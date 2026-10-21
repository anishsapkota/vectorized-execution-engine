#include "codegen/generator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <set>
#include <sstream>

#include "codegen/compiler.h"
#include "codegen/runtime.h"
#include "storage/vector.h"

namespace wb::codegen {

namespace {

// ---------------------------------------------------------------------------------------------
// Scan binding
// ---------------------------------------------------------------------------------------------

void collect_scans(const plan::Node& node, const plan::Filter* parent_filter, std::vector<ScanBinding>& out) {
    if (const auto* s = std::get_if<plan::Scan>(&node.op)) {
        ScanBinding b;
        b.scan = s;
        for (std::size_t p = 0; p < s->cols.size(); ++p)
            if (s->table->schema()[s->cols[p]].type == DataType::Int64) b.int_cols.push_back(p);
        if (b.int_cols.size() > 16) throw CodegenError("codegen: a scan reads more than 16 int64 columns");
        if (parent_filter) {
            const plan::Compare& first = parent_filter->conjuncts.front();
            Predicate            p{s->cols[first.col], first.op, {}};
            std::visit([&](auto v) { p.value = v; }, first.literal);
            b.prune = p;
        }
        std::set<std::uint32_t> sigs;
        for (const auto& seg : s->table->segments())
            sigs.insert(width_sig(b, *seg));
        b.width_sigs.assign(sigs.begin(), sigs.end());
        out.push_back(std::move(b));
    } else if (const auto* f = std::get_if<plan::Filter>(&node.op)) {
        collect_scans(*f->child, f, out);
    } else if (const auto* j = std::get_if<plan::HashJoin>(&node.op)) {
        collect_scans(*j->build, nullptr, out);
        collect_scans(*j->probe, nullptr, out);
    } else {
        collect_scans(*std::get<plan::Aggregate>(node.op).child, nullptr, out);
    }
}

// ---------------------------------------------------------------------------------------------
// Source emission
// ---------------------------------------------------------------------------------------------

std::string int_literal(int64_t v) {
    if (v == INT64_MIN) return "(-9223372036854775807LL - 1)";
    return std::to_string(v) + "LL";
}

std::string double_literal(double v) {
    if (std::isnan(v)) return "__builtin_nan(\"\")";
    if (std::isinf(v)) return v > 0 ? "__builtin_inf()" : "(-__builtin_inf())";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%a", v);  // hex float: exact round trip
    return std::string{"("} + buf + ")";
}

const char* op_token(CompareOp op) {
    switch (op) {
        case CompareOp::Eq:
            return "==";
        case CompareOp::Ne:
            return "!=";
        case CompareOp::Lt:
            return "<";
        case CompareOp::Le:
            return "<=";
        case CompareOp::Gt:
            return ">";
        case CompareOp::Ge:
            return ">=";
    }
    return "==";
}

const char* code_type(std::uint32_t code) {
    static const char* kTypes[] = {"uint8_t", "uint16_t", "uint32_t", "uint64_t"};
    return kTypes[code & 3];
}

class CodeWriter {
   public:
    void line(const std::string& text) {
        out_ << std::string(indent_ * 4, ' ') << text << '\n';
    }
    void open(const std::string& text) {
        line(text);
        ++indent_;
    }
    void close(const std::string& text = "}") {
        --indent_;
        line(text);
    }
    std::string str() const { return out_.str(); }

   private:
    std::ostringstream out_;
    int                indent_ = 0;
};

// One output column of an operator, as seen by its parent: a C++ type and a loader that emits
// the load on first use (HyPer's "load attributes when first touched") and returns the variable.
struct Column {
    bool                         is_int;
    std::function<std::string()> get;
};
using Ctx      = std::vector<Column>;
using Consumer = std::function<void(Ctx&)>;

// The code of one pipeline function under construction.
struct Pipeline {
    std::size_t              scan_index;
    std::vector<std::string> prologue;  // after the column pointers, before the row loop
    std::vector<std::string> epilogue;  // after the row loop
    CodeWriter               body;      // inside the row loop
    std::vector<std::string> after_run;  // statements wb_query runs once this pipeline finished
    bool                     has_global_acc = false;
};

class Generator {
   public:
    Generator(const plan::Node& root, const std::vector<ScanBinding>& scans) : root_(root), scans_(scans) {}

    std::string run() {
        const auto& agg = std::get<plan::Aggregate>(root_.op);
        grouped_        = agg.group_col.has_value();
        sum_col_        = supported_sum_col(agg);
        produce(*agg.child, [&](Ctx& ctx) { consume_aggregate(agg, ctx); });
        if (next_scan_ != scans_.size()) throw CodegenError("codegen: scan bindings do not match the plan");
        return assemble();
    }

   private:
    Pipeline& cur() { return *pipelines_[current_]; }

    std::string fresh(const char* prefix) { return prefix + std::to_string(counter_++); }

    void produce(const plan::Node& node, const Consumer& consume) {
        if (const auto* s = std::get_if<plan::Scan>(&node.op)) return produce_scan(*s, consume);
        if (const auto* f = std::get_if<plan::Filter>(&node.op)) {
            return produce(*f->child, [&, f](Ctx& ctx) { consume_filter(*f, ctx, consume); });
        }
        if (const auto* j = std::get_if<plan::HashJoin>(&node.op)) return produce_join(*j, consume);
        throw CodegenError("codegen: Aggregate is only supported at the root");
    }

    void produce_scan(const plan::Scan& s, const Consumer& consume) {
        const std::size_t  index   = next_scan_++;
        const ScanBinding& binding = scans_.at(index);
        if (binding.scan != &s) throw CodegenError("codegen: scan bindings are out of order");

        const std::size_t previous = current_;
        pipelines_.push_back(std::make_unique<Pipeline>());
        current_                = pipelines_.size() - 1;
        cur().scan_index        = index;

        // Column pointers, declared once per segment call.
        Ctx ctx;
        for (std::size_t p = 0; p < s.cols.size(); ++p) {
            const ColumnDef& def = s.table->schema()[s.cols[p]];
            if (def.type == DataType::Int64) {
                const auto k = static_cast<std::size_t>(
                    std::find(binding.int_cols.begin(), binding.int_cols.end(), p) - binding.int_cols.begin());
                cur().prologue.push_back("const C" + std::to_string(k) + "* c" + std::to_string(p) +
                                         " = static_cast<const C" + std::to_string(k) + "*>(seg.cols[" +
                                         std::to_string(p) + "].data);");
                cur().prologue.push_back("const uint64_t b" + std::to_string(p) + " = static_cast<uint64_t>(seg.cols[" +
                                         std::to_string(p) + "].base);");
                ctx.push_back(lazy(true, "static_cast<int64_t>(b" + std::to_string(p) + " + static_cast<uint64_t>(c" +
                                             std::to_string(p) + "[i]))"));
            } else {
                const char* elem = def.narrow_float ? "float" : "double";
                cur().prologue.push_back(std::string{"const "} + elem + "* f" + std::to_string(p) +
                                         " = static_cast<const " + elem + "*>(seg.cols[" + std::to_string(p) +
                                         "].data);");
                ctx.push_back(lazy(false, "static_cast<double>(f" + std::to_string(p) + "[i])"));
            }
        }
        consume(ctx);
        current_ = previous;
    }

    // Build pipeline(s) fill join table J; the build pipeline's completion finalizes it. Then the
    // probe side produces, and each probe tuple loops over its matches.
    void produce_join(const plan::HashJoin& j, const Consumer& consume) {
        const std::size_t id         = join_cols_.size();
        const auto        build_cols = plan::output_types(*j.build);
        join_cols_.push_back(build_cols.size());

        produce(*j.build, [&, id](Ctx& ctx) { consume_build(j, id, ctx); });
        produce(*j.probe, [&, id](Ctx& ctx) { consume_probe(j, id, build_cols, ctx, consume); });
    }

    void consume_build(const plan::HashJoin& j, std::size_t id, Ctx& ctx) {
        const std::string jt = "st.jt" + std::to_string(id);
        std::string       payload;
        for (std::size_t c = 0; c < ctx.size(); ++c) {
            const std::string v = ctx[c].get();
            payload += std::string{c ? ", " : ""} +
                       (ctx[c].is_int ? "static_cast<uint64_t>(" + v + ")" : "__builtin_bit_cast(uint64_t, " + v + ")");
        }
        const std::string key = ctx[j.build_key].get();
        cur().body.open("{");
        cur().body.line("const uint64_t payload[] = {" + payload + "};");
        cur().body.line(jt + ".append(" + key + ", payload);");
        cur().body.close();
        if (finalized_.size() <= id) finalized_.resize(id + 1, false);
        if (!finalized_[id]) {
            finalized_[id] = true;
            cur().after_run.push_back("if (rc == 0) " + jt + ".finalize();");
        }
    }

    void consume_probe(const plan::HashJoin& j, std::size_t id, const std::vector<DataType>& build_cols, Ctx& probe,
                       const Consumer& consume) {
        const std::string jt     = "st.jt" + std::to_string(id);
        const std::string bucket = fresh("jb");
        const std::string row    = fresh("j");
        cur().body.line("const wb_rt::JoinBucket* " + bucket + " = " + jt + ".find(" + probe[j.probe_key].get() + ");");
        cur().body.line("if (" + bucket + " == nullptr) continue;");
        cur().body.open("for (uint32_t " + row + " = " + bucket + "->first; " + row + " < " + bucket + "->first + " +
                        bucket + "->count; ++" + row + ") {");
        Ctx out = probe;
        for (std::size_t c = 0; c < build_cols.size(); ++c) {
            const std::string raw = jt + ".col(" + std::to_string(c) + "u, " + row + ")";
            out.push_back(build_cols[c] == DataType::Int64 ? lazy(true, "static_cast<int64_t>(" + raw + ")")
                                                           : lazy(false, "__builtin_bit_cast(double, " + raw + ")"));
        }
        consume(out);
        cur().body.close();
    }

    // A column whose load is emitted at first use, into whatever scope is current then.
    Column lazy(bool is_int, std::string expr) {
        auto name = std::make_shared<std::string>();
        return Column{is_int, [this, is_int, expr, name] {
                          if (name->empty()) {
                              *name = fresh("v");
                              cur().body.line(std::string{"const "} + (is_int ? "int64_t " : "double ") + *name +
                                              " = " + expr + ";");
                          }
                          return *name;
                      }};
    }

    void consume_filter(const plan::Filter& f, Ctx& ctx, const Consumer& consume) {
        for (const plan::Compare& c : f.conjuncts) {
            const std::string lit = std::holds_alternative<int64_t>(c.literal)
                                        ? int_literal(std::get<int64_t>(c.literal))
                                        : double_literal(std::get<double>(c.literal));
            cur().body.line("if (!(" + ctx[c.col].get() + " " + op_token(c.op) + " " + lit + ")) continue;");
        }
        consume(ctx);
    }

    // Until the generator is generalized: at least one Sum, all on one Float64 column, plus Counts.
    static std::size_t supported_sum_col(const plan::Aggregate& agg) {
        const auto                 types = plan::output_types(*agg.child);
        std::optional<std::size_t> col;
        for (const plan::AggExpr& e : agg.aggs) {
            if (e.fn == plan::AggFn::Count) continue;
            if (e.fn != plan::AggFn::Sum || types[*e.col] != DataType::Float64 || (col && *col != *e.col)) {
                throw CodegenError(std::string{"codegen: aggregate "} + plan::agg_fn_name(e.fn) + " not supported yet");
            }
            col = e.col;
        }
        if (!col) throw CodegenError("codegen: aggregate count not supported yet");
        return *col;
    }

    void consume_aggregate(const plan::Aggregate& agg, Ctx& ctx) {
        if (agg.group_col) {
            const std::string key = ctx[*agg.group_col].get();
            const std::string val = ctx[sum_col_].get();
            cur().body.open("{");
            cur().body.line("wb_rt::AggSlot& e = st.agg.find_or_insert(" + key + ");");
            cur().body.line("e.sum += " + val + ";");
            cur().body.line("++e.count;");
            cur().body.close();
            return;
        }
        // Accumulate in locals (registers) for the segment, continuing from st so the summation
        // order is exactly the vectorized engine's.
        if (!cur().has_global_acc) {
            cur().has_global_acc = true;
            cur().prologue.push_back("double acc_sum = st.sum;");
            cur().prologue.push_back("int64_t acc_count = st.count;");
            cur().epilogue.push_back("st.sum = acc_sum;");
            cur().epilogue.push_back("st.count = acc_count;");
        }
        const std::string val = ctx[sum_col_].get();
        cur().body.line("acc_sum += " + val + ";");
        cur().body.line("++acc_count;");
    }

    std::string assemble() {
        CodeWriter w;
        w.line("// Generated by wb::codegen for: " + plan::to_string(root_));
        w.line("#include \"codegen/runtime.h\"");
        w.line("");
        w.open("namespace {");
        w.line("");
        w.open("struct State {");
        w.line("double sum = 0.0;");
        w.line("int64_t count = 0;");
        w.line("wb_rt::AggTable agg;");
        for (std::size_t id = 0; id < join_cols_.size(); ++id)
            w.line("wb_rt::JoinTable jt" + std::to_string(id) + ";");
        w.close("};");

        for (const auto& pipeline : pipelines_)
            emit_pipeline(w, *pipeline);

        w.close("}  // namespace");
        w.line("");
        w.line("extern \"C\" uint32_t wb_abi_version() { return wb_rt::kAbiVersion; }");
        w.line("");
        w.open("extern \"C\" int wb_query(const wb_rt::QueryInput* in, wb_rt::QueryOutput* out) {");
        w.line("if (in->num_scans != " + std::to_string(scans_.size()) + "u) return 2;");
        w.line("State st;");
        if (grouped_) w.line("st.agg.init();");
        for (std::size_t id = 0; id < join_cols_.size(); ++id)
            w.line("st.jt" + std::to_string(id) + ".init(" + std::to_string(join_cols_[id]) + "u);");
        w.line("int rc = 0;");
        for (const auto& pipeline : pipelines_) {
            const std::string k = std::to_string(pipeline->scan_index);
            w.line("if (rc == 0) rc = p" + k + "(in->scans[" + k + "], st);");
            for (const auto& stmt : pipeline->after_run)
                w.line(stmt);
        }
        w.open("if (rc == 0) {");
        if (grouped_) {
            w.line("st.agg.for_each([&](const wb_rt::AggSlot& s) { out->emit_group(out->ctx, s.key, s.sum, s.count); });");
        } else {
            w.line("out->emit_group(out->ctx, 0, st.sum, st.count);");
        }
        w.close();
        if (grouped_) w.line("st.agg.release();");
        for (std::size_t id = 0; id < join_cols_.size(); ++id)
            w.line("st.jt" + std::to_string(id) + ".release();");
        w.line("return rc;");
        w.close();
        return w.str();
    }

    void emit_pipeline(CodeWriter& w, const Pipeline& pipeline) {
        const ScanBinding& binding = scans_[pipeline.scan_index];
        const std::string  k       = std::to_string(pipeline.scan_index);
        const std::size_t  nparams = binding.int_cols.size();

        w.line("");
        if (nparams > 0) {
            std::string params;
            for (std::size_t i = 0; i < nparams; ++i)
                params += (i ? ", class C" : "class C") + std::to_string(i);
            w.line("template <" + params + ">");
        }
        w.open("void p" + k + "_seg(const wb_rt::SegmentRef& seg, State& st) {");
        w.line("(void)st;");
        for (const auto& stmt : pipeline.prologue)
            w.line(stmt);
        w.open("for (uint32_t i = seg.lo; i < seg.hi; ++i) {");
        std::istringstream body{pipeline.body.str()};
        for (std::string line; std::getline(body, line);)
            w.line(line);
        w.close();
        for (const auto& stmt : pipeline.epilogue)
            w.line(stmt);
        w.close();

        // Dispatch each segment to the instantiation for its width signature.
        w.line("");
        w.open("int p" + k + "(const wb_rt::ScanInput& in, State& st) {");
        w.open("for (uint32_t s = 0; s < in.num_segs; ++s) {");
        w.line("const wb_rt::SegmentRef& seg = in.segs[s];");
        w.open("switch (seg.width_sig) {");
        for (std::uint32_t sig : binding.width_sigs) {
            std::string args;
            for (std::size_t i = 0; i < nparams; ++i)
                args += std::string{i ? ", " : ""} + code_type(sig >> (2 * i));
            w.line("case " + std::to_string(sig) + "u: p" + k + "_seg" + (nparams ? "<" + args + ">" : "") +
                   "(seg, st); break;");
        }
        w.line("default: return 1;");
        w.close();
        w.close();
        w.line("return 0;");
        w.close();
    }

    const plan::Node&                       root_;
    const std::vector<ScanBinding>&         scans_;
    std::vector<std::unique_ptr<Pipeline>>  pipelines_;
    std::size_t                             current_   = 0;
    std::size_t                             next_scan_ = 0;
    std::size_t                             counter_   = 0;
    bool                                    grouped_   = false;
    std::size_t                             sum_col_   = 0;
    std::vector<std::size_t>                join_cols_;  // payload columns per join table
    std::vector<bool>                       finalized_;  // finalize() already scheduled
};

}  // namespace

std::uint32_t width_sig(const ScanBinding& binding, const Segment& seg) {
    std::uint32_t sig = 0;
    for (std::size_t k = 0; k < binding.int_cols.size(); ++k) {
        const auto& col = static_cast<const Int64Vector&>(seg.column(binding.scan->cols[binding.int_cols[k]]));
        sig |= wb_rt::width_code(col.raw().width) << (2 * k);
    }
    return sig;
}

std::vector<ScanBinding> bind_scans(const plan::Node& root) {
    std::vector<ScanBinding> out;
    collect_scans(root, nullptr, out);
    return out;
}

std::string generate(const plan::Node& root, const std::vector<ScanBinding>& scans) {
    return Generator{root, scans}.run();
}

}  // namespace wb::codegen
