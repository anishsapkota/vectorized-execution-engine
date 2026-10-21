#include "codegen/host.h"

#include <bit>
#include <chrono>

#include "exec/predicate.h"
#include "exec/scan.h"
#include "storage/vector.h"

namespace wb::codegen {

namespace {

using Clock = std::chrono::steady_clock;

// Everything wb_query reads, kept alive for the duration of the call.
struct BoundInput {
    std::vector<wb_rt::ColumnRef>               columns;
    std::vector<std::vector<wb_rt::SegmentRef>> segments;  // per scan
    std::vector<wb_rt::ScanInput>               scans;
    wb_rt::QueryInput                           input{};
};

void bind_inputs(const std::vector<ScanBinding>& bindings, BoundInput& bound) {
    std::size_t total_columns = 0;
    for (const auto& b : bindings)
        total_columns += b.scan->table->num_segments() * b.scan->cols.size();
    bound.columns.reserve(total_columns);  // SegmentRefs point into it: no reallocation allowed
    bound.segments.resize(bindings.size());

    for (std::size_t s = 0; s < bindings.size(); ++s) {
        const ScanBinding& b     = bindings[s];
        const Table&       table = *b.scan->table;
        const bool on_sort_key   = b.prune && table.sort_key_col() && *table.sort_key_col() == b.prune->col_idx;

        for (const auto& seg_ptr : table.segments()) {
            const Segment& seg = *seg_ptr;
            if (b.prune && !zone_map_can_match(seg.zone(b.prune->col_idx), b.prune->op, b.prune->value)) continue;

            std::size_t lo = 0;
            if (on_sort_key && !seg.primary_index().empty()) lo = scan_start_row(seg.primary_index(), *b.prune);
            if (lo >= seg.num_rows()) continue;

            const wb_rt::ColumnRef* first = bound.columns.data() + bound.columns.size();
            for (std::size_t col : b.scan->cols) {
                const Vector& v = seg.column(col);
                if (v.type() == DataType::Int64) {
                    const Int64Raw raw = static_cast<const Int64Vector&>(v).raw();
                    bound.columns.push_back({raw.codes, raw.base, raw.width, 0});
                } else {
                    const Float64Raw raw = static_cast<const Float64Vector&>(v).raw();
                    bound.columns.push_back({raw.values, 0, 0, static_cast<uint8_t>(raw.narrow)});
                }
            }
            bound.segments[s].push_back({first, static_cast<uint32_t>(lo), static_cast<uint32_t>(seg.num_rows()),
                                         width_sig(b, seg)});
        }
    }
    for (const auto& segs : bound.segments)
        bound.scans.push_back({segs.data(), static_cast<uint32_t>(segs.size())});
    bound.input = {bound.scans.data(), static_cast<uint32_t>(bound.scans.size())};
}

struct Collector {
    plan::QueryResult*      result;
    const plan::Aggregate*  agg;
};

// Rows arrive as [key,] sum, count (the generator's current SUM/COUNT state).
void collect_row(void* ctx, const uint64_t* values, uint32_t) {
    auto*       c     = static_cast<Collector*>(ctx);
    std::size_t v     = 0;
    std::size_t i     = 0;
    if (c->result->grouped) c->result->columns[i++].push_int(static_cast<int64_t>(values[v++]));
    const double  sum   = std::bit_cast<double>(values[v++]);
    const int64_t count = static_cast<int64_t>(values[v]);
    for (const plan::AggExpr& e : c->agg->aggs) {
        auto& col = c->result->columns[i++];
        if (e.fn == plan::AggFn::Count) col.push_int(count);
        else col.push_double(sum, count > 0);  // SUM over no rows is NULL
    }
}

}  // namespace

PreparedQuery prepare(const plan::Node& root, const CompileOptions& options) {
    plan::validate_query(root);
    PreparedQuery query;
    query.root    = &root;
    query.grouped = std::get<plan::Aggregate>(root.op).group_col.has_value();

    const auto        start  = Clock::now();
    query.scans              = bind_scans(root);
    const std::string source = generate(root, query.scans);
    const double generate_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    query.compiled                   = compile_source(source, options);
    query.compiled.stats.generate_ms = generate_ms;
    return query;
}

plan::QueryResult execute(const PreparedQuery& query) {
    BoundInput bound;
    bind_inputs(query.scans, bound);

    const auto&       agg = std::get<plan::Aggregate>(query.root->op);
    plan::QueryResult result;
    result.grouped = query.grouped;
    if (result.grouped) result.columns.push_back({DataType::Int64, {}, {}, {}});
    const auto types = plan::output_types(*query.root);
    for (std::size_t i = result.grouped ? 1 : 0; i < types.size(); ++i)
        result.columns.push_back({types[i], {}, {}, {}});
    Collector          collector{&result, &agg};
    wb_rt::QueryOutput out{&collector, collect_row, 0, 0};
    const int          rc = query.compiled.fn(&bound.input, &out);
    if (rc == 1) throw CodegenError("codegen: a segment's width signature was not compiled (stale library?)");
    if (rc != 0) throw CodegenError("codegen: wb_query failed with code " + std::to_string(rc));
    result.canonicalize();
    return result;
}

plan::QueryResult run_compiled(const plan::Node& root, const CompileOptions& options, CompileStats* stats) {
    const PreparedQuery query = prepare(root, options);
    if (stats) *stats = query.compiled.stats;
    return execute(query);
}

}  // namespace wb::codegen
