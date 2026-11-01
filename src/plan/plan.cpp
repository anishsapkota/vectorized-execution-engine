#include "plan/plan.h"

#include <sstream>
#include <stdexcept>

namespace wb::plan {

namespace {

template <class... Fs>
struct Overloaded : Fs... {
    using Fs::operator()...;
};
template <class... Fs>
Overloaded(Fs...) -> Overloaded<Fs...>;

[[noreturn]] void reject(const std::string& node, const std::string& what) {
    throw std::invalid_argument("plan: " + node + ": " + what);
}

DataType column_type(const std::vector<DataType>& types, std::size_t col, const char* node,
                     const char* role) {
    if (col >= types.size()) {
        reject(node, std::string{role} + " column " + std::to_string(col) + " out of range (input has " +
                         std::to_string(types.size()) + " columns)");
    }
    return types[col];
}

const char* op_symbol(CompareOp op) {
    switch (op) {
        case CompareOp::Eq:
            return "=";
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
    return "?";
}

bool contains_aggregate(const Node& node) {
    return std::visit(Overloaded{
                          [](const Scan&) { return false; },
                          [](const Filter& f) { return contains_aggregate(*f.child); },
                          [](const HashJoin& j) {
                              return contains_aggregate(*j.probe) || contains_aggregate(*j.build);
                          },
                          [](const Aggregate&) { return true; },
                      },
                      node.op);
}

}  // namespace

NodePtr scan(const Table& table, std::vector<std::size_t> cols) {
    return std::make_unique<Node>(Node{Scan{&table, std::move(cols)}});
}

NodePtr filter(NodePtr child, std::vector<Compare> conjuncts) {
    return std::make_unique<Node>(Node{Filter{std::move(child), std::move(conjuncts)}});
}

NodePtr join(NodePtr probe, NodePtr build, std::size_t probe_key, std::size_t build_key) {
    return std::make_unique<Node>(Node{HashJoin{std::move(probe), std::move(build), probe_key, build_key}});
}

NodePtr aggregate(NodePtr child, std::optional<std::size_t> group_col, std::vector<AggExpr> aggs) {
    return std::make_unique<Node>(Node{Aggregate{std::move(child), group_col, std::move(aggs)}});
}

NodePtr sum_count(NodePtr child, std::optional<std::size_t> group_col, std::size_t sum_col) {
    return aggregate(std::move(child), group_col, {{AggFn::Sum, sum_col}, {AggFn::Count, std::nullopt}});
}

const char* agg_fn_name(AggFn fn) {
    switch (fn) {
        case AggFn::Count:
            return "count";
        case AggFn::Sum:
            return "sum";
        case AggFn::Min:
            return "min";
        case AggFn::Max:
            return "max";
        case AggFn::Avg:
            return "avg";
        case AggFn::Median:
            return "median";
    }
    return "?";
}

std::vector<DataType> output_types(const Node& node) {
    return std::visit(
        Overloaded{
            [](const Scan& s) {
                const Schema&         schema = s.table->schema();
                std::vector<DataType> types;
                for (std::size_t col : s.cols) {
                    if (col >= schema.size()) {
                        reject("Scan(" + s.table->name() + ")",
                               "table column " + std::to_string(col) + " out of range");
                    }
                    const DataType t = schema[col].type;
                    if (t != DataType::Int64 && t != DataType::Float64) {
                        reject("Scan(" + s.table->name() + ")",
                               "column '" + schema[col].name + "' has unsupported type " + type_name(t));
                    }
                    types.push_back(t);
                }
                return types;
            },
            [](const Filter& f) {
                auto types = output_types(*f.child);
                if (f.conjuncts.empty()) reject("Filter", "needs at least one conjunct");
                for (const Compare& c : f.conjuncts) {
                    const DataType t = column_type(types, c.col, "Filter", "compared");
                    const bool literal_int = std::holds_alternative<int64_t>(c.literal);
                    if (literal_int != (t == DataType::Int64)) {
                        reject("Filter", "literal type does not match column " + std::to_string(c.col) +
                                             " (" + type_name(t) + ")");
                    }
                }
                return types;
            },
            [](const HashJoin& j) {
                auto probe = output_types(*j.probe);
                auto build = output_types(*j.build);
                if (column_type(probe, j.probe_key, "HashJoin", "probe key") != DataType::Int64 ||
                    column_type(build, j.build_key, "HashJoin", "build key") != DataType::Int64) {
                    reject("HashJoin", "join keys must be int64");
                }
                probe.insert(probe.end(), build.begin(), build.end());
                return probe;
            },
            [](const Aggregate& a) {
                const auto            input = output_types(*a.child);
                std::vector<DataType> types;
                if (a.group_col) {
                    if (column_type(input, *a.group_col, "Aggregate", "group") != DataType::Int64) {
                        reject("Aggregate", "group column must be int64");
                    }
                    types.push_back(DataType::Int64);
                }
                if (a.aggs.empty()) reject("Aggregate", "needs at least one aggregate");
                for (const AggExpr& e : a.aggs) {
                    const std::string name = agg_fn_name(e.fn);
                    if (e.fn == AggFn::Count) {
                        if (e.col) reject("Aggregate", "count takes no column (COUNT(*))");
                        types.push_back(DataType::Int64);
                        continue;
                    }
                    if (!e.col) reject("Aggregate", name + " needs a column");
                    const DataType in = column_type(input, *e.col, "Aggregate", name.c_str());
                    const bool to_double = e.fn == AggFn::Avg || e.fn == AggFn::Median;
                    types.push_back(to_double ? DataType::Float64 : in);
                }
                return types;
            },
        },
        node.op);
}

void validate_query(const Node& root) {
    const auto* agg = std::get_if<Aggregate>(&root.op);
    if (!agg) reject("query", "root must be Aggregate");
    if (contains_aggregate(*agg->child)) reject("query", "Aggregate is only allowed at the root");
    output_types(root);
}

std::string to_string(const Node& node) {
    return std::visit(
        Overloaded{
            [](const Scan& s) {
                std::ostringstream out;
                out << "Scan(" << s.table->name() << ",[";
                for (std::size_t i = 0; i < s.cols.size(); ++i)
                    out << (i ? "," : "") << s.cols[i];
                out << "])";
                return out.str();
            },
            [](const Filter& f) {
                std::ostringstream out;
                out.precision(17);
                out << "Filter([";
                for (std::size_t i = 0; i < f.conjuncts.size(); ++i) {
                    const Compare& c = f.conjuncts[i];
                    out << (i ? "," : "") << c.col << op_symbol(c.op);
                    std::visit([&](auto v) { out << v; }, c.literal);
                }
                out << "], " << to_string(*f.child) << ")";
                return out.str();
            },
            [](const HashJoin& j) {
                return "HashJoin(" + std::to_string(j.probe_key) + "=" + std::to_string(j.build_key) +
                       ", " + to_string(*j.probe) + ", " + to_string(*j.build) + ")";
            },
            [](const Aggregate& a) {
                std::string out = "Aggregate(group=" +
                                  (a.group_col ? std::to_string(*a.group_col) : std::string{"none"}) + ",[";
                for (std::size_t i = 0; i < a.aggs.size(); ++i) {
                    out += (i ? "," : "") + std::string{agg_fn_name(a.aggs[i].fn)};
                    if (a.aggs[i].col) out += "(" + std::to_string(*a.aggs[i].col) + ")";
                }
                return out + "], " + to_string(*a.child) + ")";
            },
        },
        node.op);
}

}  // namespace wb::plan
