#include "plan/lower_vec.h"

#include <memory>

#include "exec/filter.h"
#include "exec/hash_join.h"
#include "exec/agg_spec.h"
#include "exec/hashagg.h"
#include "exec/scan.h"

namespace wb::plan {

namespace {

constexpr std::uint32_t kBatchSize = 4096;

static_assert(static_cast<int>(AggFn::Count) == static_cast<int>(AggKind::Count) &&
                  static_cast<int>(AggFn::Sum) == static_cast<int>(AggKind::Sum) &&
                  static_cast<int>(AggFn::Min) == static_cast<int>(AggKind::Min) &&
                  static_cast<int>(AggFn::Max) == static_cast<int>(AggKind::Max) &&
                  static_cast<int>(AggFn::Avg) == static_cast<int>(AggKind::Avg) &&
                  static_cast<int>(AggFn::Median) == static_cast<int>(AggKind::Median),
              "plan::AggFn and wb::AggKind must list the same functions in the same order");

Predicate to_predicate(std::size_t col, const Compare& c) {
    Predicate p{col, c.op, {}};
    std::visit([&](auto v) { p.value = v; }, c.literal);
    return p;
}

std::unique_ptr<Operator> lower(const Node& node) {
    if (const auto* s = std::get_if<Scan>(&node.op)) {
        return std::make_unique<ScanOperator>(*s->table, std::nullopt, kBatchSize, s->cols);
    }
    if (const auto* f = std::get_if<Filter>(&node.op)) {
        std::unique_ptr<Operator> input;
        if (const auto* s = std::get_if<Scan>(&f->child->op)) {
            // Prune on the first conjunct, expressed against the table's column numbering.
            const Compare& first = f->conjuncts.front();
            input = std::make_unique<ScanOperator>(*s->table, to_predicate(s->cols[first.col], first),
                                                   kBatchSize, s->cols);
        } else {
            input = lower(*f->child);
        }
        for (const Compare& c : f->conjuncts)
            input = std::make_unique<FilterOperator>(std::move(input), to_predicate(c.col, c));
        return input;
    }
    const auto& j = std::get<HashJoin>(node.op);
    return std::make_unique<HashJoinOperator>(lower(*j.probe), lower(*j.build), j.probe_key, j.build_key);
}

}  // namespace

QueryResult run_vectorized(const Node& root) {
    validate_query(root);
    const auto& agg         = std::get<Aggregate>(root.op);
    const auto  child_types = output_types(*agg.child);
    const auto  out_types   = output_types(root);

    std::vector<AggSpec> specs;
    for (const AggExpr& e : agg.aggs) {
        const auto kind = static_cast<AggKind>(e.fn);  // AggFn and AggKind list the same functions
        specs.push_back(e.col ? AggSpec{kind, *e.col, child_types[*e.col]} : AggSpec{kind});
    }
    HashAggOperator op{lower(*agg.child), agg.group_col, std::move(specs)};

    QueryResult result;
    result.grouped = agg.group_col.has_value();
    for (DataType t : out_types)
        result.columns.push_back({t, {}, {}, {}});
    while (auto batch = op.next()) {
        for (std::size_t c = 0; c < result.columns.size(); ++c) {
            const Vector& col = batch->column(c);
            for_each_live_row(*batch, [&](std::uint32_t row) {
                if (col.type() == DataType::Int64) {
                    result.columns[c].push_int(static_cast<const Int64Vector&>(col).value(row));
                } else {
                    result.columns[c].push_double(static_cast<const Float64Vector&>(col).value(row));
                }
            });
        }
    }
    // A global aggregate over no rows still answers one row: COUNT 0, everything else NULL.
    if (!result.grouped && result.num_rows() == 0) {
        for (std::size_t i = 0; i < agg.aggs.size(); ++i) {
            auto& col = result.columns[i];
            if (agg.aggs[i].fn == AggFn::Count) col.push_int(0);
            else if (col.type == DataType::Int64) col.push_int(0, false);
            else col.push_double(0.0, false);
        }
    }
    result.canonicalize();
    return result;
}

}  // namespace wb::plan
