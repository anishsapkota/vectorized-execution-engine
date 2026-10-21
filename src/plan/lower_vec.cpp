#include "plan/lower_vec.h"

#include <memory>

#include "exec/filter.h"
#include "exec/hash_join.h"
#include "exec/hashagg.h"
#include "exec/scan.h"

namespace wb::plan {

namespace {

constexpr std::uint32_t kBatchSize = 4096;

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
    const auto& agg   = std::get<SumCount>(root.op);
    auto        input = lower(*agg.child);

    QueryResult result;
    result.grouped = agg.group_col.has_value();
    if (result.grouped) result.columns.push_back({DataType::Int64, {}, {}, {}});
    result.columns.push_back({DataType::Float64, {}, {}, {}});
    result.columns.push_back({DataType::Int64, {}, {}, {}});
    auto& cols = result.columns;
    if (agg.group_col) {
        HashAggOperator op{std::move(input), *agg.group_col, agg.sum_col};
        while (auto batch = op.next()) {
            const auto& keys   = static_cast<const Int64Vector&>(batch->column(0));
            const auto& sums   = static_cast<const Float64Vector&>(batch->column(1));
            const auto& counts = static_cast<const Int64Vector&>(batch->column(2));
            for_each_live_row(*batch, [&](std::uint32_t row) {
                cols[0].push_int(keys.value(row));
                cols[1].push_double(sums.value(row));
                cols[2].push_int(counts.value(row));
            });
        }
    } else {
        double  sum   = 0.0;
        int64_t count = 0;
        while (auto batch = input->next()) {
            static_cast<const Float64Vector&>(batch->column(agg.sum_col)).visit([&](auto values) {
                for_each_live_row(*batch, [&](std::uint32_t row) { sum += values[row]; });
            });
            count += batch->size();
        }
        cols[0].push_double(sum);
        cols[1].push_int(count);
    }
    result.canonicalize();
    return result;
}

}  // namespace wb::plan
