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
    const auto& agg         = std::get<Aggregate>(root.op);
    const auto  child_types = output_types(*agg.child);

    // Until HashAggOperator is generalized: at least one Sum, all on one Float64 column, plus Counts.
    std::optional<std::size_t> sum_col;
    for (const AggExpr& e : agg.aggs) {
        if (e.fn == AggFn::Count) continue;
        if (e.fn != AggFn::Sum || child_types[*e.col] != DataType::Float64 || (sum_col && *sum_col != *e.col)) {
            throw std::invalid_argument(std::string{"aggregate "} + agg_fn_name(e.fn) + " not supported yet");
        }
        sum_col = e.col;
    }
    if (!sum_col) throw std::invalid_argument("aggregate count not supported yet");

    auto                 input = lower(*agg.child);
    std::vector<int64_t> keys, counts;
    std::vector<double>  sums;
    if (agg.group_col) {
        HashAggOperator op{std::move(input), *agg.group_col, *sum_col};
        while (auto batch = op.next()) {
            const auto& k = static_cast<const Int64Vector&>(batch->column(0));
            const auto& s = static_cast<const Float64Vector&>(batch->column(1));
            const auto& c = static_cast<const Int64Vector&>(batch->column(2));
            for_each_live_row(*batch, [&](std::uint32_t row) {
                keys.push_back(k.value(row));
                sums.push_back(s.value(row));
                counts.push_back(c.value(row));
            });
        }
    } else {
        double  sum   = 0.0;
        int64_t count = 0;
        while (auto batch = input->next()) {
            static_cast<const Float64Vector&>(batch->column(*sum_col)).visit([&](auto values) {
                for_each_live_row(*batch, [&](std::uint32_t row) { sum += values[row]; });
            });
            count += batch->size();
        }
        sums   = {sum};
        counts = {count};
    }

    QueryResult result;
    result.grouped = agg.group_col.has_value();
    if (result.grouped) result.columns.push_back({DataType::Int64, {}, {}, {}});
    for (const AggExpr& e : agg.aggs)
        result.columns.push_back({e.fn == AggFn::Count ? DataType::Int64 : DataType::Float64, {}, {}, {}});
    for (std::size_t row = 0; row < counts.size(); ++row) {
        std::size_t c = 0;
        if (result.grouped) result.columns[c++].push_int(keys[row]);
        for (const AggExpr& e : agg.aggs) {
            auto& col = result.columns[c++];
            if (e.fn == AggFn::Count) col.push_int(counts[row]);
            else col.push_double(sums[row], counts[row] > 0);  // SUM over no rows is NULL
        }
    }
    result.canonicalize();
    return result;
}

}  // namespace wb::plan
