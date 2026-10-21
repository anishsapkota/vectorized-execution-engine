#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "exec/agg_spec.h"
#include "exec/group_index.h"
#include "exec/operator.h"

namespace wb {

// HashAggOperator: [GROUP BY group_col] with a list of aggregates. Output columns: [group key
// INT64,] then one per aggregate — Count INT64; Sum, Min, Max the input's type; Avg FLOAT64.
// One row per group; a global aggregate (no group_col) over no rows emits no row.
//
// Vectorized: per batch, one pass assigns dense group ids, then one loop per aggregate updates
// that aggregate's state array. Semantics (shared with the compiled engine):
//   SUM starts at +0.0 / 0 and every row adds; SUM(INT64) accumulates in __int128 and throws
//   std::overflow_error at output if the result leaves int64. MIN/MAX start at the group's first
//   value and update with `v < m || v != v` (NaN wins). MEDIAN is exact: values are collected,
//   bucketed by group with a counting sort, and selected with std::nth_element (NaN sorts last);
//   an even count averages the two middle values.
class HashAggOperator final : public Operator {
   public:
    HashAggOperator(std::unique_ptr<Operator> child, std::optional<std::size_t> group_col,
                    std::vector<AggSpec> aggs);
    // GROUP BY group_col: SUM(agg_col FLOAT64), COUNT(*).
    HashAggOperator(std::unique_ptr<Operator> child, std::size_t group_col, std::size_t agg_col);

    std::optional<Batch> next() override;

   private:
    // Per-aggregate state, indexed by group id. Only the array the aggregate needs is used.
    struct State {
        AggSpec               spec;
        std::vector<double>   f64;   // Sum/Avg FLOAT64, Min/Max FLOAT64
        std::vector<__int128> i128;  // Sum/Avg INT64
        std::vector<int64_t>  i64;   // Min/Max INT64
        // Median: every (group id, value) pair, in arrival order.
        std::vector<std::uint32_t> med_gids;
        std::vector<int64_t>       med_i64;
        std::vector<double>        med_f64;
    };

    void build_();
    void consume_(const Batch& batch);
    template <typename Rows>
    void consume_rows_(const Batch& batch, const Rows& rows);
    template <typename Rows>
    void consume_global_(const Batch& batch, const Rows& rows);
    void add_group_(const Batch& batch, std::uint32_t row);
    void make_output_();
    std::vector<double> medians_(State& s) const;

    std::unique_ptr<Operator>  child_;
    std::optional<std::size_t> group_col_;
    std::vector<State>         states_;
    GroupIndex                 index_;
    std::vector<int64_t>       counts_;  // rows per group

    // Per-batch scratch, reused: group id of each live row.
    std::vector<std::uint32_t> gids_;

    bool                                 built_ = false;
    std::vector<std::unique_ptr<Vector>> out_vecs_;
    std::vector<const Vector*>           out_cols_;  // borrowed by emitted batches
    std::size_t                          emit_cursor_     = 0;
    std::uint32_t                        emit_batch_size_ = 4096;
};
}  // namespace wb
