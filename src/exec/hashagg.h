#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "exec/agg_spec.h"
#include "exec/group_slots.h"
#include "exec/median_store.h"
#include "exec/operator.h"

namespace wb {

// HashAggOperator: [GROUP BY group_col] with a list of aggregates. Output columns: [group key
// INT64,] then one per aggregate — Count INT64; Sum, Min, Max the input's type; Avg and Median
// FLOAT64. One row per group, in first-seen order; a global aggregate (no group_col) over no
// rows emits no row.
//
// Each group's aggregate state lives inline in its hash slot (GroupSlots). Per batch, one pass
// finds every row's state block, then one loop per aggregate updates it in place. Semantics
// (shared with the compiled engine): SUM starts at +0.0 / 0 and every row adds; SUM(INT64)
// accumulates in __int128 and throws std::overflow_error at output if the result leaves int64;
// MIN/MAX start at the group's first value and update with `v < m || v != v` (NaN wins); MEDIAN
// is exact, collected per group in block chains (MedianStore) in the column's storage type and
// selected under median_less.
class HashAggOperator final : public Operator {
   public:
    HashAggOperator(std::unique_ptr<Operator> child, std::optional<std::size_t> group_col,
                    std::vector<AggSpec> aggs);
    // GROUP BY group_col: SUM(agg_col FLOAT64), COUNT(*).
    HashAggOperator(std::unique_ptr<Operator> child, std::size_t group_col, std::size_t agg_col);

    std::optional<Batch> next() override;

   private:
    using Store = std::variant<std::monostate, MedianStore<int64_t>, MedianStore<double>, MedianStore<float>>;

    // One aggregate's place in the state block.
    struct Field {
        AggSpec     spec;
        std::size_t offset = 0;
        bool        narrow = false;  // Median over a float-stored column (set on the first batch)
        Store       store;           // Median only
    };

    void build_();
    void consume_(const Batch& batch);
    template <typename Rows>
    void consume_rows_(const Batch& batch, const Rows& rows);
    template <typename Rows>
    void consume_global_(const Batch& batch, const Rows& rows);
    void init_state_(std::byte* state, const Batch& batch, std::uint32_t row);
    void bind_median_stores_(const Batch& batch);
    void make_output_();

    std::unique_ptr<Operator>  child_;
    std::optional<std::size_t> group_col_;
    std::vector<Field>         fields_;
    std::size_t                count_offset_ = 0;
    std::size_t                state_bytes_  = 0;
    std::size_t                state_align_  = 8;
    std::optional<GroupSlots>  slots_;             // grouped
    std::vector<std::byte>     global_state_raw_;  // global: one state block (aligned inside)
    std::byte*                 global_state_   = nullptr;
    bool                       global_started_ = false;
    bool                       stores_bound_   = false;

    std::vector<std::byte*> ptrs_;  // per batch: each live row's state block

    bool                                 built_ = false;
    std::vector<std::unique_ptr<Vector>> out_vecs_;
    std::vector<const Vector*>           out_cols_;  // borrowed by emitted batches
    std::size_t                          out_rows_        = 0;
    std::size_t                          emit_cursor_     = 0;
    std::uint32_t                        emit_batch_size_ = 4096;
};
}  // namespace wb
