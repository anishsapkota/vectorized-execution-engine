#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "exec/hash_table.h"
#include "exec/operator.h"
#include "storage/segment.h"

namespace wb {

// HashAggOperator: GROUP BY group_col, SUM(agg_col), COUNT(*).
// Currently assumes group_col is Int64 and agg_col is Float64.
// Output schema: {group_col INT64, sum FLOAT64, count INT64}.
class HashAggOperator final : public Operator {
   public:
    HashAggOperator(std::unique_ptr<Operator> child, std::size_t group_col_idx,
                    std::size_t agg_col_idx);
    std::optional<Batch> next() override;

   private:
    struct AggState {
        double  sum   = 0.0;
        int64_t count = 0;
    };

    void build_();      // pulls all child batches, populates hash_
    void emit_init_();  // prepares iteration over hash_ for emission

    std::unique_ptr<Operator> child_;
    std::size_t               group_col_;
    std::size_t               agg_col_;

    AggHashTable hash_;
    bool         built_ = false;

    // Output emission state
    std::unique_ptr<Int64Vector>   out_key_vec_;
    std::unique_ptr<Float64Vector> out_sum_vec_;
    std::unique_ptr<Int64Vector>   out_count_vec_;
    // Column list borrowed by emitted batches.
    std::array<const Vector*, 3> out_cols_{};

    std::size_t   emit_cursor_     = 0;
    std::uint32_t emit_batch_size_ = 4096;
};
}  // namespace wb
