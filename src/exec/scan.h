#pragma once

#include <optional>
#include <vector>

#include "exec/operator.h"
#include "exec/predicate.h"
#include "storage/table.h"

namespace wb {

// First row worth scanning in a segment sorted on pred's column, using the sparse primary index.
// Shared by ScanOperator and the codegen host so both engines read identical row ranges.
std::size_t scan_start_row(const std::vector<IndexEntry>& idx, const Predicate& pred);

class ScanOperator final : public Operator {
   public:
    // Scan a table; if predicate is set, prune segments whose zone map proves
    // they cannot match. Within surviving segments, emit every row id.
    // projected_cols: original column indices to include in each batch (empty = all).
    ScanOperator(const Table& table, std::optional<Predicate> predicate,
                 std::uint32_t            target_batch_size = 4096,
                 std::vector<std::size_t> projected_cols    = {});

    std::optional<Batch> next() override;

    // Stats for EXPLAIN — the talking-point version.
    std::size_t segments_scanned() const { return segments_scanned_; }
    std::size_t segments_pruned() const { return segments_pruned_; }

   private:
    const Table&             table_;
    std::optional<Predicate> predicate_;
    std::uint32_t            target_batch_size_;
    std::vector<std::size_t> projected_cols_;  // empty = all columns
    // Iteration state
    std::size_t seg_idx_         = 0;
    std::size_t row_idx_         = 0;
    bool        seg_initialized_ = false;
    // Column pointers for the current segment; emitted batches borrow this buffer.
    std::vector<const Vector*> batch_cols_;

    // Stats.
    std::size_t segments_scanned_ = 0;
    std::size_t segments_pruned_  = 0;
};

}  // namespace wb