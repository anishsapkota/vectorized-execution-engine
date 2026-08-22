#include "exec/scan.h"

#include <algorithm>
#include <iostream>

namespace wb {

namespace {

std::size_t find_starting_row_(const std::vector<IndexEntry>& idx, const Predicate& pred) {
    int64_t value = std::get<int64_t>(pred.value);  // sort key is int64

    switch (pred.op) {
        case CompareOp::Gt:
        case CompareOp::Ge: {
            // Find first granule whose start value could satisfy the predicate.
            // Last granule with start value <= value is where we begin.
            // (Granule N starts at offset N*granule; values in [N, N+1) might satisfy.)
            // Binary search the first granule whose START is > value, back up by one.
            auto it = std::upper_bound(
                idx.begin(), idx.end(), value,
                [](int64_t v, const IndexEntry& e) { return v < e.sort_key_value; });
            if (it == idx.begin()) return 0;
            return (it - 1)->row_offset;
        }
        case CompareOp::Eq: {
            // Same logic — find the granule that may contain `value`.
            auto it = std::upper_bound(
                idx.begin(), idx.end(), value,
                [](int64_t v, const IndexEntry& e) { return v < e.sort_key_value; });
            if (it == idx.begin()) return 0;
            return (it - 1)->row_offset;
        }
        case CompareOp::Lt:
        case CompareOp::Le:
            return 0;  // can't skip — matching rows are at the front
        case CompareOp::Ne:
            return 0;  // can't skip
    }
    return 0;
}
}  // namespace

ScanOperator::ScanOperator(const Table& table, std::optional<Predicate> predicate,
                           std::uint32_t target_batch_size, std::vector<std::size_t> projected_cols)
    : table_(table),
      predicate_(std::move(predicate)),
      target_batch_size_(target_batch_size),
      projected_cols_(std::move(projected_cols)) {}

std::optional<Batch> ScanOperator::next() {
    while (true) {
        // All segments exhausted — signal end of stream.
        if (seg_idx_ >= table_.num_segments()) return std::nullopt;

        if (!seg_initialized_) {
            // First visit to this segment: check zone map before reading any rows.
            if (predicate_) {
                const ZoneMap& zm = table_.segment(seg_idx_).zone(predicate_->col_idx);
                if (!zone_map_can_match(zm, predicate_->op, predicate_->value)) {
                    // Zone map proves no row in this segment can match — skip it entirely.
                    ++segments_pruned_;
                    ++seg_idx_;
                    row_idx_ = 0;
                    continue;
                }
            }
            // Segment survived pruning; count it and mark initialised.
            ++segments_scanned_;

            // Sparse index check (new): if predicate is on the sort key, find the
            // starting granule.
            if (predicate_ && table_.sort_key_col() &&
                *table_.sort_key_col() == predicate_->col_idx) {
                const auto& idx = table_.segment(seg_idx_).primary_index();
                if (!idx.empty()) {
                    std::size_t new_row = find_starting_row_(idx, *predicate_);
                    std::cerr << "seg " << seg_idx_ << ": index has " << idx.size() << " entries, "
                              << "skipping to row " << new_row << " of " << table_.segment(seg_idx_).num_rows() << "\n";
                    row_idx_ = new_row;
                } else {
                    std::cerr << "seg " << seg_idx_ << ": empty primary_index\n";
                }
            }

            seg_initialized_ = true;
        }

        const Segment& seg       = table_.segment(seg_idx_);
        std::size_t    remaining = seg.num_rows() - row_idx_;

        // Segment exhausted — move to the next one.
        if (remaining == 0) {
            ++seg_idx_;
            row_idx_         = 0;
            seg_initialized_ = false;
            continue;
        }

        // Clamp to target_batch_size_ so no single batch grows unboundedly.
        std::size_t batch_size = std::min(remaining, static_cast<std::size_t>(target_batch_size_));

        // Borrow pointers to projected columns only (or all if no projection).
        std::vector<const Vector*> vecs;
        if (projected_cols_.empty()) {
            vecs.reserve(seg.num_columns());
            for (std::size_t i = 0; i < seg.num_columns(); ++i)
                vecs.push_back(&seg.column(i));
        } else {
            vecs.reserve(projected_cols_.size());
            for (std::size_t i : projected_cols_)
                vecs.push_back(&seg.column(i));
        }

        // Record where in the column arrays this batch starts, then advance the cursor.
        std::uint32_t offset = static_cast<std::uint32_t>(row_idx_);
        row_idx_ += batch_size;
        return Batch{std::move(vecs), offset, static_cast<std::uint32_t>(batch_size)};
    }
}
}  // namespace wb