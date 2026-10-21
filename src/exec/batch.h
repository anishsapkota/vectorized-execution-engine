#pragma once

#include <cstdint>
#include <span>

#include "storage/vector.h"

namespace wb {

/*
A batch is a non-owning view over a set of column vectors plus an optional selection vector
identifying which rows are "live". Both the column list and the selection vector are borrowed from
buffers owned by the operator that produced the batch, so building a batch allocates nothing.
Lifetime: a batch is valid only until the next call to next() on the operator that produced it (or
any operator upstream of it), and only as long as the underlying segment remains valid. Operators
must not store batches across next() calls.
*/
class Batch {
   public:
    Batch() = default;

    // Construct from borrowed vectors. offset is the first live row within the
    // column arrays; size is the number of live rows starting at offset.
    Batch(std::span<const Vector* const> columns, std::uint32_t offset, std::uint32_t size)
        : columns_(columns),
          offset_(offset),
          size_(size) {}

    // First row index within the borrowed column arrays that belongs to this batch.
    // Valid only when has_sel() is false; sel entries are absolute and do not need to be
    // offset-corrected.
    std::uint32_t offset() const { return offset_; }

    // Number of live rows in this batch (after selection)
    std::uint32_t size() const { return size_; }

    // Number of columns in this batch
    std::size_t num_columns() const { return columns_.size(); }

    // Borrowed access to the i-th column
    const Vector& column(std::size_t i) const { return *columns_[i]; }

    // Optional selection vector. If absent, rows offset()..offset()+size()-1 are all live.
    // If present, sel().size() == size() and entries are indices into the
    // underlying column vectors.
    bool                           has_sel() const { return has_sel_; }
    std::span<const std::uint32_t> sel() const { return sel_; }

    void set_sel(std::span<const std::uint32_t> sel) {
        size_    = static_cast<std::uint32_t>(sel.size());
        sel_     = sel;
        has_sel_ = true;
    }

   private:
    std::span<const Vector* const> columns_;
    std::span<const std::uint32_t> sel_;
    std::uint32_t                  offset_  = 0;
    std::uint32_t                  size_    = 0;
    bool                           has_sel_ = false;
};

// Calls f(row) for every live row, where row is an absolute index into the batch's columns.
// Hides the "selection vector or contiguous range" split that every consumer otherwise repeats.
template <typename F>
void for_each_live_row(const Batch& batch, F&& f) {
    if (batch.has_sel()) {
        for (std::uint32_t row : batch.sel())
            f(row);
        return;
    }
    const std::uint32_t end = batch.offset() + batch.size();
    for (std::uint32_t row = batch.offset(); row < end; ++row)
        f(row);
}

}  // namespace wb
