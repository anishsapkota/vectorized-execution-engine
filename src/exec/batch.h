#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "storage/vector.h"

namespace wb {

/*
A batch is a non-owning view over a set of column vectors plus an optional selection vector
identifying which rows are "live". Lifetime: a batch is valid only as long as the underlying vectors
(and any segment they live in) remain valid. Operators must not store batches accross next() calls.
*/
class Batch {
   public:
    Batch() = default;

    // Construct from borrowed vectors. offset is the first live row within the
    // column arrays; size is the number of live rows starting at offset.
    Batch(std::vector<const Vector*> vectors, std::uint32_t offset, std::uint32_t size)
        : vectors_(std::move(vectors)),
          offset_(offset),
          size_(size) {}

    // First row index within the borrowed column arrays that belongs to this batch.
    // Valid only when sel() is empty; sel entries are absolute and do not need to be
    // offset-corrected"
    std::uint32_t offset() const { return offset_; }

    // Number of live rows in this batch (after selection)
    std::uint32_t size() const { return size_; }

    // Number of columns in this batch
    std::size_t num_columns() const { return vectors_.size(); }

    // Borrowed access to the i-th column
    const Vector& column(std::size_t i) const { return *vectors_[i]; }

    // Optional selection vector. If absent, row 0..size()-1 are all live.
    // If present, sel().size() == size() and entries are indices into the
    // underlying column vectors.
    const std::optional<std::vector<std::uint32_t>>& sel() const { return sel_; }

    void set_sel(std::vector<std::uint32_t> sel) {
        size_ = static_cast<std::uint32_t>(sel.size());
        sel_  = std::move(sel);
    }

   private:
    std::vector<const Vector*>                vectors_;
    std::optional<std::vector<std::uint32_t>> sel_;
    std::uint32_t                             offset_ = 0;
    std::uint32_t                             size_   = 0;
};
}  // namespace wb