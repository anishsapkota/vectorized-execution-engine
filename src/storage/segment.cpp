#include "storage/segment.h"

#include <cassert>
#include <memory>
#include <utility>

#include "storage/vector.h"

namespace wb {

namespace {
void permute_column(Vector& col, std::span<const std::uint32_t> perm) {
    switch (col.type()) {
        case DataType::Int64: {
            auto&                v = static_cast<Int64Vector&>(col);
            std::vector<int64_t> out(perm.size());
            auto                 src = v.data();
            for (std::size_t i = 0; i < perm.size(); ++i)
                out[i] = src[perm[i]];
            v.reset(std::move(out));  // see below
            break;
        }
        case DataType::Float64: {
            auto&               v = static_cast<Float64Vector&>(col);
            std::vector<double> out(perm.size());
            auto                src = v.data();
            for (std::size_t i = 0; i < perm.size(); ++i)
                out[i] = src[perm[i]];
            v.reset(std::move(out));
            break;
        }
        case DataType::String: {
            auto&                    v = static_cast<StringVector&>(col);
            std::vector<std::string> out(perm.size());
            auto                     src = v.data();
            for (std::size_t i = 0; i < perm.size(); ++i)
                out[i] = std::move(src[perm[i]]);
            v.reset(std::move(out));
            break;
        }
    }
}
}  // namespace

SegmentBuilder::SegmentBuilder(Schema schema) : schema_(std::move(schema)) {
    // Walk each column definition in the schema.
    for (const auto& col : schema_) {
        // Allocate the right typed vector depending on the column's declared type.
        // Each Vector subclass stores one contiguous array of values of that type.
        switch (col.type) {
            case DataType::Int64:
                columns_.push_back(std::make_unique<Int64Vector>());
                break;
            case DataType::Float64:
                columns_.push_back(std::make_unique<Float64Vector>());
                break;
            case DataType::String:
                columns_.push_back(std::make_unique<StringVector>());
                break;
        }
        // Pair each column with an empty ZoneMap that will track its min/max as
        // values are appended.
        zones_.emplace_back();
    }
}

void SegmentBuilder::append_int64(std::size_t col_idx, int64_t value) {
    // Guard: col_idx must refer to an actual column.
    assert(col_idx < columns_.size());
    // Guard: the schema says this column holds int64; caller must honour that.
    assert(schema_[col_idx].type == DataType::Int64);
    // Downcast the base Vector* to Int64Vector* so we can call append().
    // Safe because we constructed an Int64Vector for every Int64 column above.
    static_cast<Int64Vector*>(columns_[col_idx].get())->append(value);
    // Update the column's min/max statistics with the new value.
    zones_[col_idx].observe(value);
    // Count rows at col 0: the counter reflects "rows started", not "rows
    // completed", so partial rows don't leave it stale.
    if (col_idx == 0) ++num_rows_;
}

void SegmentBuilder::append_float64(std::size_t col_idx, double value) {
    assert(col_idx < columns_.size());
    assert(schema_[col_idx].type == DataType::Float64);
    static_cast<Float64Vector*>(columns_[col_idx].get())->append(value);
    zones_[col_idx].observe(value);
    if (col_idx == 0) ++num_rows_;
}

void SegmentBuilder::append_string(std::size_t col_idx, std::string value) {
    assert(col_idx < columns_.size());
    assert(schema_[col_idx].type == DataType::String);
    // Observe BEFORE moving: after std::move, value is in a valid-but-unspecified
    // state (typically empty). Observing the empty string would corrupt the zone
    // map's min — "" sorts before every non-empty string.
    zones_[col_idx].observe(value);
    // Move the string into storage to avoid a heap allocation copy.
    static_cast<StringVector*>(columns_[col_idx].get())->append(std::move(value));
    if (col_idx == 0) ++num_rows_;
}

std::unique_ptr<Segment> SegmentBuilder::seal(std::optional<std::size_t> sort_key_col,
                                              std::uint32_t              granule_size) {
    // Catch any caller that forgot to append a value to one or more columns.
    // All columns must have identical lengths — a mismatch means a row is incomplete.
    // This is the only safe moment to detect it: the partial row can't be fixed after seal.
    for (std::size_t i = 1; i < columns_.size(); ++i) {
        assert(columns_[i]->size() == columns_[0]->size() &&
               "all columns must have the same length");
    }

    if (sort_key_col) {
        // Sort all columns in lockstep by the sort-key column.
        const std::size_t          n = num_rows_;
        std::vector<std::uint32_t> perm(n);
        for (std::uint32_t i = 0; i < n; ++i)
            perm[i] = i;
        assert(schema_[*sort_key_col].type == DataType::Int64);
        const auto& key_data = static_cast<const Int64Vector&>(*columns_[*sort_key_col]).data();
        std::sort(perm.begin(), perm.end(),
                  [&](std::uint32_t a, std::uint32_t b) { return key_data[a] < key_data[b]; });

        for (std::size_t c = 0; c < columns_.size(); ++c) {
            permute_column(*columns_[c], perm);
        }
    }

    // Build sparse primary index (if sorted).
    std::vector<IndexEntry> index;
    if (sort_key_col) {
        const auto& key_data = static_cast<const Int64Vector&>(*columns_[*sort_key_col]).data();
        for (std::uint32_t off = 0; off < num_rows_; off += granule_size) {
            index.push_back({key_data[off], off});
        }
    }

    // Transfer ownership of columns and zone maps into the immutable Segment.
    // After this move, columns_ and zones_ are empty — the builder is exhausted.
    return std::make_unique<Segment>(num_rows_, std::move(columns_), std::move(zones_),
                                     std::move(index));
}

}  // namespace wb