#include "storage/segment.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <utility>

#include "storage/vector.h"

namespace wb {

namespace {
template <typename T>
void permute(std::vector<T>& values, std::span<const std::uint32_t> perm) {
    std::vector<T> out;
    out.reserve(perm.size());
    for (std::uint32_t p : perm)
        out.push_back(std::move(values[p]));
    values = std::move(out);
}
}  // namespace

SegmentBuilder::SegmentBuilder(Schema schema) : schema_(std::move(schema)) {
    // One raw buffer and one empty ZoneMap per column; the ZoneMap tracks min/max as values
    // are appended.
    for (const auto& col : schema_) {
        switch (col.type) {
            case DataType::Int64:
                buffers_.emplace_back(std::vector<int64_t>{});
                break;
            case DataType::Float64:
                buffers_.emplace_back(std::vector<double>{});
                break;
            case DataType::String:
                buffers_.emplace_back(std::vector<std::string>{});
                break;
        }
        zones_.emplace_back();
    }
}

void SegmentBuilder::append_int64(std::size_t col_idx, int64_t value) {
    assert(col_idx < buffers_.size());
    assert(schema_[col_idx].type == DataType::Int64);
    std::get<std::vector<int64_t>>(buffers_[col_idx]).push_back(value);
    zones_[col_idx].observe(value);
    // Count rows at col 0: the counter reflects "rows started", not "rows
    // completed", so partial rows don't leave it stale.
    if (col_idx == 0) ++num_rows_;
}

void SegmentBuilder::append_float64(std::size_t col_idx, double value) {
    assert(col_idx < buffers_.size());
    assert(schema_[col_idx].type == DataType::Float64);
    std::get<std::vector<double>>(buffers_[col_idx]).push_back(value);
    zones_[col_idx].observe(value);
    if (col_idx == 0) ++num_rows_;
}

void SegmentBuilder::append_string(std::size_t col_idx, std::string value) {
    assert(col_idx < buffers_.size());
    assert(schema_[col_idx].type == DataType::String);
    // Observe BEFORE moving: after std::move, value is in a valid-but-unspecified
    // state (typically empty). Observing the empty string would corrupt the zone
    // map's min — "" sorts before every non-empty string.
    zones_[col_idx].observe(value);
    std::get<std::vector<std::string>>(buffers_[col_idx]).push_back(std::move(value));
    if (col_idx == 0) ++num_rows_;
}

std::unique_ptr<Segment> SegmentBuilder::seal(std::optional<std::size_t> sort_key_col,
                                              std::uint32_t              granule_size) {
    // Catch any caller that forgot to append a value to one or more columns.
    // All columns must have identical lengths — a mismatch means a row is incomplete.
    // This is the only safe moment to detect it: the partial row can't be fixed after seal.
    for (const auto& buf : buffers_) {
        assert(std::visit([](const auto& v) { return v.size(); }, buf) == num_rows_ &&
               "all columns must have the same length");
        (void)buf;
    }

    // Build sparse primary index (if sorted) from the raw key values, before encoding.
    std::vector<IndexEntry> index;
    if (sort_key_col) {
        assert(schema_[*sort_key_col].type == DataType::Int64);
        const auto& keys = std::get<std::vector<int64_t>>(buffers_[*sort_key_col]);

        // Sort all columns in lockstep by the sort-key column.
        std::vector<std::uint32_t> perm(num_rows_);
        std::iota(perm.begin(), perm.end(), 0u);
        std::sort(perm.begin(), perm.end(),
                  [&](std::uint32_t a, std::uint32_t b) { return keys[a] < keys[b]; });
        for (auto& buf : buffers_)
            std::visit([&](auto& values) { permute(values, perm); }, buf);

        for (std::uint32_t off = 0; off < num_rows_; off += granule_size)
            index.push_back({keys[off], off});
    }

    // Encode each buffer into its immutable column.
    std::vector<std::unique_ptr<Vector>> columns;
    columns.reserve(buffers_.size());
    for (std::size_t c = 0; c < buffers_.size(); ++c) {
        switch (schema_[c].type) {
            case DataType::Int64:
                columns.push_back(
                    std::make_unique<Int64Vector>(std::get<std::vector<int64_t>>(buffers_[c])));
                break;
            case DataType::Float64: {
                auto col = std::make_unique<Float64Vector>(
                    std::move(std::get<std::vector<double>>(buffers_[c])), schema_[c].narrow_float);
                if (col->narrow()) {
                    // Rounding to float can move a value past the double min/max; rebuild the
                    // zone map from the stored values so pruning agrees with the filter.
                    ZoneMap zone;
                    col->visit([&](auto values) {
                        for (auto v : values)
                            zone.observe(static_cast<double>(v));
                    });
                    zones_[c] = std::move(zone);
                }
                columns.push_back(std::move(col));
                break;
            }
            case DataType::String:
                columns.push_back(std::make_unique<StringVector>(
                    std::move(std::get<std::vector<std::string>>(buffers_[c]))));
                break;
        }
    }
    buffers_.clear();

    // After this move, zones_ is empty — the builder is exhausted.
    return std::make_unique<Segment>(num_rows_, std::move(columns), std::move(zones_),
                                     std::move(index));
}

}  // namespace wb
