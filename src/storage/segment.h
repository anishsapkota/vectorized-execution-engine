#pragma once

#include <vector>

#include "storage/schema.h"
#include "storage/vector.h"
#include "storage/zone_map.h"

namespace wb {

struct IndexEntry {
    int64_t       sort_key_value;  // value at row_offset in the sort-key column
    std::uint32_t row_offset;      // first row of this granule
};

class Segment {
   public:
    Segment(std::size_t num_rows, std::vector<std::unique_ptr<Vector>> columns,
            std::vector<ZoneMap> zones, std::vector<IndexEntry> primary_index)
        : num_rows_(num_rows),
          columns_(std::move(columns)),
          zones_(std::move(zones)),
          primary_index_(std::move(primary_index)) {}

    std::size_t                    num_rows() const { return num_rows_; }
    std::size_t                    num_columns() const { return columns_.size(); }
    const Vector&                  column(std::size_t i) const { return *columns_[i]; }
    const ZoneMap&                 zone(std::size_t i) const { return zones_[i]; }
    const std::vector<IndexEntry>& primary_index() const { return primary_index_; }

   private:
    std::size_t                          num_rows_;
    std::vector<std::unique_ptr<Vector>> columns_;
    std::vector<ZoneMap>                 zones_;
    std::vector<IndexEntry>              primary_index_;
};

class SegmentBuilder {
   public:
    explicit SegmentBuilder(Schema schema);
    // Append one row. Values must match the schema in order and type.
    void append_int64(std::size_t col_idx, int64_t value);
    void append_float64(std::size_t col_idx, double value);
    void append_string(std::size_t col_idx, std::string value);

    // Finalise into an immutable Segment. The builder is left empty;
    // construct a new one to build another segment.
    std::unique_ptr<Segment> seal(std::optional<std::size_t> sort_key_col,
                                  std::uint32_t              granule_size);
    std::size_t              num_rows() const { return num_rows_; }

   private:
    Schema                               schema_;
    std::vector<std::unique_ptr<Vector>> columns_;
    std::vector<ZoneMap>                 zones_;
    std::size_t                          num_rows_ = 0;
};

}  // namespace wb