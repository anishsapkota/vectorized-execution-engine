#include "dist/distribute.h"

#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "storage/segment.h"
#include "storage/vector.h"

namespace wb {

namespace {

constexpr std::uint32_t kGranuleSize = 8192;

// Appends `rows` of `seg` to `builder`, one column at a time, so the type and encoding switch
// runs once per column per call instead of once per value.
void append_rows(SegmentBuilder& builder, const Segment& seg, const Schema& schema,
                 std::span<const std::uint32_t> rows) {
    for (std::size_t c = 0; c < schema.size(); ++c) {
        switch (schema[c].type) {
            case DataType::Int64:
                static_cast<const Int64Vector&>(seg.column(c)).visit([&](auto codes, int64_t base) {
                    for (std::uint32_t r : rows)
                        builder.append_int64(c, Int64Vector::decode(base, codes[r]));
                });
                break;
            case DataType::Float64:
                static_cast<const Float64Vector&>(seg.column(c)).visit([&](auto values) {
                    for (std::uint32_t r : rows)
                        builder.append_float64(c, static_cast<double>(values[r]));
                });
                break;
            case DataType::String: {
                const auto& values = static_cast<const StringVector&>(seg.column(c)).data();
                for (std::uint32_t r : rows)
                    builder.append_string(c, values[r]);
                break;
            }
        }
    }
}

// One SegmentBuilder per destination part; seals a segment whenever one reaches
// rows_per_segment.
class PartWriter {
   public:
    PartWriter(Cluster& cluster, const std::string& name, const Schema& schema,
               std::optional<std::size_t> sort_key_col, std::vector<std::size_t> dest_slices,
               std::size_t rows_per_segment)
        : cluster_(cluster),
          name_(name),
          schema_(schema),
          sort_key_col_(sort_key_col),
          dest_slices_(std::move(dest_slices)),
          rows_per_segment_(rows_per_segment) {
        for (std::size_t d = 0; d < dest_slices_.size(); ++d)
            builders_.push_back(std::make_unique<SegmentBuilder>(schema_));
    }

    std::size_t num_parts() const { return dest_slices_.size(); }

    void push(std::size_t part, const Segment& seg, std::span<const std::uint32_t> rows) {
        while (!rows.empty()) {
            SegmentBuilder& builder = *builders_[part];
            std::size_t     room    = rows_per_segment_ - builder.num_rows();
            auto            take    = rows.first(std::min(room, rows.size()));
            append_rows(builder, seg, schema_, take);
            rows = rows.subspan(take.size());
            if (builder.num_rows() == rows_per_segment_) flush(part);
        }
    }

    void finish() {
        for (std::size_t part = 0; part < builders_.size(); ++part) {
            if (builders_[part]->num_rows() > 0) flush(part);
        }
    }

   private:
    void flush(std::size_t part) {
        cluster_.local_part(dest_slices_[part], name_)
            .add_segment(builders_[part]->seal(sort_key_col_, kGranuleSize));
        builders_[part] = std::make_unique<SegmentBuilder>(schema_);
    }

    Cluster&                                     cluster_;
    const std::string&                           name_;
    const Schema&                                schema_;
    std::optional<std::size_t>                   sort_key_col_;
    std::vector<std::size_t>                     dest_slices_;
    std::size_t                                  rows_per_segment_;
    std::vector<std::unique_ptr<SegmentBuilder>> builders_;
};

}  // namespace

void distribute_table(Cluster& cluster, const Table& source, const std::string& name,
                      Distribution distribution, std::size_t rows_per_segment) {
    if (rows_per_segment == 0) throw std::invalid_argument("rows_per_segment must be > 0");

    const Schema& schema = source.schema();
    cluster.declare_table(name, schema, distribution, source.sort_key_col());

    // Parts that receive rows: every slice, or each node's first slice for ALL.
    std::vector<std::size_t> dest_slices;
    if (distribution.style == DistStyle::All) {
        for (std::size_t node = 0; node < cluster.num_nodes(); ++node)
            dest_slices.push_back(cluster.first_slice_of(node));
    } else {
        for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice)
            dest_slices.push_back(slice);
    }
    PartWriter writer{cluster,      name, schema, source.sort_key_col(), std::move(dest_slices),
                      rows_per_segment};

    // Reused across segments: row indices of the current segment, per destination part.
    std::vector<std::vector<std::uint32_t>> rows_for(writer.num_parts());
    std::size_t                             next_even = 0;  // round-robin cursor, table-wide

    for (const auto& seg_ptr : source.segments()) {
        const Segment&      seg = *seg_ptr;
        const std::uint32_t n   = static_cast<std::uint32_t>(seg.num_rows());
        for (auto& rows : rows_for)
            rows.clear();

        switch (distribution.style) {
            case DistStyle::Key: {
                const std::size_t num_slices = cluster.num_slices();
                static_cast<const Int64Vector&>(seg.column(distribution.key_col))
                    .visit([&](auto codes, int64_t base) {
                        for (std::uint32_t r = 0; r < n; ++r) {
                            const int64_t key = Int64Vector::decode(base, codes[r]);
                            rows_for[slice_for_key(key, num_slices)].push_back(r);
                        }
                    });
                break;
            }
            case DistStyle::Even:
                for (std::uint32_t r = 0; r < n; ++r) {
                    rows_for[next_even].push_back(r);
                    next_even = (next_even + 1) % rows_for.size();
                }
                break;
            case DistStyle::All:
                for (std::uint32_t r = 0; r < n; ++r)
                    rows_for[0].push_back(r);
                for (std::size_t part = 1; part < rows_for.size(); ++part)
                    rows_for[part] = rows_for[0];
                break;
        }

        for (std::size_t part = 0; part < rows_for.size(); ++part)
            writer.push(part, seg, rows_for[part]);
    }
    writer.finish();
}

}  // namespace wb
