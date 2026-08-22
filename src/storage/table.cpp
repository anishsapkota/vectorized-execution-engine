#include "storage/table.h"

#include <cassert>
#include <optional>

namespace wb {

Table::Table(std::string name, Schema schema, std::optional<std::size_t> sort_key_col)
    : sort_key_col_(sort_key_col), name_(std::move(name)), schema_(std::move(schema)) {}

void Table::add_segment(std::unique_ptr<Segment> segment) {
    segments_.push_back(std::move(segment));
}

const std::string& Table::name() const { return name_; }

const Schema& Table::schema() const { return schema_; }

std::size_t Table::num_segments() const { return segments_.size(); }

const Segment& Table::segment(std::size_t i) const {
    assert(i < segments_.size());
    return *segments_[i];
}

const std::vector<std::unique_ptr<Segment>>& Table::segments() const { return segments_; }

}  // namespace wb
