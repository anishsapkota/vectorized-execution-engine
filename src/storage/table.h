#pragma once

#include "storage/schema.h"
#include "storage/segment.h"

namespace wb {

class Table {
   public:
    Table(std::string name, Schema schema, std::optional<std::size_t> sort_key_col = std::nullopt);

    void add_segment(std::unique_ptr<Segment> segment);

    const std::string& name() const;
    const Schema&      schema() const;

    std::size_t    num_segments() const;
    const Segment& segment(std::size_t i) const;

    // For iteration in the scan operator
    const std::vector<std::unique_ptr<Segment>>& segments() const;

    // Index of the sort-key column in the schema, or nullopt is not sorted.
    std::optional<std::size_t> sort_key_col() const { return sort_key_col_; }

   private:
    std::optional<std::size_t>            sort_key_col_;
    std::string                           name_;
    Schema                                schema_;
    std::vector<std::unique_ptr<Segment>> segments_;
};

}  // namespace wb