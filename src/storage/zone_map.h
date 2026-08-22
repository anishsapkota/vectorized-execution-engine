#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <variant>

#include "storage/schema.h"

namespace wb {
/*
A zone map summarises one column within one segment, holding the exact min and
max observed value. Used by the optimiser/scan operator to decide whether a
segment can possibly contain rows matching a predicate.

We use std::variant rather than a separate type per column to keep the Segment
API uniform - every column has a ZoneMap, regardless of type.
*/

class ZoneMap {
   private:
    bool     has_values_ = false;
    DataType type_       = DataType::Int64;

    std::variant<int64_t, double, std::string> min_;
    std::variant<int64_t, double, std::string> max_;

   public:
    // Empty zone map (no rows observed yet)
    ZoneMap() = default;

    void observe(int64_t v);
    void observe(double v);
    void observe(const std::string& v);

    bool     has_values() const { return has_values_; }
    DataType type() const { return type_; }

    int64_t            min_i64() const;
    int64_t            max_i64() const;
    double             min_f64() const;
    double             max_f64() const;
    const std::string& min_str() const;
    const std::string& max_str() const;
};

}  // namespace wb