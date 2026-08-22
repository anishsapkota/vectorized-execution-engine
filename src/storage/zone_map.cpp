#include "storage/zone_map.h"

#include <cassert>

namespace wb {

void ZoneMap::observe(int64_t v) {
    if (!has_values_) {
        has_values_ = true;
        type_       = DataType::Int64;
        min_        = v;
        max_        = v;
        return;
    }

    if (v < std::get<int64_t>(min_)) min_ = v;
    if (v > std::get<int64_t>(max_)) max_ = v;
}

void ZoneMap::observe(double v) {
    if (!has_values_) {
        has_values_ = true;
        type_       = DataType::Float64;
        min_        = v;
        max_        = v;
        return;
    }
    if (v < std::get<double>(min_)) min_ = v;
    if (v > std::get<double>(max_)) max_ = v;
}

void ZoneMap::observe(const std::string& v) {
    if (!has_values_) {
        has_values_ = true;
        type_       = DataType::String;
        min_        = v;
        max_        = v;
        return;
    }
    if (v < std::get<std::string>(min_)) min_ = v;
    if (v > std::get<std::string>(max_)) max_ = v;
}

int64_t ZoneMap::min_i64() const {
    assert(has_values_ && type_ == DataType::Int64);
    return std::get<int64_t>(min_);
}

int64_t ZoneMap::max_i64() const {
    assert(has_values_ && type_ == DataType::Int64);
    return std::get<int64_t>(max_);
}

double ZoneMap::min_f64() const {
    assert(has_values_ && type_ == DataType::Float64);
    return std::get<double>(min_);
}

double ZoneMap::max_f64() const {
    assert(has_values_ && type_ == DataType::Float64);
    return std::get<double>(max_);
}

const std::string& ZoneMap::min_str() const {
    assert(has_values_ && type_ == DataType::String);
    return std::get<std::string>(min_);
}

const std::string& ZoneMap::max_str() const {
    assert(has_values_ && type_ == DataType::String);
    return std::get<std::string>(max_);
}
}  // namespace wb