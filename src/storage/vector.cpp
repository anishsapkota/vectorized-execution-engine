
#include "vector.h"

#include <algorithm>
#include <limits>
#include <type_traits>

namespace wb {

const char* type_name(DataType t) {
    switch (t) {
        case DataType::Int64:
            return "int64";
        case DataType::Float64:
            return "float64";
        case DataType::String:
            return "string";
    }
    return "unknown";  // unreachable, but pacifies -Wreturn-type
}

Int64Vector::Int64Vector(std::span<const int64_t> values) : size_(values.size()) {
    if (values.empty()) return;  // codes_ holds an empty uint8_t vector

    auto [lo, hi]  = std::minmax_element(values.begin(), values.end());
    base_          = *lo;
    uint64_t range = static_cast<uint64_t>(*hi) - static_cast<uint64_t>(*lo);

    auto encode = [&]<typename Code>(std::type_identity<Code>) {
        std::vector<Code> codes(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            codes[i] =
                static_cast<Code>(static_cast<uint64_t>(values[i]) - static_cast<uint64_t>(base_));
        codes_ = std::move(codes);
    };

    if (range <= std::numeric_limits<uint8_t>::max()) {
        encode(std::type_identity<uint8_t>{});
    } else if (range <= std::numeric_limits<uint16_t>::max()) {
        encode(std::type_identity<uint16_t>{});
    } else if (range <= std::numeric_limits<uint32_t>::max()) {
        encode(std::type_identity<uint32_t>{});
    } else {
        encode(std::type_identity<uint64_t>{});
    }
}

std::size_t Int64Vector::memory_bytes() const {
    return std::visit(
        [](const auto& codes) { return codes.capacity() * sizeof(codes[0]); }, codes_);
}

std::size_t Int64Vector::width_bytes() const {
    return std::visit([](const auto& codes) { return sizeof(codes[0]); }, codes_);
}

int64_t Int64Vector::value(std::size_t i) const {
    return std::visit([&](const auto& codes) { return decode(base_, codes[i]); }, codes_);
}

Float64Vector::Float64Vector(std::vector<double> values, bool narrow) {
    if (!narrow) {
        values_ = std::move(values);
        return;
    }
    values_ = std::vector<float>(values.begin(), values.end());
}

std::size_t Float64Vector::size() const {
    return std::visit([](const auto& values) { return values.size(); }, values_);
}

std::size_t Float64Vector::memory_bytes() const {
    return std::visit(
        [](const auto& values) { return values.capacity() * sizeof(values[0]); }, values_);
}

double Float64Vector::value(std::size_t i) const {
    return std::visit([&](const auto& values) { return static_cast<double>(values[i]); }, values_);
}

}  // namespace wb
