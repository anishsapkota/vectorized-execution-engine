#include "exec/predicate.h"

#include <cassert>

namespace wb {

// Single implementation of the pruning logic, shared across all three value
// types via template. T must support <, <=, ==, !=.
// The question asked: given that every value in this column is in [zmin, zmax],
// can "col op v" ever be true?
template <typename T>
static bool can_match(CompareOp op, T v, T zmin, T zmax) {
    switch (op) {
        // col == v is possible only if v falls inside [zmin, zmax].
        case CompareOp::Eq:
            return zmin <= v && v <= zmax;
        // col != v is impossible only when every value is v (zmin == zmax == v).
        case CompareOp::Ne:
            return zmin != v || zmax != v;
        // col < v is possible only if the smallest value is less than v.
        case CompareOp::Lt:
            return zmin < v;
        // col <= v is possible only if the smallest value is <= v.
        case CompareOp::Le:
            return zmin <= v;
        // col > v is possible only if the largest value is greater than v.
        case CompareOp::Gt:
            return zmax > v;
        // col >= v is possible only if the largest value is >= v.
        case CompareOp::Ge:
            return zmax >= v;
    }
    return true;  // unreachable; silences -Wreturn-type
}

bool zone_map_can_match(const ZoneMap& zm, CompareOp op,
                        const std::variant<int64_t, double, std::string>& value) {
    // Empty segment — no rows to prune, so conservatively say "might match".
    if (!zm.has_values()) return true;

    switch (zm.type()) {
        case DataType::Int64: {
            assert(std::holds_alternative<int64_t>(value));
            int64_t v    = std::get<int64_t>(value);
            int64_t zmin = zm.min_i64();
            int64_t zmax = zm.max_i64();
            return can_match(op, v, zmin, zmax);
        }
        case DataType::Float64: {
            assert(std::holds_alternative<double>(value));
            double v    = std::get<double>(value);
            double zmin = zm.min_f64();
            double zmax = zm.max_f64();  // was zm.max_i64() — wrong type
            return can_match(op, v, zmin, zmax);
        }
        case DataType::String: {
            assert(std::holds_alternative<std::string>(value));
            const std::string& v    = std::get<std::string>(value);
            const std::string& zmin = zm.min_str();
            const std::string& zmax = zm.max_str();
            return can_match(op, v, zmin, zmax);
        }
    }
    return true;  // unreachable
}

}  // namespace wb