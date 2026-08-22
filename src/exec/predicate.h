#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "storage/zone_map.h"

namespace wb {

enum class CompareOp {
    Eq,  // ==
    Ne,  // !=
    Lt,  // <
    Le,  // <=
    Gt,  // >
    Ge   // >=
};

// A predicate over a single column. Later we ll combine these with AND
struct Predicate {
    // col_idx is interpreted relative to the input the operator sees:
    //   - In ScanOperator, this is the original table schema index (used to
    //     index into the segment's zone-map array).
    //   - In FilterOperator and HashAggOperator, this is the position within
    //     the batch the operator receives (which reflects any upstream
    //     projection).
    // Callers must construct distinct Predicate instances for each operator
    // if a projection sits between scan and filter. A planner would do this
    // mapping automatically.
    std::size_t                                col_idx;
    CompareOp                                  op;
    std::variant<int64_t, double, std::string> value;
};

// Return true if a segment with the given zone map CAN POSSIBLY contain rows
// matching the predicate. Returns false only if the zone map proves no row
// in the segment matches. (A "false negative" - saying "might match" when actually no rows do - is
// safe; a "false positive" - saying "no match" when some row does - would corrupt results.)
bool zone_map_can_match(const ZoneMap& zm, CompareOp op,
                        const std::variant<int64_t, double, std::string>& value);

}  // namespace wb