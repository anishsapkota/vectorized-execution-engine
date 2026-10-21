#include "dist/planner.h"

#include <optional>
#include <stdexcept>

namespace wb {

const char* join_dist_label(JoinDist strategy) {
    switch (strategy) {
        case JoinDist::DistNone:
            return "DS_DIST_NONE";
        case JoinDist::DistAllNone:
            return "DS_DIST_ALL_NONE";
        case JoinDist::DistInner:
            return "DS_DIST_INNER";
        case JoinDist::DistOuter:
            return "DS_DIST_OUTER";
        case JoinDist::BcastInner:
            return "DS_BCAST_INNER";
        case JoinDist::DistBoth:
            return "DS_DIST_BOTH";
    }
    return "unknown";
}

namespace {

// KEY distribution helps only when it is on the join column: then equal join keys already
// share a slice.
bool keyed_on_join(const JoinSide& side) {
    return side.distribution.style == DistStyle::Key &&
           side.distribution.key_col == side.join_key_col;
}

bool is_all(const JoinSide& side) { return side.distribution.style == DistStyle::All; }

}  // namespace

bool is_valid_strategy(JoinDist strategy, const JoinSide& outer, const JoinSide& inner) {
    // Every slice scans its node's full copy of an ALL table, so an ALL outer would emit each
    // match once per slice, and moving an ALL inner would send every row once per slice.
    if (is_all(outer)) return false;

    switch (strategy) {
        case JoinDist::DistNone:
            return keyed_on_join(outer) && keyed_on_join(inner);
        case JoinDist::DistAllNone:
            return is_all(inner);
        case JoinDist::DistInner:
            return keyed_on_join(outer) && !is_all(inner);
        case JoinDist::DistOuter:
            return keyed_on_join(inner);
        case JoinDist::BcastInner:
        case JoinDist::DistBoth:
            return !is_all(inner);
    }
    return false;
}

JoinDist choose_join_strategy(const JoinSide& outer, const JoinSide& inner, std::size_t num_nodes) {
    if (num_nodes == 0) throw std::invalid_argument("choose_join_strategy: num_nodes must be > 0");

    // Rows sent across nodes, scaled by N so every cost is an integer:
    //   true cost = scaled / N.
    const std::uint64_t n = num_nodes, a = outer.estimated_rows, b = inner.estimated_rows;
    auto scaled_cost = [&](JoinDist strategy) -> std::uint64_t {
        switch (strategy) {
            case JoinDist::DistNone:
            case JoinDist::DistAllNone:
                return 0;
            case JoinDist::DistInner:
                return b * (n - 1);
            case JoinDist::DistOuter:
                return a * (n - 1);
            case JoinDist::BcastInner:
                return b * (n - 1) * n;
            case JoinDist::DistBoth:
                return (a + b) * (n - 1);
        }
        return 0;
    };

    std::optional<JoinDist> best;
    std::uint64_t           best_cost = 0;
    for (JoinDist strategy : {JoinDist::DistNone, JoinDist::DistAllNone, JoinDist::DistInner,
                              JoinDist::DistOuter, JoinDist::BcastInner, JoinDist::DistBoth}) {
        if (!is_valid_strategy(strategy, outer, inner)) continue;
        const std::uint64_t cost = scaled_cost(strategy);
        if (!best || cost < best_cost) {  // strict: ties keep the earlier enum value
            best      = strategy;
            best_cost = cost;
        }
    }
    if (!best) throw std::invalid_argument("choose_join_strategy: no valid strategy");
    return *best;
}

}  // namespace wb
