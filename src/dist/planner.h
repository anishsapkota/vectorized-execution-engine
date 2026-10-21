#pragma once

#include <cstddef>
#include <cstdint>

#include "dist/cluster.h"

namespace wb {

// Join data-movement strategies, named after the labels in Redshift's EXPLAIN output.
// "Outer" is the probe side, "inner" the build side.
enum class JoinDist : std::uint8_t {
    DistNone,     // DS_DIST_NONE:     both sides KEY-distributed on the join key; no movement
    DistAllNone,  // DS_DIST_ALL_NONE: inner is ALL; every node already has all of it
    DistInner,    // DS_DIST_INNER:    redistribute inner on its join key to match outer
    DistOuter,    // DS_DIST_OUTER:    redistribute outer on its join key to match inner
    BcastInner,   // DS_BCAST_INNER:   send all of inner to every node
    DistBoth      // DS_DIST_BOTH:     redistribute both sides on their join keys
};

const char* join_dist_label(JoinDist strategy);

struct JoinSide {
    Distribution  distribution;
    std::size_t   join_key_col   = 0;
    std::uint64_t estimated_rows = 0;
};

// Can `strategy` produce a correct result for these two sides, given what each slice holds
// after movement? KEY distribution counts only when it is on the join column, and an ALL table
// is scanned in full on every slice, so it must never be moved. An ALL outer is rejected
// (Redshift doesn't plan that shape either).
bool is_valid_strategy(JoinDist strategy, const JoinSide& outer, const JoinSide& inner);

// The planner rule. Among valid strategies, pick the one that sends the fewest
// rows across nodes. Cost model, with N = num_nodes, |A| outer rows, |B| inner rows:
//   DistNone, DistAllNone  0
//   DistInner              |B| * (N-1)/N
//   DistOuter              |A| * (N-1)/N
//   BcastInner             |B| * (N-1)
//   DistBoth               (|A| + |B|) * (N-1)/N
// Break ties by enum order (prefer the strategies listed first). Throws std::invalid_argument if
// no strategy is valid.
JoinDist choose_join_strategy(const JoinSide& outer, const JoinSide& inner, std::size_t num_nodes);

}  // namespace wb
