#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "dist/cluster.h"
#include "dist/network.h"
#include "dist/planner.h"
#include "dist/stream.h"

namespace wb {

struct JoinQuery {
    std::string outer_table;
    std::size_t outer_key_col = 0;
    std::string inner_table;
    std::size_t inner_key_col = 0;
};

struct NamedStream {
    std::string  name;
    StreamTiming timing;
};

struct JoinRunResult {
    JoinDist                   strategy = JoinDist::DistNone;
    TrafficSnapshot            traffic;
    std::vector<NamedStream>   streams;          // in execution order; the last one is the join
    std::vector<std::uint64_t> rows_per_slice;   // join output rows produced by each slice
    std::uint64_t              output_rows = 0;
    // Order-independent fingerprint of the result: wrapping sum of every int64 output value.
    // Equal across strategies when they compute the same join, which is what the tests check.
    std::uint64_t checksum = 0;

    double wall_ms() const;
};

// Executes `query` on the cluster with a fixed movement strategy, so experiments can compare
// strategies on the same data. Throws std::invalid_argument if is_valid_strategy rejects it.
JoinRunResult run_distributed_join(Cluster& cluster, const JoinQuery& query, JoinDist strategy);

// The planner's view of a query: distributions and sizes from the cluster catalog.
JoinSide outer_side(const Cluster& cluster, const JoinQuery& query);
JoinSide inner_side(const Cluster& cluster, const JoinQuery& query);

}  // namespace wb
