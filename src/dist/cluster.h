#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "dist/network.h"
#include "storage/schema.h"
#include "storage/table.h"

namespace wb {

// How a table's rows are spread over slices. Mirrors Redshift's DISTSTYLE.
enum class DistStyle : std::uint8_t {
    Even,  // round-robin over all slices
    Key,   // row goes to slice_for_key(row[key_col]); equal keys always share a slice
    All    // one full copy per node, stored on the node's first slice
};

struct Distribution {
    DistStyle   style   = DistStyle::Even;
    std::size_t key_col = 0;  // meaningful only for DistStyle::Key
};

const char* dist_style_name(DistStyle style);

// The single function that decides which slice owns a key. KEY distribution at load time and
// Redistribute at query time must both use it: if they disagree, tables that look co-located
// are not, and DS_DIST_NONE joins silently drop matches.
//
// Uses the high 32 bits of the multiplicative hash. AggHashTable buckets on the low bits of the
// same hash; partitioning on the low bits too would hand every slice keys that all collide into
// the same fraction of its hash-table buckets.
inline std::size_t slice_for_key(int64_t key, std::size_t num_slices) {
    const std::uint64_t h = static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ULL;
    return static_cast<std::size_t>((h >> 32) % num_slices);
}

struct ClusterConfig {
    std::size_t nodes           = 2;
    std::size_t slices_per_node = 2;
};

// A simulated Redshift cluster: nodes x slices, each slice holding its own part of every table.
// The leader is whoever owns the Cluster (main thread); slices are driven by run_stream().
class Cluster {
   public:
    explicit Cluster(ClusterConfig config);

    std::size_t num_nodes() const { return config_.nodes; }
    std::size_t slices_per_node() const { return config_.slices_per_node; }
    std::size_t num_slices() const { return config_.nodes * config_.slices_per_node; }
    std::size_t node_of(std::size_t slice) const { return slice / config_.slices_per_node; }
    std::size_t first_slice_of(std::size_t node) const { return node * config_.slices_per_node; }

    // Catalog. Creates an empty part of the table on every slice. Throws std::invalid_argument
    // if the name is taken or a KEY distribution names a non-int64 column.
    void declare_table(const std::string& name, Schema schema, Distribution distribution,
                       std::optional<std::size_t> sort_key_col = std::nullopt);

    bool                has_table(const std::string& name) const;
    const Schema&       schema(const std::string& name) const;
    const Distribution& distribution(const std::string& name) const;

    // The part of a table that `slice` stores, for loading. ALL tables exist only on each node's
    // first slice; asking another slice for its part of one throws std::invalid_argument.
    Table& local_part(std::size_t slice, const std::string& name);

    // What a query running on `slice` scans: its own part, or for an ALL table, its node's copy.
    const Table& scan_target(std::size_t slice, const std::string& name) const;

    // Rows in the logical table. An ALL table counts one node's copy, not every copy.
    std::uint64_t row_count(const std::string& name) const;

    NetworkStats& network() { return network_; }

   private:
    struct TableEntry {
        Schema                              schema;
        Distribution                        distribution;
        std::vector<std::unique_ptr<Table>> parts;  // index = slice
    };

    const TableEntry& entry(const std::string& name) const;

    ClusterConfig                               config_;
    std::unordered_map<std::string, TableEntry> tables_;
    NetworkStats                                network_;
};

}  // namespace wb
