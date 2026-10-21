#pragma once

#include <cstddef>
#include <string>

#include "dist/cluster.h"
#include "storage/table.h"

namespace wb {

// The loader. Spreads `source` (a table on the leader) over the cluster's slices as table
// `name`, according to `distribution`.
//
// Contract:
//   - Declares `name` in the cluster with source's schema, the given distribution and source's
//     sort key, then fills each slice's part via cluster.local_part(slice, name).
//   - KEY:  each row goes to slice_for_key(row[key_col], num_slices). Must be exactly that
//           function — Exchange::redistribute uses it too, and co-location depends on it.
//   - EVEN: round-robin over all slices, so part sizes differ by at most one row.
//   - ALL:  one full copy per node, on cluster.first_slice_of(node).
//   - Each part is cut into segments of at most rows_per_segment rows, sealed with source's sort
//     key so zone maps and the sparse index keep working per slice.
// Rows are routed per source segment, then copied column-at-a-time into one SegmentBuilder per
// destination, so the type switch runs per column, not per value.
void distribute_table(Cluster& cluster, const Table& source, const std::string& name,
                      Distribution distribution, std::size_t rows_per_segment = 65536);

}  // namespace wb
