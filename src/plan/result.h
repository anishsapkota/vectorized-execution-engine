#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wb::plan {

// The answer to a SumCount query, in the same form from both engines so they can be compared
// exactly. Column-major; keys is empty for a global (ungrouped) aggregate.
struct QueryResult {
    bool                 grouped = false;
    std::vector<int64_t> keys;
    std::vector<double>  sums;
    std::vector<int64_t> counts;

    std::size_t num_rows() const { return sums.size(); }

    // Sorts rows by key (grouped results only), so hash-table emission order does not matter.
    void canonicalize();

    // Exact: doubles compare by bit pattern, so 0.1 + 0.2 and 0.3 differ.
    bool operator==(const QueryResult& other) const;

    // FNV-1a over every row's bits, in current row order (canonicalize first).
    std::uint64_t checksum() const;

    // Human-readable description of the first difference, or "" if equal. For test messages.
    std::string diff(const QueryResult& other) const;
};

}  // namespace wb::plan
