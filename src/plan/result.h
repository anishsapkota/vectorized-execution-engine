#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "storage/schema.h"

namespace wb::plan {

// One output column. Values live in `ints` or `doubles` depending on `type`; `valid[i] == 0`
// marks row i NULL (its stored value is meaningless).
struct ResultColumn {
    DataType             type = DataType::Int64;
    std::vector<int64_t> ints;
    std::vector<double>  doubles;
    std::vector<uint8_t> valid;

    void        push_int(int64_t v, bool is_valid = true);
    void        push_double(double v, bool is_valid = true);
    std::size_t size() const { return valid.size(); }
};

// The answer to an aggregate query, in the same form from both engines so they can be compared
// exactly. The group key is column 0 when grouped.
struct QueryResult {
    bool                      grouped = false;
    std::vector<ResultColumn> columns;

    std::size_t num_rows() const { return columns.empty() ? 0 : columns.front().size(); }

    // Sorts rows by key (grouped results only), so hash-table emission order does not matter.
    void canonicalize();

    // Exact: doubles compare by bit pattern (0.1 + 0.2 differs from 0.3); NULL equals NULL
    // whatever bytes it stores.
    bool operator==(const QueryResult& other) const;

    // FNV-1a over every cell in current row order (canonicalize first).
    std::uint64_t checksum() const;

    // First difference ("col C row R: ..."), or "" if equal. For test messages.
    std::string diff(const QueryResult& other) const;

    int64_t int_at(std::size_t col, std::size_t row) const { return columns[col].ints[row]; }
    double  double_at(std::size_t col, std::size_t row) const { return columns[col].doubles[row]; }
    bool    is_null(std::size_t col, std::size_t row) const { return columns[col].valid[row] == 0; }
};

}  // namespace wb::plan
