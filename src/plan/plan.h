#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "exec/predicate.h"
#include "storage/schema.h"
#include "storage/table.h"

namespace wb::plan {

// A logical query plan, shared by the vectorized engine (lower_vec.h) and the compiled engine
// (codegen/). Column references are positions in the child's output, as with the operators.

struct Compare {
    std::size_t                    col;
    CompareOp                      op;
    std::variant<int64_t, double> literal;
};

struct Node;
using NodePtr = std::unique_ptr<Node>;

struct Scan {
    const Table*             table;
    std::vector<std::size_t> cols;  // table column indices, in output order
};

struct Filter {
    NodePtr              child;
    std::vector<Compare> conjuncts;  // AND; non-empty
};

struct HashJoin {
    NodePtr     probe;
    NodePtr     build;
    std::size_t probe_key;
    std::size_t build_key;
};  // output: probe columns, then build columns

struct SumCount {
    NodePtr                    child;
    std::optional<std::size_t> group_col;
    std::size_t                sum_col;
};  // output: [key int64,] sum float64, count int64

struct Node {
    std::variant<Scan, Filter, HashJoin, SumCount> op;
};

NodePtr scan(const Table& table, std::vector<std::size_t> cols);
NodePtr filter(NodePtr child, std::vector<Compare> conjuncts);
NodePtr join(NodePtr probe, NodePtr build, std::size_t probe_key, std::size_t build_key);
NodePtr sum_count(NodePtr child, std::optional<std::size_t> group_col, std::size_t sum_col);

// Output column types of `node`. Validates the subtree: only Int64/Float64 columns, column
// references in range, literal types matching their columns, Int64 join/group keys, a Float64
// sum column, non-empty conjunctions. Throws std::invalid_argument naming the node and column.
std::vector<DataType> output_types(const Node& node);

// output_types(root), plus: the root is a SumCount and no SumCount appears below it.
void validate_query(const Node& root);

// One-line form, e.g. "SumCount(group=0,sum=1, Filter([0>5000], Scan(readings,[0,2])))".
std::string to_string(const Node& node);

}  // namespace wb::plan
