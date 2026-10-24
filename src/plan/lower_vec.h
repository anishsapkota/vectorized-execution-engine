#pragma once

#include "plan/plan.h"
#include "plan/result.h"

namespace wb::plan {

// Runs a validated Aggregate plan on the vectorized engine:
//   Filter over Scan -> ScanOperator pruning on the first conjunct (translated to the table's
//                       column), then one FilterOperator per conjunct;
//   HashJoin         -> HashJoinOperator;
//   Aggregate        -> HashAggOperator, grouped or global. A global aggregate over no rows
//                       yields one row: COUNT 0, every other aggregate NULL.
// Returns a canonicalized result. Throws std::invalid_argument if the plan is invalid and
// std::overflow_error if a SUM over int64 leaves the int64 range.
QueryResult run_vectorized(const Node& root);

}  // namespace wb::plan
