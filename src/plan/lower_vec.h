#pragma once

#include "plan/plan.h"
#include "plan/result.h"

namespace wb::plan {

// Runs a validated SumCount plan on the vectorized engine:
//   Filter over Scan -> ScanOperator pruning on the first conjunct (translated to the table's
//                       column), then one FilterOperator per conjunct;
//   HashJoin         -> HashJoinOperator;
//   grouped SumCount -> HashAggOperator; global SumCount -> a drain loop.
// Returns a canonicalized result. Throws std::invalid_argument if the plan is invalid.
QueryResult run_vectorized(const Node& root);

}  // namespace wb::plan
