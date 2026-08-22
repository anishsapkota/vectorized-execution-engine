#pragma once
#include <cstddef>
#include <memory>

#include "exec/hash_table.h"
#include "storage/table.h"

namespace wb {

AggHashTable parallel_group_sum_count(const Table& table, std::size_t group_col_idx,
                                      std::size_t agg_col_idx, std::size_t num_workers);

}  // namespace wb