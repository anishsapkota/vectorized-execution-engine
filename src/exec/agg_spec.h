#pragma once

#include <cstddef>
#include <cstdint>

#include "storage/schema.h"

namespace wb {

enum class AggKind : std::uint8_t { Count, Sum, Min, Max, Avg, Median };

// One aggregate computed by HashAggOperator. `col` and `type` describe the input column and are
// ignored for Count (COUNT(*)).
struct AggSpec {
    AggKind     kind;
    std::size_t col  = 0;
    DataType    type = DataType::Int64;
};

}  // namespace wb
