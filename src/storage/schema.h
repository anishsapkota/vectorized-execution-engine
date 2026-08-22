#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wb {

enum class DataType : std::uint8_t { Int64, Float64, String };

struct ColumnDef {
    std::string name;
    DataType    type;
};

using Schema = std::vector<ColumnDef>;

const char* type_name(DataType t);

}  // namespace wb