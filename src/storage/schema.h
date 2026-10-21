#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wb {

enum class DataType : std::uint8_t { Int64, Float64, String };

struct ColumnDef {
    std::string name;
    DataType    type;
    // Float64 only: store values as float. Lossy in general; exact for inputs with at most
    // 6 significant decimal digits.
    bool narrow_float = false;
};

using Schema = std::vector<ColumnDef>;

const char* type_name(DataType t);

}  // namespace wb