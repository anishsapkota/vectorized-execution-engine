
#include "vector.h"

namespace wb {

const char* type_name(DataType t) {
    switch (t) {
        case DataType::Int64:
            return "int64";
        case DataType::Float64:
            return "float64";
        case DataType::String:
            return "string";
    }
    return "unknown";  // unreachable, but pacifies -Wreturn-type
}

}  // namespace wb