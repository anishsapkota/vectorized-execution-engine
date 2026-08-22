#pragma once

#include <cstddef>
#include <string>

#include "storage/table.h"

namespace wb {
// Load a csv file into the table. The file must start with a header row whose columns match
// table.schema() in order. Each subsequent row becomes one row in the table; the loader builds
// segments of up to rows_per_segment rows.

void load_csv(Table& table, const std::string& path, std::size_t rows_per_segment = 65536);

}