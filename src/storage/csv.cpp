
#include "storage/csv.h"

#include <charconv>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "storage/segment.h"

namespace wb {

// Split a single CSV line on commas into out (cleared first). No quote handling. The views
// point into line and are valid only while line is unchanged.
void split_csv_line(std::string_view line, std::vector<std::string_view>& out) {
    out.clear();
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            out.push_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
}

int64_t parse_int64(std::string_view field) {
    int64_t value = 0;
    auto [end, ec] = std::from_chars(field.data(), field.data() + field.size(), value);
    if (ec != std::errc{} || end != field.data() + field.size()) {
        throw std::runtime_error{"bad int64 field '" + std::string{field} + "'"};
    }
    return value;
}

// strtod rather than std::from_chars: not every libc++ we build against has the floating-point
// overload. strtod stops at the ',' (or the line's terminating NUL) that ends the field.
double parse_float64(std::string_view field) {
    char*  end   = nullptr;
    double value = field.empty() ? 0.0 : std::strtod(field.data(), &end);
    if (field.empty() || end != field.data() + field.size()) {
        throw std::runtime_error{"bad float64 field '" + std::string{field} + "'"};
    }
    return value;
}

// Append one parsed value to the builder, dispatching on the column type.
void append_field(SegmentBuilder& builder, std::size_t col_idx, DataType type,
                  std::string_view field) {
    switch (type) {
        case DataType::Int64:
            builder.append_int64(col_idx, parse_int64(field));
            break;
        case DataType::Float64:
            builder.append_float64(col_idx, parse_float64(field));
            break;
        case DataType::String:
            builder.append_string(col_idx, std::string{field});
            break;
    }
}

void load_csv(Table& table, const std::string& path, std::size_t rows_per_segment) {
    std::ifstream in{path};
    if (!in) throw std::runtime_error{"cannot open " + path};

    const Schema& schema = table.schema();

    // Read header line. Validate it matches the schema.
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error{"empty CSV: " + path};
    }
    {
        std::vector<std::string_view> header;
        split_csv_line(line, header);
        if (header.size() != schema.size()) {
            throw std::runtime_error{"CSV header column count mismatch"};
        }

        for (std::size_t i = 0; i < schema.size(); ++i) {
            if (header[i] != schema[i].name) {
                throw std::runtime_error{"CSV header column " + std::to_string(i) + ": expected '" +
                                         schema[i].name + "', got '" + std::string{header[i]} + "'"};
            }
        }
    }

    // Read data rows. Every rows_per_segment rows, seal a segment.
    auto        builder         = std::make_unique<SegmentBuilder>(schema);
    std::size_t rows_in_builder = 0;

    // Reused for every row: after the first row it has enough capacity, so splitting a row
    // allocates nothing.
    std::vector<std::string_view> fields;
    while (std::getline(in, line)) {
        if (line.empty()) continue;  // skip blank lines

        split_csv_line(line, fields);

        if (fields.size() != schema.size()) {
            throw std::runtime_error{"row column count mismatch"};
        }

        for (std::size_t i = 0; i < schema.size(); ++i) {
            append_field(*builder, i, schema[i].type, fields[i]);
        }
        ++rows_in_builder;

        if (rows_in_builder >= rows_per_segment) {
            table.add_segment(builder->seal(table.sort_key_col(), /*granule_size=*/8192));
            builder         = std::make_unique<SegmentBuilder>(schema);
            rows_in_builder = 0;
        }
    }

    // Seal any final partial segment
    if (rows_in_builder > 0) {
        table.add_segment(builder->seal(table.sort_key_col(), 8192));
    }
};

}  // namespace wb