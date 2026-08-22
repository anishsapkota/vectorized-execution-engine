
#include "storage/csv.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "storage/segment.h"

namespace wb {

// Split a single CSV line on commas. No quote handling. Returns empty vector if the input is empty
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::size_t              start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            out.emplace_back(line, start, i - start);
            start = i + 1;
        }
    }
    return out;
}

// Append one parsed value to the builder, dispatching on the column type.
void append_field(SegmentBuilder& builder, std::size_t col_idx, DataType type,
                  const std::string& field) {
    switch (type) {
        case DataType::Int64:
            builder.append_int64(col_idx, std::stoll(field));
            break;
        case DataType::Float64:
            builder.append_float64(col_idx, std::stod(field));
            break;
        case DataType::String:
            builder.append_string(col_idx, field);
            break;
        default:
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
        std::vector<std::string> header = split_csv_line(line);
        if (header.size() != schema.size()) {
            throw std::runtime_error{"CSV header column count mismatch"};
        }

        for (std::size_t i = 0; i < schema.size(); ++i) {
            if (header[i] != schema[i].name) {
                throw std::runtime_error{"CSV header column " + std::to_string(i) + ": expected '" +
                                         schema[i].name + "', got '" + header[i] + "'"};
            }
        }
    }

    // Read data rows. Every rows_per_segment rows, seal a segment.
    auto        builder         = std::make_unique<SegmentBuilder>(schema);
    std::size_t rows_in_builder = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;  // skip blank lines

        std::vector<std::string> fields = split_csv_line(line);

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