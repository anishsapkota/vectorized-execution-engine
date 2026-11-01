#include "plan/result.h"

#include <algorithm>
#include <bit>
#include <numeric>
#include <sstream>

namespace wb::plan {

namespace {

void fnv_mix(std::uint64_t& h, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xFF;
        h *= 0x100000001B3ULL;
    }
}

// Bit pattern of a cell; 0 with a NULL marker for NULLs so stored bytes do not matter.
std::uint64_t cell_bits(const ResultColumn& c, std::size_t row) {
    if (!c.valid[row]) return 0;
    return c.type == DataType::Int64 ? static_cast<std::uint64_t>(c.ints[row])
                                     : std::bit_cast<std::uint64_t>(c.doubles[row]);
}

std::string cell_text(const ResultColumn& c, std::size_t row) {
    if (!c.valid[row]) return "NULL";
    std::ostringstream out;
    out.precision(17);
    if (c.type == DataType::Int64) out << c.ints[row];
    else out << c.doubles[row];
    return out.str();
}

}  // namespace

void ResultColumn::push_int(int64_t v, bool is_valid) {
    ints.push_back(v);
    valid.push_back(is_valid);
}

void ResultColumn::push_double(double v, bool is_valid) {
    doubles.push_back(v);
    valid.push_back(is_valid);
}

void QueryResult::append_empty_aggregate_row(const std::vector<bool>& is_count) {
    for (std::size_t i = 0; i < columns.size(); ++i) {
        auto& col = columns[i];
        if (is_count[i]) col.push_int(0);
        else if (col.type == DataType::Int64) col.push_int(0, false);
        else col.push_double(0.0, false);
    }
}

void QueryResult::canonicalize() {
    if (!grouped || columns.empty()) return;
    const auto&              keys = columns[0].ints;
    std::vector<std::size_t> order(keys.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });

    auto permute = [&](auto& values) {
        if (values.empty()) return;
        auto copy = values;
        for (std::size_t i = 0; i < order.size(); ++i)
            values[i] = copy[order[i]];
    };
    for (auto& c : columns) {
        permute(c.ints);
        permute(c.doubles);
        permute(c.valid);
    }
}

bool QueryResult::operator==(const QueryResult& other) const { return diff(other).empty(); }

std::uint64_t QueryResult::checksum() const {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    fnv_mix(h, grouped);
    for (std::size_t row = 0; row < num_rows(); ++row) {
        for (const auto& c : columns) {
            fnv_mix(h, c.valid[row]);
            fnv_mix(h, cell_bits(c, row));
        }
    }
    return h;
}

std::string QueryResult::diff(const QueryResult& other) const {
    std::ostringstream out;
    if (grouped != other.grouped) {
        out << "grouped " << grouped << " vs " << other.grouped;
        return out.str();
    }
    if (columns.size() != other.columns.size()) {
        out << "column count " << columns.size() << " vs " << other.columns.size();
        return out.str();
    }
    for (std::size_t c = 0; c < columns.size(); ++c) {
        if (columns[c].type != other.columns[c].type) {
            out << "col " << c << ": type " << type_name(columns[c].type) << " vs " << type_name(other.columns[c].type);
            return out.str();
        }
    }
    if (num_rows() != other.num_rows()) {
        out << "row count " << num_rows() << " vs " << other.num_rows();
        return out.str();
    }
    for (std::size_t row = 0; row < num_rows(); ++row) {
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const auto& a = columns[c];
            const auto& b = other.columns[c];
            if (a.valid[row] != b.valid[row] || cell_bits(a, row) != cell_bits(b, row)) {
                out << "col " << c << " row " << row << ": " << cell_text(a, row) << " vs " << cell_text(b, row);
                return out.str();
            }
        }
    }
    return "";
}

}  // namespace wb::plan
