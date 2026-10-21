#pragma once

#include <cstdint>
#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "dist/chunk.h"
#include "exec/operator.h"
#include "exec/scan.h"
#include "storage/segment.h"
#include "storage/table.h"

namespace wbtest {

using KeyValueRows = std::vector<std::pair<int64_t, int64_t>>;

inline wb::Schema key_value_schema() {
    return {{"key", wb::DataType::Int64}, {"value", wb::DataType::Int64}};
}

inline void append_segments(wb::Table& table, const KeyValueRows& rows,
                            std::size_t rows_per_segment) {
    for (std::size_t start = 0; start < rows.size(); start += rows_per_segment) {
        wb::SegmentBuilder builder{table.schema()};
        const std::size_t  end = std::min(rows.size(), start + rows_per_segment);
        for (std::size_t i = start; i < end; ++i) {
            builder.append_int64(0, rows[i].first);
            builder.append_int64(1, rows[i].second);
        }
        table.add_segment(builder.seal(std::nullopt, 1024));
    }
}

inline std::unique_ptr<wb::Table> make_table(const std::string& name, const KeyValueRows& rows,
                                             std::size_t rows_per_segment = 1000) {
    auto table = std::make_unique<wb::Table>(name, key_value_schema());
    append_segments(*table, rows, rows_per_segment);
    return table;
}

// Every live (key, value) row an operator produces, in emission order.
inline KeyValueRows collect(wb::Operator& op) {
    KeyValueRows rows;
    while (auto batch = op.next()) {
        const auto& keys   = static_cast<const wb::Int64Vector&>(batch->column(0));
        const auto& values = static_cast<const wb::Int64Vector&>(batch->column(1));
        wb::for_each_live_row(*batch, [&](std::uint32_t row) {
            rows.emplace_back(keys.value(row), values.value(row));
        });
    }
    return rows;
}

inline KeyValueRows collect(const wb::Table& table) {
    wb::ScanOperator scan{table, std::nullopt};
    return collect(scan);
}

// readings-shaped table: device_id 1..300, ts a random walk (sorted per segment), value ~N(50,15).
inline std::unique_ptr<wb::Table> make_readings(std::size_t rows, std::size_t rows_per_segment = 8'000,
                                     bool narrow_float = false) {
    wb::Schema schema{{"device_id", wb::DataType::Int64},
                  {"ts", wb::DataType::Int64},
                  {"value", wb::DataType::Float64, narrow_float}};
    auto   t = std::make_unique<wb::Table>("readings", schema, 1);
    std::mt19937_64                        rng{7};
    std::uniform_int_distribution<int64_t> device{1, 300};
    std::uniform_int_distribution<int64_t> jitter{-30, 30};
    std::normal_distribution<double>       value{50.0, 15.0};
    int64_t                                ts = 1'700'000'000;
    for (std::size_t start = 0; start < rows; start += rows_per_segment) {
        wb::SegmentBuilder b{schema};
        for (std::size_t r = start; r < std::min(rows, start + rows_per_segment); ++r) {
            ts += 1 + jitter(rng);
            b.append_int64(0, device(rng));
            b.append_int64(1, ts);
            b.append_float64(2, value(rng));
        }
        t->add_segment(b.seal(1, 1024));
    }
    return t;
}

}  // namespace wbtest
