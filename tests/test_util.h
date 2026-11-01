#pragma once

#include <cstdint>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

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

// probe(k int64, amount float64) and build(k int64, grp int64), several segments each.
struct JoinTables {
    wb::Table probe{"probe", {{"k", wb::DataType::Int64}, {"amount", wb::DataType::Float64}}};
    wb::Table build{"build", {{"k", wb::DataType::Int64}, {"grp", wb::DataType::Int64}}};
};

inline std::unique_ptr<JoinTables> make_join_tables(int64_t probe_rows, int64_t (*probe_key)(int64_t), int64_t build_rows,
                                             int64_t (*build_key)(int64_t)) {
    auto t = std::make_unique<JoinTables>();
    for (int64_t start = 0; start < probe_rows; start += 700) {
        wb::SegmentBuilder b{t->probe.schema()};
        for (int64_t i = start; i < std::min(probe_rows, start + 700); ++i) {
            b.append_int64(0, probe_key(i));
            b.append_float64(1, 0.25 * static_cast<double>(i));
        }
        t->probe.add_segment(b.seal(std::nullopt, 1024));
    }
    for (int64_t start = 0; start < build_rows; start += 250) {
        wb::SegmentBuilder b{t->build.schema()};
        for (int64_t i = start; i < std::min(build_rows, start + 250); ++i) {
            b.append_int64(0, build_key(i));
            b.append_int64(1, i % 7);
        }
        t->build.add_segment(b.seal(std::nullopt, 1024));
    }
    return t;
}


// (k int64, v float64) rows, one segment per `rows_per_segment` rows.
inline std::unique_ptr<wb::Table> make_kv_double(const std::vector<std::pair<int64_t, double>>& rows,
                                                 std::size_t rows_per_segment = 1000) {
    auto t = std::make_unique<wb::Table>("kv", wb::Schema{{"k", wb::DataType::Int64}, {"v", wb::DataType::Float64}});
    for (std::size_t start = 0; start < rows.size(); start += rows_per_segment) {
        wb::SegmentBuilder b{t->schema()};
        for (std::size_t i = start; i < std::min(rows.size(), start + rows_per_segment); ++i) {
            b.append_int64(0, rows[i].first);
            b.append_float64(1, rows[i].second);
        }
        t->add_segment(b.seal(std::nullopt, 1024));
    }
    return t;
}

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Groups 1-3 put NaN first, in the middle, everywhere; group 4 holds only -0.0.
inline std::unique_ptr<wb::Table> make_nan_table() {
    return make_kv_double({{1, kNaN}, {1, 1.0}, {1, 2.0}, {2, 1.0}, {2, kNaN}, {2, 2.0},
                           {3, kNaN}, {3, kNaN}, {4, -0.0}, {4, -0.0}});
}

// One group whose running int64 sum leaves int64 but whose final sum is 5.
inline KeyValueRows int128_intermediate_rows() {
    return {{1, INT64_MAX}, {1, INT64_MAX}, {1, -INT64_MAX}, {1, -INT64_MAX + 5}};
}

// Group 42 whose sum overflows int64.
inline KeyValueRows overflow_rows() { return {{42, INT64_MAX}, {42, 1}}; }

// Groups 1-4 with 1, 2, 7 and 10 values (odd, even, single), deterministic pseudo-random values.
inline std::vector<std::pair<int64_t, double>> median_double_rows() {
    std::vector<std::pair<int64_t, double>> rows;
    uint64_t state = 99;
    const int sizes[] = {1, 2, 7, 10};
    for (int g = 0; g < 4; ++g) {
        for (int i = 0; i < sizes[g]; ++i) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            rows.emplace_back(g + 1, static_cast<double>(state >> 40) / 1024.0 - 4096.0);
        }
    }
    return rows;
}

inline KeyValueRows median_int_rows() {
    KeyValueRows rows;
    for (const auto& [k, v] : median_double_rows())
        rows.emplace_back(k, static_cast<int64_t>(v * 1000));
    return rows;
}

// Group 1: even count with NaN in the middle pair -> NaN. Group 2: odd count, NaN sorts last -> 2.0.
inline std::vector<std::pair<int64_t, double>> nan_median_rows() {
    return {{1, 1.0}, {1, kNaN}, {1, 3.0}, {1, kNaN}, {2, 1.0}, {2, 2.0}, {2, kNaN}};
}

// Signed zeros (and NaNs with two payloads) are equal under `<`, so a median whose middle values
// include them is only deterministic under a total order: -0.0 before +0.0, NaNs last by bits.
inline std::vector<std::pair<int64_t, double>> signed_zero_median_rows() {
    const double nan_a = std::bit_cast<double>(uint64_t{0x7FF8000000000001ULL});
    const double nan_b = std::bit_cast<double>(uint64_t{0x7FF8000000000002ULL});
    return {{1, +0.0}, {1, 1.0}, {1, -0.0}, {1, +0.0}, {1, 1.0},                 // odd: middle is +0.0
            {2, -0.0}, {2, +0.0}, {2, -0.0}, {2, +0.0},                          // even: (-0 + +0) / 2
            {3, -0.0}, {3, -0.0}, {3, +0.0}, {3, 5.0},                           // even: (-0 + +0) / 2
            {4, nan_b}, {4, 1.0}, {4, nan_a}, {4, nan_b}, {4, nan_a}, {4, 2.0}};  // even: (nan_a + nan_a) / 2
}

// MEDIAN groups of 1, 255, 256, 257 and 512 values: chains ending at, just after and well past a
// 256-value block boundary.
inline std::vector<std::pair<int64_t, double>> median_boundary_rows() {
    std::vector<std::pair<int64_t, double>> rows;
    uint64_t state = 17;
    for (int64_t g : {1, 255, 256, 257, 512})
        for (int64_t i = 0; i < g; ++i) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            rows.emplace_back(g, static_cast<double>(state >> 40) / 512.0);
        }
    return rows;
}

// Keys alternate between INT64_MIN (held outside the hash table's slot array) and 0.
inline std::vector<std::pair<int64_t, double>> int64_min_median_rows() {
    std::vector<std::pair<int64_t, double>> rows;
    for (int64_t i = 0; i < 1'000; ++i)
        rows.emplace_back(i % 2 ? INT64_MIN : 0, 0.25 * static_cast<double>(i % 37) - 3.0);
    return rows;
}

inline KeyValueRows int64_extreme_rows() { return {{1, INT64_MIN}, {1, INT64_MAX}}; }

}  // namespace wbtest
