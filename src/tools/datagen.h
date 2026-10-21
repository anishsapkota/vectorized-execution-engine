#pragma once

// Synthetic join tables shared by wb_dist and wb_codegen, so both experiments run on the same data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>

#include "storage/schema.h"
#include "storage/segment.h"
#include "storage/table.h"

namespace wb {

// orders(customer_id INT64, amount FLOAT64). With exponent 1 customer ids are uniform; larger
// exponents push orders toward low ids, the same power trick as tools/gen.cpp.
inline std::unique_ptr<Table> make_orders(std::int64_t rows, std::int64_t customers, double exponent) {
    auto table = std::make_unique<Table>(
        "orders", Schema{{"customer_id", DataType::Int64}, {"amount", DataType::Float64}});
    std::mt19937_64                        rng{42};
    std::uniform_real_distribution<double> unit{0.0, 1.0};
    for (std::int64_t start = 0; start < rows; start += 65536) {
        SegmentBuilder builder{table->schema()};
        for (std::int64_t i = start; i < std::min(rows, start + 65536); ++i) {
            const auto id = static_cast<std::int64_t>(std::pow(unit(rng), exponent) * customers);
            builder.append_int64(0, id);
            builder.append_float64(1, unit(rng) * 100.0);
        }
        table->add_segment(builder.seal(std::nullopt, 8192));
    }
    return table;
}

// customers(customer_id INT64, region INT64), ids 0..rows-1.
inline std::unique_ptr<Table> make_customers(std::int64_t rows) {
    auto table = std::make_unique<Table>(
        "customers", Schema{{"customer_id", DataType::Int64}, {"region", DataType::Int64}});
    for (std::int64_t start = 0; start < rows; start += 65536) {
        SegmentBuilder builder{table->schema()};
        for (std::int64_t id = start; id < std::min(rows, start + 65536); ++id) {
            builder.append_int64(0, id);
            builder.append_int64(1, id % 50);
        }
        table->add_segment(builder.seal(std::nullopt, 8192));
    }
    return table;
}

}  // namespace wb
