// Logical plan IR, its type checker, and lowering to the vectorized engine.

#include <stdexcept>
#include <string>
#include <vector>

#include "plan/plan.h"
#include "plan/result.h"
#include "storage/table.h"
#include "test_framework.h"

using namespace wb;
using namespace wb::plan;

namespace {

QueryResult grouped(std::vector<int64_t> keys, std::vector<double> sums,
                    std::vector<int64_t> counts) {
    QueryResult r;
    r.grouped = true;
    r.keys    = std::move(keys);
    r.sums    = std::move(sums);
    r.counts  = std::move(counts);
    return r;
}

Table readings_schema_table() {
    return Table{"readings",
                 {{"device_id", DataType::Int64}, {"ts", DataType::Int64}, {"value", DataType::Float64}},
                 1};
}

template <typename F>
bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

TEST(result_equality_is_bitwise) {
    // 0.1 + 0.2 != 0.3 in binary floating point; an epsilon comparison would hide that.
    auto a = grouped({1}, {0.1 + 0.2}, {1});
    auto b = grouped({1}, {0.3}, {1});
    CHECK(!(a == b));
    CHECK(a.diff(b).find("row 0") != std::string::npos);
    CHECK(a == grouped({1}, {0.1 + 0.2}, {1}));
}

TEST(result_canonicalize_sorts_by_key) {
    auto r = grouped({3, 1, 2}, {30.0, 10.0, 20.0}, {3, 1, 2});
    r.canonicalize();
    CHECK(r.keys == (std::vector<int64_t>{1, 2, 3}));
    CHECK(r.sums == (std::vector<double>{10.0, 20.0, 30.0}));
    CHECK(r.counts == (std::vector<int64_t>{1, 2, 3}));
}

TEST(result_checksum_order_independent) {
    auto a = grouped({3, 1, 2}, {30.0, 10.0, 20.0}, {3, 1, 2});
    auto b = grouped({1, 2, 3}, {10.0, 20.0, 30.0}, {1, 2, 3});
    a.canonicalize();
    b.canonicalize();
    CHECK_EQ(a.checksum(), b.checksum());
    CHECK(a.checksum() != grouped({1, 2, 3}, {10.0, 20.0, 31.0}, {1, 2, 3}).checksum());
}

TEST(plan_output_types) {
    const Table t = readings_schema_table();
    using V       = std::vector<DataType>;
    CHECK(output_types(*scan(t, {1, 2})) == (V{DataType::Int64, DataType::Float64}));
    CHECK(output_types(*join(scan(t, {0, 1}), scan(t, {0, 2}), 0, 0)) ==
          (V{DataType::Int64, DataType::Int64, DataType::Int64, DataType::Float64}));
    CHECK(output_types(*sum_count(scan(t, {0, 2}), 0, 1)) ==
          (V{DataType::Int64, DataType::Float64, DataType::Int64}));
    CHECK(output_types(*sum_count(scan(t, {2}), std::nullopt, 0)) ==
          (V{DataType::Float64, DataType::Int64}));
}

TEST(plan_rejects_invalid) {
    const Table t = readings_schema_table();
    const Table s{"s", {{"name", DataType::String}, {"id", DataType::Int64}}};

    CHECK(throws_invalid([&] { output_types(*scan(s, {0})); }));                     // String column
    CHECK(throws_invalid([&] { output_types(*filter(scan(t, {0}), {{0, CompareOp::Gt, 1.0}})); }));
    CHECK(throws_invalid([&] { output_types(*sum_count(scan(t, {2, 2}), 0, 1)); }));  // Float64 group
    CHECK(throws_invalid([&] { output_types(*sum_count(scan(t, {0, 1}), 0, 1)); }));  // Int64 sum
    CHECK(throws_invalid([&] { output_types(*scan(t, {9})); }));                      // bad table col
    CHECK(throws_invalid([&] { output_types(*filter(scan(t, {0}), {{9, CompareOp::Gt, int64_t{1}}})); }));
    CHECK(throws_invalid([&] { output_types(*filter(scan(t, {0}), {})); }));          // empty AND
    CHECK(throws_invalid([&] { validate_query(*filter(scan(t, {0}), {{0, CompareOp::Gt, int64_t{1}}})); }));
    CHECK(throws_invalid([&] {
        validate_query(*sum_count(filter(sum_count(scan(t, {0, 2}), 0, 1), {{0, CompareOp::Gt, int64_t{1}}}), 0, 1));
    }));
    validate_query(*sum_count(scan(t, {0, 2}), 0, 1));  // valid: must not throw
}

TEST(plan_to_string) {
    const Table t = readings_schema_table();
    CHECK_EQ(to_string(*sum_count(filter(scan(t, {0, 2}), {{0, CompareOp::Gt, int64_t{5000}}}), 0, 1)),
             std::string{"SumCount(group=0,sum=1, Filter([0>5000], Scan(readings,[0,2])))"});
}
