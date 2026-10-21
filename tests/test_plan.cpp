// Logical plan IR, its type checker, and lowering to the vectorized engine.

#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "plan/lower_vec.h"
#include "plan/plan.h"
#include "plan/result.h"
#include "storage/segment.h"
#include "storage/table.h"
#include "test_framework.h"
#include "test_util.h"

using namespace wb;
using namespace wb::plan;
using wbtest::make_readings;

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

// --- A row-at-a-time reference interpreter. It reads storage in segment/row order, which is the
// order both engines accumulate in, so its sums are bit-identical to theirs. ---

struct Value {
    bool    is_int;
    int64_t i;
    double  d;
};
using Row = std::vector<Value>;

std::vector<Row> reference_rows(const Node& node);

bool compare(const Value& v, CompareOp op, const std::variant<int64_t, double>& lit) {
    auto cmp = [&](auto a, auto b) {
        switch (op) {
            case CompareOp::Eq: return a == b;
            case CompareOp::Ne: return a != b;
            case CompareOp::Lt: return a < b;
            case CompareOp::Le: return a <= b;
            case CompareOp::Gt: return a > b;
            case CompareOp::Ge: return a >= b;
        }
        return false;
    };
    return v.is_int ? cmp(v.i, std::get<int64_t>(lit)) : cmp(v.d, std::get<double>(lit));
}

std::vector<Row> reference_rows(const Node& node) {
    std::vector<Row> out;
    if (const auto* s = std::get_if<Scan>(&node.op)) {
        for (const auto& seg : s->table->segments()) {
            for (std::size_t r = 0; r < seg->num_rows(); ++r) {
                Row row;
                for (std::size_t c : s->cols) {
                    const Vector& col = seg->column(c);
                    if (col.type() == DataType::Int64) {
                        row.push_back({true, static_cast<const Int64Vector&>(col).value(r), 0});
                    } else {
                        row.push_back({false, 0, static_cast<const Float64Vector&>(col).value(r)});
                    }
                }
                out.push_back(std::move(row));
            }
        }
    } else if (const auto* f = std::get_if<Filter>(&node.op)) {
        for (auto& row : reference_rows(*f->child)) {
            bool keep = true;
            for (const auto& c : f->conjuncts)
                keep = keep && compare(row[c.col], c.op, c.literal);
            if (keep) out.push_back(std::move(row));
        }
    } else if (const auto* j = std::get_if<HashJoin>(&node.op)) {
        const auto build = reference_rows(*j->build);
        for (const auto& p : reference_rows(*j->probe)) {
            for (const auto& b : build) {
                if (p[j->probe_key].i != b[j->build_key].i) continue;
                Row row = p;
                row.insert(row.end(), b.begin(), b.end());
                out.push_back(std::move(row));
            }
        }
    }
    return out;
}

QueryResult reference(const Node& root) {
    const auto& agg = std::get<SumCount>(root.op);
    QueryResult r;
    r.grouped = agg.group_col.has_value();
    std::map<int64_t, std::size_t> slot;  // key -> row in r
    if (!r.grouped) {
        r.sums   = {0.0};
        r.counts = {0};
    }
    for (const auto& row : reference_rows(*agg.child)) {
        std::size_t at = 0;
        if (r.grouped) {
            auto [it, fresh] = slot.try_emplace(row[*agg.group_col].i, r.keys.size());
            if (fresh) {
                r.keys.push_back(it->first);
                r.sums.push_back(0.0);
                r.counts.push_back(0);
            }
            at = it->second;
        }
        r.sums[at] += row[agg.sum_col].d;
        ++r.counts[at];
    }
    r.canonicalize();
    return r;
}

void check_matches_reference(const Node& root) {
    const auto got  = run_vectorized(root);
    const auto want = reference(root);
    if (!(got == want)) throw wbtest::TestFailure(to_string(root) + ": " + got.diff(want));
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

TEST(vec_global_filter_sum) {
    auto          t   = make_readings(50'000);
    const int64_t mid = t->segment(3).zone(1).min_i64();
    check_matches_reference(*sum_count(filter(scan(*t, {1, 2}), {{0, CompareOp::Gt, mid}}), std::nullopt, 1));
}

TEST(vec_grouped_sum) {
    auto t = make_readings(50'000);
    check_matches_reference(*sum_count(scan(*t, {0, 2}), 0, 1));
}

TEST(vec_two_conjuncts) {
    auto t = make_readings(50'000);
    check_matches_reference(*sum_count(
        filter(scan(*t, {0, 2}), {{1, CompareOp::Gt, 50.0}, {0, CompareOp::Lt, int64_t{150}}}), std::nullopt, 1));
}

TEST(vec_join_grouped) {
    Schema probe_schema{{"k", DataType::Int64}, {"amount", DataType::Float64}};
    Schema build_schema{{"k", DataType::Int64}, {"grp", DataType::Int64}};
    Table  probe{"probe", probe_schema}, build{"build", build_schema};
    for (int64_t start = 0; start < 3'000; start += 700) {
        SegmentBuilder b{probe_schema};
        for (int64_t i = start; i < std::min<int64_t>(3'000, start + 700); ++i) {
            b.append_int64(0, i % 400);
            b.append_float64(1, 0.25 * static_cast<double>(i));
        }
        probe.add_segment(b.seal(std::nullopt, 1024));
    }
    for (int64_t start = 0; start < 900; start += 250) {
        SegmentBuilder b{build_schema};
        for (int64_t i = start; i < std::min<int64_t>(900, start + 250); ++i) {
            b.append_int64(0, 200 + i % 300);
            b.append_int64(1, i % 7);
        }
        build.add_segment(b.seal(std::nullopt, 1024));
    }
    check_matches_reference(*sum_count(join(scan(probe, {0, 1}), scan(build, {0, 1}), 0, 0), 3, 1));
}

TEST(prune_uses_table_column_through_projection) {
    // Plan column 0 is table column 2 (value, Float64). Pruning on table column 0 (device_id,
    // Int64) with a double literal would be wrong, and would throw.
    auto t = make_readings(50'000);
    check_matches_reference(*sum_count(filter(scan(*t, {2, 1}), {{0, CompareOp::Gt, 80.0}}), std::nullopt, 0));
}

TEST(empty_table_vec) {
    auto t      = make_readings(0);
    auto global = run_vectorized(*sum_count(scan(*t, {2}), std::nullopt, 0));
    CHECK_EQ(global.num_rows(), std::size_t{1});
    CHECK_EQ(global.counts[0], int64_t{0});
    CHECK_EQ(global.sums[0], 0.0);
    CHECK_EQ(run_vectorized(*sum_count(scan(*t, {0, 2}), 0, 1)).num_rows(), std::size_t{0});
}
