// Logical plan IR, its type checker, and lowering to the vectorized engine.

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "exec/group_index.h"
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

ResultColumn int_column(const std::vector<int64_t>& values) {
    ResultColumn c{DataType::Int64, {}, {}, {}};
    for (int64_t v : values)
        c.push_int(v);
    return c;
}

ResultColumn double_column(const std::vector<double>& values) {
    ResultColumn c{DataType::Float64, {}, {}, {}};
    for (double v : values)
        c.push_double(v);
    return c;
}

// [key, sum, count], the SumCount shape.
QueryResult grouped(std::vector<int64_t> keys, std::vector<double> sums, std::vector<int64_t> counts) {
    QueryResult r;
    r.grouped = true;
    r.columns = {int_column(keys), double_column(sums), int_column(counts)};
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

// Every aggregate computed the obvious way over each group's rows, in scan order. Must agree
// exactly with spec section 3; both engines are checked against this.
bool less_nan_last(double a, double b) { return !std::isnan(a) && (std::isnan(b) || a < b); }

void reference_aggregate(const AggExpr& e, DataType in, const std::vector<const Row*>& rows, bool grouped,
                         int64_t key, ResultColumn& out) {
    const std::size_t n = rows.size();
    if (e.fn == AggFn::Count) return out.push_int(static_cast<int64_t>(n));
    if (n == 0) {  // global aggregate over no rows
        if (in == DataType::Int64 && (e.fn == AggFn::Sum || e.fn == AggFn::Min || e.fn == AggFn::Max)) {
            return out.push_int(0, false);
        }
        return out.push_double(0.0, false);
    }
    const std::size_t c = *e.col;
    if (in == DataType::Int64) {
        std::vector<int64_t> v;
        for (const Row* r : rows)
            v.push_back((*r)[c].i);
        __int128 sum = 0;
        for (int64_t x : v)
            sum += x;
        switch (e.fn) {
            case AggFn::Sum:
                if (sum > INT64_MAX || sum < INT64_MIN) {
                    throw std::overflow_error("SUM overflow in group " + (grouped ? std::to_string(key) : "global"));
                }
                return out.push_int(static_cast<int64_t>(sum));
            case AggFn::Min:
                return out.push_int(*std::min_element(v.begin(), v.end()));
            case AggFn::Max:
                return out.push_int(*std::max_element(v.begin(), v.end()));
            case AggFn::Avg:
                return out.push_double(static_cast<double>(sum) / static_cast<double>(n));
            case AggFn::Median: {
                std::sort(v.begin(), v.end());
                return out.push_double(n % 2 ? static_cast<double>(v[n / 2])
                                             : (static_cast<double>(v[n / 2 - 1]) + static_cast<double>(v[n / 2])) / 2.0);
            }
            case AggFn::Count:
                break;
        }
    }
    std::vector<double> v;
    for (const Row* r : rows)
        v.push_back((*r)[c].d);
    double sum = 0.0;  // starts at +0.0; every row adds
    for (double x : v)
        sum += x;
    switch (e.fn) {
        case AggFn::Sum:
            return out.push_double(sum);
        case AggFn::Min: {
            double m = v[0];
            for (double x : v)
                if (x < m || x != x) m = x;
            return out.push_double(m);
        }
        case AggFn::Max: {
            double m = v[0];
            for (double x : v)
                if (x > m || x != x) m = x;
            return out.push_double(m);
        }
        case AggFn::Avg:
            return out.push_double(sum / static_cast<double>(n));
        case AggFn::Median:
            std::sort(v.begin(), v.end(), less_nan_last);
            return out.push_double(n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0);
        case AggFn::Count:
            break;
    }
}

QueryResult reference(const Node& root) {
    const auto& agg   = std::get<Aggregate>(root.op);
    const auto  in    = output_types(*agg.child);
    const auto  types = output_types(root);
    const auto  rows  = reference_rows(*agg.child);

    // Group rows in first-seen order; a global aggregate is one (possibly empty) group.
    std::map<int64_t, std::size_t>        slot;
    std::vector<int64_t>                  keys;
    std::vector<std::vector<const Row*>> groups;
    if (!agg.group_col) groups.emplace_back();
    for (const Row& row : rows) {
        std::size_t at = 0;
        if (agg.group_col) {
            auto [it, fresh] = slot.try_emplace(row[*agg.group_col].i, keys.size());
            if (fresh) {
                keys.push_back(it->first);
                groups.emplace_back();
            }
            at = it->second;
        }
        groups[at].push_back(&row);
    }

    QueryResult r;
    r.grouped = agg.group_col.has_value();
    for (DataType t : types)
        r.columns.push_back({t, {}, {}, {}});
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::size_t c = 0;
        if (r.grouped) r.columns[c++].push_int(keys[g]);
        for (const AggExpr& e : agg.aggs) {
            const DataType input = e.col ? in[*e.col] : DataType::Int64;
            reference_aggregate(e, input, groups[g], r.grouped, r.grouped ? keys[g] : 0, r.columns[c++]);
        }
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
    CHECK(r.columns[0].ints == (std::vector<int64_t>{1, 2, 3}));
    CHECK(r.columns[1].doubles == (std::vector<double>{10.0, 20.0, 30.0}));
    CHECK(r.columns[2].ints == (std::vector<int64_t>{1, 2, 3}));
}

TEST(result_nulls_compare_equal_regardless_of_bytes) {
    QueryResult a, b, c;
    a.columns = {double_column({0.0})};
    b.columns = {double_column({0.0})};
    c.columns = {double_column({1.0})};
    a.columns[0].doubles[0] = 1.0;
    a.columns[0].valid[0]   = 0;
    b.columns[0].doubles[0] = 2.0;
    b.columns[0].valid[0]   = 0;
    CHECK(a == b);
    CHECK(!(a == c));
    CHECK(a.is_null(0, 0));
    CHECK(!c.is_null(0, 0));
}

TEST(result_diff_names_column) {
    auto a = grouped({1}, {1.0}, {1});
    auto b = grouped({1}, {1.0}, {2});
    const auto d = a.diff(b);
    CHECK(d.find("col 2") != std::string::npos);
    CHECK(d.find("row 0") != std::string::npos);
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
             std::string{"Aggregate(group=0,[sum(1),count], Filter([0>5000], Scan(readings,[0,2])))"});
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
    CHECK_EQ(global.int_at(1, 0), int64_t{0});
    CHECK(global.is_null(0, 0));  // SQL: SUM over no rows is NULL
    CHECK_EQ(run_vectorized(*sum_count(scan(*t, {0, 2}), 0, 1)).num_rows(), std::size_t{0});
}

TEST(vec_int64_min_group_key) {
    // INT64_MIN is the hash tables' empty-slot marker; as a real key it must still be its own
    // group, and must not leak into whichever key next probes the same slot.
    Schema schema{{"k", DataType::Int64}, {"v", DataType::Float64}};
    Table  t{"t", schema};
    SegmentBuilder b{schema};
    for (int i = 0; i < 100; ++i) {
        b.append_int64(0, i % 3 == 0 ? INT64_MIN : i % 5);
        b.append_float64(1, static_cast<double>(i));
    }
    t.add_segment(b.seal(std::nullopt, 1024));
    check_matches_reference(*sum_count(scan(t, {0, 1}), 0, 1));
}

TEST(plan_aggregate_rejects_invalid) {
    const Table t = readings_schema_table();
    CHECK(throws_invalid([&] { output_types(*aggregate(scan(t, {0, 2}), 0, {{AggFn::Count, 0}})); }));
    CHECK(throws_invalid([&] { output_types(*aggregate(scan(t, {0, 2}), 0, {{AggFn::Min, std::nullopt}})); }));
    CHECK(throws_invalid([&] { output_types(*aggregate(scan(t, {0, 2}), 0, {{AggFn::Max, 9}})); }));
    CHECK(throws_invalid([&] { output_types(*aggregate(scan(t, {0, 2}), 0, {})); }));
    CHECK(throws_invalid([&] { output_types(*aggregate(scan(t, {2, 0}), 0, {{AggFn::Count, std::nullopt}})); }));
    CHECK(throws_invalid([&] {
        validate_query(*aggregate(filter(aggregate(scan(t, {0, 2}), 0, {{AggFn::Count, std::nullopt}}),
                                         {{0, CompareOp::Gt, int64_t{1}}}),
                                  0, {{AggFn::Count, std::nullopt}}));
    }));
}

TEST(plan_aggregate_output_types) {
    const Table t = readings_schema_table();
    using V       = std::vector<DataType>;
    const auto q  = aggregate(scan(t, {0, 2}), 0,
                              {{AggFn::Min, 0}, {AggFn::Max, 1}, {AggFn::Avg, 0}, {AggFn::Median, 1},
                               {AggFn::Sum, 0}, {AggFn::Sum, 1}, {AggFn::Count, std::nullopt}});
    CHECK(output_types(*q) == (V{DataType::Int64, DataType::Int64, DataType::Float64, DataType::Float64,
                                 DataType::Float64, DataType::Int64, DataType::Float64, DataType::Int64}));
}

TEST(plan_aggregate_to_string) {
    const Table t = readings_schema_table();
    CHECK_EQ(to_string(*sum_count(scan(t, {0, 2}), 0, 1)),
             std::string{"Aggregate(group=0,[sum(1),count], Scan(readings,[0,2]))"});
    CHECK_EQ(to_string(*aggregate(scan(t, {2}), std::nullopt, {{AggFn::Median, 0}, {AggFn::Avg, 0}})),
             std::string{"Aggregate(group=none,[median(0),avg(0)], Scan(readings,[2]))"});
}

TEST(reference_median_and_nulls) {
    Schema schema{{"k", DataType::Int64}, {"v", DataType::Float64}};
    Table  t{"t", schema};
    SegmentBuilder b{schema};
    for (double v : {3.0, 1.0, std::numeric_limits<double>::quiet_NaN(), 2.0}) {
        b.append_int64(0, 1);
        b.append_float64(1, v);
    }
    t.add_segment(b.seal(std::nullopt, 1024));
    // Sorted with NaN last: 1, 2, 3, NaN -> median = (2 + 3) / 2.
    const auto r = reference(*aggregate(scan(t, {0, 1}), 0, {{AggFn::Median, 1}}));
    CHECK_EQ(r.double_at(1, 0), 2.5);

    Table      empty{"e", schema};
    const auto g = reference(*aggregate(scan(empty, {0, 1}), std::nullopt, {{AggFn::Median, 1}, {AggFn::Count, std::nullopt}}));
    CHECK_EQ(g.num_rows(), std::size_t{1});
    CHECK(g.is_null(0, 0));
    CHECK_EQ(g.int_at(1, 0), int64_t{0});
}

TEST(group_index_dense_first_seen) {
    GroupIndex                  index;
    std::vector<uint32_t>       gids;
    for (int64_t k : {int64_t{7}, int64_t{-3}, int64_t{7}, INT64_MIN, int64_t{-3}, int64_t{0}})
        gids.push_back(index.find_or_insert(k));
    CHECK(gids == (std::vector<uint32_t>{0, 1, 0, 2, 1, 3}));
    CHECK_EQ(index.size(), uint32_t{4});
    CHECK_EQ(index.key_of(2), INT64_MIN);
    CHECK_EQ(index.key_of(3), int64_t{0});
}

TEST(group_index_growth_keeps_ids) {
    GroupIndex index;
    for (int64_t i = 0; i < 5'000; ++i)
        CHECK_EQ(index.find_or_insert(i * 7'919), static_cast<uint32_t>(i));
    for (int64_t i = 0; i < 5'000; ++i)
        CHECK_EQ(index.find_or_insert(i * 7'919), static_cast<uint32_t>(i));
    CHECK_EQ(index.size(), uint32_t{5'000});
    CHECK(index.capacity() >= 2 * static_cast<std::size_t>(index.size()));
}
