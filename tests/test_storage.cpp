// Correctness checks for column encodings, borrowed selection vectors and the CSV parser.
// Each check compares the engine against a brute-force loop over the raw values.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "exec/filter.h"
#include "exec/hashagg.h"
#include "exec/parallel_agg.h"
#include "exec/scan.h"
#include "storage/csv.h"
#include "storage/table.h"
#include "storage/vector.h"
#include "test_framework.h"

using namespace wb;

namespace {

TEST(int64_encoding) {
    struct Case {
        std::vector<int64_t> values;
        std::size_t          width;
    };
    const int64_t lo = std::numeric_limits<int64_t>::min();
    const int64_t hi = std::numeric_limits<int64_t>::max();
    std::vector<Case> cases = {
        {{}, 1},
        {{7, 7, 7}, 1},
        {{-100, 155}, 1},
        {{0, 256}, 2},
        {{-5, 70'000}, 4},
        {{1'700'000'000, 1'700'065'536}, 4},
        {{0, int64_t{1} << 40}, 8},
        {{lo, hi, 0, -1}, 8},
    };
    for (const auto& c : cases) {
        Int64Vector v{c.values};
        CHECK(v.size() == c.values.size());
        CHECK(v.width_bytes() == c.width);
        for (std::size_t i = 0; i < c.values.size(); ++i)
            CHECK(v.value(i) == c.values[i]);
    }
}

struct Row {
    int64_t device;
    int64_t ts;
    double  value;
};

Table make_table(const std::vector<Row>& rows, bool narrow_float, std::size_t rows_per_segment) {
    Schema schema = {{"device_id", DataType::Int64},
                     {"ts", DataType::Int64},
                     {"value", DataType::Float64, narrow_float}};
    Table  t{"t", schema, /*sort_key_col=*/1};
    auto   builder = std::make_unique<SegmentBuilder>(schema);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        builder->append_int64(0, rows[r].device);
        builder->append_int64(1, rows[r].ts);
        builder->append_float64(2, rows[r].value);
        if (builder->num_rows() == rows_per_segment || r + 1 == rows.size()) {
            t.add_segment(builder->seal(1, 1024));
            builder = std::make_unique<SegmentBuilder>(schema);
        }
    }
    return t;
}

std::vector<Row> random_rows(std::size_t n) {
    std::mt19937_64                        rng{7};
    std::uniform_int_distribution<int64_t> device{1, 300};
    std::uniform_int_distribution<int64_t> jitter{-30, 30};
    std::normal_distribution<double>       value{50.0, 15.0};
    std::vector<Row>                       rows(n);
    int64_t                                ts = 1'700'000'000;
    for (auto& r : rows) {
        ts += 1 + jitter(rng);
        r = {device(rng), ts, value(rng)};
    }
    return rows;
}

// SELECT count(*), sum(value) WHERE ts > ts_cut AND value > value_cut, as two chained filters so
// the second one reads the first one's borrowed selection vector.
void check_chained_filters(bool narrow_float) {
    auto        rows = random_rows(50'000);
    Table       t    = make_table(rows, narrow_float, 8'000);
    int64_t     ts_cut    = rows[rows.size() / 2].ts;
    double      value_cut = 55.0;

    auto stored = [&](double v) { return narrow_float ? double(float(v)) : v; };
    int64_t want_rows = 0;
    double  want_sum  = 0.0;
    for (const auto& r : rows) {
        if (r.ts > ts_cut && stored(r.value) > value_cut) {
            ++want_rows;
            want_sum += stored(r.value);
        }
    }

    auto scan = std::make_unique<ScanOperator>(t, Predicate{1, CompareOp::Gt, ts_cut}, 4096,
                                               std::vector<std::size_t>{1, 2});
    auto f1   = std::make_unique<FilterOperator>(std::move(scan), Predicate{0, CompareOp::Gt, ts_cut});
    FilterOperator f2{std::move(f1), Predicate{1, CompareOp::Gt, value_cut}};

    int64_t got_rows = 0;
    double  got_sum  = 0.0;
    while (auto batch = f2.next()) {
        CHECK(batch->has_sel());
        CHECK(batch->sel().size() == batch->size());
        got_rows += batch->size();
        const auto& vals = static_cast<const Float64Vector&>(batch->column(1));
        for (std::uint32_t i : batch->sel())
            got_sum += vals.value(i);
    }
    CHECK(got_rows == want_rows);
    // Summation order differs (segments are sorted by ts), so compare with a tolerance.
    CHECK(std::abs(got_sum - want_sum) <= 1e-9 * std::abs(want_sum));
}

// Zone maps on a narrowed column must describe the stored (rounded) values: a literal between
// the double max and the float max must not prune a segment whose stored max exceeds it.
TEST(narrow_float_zone_map) {
    double v   = 0.1;  // float(0.1) > 0.1
    double lit = (v + double(float(v))) / 2;
    CHECK(double(float(v)) > lit && lit > v);

    Table t = make_table({{1, 1, v}}, /*narrow_float=*/true, 10);
    CHECK(t.segment(0).zone(2).max_f64() == double(float(v)));

    auto           scan = std::make_unique<ScanOperator>(t, Predicate{2, CompareOp::Gt, lit});
    FilterOperator filter{std::move(scan), Predicate{2, CompareOp::Gt, lit}};
    int64_t        rows = 0;
    while (auto batch = filter.next())
        rows += batch->size();
    CHECK(rows == 1);
}

TEST(hashagg) {
    auto  rows = random_rows(30'000);
    Table t    = make_table(rows, false, 4'000);

    std::map<int64_t, std::pair<double, int64_t>> want;
    for (const auto& r : rows) {
        want[r.device].first += r.value;
        want[r.device].second += 1;
    }

    auto            scan = std::make_unique<ScanOperator>(t, std::nullopt);
    HashAggOperator agg{std::move(scan), 0, 2};
    std::size_t     groups = 0;
    while (auto batch = agg.next()) {
        const auto& keys   = static_cast<const Int64Vector&>(batch->column(0));
        const auto& sums   = static_cast<const Float64Vector&>(batch->column(1));
        const auto& counts = static_cast<const Int64Vector&>(batch->column(2));
        for (std::uint32_t i = batch->offset(); i < batch->offset() + batch->size(); ++i) {
            auto it = want.find(keys.value(i));
            CHECK(it != want.end());
            if (it == want.end()) continue;
            CHECK(counts.value(i) == it->second.second);
            CHECK(std::abs(sums.value(i) - it->second.first) <= 1e-9 * it->second.first);
            ++groups;
        }
    }
    CHECK(groups == want.size());

    AggHashTable par = parallel_group_sum_count(t, 0, 2, 4);
    CHECK(par.size() == want.size());
    par.for_each([&](int64_t k, const AggHashTable::Value& v) {
        CHECK(v.count == want[k].second);
    });
}

bool csv_throws(const std::string& body) {
    auto path = std::filesystem::temp_directory_path() / "wb_test.csv";
    std::ofstream{path} << "device_id,ts,value\n" << body;
    Table t{"t", {{"device_id", DataType::Int64}, {"ts", DataType::Int64}, {"value", DataType::Float64}}};
    try {
        load_csv(t, path.string(), 100);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

TEST(csv) {
    CHECK(!csv_throws("1,2,3.5\n-4,5,-0.25\n"));
    CHECK(csv_throws("1,2,\n"));       // empty float
    CHECK(csv_throws("1,,3.5\n"));     // empty int
    CHECK(csv_throws("1x,2,3.5\n"));   // trailing junk in int
    CHECK(csv_throws("1,2,3.5abc\n")); // trailing junk in float
    CHECK(csv_throws("1,2\n"));        // column count
}

TEST(chained_filters_f64) { check_chained_filters(false); }
TEST(chained_filters_f32) { check_chained_filters(true); }

TEST(raw_accessors) {
    std::vector<int64_t> ints{-5, 70'000};
    Int64Vector          v{ints};
    CHECK(v.raw().width == 4);
    CHECK(v.raw().base == -5);
    CHECK(static_cast<const uint32_t*>(v.raw().codes)[1] == 70'005);

    Float64Vector f{{1.5, 2.5}, true};
    CHECK(f.raw().narrow);
    CHECK(static_cast<const float*>(f.raw().values)[1] == 2.5f);
}

TEST(scan_start_row_public) {
    const std::vector<IndexEntry> idx{{100, 0}, {200, 1024}, {300, 2048}};
    CHECK(scan_start_row(idx, Predicate{0, CompareOp::Gt, int64_t{250}}) == 1024);
    CHECK(scan_start_row(idx, Predicate{0, CompareOp::Eq, int64_t{100}}) == 0);
    CHECK(scan_start_row(idx, Predicate{0, CompareOp::Lt, int64_t{250}}) == 0);
    CHECK(scan_start_row(idx, Predicate{0, CompareOp::Ge, int64_t{99}}) == 0);
}

}  // namespace
