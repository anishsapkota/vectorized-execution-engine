// Compiled engine: runtime header, compiler/cache, generator, and differential tests against the
// vectorized engine (results must be bit-identical).

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "codegen/compiler.h"
#include "codegen/host.h"
#include "codegen/runtime.h"
#include "exec/group_slots.h"
#include "exec/median_store.h"
#include "exec/hash_table.h"
#include "plan/lower_vec.h"
#include "plan/plan.h"
#include "storage/segment.h"
#include "test_framework.h"
#include "test_util.h"
#include "tools/datagen.h"

using namespace wb;
using namespace wb::codegen;
using namespace wb::plan;
using wbtest::make_readings;
using wbtest::make_join_tables;
using namespace wbtest;

namespace {

// Points the codegen cache at a fresh directory for the rest of the test.
std::string fresh_cache_dir() {
    char tmpl[] = "/tmp/wb_codegen_test_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    if (!dir) throw wbtest::TestFailure("mkdtemp failed");
    setenv("WB_CODEGEN_CACHE", dir, 1);
    unsetenv("WB_CXX");
    return dir;
}

// A hand-written "generated" library. `tag` makes each source (and so its cache key) unique.
std::string test_source(const std::string& tag, int abi_version = -1) {
    const std::string version =
        abi_version < 0 ? "wb_rt::kAbiVersion" : std::to_string(abi_version) + "u";
    return "// " + tag + "\n#include \"codegen/runtime.h\"\n"
           "extern \"C\" uint32_t wb_abi_version() { return " + version + "; }\n"
           "extern \"C\" int wb_query(const wb_rt::QueryInput*, wb_rt::QueryOutput* out) {\n"
           "    const uint64_t row[3] = {7, __builtin_bit_cast(uint64_t, 1.5), 2};\n"
           "    out->emit_row(out->ctx, row, 3);\n    return 0;\n}\n";
}

struct Emitted {
    int64_t key   = 0;
    double  sum   = 0;
    int64_t count = 0;
    int     calls = 0;
};

void record_row(void* ctx, const uint64_t* values, uint32_t n) {
    auto* e = static_cast<Emitted*>(ctx);
    if (n != 3) return;
    *e = {static_cast<int64_t>(values[0]), std::bit_cast<double>(values[1]), static_cast<int64_t>(values[2]), e->calls + 1};
}

template <typename F>
std::string codegen_error_text(F&& f) {
    try {
        f();
    } catch (const CodegenError& e) {
        return e.what();
    }
    return "";
}

// Runs `root` on both engines; they must agree bit for bit.
QueryResult check_engines_agree(const Node& root) {
    const auto vec = run_vectorized(root);
    const auto cg  = run_compiled(root);
    if (!(cg == vec)) throw wbtest::TestFailure(to_string(root) + ": compiled vs vectorized: " + cg.diff(vec));
    return cg;
}

std::pair<int64_t, int64_t> ts_range(const Table& t) {
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    for (const auto& seg : t.segments()) {
        lo = std::min(lo, seg->zone(1).min_i64());
        hi = std::max(hi, seg->zone(1).max_i64());
    }
    return {lo, hi};
}

}  // namespace

TEST(runtime_hash_matches_engine) {
    std::vector<int64_t> keys{INT64_MIN, INT64_MAX, -1, 0, 1};
    for (int64_t k = -500; k < 495; ++k)
        keys.push_back(k * 7'919);
    for (int64_t k : keys)
        CHECK_EQ(wb_rt::hash64(k), static_cast<uint64_t>(k) * 0x9E3779B97F4A7C15ULL);
}

TEST(runtime_group_rows_match_group_slots) {
    struct Row {
        int64_t count;
    };
    GroupSlots              engine{8, 8};
    wb_rt::GroupRows<Row>   rt;
    rt.init();
    uint64_t state = 1;
    for (int i = 0; i < 50'000; ++i) {
        state             = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const int64_t key = i % 997 == 0 ? INT64_MIN : static_cast<int64_t>(state >> 33) % 3'000 - 1'500;
        bool          f1 = false, f2 = false;
        std::byte*    es = engine.find_or_insert(key, f1);
        Row*          rr = rt.find_or_insert(key, f2);
        CHECK_EQ(f1, f2);
        if (f1) *reinterpret_cast<int64_t*>(es) = 0;
        if (f2) *rr = Row{0};
        ++*reinterpret_cast<int64_t*>(es);
        ++rr->count;
    }
    CHECK_EQ(rt.size(), engine.size());
    for (uint32_t g = 0; g < engine.size(); ++g) {
        CHECK_EQ(rt.key_of(g), engine.key_of(g));
        CHECK_EQ(rt.row_of(g).count, *reinterpret_cast<const int64_t*>(engine.state_of(g)));
    }
    rt.release();
}

namespace {
template <typename T>
T rt_random_value(uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    if constexpr (std::is_floating_point_v<T>) {
        const uint64_t pick = state >> 58;
        if (pick == 0) return std::numeric_limits<T>::quiet_NaN();
        if (pick == 1) return -T{0};
        if (pick == 2) return T{0};
        return static_cast<T>(static_cast<double>(state >> 40) / 64.0 - 100.0);
    } else {
        return static_cast<int64_t>(state >> 20) - (int64_t{1} << 43);
    }
}

template <typename T>
void check_runtime_median_store() {
    std::vector<std::size_t> sizes{254, 255, 256, 509, 510, 511, 513};
    uint64_t                 state = 23;
    for (int g = 0; g < 200; ++g)
        sizes.push_back(1 + static_cast<std::size_t>((state = state * 6364136223846793005ULL + 1) >> 54) % 700);
    MedianStore<T>                 engine;
    wb_rt::MedianStore<T>          rt;
    std::vector<MedianHandle>      eh(sizes.size());
    std::vector<wb_rt::MedianHandle> rh(sizes.size());
    for (std::size_t k = 0; k < 700; ++k) {
        for (std::size_t g = 0; g < sizes.size(); ++g) {
            if (k >= sizes[g]) continue;
            const T v = rt_random_value<T>(state);
            engine.append(eh[g], v);
            rt.append(rh[g], v);
        }
    }
    std::vector<T> scratch_e;
    std::vector<T> scratch_r(rt.max_group);
    for (std::size_t g = 0; g < sizes.size(); ++g) {
        CHECK_EQ(std::bit_cast<uint64_t>(rt.finalize_group(rh[g], scratch_r.data())),
                 std::bit_cast<uint64_t>(engine.finalize_group(eh[g], scratch_e)));
    }
    rt.release();

    MedianStore<T>        e1;
    wb_rt::MedianStore<T> r1;
    MedianHandle          h1;
    wb_rt::MedianHandle   h2;
    for (int i = 0; i < 100'000; ++i) {
        const T v = rt_random_value<T>(state);
        e1.append(h1, v);
        r1.append(h2, v);
    }
    CHECK_EQ(std::bit_cast<uint64_t>(r1.finalize_single(h2)), std::bit_cast<uint64_t>(e1.finalize_single(h1)));
    r1.release();
}
}  // namespace

TEST(runtime_median_store_matches_vectorized) {
    check_runtime_median_store<double>();
    check_runtime_median_store<float>();
    check_runtime_median_store<int64_t>();
}

TEST(runtime_join_matches_engine) {
    // Payload column 0 = insertion index, column 1 = a double, to check bit-exact storage.
    wb_rt::JoinTable jt;
    jt.init(2);
    std::map<int64_t, std::vector<uint64_t>> inserted;  // key -> insertion indices, in order
    for (uint64_t i = 0; i < 5'000; ++i) {
        const int64_t  key        = static_cast<int64_t>((i * 7'919) % 613) - 300;
        const double   d          = 0.1 * static_cast<double>(i);
        const uint64_t payload[2] = {i, std::bit_cast<uint64_t>(d)};
        jt.append(key, payload);
        inserted[key].push_back(i);
    }
    jt.finalize();
    for (const auto& [key, rows] : inserted) {
        const wb_rt::JoinBucket* b = jt.find(key);
        CHECK(b != nullptr);
        CHECK_EQ(b->count, static_cast<uint32_t>(rows.size()));
        for (uint32_t j = 0; j < b->count; ++j) {
            CHECK_EQ(jt.col(0, b->first + j), rows[j]);
            CHECK_EQ(jt.col(1, b->first + j), std::bit_cast<uint64_t>(0.1 * static_cast<double>(rows[j])));
        }
    }
    CHECK(jt.find(10'000) == nullptr);
    jt.release();

    wb_rt::JoinTable empty;
    empty.init(1);
    empty.finalize();
    CHECK(empty.find(0) == nullptr);
    empty.release();
}

TEST(compile_and_call) {
    fresh_cache_dir();
    const auto     compiled = compile_source(test_source("compile_and_call"), {});
    Emitted        emitted;
    wb_rt::QueryOutput out{&emitted, record_row, 0, 0};
    CHECK_EQ(compiled.fn(nullptr, &out), 0);
    CHECK_EQ(emitted.calls, 1);
    CHECK_EQ(emitted.key, int64_t{7});
    CHECK_EQ(emitted.sum, 1.5);
    CHECK_EQ(emitted.count, int64_t{2});
    CHECK(!compiled.stats.cache_hit);
    CHECK(compiled.stats.compile_ms > 0.0);
}

TEST(compile_cache_hit) {
    fresh_cache_dir();
    const auto src = test_source("compile_cache_hit");
    compile_source(src, {});
    const std::size_t before = compiler_invocations();
    const auto        again  = compile_source(src, {});
    CHECK(again.stats.cache_hit);
    CHECK_EQ(compiler_invocations(), before);
}

TEST(compile_missing_compiler_raises) {
    fresh_cache_dir();
    setenv("WB_CXX", "/nonexistent", 1);
    const auto text = codegen_error_text([] { compile_source(test_source("missing_compiler"), {}); });
    unsetenv("WB_CXX");
    CHECK(text.find("/nonexistent") != std::string::npos);
}

TEST(compile_abi_mismatch_raises) {
    fresh_cache_dir();
    const auto text = codegen_error_text([] { compile_source(test_source("abi", 999), {}); });
    CHECK(text.find("999") != std::string::npos);
}

TEST(corrupt_cached_library_raises) {
    const auto dir = fresh_cache_dir();
    const auto src = test_source("corrupt_cache");
    char       name[32];
    std::snprintf(name, sizeof name, "%016llx", static_cast<unsigned long long>(cache_key(src, {})));
    std::ofstream{dir + "/" + name + "." + library_extension()} << "not a library";
    const auto text = codegen_error_text([&] { compile_source(src, {}); });
    CHECK(!text.empty());
}

TEST(cache_key_depends_on_opt_level) {
    const auto src = test_source("key_opt");
    CHECK(cache_key(src, {.opt_level = 1}) != cache_key(src, {.opt_level = 3}));
}

TEST(cache_key_depends_on_runtime_header) {
    const auto src = test_source("key_runtime");
    CHECK(cache_key(src, {}, "a") != cache_key(src, {}, "b"));
}

TEST(cg_q1_q4_shapes) {
    fresh_cache_dir();
    auto t        = make_readings(200'000);
    auto [lo, hi] = ts_range(*t);
    const int64_t mid = lo + (hi - lo) / 2;
    check_engines_agree(*sum_count(filter(scan(*t, {1, 2}), {{0, CompareOp::Gt, mid}}), std::nullopt, 1));
    check_engines_agree(*sum_count(filter(scan(*t, {2}), {{0, CompareOp::Gt, 50.0}}), std::nullopt, 0));
    check_engines_agree(*sum_count(
        filter(scan(*t, {2, 0}), {{0, CompareOp::Gt, 50.0}, {1, CompareOp::Lt, int64_t{150}}}), std::nullopt, 0));
    const auto grouped = check_engines_agree(*sum_count(scan(*t, {0, 2}), 0, 1));
    CHECK_EQ(grouped.num_rows(), std::size_t{300});
}

TEST(cg_sort_key_ops) {
    fresh_cache_dir();
    auto          t    = make_readings(100'000);
    const int64_t real = t->segment(5).zone(1).min_i64();  // a ts value that exists
    for (CompareOp op : {CompareOp::Gt, CompareOp::Ge, CompareOp::Eq, CompareOp::Lt, CompareOp::Ne}) {
        check_engines_agree(*sum_count(filter(scan(*t, {1, 2}), {{0, op, real}}), std::nullopt, 1));
    }
}

TEST(cg_extreme_keys) {
    // One segment needs 8-byte FOR codes (range spans all of int64); the others need 2 bytes.
    fresh_cache_dir();
    Schema schema{{"k", DataType::Int64}, {"v", DataType::Float64}};
    Table  t{"extreme", schema};
    const std::vector<int64_t> wide{INT64_MIN, INT64_MIN + 1, INT64_MAX, -1, 0, -9'000'000'000'000LL, 42};
    std::size_t                rows = 0;
    for (int seg = 0; seg < 4; ++seg) {
        SegmentBuilder b{schema};
        for (int i = 0; i < 1'000; ++i) {
            const int64_t k = seg == 2 ? wide[i % wide.size()] : 1'000 + (i % 37) - 18;
            b.append_int64(0, k);
            b.append_float64(1, 0.5 * i + seg);
            ++rows;
        }
        t.add_segment(b.seal(std::nullopt, 1024));
    }
    CHECK_EQ(static_cast<const Int64Vector&>(t.segment(2).column(0)).width_bytes(), std::size_t{8});

    const auto grouped = check_engines_agree(*sum_count(scan(t, {0, 1}), 0, 1));
    int64_t    total   = 0;
    for (int64_t c : grouped.columns[2].ints)
        total += c;
    CHECK_EQ(total, static_cast<int64_t>(rows));
    check_engines_agree(*sum_count(filter(scan(t, {0, 1}), {{0, CompareOp::Gt, int64_t{INT64_MIN}}}), 0, 1));
    check_engines_agree(*sum_count(filter(scan(t, {0, 1}), {{0, CompareOp::Lt, int64_t{0}}}), std::nullopt, 1));
}

TEST(cg_all_pruned) {
    fresh_cache_dir();
    auto t        = make_readings(50'000);
    auto [lo, hi] = ts_range(*t);
    const auto r  = check_engines_agree(*sum_count(filter(scan(*t, {1, 2}), {{0, CompareOp::Gt, hi}}), std::nullopt, 1));
    CHECK_EQ(r.num_rows(), std::size_t{1});
    CHECK_EQ(r.int_at(1, 0), int64_t{0});
}

TEST(narrow_float_filter_matches) {
    fresh_cache_dir();
    auto t = make_readings(50'000, 8'000, /*narrow_float=*/true);
    check_engines_agree(*sum_count(filter(scan(*t, {2}), {{0, CompareOp::Gt, 50.1}}), std::nullopt, 0));
    check_engines_agree(*sum_count(filter(scan(*t, {0, 2}), {{1, CompareOp::Le, 50.1}}), 0, 1));
}

TEST(cg_prune_uses_table_column_through_projection) {
    fresh_cache_dir();
    auto t = make_readings(50'000);
    check_engines_agree(*sum_count(filter(scan(*t, {2, 1}), {{0, CompareOp::Gt, 80.0}}), std::nullopt, 0));
}

TEST(empty_table_cg) {
    fresh_cache_dir();
    auto       t      = make_readings(0);
    const auto global = check_engines_agree(*sum_count(scan(*t, {2}), std::nullopt, 0));
    CHECK_EQ(global.num_rows(), std::size_t{1});
    CHECK(global.is_null(0, 0));  // SQL: SUM over no rows is NULL
    CHECK_EQ(global.int_at(1, 0), int64_t{0});
    CHECK_EQ(check_engines_agree(*sum_count(scan(*t, {0, 2}), 0, 1)).num_rows(), std::size_t{0});
}

TEST(cg_q5_join_agg) {
    fresh_cache_dir();
    auto orders    = make_orders(200'000, 10'000, 1.0);
    auto customers = make_customers(10'000);
    const auto r   = check_engines_agree(
        *sum_count(join(filter(scan(*orders, {0, 1}), {{1, CompareOp::Gt, 50.0}}), scan(*customers, {0, 1}), 0, 0), 3, 1));
    CHECK_EQ(r.num_rows(), std::size_t{50});
}

TEST(cg_join_duplicates_both_sides) {
    fresh_cache_dir();
    auto t = make_join_tables(3'000, [](int64_t i) { return i % 400; }, 900, [](int64_t i) { return 200 + i % 300; });
    check_engines_agree(*sum_count(join(scan(t->probe, {0, 1}), scan(t->build, {0, 1}), 0, 0), 0, 1));
    check_engines_agree(*sum_count(join(scan(t->probe, {0, 1}), scan(t->build, {0, 1}), 0, 0), 3, 1));
    check_engines_agree(*sum_count(join(scan(t->probe, {1, 0}), scan(t->build, {1, 0}), 1, 1), std::nullopt, 0));
}

TEST(cg_join_no_matches) {
    fresh_cache_dir();
    auto t = make_join_tables(1'000, [](int64_t i) { return i % 100; }, 500, [](int64_t i) { return 1'000 + i % 100; });
    CHECK_EQ(check_engines_agree(*sum_count(join(scan(t->probe, {0, 1}), scan(t->build, {0, 1}), 0, 0), 0, 1)).num_rows(),
             std::size_t{0});
    const auto global =
        check_engines_agree(*sum_count(join(scan(t->probe, {0, 1}), scan(t->build, {0, 1}), 0, 0), std::nullopt, 1));
    CHECK_EQ(global.int_at(1, 0), int64_t{0});
}

TEST(empty_table_cg_join) {
    fresh_cache_dir();
    auto t = make_join_tables(1'000, [](int64_t i) { return i % 100; }, 0, [](int64_t i) { return i; });
    CHECK_EQ(check_engines_agree(*sum_count(join(scan(t->probe, {0, 1}), scan(t->build, {0, 1}), 0, 0), 0, 1)).num_rows(),
             std::size_t{0});
    check_engines_agree(*sum_count(join(scan(t->probe, {0, 1}), scan(t->build, {0, 1}), 0, 0), std::nullopt, 1));
}

TEST(join_with_filters_on_both_sides) {
    // First conjuncts on different columns and types on each side, so the host must prune each
    // scan with its own predicate and number the scans the way the generator does.
    fresh_cache_dir();
    auto t = make_join_tables(3'000, [](int64_t i) { return i % 400; }, 900, [](int64_t i) { return 200 + i % 300; });
    check_engines_agree(*sum_count(join(filter(scan(t->probe, {1, 0}), {{0, CompareOp::Gt, 300.0}}),
                                        filter(scan(t->build, {1, 0}), {{0, CompareOp::Lt, int64_t{4}}}), 1, 1),
                                   3, 0));
}

TEST(concurrent_processes_share_cache) {
    // Several processes missing the cache for the same key at once must all succeed.
    fresh_cache_dir();
    const auto        src = test_source("concurrent_cache");
    std::vector<pid_t> children;
    for (int i = 0; i < 8; ++i) {
        const pid_t pid = fork();
        if (pid == 0) {
            int code = 0;
            try {
                compile_source(src, {});
            } catch (...) {
                code = 1;
            }
            _exit(code);
        }
        children.push_back(pid);
    }
    int failures = 0;
    for (pid_t pid : children) {
        int status = 0;
        waitpid(pid, &status, 0);
        failures += !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    CHECK_EQ(failures, 0);
}

TEST(cache_dir_writable_by_others_is_rejected) {
    // Libraries in the cache are dlopen'ed, so a directory others can write to is code injection.
    const auto dir = fresh_cache_dir();
    chmod(dir.c_str(), 0777);
    const auto text = codegen_error_text([] { compile_source(test_source("shared_cache_dir"), {}); });
    chmod(dir.c_str(), 0700);
    CHECK(text.find("writable") != std::string::npos);
    CHECK(text.find(dir) != std::string::npos);
}

namespace {
const std::vector<AggExpr> kDecomposableAggs{{AggFn::Count, std::nullopt}, {AggFn::Sum, 2}, {AggFn::Sum, 0},
                                             {AggFn::Min, 2},              {AggFn::Max, 2}, {AggFn::Min, 1},
                                             {AggFn::Max, 1},              {AggFn::Avg, 2}, {AggFn::Avg, 0}};

template <typename F>
std::string overflow_message(F&& f) {
    try {
        f();
    } catch (const std::overflow_error& e) {
        return e.what();
    }
    return "";
}
}  // namespace

TEST(cg_all_decomposable_aggs) {
    fresh_cache_dir();
    auto t = make_readings(50'000);
    check_engines_agree(*aggregate(scan(*t, {0, 1, 2}), 0, kDecomposableAggs));
    check_engines_agree(*aggregate(scan(*t, {0, 1, 2}), std::nullopt, kDecomposableAggs));
}

TEST(cg_min_max_nan_and_negative_zero) {
    fresh_cache_dir();
    auto t = make_nan_table();
    check_engines_agree(*aggregate(scan(*t, {0, 1}), 0, {{AggFn::Min, 1}, {AggFn::Max, 1}, {AggFn::Sum, 1}}));
    check_engines_agree(*aggregate(scan(*t, {0, 1}), std::nullopt, {{AggFn::Min, 1}, {AggFn::Max, 1}, {AggFn::Sum, 1}}));
}

TEST(cg_sum_int128_and_overflow) {
    fresh_cache_dir();
    auto ok = make_table("t", int128_intermediate_rows());
    check_engines_agree(*aggregate(scan(*ok, {0, 1}), 0, {{AggFn::Sum, 1}, {AggFn::Avg, 1}}));
    auto bad     = make_table("t", overflow_rows());
    auto grouped = aggregate(scan(*bad, {0, 1}), 0, {{AggFn::Sum, 1}});
    auto global  = aggregate(scan(*bad, {0, 1}), std::nullopt, {{AggFn::Sum, 1}});
    CHECK(overflow_message([&] { run_compiled(*grouped); }).find("42") != std::string::npos);
    CHECK(overflow_message([&] { run_vectorized(*grouped); }).find("42") != std::string::npos);
    CHECK(overflow_message([&] { run_compiled(*global); }).find("global") != std::string::npos);
    CHECK(overflow_message([&] { run_vectorized(*global); }).find("global") != std::string::npos);
}

TEST(cg_aggs_above_join) {
    fresh_cache_dir();
    auto t = make_join_tables(3'000, [](int64_t i) { return i % 400; }, 900, [](int64_t i) { return 200 + i % 300; });
    check_engines_agree(*aggregate(join(scan(t->probe, {0, 1}), scan(t->build, {0, 1}), 0, 0), 3,
                                   {{AggFn::Min, 1}, {AggFn::Max, 1}, {AggFn::Avg, 1}, {AggFn::Count, std::nullopt}}));
}

TEST(cg_global_all_pruned_is_null) {
    fresh_cache_dir();
    auto t        = make_readings(50'000);
    auto [lo, hi] = ts_range(*t);
    const auto r  = check_engines_agree(*aggregate(filter(scan(*t, {1, 2}), {{0, CompareOp::Gt, hi}}), std::nullopt,
                                                   {{AggFn::Count, std::nullopt}, {AggFn::Sum, 1}, {AggFn::Min, 1}, {AggFn::Avg, 1}}));
    CHECK_EQ(r.num_rows(), std::size_t{1});
    CHECK_EQ(r.int_at(0, 0), int64_t{0});
    CHECK(r.is_null(1, 0));
    CHECK(r.is_null(2, 0));
    CHECK(r.is_null(3, 0));
}

TEST(cg_count_only) {
    fresh_cache_dir();
    auto t = make_readings(50'000);
    check_engines_agree(*aggregate(scan(*t, {0}), 0, {{AggFn::Count, std::nullopt}}));
    check_engines_agree(*aggregate(scan(*t, {0}), std::nullopt, {{AggFn::Count, std::nullopt}}));
}

TEST(cg_median_matrix) {
    fresh_cache_dir();
    auto d = make_kv_double(median_double_rows(), 5);
    auto i = make_table("t", median_int_rows(), 5);
    auto x = make_table("t", int64_extreme_rows());
    auto n = make_kv_double(nan_median_rows());
    for (const Table* t : {d.get(), i.get(), x.get(), n.get()}) {
        check_engines_agree(*aggregate(scan(*t, {0, 1}), 0, {{AggFn::Median, 1}, {AggFn::Count, std::nullopt}}));
        check_engines_agree(*aggregate(scan(*t, {0, 1}), std::nullopt, {{AggFn::Median, 1}}));
    }
    auto r        = make_readings(50'000);
    auto [lo, hi] = ts_range(*r);
    check_engines_agree(*aggregate(scan(*r, {0, 2}), 0, {{AggFn::Median, 1}}));
    check_engines_agree(
        *aggregate(filter(scan(*r, {1, 2}), {{0, CompareOp::Gt, lo + (hi - lo) / 2}}), std::nullopt, {{AggFn::Median, 1}}));
    check_engines_agree(*aggregate(filter(scan(*r, {1, 2}), {{0, CompareOp::Gt, hi}}), std::nullopt, {{AggFn::Median, 1}}));
}

TEST(cg_many_aggs_same_and_different_columns) {
    fresh_cache_dir();
    auto t = make_readings(50'000);
    check_engines_agree(*aggregate(scan(*t, {0, 1, 2}), 0,
                                   {{AggFn::Min, 2}, {AggFn::Max, 2}, {AggFn::Median, 2}, {AggFn::Median, 1},
                                    {AggFn::Avg, 1}, {AggFn::Count, std::nullopt}}));
    check_engines_agree(*aggregate(scan(*t, {0, 1, 2}), std::nullopt,
                                   {{AggFn::Median, 2}, {AggFn::Median, 1}, {AggFn::Min, 2}, {AggFn::Count, std::nullopt}}));
}

TEST(cg_group_column_aggregated) {
    fresh_cache_dir();
    auto t = make_readings(20'000);
    check_engines_agree(*aggregate(scan(*t, {0, 2}), 0,
                                   {{AggFn::Min, 0}, {AggFn::Max, 0}, {AggFn::Median, 0}, {AggFn::Count, std::nullopt}}));
}

TEST(cg_median_above_join) {
    fresh_cache_dir();
    auto orders    = make_orders(200'000, 10'000, 1.0);
    auto customers = make_customers(10'000);
    check_engines_agree(*aggregate(join(filter(scan(*orders, {0, 1}), {{1, CompareOp::Gt, 50.0}}), scan(*customers, {0, 1}), 0, 0),
                                   3, {{AggFn::Avg, 1}, {AggFn::Median, 1}}));
}

TEST(cg_median_signed_zeros_and_nan_payloads) {
    fresh_cache_dir();
    auto t = make_kv_double(signed_zero_median_rows(), 3);
    check_engines_agree(*aggregate(scan(*t, {0, 1}), 0, {{AggFn::Median, 1}}));
    check_engines_agree(*aggregate(scan(*t, {0, 1}), std::nullopt, {{AggFn::Median, 1}}));
}

TEST(cg_median_block_boundaries) {
    fresh_cache_dir();
    auto t = make_kv_double(median_boundary_rows(), 300);
    check_engines_agree(*aggregate(scan(*t, {0, 1}), 0, {{AggFn::Median, 1}, {AggFn::Count, std::nullopt}}));
}

TEST(cg_int64_min_key_with_median) {
    fresh_cache_dir();
    auto t = make_kv_double(int64_min_median_rows(), 128);
    check_engines_agree(*aggregate(scan(*t, {0, 1}), 0, {{AggFn::Sum, 1}, {AggFn::Median, 1}, {AggFn::Count, std::nullopt}}));
}

TEST(cg_many_int128_fields) {
    fresh_cache_dir();
    auto t = make_readings(50'000);
    check_engines_agree(*aggregate(scan(*t, {0, 1, 2}), 0,
                                   {{AggFn::Sum, 0}, {AggFn::Avg, 0}, {AggFn::Sum, 1}, {AggFn::Min, 2},
                                    {AggFn::Count, std::nullopt}}));
}

TEST(cg_median_narrow_float_scan_and_join) {
    fresh_cache_dir();
    auto t = make_readings(50'000, 8'000, /*narrow_float=*/true);
    check_engines_agree(*aggregate(scan(*t, {0, 2}), 0, {{AggFn::Median, 1}}));
    check_engines_agree(*aggregate(scan(*t, {2}), std::nullopt, {{AggFn::Median, 0}}));
    // Narrow table on both sides of a join: probe column 2 is a scan column, build column 4 comes
    // out of the join table as double.
    auto devices = make_readings(2'000, 500, /*narrow_float=*/true);
    check_engines_agree(*aggregate(join(scan(*t, {0, 2}), scan(*devices, {0, 2}), 0, 0), 0,
                                   {{AggFn::Median, 1}, {AggFn::Median, 3}}));
}

TEST(median_selection_is_not_quadratic_on_organ_pipe_data) {
    // Rise-then-fall values defeat a middle-element pivot (quadratic); selection must stay
    // O(n log n) in both engines. 400k values took ~28 s with the old middle pivot.
    constexpr int64_t n = 400'000;
    MedianStore<double>        engine;
    wb_rt::MedianStore<double> rt;
    MedianHandle               eh;
    wb_rt::MedianHandle        rh;
    std::vector<double>        values;
    for (int64_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(i < n / 2 ? i : n - i);
        values.push_back(v);
        engine.append(eh, v);
        rt.append(rh, v);
    }
    std::sort(values.begin(), values.end());
    const double want  = (values[n / 2 - 1] + values[n / 2]) / 2.0;
    const auto   start = std::chrono::steady_clock::now();
    CHECK_EQ(engine.finalize_single(eh), want);
    CHECK_EQ(rt.finalize_single(rh), want);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CHECK(seconds < 2.0);
    rt.release();
}
