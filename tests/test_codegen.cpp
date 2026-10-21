// Compiled engine: runtime header, compiler/cache, generator, and differential tests against the
// vectorized engine (results must be bit-identical).

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <bit>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstdint>
#include <map>
#include <vector>

#include "codegen/compiler.h"
#include "codegen/host.h"
#include "codegen/runtime.h"
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
           "    out->emit_group(out->ctx, 7, 1.5, 2);\n    return 0;\n}\n";
}

struct Emitted {
    int64_t key   = 0;
    double  sum   = 0;
    int64_t count = 0;
    int     calls = 0;
};

void record_group(void* ctx, int64_t key, double sum, int64_t count) {
    auto* e = static_cast<Emitted*>(ctx);
    *e      = {key, sum, count, e->calls + 1};
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

// probe(k int64, amount float64) and build(k int64, grp int64), several segments each.
struct JoinTables {
    Table probe{"probe", {{"k", DataType::Int64}, {"amount", DataType::Float64}}};
    Table build{"build", {{"k", DataType::Int64}, {"grp", DataType::Int64}}};
};

std::unique_ptr<JoinTables> make_join_tables(int64_t probe_rows, int64_t (*probe_key)(int64_t), int64_t build_rows,
                                             int64_t (*build_key)(int64_t)) {
    auto t = std::make_unique<JoinTables>();
    for (int64_t start = 0; start < probe_rows; start += 700) {
        SegmentBuilder b{t->probe.schema()};
        for (int64_t i = start; i < std::min(probe_rows, start + 700); ++i) {
            b.append_int64(0, probe_key(i));
            b.append_float64(1, 0.25 * static_cast<double>(i));
        }
        t->probe.add_segment(b.seal(std::nullopt, 1024));
    }
    for (int64_t start = 0; start < build_rows; start += 250) {
        SegmentBuilder b{t->build.schema()};
        for (int64_t i = start; i < std::min(build_rows, start + 250); ++i) {
            b.append_int64(0, build_key(i));
            b.append_int64(1, i % 7);
        }
        t->build.add_segment(b.seal(std::nullopt, 1024));
    }
    return t;
}

}  // namespace

TEST(runtime_hash_matches_engine) {
    std::vector<int64_t> keys{INT64_MIN, INT64_MAX, -1, 0, 1};
    for (int64_t k = -500; k < 495; ++k)
        keys.push_back(k * 7'919);
    for (int64_t k : keys)
        CHECK_EQ(wb_rt::hash64(k), static_cast<uint64_t>(k) * 0x9E3779B97F4A7C15ULL);
}

TEST(runtime_agg_matches_engine) {
    AggHashTable     engine;  // default initial capacity, as HashAggOperator uses
    wb_rt::AggTable  rt;
    rt.init();
    uint64_t state = 1;
    for (int i = 0; i < 50'000; ++i) {
        state         = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const int64_t key = static_cast<int64_t>(state >> 33) % 3'000 - 1'500;
        const double  v   = static_cast<double>(state >> 11) * 0x1p-53;
        auto& e = engine.find_or_insert(key);
        e.sum += v;
        e.count += 1;
        auto& r = rt.find_or_insert(key);
        r.sum += v;
        r.count += 1;
    }
    CHECK_EQ(rt.size, engine.size());
    CHECK_EQ(rt.capacity, engine.capacity());
    std::map<int64_t, AggHashTable::Value> want;
    engine.for_each([&](int64_t k, const AggHashTable::Value& v) { want[k] = v; });
    std::size_t seen = 0;
    rt.for_each([&](const wb_rt::AggSlot& s) {
        const auto it = want.find(s.key);
        CHECK(it != want.end());
        CHECK_EQ(std::bit_cast<uint64_t>(s.sum), std::bit_cast<uint64_t>(it->second.sum));
        CHECK_EQ(s.count, it->second.count);
        ++seen;
    });
    CHECK_EQ(seen, want.size());
    rt.release();
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
    wb_rt::QueryOutput out{&emitted, record_group};
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
    const std::vector<int64_t> wide{INT64_MIN + 1, INT64_MAX, -1, 0, -9'000'000'000'000LL, 42};
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
    for (int64_t c : grouped.counts)
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
    CHECK_EQ(r.counts[0], int64_t{0});
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
    CHECK_EQ(global.counts[0], int64_t{0});
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
