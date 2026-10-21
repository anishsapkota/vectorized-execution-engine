// Compiled engine: runtime header, compiler/cache, generator, and differential tests against the
// vectorized engine (results must be bit-identical).

#include <unistd.h>

#include <bit>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstdint>
#include <map>
#include <vector>

#include "codegen/compiler.h"
#include "codegen/runtime.h"
#include "exec/hash_table.h"
#include "test_framework.h"

using namespace wb;
using namespace wb::codegen;

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
