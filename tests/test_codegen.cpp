// Compiled engine: runtime header, compiler/cache, generator, and differential tests against the
// vectorized engine (results must be bit-identical).

#include <bit>
#include <cstdint>
#include <map>
#include <vector>

#include "codegen/runtime.h"
#include "exec/hash_table.h"
#include "test_framework.h"

using namespace wb;

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
