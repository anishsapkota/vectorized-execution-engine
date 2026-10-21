// Logical plan IR, its type checker, and lowering to the vectorized engine.

#include <string>
#include <vector>

#include "plan/result.h"
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
