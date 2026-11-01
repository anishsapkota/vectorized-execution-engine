// wb_codegen: the codegen-vs-vectorized experiment.
// break-even = runs needed before the one-off cost (generate + compile + load) is repaid. Runs Q1-Q5 (spec section 8) on the
// vectorized engine and on generated code at -O1 and -O3, and checks they agree bit for bit.
//
//   ./wb_codegen [--csv data/readings.csv] [--orders 2000000] [--customers 100000] [--iters 7]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "codegen/host.h"
#include "plan/lower_vec.h"
#include "plan/plan.h"
#include "storage/csv.h"
#include "storage/table.h"
#include "tools/allocator_info.h"
#include "tools/datagen.h"

// Counts operator new calls (the engine's allocations). Generated code allocates its hash tables
// with malloc, so its tables are not counted here; see the results notes.
namespace {
std::atomic<std::uint64_t> g_allocs{0};
void*                      counted_alloc(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc{};
}
}  // namespace

void* operator new(std::size_t n) { return counted_alloc(n); }
void* operator new[](std::size_t n) { return counted_alloc(n); }
void  operator delete(void* p) noexcept { std::free(p); }
void  operator delete[](void* p) noexcept { std::free(p); }
void  operator delete(void* p, std::size_t) noexcept { std::free(p); }
void  operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

using namespace wb;
using namespace wb::plan;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string  csv       = "data/readings.csv";
    std::int64_t orders    = 2'000'000;
    std::int64_t customers = 100'000;
    int          iters     = 7;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--csv") o.csv = value;
        else if (flag == "--orders") o.orders = std::stoll(value);
        else if (flag == "--customers") o.customers = std::stoll(value);
        else if (flag == "--iters") o.iters = std::stoi(value);
        else throw std::invalid_argument("unknown flag " + flag);
    }
    return o;
}

struct Timing {
    double        best_ms   = 0;
    double        median_ms = 0;
    std::uint64_t allocs    = 0;  // per run
    QueryResult   result;
};

Timing time_runs(int iters, const std::function<QueryResult()>& run) {
    std::vector<double> ms;
    Timing              t;
    const auto          a0 = g_allocs.load();
    for (int i = 0; i < iters; ++i) {
        const auto start = Clock::now();
        t.result         = run();
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    t.allocs = (g_allocs.load() - a0) / static_cast<std::uint64_t>(iters);
    std::sort(ms.begin(), ms.end());
    t.best_ms   = ms.front();
    t.median_ms = ms[ms.size() / 2];
    return t;
}

bool g_all_match = true;

void bench_query(const char* id, const Node& plan, int iters) {
    const Timing vec = time_runs(iters, [&] { return run_vectorized(plan); });
    std::printf("%-3s %-7s %8s %10s %8s %10.2f %10.2f %8s %10s %9llu %5s\n", id, "vec", "-", "-", "-", vec.best_ms,
                vec.median_ms, "1.00x", "-", static_cast<unsigned long long>(vec.allocs), "-");

    for (int opt : {1, 3}) {
        // Cold: generate + compile + load a fresh library; warm: run that library repeatedly.
        const codegen::PreparedQuery q  = codegen::prepare(plan, {.opt_level = opt, .use_cache = false});
        const auto&                  st = q.compiled.stats;
        const Timing                 cg = time_runs(iters, [&] { return codegen::execute(q); });

        const bool  match   = cg.result == vec.result;
        g_all_match         = g_all_match && match;
        const double gain   = vec.best_ms - cg.best_ms;
        char         breakeven[32];
        if (gain <= 0) std::snprintf(breakeven, sizeof breakeven, "inf");
        else std::snprintf(breakeven, sizeof breakeven, "%.0f", (st.generate_ms + st.compile_ms + st.load_ms) / gain);
        char speedup[32];
        std::snprintf(speedup, sizeof speedup, "%.2fx", vec.best_ms / cg.best_ms);
        const std::string engine = "cg-O" + std::to_string(opt);
        std::printf("%-3s %-7s %8.2f %10.1f %8.2f %10.2f %10.2f %8s %10s %9llu %5s\n", id, engine.c_str(),
                    st.generate_ms, st.compile_ms, st.load_ms, cg.best_ms, cg.median_ms, speedup, breakeven,
                    static_cast<unsigned long long>(cg.allocs), match ? "yes" : "NO");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);

        Table readings{"readings",
                       {{"device_id", DataType::Int64}, {"ts", DataType::Int64}, {"value", DataType::Float64}},
                       /*sort_key_col=*/1};
        load_csv(readings, o.csv, 65536);
        int64_t lo = std::numeric_limits<int64_t>::max(), hi = std::numeric_limits<int64_t>::min();
        for (const auto& seg : readings.segments()) {
            lo = std::min(lo, seg->zone(1).min_i64());
            hi = std::max(hi, seg->zone(1).max_i64());
        }
        const int64_t mid       = lo + (hi - lo) / 2;
        auto          orders    = make_orders(o.orders, o.customers, 1.0);
        auto          customers = make_customers(o.customers);

        const auto q1 = sum_count(filter(scan(readings, {1, 2}), {{0, CompareOp::Gt, mid}}), std::nullopt, 1);
        const auto q2 = sum_count(filter(scan(readings, {2}), {{0, CompareOp::Gt, 50.0}}), std::nullopt, 0);
        const auto q3 = sum_count(
            filter(scan(readings, {2, 0}), {{0, CompareOp::Gt, 50.0}, {1, CompareOp::Lt, int64_t{5000}}}), std::nullopt, 0);
        const auto q4 = sum_count(scan(readings, {0, 2}), 0, 1);
        const auto q5 = sum_count(
            join(filter(scan(*orders, {0, 1}), {{1, CompareOp::Gt, 50.0}}), scan(*customers, {0, 1}), 0, 0), 3, 1);
        const auto q6 = aggregate(scan(readings, {0, 2}), 0,
                                  {{AggFn::Min, 1}, {AggFn::Max, 1}, {AggFn::Avg, 1}, {AggFn::Count, std::nullopt}});
        const auto q7 = aggregate(scan(readings, {0, 2}), 0, {{AggFn::Median, 1}});
        const auto q8 = aggregate(filter(scan(readings, {1, 2}), {{0, CompareOp::Gt, mid}}), std::nullopt,
                                  {{AggFn::Median, 1}});
        const auto q9 = aggregate(
            join(filter(scan(*orders, {0, 1}), {{1, CompareOp::Gt, 50.0}}), scan(*customers, {0, 1}), 0, 0), 3,
            {{AggFn::Avg, 1}, {AggFn::Median, 1}});

        std::size_t rows = 0;
        for (const auto& seg : readings.segments())
            rows += seg->num_rows();
        std::printf("allocator: %s\n", allocator_name().c_str());
        std::printf("readings %zu rows (%zu segments), orders %lld, customers %lld, iters %d, single-threaded\n", rows,
                    readings.num_segments(), static_cast<long long>(o.orders), static_cast<long long>(o.customers),
                    o.iters);
        int n = 0;
        for (const auto* q : {&q1, &q2, &q3, &q4, &q5, &q6, &q7, &q8, &q9})
            std::printf("Q%d %s\n", ++n, to_string(**q).c_str());
        std::printf("\n");
        std::printf("%-3s %-7s %8s %10s %8s %10s %10s %8s %10s %9s %5s\n", "q", "engine", "gen ms", "compile ms",
                    "load ms", "best ms", "median ms", "speedup", "break-even", "allocs", "match");

        bench_query("Q1", *q1, o.iters);
        bench_query("Q2", *q2, o.iters);
        bench_query("Q3", *q3, o.iters);
        bench_query("Q4", *q4, o.iters);
        bench_query("Q5", *q5, o.iters);
        bench_query("Q6", *q6, o.iters);
        bench_query("Q7", *q7, o.iters);
        bench_query("Q8", *q8, o.iters);
        bench_query("Q9", *q9, o.iters);

        if (!g_all_match) {
            std::fprintf(stderr, "error: compiled and vectorized results differ\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
