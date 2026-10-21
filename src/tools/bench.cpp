// wb_bench: load time, memory footprint, allocation counts and query latency for the
// readings workload. Run the same binary before and after a storage/exec change and diff
// the output.
//
//   ./wb_gen --rows 10000000 --out data/readings.csv
//   ./wb_bench --csv data/readings.csv --iters 7 > bench/results/<label>.txt

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "exec/filter.h"
#include "exec/hashagg.h"
#include "exec/parallel_agg.h"
#include "exec/scan.h"
#include "storage/csv.h"
#include "storage/table.h"
#include "storage/vector.h"

// ---------------------------------------------------------------------------
// Allocation counting: replace global operator new/delete. Every std::vector,
// std::string spill and make_unique goes through here.
// ---------------------------------------------------------------------------

namespace {
std::atomic<std::uint64_t> g_alloc_count{0};
std::atomic<std::uint64_t> g_alloc_bytes{0};

void* counted_alloc(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
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
using Clock = std::chrono::steady_clock;

struct AllocSnapshot {
    std::uint64_t count = g_alloc_count.load(std::memory_order_relaxed);
    std::uint64_t bytes = g_alloc_bytes.load(std::memory_order_relaxed);
};

// Current physical footprint of the process (what Activity Monitor shows as "Memory").
std::size_t current_rss_bytes() {
#if defined(__APPLE__)
    task_vm_info_data_t    info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
        KERN_SUCCESS)
        return info.phys_footprint;
    return 0;
#else
    long  pages = 0, resident = 0;
    FILE* f     = std::fopen("/proc/self/statm", "r");
    if (f) {
        if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
        std::fclose(f);
    }
    return static_cast<std::size_t>(resident) * 4096;
#endif
}

std::size_t peak_rss_bytes() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return static_cast<std::size_t>(ru.ru_maxrss);  // bytes on macOS
#else
    return static_cast<std::size_t>(ru.ru_maxrss) * 1024;  // KiB on Linux
#endif
}

double mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

// ---------------------------------------------------------------------------
// API-dependent consumers. These are the only lines that touch column data
// directly, so they are the only lines that change when the column API does.
// ---------------------------------------------------------------------------

// Sum column `col` over the live rows of a filtered batch.
double sum_selected(const Batch& batch, std::size_t col) {
    double sum = 0.0;
    static_cast<const Float64Vector&>(batch.column(col)).visit([&](auto vals) {
        for (std::uint32_t i : batch.sel())
            sum += vals[i];
    });
    return sum;
}

// Fold one HashAgg output batch {key, sum, count} into running totals.
void fold_agg_batch(const Batch& batch, double& total_sum, std::int64_t& total_count) {
    const auto&   sums   = static_cast<const Float64Vector&>(batch.column(1));
    const auto&   counts = static_cast<const Int64Vector&>(batch.column(2));
    std::uint32_t end    = batch.offset() + batch.size();
    for (std::uint32_t i = batch.offset(); i < end; ++i) {
        total_sum += sums.value(i);
        total_count += counts.value(i);
    }
}

// ---------------------------------------------------------------------------
// Queries. Each returns a checksum line so before/after results can be compared.
// ---------------------------------------------------------------------------

struct QueryResult {
    std::int64_t rows  = 0;
    double       sum   = 0.0;
    std::int64_t count = 0;  // groups for aggregations
};

// SELECT count(*), sum(value) FROM readings WHERE <col> <op> <lit>
QueryResult filter_sum(const Table& t, std::size_t col, CompareOp op,
                       std::variant<std::int64_t, double, std::string> lit) {
    // Project {col, value} (or just {value} when filtering on value itself).
    std::vector<std::size_t> proj = col == 2 ? std::vector<std::size_t>{2}
                                             : std::vector<std::size_t>{col, 2};
    std::size_t sum_pos = proj.size() - 1;

    auto scan   = std::make_unique<ScanOperator>(t, Predicate{col, op, lit}, 4096, proj);
    auto filter = std::make_unique<FilterOperator>(std::move(scan), Predicate{0, op, lit});

    QueryResult r;
    while (auto batch = filter->next()) {
        r.rows += batch->size();
        r.sum += sum_selected(*batch, sum_pos);
    }
    return r;
}

// SELECT device_id, sum(value), count(*) FROM readings GROUP BY device_id
QueryResult hashagg(const Table& t) {
    auto        scan = std::make_unique<ScanOperator>(t, std::nullopt);
    auto        agg  = std::make_unique<HashAggOperator>(std::move(scan), 0, 2);
    QueryResult r;
    while (auto batch = agg->next()) {
        r.count += batch->size();
        fold_agg_batch(*batch, r.sum, r.rows);
    }
    return r;
}

QueryResult hashagg_parallel(const Table& t, std::size_t workers) {
    AggHashTable ht = parallel_group_sum_count(t, 0, 2, workers);
    QueryResult  r;
    r.count = static_cast<std::int64_t>(ht.size());
    ht.for_each([&](std::int64_t, const AggHashTable::Value& v) {
        r.sum += v.sum;
        r.rows += v.count;
    });
    return r;
}

void run_query(const char* name, int iters, const std::function<QueryResult()>& q) {
    std::vector<double> us;
    us.reserve(iters);
    QueryResult   result;
    AllocSnapshot a0;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        result  = q();
        auto t1 = Clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    AllocSnapshot a1;
    std::sort(us.begin(), us.end());
    std::printf("%-22s best %9.0f us  median %9.0f us  allocs/iter %8llu  alloc MiB/iter %8.2f"
                "  | rows=%lld count=%lld sum=%.17g\n",
                name, us.front(), us[us.size() / 2],
                static_cast<unsigned long long>((a1.count - a0.count) / iters),
                mib((a1.bytes - a0.bytes) / iters), static_cast<long long>(result.rows),
                static_cast<long long>(result.count), result.sum);
}

}  // namespace

int main(int argc, char** argv) {
    std::string csv   = "data/readings.csv";
    int         iters = 7;
    bool        narrow_float = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc) {
            csv = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--narrow-float") {
            narrow_float = true;
        } else {
            std::fprintf(stderr, "usage: wb_bench [--csv path] [--iters n] [--narrow-float]\n");
            return 1;
        }
    }

    Table t{"readings",
            {{"device_id", DataType::Int64},
             {"ts", DataType::Int64},
             {"value", DataType::Float64, narrow_float}},
            /*sort_key_col=*/1};

    // --- Load -------------------------------------------------------------
    const std::size_t rss0 = current_rss_bytes();
    AllocSnapshot     a0;
    auto              t0 = Clock::now();
    load_csv(t, csv, 65536);
    auto          t1 = Clock::now();
    AllocSnapshot a1;
    const std::size_t rss1 = current_rss_bytes();

    std::size_t rows = 0, col_bytes[3] = {0, 0, 0};
    for (const auto& seg : t.segments()) {
        rows += seg->num_rows();
        for (std::size_t c = 0; c < 3; ++c)
            col_bytes[c] += seg->column(c).memory_bytes();
    }
    const std::size_t total_col = col_bytes[0] + col_bytes[1] + col_bytes[2];

    std::printf("== load\n");
    std::printf("rows %zu  segments %zu  load %.0f ms  allocs %llu (%.2f/row)  alloc MiB %.1f\n",
                rows, t.num_segments(),
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                static_cast<unsigned long long>(a1.count - a0.count),
                static_cast<double>(a1.count - a0.count) / static_cast<double>(rows),
                mib(a1.bytes - a0.bytes));
    std::printf("== memory\n");
    std::printf("columns  device_id %.1f MiB  ts %.1f MiB  value %.1f MiB  total %.1f MiB "
                "(%.2f B/row)\n",
                mib(col_bytes[0]), mib(col_bytes[1]), mib(col_bytes[2]), mib(total_col),
                static_cast<double>(total_col) / static_cast<double>(rows));
    const Segment& seg0 = t.segment(0);
    std::printf("encoding seg0 widths  device_id %zu B  ts %zu B  value %s\n",
                static_cast<const Int64Vector&>(seg0.column(0)).width_bytes(),
                static_cast<const Int64Vector&>(seg0.column(1)).width_bytes(),
                static_cast<const Float64Vector&>(seg0.column(2)).narrow() ? "f32" : "f64");
    std::printf("process  footprint after load %.1f MiB (delta %.1f MiB)  peak RSS %.1f MiB\n",
                mib(rss1), mib(rss1 - rss0), mib(peak_rss_bytes()));

    // --- Queries ----------------------------------------------------------
    // ts cutoffs from the zone maps: mid-range (~50% selected, ~half the
    // segments pruned) and min (~100% selected, nothing pruned).
    std::int64_t ts_min = t.segment(0).zone(1).min_i64();
    std::int64_t ts_max = t.segment(0).zone(1).max_i64();
    for (const auto& seg : t.segments()) {
        ts_min = std::min(ts_min, seg->zone(1).min_i64());
        ts_max = std::max(ts_max, seg->zone(1).max_i64());
    }
    const std::int64_t ts_mid  = ts_min + (ts_max - ts_min) / 2;
    const std::size_t  workers = std::thread::hardware_concurrency();

    std::printf("== queries (iters=%d, workers=%zu)\n", iters, workers);
    run_query("ts>mid sum(value)", iters,
              [&] { return filter_sum(t, 1, CompareOp::Gt, ts_mid); });
    run_query("ts>=min sum(value)", iters,
              [&] { return filter_sum(t, 1, CompareOp::Ge, ts_min); });
    run_query("value>50 sum(value)", iters,
              [&] { return filter_sum(t, 2, CompareOp::Gt, 50.0); });
    run_query("groupby device 1T", iters, [&] { return hashagg(t); });
    run_query("groupby device MT", iters, [&] { return hashagg_parallel(t, workers); });
    return 0;
}
