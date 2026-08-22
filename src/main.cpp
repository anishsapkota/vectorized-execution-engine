#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <thread>
#include <tuple>
#include <vector>

#include "exec/filter.h"
#include "exec/hashagg.h"
#include "exec/parallel_agg.h"
#include "exec/scan.h"
#include "storage/csv.h"
#include "storage/schema.h"
#include "storage/segment.h"
#include "storage/vector.h"
#include "storage/zone_map.h"

int main() {
    using namespace wb;

    // Int64Vector ints;
    // ints.append(10);
    // ints.append(20);
    // ints.append(30);

    // StringVector strings;
    // strings.append("alpha");
    // strings.append(std::string{"beta"});  // try both lvalue and rvalue

    // std::cout << "int column: type=" << type_name(ints.type()) << " size=" << ints.size() <<
    // "\n"; std::cout << "string column: type=" << type_name(strings.type()) << " size=" <<
    // strings.size()
    //           << "\n";

    // // Hold a concrete vector through the base pointer, like an operator would.
    // std::unique_ptr<Vector> col =
    //     std::make_unique<Float64Vector>(std::vector<double>{1.5, 2.5, 3.5});
    // std::cout << "polymorphic: type=" << type_name(col->type()) << " size=" << col->size() <<
    // "\n";

    // // Read through span — the kernel-style access.
    // auto    view = ints.data();
    // int64_t sum  = 0;
    // for (auto v : view)
    //     sum += v;
    // std::cout << "sum of int column: " << sum << "\n";

    // ZoneMap z;
    // z.observe(int64_t{10});
    // z.observe(int64_t{5});
    // z.observe(int64_t{20});
    // z.observe(int64_t{15});

    // std::cout << "zone: min=" << z.min_i64() << " max=" << z.max_i64() << "\n";

    // Schema schema = {{"id", DataType::Int64}, {"price", DataType::Float64}};

    // SegmentBuilder builder{schema};
    // builder.append_int64(0, 100);
    // builder.append_float64(1, 9.99);
    // builder.append_int64(0, 200);
    // builder.append_float64(1, 19.99);
    // builder.append_int64(0, 150);
    // builder.append_float64(1, 14.99);

    // auto seg = builder.seal();
    // std::cout << "segment: " << seg->num_rows() << " rows, " << seg->num_columns() << "
    // columns\n";

    // const auto& id_zone = seg->zone(0);
    // std::cout << "id zone: min=" << id_zone.min_i64() << " max=" << id_zone.max_i64() << "\n";

    // const auto& price_zone = seg->zone(1);
    // std::cout << "price zone: min=" << price_zone.min_f64() << " max=" << price_zone.max_f64()
    //           << "\n";

    // Table t{"events", {{"id", DataType::Int64}, {"value", DataType::Float64}}};

    // // Segment 0: ids 1..3
    // {
    //     SegmentBuilder b{t.schema()};
    //     b.append_int64(0, 1);
    //     b.append_float64(1, 10.0);
    //     b.append_int64(0, 2);
    //     b.append_float64(1, 20.0);
    //     b.append_int64(0, 3);
    //     b.append_float64(1, 30.0);
    //     t.add_segment(b.seal());
    // }
    // // Segment 1: ids 100..102
    // {
    //     SegmentBuilder b{t.schema()};
    //     b.append_int64(0, 100);
    //     b.append_float64(1, 1.0);
    //     b.append_int64(0, 101);
    //     b.append_float64(1, 2.0);
    //     b.append_int64(0, 102);
    //     b.append_float64(1, 3.0);
    //     t.add_segment(b.seal());
    // }
    // // Segment 2: ids 1000..1002
    // {
    //     SegmentBuilder b{t.schema()};
    //     b.append_int64(0, 1000);
    //     b.append_float64(1, 0.1);
    //     b.append_int64(0, 1001);
    //     b.append_float64(1, 0.2);
    //     b.append_int64(0, 1002);
    //     b.append_float64(1, 0.3);
    //     t.add_segment(b.seal());
    // }

    // // Scan #1: no predicate — should emit all 9 rows.
    // {
    //     ScanOperator scan{t, std::nullopt};
    //     std::size_t  count = 0;
    //     while (auto batch = scan.next())
    //         count += batch->size();
    //     std::cout << "no predicate: emitted " << count << " rows, scanned "
    //               << scan.segments_scanned() << " segments, pruned " << scan.segments_pruned()
    //               << "\n";
    // }

    // // Scan #2: id > 50 — should prune segment 0 only, emit 6 rows.
    // {
    //     Predicate    p{.col_idx = 0, .op = CompareOp::Gt, .value = int64_t{50}};
    //     ScanOperator scan{t, p};
    //     std::size_t  count = 0;
    //     while (auto batch = scan.next())
    //         count += batch->size();
    //     std::cout << "id > 50: emitted " << count << " rows, scanned " << scan.segments_scanned()
    //               << " segments, pruned " << scan.segments_pruned() << "\n";
    // }

    // // Scan #3: id > 500 AND id < 2000 conceptually — we only have AND-of-one for now.
    // // Use id > 500 — should prune segments 0 and 1, emit 3 rows.
    // {
    //     Predicate    p{.col_idx = 0, .op = CompareOp::Gt, .value = int64_t{500}};
    //     ScanOperator scan{t, p};
    //     std::size_t  count = 0;
    //     while (auto batch = scan.next())
    //         count += batch->size();
    //     std::cout << "id > 500: emitted " << count << " rows, scanned " <<
    //     scan.segments_scanned()
    //               << " segments, pruned " << scan.segments_pruned() << "\n";
    // }

    // // Test A: scan with no predicate, filter id > 50 -> should emit 6 rows (segs 1+2).
    // {
    //     auto           scan = std::make_unique<ScanOperator>(t, std::nullopt);
    //     Predicate      p{.col_idx = 0, .op = CompareOp::Gt, .value = int64_t{50}};
    //     FilterOperator filter{std::move(scan), p};
    //     std::size_t    count = 0;
    //     while (auto batch = filter.next())
    //         count += batch->size();
    //     std::cout << "scan(none) | filter(id > 50): emitted " << count << " rows\n";
    // }

    // // Test B: scan prunes with id > 500 (zone-map skips segs 0+1), filter id < 1002 -> 2 rows.
    // {
    //     Predicate      scan_pred{.col_idx = 0, .op = CompareOp::Gt, .value = int64_t{500}};
    //     auto           scan = std::make_unique<ScanOperator>(t, scan_pred);
    //     Predicate      filter_pred{.col_idx = 0, .op = CompareOp::Lt, .value = int64_t{1002}};
    //     FilterOperator filter{std::move(scan), filter_pred};
    //     std::size_t    count = 0;
    //     while (auto batch = filter.next())
    //         count += batch->size();
    //     std::cout << "scan(id>500) | filter(id<1002): emitted " << count << " rows\n";
    // }

    // {
    //     Table loaded{"loaded",
    //                  {
    //                      {"id", DataType::Int64},
    //                      {"price", DataType::Float64},
    //                      {"name", DataType::String},
    //                  }};

    //     load_csv(loaded, "data/test.csv");

    //     std::cout << "loaded: " << loaded.num_segments() << " segments\n";
    //     if (loaded.num_segments() > 0) {
    //         const auto& seg = loaded.segment(0);
    //         std::cout << "first segment: " << seg.num_rows() << " rows\n";
    //         std::cout << "  id zone:    [" << seg.zone(0).min_i64() << ", " <<
    //         seg.zone(0).max_i64()
    //                   << "]\n";
    //         std::cout << "  price zone: [" << seg.zone(1).min_f64() << ", " <<
    //         seg.zone(1).max_f64()
    //                   << "]\n";
    //         std::cout << "  name zone:  [" << seg.zone(2).min_str() << ", " <<
    //         seg.zone(2).max_str()
    //                   << "]\n";
    //     }
    // }

    // {
    //     Table t{"bigger",
    //             {
    //                 {"id", DataType::Int64},
    //                 {"price", DataType::Float64},
    //                 {"name", DataType::String},
    //             }};
    //     load_csv(t, "data/bigger.csv", /* rows_per_segment = */ 100);
    //     std::cout << "bigger: " << t.num_segments() << " segments\n";  // expect 3
    //     for (std::size_t i = 0; i < t.num_segments(); ++i) {
    //         const auto& seg = t.segment(i);
    //         std::cout << "  seg " << i << ": " << seg.num_rows() << " rows, "
    //                   << "id [" << seg.zone(0).min_i64() << ", " << seg.zone(0).max_i64() <<
    //                   "]\n";
    //     }
    // }

    {
        using namespace std::chrono;

        Table t{"readings",
                {
                    {"device_id", DataType::Int64},
                    {"ts", DataType::Int64},
                    {"value", DataType::Float64},
                },
                /*sort_key_col=*/1};  // ts

        auto t0 = steady_clock::now();
        load_csv(t, "data/readings.csv", /* rows_per_segment = */ 65536);
        auto t1 = steady_clock::now();

        auto ms = duration_cast<milliseconds>(t1 - t0).count();
        std::cout << "loaded " << t.num_segments() << " segments in " << ms << " ms\n";

        // Inspect the first and last segments.
        const auto& first = t.segment(0);
        const auto& last  = t.segment(t.num_segments() - 1);
        std::cout << "first seg: ts [" << first.zone(1).min_i64() << ", " << first.zone(1).max_i64()
                  << "]\n";
        std::cout << "last  seg: ts [" << last.zone(1).min_i64() << ", " << last.zone(1).max_i64()
                  << "]\n";

        {
            // SELECT sum(value) FROM readings WHERE ts > 1_705_000_000;
            // Projection: {1=ts, 2=value} -> batch positions {0, 1}.
            // Filter predicate uses batch col 0 (ts); sum uses batch col 1 (value).
            using namespace std::chrono;

            int64_t   cutoff = 1'710'000'000;
            Predicate scan_pred{.col_idx = 1, .op = CompareOp::Gt, .value = cutoff};
            Predicate filter_pred{.col_idx = 0, .op = CompareOp::Gt, .value = cutoff};

            constexpr int iters        = 5;
            int64_t       best_us      = std::numeric_limits<int64_t>::max();
            double        last_sum     = 0.0;
            int64_t       last_rows    = 0;
            std::size_t   last_scanned = 0, last_pruned = 0;

            for (int iter = 0; iter < iters; ++iter) {
                auto          scan     = std::make_unique<ScanOperator>(t, scan_pred, 4096,
                                                           std::vector<std::size_t>{1, 2});
                ScanOperator* scan_ptr = scan.get();
                auto filter = std::make_unique<FilterOperator>(std::move(scan), filter_pred);

                auto    t0   = steady_clock::now();
                double  sum  = 0.0;
                int64_t rows = 0;
                while (auto batch = filter->next()) {
                    // batch col 1 = original col 2 = value
                    const auto& vals = static_cast<const Float64Vector&>(batch->column(1)).data();
                    for (std::uint32_t i : *batch->sel()) {
                        sum += vals[i];
                        ++rows;
                    }
                }
                auto t1 = steady_clock::now();

                auto us = duration_cast<microseconds>(t1 - t0).count();
                if (us < best_us) {
                    best_us      = us;
                    last_sum     = sum;
                    last_rows    = rows;
                    last_scanned = scan_ptr->segments_scanned();
                    last_pruned  = scan_ptr->segments_pruned();
                }
            }

            std::cout << "filter+sum: best of " << iters << " = " << best_us << " us"
                      << "  (" << last_rows << " rows, sum=" << last_sum << ")"
                      << "  scanned=" << last_scanned << " pruned=" << last_pruned << "\n";
        }
        {
            // SELECT device_id, SUM(value), COUNT(*) FROM readings GROUP BY device_id.
            using namespace std::chrono;

            auto scan = std::make_unique<ScanOperator>(t, std::nullopt);
            auto agg  = std::make_unique<HashAggOperator>(std::move(scan), /*group=*/0,
                                                         /*agg=*/2);  // device_id, value

            constexpr int iters            = 5;
            int64_t       best_us          = std::numeric_limits<int64_t>::max();
            std::size_t   group_count_seen = 0;

            for (int iter = 0; iter < iters; ++iter) {
                auto s   = std::make_unique<ScanOperator>(t, std::nullopt);
                auto agg = std::make_unique<HashAggOperator>(std::move(s), 0, 2);

                auto        t0          = steady_clock::now();
                std::size_t groups      = 0;
                double      total_sum   = 0.0;
                int64_t     total_count = 0;
                while (auto batch = agg->next()) {
                    const auto&   sums = static_cast<const Float64Vector&>(batch->column(1)).data();
                    const auto&   counts = static_cast<const Int64Vector&>(batch->column(2)).data();
                    std::uint32_t end    = batch->offset() + batch->size();
                    for (std::uint32_t i = batch->offset(); i < end; ++i) {
                        total_sum += sums[i];
                        total_count += counts[i];
                        ++groups;
                    }
                }
                auto t1          = steady_clock::now();
                best_us          = std::min(best_us, duration_cast<microseconds>(t1 - t0).count());
                group_count_seen = groups;

                if (iter == 0) {
                    std::cout << "  (sanity: " << groups << " groups, "
                              << "total_sum=" << total_sum << ", "
                              << "total_count=" << total_count << ")\n";
                }
            }

            std::cout << "hashagg: best of " << iters << " runs: " << best_us << " us, "
                      << group_count_seen << " groups\n";
        }

        {
            // Same query as the unordered_map / flat-hash benchmarks, but parallel.
            using namespace std::chrono;

            constexpr int     iters   = 5;
            const std::size_t workers = std::thread::hardware_concurrency();

            int64_t     best_us     = std::numeric_limits<int64_t>::max();
            std::size_t groups_seen = 0;

            for (int iter = 0; iter < iters; ++iter) {
                auto         t0     = steady_clock::now();
                AggHashTable result = parallel_group_sum_count(t, /*group=*/0, /*agg=*/2, workers);
                auto         t1     = steady_clock::now();

                best_us     = std::min(best_us, duration_cast<microseconds>(t1 - t0).count());
                groups_seen = result.size();

                if (iter == 0) {
                    // Sanity check: same totals as the single-threaded version.
                    double  total_sum   = 0.0;
                    int64_t total_count = 0;
                    result.for_each([&](int64_t, const AggHashTable::Value& v) {
                        total_sum += v.sum;
                        total_count += v.count;
                    });
                    std::cout << "  (parallel sanity: " << result.size() << " groups, "
                              << "total_sum=" << total_sum << ", "
                              << "total_count=" << total_count << ")\n";
                }
            }

            std::cout << "hashagg (parallel, " << workers << " workers): best of " << iters
                      << " runs: " << best_us << " us, " << groups_seen << " groups\n";
        }
    }
}