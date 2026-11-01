#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "dist/cluster.h"
#include "dist/dist_join.h"
#include "dist/distribute.h"
#include "dist/exchange.h"
#include "dist/planner.h"
#include "dist/stream.h"
#include "exec/filter.h"
#include "exec/hash_join.h"
#include "exec/scan.h"
#include "test_framework.h"
#include "test_util.h"

using namespace wb;
using namespace wbtest;

namespace {

// Slice s holds keys [s*1000, s*1000 + 999], value = 2 * key.
void fill_parts_with_ranges(Cluster& cluster, const std::string& name) {
    cluster.declare_table(name, key_value_schema(), {DistStyle::Even, 0});
    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        KeyValueRows rows;
        for (int64_t key = slice * 1000; key < static_cast<int64_t>(slice * 1000 + 1000); ++key)
            rows.emplace_back(key, 2 * key);
        append_segments(cluster.local_part(slice, name), rows, 300);
    }
}

std::size_t source_slice_of(int64_t key) { return static_cast<std::size_t>(key / 1000); }

using JoinRow = std::tuple<int64_t, int64_t, int64_t, int64_t>;

std::vector<JoinRow> collect_join(Operator& join) {
    std::vector<JoinRow> rows;
    while (auto batch = join.next()) {
        CHECK_EQ(batch->num_columns(), std::size_t{4});
        auto col = [&](std::size_t c) -> const Int64Vector& {
            return static_cast<const Int64Vector&>(batch->column(c));
        };
        const auto &a = col(0), &b = col(1), &c = col(2), &d = col(3);
        for_each_live_row(*batch, [&](std::uint32_t row) {
            rows.emplace_back(a.value(row), b.value(row), c.value(row), d.value(row));
        });
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

std::vector<JoinRow> nested_loop_join(const KeyValueRows& probe, const KeyValueRows& build) {
    std::vector<JoinRow> rows;
    for (const auto& [pk, pv] : probe)
        for (const auto& [bk, bv] : build)
            if (pk == bk) rows.emplace_back(pk, pv, bk, bv);
    std::sort(rows.begin(), rows.end());
    return rows;
}

KeyValueRows rows_of_all_parts(Cluster& cluster, const std::string& name) {
    KeyValueRows rows;
    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        if (cluster.distribution(name).style == DistStyle::All &&
            slice != cluster.first_slice_of(cluster.node_of(slice)))
            continue;
        auto part = collect(cluster.local_part(slice, name));
        rows.insert(rows.end(), part.begin(), part.end());
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

KeyValueRows sorted(KeyValueRows rows) {
    std::sort(rows.begin(), rows.end());
    return rows;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Infrastructure (should pass now)
// ---------------------------------------------------------------------------------------------

TEST(network_counts_same_node_traffic_as_local) {
    NetworkStats stats;
    stats.record(0, 0, 100, 10);
    stats.record(0, 1, 40, 4);
    stats.record(1, 0, 60, 6);
    const auto traffic = stats.snapshot();
    CHECK_EQ(traffic.local_bytes, std::uint64_t{100});
    CHECK_EQ(traffic.local_rows, std::uint64_t{10});
    CHECK_EQ(traffic.cross_node_bytes, std::uint64_t{100});
    CHECK_EQ(traffic.cross_node_rows, std::uint64_t{10});
}

TEST(slice_for_key_spreads_sequential_keys_evenly) {
    constexpr std::size_t kSlices = 8;
    constexpr int64_t     kKeys   = 80'000;
    std::vector<int>      per_slice(kSlices, 0);
    for (int64_t key = 0; key < kKeys; ++key) {
        const std::size_t slice = slice_for_key(key, kSlices);
        CHECK(slice < kSlices);
        CHECK_EQ(slice, slice_for_key(key, kSlices));
        ++per_slice[slice];
    }
    for (int count : per_slice) {
        CHECK(count > 9'000);
        CHECK(count < 11'000);
    }
}

TEST(all_tables_live_on_the_first_slice_of_each_node) {
    Cluster cluster{{.nodes = 2, .slices_per_node = 2}};
    cluster.declare_table("dim", key_value_schema(), {DistStyle::All, 0});

    bool rejected = false;
    try {
        cluster.local_part(1, "dim");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
    CHECK(&cluster.scan_target(1, "dim") == &cluster.local_part(0, "dim"));
    CHECK(&cluster.scan_target(3, "dim") == &cluster.local_part(2, "dim"));
}

TEST(redistribute_routes_every_row_to_its_key_slice) {
    Cluster cluster{{.nodes = 2, .slices_per_node = 2}};
    fill_parts_with_ranges(cluster, "t");

    Exchange exchange{cluster};
    run_stream(cluster.num_slices(), [&](std::size_t slice) {
        ScanOperator scan{cluster.scan_target(slice, "t"), std::nullopt};
        exchange.redistribute(scan, slice, 0);
    });

    std::uint64_t received = 0, expected_cross_rows = 0;
    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        auto source = exchange.receive(slice);
        for (const auto& [key, value] : collect(*source)) {
            CHECK_EQ(slice_for_key(key, cluster.num_slices()), slice);
            CHECK_EQ(value, 2 * key);
            if (cluster.node_of(source_slice_of(key)) != cluster.node_of(slice))
                ++expected_cross_rows;
            ++received;
        }
    }
    CHECK_EQ(received, std::uint64_t{4000});

    const auto traffic = cluster.network().snapshot();
    CHECK_EQ(traffic.cross_node_rows, expected_cross_rows);
    CHECK_EQ(traffic.cross_node_rows + traffic.local_rows, std::uint64_t{4000});
    CHECK_EQ(traffic.cross_node_bytes, expected_cross_rows * 16);
}

TEST(broadcast_crosses_the_wire_once_per_remote_node) {
    Cluster cluster{{.nodes = 3, .slices_per_node = 2}};
    fill_parts_with_ranges(cluster, "t");
    const std::uint64_t total_rows = 6000;

    Exchange exchange{cluster};
    run_stream(cluster.num_slices(), [&](std::size_t slice) {
        ScanOperator scan{cluster.scan_target(slice, "t"), std::nullopt};
        exchange.broadcast(scan, slice);
    });

    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        auto source = exchange.receive(slice);
        CHECK_EQ(collect(*source).size(), total_rows);
    }
    const auto traffic = cluster.network().snapshot();
    CHECK_EQ(traffic.cross_node_rows, total_rows * (cluster.num_nodes() - 1));
    CHECK_EQ(traffic.local_rows, total_rows);
}

TEST(exchange_moves_only_rows_selected_upstream) {
    Cluster cluster{{.nodes = 2, .slices_per_node = 1}};
    fill_parts_with_ranges(cluster, "t");

    Exchange exchange{cluster};
    run_stream(cluster.num_slices(), [&](std::size_t slice) {
        auto scan = std::make_unique<ScanOperator>(cluster.scan_target(slice, "t"), std::nullopt);
        FilterOperator filter{std::move(scan), Predicate{0, CompareOp::Ge, int64_t{1500}}};
        exchange.redistribute(filter, slice, 0);
    });

    std::uint64_t received = 0;
    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        auto source = exchange.receive(slice);
        for (const auto& [key, value] : collect(*source)) {
            CHECK(key >= 1500);
            ++received;
        }
    }
    CHECK_EQ(received, std::uint64_t{500});
}

TEST(run_stream_times_every_slice_and_rethrows_failures) {
    const auto timing = run_stream(4, [](std::size_t) {});
    CHECK_EQ(timing.slice_ms.size(), std::size_t{4});

    bool rethrown = false;
    try {
        run_stream(4, [](std::size_t slice) {
            if (slice == 2) throw std::runtime_error("slice 2 failed");
        });
    } catch (const std::runtime_error&) {
        rethrown = true;
    }
    CHECK(rethrown);
}

// ---------------------------------------------------------------------------------------------
// Task 1: distribute_table
// ---------------------------------------------------------------------------------------------

namespace {
KeyValueRows sample_rows(int64_t count, int64_t distinct_keys) {
    KeyValueRows rows;
    for (int64_t i = 0; i < count; ++i)
        rows.emplace_back((i * 7919) % distinct_keys, i);
    return rows;
}
}  // namespace

TEST(distribute_key_places_every_row_on_its_key_slice) {
    Cluster cluster{{.nodes = 2, .slices_per_node = 3}};
    const auto source_rows = sample_rows(10'000, 997);
    auto       source      = make_table("src", source_rows);

    distribute_table(cluster, *source, "t", {DistStyle::Key, 0}, 512);

    CHECK(cluster.distribution("t").style == DistStyle::Key);
    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        for (const auto& [key, value] : collect(cluster.local_part(slice, "t")))
            CHECK_EQ(slice_for_key(key, cluster.num_slices()), slice);
    }
    CHECK(rows_of_all_parts(cluster, "t") == sorted(source_rows));
}

TEST(distribute_even_balances_parts_within_one_row) {
    Cluster cluster{{.nodes = 2, .slices_per_node = 3}};
    const auto source_rows = sample_rows(10'001, 50);
    auto       source      = make_table("src", source_rows);

    distribute_table(cluster, *source, "t", {DistStyle::Even, 0}, 512);

    std::size_t smallest = SIZE_MAX, largest = 0;
    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        const std::size_t rows = collect(cluster.local_part(slice, "t")).size();
        smallest               = std::min(smallest, rows);
        largest                = std::max(largest, rows);
    }
    CHECK(largest - smallest <= 1);
    CHECK(rows_of_all_parts(cluster, "t") == sorted(source_rows));
}

TEST(distribute_all_stores_one_full_copy_per_node) {
    Cluster cluster{{.nodes = 3, .slices_per_node = 2}};
    const auto source_rows = sample_rows(2'000, 2'000);
    auto       source      = make_table("src", source_rows);

    distribute_table(cluster, *source, "t", {DistStyle::All, 0}, 512);

    for (std::size_t node = 0; node < cluster.num_nodes(); ++node) {
        const Table& copy = cluster.local_part(cluster.first_slice_of(node), "t");
        CHECK(sorted(collect(copy)) == sorted(source_rows));
    }
    CHECK_EQ(cluster.row_count("t"), std::uint64_t{2'000});
}

TEST(distribute_cuts_parts_into_segments_of_at_most_rows_per_segment) {
    Cluster cluster{{.nodes = 1, .slices_per_node = 2}};
    auto    source = make_table("src", sample_rows(5'000, 100));

    distribute_table(cluster, *source, "t", {DistStyle::Even, 0}, 300);

    for (std::size_t slice = 0; slice < cluster.num_slices(); ++slice) {
        const Table& part = cluster.local_part(slice, "t");
        CHECK(part.num_segments() >= 2);
        for (const auto& segment : part.segments())
            CHECK(segment->num_rows() <= 300);
    }
}

// ---------------------------------------------------------------------------------------------
// Task 2: HashJoinOperator
// ---------------------------------------------------------------------------------------------

TEST(hash_join_matches_a_nested_loop_reference) {
    // Duplicate keys on both sides, keys present on only one side, several segments per side.
    KeyValueRows probe_rows, build_rows;
    for (int64_t i = 0; i < 3'000; ++i)
        probe_rows.emplace_back(i % 400, i);
    for (int64_t i = 0; i < 900; ++i)
        build_rows.emplace_back(200 + i % 300, -i);

    auto probe_table = make_table("probe", probe_rows, 700);
    auto build_table = make_table("build", build_rows, 250);

    HashJoinOperator join{std::make_unique<ScanOperator>(*probe_table, std::nullopt),
                          std::make_unique<ScanOperator>(*build_table, std::nullopt), 0, 0};
    const auto actual = collect_join(join);
    CHECK_EQ(actual.size(), nested_loop_join(probe_rows, build_rows).size());
    CHECK(actual == nested_loop_join(probe_rows, build_rows));
}

TEST(hash_join_respects_a_selection_vector_on_the_probe_side) {
    KeyValueRows probe_rows, build_rows;
    for (int64_t i = 0; i < 2'000; ++i)
        probe_rows.emplace_back(i % 100, i);
    for (int64_t key = 0; key < 100; ++key)
        build_rows.emplace_back(key, key * 10);

    auto probe_table = make_table("probe", probe_rows);
    auto build_table = make_table("build", build_rows);

    auto probe = std::make_unique<FilterOperator>(
        std::make_unique<ScanOperator>(*probe_table, std::nullopt),
        Predicate{1, CompareOp::Lt, int64_t{500}});
    HashJoinOperator join{std::move(probe), std::make_unique<ScanOperator>(*build_table, std::nullopt),
                          0, 0};

    KeyValueRows filtered;
    for (const auto& row : probe_rows)
        if (row.second < 500) filtered.push_back(row);
    CHECK(collect_join(join) == nested_loop_join(filtered, build_rows));
}

TEST(hash_join_with_an_empty_build_side_emits_nothing) {
    auto probe_table = make_table("probe", sample_rows(1'000, 10));
    auto build_table = make_table("build", {});
    HashJoinOperator join{std::make_unique<ScanOperator>(*probe_table, std::nullopt),
                          std::make_unique<ScanOperator>(*build_table, std::nullopt), 0, 0};
    CHECK(collect_join(join).empty());
}

// ---------------------------------------------------------------------------------------------
// Task 3: planner
// ---------------------------------------------------------------------------------------------

namespace {
JoinSide side(DistStyle style, std::size_t dist_key_col, std::uint64_t rows) {
    return {{style, dist_key_col}, /*join_key_col=*/0, rows};
}
}  // namespace

TEST(planner_validity_follows_what_each_slice_holds) {
    const auto key_on_join  = side(DistStyle::Key, 0, 1000);
    const auto key_on_other = side(DistStyle::Key, 1, 1000);
    const auto even         = side(DistStyle::Even, 0, 1000);
    const auto all          = side(DistStyle::All, 0, 1000);

    CHECK(is_valid_strategy(JoinDist::DistNone, key_on_join, key_on_join));
    CHECK(!is_valid_strategy(JoinDist::DistNone, key_on_other, key_on_join));
    CHECK(!is_valid_strategy(JoinDist::DistNone, even, key_on_join));

    CHECK(is_valid_strategy(JoinDist::DistAllNone, even, all));
    CHECK(!is_valid_strategy(JoinDist::DistAllNone, even, even));

    CHECK(is_valid_strategy(JoinDist::DistInner, key_on_join, even));
    CHECK(!is_valid_strategy(JoinDist::DistInner, key_on_other, even));

    CHECK(is_valid_strategy(JoinDist::DistOuter, even, key_on_join));
    CHECK(!is_valid_strategy(JoinDist::DistOuter, even, key_on_other));

    CHECK(is_valid_strategy(JoinDist::BcastInner, even, even));
    CHECK(is_valid_strategy(JoinDist::DistBoth, even, even));

    // Moving an ALL table would duplicate it: every slice scans its node's full copy.
    CHECK(!is_valid_strategy(JoinDist::BcastInner, even, all));
    CHECK(!is_valid_strategy(JoinDist::DistBoth, even, all));
    CHECK(!is_valid_strategy(JoinDist::DistInner, key_on_join, all));
}

TEST(planner_picks_the_cheapest_valid_strategy) {
    auto pick = [](JoinSide outer, JoinSide inner, std::size_t nodes) {
        return std::string{join_dist_label(choose_join_strategy(outer, inner, nodes))};
    };
    using std::string;

    CHECK_EQ(pick(side(DistStyle::Key, 0, 1'000'000), side(DistStyle::Key, 0, 1'000), 4),
             string{"DS_DIST_NONE"});
    CHECK_EQ(pick(side(DistStyle::Even, 0, 1'000'000), side(DistStyle::All, 0, 1'000), 4),
             string{"DS_DIST_ALL_NONE"});
    // Even a tiny inner is cheaper to redistribute than to broadcast when outer is in place.
    CHECK_EQ(pick(side(DistStyle::Key, 0, 1'000'000), side(DistStyle::Even, 0, 10), 4),
             string{"DS_DIST_INNER"});
    // Outer would move 500k rows; broadcasting a 1k inner moves 1k.
    CHECK_EQ(pick(side(DistStyle::Even, 0, 1'000'000), side(DistStyle::Key, 0, 1'000), 2),
             string{"DS_BCAST_INNER"});
    // Outer moves 500k; broadcasting a 600k inner moves 600k.
    CHECK_EQ(pick(side(DistStyle::Even, 0, 1'000'000), side(DistStyle::Key, 0, 600'000), 2),
             string{"DS_DIST_OUTER"});
    // KEY on the wrong column is as good as EVEN.
    CHECK_EQ(pick(side(DistStyle::Key, 1, 1'000'000), side(DistStyle::Key, 0, 600'000), 2),
             string{"DS_DIST_OUTER"});
    // Broadcast 100k to 3 remote nodes (300k) vs redistribute 1.1M * 3/4 (825k).
    CHECK_EQ(pick(side(DistStyle::Even, 0, 1'000'000), side(DistStyle::Even, 0, 100'000), 4),
             string{"DS_BCAST_INNER"});
    // Broadcast 500k (1.5M) vs redistribute 1.5M * 3/4 (1.125M).
    CHECK_EQ(pick(side(DistStyle::Even, 0, 1'000'000), side(DistStyle::Even, 0, 500'000), 4),
             string{"DS_DIST_BOTH"});
}

// ---------------------------------------------------------------------------------------------
// All three together
// ---------------------------------------------------------------------------------------------

TEST(every_valid_strategy_computes_the_same_join) {
    Cluster cluster{{.nodes = 2, .slices_per_node = 2}};

    KeyValueRows orders, customers;
    for (int64_t i = 0; i < 20'000; ++i)
        orders.emplace_back((i * 31) % 1'500, i);  // some customer ids never ordered
    for (int64_t id = 0; id < 1'200; ++id)
        customers.emplace_back(id, id % 7);  // some orders have no customer

    auto orders_source    = make_table("orders", orders, 4'096);
    auto customers_source = make_table("customers", customers, 4'096);
    distribute_table(cluster, *orders_source, "orders_key", {DistStyle::Key, 0});
    distribute_table(cluster, *orders_source, "orders_even", {DistStyle::Even, 0});
    distribute_table(cluster, *customers_source, "customers_key", {DistStyle::Key, 0});
    distribute_table(cluster, *customers_source, "customers_even", {DistStyle::Even, 0});
    distribute_table(cluster, *customers_source, "customers_all", {DistStyle::All, 0});

    std::uint64_t expected_rows = 0, expected_checksum = 0;
    for (const auto& [ok, ov, ck, cv] : nested_loop_join(orders, customers)) {
        ++expected_rows;
        expected_checksum += static_cast<std::uint64_t>(ok) + static_cast<std::uint64_t>(ov) +
                             static_cast<std::uint64_t>(ck) + static_cast<std::uint64_t>(cv);
    }

    const JoinDist all_strategies[] = {JoinDist::DistNone,  JoinDist::DistAllNone,
                                       JoinDist::DistInner, JoinDist::DistOuter,
                                       JoinDist::BcastInner, JoinDist::DistBoth};
    int runs = 0;
    for (const char* outer : {"orders_key", "orders_even"}) {
        for (const char* inner : {"customers_key", "customers_even", "customers_all"}) {
            const JoinQuery query{outer, 0, inner, 0};
            for (JoinDist strategy : all_strategies) {
                if (!is_valid_strategy(strategy, outer_side(cluster, query),
                                       inner_side(cluster, query)))
                    continue;
                const auto result = run_distributed_join(cluster, query, strategy);
                CHECK_EQ(result.output_rows, expected_rows);
                CHECK_EQ(result.checksum, expected_checksum);
                if (strategy == JoinDist::DistNone || strategy == JoinDist::DistAllNone)
                    CHECK_EQ(result.traffic.cross_node_bytes, std::uint64_t{0});
                ++runs;
            }
        }
    }
    CHECK(runs >= 10);
}

TEST(hash_join_output_columns_are_reused) {
    // 20,000 output rows (one match per probe row) span several output batches; every batch must
    // be served from the same operator-owned columns.
    KeyValueRows probe_rows, build_rows;
    for (int64_t i = 0; i < 20'000; ++i)
        probe_rows.emplace_back(i % 100, i);
    for (int64_t key = 0; key < 100; ++key)
        build_rows.emplace_back(key, -key);
    auto probe_table = make_table("probe", probe_rows, 3'000);
    auto build_table = make_table("build", build_rows);

    {
        HashJoinOperator join{std::make_unique<ScanOperator>(*probe_table, std::nullopt),
                              std::make_unique<ScanOperator>(*build_table, std::nullopt), 0, 0};
        std::vector<std::vector<const Vector*>> batches;
        while (auto batch = join.next()) {
            std::vector<const Vector*> cols;
            for (std::size_t c = 0; c < batch->num_columns(); ++c)
                cols.push_back(&batch->column(c));
            batches.push_back(cols);
        }
        CHECK(batches.size() >= 2);
        for (const auto& cols : batches)
            CHECK(cols == batches.front());
    }

    HashJoinOperator join{std::make_unique<ScanOperator>(*probe_table, std::nullopt),
                          std::make_unique<ScanOperator>(*build_table, std::nullopt), 0, 0};
    CHECK(collect_join(join) == nested_loop_join(probe_rows, build_rows));
}
