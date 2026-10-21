// wb_dist: the Weekend 1 experiments. Needs tasks 1-3 implemented (see src/dist/*.h).
//
//   ./wb_dist [--nodes N] [--slices S] [--orders ROWS] [--customers ROWS] [--skew EXP]
//             [--experiment all|strategies|skew|crossover]

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dist/cluster.h"
#include "dist/dist_join.h"
#include "dist/distribute.h"
#include "dist/planner.h"
#include "storage/segment.h"
#include "storage/table.h"
#include "todo.h"
#include "tools/datagen.h"

using namespace wb;

namespace {

struct Options {
    ClusterConfig cluster{.nodes = 4, .slices_per_node = 2};
    std::int64_t  orders        = 2'000'000;
    std::int64_t  customers     = 100'000;
    double        skew_exponent = 8.0;  // >1 concentrates orders on low customer ids
    std::string   experiment    = "all";
};

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i], value = argv[i + 1];
        if (flag == "--nodes") options.cluster.nodes = std::stoul(value);
        else if (flag == "--slices") options.cluster.slices_per_node = std::stoul(value);
        else if (flag == "--orders") options.orders = std::stoll(value);
        else if (flag == "--customers") options.customers = std::stoll(value);
        else if (flag == "--skew") options.skew_exponent = std::stod(value);
        else if (flag == "--experiment") options.experiment = value;
        else throw std::invalid_argument("unknown flag " + flag);
    }
    return options;
}

double mb(std::uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

void print_header() {
    std::printf("%-12s %-15s %-17s %10s %10s %9s %9s %9s %10s\n", "outer", "inner", "strategy",
                "cross MB", "local MB", "xchg ms", "join max", "join min", "out rows");
}

void print_run(const std::string& outer, const std::string& inner, const JoinRunResult& run,
               bool planner_choice) {
    const auto& join     = run.streams.back().timing;
    const double exchange = run.streams.size() > 1 ? run.streams.front().timing.wall_ms : 0.0;
    std::printf("%-12s %-15s %-16s%s %10.1f %10.1f %9.1f %9.1f %9.1f %10llu\n", outer.c_str(),
                inner.c_str(), join_dist_label(run.strategy), planner_choice ? "*" : " ",
                mb(run.traffic.cross_node_bytes), mb(run.traffic.local_bytes), exchange,
                join.slowest_slice_ms(), join.fastest_slice_ms(),
                static_cast<unsigned long long>(run.output_rows));
}

// Experiment 1: the same join under every strategy the data layout allows.
void strategies(const Options& options) {
    std::printf("\n== Experiment 1: one join, every strategy ==\n");
    std::printf("%zu nodes x %zu slices, %lld orders, %lld customers. * = planner's choice\n\n",
                options.cluster.nodes, options.cluster.slices_per_node,
                static_cast<long long>(options.orders), static_cast<long long>(options.customers));

    Cluster cluster{options.cluster};
    auto    orders    = make_orders(options.orders, options.customers, 1.0);
    auto    customers = make_customers(options.customers);
    distribute_table(cluster, *orders, "orders_key", {DistStyle::Key, 0});
    distribute_table(cluster, *orders, "orders_even", {DistStyle::Even, 0});
    distribute_table(cluster, *customers, "customers_key", {DistStyle::Key, 0});
    distribute_table(cluster, *customers, "customers_even", {DistStyle::Even, 0});
    distribute_table(cluster, *customers, "customers_all", {DistStyle::All, 0});

    print_header();
    std::uint64_t first_checksum = 0;
    bool          checksums_agree = true;
    for (const char* outer : {"orders_key", "orders_even"}) {
        for (const char* inner : {"customers_key", "customers_even", "customers_all"}) {
            const JoinQuery query{outer, 0, inner, 0};
            const JoinSide  outer_info = outer_side(cluster, query);
            const JoinSide  inner_info = inner_side(cluster, query);
            const JoinDist  chosen = choose_join_strategy(outer_info, inner_info, cluster.num_nodes());
            for (JoinDist strategy : {JoinDist::DistNone, JoinDist::DistAllNone, JoinDist::DistInner,
                                      JoinDist::DistOuter, JoinDist::BcastInner, JoinDist::DistBoth}) {
                if (!is_valid_strategy(strategy, outer_info, inner_info)) continue;
                const auto run = run_distributed_join(cluster, query, strategy);
                print_run(outer, inner, run, strategy == chosen);
                if (first_checksum == 0) first_checksum = run.checksum;
                checksums_agree = checksums_agree && run.checksum == first_checksum;
            }
        }
    }
    std::printf("\nall strategies returned the same result: %s\n", checksums_agree ? "yes" : "NO");
}

// Experiment 2: KEY distribution on a skewed column. No data moves (DS_DIST_ALL_NONE), so any
// difference between slices is purely how many rows each one owns.
void skew(const Options& options) {
    std::printf("\n== Experiment 2: skew (exponent %.1f) ==\n", options.skew_exponent);
    Cluster cluster{options.cluster};
    auto    orders    = make_orders(options.orders, options.customers, options.skew_exponent);
    auto    customers = make_customers(options.customers);
    distribute_table(cluster, *orders, "orders_key", {DistStyle::Key, 0});
    distribute_table(cluster, *orders, "orders_even", {DistStyle::Even, 0});
    distribute_table(cluster, *customers, "customers_all", {DistStyle::All, 0});

    for (const char* outer : {"orders_key", "orders_even"}) {
        const auto  run  = run_distributed_join(cluster, {outer, 0, "customers_all", 0},
                                                JoinDist::DistAllNone);
        const auto& join = run.streams.back().timing;
        std::printf("\n%s: join wall %.1f ms (slowest slice %.1f, fastest %.1f)\n", outer,
                    join.wall_ms, join.slowest_slice_ms(), join.fastest_slice_ms());
        for (std::size_t slice = 0; slice < run.rows_per_slice.size(); ++slice) {
            std::printf("  slice %2zu: %9llu rows  %7.1f ms\n", slice,
                        static_cast<unsigned long long>(run.rows_per_slice[slice]),
                        join.slice_ms[slice]);
        }
    }
}

// Experiment 3: broadcast vs redistribute-both as the inner side grows. Rows on both sides are
// 16 bytes, so the predicted crossover in bytes is the same as in rows: |B| = |A| / (N-1).
void crossover(const Options& options) {
    const std::size_t nodes     = options.cluster.nodes;
    const double      predicted = static_cast<double>(options.orders) / (nodes - 1);
    std::printf("\n== Experiment 3: broadcast vs redistribute ==\n");
    std::printf("predicted crossover at |customers| = |orders|/(N-1) = %.0f\n\n", predicted);
    std::printf("%12s %8s %16s %16s %12s %12s\n", "customers", "x pred", "BCAST cross MB",
                "DIST_BOTH cross MB", "BCAST ms", "BOTH ms");

    auto orders = make_orders(options.orders, options.customers, 1.0);
    for (double fraction : {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}) {
        const auto inner_rows = static_cast<std::int64_t>(predicted * fraction);
        Cluster    cluster{options.cluster};
        auto       customers = make_customers(inner_rows);
        distribute_table(cluster, *orders, "orders", {DistStyle::Even, 0});
        distribute_table(cluster, *customers, "customers", {DistStyle::Even, 0});

        const JoinQuery query{"orders", 0, "customers", 0};
        const auto      bcast = run_distributed_join(cluster, query, JoinDist::BcastInner);
        const auto      both  = run_distributed_join(cluster, query, JoinDist::DistBoth);
        std::printf("%12lld %8.2f %16.1f %16.1f %12.1f %12.1f\n",
                    static_cast<long long>(inner_rows), fraction,
                    mb(bcast.traffic.cross_node_bytes), mb(both.traffic.cross_node_bytes),
                    bcast.wall_ms(), both.wall_ms());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const bool    all     = options.experiment == "all";
        if (all || options.experiment == "strategies") strategies(options);
        if (all || options.experiment == "skew") skew(options);
        if (all || options.experiment == "crossover") crossover(options);
    } catch (const NotImplemented& stub) {
        std::cerr << "not implemented yet: " << stub.what() << "\nrun `make test` to see which "
                  << "tasks remain.\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
