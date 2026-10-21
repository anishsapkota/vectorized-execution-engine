#include "dist/dist_join.h"

#include <memory>
#include <stdexcept>

#include "dist/exchange.h"
#include "exec/hash_join.h"
#include "exec/scan.h"

namespace wb {

namespace {

std::unique_ptr<Operator> scan(const Cluster& cluster, std::size_t slice,
                               const std::string& table) {
    return std::make_unique<ScanOperator>(cluster.scan_target(slice, table), std::nullopt);
}

struct SliceOutput {
    std::uint64_t rows     = 0;
    std::uint64_t checksum = 0;
};

SliceOutput drain(Operator& join) {
    SliceOutput output;
    while (auto batch = join.next()) {
        output.rows += batch->size();
        for (std::size_t c = 0; c < batch->num_columns(); ++c) {
            if (batch->column(c).type() != DataType::Int64) continue;
            static_cast<const Int64Vector&>(batch->column(c)).visit([&](auto codes, int64_t base) {
                for_each_live_row(*batch, [&](std::uint32_t row) {
                    output.checksum += static_cast<std::uint64_t>(Int64Vector::decode(base, codes[row]));
                });
            });
        }
    }
    return output;
}

bool moves_outer(JoinDist strategy) {
    return strategy == JoinDist::DistOuter || strategy == JoinDist::DistBoth;
}

bool moves_inner(JoinDist strategy) {
    return strategy == JoinDist::DistInner || strategy == JoinDist::BcastInner ||
           strategy == JoinDist::DistBoth;
}

}  // namespace

double JoinRunResult::wall_ms() const {
    double total = 0.0;
    for (const auto& stream : streams)
        total += stream.timing.wall_ms;
    return total;
}

JoinSide outer_side(const Cluster& cluster, const JoinQuery& query) {
    return {cluster.distribution(query.outer_table), query.outer_key_col,
            cluster.row_count(query.outer_table)};
}

JoinSide inner_side(const Cluster& cluster, const JoinQuery& query) {
    return {cluster.distribution(query.inner_table), query.inner_key_col,
            cluster.row_count(query.inner_table)};
}

JoinRunResult run_distributed_join(Cluster& cluster, const JoinQuery& query, JoinDist strategy) {
    if (!is_valid_strategy(strategy, outer_side(cluster, query), inner_side(cluster, query))) {
        throw std::invalid_argument(std::string{join_dist_label(strategy)} + " cannot join " +
                                    query.outer_table + " with " + query.inner_table);
    }

    const std::size_t num_slices = cluster.num_slices();
    JoinRunResult     result;
    result.strategy = strategy;
    cluster.network().reset();

    // Stream 1 (only if something moves): every slice sends its rows to their join slices.
    Exchange outer_exchange{cluster};
    Exchange inner_exchange{cluster};
    if (moves_outer(strategy) || moves_inner(strategy)) {
        auto timing = run_stream(num_slices, [&](std::size_t slice) {
            if (moves_outer(strategy)) {
                auto outer = scan(cluster, slice, query.outer_table);
                outer_exchange.redistribute(*outer, slice, query.outer_key_col);
            }
            if (moves_inner(strategy)) {
                auto inner = scan(cluster, slice, query.inner_table);
                if (strategy == JoinDist::BcastInner) {
                    inner_exchange.broadcast(*inner, slice);
                } else {
                    inner_exchange.redistribute(*inner, slice, query.inner_key_col);
                }
            }
        });
        result.streams.push_back({"exchange", std::move(timing)});
    }

    // Stream 2: every slice joins what it now holds.
    std::vector<SliceOutput> outputs(num_slices);
    auto                     timing = run_stream(num_slices, [&](std::size_t slice) {
        auto probe = moves_outer(strategy) ? outer_exchange.receive(slice)
                                                               : scan(cluster, slice, query.outer_table);
        auto build = moves_inner(strategy) ? inner_exchange.receive(slice)
                                                               : scan(cluster, slice, query.inner_table);
        HashJoinOperator join{std::move(probe), std::move(build), query.outer_key_col,
                              query.inner_key_col};
        outputs[slice] = drain(join);
    });
    result.streams.push_back({"join", std::move(timing)});

    result.traffic = cluster.network().snapshot();
    for (const auto& output : outputs) {
        result.rows_per_slice.push_back(output.rows);
        result.output_rows += output.rows;
        result.checksum += output.checksum;
    }
    return result;
}

}  // namespace wb
