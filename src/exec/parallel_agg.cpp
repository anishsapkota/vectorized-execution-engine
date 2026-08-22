#include "exec/parallel_agg.h"

#include <atomic>
#include <thread>
#include <vector>

#include "storage/segment.h"
#include "storage/vector.h"

namespace wb {

namespace {

void worker_loop(const Table& table, std::size_t group_col_idx, std::size_t agg_col_idx,
                 std::atomic<size_t>& next_seg, AggHashTable& local_ht) {
    const std::size_t total = table.num_segments();
    while (true) {
        // Claim one segment atomically. fetch_add returns the OLD value, which is the index this
        // worker just claimed.
        std::size_t seg_idx = next_seg.fetch_add(1, std::memory_order_relaxed);
        if (seg_idx >= total) break;
        const Segment& seg     = table.segment(seg_idx);
        const auto&    key_vec = static_cast<const Int64Vector&>(seg.column(group_col_idx));
        const auto&    val_vec = static_cast<const Float64Vector&>(seg.column(agg_col_idx));

        auto keys = key_vec.data();
        auto vals = val_vec.data();

        const std::size_t n = seg.num_rows();
        for (std::size_t i = 0; i < n; ++i) {
            auto& v = local_ht.find_or_insert(keys[i]);
            v.sum += vals[i];
            v.count += 1;
        }
    }
}

}  // namespace

AggHashTable parallel_group_sum_count(const Table& table, std::size_t group_col_idx,
                                      std::size_t agg_col_idx, std::size_t num_workers) {
    if (num_workers <= 1) {
        // single threaded path - no thread overhead
        AggHashTable        ht{32768};
        std::atomic<size_t> ctr{0};
        worker_loop(table, group_col_idx, agg_col_idx, ctr, ht);
        return ht;
    }

    std::atomic<size_t>       next_seg{0};
    std::vector<AggHashTable> locals;
    locals.reserve(num_workers);

    for (std::size_t w = 0; w < num_workers; ++w) {
        locals.emplace_back(32768);  // pre-sized to ~10k expected groups
    }
    std::vector<std::thread> threads;
    threads.reserve(num_workers);
    for (std::size_t w = 0; w < num_workers; ++w) {
        threads.emplace_back(worker_loop, std::cref(table), group_col_idx, agg_col_idx,
                             std::ref(next_seg), std::ref(locals[w]));
    }

    for (auto& t : threads)
        t.join();

    // Merge phase: collapse all locals into the first.

    AggHashTable& global = locals[0];
    for (std::size_t w = 1; w < num_workers; ++w) {
        locals[w].for_each([&](int64_t k, const AggHashTable::Value& v) {
            auto& dst = global.find_or_insert(k);
            dst.sum += v.sum;
            dst.count += v.count;
        });
    }
    return std::move(global);
}

}  // namespace wb