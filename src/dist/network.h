#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace wb {

struct TrafficSnapshot {
    std::uint64_t cross_node_bytes = 0;  // would travel over the real network
    std::uint64_t cross_node_rows  = 0;
    std::uint64_t local_bytes      = 0;  // slice to slice on the same node: a memory copy
    std::uint64_t local_rows       = 0;
};

// Counts data movement between slices. There is no simulated latency or bandwidth yet: bytes
// moved is the quantity Redshift's DS_* join strategies trade off, so we measure that first.
// Thread-safe; every slice thread records into the same instance.
class NetworkStats {
   public:
    void record(std::size_t src_node, std::size_t dst_node, std::uint64_t bytes,
                std::uint64_t rows) {
        if (src_node == dst_node) {
            local_bytes_.fetch_add(bytes, std::memory_order_relaxed);
            local_rows_.fetch_add(rows, std::memory_order_relaxed);
            return;
        }
        cross_node_bytes_.fetch_add(bytes, std::memory_order_relaxed);
        cross_node_rows_.fetch_add(rows, std::memory_order_relaxed);
    }

    // Read after the streams that moved data have joined; the thread joins provide the
    // happens-before edge, so relaxed loads see every recorded value.
    TrafficSnapshot snapshot() const {
        return {cross_node_bytes_.load(std::memory_order_relaxed),
                cross_node_rows_.load(std::memory_order_relaxed),
                local_bytes_.load(std::memory_order_relaxed),
                local_rows_.load(std::memory_order_relaxed)};
    }

    void reset() {
        cross_node_bytes_ = 0;
        cross_node_rows_  = 0;
        local_bytes_      = 0;
        local_rows_       = 0;
    }

   private:
    std::atomic<std::uint64_t> cross_node_bytes_{0};
    std::atomic<std::uint64_t> cross_node_rows_{0};
    std::atomic<std::uint64_t> local_bytes_{0};
    std::atomic<std::uint64_t> local_rows_{0};
};

}  // namespace wb
