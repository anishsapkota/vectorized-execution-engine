#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace wb {

// MEDIAN's total order (shared with the codegen runtime): numbers ascending with -0.0 before
// +0.0, then NaNs ordered by bit pattern. For int64 it is plain <.
template <typename T>
bool median_less(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        using Bits       = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
        const bool a_nan = std::isnan(a), b_nan = std::isnan(b);
        if (a_nan || b_nan) return !a_nan || (b_nan && std::bit_cast<Bits>(a) < std::bit_cast<Bits>(b));
        return a < b || (a == b && std::signbit(a) && !std::signbit(b));
    } else {
        return a < b;
    }
}

// Where one group's MEDIAN values live: a chain of blocks, plus the number of values. Stored in
// the group's inline aggregate state.
struct MedianHandle {
    uint32_t head  = UINT32_MAX;
    uint32_t tail  = UINT32_MAX;
    uint32_t count = 0;
};

// Values for one MEDIAN aggregate, in per-group chains of 256-value blocks carved from 64-block
// chunks that never move: no doubling, no copy on growth, no group-id array. T is the column's
// storage type (int64_t, double, or float for narrow columns); values widen to double only
// after selection.
template <typename T>
class MedianStore {
   public:
    static constexpr uint32_t kBlockValues = 256;
    static constexpr uint32_t kChunkBlocks = 64;

    void append(MedianHandle& h, T v) {
        if (h.tail == UINT32_MAX || block_(h.tail).n == kBlockValues) {
            const uint32_t b = new_block_();
            if (h.tail == UINT32_MAX) h.head = b;
            else block_(h.tail).next = b;
            h.tail = b;
        }
        Block& block = block_(h.tail);
        block.v[block.n++] = v;
        ++h.count;
    }

    // Median of a non-empty group; copies its chain into `scratch` (reused across groups).
    double finalize_group(const MedianHandle& h, std::vector<T>& scratch) {
        scratch.resize(h.count);
        std::size_t k = 0;
        for (uint32_t b = h.head; b != UINT32_MAX; b = block_(b).next) {
            const Block& block = block_(b);
            std::copy(block.v, block.v + block.n, scratch.data() + k);
            k += block.n;
        }
        return select_(scratch.data(), scratch.size());
    }

    // Median of the only chain in the store (a global aggregate). Frees each chunk as soon as its
    // values are copied out, so the peak is the values plus one chunk.
    double finalize_single(const MedianHandle& h) {
        std::vector<T> all;
        all.reserve(h.count);
        for (uint32_t b = h.head; b != UINT32_MAX;) {
            const Block&   block = block_(b);
            const uint32_t next  = block.next;
            all.insert(all.end(), block.v, block.v + block.n);
            if (next == UINT32_MAX || (next / kChunkBlocks) != (b / kChunkBlocks)) chunks_[b / kChunkBlocks].reset();
            b = next;
        }
        return select_(all.data(), all.size());
    }

    std::size_t bytes_reserved() const {
        std::size_t live = 0;
        for (const auto& c : chunks_)
            live += c ? sizeof(Chunk) : 0;
        return live;
    }

   private:
    struct Block {
        uint32_t next = UINT32_MAX;
        uint32_t n    = 0;
        T        v[kBlockValues];
    };
    struct Chunk {
        Block blocks[kChunkBlocks];
    };

    Block& block_(uint32_t id) { return chunks_[id / kChunkBlocks]->blocks[id % kChunkBlocks]; }

    uint32_t new_block_() {
        if (next_block_ % kChunkBlocks == 0) chunks_.push_back(std::unique_ptr<Chunk>(new Chunk));
        return next_block_++;
    }

    static double select_(T* values, std::size_t n) {
        T* mid = values + n / 2;
        std::nth_element(values, mid, values + n, median_less<T>);
        if (n % 2) return static_cast<double>(*mid);
        const T lo = *std::max_element(values, mid, median_less<T>);  // everything left of mid is <= *mid
        return (static_cast<double>(lo) + static_cast<double>(*mid)) / 2.0;
    }

    std::vector<std::unique_ptr<Chunk>> chunks_;
    uint32_t                            next_block_ = 0;
};

}  // namespace wb
