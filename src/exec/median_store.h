#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
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

// Values for one MEDIAN aggregate, in per-group chains of 2 KiB blocks carved from 64-block
// chunks that never move: no doubling, no copy on growth, no group-id array. T is the column's
// storage type (int64_t, double, or float for narrow columns); values widen to double only
// after selection.
template <typename T>
class MedianStore {
   public:
    using value_type = T;
    // Blocks hold exactly 2 KiB of values (256 int64/double or 512 float, a power of two) with
    // their next/size headers kept out of band, so chunks are exactly 128 KiB (whole pages) and
    // value i of a chain is found with a shift and a mask.
    static constexpr uint32_t kBlockValues = static_cast<uint32_t>(2048 / sizeof(T));
    static constexpr uint32_t kChunkBlocks = 64;

    void append(MedianHandle& h, T v) {
        if (h.tail == UINT32_MAX || size_[h.tail] == kBlockValues) {
            const uint32_t b = new_block_();
            if (h.tail == UINT32_MAX) h.head = b;
            else next_[h.tail] = b;
            h.tail = b;
        }
        block_(h.tail).v[size_[h.tail]++] = v;
        ++h.count;
    }

    // Median of a non-empty group; copies its chain into `scratch` (reused across groups).
    double finalize_group(const MedianHandle& h, std::vector<T>& scratch) {
        scratch.resize(h.count);
        std::size_t k = 0;
        for (uint32_t b = h.head; b != UINT32_MAX; b = next_[b]) {
            const Block& block = block_(b);
            std::copy(block.v, block.v + size_[b], scratch.data() + k);
            k += size_[b];
        }
        return select_(scratch.data(), scratch.size());
    }

    // Median of the only chain in the store (a global aggregate), selected in place: the chain's
    // blocks are consecutive and all full but the last, so value i is at block
    // head + i / kBlockValues, slot i % kBlockValues. No second copy of the values is made; the blocks are released afterwards.
    double finalize_single(const MedianHandle& h) {
        const uint32_t head = h.head;
        auto           get  = [&](std::size_t i) -> T& {
            return block_(head + static_cast<uint32_t>(i / kBlockValues)).v[i % kBlockValues];
        };
        const double median = select_in_place_(h.count, get);
        chunks_.clear();
        next_.clear();
        size_.clear();
        next_block_ = 0;
        return median;
    }

    std::size_t bytes_reserved() const {
        std::size_t live = 0;
        for (const auto& c : chunks_)
            live += c ? sizeof(Chunk) : 0;
        return live;
    }

   private:
    struct Block {
        T v[kBlockValues];
    };
    struct Chunk {
        Block blocks[kChunkBlocks];
    };
    static_assert(sizeof(Block) == 2048, "MEDIAN blocks must be exactly 2 KiB");
    static_assert((kBlockValues & (kBlockValues - 1)) == 0, "values per block must be a power of two");

    Block& block_(uint32_t id) { return chunks_[id / kChunkBlocks]->blocks[id % kChunkBlocks]; }

    uint32_t new_block_() {
        if (next_block_ % kChunkBlocks == 0) chunks_.push_back(std::unique_ptr<Chunk>(new Chunk));
        next_.push_back(UINT32_MAX);
        size_.push_back(0);
        return next_block_++;
    }

    // Three-way quickselect through an accessor: afterwards get(k) holds the k-th smallest value
    // and every element before k is <= it (same algorithm as the codegen runtime).
    template <typename Get>
    static double select_in_place_(std::size_t n, Get get) {
        const std::size_t k  = n / 2;
        std::size_t       lo = 0, hi = n;
        while (hi - lo > 1) {
            const T     pivot = get(lo + (hi - lo) / 2);
            std::size_t lt = lo, i = lo, gt = hi;  // [lo,lt) < pivot, [lt,i) == pivot, [gt,hi) > pivot
            while (i < gt) {
                if (median_less(get(i), pivot)) std::swap(get(lt++), get(i++));
                else if (median_less(pivot, get(i))) std::swap(get(--gt), get(i));
                else ++i;
            }
            if (k < lt) hi = lt;
            else if (k >= gt) lo = gt;
            else break;
        }
        if (n % 2) return static_cast<double>(get(k));
        T left = get(0);
        for (std::size_t i = 1; i < k; ++i)
            if (median_less(left, get(i))) left = get(i);
        return (static_cast<double>(left) + static_cast<double>(get(k))) / 2.0;
    }

    static double select_(T* values, std::size_t n) {
        T* mid = values + n / 2;
        std::nth_element(values, mid, values + n, median_less<T>);
        if (n % 2) return static_cast<double>(*mid);
        const T lo = *std::max_element(values, mid, median_less<T>);  // everything left of mid is <= *mid
        return (static_cast<double>(lo) + static_cast<double>(*mid)) / 2.0;
    }

    std::vector<std::unique_ptr<Chunk>> chunks_;
    std::vector<uint32_t>               next_;  // per block: next block of the chain
    std::vector<uint32_t>               size_;  // per block: values stored
    uint32_t                            next_block_ = 0;
};

}  // namespace wb
