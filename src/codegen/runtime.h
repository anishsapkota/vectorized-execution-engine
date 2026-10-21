#pragma once

// Runtime for generated query code. Compiled into every generated library and included by the
// host that calls it, so both sides share one definition of the ABI structs.
//
// Self-contained on purpose: only <cstdint>/<cstddef>, no STL, so a generated query compiles in
// a fraction of a second and never depends on engine struct layouts. Memory comes from
// __builtin_malloc/__builtin_free.
//
// The tables deliberately mirror the engine's algorithms (AggHashTable, HashJoinOperator), so
// benchmarks compare execution models, not hash tables.

#include <cstddef>
#include <cstdint>

namespace wb_rt {

// Bump on any change to the structs below; the host rejects libraries built against another.
constexpr uint32_t kAbiVersion = 4;

// ---------------------------------------------------------------------------------------------
// Inputs and outputs
// ---------------------------------------------------------------------------------------------

struct ColumnRef {
    const void* data;    // Int64: width-byte unsigned codes; Float64: float or double values
    int64_t     base;    // Int64 frame-of-reference base; value = base + code
    uint8_t     width;   // Int64 code width in bytes: 1, 2, 4 or 8
    uint8_t     is_f32;  // Float64 stored as float
};

struct SegmentRef {
    const ColumnRef* cols;       // one per projected column of the scan
    uint32_t         lo, hi;     // row range left after pruning
    uint32_t         width_sig;  // see width_sig()
};

struct ScanInput {
    const SegmentRef* segs;
    uint32_t          num_segs;
};

struct QueryInput {
    const ScanInput* scans;  // in the generator's scan numbering (codegen::bind_scans)
    uint32_t         num_scans;
};

// Results leave as rows: one 64-bit pattern per output column (int64 as is, double bit-copied).
// On SUM(INT64) overflow wb_query sets error_key (or error_global) and returns 3.
struct QueryOutput {
    void*   ctx;
    void    (*emit_row)(void* ctx, const uint64_t* values, uint32_t n);
    int64_t error_key;
    uint8_t error_global;
};

inline uint32_t width_code(uint8_t width) {
    return width == 1 ? 0u : width == 2 ? 1u : width == 4 ? 2u : 3u;
}

// A segment's width signature: width_code of the scan's k-th Int64 column (in projection order)
// in bits 2k..2k+1. Generated code is specialized per signature.

inline uint64_t hash64(int64_t key) { return static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL; }

// ---------------------------------------------------------------------------------------------
// Group rows: same algorithm as wb::GroupSlots. Maps a key to its group's state record (Row,
// defined by the generated code) stored inline in the hash slot; group ids are dense and
// first-seen, for output order.
// ---------------------------------------------------------------------------------------------

template <class Row>
struct GroupRows {
    static constexpr int64_t  kEmpty   = INT64_MIN;
    static constexpr uint32_t kMinSlot = 0xFFFFFFFFu;

    struct Slot {
        int64_t  key;
        uint32_t gid;
        Row      row;
    };

    Slot*     slots    = nullptr;
    size_t    capacity = 0;  // power of two
    int64_t*  keys     = nullptr;  // gid -> key
    uint32_t* slot_of  = nullptr;  // gid -> slot index (kMinSlot for INT64_MIN)
    uint32_t  count    = 0;
    uint32_t  ids_cap  = 0;

    // INT64_MIN is the empty-slot marker, so that key's row lives outside the slot array.
    bool has_min = false;
    Row  min_row{};

    void init(size_t initial = 1024) {
        size_t cap = 1;
        while (cap < initial)
            cap <<= 1;
        allocate_(cap);
    }

    uint32_t size() const { return count; }
    int64_t  key_of(uint32_t gid) const { return keys[gid]; }
    Row&     row_of(uint32_t gid) { return slot_of[gid] == kMinSlot ? min_row : slots[slot_of[gid]].row; }

    // The row of `key`'s group; for a new group `fresh` is set and the caller initializes *row.
    Row* find_or_insert(int64_t key, bool& fresh) {
        if (key == kEmpty) {
            fresh = !has_min;
            if (fresh) {
                has_min = true;
                push_id_(key, kMinSlot);
            }
            return &min_row;
        }
        if (static_cast<size_t>(count) * 2 >= capacity) grow_();
        const size_t mask = capacity - 1;
        for (size_t i = hash64(key) & mask;; i = (i + 1) & mask) {
            Slot& s = slots[i];
            if (s.key == key) {
                fresh = false;
                return &s.row;
            }
            if (s.key == kEmpty) {
                s.key = key;
                s.gid = push_id_(key, static_cast<uint32_t>(i));
                fresh = true;
                return &s.row;
            }
        }
    }

    void release() {
        __builtin_free(slots);
        __builtin_free(keys);
        __builtin_free(slot_of);
        *this = GroupRows{};
    }

   private:
    void allocate_(size_t cap) {
        slots    = static_cast<Slot*>(__builtin_malloc(cap * sizeof(Slot)));
        capacity = cap;
        for (size_t i = 0; i < cap; ++i)
            slots[i].key = kEmpty;
    }

    uint32_t push_id_(int64_t key, uint32_t slot) {
        if (count == ids_cap) {
            ids_cap       = ids_cap ? ids_cap * 2 : 1024;
            auto* nkeys   = static_cast<int64_t*>(__builtin_malloc(ids_cap * sizeof(int64_t)));
            auto* nslots  = static_cast<uint32_t*>(__builtin_malloc(ids_cap * sizeof(uint32_t)));
            if (count) {
                __builtin_memcpy(nkeys, keys, count * sizeof(int64_t));
                __builtin_memcpy(nslots, slot_of, count * sizeof(uint32_t));
            }
            __builtin_free(keys);
            __builtin_free(slot_of);
            keys    = nkeys;
            slot_of = nslots;
        }
        keys[count]    = key;
        slot_of[count] = slot;
        return count++;
    }

    void grow_() {
        Slot*        old     = slots;
        const size_t old_cap = capacity;
        allocate_(old_cap * 2);
        const size_t mask = capacity - 1;
        for (size_t i = 0; i < old_cap; ++i) {
            if (old[i].key == kEmpty) continue;
            size_t j = hash64(old[i].key) & mask;
            while (slots[j].key != kEmpty)
                j = (j + 1) & mask;
            __builtin_memcpy(&slots[j], &old[i], sizeof(Slot));  // key, gid and row move together
            slot_of[old[i].gid] = static_cast<uint32_t>(j);
        }
        __builtin_free(old);
    }
};

// ---------------------------------------------------------------------------------------------
// Exact median: same design as wb::MedianStore. Values live in per-group chains of 2 KiB blocks
// (headers kept out of band) carved from 128 KiB chunks that never move; a group's
// {head, tail, count} handle lives in its Row. Ordering is a total order: numbers ascending with -0.0 before +0.0, then NaNs ordered by
// bit pattern; an even count averages the two middle values in double.
// ---------------------------------------------------------------------------------------------

inline bool median_less(int64_t a, int64_t b) { return a < b; }
inline bool median_less(double a, double b) {
    const bool a_nan = a != a, b_nan = b != b;
    if (a_nan || b_nan) {
        return !a_nan || (b_nan && __builtin_bit_cast(uint64_t, a) < __builtin_bit_cast(uint64_t, b));
    }
    return a < b || (a == b && __builtin_signbit(a) && !__builtin_signbit(b));
}
inline bool median_less(float a, float b) {
    const bool a_nan = a != a, b_nan = b != b;
    if (a_nan || b_nan) {
        return !a_nan || (b_nan && __builtin_bit_cast(uint32_t, a) < __builtin_bit_cast(uint32_t, b));
    }
    return a < b || (a == b && __builtin_signbit(a) && !__builtin_signbit(b));
}

struct MedianHandle {
    uint32_t head  = 0xFFFFFFFFu;
    uint32_t tail  = 0xFFFFFFFFu;
    uint32_t count = 0;
};

template <class T>
struct MedianStore {
    static constexpr uint32_t kNone        = 0xFFFFFFFFu;
    static constexpr uint32_t kBlockValues = static_cast<uint32_t>(2048 / sizeof(T));
    static constexpr uint32_t kChunkBlocks = 64;

    struct Block {
        T v[kBlockValues];
    };
    static_assert(sizeof(Block) == 2048, "MEDIAN blocks must be exactly 2 KiB");
    static_assert((kBlockValues & (kBlockValues - 1)) == 0, "values per block must be a power of two");

    Block**   chunks     = nullptr;
    uint32_t  num_chunks = 0;
    uint32_t  chunks_cap = 0;
    uint32_t* next       = nullptr;  // per block: next block of the chain
    uint32_t* used       = nullptr;  // per block: values stored
    uint32_t  next_block = 0;
    uint32_t  header_cap = 0;
    uint32_t  max_group  = 0;  // largest group seen, for sizing finalize scratch

    void append(MedianHandle& h, T v) {
        if (h.tail == kNone || used[h.tail] == kBlockValues) {
            const uint32_t b = new_block_();
            if (h.tail == kNone) h.head = b;
            else next[h.tail] = b;
            h.tail = b;
        }
        block_(h.tail).v[used[h.tail]++] = v;
        if (++h.count > max_group) max_group = h.count;
    }

    // Median of a non-empty group; copies its chain into scratch (max_group elements).
    double finalize_group(const MedianHandle& h, T* scratch) {
        size_t k = 0;
        for (uint32_t b = h.head; b != kNone; b = next[b]) {
            __builtin_memcpy(scratch + k, block_(b).v, used[b] * sizeof(T));
            k += used[b];
        }
        return select_(h.count, [&](size_t i) -> T& { return scratch[i]; });
    }

    // Median of the only chain (a global aggregate), selected in place over its consecutive blocks.
    double finalize_single(const MedianHandle& h) {
        const uint32_t head = h.head;
        return select_(h.count, [&](size_t i) -> T& {
            return block_(head + static_cast<uint32_t>(i / kBlockValues)).v[i % kBlockValues];
        });
    }

    void release() {
        for (uint32_t c = 0; c < num_chunks; ++c)
            __builtin_free(chunks[c]);
        __builtin_free(chunks);
        __builtin_free(next);
        __builtin_free(used);
        *this = MedianStore{};
    }

   private:
    Block& block_(uint32_t id) { return chunks[id / kChunkBlocks][id % kChunkBlocks]; }

    uint32_t new_block_() {
        if (next_block % kChunkBlocks == 0) {
            if (num_chunks == chunks_cap) {
                chunks_cap  = chunks_cap ? chunks_cap * 2 : 64;
                auto* grown = static_cast<Block**>(__builtin_malloc(chunks_cap * sizeof(Block*)));
                if (num_chunks) __builtin_memcpy(grown, chunks, num_chunks * sizeof(Block*));
                __builtin_free(chunks);
                chunks = grown;
            }
            chunks[num_chunks++] = static_cast<Block*>(__builtin_malloc(kChunkBlocks * sizeof(Block)));
        }
        if (next_block == header_cap) {  // per-block headers: 8 B per 2 KiB block, grown by doubling
            header_cap = header_cap ? header_cap * 2 : kChunkBlocks;
            auto* nn   = static_cast<uint32_t*>(__builtin_malloc(header_cap * sizeof(uint32_t)));
            auto* nu   = static_cast<uint32_t*>(__builtin_malloc(header_cap * sizeof(uint32_t)));
            if (next_block) {
                __builtin_memcpy(nn, next, next_block * sizeof(uint32_t));
                __builtin_memcpy(nu, used, next_block * sizeof(uint32_t));
            }
            __builtin_free(next);
            __builtin_free(used);
            next = nn;
            used = nu;
        }
        next[next_block] = kNone;
        used[next_block] = 0;
        return next_block++;
    }

    // Three-way quickselect through an accessor: afterwards get(k) holds the k-th smallest value
    // and every element before k is <= it (same algorithm as wb::MedianStore).
    template <class Get>
    static double select_(size_t n, Get get) {
        const size_t k  = n / 2;
        size_t       lo = 0, hi = n;
        while (hi - lo > 1) {
            const T pivot = get(lo + (hi - lo) / 2);
            size_t  lt = lo, i = lo, gt = hi;
            while (i < gt) {
                if (median_less(get(i), pivot)) {
                    const T t = get(lt);
                    get(lt++) = get(i);
                    get(i++)  = t;
                } else if (median_less(pivot, get(i))) {
                    const T t = get(--gt);
                    get(gt)   = get(i);
                    get(i)    = t;
                } else {
                    ++i;
                }
            }
            if (k < lt) hi = lt;
            else if (k >= gt) lo = gt;
            else break;
        }
        if (n % 2) return static_cast<double>(get(k));
        T left = get(0);
        for (size_t i = 1; i < k; ++i)
            if (median_less(left, get(i))) left = get(i);
        return (static_cast<double>(left) + static_cast<double>(get(k))) / 2.0;
    }
};

// ---------------------------------------------------------------------------------------------
// Join table: same algorithm as wb::HashJoinOperator. Rows are appended, then finalize()
// stable-sorts them by key and indexes each run of equal keys as (key, first, count).
// Payload values are stored as 64-bit patterns (doubles bit-copied).
// ---------------------------------------------------------------------------------------------

struct JoinBucket {
    int64_t  key;
    uint32_t first;
    uint32_t count;  // 0 marks an empty bucket
};

struct JoinTable {
    uint32_t    ncols     = 0;
    uint32_t    rows      = 0;
    uint32_t    row_cap   = 0;
    int64_t*    keys      = nullptr;  // insertion order until finalize, then sorted
    uint64_t*   staged    = nullptr;  // row-major payload, insertion order
    uint64_t*   sorted    = nullptr;  // column-major payload, key order (after finalize)
    JoinBucket* buckets   = nullptr;
    size_t      nbuckets  = 0;
    unsigned    shift     = 64;

    void init(uint32_t num_cols) {
        ncols = num_cols;
        rows  = 0;
    }

    void append(int64_t key, const uint64_t* payload) {
        if (rows == row_cap) grow_();
        keys[rows] = key;
        for (uint32_t c = 0; c < ncols; ++c)
            staged[static_cast<size_t>(rows) * ncols + c] = payload[c];
        ++rows;
    }

    void finalize() {
        if (rows == 0) return;
        uint32_t* order = sort_rows_by_key_();
        int64_t*  sorted_keys = static_cast<int64_t*>(__builtin_malloc(rows * sizeof(int64_t)));
        sorted = static_cast<uint64_t*>(__builtin_malloc(static_cast<size_t>(rows) * (ncols ? ncols : 1) *
                                                         sizeof(uint64_t)));
        for (uint32_t i = 0; i < rows; ++i) {
            sorted_keys[i] = keys[order[i]];
            for (uint32_t c = 0; c < ncols; ++c)
                sorted[static_cast<size_t>(c) * rows + i] = staged[static_cast<size_t>(order[i]) * ncols + c];
        }
        __builtin_free(order);
        __builtin_free(keys);
        __builtin_free(staged);
        keys   = sorted_keys;
        staged = nullptr;
        build_index_();
    }

    const JoinBucket* find(int64_t key) const {
        if (nbuckets == 0) return nullptr;
        const size_t mask = nbuckets - 1;
        for (size_t i = bucket_of_(key);; i = (i + 1) & mask) {
            const JoinBucket& b = buckets[i];
            if (b.count == 0) return nullptr;
            if (b.key == key) return &b;
        }
    }

    // Payload column c of the row at sorted position `row`.
    uint64_t col(uint32_t c, uint32_t row) const { return sorted[static_cast<size_t>(c) * rows + row]; }

    void release() {
        __builtin_free(keys);
        __builtin_free(staged);
        __builtin_free(sorted);
        __builtin_free(buckets);
        *this = JoinTable{};
    }

   private:
    void grow_() {
        const uint32_t cap = row_cap ? row_cap * 2 : 1024;
        auto* new_keys   = static_cast<int64_t*>(__builtin_malloc(cap * sizeof(int64_t)));
        auto* new_staged = static_cast<uint64_t*>(
            __builtin_malloc(static_cast<size_t>(cap) * (ncols ? ncols : 1) * sizeof(uint64_t)));
        if (rows) {
            __builtin_memcpy(new_keys, keys, rows * sizeof(int64_t));
            __builtin_memcpy(new_staged, staged, static_cast<size_t>(rows) * ncols * sizeof(uint64_t));
        }
        __builtin_free(keys);
        __builtin_free(staged);
        keys    = new_keys;
        staged  = new_staged;
        row_cap = cap;
    }

    // Stable merge sort of row indices by key; equal keys keep insertion order.
    uint32_t* sort_rows_by_key_() const {
        auto* a = static_cast<uint32_t*>(__builtin_malloc(rows * sizeof(uint32_t)));
        auto* b = static_cast<uint32_t*>(__builtin_malloc(rows * sizeof(uint32_t)));
        for (uint32_t i = 0; i < rows; ++i)
            a[i] = i;
        for (uint32_t width = 1; width < rows; width *= 2) {
            for (uint32_t lo = 0; lo < rows; lo += 2 * width) {
                const uint32_t mid = lo + width < rows ? lo + width : rows;
                const uint32_t hi  = lo + 2 * width < rows ? lo + 2 * width : rows;
                uint32_t       i = lo, j = mid, k = lo;
                while (i < mid && j < hi)
                    b[k++] = keys[a[j]] < keys[a[i]] ? a[j++] : a[i++];
                while (i < mid)
                    b[k++] = a[i++];
                while (j < hi)
                    b[k++] = a[j++];
            }
            uint32_t* t = a;
            a           = b;
            b           = t;
        }
        __builtin_free(b);
        return a;
    }

    size_t bucket_of_(int64_t key) const { return static_cast<size_t>(hash64(key) >> shift); }

    void build_index_() {
        uint32_t distinct = 0;
        for (uint32_t i = 0; i < rows; ++i)
            distinct += (i == 0 || keys[i] != keys[i - 1]);
        size_t want = 2 * static_cast<size_t>(distinct);
        nbuckets    = 2;
        unsigned log2 = 1;
        while (nbuckets < want) {
            nbuckets <<= 1;
            ++log2;
        }
        shift   = 64 - log2;
        buckets = static_cast<JoinBucket*>(__builtin_malloc(nbuckets * sizeof(JoinBucket)));
        for (size_t i = 0; i < nbuckets; ++i)
            buckets[i] = JoinBucket{0, 0, 0};

        const size_t mask = nbuckets - 1;
        for (uint32_t start = 0; start < rows;) {
            uint32_t end = start + 1;
            while (end < rows && keys[end] == keys[start])
                ++end;
            size_t i = bucket_of_(keys[start]);
            while (buckets[i].count != 0)
                i = (i + 1) & mask;
            buckets[i] = JoinBucket{keys[start], start, end - start};
            start      = end;
        }
    }
};

}  // namespace wb_rt
