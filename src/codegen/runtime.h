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
constexpr uint32_t kAbiVersion = 1;

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

struct QueryOutput {
    void* ctx;
    void (*emit_group)(void* ctx, int64_t key, double sum, int64_t count);
};

inline uint32_t width_code(uint8_t width) {
    return width == 1 ? 0u : width == 2 ? 1u : width == 4 ? 2u : 3u;
}

// A segment's width signature: width_code of the scan's k-th Int64 column (in projection order)
// in bits 2k..2k+1. Generated code is specialized per signature.

inline uint64_t hash64(int64_t key) { return static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL; }

// ---------------------------------------------------------------------------------------------
// Aggregation table: same algorithm as wb::AggHashTable
// ---------------------------------------------------------------------------------------------

struct AggSlot {
    int64_t key;
    double  sum;
    int64_t count;
};

struct AggTable {
    static constexpr int64_t kEmpty = INT64_MIN;

    AggSlot* slots    = nullptr;
    size_t   size     = 0;
    size_t   capacity = 0;  // power of two

    void init(size_t initial = 1024) {
        size_t cap = 1;
        while (cap < initial)
            cap <<= 1;
        allocate_(cap);
    }

    AggSlot& find_or_insert(int64_t key) {
        if (size * 2 >= capacity) grow_();
        const size_t mask = capacity - 1;
        for (size_t i = hash64(key) & mask;; i = (i + 1) & mask) {
            AggSlot& s = slots[i];
            if (s.key == key) return s;
            if (s.key == kEmpty) {
                s.key = key;
                ++size;
                return s;
            }
        }
    }

    template <class F>
    void for_each(F&& f) const {
        for (size_t i = 0; i < capacity; ++i)
            if (slots[i].key != kEmpty) f(slots[i]);
    }

    void release() {
        __builtin_free(slots);
        slots    = nullptr;
        size     = 0;
        capacity = 0;
    }

   private:
    void allocate_(size_t cap) {
        slots    = static_cast<AggSlot*>(__builtin_malloc(cap * sizeof(AggSlot)));
        capacity = cap;
        size     = 0;
        for (size_t i = 0; i < cap; ++i)
            slots[i] = AggSlot{kEmpty, 0.0, 0};
    }

    void grow_() {
        AggSlot*     old     = slots;
        const size_t old_cap = capacity;
        allocate_(old_cap * 2);
        for (size_t i = 0; i < old_cap; ++i) {
            if (old[i].key == kEmpty) continue;
            AggSlot& s = find_or_insert(old[i].key);
            s.sum      = old[i].sum;
            s.count    = old[i].count;
        }
        __builtin_free(old);
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
