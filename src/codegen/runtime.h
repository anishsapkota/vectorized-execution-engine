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
constexpr uint32_t kAbiVersion = 3;

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
// Group table: same algorithm as wb::GroupIndex. Maps a key to a dense group id in first-seen
// order; per-group state lives in a StateRows indexed by that id.
// ---------------------------------------------------------------------------------------------

struct GroupTable {
    static constexpr int64_t kEmpty = INT64_MIN;

    struct Slot {
        int64_t  key;
        uint32_t gid;
    };

    Slot*    slots    = nullptr;
    size_t   capacity = 0;  // power of two
    int64_t* keys     = nullptr;  // gid -> key
    uint32_t count    = 0;        // groups, including the INT64_MIN key if present
    uint32_t keys_cap = 0;

    // INT64_MIN is the empty-slot marker, so that key lives outside the slot array.
    bool     has_min = false;
    uint32_t min_gid = 0;

    void init(size_t initial = 1024) {
        size_t cap = 1;
        while (cap < initial)
            cap <<= 1;
        allocate_(cap);
    }

    uint32_t size() const { return count; }
    int64_t  key_of(uint32_t gid) const { return keys[gid]; }

    uint32_t find_or_insert(int64_t key) {
        if (key == kEmpty) {
            if (!has_min) {
                has_min = true;
                min_gid = push_key_(key);
            }
            return min_gid;
        }
        if (static_cast<size_t>(count) * 2 >= capacity) grow_();
        const size_t mask = capacity - 1;
        for (size_t i = hash64(key) & mask;; i = (i + 1) & mask) {
            Slot& s = slots[i];
            if (s.key == key) return s.gid;
            if (s.key == kEmpty) {
                s = Slot{key, push_key_(key)};
                return s.gid;
            }
        }
    }

    void release() {
        __builtin_free(slots);
        __builtin_free(keys);
        *this = GroupTable{};
    }

   private:
    void allocate_(size_t cap) {
        slots    = static_cast<Slot*>(__builtin_malloc(cap * sizeof(Slot)));
        capacity = cap;
        for (size_t i = 0; i < cap; ++i)
            slots[i] = Slot{kEmpty, 0};
    }

    uint32_t push_key_(int64_t key) {
        if (count == keys_cap) {
            keys_cap      = keys_cap ? keys_cap * 2 : 1024;
            auto* grown   = static_cast<int64_t*>(__builtin_malloc(keys_cap * sizeof(int64_t)));
            if (count) __builtin_memcpy(grown, keys, count * sizeof(int64_t));
            __builtin_free(keys);
            keys = grown;
        }
        keys[count] = key;
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
            slots[j] = old[i];
        }
        __builtin_free(old);
    }
};

// Growable array of per-group state records. Row is defined by the generated code.
template <class Row>
struct StateRows {
    Row*     data     = nullptr;
    uint32_t size     = 0;
    uint32_t capacity = 0;

    Row& push(const Row& row) {
        if (size == capacity) {
            capacity    = capacity ? capacity * 2 : 1024;
            auto* grown = static_cast<Row*>(__builtin_malloc(capacity * sizeof(Row)));
            if (size) __builtin_memcpy(grown, data, size * sizeof(Row));
            __builtin_free(data);
            data = grown;
        }
        data[size] = row;
        return data[size++];
    }

    void release() {
        __builtin_free(data);
        *this = StateRows{};
    }
};

// ---------------------------------------------------------------------------------------------
// Exact median: collects (group id, value) pairs, then per group selects the middle value(s).
// Ordering puts NaN after every number, matching the vectorized engine; an even count averages
// the two middle values in double.
// ---------------------------------------------------------------------------------------------

inline bool median_less(int64_t a, int64_t b) { return a < b; }
// Total order: numbers ascending with -0.0 before +0.0, then NaNs ordered by bit pattern, so
// values `<` cannot tell apart still get a fixed position (same order as HashAggOperator).
inline bool median_less(double a, double b) {
    const bool a_nan = a != a, b_nan = b != b;
    if (a_nan || b_nan) {
        return !a_nan || (b_nan && __builtin_bit_cast(uint64_t, a) < __builtin_bit_cast(uint64_t, b));
    }
    return a < b || (a == b && __builtin_signbit(a) && !__builtin_signbit(b));
}

template <class T>
struct MedianBuffer {
    uint32_t* gids     = nullptr;
    T*        values   = nullptr;
    size_t    size     = 0;
    size_t    capacity = 0;

    void append(uint32_t gid, T v) {
        if (size == capacity) grow_();
        gids[size]   = gid;
        values[size] = v;
        ++size;
    }

    // counts[g] = number of values appended for group g; writes each group's median to out[g].
    void finalize(const int64_t* counts, uint32_t groups, double* out) {
        size_t* start = static_cast<size_t*>(__builtin_malloc((static_cast<size_t>(groups) + 1) * sizeof(size_t)));
        size_t* next  = static_cast<size_t*>(__builtin_malloc((static_cast<size_t>(groups) + 1) * sizeof(size_t)));
        start[0]      = 0;
        for (uint32_t g = 0; g < groups; ++g)
            start[g + 1] = start[g] + static_cast<size_t>(counts[g]);
        for (uint32_t g = 0; g < groups; ++g)
            next[g] = start[g];
        T* bucketed = static_cast<T*>(__builtin_malloc((size ? size : 1) * sizeof(T)));
        for (size_t i = 0; i < size; ++i)
            bucketed[next[gids[i]]++] = values[i];
        for (uint32_t g = 0; g < groups; ++g)
            out[g] = median_(bucketed + start[g], start[g + 1] - start[g]);
        __builtin_free(bucketed);
        __builtin_free(next);
        __builtin_free(start);
    }

    void release() {
        __builtin_free(gids);
        __builtin_free(values);
        *this = MedianBuffer{};
    }

   private:
    void grow_() {
        capacity          = capacity ? capacity * 2 : 4096;
        auto* new_gids    = static_cast<uint32_t*>(__builtin_malloc(capacity * sizeof(uint32_t)));
        auto* new_values  = static_cast<T*>(__builtin_malloc(capacity * sizeof(T)));
        if (size) {
            __builtin_memcpy(new_gids, gids, size * sizeof(uint32_t));
            __builtin_memcpy(new_values, values, size * sizeof(T));
        }
        __builtin_free(gids);
        __builtin_free(values);
        gids   = new_gids;
        values = new_values;
    }

    // Quickselect with three-way partitioning: afterwards a[k] holds the k-th smallest value and
    // every element before k is <= it.
    static void select_(T* a, size_t n, size_t k) {
        size_t lo = 0, hi = n;  // active range [lo, hi)
        while (hi - lo > 1) {
            const T pivot = a[lo + (hi - lo) / 2];
            size_t  lt = lo, i = lo, gt = hi;  // [lo,lt) < pivot, [lt,i) == pivot, [gt,hi) > pivot
            while (i < gt) {
                if (median_less(a[i], pivot)) {
                    const T t = a[lt];
                    a[lt++]   = a[i];
                    a[i++]    = t;
                } else if (median_less(pivot, a[i])) {
                    const T t = a[--gt];
                    a[gt]     = a[i];
                    a[i]      = t;
                } else {
                    ++i;
                }
            }
            if (k < lt) hi = lt;
            else if (k >= gt) lo = gt;
            else return;  // a[k] equals the pivot
        }
    }

    static double median_(T* a, size_t n) {
        const size_t k = n / 2;
        select_(a, n, k);
        if (n % 2) return static_cast<double>(a[k]);
        T lo = a[0];
        for (size_t i = 1; i < k; ++i)
            if (median_less(lo, a[i])) lo = a[i];
        return (static_cast<double>(lo) + static_cast<double>(a[k])) / 2.0;
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
