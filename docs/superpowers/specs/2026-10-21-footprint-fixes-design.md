# Footprint fixes: inline group state, block-chained MEDIAN, reusable join output

Date: 2026-10-21
Status: design approved, spec under review
Builds on: `2026-10-21-aggregates-design.md` and `2026-10-21-codegen-vs-vectorized-design.md`

## 1. Goal

Fix the four findings of the footprint review of branch `codegen`, in both engines, without changing
any result:

| ID | Finding | Measured now | Target |
|---|---|---|---|
| F1 | MEDIAN buffers grow by doubling and carry a gid per value, plus a bucketed copy at finalize | Q7 382 MiB, Q8 381 MiB peak (40 B/row) | Q7 ≤ 100 MiB, Q8 ≤ 80 MiB, both engines |
| F2 | Group state sits behind a gid indirection (slot → gid → state arrays) | Q4 26.7 ms vec, 25.4 ms cg (old fused table: 16.9 / 17.9 ms) | Q4 ≤ ~18 ms, both engines |
| F3 | `HashJoinOperator` builds fresh FOR-encoded chunks per output batch | Q5/Q9 vec ~31k allocations/run | ≤ 100 allocations/run |
| F4 | MEDIAN over a `narrow_float` column stores doubles | 8 B/value | 4 B/value |

Plus the doc fix: aggregates spec §10 states Q7 MEDIAN memory as "about 280–300 MB"; the measured
value is 382 MiB (40 B/row).

Hard constraints:

- Both engines stay **bit-identical** to each other and to the reference interpreter. The whole
  existing reference and differential suite passes unchanged.
- Arithmetic, row order per group, hashing, probing and growth rules are unchanged, and identical
  in both engines.
- The runtime header still uses only `<cstdint>`/`<cstddef>`; `kAbiVersion` becomes 4.
- `exec/` still does not include `plan/`.
- `wb_bench` checksums equal `bench/results/after-aggregates.txt`.

If a target is missed, the results section reports the measured number and the cause; targets are
not adjusted after the fact.

Non-goals: `AggHashTable`/`parallel_agg`, the exchange path, `ChunkBuilder`'s `wire_bytes`, build-side
materialization of the join, approximate medians.

## 2. Inline group state (F2)

### 2.1 Vectorized: `exec/group_slots.h`

`GroupSlots` replaces `GroupIndex` and the per-aggregate state arrays and `counts_` inside
`HashAggOperator`.

- Slot layout: `[key int64 | gid uint32 | pad uint32 | state]`; stride = 16 + state bytes, rounded
  up to 16. The slot array is 16-byte aligned.
- State layout, computed once from the `AggSpec` list, in this order starting at state offset 0
  (which is 16-aligned, since the slot header is 16 B):
  1. every `__int128` field (SUM/AVG over Int64), 16 B each;
  2. `count` (int64);
  3. every 8-byte field (Float64 SUM/AVG, MIN, MAX), in aggregate order;
  4. per MEDIAN aggregate a `{head, tail}` pair of uint32 block ids (section 3).
- API:

  ```cpp
  class GroupSlots {
     public:
      GroupSlots(std::size_t state_bytes, std::size_t state_align, std::size_t initial_capacity = 1024);
      void       reserve(std::size_t groups);                    // grow now if needed
      std::byte* find_or_insert(int64_t key, bool& fresh);      // fresh: new group, state uninitialized
      uint32_t   size() const;
      int64_t    key_of(uint32_t gid) const;
      std::byte* state_of(uint32_t gid);                        // via a gid -> slot index array
      std::size_t capacity() const;
  };
  ```

- Same hash (`key * 0x9E3779B97F4A7C15`, low bits), linear probing, initial capacity 1024, grow when
  `size * 2 >= capacity`, INT64_MIN held outside the slot array, gids dense in first-seen order. A
  rehash moves each slot's state with it and updates the gid → slot index array.

`HashAggOperator`, per batch:

1. `slots_.reserve(slots_.size() + live_rows)`, so no rehash happens during the batch and every state
   pointer taken in pass 1 stays valid for the batch.
2. Pass 1: for each live row, `ptrs_[k] = find_or_insert(key, fresh)`; a fresh group's state is
   initialized (count 0, sums +0.0 / 0, MIN/MAX = this row's value, MEDIAN chain empty); then
   `++count`. Global aggregates keep one state block and the register-accumulating path.
3. One loop per aggregate: `*reinterpret_cast<T*>(ptrs_[k] + offset)` updated by the existing rules.

Output walks gids 0..n-1 through `state_of(gid)`.

### 2.2 Compiled: `wb_rt::GroupRows<Row>`

Replaces `GroupTable` + `StateRows<Row>`:

```cpp
template <class Row> struct GroupRows {
    struct Slot { int64_t key; uint32_t gid; Row row; };
    void     init(size_t initial = 1024);
    Row*     find_or_insert(int64_t key, bool& fresh);   // fresh: caller initializes *row
    uint32_t size() const;
    int64_t  key_of(uint32_t gid) const;
    Row&     row_of(uint32_t gid);                       // via slot_of[gid]
    void     release();
};
```

Same hash/probe/growth rules and INT64_MIN handling as `GroupSlots`. Generated code uses the
returned `Row*` immediately within the row, so a rehash between rows is harmless.

## 3. MEDIAN storage (F1, F4)

`MedianStore<T>`: vectorized in `exec/median_store.h`, runtime in `runtime.h` (no STL), same design.

- Block: `{uint32_t next; uint32_t n; T v[256];}`. Blocks come from 64-block chunks that are never
  moved; only the chunk pointer table grows. Block ids are uint32 (`chunk = id >> 6`,
  `slot = id & 63`).
- A MEDIAN aggregate keeps `{head, tail}` per group in the group's inline state (empty: both
  `UINT32_MAX`). `append(handle, v)` writes into the tail block, linking a new block when it is full.
- Grouped finalize: for each gid, copy its chain into a scratch array sized to the largest group,
  then select with the existing algorithm (vectorized `std::nth_element` + left max; runtime
  three-way quickselect + left max) under the existing total order (numbers ascending with
  `-0.0` before `+0.0`, NaNs last ordered by bit pattern).
- Global finalize: one chain; copy it into an exact-size array, **freeing each chunk as soon as it is
  copied**, then select in place.

Element type (F4) = the input's storage type:

- `float` when the aggregated column comes straight from a scan of a `narrow_float` column;
- `int64_t` for Int64;
- `double` otherwise (including build-side columns that pass through a join, which both engines
  carry as doubles).

Widening float → double preserves the total order (monotone; `-0.0` before `+0.0`; NaN bit
patterns stay ordered by bits), and both engines widen only the selected values, so results are
unchanged. Vectorized detects the storage type per aggregate from the first batch
(`Float64Vector::narrow()`); the generator knows it from the scan column's `ColumnDef`.

Estimated peaks for 10M rows: Q7 ≈ 76 MiB values + ~10 MiB half-block slack (10k groups) + ~0.3 MiB
headers + ~0.8 MiB scratch ≈ **88 MiB**; Q8 ≈ 76 MiB + one 128 KiB chunk ≈ **77 MiB**; Q7 on
`narrow_float` ≈ **48 MiB**.

## 4. Join output reuse (F3)

`HashJoinOperator` owns its output columns, created at the first output batch with capacity for
`kOutputRows` and refilled in place by every `next()`:

- `int64_t* Int64Vector::reset_plain(std::size_t n)`: the vector becomes width 8, base 0 (decode is
  the identity), keeps its code capacity, returns the writable codes.
- `double* Float64Vector::reset(std::size_t n)`: the vector becomes a non-narrow double vector,
  keeps its capacity, returns the writable values.
- Both are documented "for operator-owned output vectors only"; the Batch lifetime contract
  ("valid until the next `next()`") already permits overwriting.
- Gather: per output batch, resolve each probe column's encoding once (`visit`) and write
  `out[k] = value(probe_row[k])`; build columns likewise from the sorted build chunk.
- `out_cols_` points at the same `Vector` objects for the operator's lifetime.

Unchanged: build-side materialization (one sorted chunk per query), `ChunkBuilder`, the exchange
path, output values (narrow probe floats were already widened to double).

## 5. Tools, testing, verification

New tool `wb_mem` (`src/tools/mem_probe.cpp`, Makefile target): loads readings, runs one named
query (`q4`, `q7`, `q8`, `q7f` = Q7 on a `narrow_float` copy) on one engine (`vec` / `cg`) once, and
prints the peak physical footprint above the loaded table in MiB and B/row (macOS
`ledger_phys_footprint_peak`; Linux `VmHWM`). One query per process.

Unit tests:

- `GroupSlots`: dense first-seen gids, INT64_MIN key, growth preserves gids and state bytes,
  `reserve(size + n)` keeps earlier state pointers stable across n new inserts, `__int128` fields
  16-aligned in every slot.
- `GroupRows`: identical gid sequence to `GroupSlots` on a random key stream including INT64_MIN.
- `MedianStore` (both): equals a sorted reference bit for bit with NaN payloads and ±0; group sizes
  255, 256, 257, 513; a global chain over several chunks; float, double and int64 element types.
- `reset_plain` / `reset`: values round-trip; the data pointer is unchanged across resets with
  `n ≤ capacity`.

Regression: the existing suite passes unchanged; ASan+UBSan run including generated libraries.

Measured verification (results section of this spec): `wb_mem` for Q7, Q8, Q7f × both engines;
`wb_codegen` Q1–Q9 timings, allocations and `match`; `wb_bench` checksums.

Docs: aggregates spec §10 Q7 memory corrected to the measured 382 MiB (40 B/row) with a pointer to
this spec.

## 6. Risks

- Wide aggregate lists widen every slot (Q6: 48 B, 1.5 MiB); accepted by design (always inline).
- `reserve(size + live_rows)` grows the table up to one batch (4,096 groups) earlier than needed;
  negligible.
- `reset_plain`/`reset` weaken column immutability for one operator; contained by documentation and
  the existing Batch lifetime contract.
