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

## 7. Results (2026-10-21)

Setup: Apple M3 Pro, single-threaded, best of 7 runs, 10M-row readings, 2M orders, 100k customers.
Raw output: `bench/results/mem-before.txt`, `mem-after.txt`, `codegen.txt` and
`after-footprint.txt`. All 18 compiled rows in `wb_codegen` still match the vectorized engine bit
for bit, and the suite (111 tests, including ASan+UBSan over the generated libraries) passes.

### Targets

| Target | Before | After | Result |
|---|---|---|---|
| Q7 MEDIAN peak, vec / cg | 382.9 / 383.3 MiB (40.2 B/row) | **89.0 / 88.8 MiB** (9.3 B/row) | met (≤ 100) |
| Q8 MEDIAN peak, vec / cg | 381.3 / 381.9 MiB | **75.9 / 75.8 MiB** (8.0 B/row) | met (≤ 80) |
| Q7 on `narrow_float`, vec / cg | 380.3 / 382.6 MiB | **51.2 / 50.5 MiB** | met (≈ half of Q7) |
| Q4 grouped SUM/COUNT, cg-O3 | 25.4 ms | **12.4 ms** | met (≤ ~18) |
| Q4 grouped SUM/COUNT, vec | 26.8 ms | 28.8 ms | **missed** |
| Q5 / Q9 vectorized allocations per run | ~31,000 | 301 / 397 | **missed** (≤ 100) |

**Q4 vectorized: missed, and slightly worse.** Phase probe of the vectorized operator: pass 1
alone (find each row's state, count) takes 21.2 ms; the per-aggregate SUM pass adds about 7 ms;
the old fused `AggHashTable` loop takes 18.9 ms (measured). Inline state helps a fused per-row
loop: the generated code is 2.33× faster than the vectorized engine. It does not help per-aggregate
passes, which read better from a dense state array (the 80 KB sum array stayed cache-resident)
than from state spread across a 1 MiB slot table. The review's premise, that the slot → gid → state
hop was the main cost, was right for the compiled engine and wrong for the vectorized one.

**Join allocations: missed, but per-batch allocation is gone.** At a quarter of the data, Q5 still
makes 294 allocations (against 301), so what remains is a fixed per-query cost: build-side
materialization, the aggregate's tables and operator setup. The per-batch output allocations
(~31k) are eliminated. Q9's count grows with data only through MEDIAN chunks (128 KiB each).

### Effect on the experiment (timings, cg-O3 against vec)

| Query | vec before → after | cg before → after | cg speedup after |
|---|---|---|---|
| Q4 group by | 26.8 → 28.8 ms | 25.4 → 12.4 ms | **2.33×** (was 1.05×) |
| Q5 join + group | 21.8 → **11.5 ms** | 12.9 → 12.4 ms | **0.93×** (was 1.68×) |
| Q6 min/max/avg/count | 38.1 → 41.3 ms | 32.5 → 20.6 ms | **2.00×** (was 1.17×) |
| Q7 median by group | 149.3 → 158.8 ms | 136.4 → 130.8 ms | 1.21× |
| Q8 global median | 67.3 → 69.8 ms | 64.1 → 63.3 ms | 1.10× |
| Q9 join + avg/median | 33.7 → 23.0 ms | 23.1 → 26.2 ms | **0.88×** (was 1.46×) |

Two earlier conclusions change:

- **Joins.** The compiled engine's 1.7× win on Q5 (and 1.5× on Q9) was mostly the vectorized
  join's per-batch materialization, not the execution model. With reused output columns the
  vectorized join is as fast or faster.
- **Grouped aggregation.** With state inline in the hash slot, the compiled engine is now about 2×
  faster on Q4 and Q6. The fused per-row update touches one cache line per row; the vectorized
  engine's per-aggregate passes cannot exploit that layout.

### Deviations from §3

- The global MEDIAN is selected in place over its block chain (three-way quickselect through an
  index accessor) instead of being copied into an exact-size array while chunks are freed. On this
  allocator, freeing ~128 KiB chunks does not lower the process footprint (standalone repro:
  162 MiB with and without freeing), so the copy alone kept Q8 at 161.5 MiB.
- Blocks hold exactly 2 KiB of values (256 int64/double or 512 float) with their next/size headers
  in side arrays. 256-value blocks with inline headers spilled each 128 KiB chunk into an extra
  16 KiB page (Q8 85.2 MiB), and 255-value blocks made in-place indexing a division
  (Q8 67 → 88 ms).

The `wb_bench` checksums are unchanged except "groupby device MT": that total varies between runs
(…0325672 / …03256726) because the parallel merge order depends on thread scheduling. That path
was not touched by this work.

## 8. Allocator comparison: system malloc vs jemalloc 5.4.0 (2026-10-21)

`make ALLOCATOR=jemalloc` links every binary against Homebrew's jemalloc. It registers as the
default malloc zone, so it also serves the dlopen'ed generated query libraries. That was checked:
a separately built library's 100 MiB allocation appears in jemalloc's `stats.allocated`. Every tool
prints the allocator it ran on. `bench/run_allocators.sh` reproduces everything below; raw output
is in `bench/results/alloc/`. Both allocators: 112/112 tests, 18/18 compiled rows matching at 10M
and at 100M rows.

**Speed: no meaningful difference.** Every query is within ±5% at both sizes, for example:

| Query | 100M vec | 100M cg-O3 |
|---|---|---|
| Q4 | 280 → 278 ms | 119 → 119 ms |
| Q7 | 1633 → 1691 ms (+4%) | 1336 → 1370 ms (+3%) |
| Q8 | 564 → 567 ms | 526 → 534 ms |

The largest gap is Q8 cg at 10M (48.8 → 54.1 ms, +11%). This is expected: hot loops allocate
nothing, and a query makes only tens to a few thousand allocations. The parallel group-by is
slightly slower under jemalloc (100M: 33.0 → 35.6 ms; 10M: 4.15 → 4.49 ms).

**Memory: jemalloc uses more.**

| Measure | system | jemalloc |
|---|---|---|
| Q7 MEDIAN peak, 100M | 791 MiB | 883 MiB (+12%) |
| Q8 MEDIAN peak, 100M | 768 MiB | 856 MiB (+11%) |
| Q7 on `narrow_float`, 100M | 400 MiB | 447 MiB (+12%) |
| Q7 MEDIAN peak, 10M | 89 MiB | 100 MiB (+12%) |
| Process footprint after loading 100M rows | 1,255 MiB | 1,340 MiB (+7%) |

`MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0` (return freed pages immediately) does not reduce
these numbers, so the extra is not freed-page retention. Its cause was not determined.

**Conclusion:** for this engine the system allocator stays the default. Allocation is off the hot
path, so jemalloc buys no speed and costs 7–12% memory. The option remains for experiments.
