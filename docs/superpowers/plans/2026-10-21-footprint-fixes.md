# Footprint Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix findings F1–F4 without changing any result:
- F1: cut MEDIAN memory about 4×;
- F2: recover the grouped SUM/COUNT regression;
- F3: remove per-batch join output allocation;
- F4: store float32 MEDIAN inputs at 4 B.

**Architecture:**
- Group state moves inline into hash slots: `GroupSlots` in the vectorized engine, `wb_rt::GroupRows<Row>` in generated code.
- MEDIAN values move into per-group chains of 256-value blocks inside fixed 64-block chunks (`MedianStore<T>`, one per engine).
- `HashJoinOperator` refills operator-owned output vectors through two narrow mutators.
- A new `wb_mem` tool measures peak memory before and after.

**Tech Stack:** C++20 (clang, `__int128`), Makefile, `tests/test_framework.h`, branch `codegen`.

**Spec:** `docs/superpowers/specs/2026-10-21-footprint-fixes-design.md`

## Global Constraints

- Results must not change:
  - the existing reference suite (`tests/test_plan.cpp`, vectorized == reference) and differential suite (`tests/test_codegen.cpp`, codegen == vectorized) pass **unchanged**;
  - comparisons are exact: doubles compare by bit pattern, and NULL == NULL.
- Arithmetic, per-group row order, hashing (`key * 0x9E3779B97F4A7C15`, low bits), linear probing, initial capacity 1024, growth rule (`size * 2 >= capacity`), INT64_MIN held outside the slot array, and dense first-seen gids are all unchanged. They stay identical in both engines.
- MEDIAN total order: numbers ascending with `-0.0` before `+0.0`, then NaNs ordered by bit pattern. Odd n → `x[n/2]`; even n → `(lo + hi) / 2.0`; values are widened to double only after selection.
- `runtime.h` uses only `<cstdint>`/`<cstddef>`, no STL; `kAbiVersion = 4`.
- `exec/` does not include `plan/`.
- `wb_bench` checksums equal `bench/results/after-aggregates.txt`.
- Targets:

| Metric | Goal |
|---|---|
| Q7 peak | ≤ 100 MiB, both engines |
| Q8 peak | ≤ 80 MiB, both engines |
| Q4 time | ≤ ~18 ms, both engines |
| Q5/Q9 vectorized allocations | ≤ 100 per run |
| Narrow-float MEDIAN storage | 4 B per value |

  A missed target is reported with the measured number and its cause; targets are never adjusted after the fact.
- Commit each task on `codegen`, ending the message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A batch in which every live row starts a new group**, enough to force table growth during a run. Expected: `reserve` keeps every state pointer valid, and results equal the reference. Test: `vec_batch_of_all_new_groups` (Task 6).
2. **MEDIAN groups of exactly 256 and 257 values** (a chain ending exactly at a block boundary, then one value past it), end to end in both engines. Expected: correct and identical. Tests: `vec_median_block_boundaries` (Task 6) and `cg_median_block_boundaries` (Task 8).
3. **INT64_MIN as a group key combined with MEDIAN and SUM**, so the out-of-band slot carries inline state and a median chain. Expected: correct in both engines. Tests: `vec_int64_min_key_with_median` (Task 6) and `cg_int64_min_key_with_median` (Task 8).
4. **A narrow-float column's median read straight from a scan and also through a join.** The scan path uses `float` storage and the join path `double`. Expected: both identical to the reference. Tests: `vec_median_narrow_float` (Task 6) and `cg_median_narrow_float_scan_and_join` (Task 8).
5. **Several `__int128` fields in one state record** (`sum(i64 a), avg(i64 a), sum(i64 b), min(f64)`), where misalignment would be undefined behaviour. Expected: results correct, and the UBSan run is clean. Tests: `vec_many_int128_fields` (Task 6) and `cg_many_int128_fields` (Task 8).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/tools/mem_probe.cpp` | `wb_mem`: peak footprint of one query on one engine | 1 |
| `src/storage/vector.h`, `vector.cpp` | `Int64Vector::reset_plain`, `Float64Vector::reset` | 2 |
| `src/exec/hash_join.h`, `hash_join.cpp` | reusable output columns | 3 |
| `src/exec/group_slots.h`, `group_slots.cpp` | `GroupSlots` (replaces `GroupIndex`) | 4 |
| `src/exec/median_store.h` | `MedianStore<T>` (vectorized) | 5 |
| `src/exec/hashagg.h`, `hashagg.cpp` | `HashAggOperator` on `GroupSlots` + `MedianStore` | 6 |
| `src/codegen/runtime.h` | `GroupRows<Row>`, `MedianStore<T>`, ABI 4 | 7 |
| `src/codegen/generator.cpp`, `host.cpp` | generated code on `GroupRows` / `MedianStore` | 8 |
| specs, `bench/results/*` | results, doc fix | 9 |

---

### Task 1: `wb_mem` and baseline numbers

**Files:** Create `src/tools/mem_probe.cpp`. Modify `Makefile` (target `wb_mem` from `src/tools/mem_probe.cpp $(LIB_SRC)`; add it to `all`, `clean` and `-include`).

**Interfaces:**
- Produces the CLI `./wb_mem <q4|q7|q8|q7f> <vec|cg> [--csv data/readings.csv]`. It prints exactly one line:
  `<query> <engine> peak above loaded table: <MiB> MiB (<B/row> B/row)`
- Queries:
  - `q4` = `sum_count(scan(readings,{0,2}),0,1)`;
  - `q7` = `aggregate(scan(readings,{0,2}),0,{median(1)})`;
  - `q8` = `aggregate(scan(readings,{2}),∅,{median(0)})`;
  - `q7f` = `q7` on a table loaded with `value` as `narrow_float`.
- `cg` runs `prepare` before measuring. The peak comes from macOS `TASK_VM_INFO.ledger_phys_footprint_peak` minus `phys_footprint` after load and prepare. On Linux it is `/proc/self/status` VmHWM minus VmRSS.

- [ ] **Step 1: Implement** the tool. This is a measurement tool, not library code, so it has no unit test; its output is checked in Step 2.
- [ ] **Step 2: Run** `make wb_mem && for q in q4 q7 q8 q7f; do for e in vec cg; do ./wb_mem $q $e; done; done | tee bench/results/mem-before.txt`. Expected: q7 and q8 at about 380 MiB (about 40 B/row) for both engines, q4 under 5 MiB, and q7f at about 380 MiB. Note: before Task 6, MEDIAN still stores doubles.
- [ ] **Step 3: Commit** `feat(tools): wb_mem peak-memory probe and baseline`.

### Task 2: Reusable output vectors

**Files:** Modify `src/storage/vector.h`, `src/storage/vector.cpp`. Test: `tests/test_storage.cpp`.

**Interfaces:**
- Produces, both documented "for operator-owned output vectors only":
  - `int64_t* Int64Vector::reset_plain(std::size_t n);`: afterwards `width_bytes() == 8`, `base() == 0`, `size() == n`, and the code capacity is kept.
  - `double* Float64Vector::reset(std::size_t n);`: afterwards `!narrow()`, `size() == n`, and the capacity is kept.

- [ ] **Step 1: Write failing tests:**
  - `TEST(int64_reset_plain_reuses_storage)`:
    - `Int64Vector v; int64_t* p = v.reset_plain(4096);` then write `INT64_MIN, -1, 0, INT64_MAX` at 0..3;
    - check `v.value(0..3)` round-trips and `v.width_bytes() == 8`;
    - then `int64_t* q = v.reset_plain(100);` and check `q == p` and `v.size() == 100`.
  - `TEST(float64_reset_reuses_storage)`: the same pattern for `Float64Vector` with `-0.0`, NaN and `1.5`. The NaN must round-trip bit for bit, and `!v.narrow()`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error (`reset_plain` is not a member).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(storage): reusable plain output vectors`.

### Task 3: Join output reuse (F3)

**Files:** Modify `src/exec/hash_join.h`, `src/exec/hash_join.cpp`. Test: `tests/test_dist.cpp`.

**Interfaces:**
- Consumes `reset_plain`/`reset` (Task 2).
- `HashJoinOperator` drops `out_probe_builder_`/`out_build_builder_`/`out_probe_`/`out_build_`. It gains `std::vector<std::unique_ptr<Vector>> out_vecs_`, created on the first output batch from the probe and build column types, with `out_cols_` pointing at them for the operator's lifetime.

- [ ] **Step 1: Write a failing test**, `TEST(hash_join_output_columns_are_reused)`:
  - probe rows with keys `i % 100` for i < 20,000, against a build with keys 0..99 (one match each), so there are 20,000 output rows and several batches;
  - collect `&batch->column(c)` for every column of the first and second batch, and check that they are pointer-equal;
  - `collect_join(...)` must equal `nested_loop_join(...)`.
- [ ] **Step 2: Run** the suite. Expected: `hash_join_output_columns_are_reused` FAILs on pointer equality, because each batch seals a new chunk today.
- [ ] **Step 3: Implement** the in-place gather per spec §4.
- [ ] **Step 4: Run** the suite. Expected: all pass, including every `hash_join_*`, `cg_*join*` and `every_valid_strategy_computes_the_same_join` test.
- [ ] **Step 5: Commit** `perf(exec): reuse hash join output columns`.

### Task 4: `GroupSlots`

**Files:** Create `src/exec/group_slots.h`, `src/exec/group_slots.cpp`. Test: `tests/test_plan.cpp`.

**Interfaces:**
- Produces `class wb::GroupSlots` exactly as spec §2.1. It has no `plan/` dependency.
- `state_align` is 8 or 16, and each slot's state address is a multiple of it.

- [ ] **Step 1: Write failing tests:**
  - `TEST(group_slots_dense_first_seen)`: the key sequence from `group_index_dense_first_seen` gives the same gids, and `key_of(2) == INT64_MIN`.
  - `TEST(group_slots_growth_preserves_state)`:
    - state 8 B; for 5,000 keys `i*7919`, on fresh insert write `int64_t(i)` into the state;
    - after all inserts (several rehashes), `*(int64_t*)state_of(gid) == gid` for every gid, and `find_or_insert` on each key returns `fresh == false` with the same state value.
  - `TEST(group_slots_reserve_keeps_pointers)`: `reserve(size() + 4096)`, then insert 4,096 new keys while keeping every returned pointer; afterwards each pointer still equals `state_of(its gid)`.
  - `TEST(group_slots_int128_alignment)`: `GroupSlots s{48, 16}`; for 1,000 keys, `reinterpret_cast<uintptr_t>(find_or_insert(k, f)) % 16 == 0`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement.** Use 16-aligned slot storage (e.g. `std::vector` of an `alignas(16)` 16-byte unit) with a stride of `16 + round_up(state_bytes, 16)`.
- [ ] **Step 4: Run** the suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(exec): GroupSlots with inline per-group state`.

### Task 5: Vectorized `MedianStore<T>`

**Files:** Create `src/exec/median_store.h`. Test: `tests/test_plan.cpp`.

**Interfaces:**
- Produces:

```cpp
struct MedianHandle { uint32_t head = UINT32_MAX, tail = UINT32_MAX; uint32_t count = 0; };
template <class T> class MedianStore {          // T: int64_t, double or float
 public:
  void   append(MedianHandle& h, T v);
  double finalize_group(const MedianHandle& h, std::vector<T>& scratch);  // grouped; non-empty h
  double finalize_single(const MedianHandle& h);                          // global; frees chunks as copied
  std::size_t bytes_reserved() const;                                      // for tests
};
```

- Selection: `std::nth_element` plus the left max, under the total order from the Global Constraints (for `float`, the same order on float bit patterns).
- `MedianHandle` is 12 B and lives in the group's inline state. Ruling: the spec's `{head, tail}` gains `count`, so finalize knows the length without walking the chain.

- [ ] **Step 1: Write failing tests:**
  - `TEST(median_store_matches_sort)`: for each `T` in {double, float, int64}, 300 groups with random sizes 1–700 plus the exact sizes 255, 256, 257 and 513, appended interleaved. Doubles and floats include 2% NaNs with two payloads, and ±0. Each `finalize_group` equals the sorted-reference formula bit for bit.
  - `TEST(median_store_global_frees_chunks)`: append 1,000,000 doubles to one handle, note `bytes_reserved()`, call `finalize_single`, and check `bytes_reserved() == 0` afterwards and the result equals the sorted reference.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement** per spec §3: 256-value blocks and 64-block chunks behind `std::unique_ptr`, never moved.
- [ ] **Step 4: Run** the suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(exec): block-chained MedianStore`.

### Task 6: `HashAggOperator` on `GroupSlots` + `MedianStore` (F2, F1, F4 vectorized)

**Files:** Modify `src/exec/hashagg.h`, `src/exec/hashagg.cpp`. Delete `src/exec/group_index.h` and `.cpp`, and port their tests to `GroupSlots` (Task 4 already covers them; remove `group_index_*`). Test: `tests/test_plan.cpp`, `tests/test_util.h`.

**Interfaces:**
- Consumes `GroupSlots` (Task 4) and `MedianStore` (Task 5).
- State layout per spec §2.1, computed in the constructor from `AggSpec`s; a MEDIAN field is a `MedianHandle` (12 B, padded to 16).
- F4: a MEDIAN over a Float64 column whose first batch is `narrow()` uses `MedianStore<float>`, and otherwise `MedianStore<double>`. If a later batch differs in narrowness, throw `std::logic_error`; storage is per table, so it can't happen in this engine.

- [ ] **Step 1: Write failing tests** (`check_matches_reference`):
  - `TEST(vec_batch_of_all_new_groups)`: a 50,000-row kv table with distinct keys `i*2654435761 % 1'000'003`, in 4,096-row segments, with `{count, sum(1), min(1), median(1)}` grouped by key.
  - `TEST(vec_median_block_boundaries)`: groups of exactly 1, 255, 256, 257 and 512 rows.
  - `TEST(vec_int64_min_key_with_median)`: keys `{INT64_MIN, 0}`, alternating, 1,000 rows, with `{sum(1), median(1), count}`.
  - `TEST(vec_median_narrow_float)`: `make_readings(50'000, 8'000, /*narrow=*/true)`, `median(value)` grouped and global.
  - `TEST(vec_many_int128_fields)`: readings with `{sum(0), avg(0), sum(1), min(2), count}` grouped by device.
  - Add the fixtures to `tests/test_util.h` (`median_boundary_rows()`, `int64_min_median_rows()`), so Task 8 reuses them.
- [ ] **Step 2: Run** the suite. Expected: the new tests PASS on the old implementation, because they're behavioural regression guards. Record that in the ledger. RED comes from Step 3's wiring: build once with the new operator against a deliberately wrong state offset for MIN (offset + 8), and check that the suite FAILs. Then revert that.
- [ ] **Step 3: Implement** per spec §2.1 and §3: reserve per batch, state pointers in pass 1, one loop per aggregate through pointer + offset, the global register path kept, and output in gid order.
- [ ] **Step 4: Run** the suite, then `./wb_bench --iters 5` and `./wb_mem q7 vec`, `./wb_mem q8 vec`, `./wb_mem q7f vec`. Expected:
  - all tests pass;
  - `wb_bench` checksums equal `after-aggregates.txt`;
  - "groupby device 1T" at ≤ ~18 ms;
  - q7 at ≤ 100 MiB, q8 at ≤ 80 MiB, and q7f at about half of q7.
- [ ] **Step 5: Commit** `perf(exec): inline group state and block-chained MEDIAN in HashAggOperator`.

### Task 7: Runtime `GroupRows<Row>` and `MedianStore<T>` (ABI 4)

**Files:** Modify `src/codegen/runtime.h` (remove `GroupTable`, `StateRows` and `MedianBuffer`; set `kAbiVersion = 4`). Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Produces `wb_rt::GroupRows<Row>` (spec §2.2) and `wb_rt::MedianHandle` plus `wb_rt::MedianStore<T>`. The latter mirror Task 5's API with no STL:
  - `append(MedianHandle&, T)`;
  - `double finalize_group(const MedianHandle&, T* scratch)`;
  - `double finalize_single(const MedianHandle&)`;
  - `void release()`;
  - `uint32_t max_group` for sizing scratch, tracked during `append`.
  - Selection is the existing three-way quickselect plus the left max.

- [ ] **Step 1: Write failing tests**, replacing `runtime_group_table_matches_group_index`, `runtime_median_buffer_matches_sort` and `runtime_state_rows_grow`:
  - `TEST(runtime_group_rows_match_group_slots)`: the key stream from the old test → the same gid per key as `GroupSlots`, the same `size`, and `key_of`.
  - `TEST(runtime_median_store_matches_vectorized)`: the Task 5 inputs → runtime `finalize_group`/`finalize_single` equal the vectorized `MedianStore` bit for bit, for all three `T`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement.** The generator still uses the old types until Task 8. Ruling: Tasks 7 and 8 land in one commit, so the build never breaks between them. Task 7's tests are written and run first, then Task 8 follows before the commit.
- [ ] **Step 4: Run** the runtime tests (after Task 8's migration compiles).
- [ ] **Step 5: Commit**, together with Task 8.

### Task 8: Generated code on `GroupRows` / `MedianStore` (F2, F1, F4 compiled)

**Files:** Modify `src/codegen/generator.cpp`, `src/codegen/host.cpp`. Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Consumes Task 7.
- `State` holds:
  - `wb_rt::GroupRows<Row> groups`;
  - one `wb_rt::MedianStore<T> medN` per MEDIAN, with `T` = `float` when the aggregated column is a scan column whose `ColumnDef::narrow_float` is set, otherwise `int64_t` or `double`;
  - `Row` includes one `wb_rt::MedianHandle medN` per MEDIAN.
- The generator's `Ctx` `Column` gains `bool is_f32`, true only for narrow-float scan columns. Values loaded from those columns keep a `float` variable for MEDIAN appends.

- [ ] **Step 1: Write failing tests** (`check_engines_agree`):
  - `TEST(cg_median_block_boundaries)`, `TEST(cg_int64_min_key_with_median)` and `TEST(cg_many_int128_fields)`: the plans from Task 6;
  - `TEST(cg_median_narrow_float_scan_and_join)`: narrow readings `median(value)` grouped and global, plus a join of a narrow-float table on the build side, aggregating `median` of the build column.
- [ ] **Step 2: Run** the suite. Expected: the new tests PASS on the old implementation (regression guards; record that in the ledger). RED comes from Task 7's compile error and the runtime tests.
- [ ] **Step 3: Implement**, then run `./wb_codegen` for Q4 only (any quick filter of the output) and `./wb_mem q7 cg`, `q8 cg`, `q7f cg`.
- [ ] **Step 4: Run** the suite, then the sanitizer suite: `make clean && make wb_test CXX="g++ -fsanitize=address,undefined" && WB_CODEGEN_EXTRA_FLAGS="-fsanitize=address,undefined" ./wb_test | tail -1 && make clean`. Expected: `0 failed` and no sanitizer report.
- [ ] **Step 5: Commit** Tasks 7 and 8: `perf(codegen): inline group rows and block-chained MEDIAN (ABI 4)`.

### Task 9: Measure, write up, fix the doc

**Files:** Modify `bench/results/codegen.txt`; create `bench/results/mem-after.txt` and `bench/results/after-footprint.txt` (`wb_bench`); modify the footprint spec (append `## 7. Results`) and the aggregates spec §10 (the Q7 memory line).

- [ ] **Step 1: Run** `./wb_codegen | tee bench/results/codegen.txt`. Expected: every match = `yes`, exit 0.
- [ ] **Step 2: Run** `for q in q4 q7 q8 q7f; do for e in vec cg; do ./wb_mem $q $e; done; done | tee bench/results/mem-after.txt` and `./wb_bench --iters 5 | tee bench/results/after-footprint.txt`.
- [ ] **Step 3: Write the results**: one table per target with before (Task 1 / earlier measurements) and after, marked met or missed, with the measured cause for any miss. In aggregates spec §10, replace "roughly 280–300 MB transient for Q7" with "measured 382 MiB (40 B/row) before the footprint fixes; see `2026-10-21-footprint-fixes-design.md` §7".
- [ ] **Step 4: Run the full verification**: `make clean && make all wb_test && ./wb_test | tail -1`. Expected: `0 failed`.
- [ ] **Step 5: Commit** `docs: footprint fix results`.
