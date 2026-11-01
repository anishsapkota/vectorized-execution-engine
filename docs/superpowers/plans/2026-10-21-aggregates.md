# Aggregates (COUNT/SUM/MIN/MAX/AVG/MEDIAN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SumCount` with general aggregates (COUNT, SUM, MIN, MAX, AVG, exact MEDIAN), running on both the vectorized and the compiled engine with bit-identical results, and benchmark them as Q6–Q9.

**Architecture:**
- A column-generic `QueryResult` and an `Aggregate` plan node come first.
- The vectorized engine gets a `GroupIndex` (key → dense gid) and struct-of-arrays state, with one loop per aggregate per batch.
- The runtime gets a matching `GroupTable`, `StateRows<Row>` and `MedianBuffer<T>`. The generator emits a per-query `Row` struct and fused per-row updates.
- MEDIAN collects `(gid, value)` pairs, then runs a counting sort by gid and a per-group selection.

**Tech Stack:** C++20 (clang, `__int128`), existing Makefile and `tests/test_framework.h`, the codegen engine on branch `codegen`.

**Spec:** `docs/superpowers/specs/2026-10-21-aggregates-design.md`. It builds on `docs/superpowers/specs/2026-10-21-codegen-vs-vectorized-design.md`.

## Global Constraints

- Results: the generated code must equal the vectorized engine **exactly**, and the vectorized engine must equal the reference interpreter in `tests/test_plan.cpp` exactly. Doubles compare by bit pattern; NULL == NULL.
- Row order: each group sees its rows in scan order. A join emits probe-row-major output, with matches in build insertion order.
- SUM:
  - the state starts at `+0.0` (Float64) or `0` (`__int128`), and **every** row adds, including the group's first;
  - a SUM(Int64) result outside `[INT64_MIN, INT64_MAX]` → `std::overflow_error`, whose message contains the group key, or `global`.
- MIN/MAX: the state starts at the group's first value. Updates are `if (v < m || v != v) m = v;` (MIN) and `if (v > m || v != v) m = v;` (MAX).
- AVG: Int64 → `static_cast<double>(sum128) / static_cast<double>(count)`; Float64 → `sum / static_cast<double>(count)`.
- MEDIAN:
  - odd n → `static_cast<double>(x[n/2])`;
  - even n → `(static_cast<double>(x[n/2-1]) + static_cast<double>(x[n/2])) / 2.0`;
  - values are ordered by a total order: numbers ascending with `-0.0` before `+0.0`, then NaNs ordered by bit pattern.
- Result types:
  - Count → Int64;
  - Sum, Min, Max → same as the input column;
  - Avg, Median → Float64.
- Empty input: grouped → 0 rows; global → 1 row with COUNT = 0 and SUM/MIN/MAX/AVG/MEDIAN NULL.
- Group hashing: `key * 0x9E3779B97F4A7C15`, low-bit index, linear probing, initial capacity 1024, grow when `size*2 >= capacity`, INT64_MIN held outside the slot array, gids dense in first-seen order.
- `runtime.h`: `kAbiVersion = 3`; still only `<cstdint>`/`<cstddef>`, no STL.
- `exec/` must not include `plan/`.
- `AggHashTable` and `parallel_agg` are unchanged.
- `wb_bench` checksums stay equal to `bench/results/after-f64.txt`, except the group-by total, which changes with group emission order (ruled in Task 4; see spec §10).
- Commit each task on branch `codegen`, ending the message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **One column feeding several aggregates**, e.g. `min(v), max(v), median(v)`, plus two MEDIANs on different columns. Expected: distinct state fields and buffers, all correct. Test: `cg_many_aggs_same_and_different_columns` (Task 8).
2. **The group column also aggregated**: `aggregate(scan(t,{0,1}), 0, {min(0), max(0), median(0)})`. Expected: correct in both engines. Tests: `vec_group_column_aggregated` (Task 5) and `cg_group_column_aggregated` (Task 8).
3. **A global aggregate whose every segment is pruned** (the table isn't empty, but no rows survive). Expected: one row, `count = 0`, the rest NULL, in both engines. Test: `cg_global_all_pruned_is_null` (Task 7).
4. **An even-count median whose middle pair includes NaN.** Expected: NaN, identical in both engines. Tests: `vec_median_nan_middle` (Task 5) and covered again in `cg_median_matrix` (Task 8).
5. **An aggregate list with only COUNT** (no value column at all), grouped and global. Expected: works; the generated `Row` holds just `count`. Test: `cg_count_only` (Task 7).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/plan/result.h`, `result.cpp` | `ResultColumn`, column-generic `QueryResult` | 1 |
| `src/plan/plan.h`, `plan.cpp` | `AggFn`, `AggExpr`, `Aggregate` (replaces `SumCount`), builders, type checker | 2 |
| `src/plan/lower_vec.cpp` | lowering of `Aggregate`; NULL row for empty global | 2, 4, 5 |
| `src/exec/agg_spec.h` | `AggKind`, `AggSpec` | 4 |
| `src/exec/group_index.h`, `group_index.cpp` | `GroupIndex` | 3 |
| `src/exec/hashagg.h`, `hashagg.cpp` | general `HashAggOperator` | 4, 5 |
| `src/codegen/runtime.h` | `GroupTable`, `StateRows`, `MedianBuffer`, `emit_row`, ABI 3 | 6 |
| `src/codegen/generator.cpp`, `host.cpp`, `host.h` | aggregate codegen and result decoding | 2, 7, 8 |
| `src/tools/codegen_bench.cpp` | Q6–Q9, generic result columns | 1, 9 |
| `tests/test_plan.cpp` | result, IR and reference-interpreter tests; vectorized vs reference | 1–5 |
| `tests/test_codegen.cpp` | runtime and differential tests | 1, 2, 6–8 |
| `tests/test_util.h` | shared helpers if needed | any |

---

### Task 1: Column-generic QueryResult

**Files:** Modify `src/plan/result.h`, `src/plan/result.cpp`, `src/plan/lower_vec.cpp`, `src/codegen/host.cpp`, `src/tools/codegen_bench.cpp`, `tests/test_plan.cpp`, `tests/test_codegen.cpp`.

**Interfaces:**
- Produces (`namespace wb::plan`): `ResultColumn` and `QueryResult` exactly as spec §4, plus:
  - `void ResultColumn::push_int(int64_t v, bool valid = true);`
  - `void ResultColumn::push_double(double v, bool valid = true);`
  - `std::size_t ResultColumn::size() const;`
- Behaviour stays as today: `SumCount` results become columns `[key Int64,] sum Float64, count Int64`. A global aggregate on empty input is still `(0.0, 0)`; Task 2 changes that.
- Equality:
  - a type mismatch, a row-count mismatch or a column-count mismatch makes results unequal;
  - NULL cells compare equal whatever bytes they hold;
  - `diff` names the column index and the row.

- [ ] **Step 1: Rewrite the three `result_*` tests** in `tests/test_plan.cpp` against the new shape, and add:
  - `TEST(result_nulls_compare_equal_regardless_of_bytes)`: two Float64 columns holding NULL over `1.0` and NULL over `2.0` → equal; NULL against `1.0` → unequal.
  - `TEST(result_diff_names_column)`: a difference in column 2, row 0 → `diff()` contains `"col 2"` and `"row 0"`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error (`ResultColumn` unknown).
- [ ] **Step 3: Implement** `result.h`/`result.cpp`. Migrate every producer and consumer: `lower_vec.cpp`, `host.cpp` (`collect_group`), `codegen_bench.cpp`, and the test helpers (`reference()`, `check_*`, the `.counts[0]`/`.num_rows()` uses) to `int_at`/`double_at`/`is_null`.
- [ ] **Step 4: Run** the suite with `make wb_test && ./wb_test > LOG; echo $?; tail -1 LOG`. Expected: exit 0, `65 passed, 0 failed`.
- [ ] **Step 5: Commit** `refactor(plan): column-generic QueryResult with NULLs`.

### Task 2: Aggregate plan node, reference interpreter, NULL-on-empty

**Files:** Modify `src/plan/plan.h`, `src/plan/plan.cpp`, `src/plan/lower_vec.cpp`, `src/codegen/generator.cpp`, `src/codegen/generator.h`, `src/codegen/host.cpp`, `tests/test_plan.cpp`, `tests/test_codegen.cpp`.

**Interfaces:**
- Produces: the spec §2 declarations (`AggFn`, `AggExpr`, `Aggregate`, `aggregate()`, `sum_count()` shorthand), and `std::vector<DataType> output_types(const Node&)` with the new rules 1–4. `validate_query` requires an `Aggregate` root.
- `to_string`: `Aggregate(group=0,[sum(1),count], <child>)`, with lowercase function names and `group=none` for a global aggregate.
- Temporary limit until Tasks 4 and 7: both engines accept only aggregate lists made of `Sum` over Float64 and `Count`, in any order and repeated. Anything else throws:
  - `std::invalid_argument("aggregate <fn> not supported yet")` from `run_vectorized`;
  - `CodegenError("codegen: aggregate <fn> not supported yet")` from the generator.
- Both engines now return **NULL SUM** for a global aggregate over empty input.
  - `run_vectorized` builds the row in its global drain path.
  - `execute` in the host builds it when `wb_query` emitted no row for a global query.
- **The reference interpreter** (`reference()` in `tests/test_plan.cpp`) implements every function exactly as the spec says:
  - MEDIAN via `std::sort` with a NaN-last comparator over a copy of the group's values;
  - SUM(Int64) in `__int128`, throwing `std::overflow_error` with the key;
  - NULL semantics for an empty global aggregate.

- [ ] **Step 1: Write failing tests:**
  - `TEST(plan_aggregate_rejects_invalid)`: each case below throws `std::invalid_argument`:
    - `Count` with `col = 0`;
    - `Min` with no column;
    - column index 9;
    - empty `aggs`;
    - a Float64 group column;
    - an `Aggregate` under a `Filter`.
  - `TEST(plan_aggregate_output_types)`: `aggregate(scan(t,{0,2}), 0, {{Min,0},{Max,1},{Avg,0},{Median,1},{Sum,0},{Sum,1},{Count,{}}})` → `{Int64, Int64, Float64, Float64, Float64, Int64, Float64, Int64}`.
  - `TEST(plan_aggregate_to_string)`: the exact string `"Aggregate(group=0,[sum(1),count], Scan(readings,[0,2]))"` for `sum_count(scan(t,{0,2}),0,1)`.
  - `TEST(reference_median_and_nulls)`: on a hand-built table with values `{3, 1, NaN, 2}` in one group, `reference()` of median gives `2.5`. Global median over an empty table → one row, NULL median, count 0.
  - Change `empty_table_vec` and `empty_table_cg`: a global SUM is now NULL (`is_null(0,0)`), count 0.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error (`aggregate` / `AggFn` undeclared).
- [ ] **Step 3: Implement.** Generator and host pattern-match the `Aggregate` node where they used to match `SumCount`.
- [ ] **Step 4: Run** the suite. Expected: exit 0, all pass. `wb_codegen` Q1–Q5 are not run yet (that's Task 9).
- [ ] **Step 5: Commit** `feat(plan): Aggregate node with SQL NULL on empty global input`.

### Task 3: GroupIndex

**Files:** Create `src/exec/group_index.h`, `src/exec/group_index.cpp`. Test: `tests/test_plan.cpp`.

**Interfaces:**
- Produces: `class wb::GroupIndex { explicit GroupIndex(std::size_t initial_capacity = 1024); uint32_t find_or_insert(int64_t key); uint32_t size() const; int64_t key_of(uint32_t gid) const; std::size_t capacity() const; }`, with the algorithm from Global Constraints.

- [ ] **Step 1: Write failing tests:**
  - `TEST(group_index_dense_first_seen)`: keys `7, -3, 7, INT64_MIN, -3, 0` → gids `0, 1, 0, 2, 1, 3`; `size() == 4`; `key_of(2) == INT64_MIN`.
  - `TEST(group_index_growth_keeps_ids)`: insert 5,000 distinct keys `i*7919`, then look each up again; every gid equals its insertion index, and `capacity() >= 2*size()`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement.** Keep a `std::vector<int64_t> keys_` (gid → key) alongside the slot array of `{key, gid}`.
- [ ] **Step 4: Run** the suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(exec): GroupIndex with dense first-seen group ids`.

### Task 4: General HashAggOperator (all but MEDIAN)

**Files:** Create `src/exec/agg_spec.h`. Modify `src/exec/hashagg.h`, `src/exec/hashagg.cpp`, `src/plan/lower_vec.cpp`. Test: `tests/test_plan.cpp`.

**Interfaces:**
- Consumes: `GroupIndex` (Task 3).
- Produces:
  - `enum class AggKind : uint8_t { Count, Sum, Min, Max, Avg, Median }; struct AggSpec { AggKind kind; std::size_t col; DataType type; };`
  - `HashAggOperator(std::unique_ptr<Operator> child, std::optional<std::size_t> group_col, std::vector<AggSpec> aggs);`
  - The existing 3-argument constructor stays, forwarding to `{Sum Float64 on sum_col, Count}`.
  - Output batch columns follow spec §5.
- Behaviour:
  - Global aggregation uses gid 0, created on the first row.
  - An empty global aggregate emits **no** rows; `run_vectorized` builds the NULL row.
  - MEDIAN throws `std::invalid_argument("aggregate median not supported yet")` until Task 5.
- `lower_vec.cpp`: maps `AggExpr` to `AggSpec` using the child's `output_types`. Delete the Task 2 limit and the separate global drain loop.

- [ ] **Step 1: Write failing tests.** Each one is `check_matches_reference(plan)` (exact).
  - `TEST(vec_all_decomposable_aggs)`: `make_readings(50'000)`, grouped by device and global, with `{count, sum(value), sum(device_id), min(value), max(value), min(ts), max(ts), avg(value), avg(device_id)}`.
  - `TEST(vec_min_max_nan)`: one Float64 group each where NaN is the first value, a middle value, and every value. MIN and MAX return NaN in all three, and match the reference.
  - `TEST(vec_sum_negative_zero)`: a group of only `-0.0` values → the SUM bit pattern equals `+0.0`, and matches the reference.
  - `TEST(vec_sum_int128_intermediate)`: Int64 values `INT64_MAX, INT64_MAX, -INT64_MAX, -INT64_MAX + 5` → SUM 5 (exact).
  - `TEST(vec_sum_int64_overflow_throws)`: values `INT64_MAX, 1` in group key 42 → `std::overflow_error` whose `what()` contains `"42"`. A global aggregate → the message contains `"global"`.
  - `TEST(vec_aggs_above_join)`: the `vec_join_grouped` tables with `{min(1), max(1), avg(1), count}` grouped by column 3.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error (`AggSpec` undeclared).
- [ ] **Step 3: Implement** per spec §5: struct-of-arrays state, one loop per aggregate per batch, dispatch once per batch, and the overflow check while producing output.
- [ ] **Step 4: Run** the suite, then `./wb_bench --iters 3`. Expected: all tests pass, and the `wb_bench` checksums equal `bench/results/after-f64.txt` (the forwarding constructor path).
- [ ] **Step 5: Commit** `feat(exec): HashAggOperator with COUNT/SUM/MIN/MAX/AVG`.

### Task 5: Vectorized MEDIAN

**Files:** Modify `src/exec/hashagg.h`, `src/exec/hashagg.cpp`. Test: `tests/test_plan.cpp`.

**Interfaces:**
- Produces: `AggKind::Median` support.
  - Per-aggregate `std::vector<uint32_t> gids` and `std::vector<T> values`.
  - The end-of-input finalize is the counting sort plus `std::nth_element` with the comparator `less_nan_last(a, b) = !isnan(a) && (isnan(b) || a < b)`; for Int64 it is plain `<`.

- [ ] **Step 1: Write failing tests** (`check_matches_reference`):
  - `TEST(vec_median_odd_even_single)`: groups with 1, 2, 7 and 10 values, Int64 and Float64.
  - `TEST(vec_median_int64_extremes)`: one group `{INT64_MIN, INT64_MAX}` → `(double(INT64_MIN) + double(INT64_MAX)) / 2.0`.
  - `TEST(vec_median_nan_middle)`: group `{1.0, NaN, 3.0, NaN}` → NaN. Group `{1.0, 2.0, NaN}` → `2.0`.
  - `TEST(vec_median_grouped_and_global_readings)`: `make_readings(50'000)`: `median(value)` grouped by device, and a global `median(value)` behind a `ts > mid` filter.
  - `TEST(vec_group_column_aggregated)`: `aggregate(scan(t,{0,2}), 0, {{Min,0},{Max,0},{Median,0},{Count,{}}})`.
- [ ] **Step 2: Run** the suite. Expected: FAIL with `aggregate median not supported yet`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(exec): exact MEDIAN in HashAggOperator`.

### Task 6: Runtime GroupTable, StateRows, MedianBuffer, emit_row

**Files:** Modify `src/codegen/runtime.h` (remove `AggTable`; `QueryOutput` becomes `emit_row`; `kAbiVersion = 3`), `src/codegen/generator.cpp` and `src/codegen/host.cpp` (switch the existing sum/count emission to `GroupTable` + `StateRows` + `emit_row`). Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Produces (`namespace wb_rt`):
  - `struct GroupTable { void init(size_t initial = 1024); uint32_t find_or_insert(int64_t key); uint32_t size() const; int64_t key_of(uint32_t gid) const; void release(); };`
  - `template <class Row> struct StateRows { Row* data = nullptr; uint32_t size = 0, capacity = 0; Row& push(const Row& r); void release(); };`
  - `template <class T> struct MedianBuffer { void append(uint32_t gid, T v); void finalize(const int64_t* counts, uint32_t groups, double* out); void release(); };`, using its own quickselect with NaN ordered last.
  - `struct QueryOutput { void* ctx; void (*emit_row)(void* ctx, const uint64_t* values, uint32_t n); int64_t error_key; uint8_t error_global; };`: `wb_query` sets `error_key` or `error_global` before returning 3 (SUM overflow).
- Replace the tests `runtime_agg_matches_engine` and `runtime_agg_int64_min_key` with the ones below. The existing compile tests' hand-written source must call `emit_row`.

- [ ] **Step 1: Write failing tests:**
  - `TEST(runtime_group_table_matches_group_index)`: 50,000 keys from the LCG in the old agg test, including `INT64_MIN` → identical gid sequence, `size`, and `key_of` for every gid.
  - `TEST(runtime_median_buffer_matches_sort)`: 300 groups with random sizes 1–60, Float64 with 2% NaN and an Int64 variant. `finalize` output equals, bit for bit, a reference that sorts each group with NaN last and applies the spec formula.
  - `TEST(runtime_state_rows_grow)`: push 10,000 `struct R { int64_t a; double b; }` → `size == 10'000`, and values read back intact.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement the runtime.** Then migrate the generator's existing grouped `SumCount` path to `GroupTable` + `StateRows<Row>` + `emit_row`, and the host's collector to decode `emit_row` by output type.
- [ ] **Step 4: Run** the suite. Expected: all pass, including every existing `cg_*` test.
- [ ] **Step 5: Commit** `feat(codegen): runtime group table, state rows, median buffer (ABI 3)`.

### Task 7: Codegen COUNT/SUM/MIN/MAX/AVG

**Files:** Modify `src/codegen/generator.cpp`, `src/codegen/host.cpp`. Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Consumes: Task 6 runtime; `output_types` (Task 2).
- Produces: generator support for every non-median `AggFn`, per spec §6.
  - Row fields are named `<fn><agg index>` (e.g. `min0`, `sum1`). SUM(Int64) and AVG(Int64) use `__int128` fields.
  - The global path uses per-segment locals named `acc_<fn><index>`, with MIN/MAX initialised under `if (acc_count == 0)`.
  - On SUM(Int64) overflow, `wb_query` sets `QueryOutput::error_key` or `error_global` (Task 6) and returns 3. The host throws `std::overflow_error` naming the key, or `global`.
  - MEDIAN still throws `CodegenError("codegen: aggregate median not supported yet")`.

- [ ] **Step 1: Write failing tests** (`check_engines_agree`):
  - `TEST(cg_all_decomposable_aggs)`: the same plans as `vec_all_decomposable_aggs`.
  - `TEST(cg_min_max_nan_and_negative_zero)`: the `vec_min_max_nan` and `vec_sum_negative_zero` tables.
  - `TEST(cg_sum_int128_and_overflow)`: the intermediate-overflow case is equal in both engines. In the overflow case both throw `std::overflow_error` containing `"42"`, and `"global"` for a global aggregate.
  - `TEST(cg_aggs_above_join)`: the plan from `vec_aggs_above_join`.
  - `TEST(cg_global_all_pruned_is_null)`: `make_readings(50'000)`, filter `ts > max_ts`, `{count, sum(value), min(value), avg(value)}` → equal in both engines; `int_at(0,0) == 0`; columns 1–3 NULL.
  - `TEST(cg_count_only)`: grouped `{count}` and global `{count}` on readings.
- [ ] **Step 2: Run** the suite. Expected: FAIL with `aggregate min not supported yet` (or similar).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite. Expected: all pass.
- [ ] **Step 5: Commit** `feat(codegen): COUNT/SUM/MIN/MAX/AVG codegen`.

### Task 8: Codegen MEDIAN

**Files:** Modify `src/codegen/generator.cpp`. Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Consumes: `MedianBuffer<T>` (Task 6).
- Produces: `State` gains `wb_rt::MedianBuffer<T> med<agg index>` per MEDIAN aggregate. `wb_query` finalizes them into `double` arrays after all pipelines run, and frees them before returning.

- [ ] **Step 1: Write failing tests** (`check_engines_agree`):
  - `TEST(cg_median_matrix)`: all `vec_median_*` tables and plans, grouped and global.
  - `TEST(cg_many_aggs_same_and_different_columns)`: readings grouped by device, `{min(value), max(value), median(value), median(ts), avg(ts), count}`.
  - `TEST(cg_group_column_aggregated)`: the `vec_group_column_aggregated` plan.
  - `TEST(cg_median_above_join)`: Q9's shape on `make_orders(200'000, 10'000, 1.0)` and `make_customers(10'000)`.
- [ ] **Step 2: Run** the suite. Expected: FAIL with `aggregate median not supported yet`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suite, then the sanitizer suite: `make clean && make wb_test CXX="g++ -fsanitize=address,undefined" && WB_CODEGEN_EXTRA_FLAGS="-fsanitize=address,undefined" ./wb_test | tail -1 && make clean`. Expected: `0 failed`, no sanitizer report.
- [ ] **Step 5: Commit** `feat(codegen): exact MEDIAN codegen`.

### Task 9: Q6–Q9 benchmark and results

**Files:** Modify `src/tools/codegen_bench.cpp`, `bench/results/codegen.txt`, and the spec (append `## 10. Results`).

**Interfaces:**
- Consumes: everything above.
- The Q6–Q9 plans are exactly as in spec §7. The output format is unchanged, with Q6–Q9 rows after Q5.

- [ ] **Step 1: Add Q6–Q9** to `codegen_bench.cpp`.
- [ ] **Step 2: Run** `make wb_codegen && ./wb_codegen | tee bench/results/codegen.txt`. Expected: 27 rows (9 queries × 3 engines), every `match` = `yes`, exit 0.
- [ ] **Step 3: Append the results** to the aggregates spec: a table, then one paragraph per prediction (confirmed or refuted, with the measured cause), including MEDIAN memory per row (`gid + value` bytes).
- [ ] **Step 4: Run the full verification**: `make clean && make all wb_test && ./wb_test | tail -1 && ./wb_bench --iters 3 | tail -5`. Expected: `0 failed`, and checksums unchanged.
- [ ] **Step 5: Commit** `feat: aggregate benchmark queries Q6-Q9 and results`.
