# Codegen vs Vectorized Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a HyPer-style compiled execution engine (generate C++ → clang → dlopen) that runs the same logical plans as the existing vectorized engine, and a benchmark comparing them.

**Architecture:**
- One logical plan IR (`src/plan`) is lowered either to the existing vectorized operators or to generated C++.
- Generated C++ uses produce/consume over a self-contained `codegen/runtime.h`, with one fused loop per pipeline, specialized to the FOR width signatures present in the table.
- A compiler module caches libraries by source hash, and a host module binds table segments into C-ABI structs and calls `wb_query`.

**Tech Stack:** C++20, Apple clang / clang++, POSIX (`posix_spawn`, `dlopen`), existing Makefile, existing `tests/test_framework.h` (`TEST`, `CHECK`, `CHECK_EQ`).

**Spec:** `docs/superpowers/specs/2026-10-21-codegen-vs-vectorized-design.md`

## Global Constraints

- Both engines must return **bit-identical** `QueryResult`s, sums included. Compare doubles with `std::bit_cast<uint64_t>`, never with a tolerance.
- Supported column types: Int64 and Float64 only. String anywhere in a plan → `std::invalid_argument`.
- The root is `SumCount`, and `SumCount` appears only at the root.
- Pruning: zone map plus sparse-index start row, using only the **first conjunct** of a `Filter` whose child is a `Scan`, in both engines.
- Generated code includes only `codegen/runtime.h`, and `runtime.h` includes only `<cstdint>` and `<cstddef>`. No STL in generated code or the runtime.
- Exported symbols: `extern "C" int wb_query(const wb_rt::QueryInput*, wb_rt::QueryOutput*)` (0 = ok) and `extern "C" uint32_t wb_abi_version()`.
- Runtime algorithms match the engine:
  - Hash: `static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL`.
  - `AggTable`: 24 B slots, `INT64_MIN` empty sentinel, load factor ≤ 0.5 (grow when `size*2 >= capacity`), low-bit index, initial capacity 1024.
  - `JoinTable`: rows stable-sorted by key; `(key, first, count)` index; top-bit index; load factor ≤ 0.5.
- Compile command: `$WB_CXX` (default `clang++`) `-std=c++20 -O<n> -march=native -fPIC -shared -I$WB_SRC_DIR $WB_CODEGEN_EXTRA_FLAGS -o <key>.tmp.<ext> <key>.cpp`, then `rename` to `<key>.<ext>`. `<ext>` is `dylib` on macOS and `so` elsewhere.
- Cache key: FNV-1a 64 over `source + compiler path + flags + runtime.h contents`. The cache dir is `$WB_CODEGEN_CACHE` (default `.wb_cache/`), mode 0700.
- Literals: doubles via `%a`; `INT64_MIN` as `(-9223372036854775807LL - 1)`; other int64 values as `<n>LL`.
- All codegen failures throw `wb::codegen::CodegenError`. There is no silent fallback to vectorized.
- Global `SumCount` always yields exactly one row, `(sum 0.0, count 0)` on empty input. Grouped `SumCount` on empty input yields zero rows.
- Both engines are single-threaded.
- Commits: one per task, on a branch the user picks before execution starts. The spec is still uncommitted; commit it with Task 1.

## Review Focus

1. **Projected filter column ≠ table column.** For `scan(t, {2, 1})` with a filter on plan column 0, pruning must use table column 2. Tests: `prune_uses_table_column_through_projection` (Task 4) and `cg_prune_uses_table_column_through_projection` (Task 7).
2. **Narrow float compared with a double literal** (`value > 50.1` on a `narrow_float` column). Both engines widen to double before comparing. Test: `narrow_float_filter_matches` (Task 7).
3. **Empty input.** A global agg returns exactly one `(0, 0)` row; a grouped agg returns zero rows; a join with an empty build side returns zero rows. Tests: `empty_table_*` (Tasks 4, 7, 8).
4. **Filters on both join inputs.** The scan numbering used by the generator must match the host's `QueryInput.scans` order. Test: `join_with_filters_on_both_sides` (Task 8).
5. **Corrupt or partial `.dylib` already in the cache dir.** This must raise `CodegenError` carrying the `dlerror()` text, not crash. Test: `corrupt_cached_library_raises` (Task 6).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/exec/scan.h`, `scan.cpp` | expose `scan_start_row` | 1 |
| `src/storage/vector.h`, `vector.cpp` | `Int64Vector::raw()`, `Float64Vector::raw()` | 1 |
| `src/tools/datagen.h` | `make_orders`, `make_customers` (moved from `dist_bench.cpp`) | 1 |
| `src/plan/result.h`, `result.cpp` | `QueryResult` | 2 |
| `src/plan/plan.h`, `plan.cpp` | IR, builders, type checker, `to_string` | 3 |
| `src/plan/lower_vec.h`, `lower_vec.cpp` | `run_vectorized` | 4 |
| `src/codegen/runtime.h` | ABI structs, hash, `AggTable`, `JoinTable` | 5 |
| `src/codegen/compiler.h`, `compiler.cpp` | cache, clang, dlopen, `CodegenError` | 6 |
| `src/codegen/generator.h`, `generator.cpp` | plan → C++ | 7, 8 |
| `src/codegen/host.h`, `host.cpp` | bind tables, call, collect; `run_compiled` | 7, 8 |
| `src/tools/codegen_bench.cpp` | `wb_codegen` | 9 |
| `tests/test_plan.cpp` | Tasks 2–4 tests | 2–4 |
| `tests/test_codegen.cpp` | Tasks 5–8 tests | 5–8 |
| `Makefile` | `src/plan`, `src/codegen` in `LIB_SRC`; `-DWB_SRC_DIR`; `wb_codegen`; `-ldl` on Linux | 1, 6, 9 |

---

### Task 1: Small engine hooks + shared data generator

**Files:**
- Modify: `src/exec/scan.h`, `src/exec/scan.cpp` (rename file-local `find_starting_row_` → public `scan_start_row`)
- Modify: `src/storage/vector.h`, `src/storage/vector.cpp`
- Create: `src/tools/datagen.h`; Modify: `src/tools/dist_bench.cpp` (use it, delete local copies)
- Modify: `Makefile` (`LIB_SRC += $(wildcard src/plan/*.cpp) $(wildcard src/codegen/*.cpp)`)
- Test: `tests/test_storage.cpp`

**Interfaces:**
- Produces:
  - `std::size_t wb::scan_start_row(const std::vector<IndexEntry>& idx, const Predicate& pred);` with behaviour identical to today's `find_starting_row_`.
  - `struct Int64Raw { const void* codes; int64_t base; std::uint8_t width; }; Int64Raw Int64Vector::raw() const;` (`width` ∈ {1, 2, 4, 8}).
  - `struct Float64Raw { const void* values; bool narrow; }; Float64Raw Float64Vector::raw() const;`
  - `std::unique_ptr<Table> make_orders(std::int64_t rows, std::int64_t customers, double exponent);` and `std::unique_ptr<Table> make_customers(std::int64_t rows);`, both `inline` in `src/tools/datagen.h`, byte-for-byte the same output as the current `dist_bench.cpp` versions.

- [ ] **Step 1: Write failing tests** in `tests/test_storage.cpp`:
  - `TEST(raw_accessors)`:
    - `Int64Vector v{{-5, 70'000}}` → `v.raw().width == 4`, `v.raw().base == -5`, `((const uint32_t*)v.raw().codes)[1] == 70'005`.
    - `Float64Vector f{{1.5, 2.5}, true}` → `f.raw().narrow` and `((const float*)f.raw().values)[1] == 2.5f`.
  - `TEST(scan_start_row_public)`: for an index `{{100, 0}, {200, 1024}, {300, 2048}}`, `Gt 250 → 1024`, `Eq 100 → 0`, `Lt 250 → 0`, `Ge 99 → 0`.
- [ ] **Step 2: Run** `make wb_test && ./wb_test | grep -E "raw_accessors|scan_start_row"`. Expected: a compile error (symbols not declared).
- [ ] **Step 3: Implement** the three hooks and `datagen.h`. Make `ScanOperator::next` call `scan_start_row`.
- [ ] **Step 4: Run** `make all wb_test && ./wb_test | tail -1 && ./wb_dist --experiment strategies | tail -1`. Expected: `0 failed`, and `all strategies returned the same result: yes`.
- [ ] **Step 5: Commit** the spec plus Task 1 files: `git commit -m "feat: expose scan_start_row, raw column accessors, shared datagen"`

### Task 2: QueryResult

**Files:** Create: `src/plan/result.h`, `src/plan/result.cpp`. Test: Create `tests/test_plan.cpp`.

**Interfaces:**
- Produces, in `namespace wb::plan`:

```cpp
struct QueryResult {
    bool                 grouped = false;
    std::vector<int64_t> keys;    // empty when !grouped
    std::vector<double>  sums;
    std::vector<int64_t> counts;

    void          canonicalize();                         // sort rows by key (grouped only)
    bool          operator==(const QueryResult&) const;   // exact; doubles via std::bit_cast<uint64_t>
    std::uint64_t checksum() const;                       // FNV-1a over canonical rows' bits
    std::string   diff(const QueryResult& other) const;   // first differing row, for test messages
};
```

- [ ] **Step 1: Write failing tests:**
  - `TEST(result_equality_is_bitwise)`: two results with sums `0.1+0.2` and `0.3` are `!=`, and their `diff()` names row 0.
  - `TEST(result_canonicalize_sorts_by_key)`: keys `{3, 1, 2}` with sums and counts permuted alongside, after `canonicalize()`, become keys `{1, 2, 3}` with sums and counts following their keys.
  - `TEST(result_checksum_order_independent)`: two copies with different row order have equal `checksum()` after `canonicalize()`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `./wb_test | grep result_`. Expected: 3 PASS.
- [ ] **Step 5: Commit** `feat(plan): QueryResult with exact comparison`.

### Task 3: Plan IR and type checker

**Files:** Create: `src/plan/plan.h`, `src/plan/plan.cpp`. Test: `tests/test_plan.cpp`.

**Interfaces:**
- Consumes: `CompareOp` (`exec/predicate.h`), `Table`, `DataType`.
- Produces: exactly the declarations in spec §4. In addition:
  - `void validate_query(const Node& root);` enforces rule 5 (root `SumCount`, none below it) and calls `output_types`.
  - `output_types` enforces rules 1–4 and 6.
  - Both throw `std::invalid_argument` with a message naming the node and the column.
  - `to_string` produces a one-line form, e.g. `SumCount(group=0,sum=1, Filter([0>5000], Scan(readings,[0,2])))`.

- [ ] **Step 1: Write failing tests:**
  - `TEST(plan_output_types)`: on a readings-shaped table (`device_id` Int64, `ts` Int64, `value` Float64):
    - `scan(t,{1,2})` → `{Int64, Float64}`;
    - `join(scan(a,{0,1}), scan(b,{0,1}), 0, 0)` → 4 types, probe first;
    - grouped `sum_count` → `{Int64, Float64, Int64}`; global → `{Float64, Int64}`.
  - `TEST(plan_rejects_invalid)`: each of the following throws `std::invalid_argument`:
    - a String column in a scan;
    - `Compare{0, Gt, 1.0}` on an Int64 column;
    - a Float64 group column;
    - an Int64 sum column;
    - column index 9;
    - empty conjuncts;
    - `validate_query` on a root `Filter`;
    - a `SumCount` nested under a `Filter`.
  - `TEST(plan_to_string)`: assert the exact string above for `sum_count(filter(scan(t,{0,2}),{{0,Gt,int64_t{5000}}}),0,1)`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `./wb_test | grep plan_`. Expected: 3 PASS.
- [ ] **Step 5: Commit** `feat(plan): logical plan IR with type checker`.

### Task 4: Vectorized lowering

**Files:** Create: `src/plan/lower_vec.h`, `src/plan/lower_vec.cpp`. Test: `tests/test_plan.cpp`.

**Interfaces:**
- Consumes: Tasks 2–3, `ScanOperator`, `FilterOperator`, `HashJoinOperator`, `HashAggOperator`.
- Produces: `QueryResult run_vectorized(const Node& root);` (calls `validate_query`, returns a canonicalized result).
- Lowering rules from spec §4. For the pruning predicate, translate the first conjunct's plan column `p` to `scan.cols[p]`, giving `Predicate{scan.cols[p], op, literal}`.
- Global `SumCount` drains with `for_each_live_row` and adds in row order.

- [ ] **Step 1: Write failing tests.** Each compares against a brute-force loop over the raw rows in a test helper. Use exact equality: the brute force adds in the same order, segment by segment and row by row after the sort.
  - `TEST(vec_global_filter_sum)`;
  - `TEST(vec_grouped_sum)`;
  - `TEST(vec_two_conjuncts)`;
  - `TEST(vec_join_grouped)`: probe 3,000 rows with keys `i%400`, build 900 rows with keys `200+i%300`, group by a build column;
  - `TEST(prune_uses_table_column_through_projection)`: `scan(t,{2,1})`, filter `[0 > x]` on the Float64 `value`, with a table where pruning on table column 0 would wrongly drop segments; the result must equal brute force;
  - `TEST(empty_table_vec)`: global → one row `(0.0, 0)`; grouped → zero rows.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `./wb_test | grep -E "vec_|prune_|empty_table_vec"`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat(plan): lower plans to vectorized operators`.

### Task 5: Runtime header

**Files:** Create: `src/codegen/runtime.h`. Test: Create `tests/test_codegen.cpp`.

**Interfaces:**
- Produces (`namespace wb_rt`, everything `inline`, no STL):

```cpp
constexpr uint32_t kAbiVersion = 1;
struct ColumnRef  { const void* data; int64_t base; uint8_t width; uint8_t is_f32; };
struct SegmentRef { const ColumnRef* cols; uint32_t lo, hi; uint32_t width_sig; };
struct ScanInput  { const SegmentRef* segs; uint32_t num_segs; };
struct QueryInput { const ScanInput* scans; uint32_t num_scans; };
struct QueryOutput { void* ctx; void (*emit_group)(void*, int64_t key, double sum, int64_t count); };

uint64_t hash64(int64_t key);
uint32_t width_code(uint8_t width);           // 1→0, 2→1, 4→2, 8→3
// width_sig = OR of width_code(col_k) << (2*k) over the scan's Int64 columns in projection order

struct AggSlot  { int64_t key; double sum; int64_t count; };
struct AggTable { AggSlot* slots; size_t size, capacity;
                  void init(size_t initial = 1024); AggSlot& find_or_insert(int64_t key);
                  template <class F> void for_each(F&& f) const; void release(); };

struct JoinBucket { int64_t key; uint32_t first, count; };
struct JoinTable  { /* ncols payload columns as uint64 bit patterns */
                    void init(uint32_t ncols); void append(int64_t key, const uint64_t* payload);
                    void finalize(); const JoinBucket* find(int64_t key) const;
                    uint64_t col(uint32_t c, uint32_t row) const; void release(); };
```

- `JoinTable::finalize`: write a **stable** merge sort over a row-index array (no `<algorithm>`). Then build buckets with `capacity = bit_ceil(max(2, 2*distinct))` and index `hash64(k) >> (64 - log2(capacity))`, with linear probing.
- `col(c, row)` indexes the sorted order. Payload doubles are stored via `__builtin_memcpy` bit copies.

- [ ] **Step 1: Write failing tests:**
  - `TEST(runtime_hash_matches_engine)`: over 1,000 keys including `INT64_MIN` and `INT64_MAX`, `hash64(k) == uint64_t(k) * 0x9E3779B97F4A7C15ULL`.
  - `TEST(runtime_agg_matches_engine)`: 50,000 inserts over 3,000 keys; `AggTable` and `AggHashTable` give identical sums (bitwise) and counts per key, and the same final capacity.
  - `TEST(runtime_join_matches_engine)`: build rows with duplicate keys; for each probe key, the sequence of `col(1, j)` over `[first, first+count)` equals the build rows with that key in original insertion order; a missing key → `find == nullptr`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `./wb_test | grep runtime_`. Expected: 3 PASS.
- [ ] **Step 5: Commit** `feat(codegen): self-contained runtime header`.

### Task 6: Compiler and cache

**Files:** Create: `src/codegen/compiler.h`, `src/codegen/compiler.cpp`. Modify: `Makefile` (add `-DWB_SRC_DIR=\"$(CURDIR)/src\"` to `CXXFLAGS`; add `-ldl` when `uname` ≠ Darwin). Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Consumes: `wb_rt::kAbiVersion`, `QueryInput`, `QueryOutput`.
- Produces (`namespace wb::codegen`):

```cpp
struct CodegenError : std::runtime_error { using std::runtime_error::runtime_error; };
struct CompileOptions { int opt_level = 3; bool use_cache = true; };
struct CompileStats { double generate_ms = 0, compile_ms = 0, load_ms = 0; bool cache_hit = false; };
using QueryFn = int (*)(const wb_rt::QueryInput*, wb_rt::QueryOutput*);
struct CompiledQuery { QueryFn fn; CompileStats stats; std::uint64_t key; };

std::uint64_t cache_key(const std::string& source, const CompileOptions&);
std::uint64_t cache_key(const std::string& source, const CompileOptions&,
                        const std::string& runtime_text);   // test seam; the 2-arg form reads runtime.h
CompiledQuery compile_source(const std::string& source, const CompileOptions&);
std::size_t   compiler_invocations();   // process-wide count, for tests
```

- `use_cache=false` skips both caches and compiles into `<cache>/<key>.<pid>.<counter>.<ext>`, so cold runs never reuse a library.
- Environment variables are read on every call.
- Compiler failure: the message contains the compiler path, its exit status, stderr, and the kept `.cpp` path.

- [ ] **Step 1: Write failing tests.** Each runs with `setenv("WB_CODEGEN_CACHE", <fresh mkdtemp dir>, 1)`. Use a hand-written source string that includes `codegen/runtime.h` and defines both exports, with `wb_query` calling `out->emit_group(out->ctx, 7, 1.5, 2)`.
  - `TEST(compile_and_call)`: `fn` runs and emits `(7, 1.5, 2)`.
  - `TEST(compile_cache_hit)`: the second `compile_source` → `stats.cache_hit`, and `compiler_invocations()` is unchanged.
  - `TEST(compile_missing_compiler_raises)`: `WB_CXX=/nonexistent` → `CodegenError` whose `what()` contains `/nonexistent`.
  - `TEST(compile_abi_mismatch_raises)`: a source where `wb_abi_version` returns 999 → `CodegenError` containing `999`.
  - `TEST(corrupt_cached_library_raises)`: write garbage to `<dir>/<cache_key(src)>.<ext>` → `CodegenError` with non-empty `dlerror()` text.
  - `TEST(cache_key_depends_on_opt_level)`: `-O1` and `-O3` give different keys.
  - `TEST(cache_key_depends_on_runtime_header)`: `cache_key(src, opts, "a") != cache_key(src, opts, "b")`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement**, using `posix_spawnp` with stderr redirected to `<key>.err` via `posix_spawn_file_actions_addopen`.
- [ ] **Step 4: Run** `./wb_test | grep -E "compile_|corrupt_|cache_key"`. Expected: 7 PASS.
- [ ] **Step 5: Commit** `feat(codegen): clang compiler with on-disk cache`.

### Task 7: Generator and host for single-table plans

**Files:** Create: `src/codegen/generator.h`, `src/codegen/generator.cpp`, `src/codegen/host.h`, `src/codegen/host.cpp`. Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces (`namespace wb::codegen`):

```cpp
struct ScanBinding { const plan::Scan* scan; std::vector<std::size_t> int_cols; // plan positions of Int64 cols
                     std::vector<std::uint32_t> width_sigs; };                  // distinct, sorted
std::vector<ScanBinding> bind_scans(const plan::Node& root);   // pre-order; computes width_sigs from segments
std::string generate(const plan::Node& root, const std::vector<ScanBinding>& scans);
plan::QueryResult run_compiled(const plan::Node& root, const CompileOptions& = {}, CompileStats* stats = nullptr);
```

- Pre-order numbering: a `HashJoin` visits `build` before `probe`. Generator and host share `bind_scans`, so the order can't drift.
- Host pruning: apply the Global Constraints rule using `zone_map_can_match` and `scan_start_row` on the **table** column (`scan.cols[p]`). A segment that survives pruning gets `SegmentRef{cols, start_row, num_rows, width_sig}`.
- `run_compiled` order: `validate_query`, `bind_scans`, `generate` (timed as `generate_ms`), `compile_source`, build refs, call `fn`, then a non-zero return → `CodegenError("width signature … not compiled")`; finally collect via the `emit_group` callback and `canonicalize`.
- In this task, `generate` throws `CodegenError("HashJoin not supported yet")` on a join (removed in Task 8).

- [ ] **Step 1: Write failing tests.** Every test asserts `run_compiled(plan) == run_vectorized(plan)` and prints `diff()` on failure.
  - `TEST(cg_q1_q4_shapes)`: the Q1–Q4 plans from spec §8 on a 200k-row readings-shaped table (generated as in `tests/test_storage.cpp::random_rows`, 8,000 rows per segment, sorted by `ts`).
  - `TEST(cg_sort_key_ops)`: a filter on `ts` with each of Gt, Ge, Eq, Lt, Ne.
  - `TEST(cg_extreme_keys)`: group keys including `INT64_MIN`, `INT64_MAX` and negatives in one segment (8-byte width), alongside other segments of width 2.
  - `TEST(cg_all_pruned)`: `ts > max_ts` → one row `(0.0, 0)`.
  - `TEST(narrow_float_filter_matches)`: a `narrow_float` value column, filter `value > 50.1`.
  - `TEST(cg_prune_uses_table_column_through_projection)`: the same plan as Task 4's test, run compiled.
  - `TEST(empty_table_cg)`: global and grouped on an empty table.
  - `TEST(cg_rejects_join_for_now)`: a join plan → `CodegenError`.
- [ ] **Step 2: Run** `make wb_test`. Expected: a compile error.
- [ ] **Step 3: Implement the generator.**
  - A `CodeWriter` holds an indent level, a fresh-name counter and the output string.
  - `produce`/`consume` follow spec §5.
  - Each scan pipeline becomes `template <class C0, …> static void pK_seg(const wb_rt::SegmentRef&, State&)`, with one template parameter per entry of `int_cols`.
  - `wb_query` switches on `width_sig` over `width_sigs`, with `default: return 1;`.
  - Literal formatting follows the Global Constraints.
- [ ] **Step 4: Implement the host** (`bind_scans`, ref building, `run_compiled`).
- [ ] **Step 5: Run** `./wb_test | grep -E "cg_|narrow_float_filter|empty_table_cg"`. Expected: all PASS.
- [ ] **Step 6: Commit** `feat(codegen): produce/consume generator for scan/filter/agg`.

### Task 8: HashJoin codegen

**Files:** Modify: `src/codegen/generator.cpp`, `src/codegen/host.cpp` (only if binding needs changes). Test: `tests/test_codegen.cpp`.

**Interfaces:**
- Consumes: `JoinTable` (Task 5), `generate` and `bind_scans` (Task 7).
- Produces: `generate` supports `HashJoin` per spec §5.
  - The build pipeline runs first and ends with `jt.finalize()`.
  - Build columns reach `consume(parent)` as expressions `jt.col(c, j)` (bit-cast to `double` for Float64 columns).
  - `jt.release()` and `agg.release()` run before `wb_query` returns.

- [ ] **Step 1: Write failing tests.** Each asserts `run_compiled == run_vectorized`.
  - `TEST(cg_q5_join_agg)`: Q5 on `make_orders(200'000, 10'000, 1.0)` and `make_customers(10'000)`.
  - `TEST(cg_join_duplicates_both_sides)`: the data from `hash_join_matches_a_nested_loop_reference`, grouped by probe key.
  - `TEST(cg_join_no_matches)`: disjoint key ranges → grouped gives zero rows, global gives `(0, 0)`.
  - `TEST(empty_table_cg_join)`: an empty build side.
  - `TEST(join_with_filters_on_both_sides)`: `Filter` on both the probe scan and the build scan, with different first-conjunct columns.
  - Delete `cg_rejects_join_for_now`.
- [ ] **Step 2: Run** `make wb_test && ./wb_test | grep -E "cg_q5|cg_join|join_with_filters|empty_table_cg_join"`. Expected: FAIL with `HashJoin not supported yet`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `./wb_test | tail -1`. Expected: `0 failed, 0 todo`.
- [ ] **Step 5: Run the sanitizer suite**: `make clean && make wb_test CXX="g++ -fsanitize=address,undefined" && WB_CODEGEN_EXTRA_FLAGS="-fsanitize=address,undefined" ./wb_test | tail -1 && make clean`. Expected: `0 failed` and no sanitizer report.
- [ ] **Step 6: Commit** `feat(codegen): hash join pipelines`.

### Task 9: `wb_codegen` benchmark and results

**Files:** Create: `src/tools/codegen_bench.cpp`. Modify: `Makefile` (target `wb_codegen` from `src/tools/codegen_bench.cpp $(LIB_SRC)`; add to `all`, `clean`, `-include`). Modify: spec (append "Results"). Create: `bench/results/codegen.txt`.

**Interfaces:**
- Consumes: `run_vectorized`, `run_compiled`, `compile_source`, `datagen.h`, the operator-new counter pattern from `src/tools/bench.cpp`.
- CLI: `./wb_codegen [--csv data/readings.csv] [--orders 2000000] [--customers 100000] [--iters 7]`.
- One output row per (query, engine), with the columns `query | engine (vec, cg-O1, cg-O3) | gen ms | compile ms | load ms | exec best ms | exec median ms | speedup | break-even runs | allocs/run | match`.
  - `break-even = compile_ms / (vec_best − cg_best)`, printed as `inf` when the denominator is ≤ 0.
  - Cold stats come from one `use_cache=false` compile; exec timings come from warm runs.
- Exit code 1 if any `match` is false.

- [ ] **Step 1: Implement** the binary with the Q1–Q5 plans from spec §8.
- [ ] **Step 2: Run** `make wb_codegen && ./wb_codegen | tee bench/results/codegen.txt`. Expected: 15 rows (5 queries × 3 engines), every `match` = `yes`, exit 0.
- [ ] **Step 3: Append `## 11. Results`** to the spec: the table, then one paragraph per prediction stating whether it was confirmed and the measured reason (selectivity, materialized bytes, compile cost).
- [ ] **Step 4: Run the full verification**: `make clean && make all wb_test && ./wb_test | tail -1 && ./wb_bench --iters 3 | tail -5`. Expected: `0 failed`, and `wb_bench` checksums unchanged from `bench/results/after-f64.txt`.
- [ ] **Step 5: Commit** `feat: wb_codegen benchmark and results`.
