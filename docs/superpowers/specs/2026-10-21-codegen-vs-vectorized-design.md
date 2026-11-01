# Codegen vs vectorized: a HyPer-style compiled engine next to the vectorized one

Date: 2026-10-21
Status: design approved, spec under review

## 1. Goal

Build a second execution engine that compiles each query to native code (HyPer's data-centric
produce/consume model, as used by Redshift), run it on the same plans and data as the existing
vectorized pull engine, and measure where each model wins and why.

Success criteria:

- Both engines return **bit-identical** results for every supported plan (including sums).
- `wb_codegen` reports, per query: vectorized time, codegen generate/compile/load/exec time at
  `-O1` and `-O3`, speedup, break-even run count, allocations, checksum agreement.
- Each prediction in section 8 is confirmed or the write-up explains the deviation.

Non-goals (v1): String columns, arithmetic expressions, OR/NOT, parallel or distributed compiled
execution, SQL parsing, LLVM IR / JIT, adaptive execution (interpret-then-compile).

## 2. Decisions

| Decision | Choice | Rejected alternatives |
|---|---|---|
| Code production | Emit C++ source per query, compile with system clang into a `.dylib`, `dlopen` it (Redshift's approach) | LLVM IR + ORC JIT (more code, new dependency); compile-time templates (no compile latency to study) |
| Scope | Scan, Filter, HashJoin, grouped/global SUM+COUNT | Single-table only; distributed per-slice |
| Engine input | Shared logical plan, lowered to both engines | Mini SQL; hand-written per-query code |
| Generated-code dependencies | Self-contained `codegen/runtime.h` (C ABI structs, hash, tables) | Including engine headers (slow compiles, ABI coupling) |
| FOR width handling | Specialize pipeline bodies to the width signatures present in the bound table; per-segment switch | All 4^k combinations; per-row width dispatch |
| Filter code shape | Branches (`if (!pred) continue;`), as HyPer | Branchless (that is the vectorized engine's technique; the contrast is measured) |
| Threads | Single-threaded, both engines | Parallel compiled pipelines |

## 3. Architecture

```
                   ┌─ lower_vectorized() ─▶ Operator tree ──────────────────────────────┐
 plan::Node tree ──┤                                                                     ├─▶ QueryResult
                   └─ generate() ─▶ C++ source ─▶ compile() ─▶ wb_query() ─▶ host binds ─┘
                                                 (cached by key)
```

| Path | Responsibility | Depends on |
|---|---|---|
| `src/plan/plan.h`, `plan.cpp` | Plan nodes, builders, type checker (`output_types`), `to_string` | `storage/schema.h`, `exec/predicate.h` (CompareOp) |
| `src/plan/lower_vec.cpp` | Plan → existing operators; drain into `QueryResult` | `exec/*` |
| `src/plan/result.h` | `QueryResult`: column-major rows, canonical sort, checksum, equality | — |
| `src/codegen/runtime.h` | Compiled into generated code and included by the host: ABI structs, `kAbiVersion`, hash function, agg table, join table | `<cstdint>`, `<cstddef>` only |
| `src/codegen/generator.{h,cpp}` | Plan + width signatures → C++ source | `plan`, `runtime.h` |
| `src/codegen/compiler.{h,cpp}` | Cache, clang invocation, `dlopen`, timings | POSIX |
| `src/codegen/host.{h,cpp}` | Build `QueryInput` from tables (segment pointers, row ranges), call `wb_query`, collect `QueryResult` | `storage`, `runtime.h`, `exec/scan.h` |
| `src/tools/datagen.h` | `make_orders`, `make_customers` moved out of `dist_bench.cpp` | `storage` |
| `src/tools/codegen_bench.cpp` | `wb_codegen` experiment binary | all |
| `tests/test_codegen.cpp` | Differential, plan-checker and compiler tests | all |

Changes to existing code (small, behaviour-preserving):

- `exec/scan.h`: expose `scan_start_row(const std::vector<IndexEntry>&, const Predicate&)`
  (today the file-local `find_starting_row_`), so both engines compute identical row ranges.
- `storage/vector.h`: a read-only raw accessor on `Int64Vector` returning `{const void* codes,
  int64_t base, uint8_t width}` and on `Float64Vector` returning `{const void* values, bool
  narrow}`, for the host to bind pointers.
- `dist_bench.cpp`: use `tools/datagen.h`.
- `Makefile`: `wb_codegen` target, `-DWB_SRC_DIR="$(CURDIR)/src"`, `-ldl` if needed.

## 4. Plan IR (`src/plan/plan.h`)

```cpp
namespace wb::plan {
struct Compare  { std::size_t col; CompareOp op; std::variant<int64_t, double> literal; };

struct Node;    using NodePtr = std::unique_ptr<Node>;
struct Scan     { const Table* table; std::vector<std::size_t> cols; };
struct Filter   { NodePtr child; std::vector<Compare> conjuncts; };          // AND
struct HashJoin { NodePtr probe, build; std::size_t probe_key, build_key; };
struct SumCount { NodePtr child; std::optional<std::size_t> group_col; std::size_t sum_col; };
struct Node     { std::variant<Scan, Filter, HashJoin, SumCount> op; };

std::vector<DataType> output_types(const Node&);   // validates; throws std::invalid_argument
NodePtr scan(const Table&, std::vector<std::size_t> cols);
NodePtr filter(NodePtr child, std::vector<Compare> conjuncts);
NodePtr join(NodePtr probe, NodePtr build, std::size_t probe_key, std::size_t build_key);
NodePtr sum_count(NodePtr child, std::optional<std::size_t> group_col, std::size_t sum_col);
std::string to_string(const Node&);
}
```

Output columns:

- Scan: the projected columns in `cols` order.
- Filter: child's columns unchanged.
- HashJoin: probe columns, then build columns.
- SumCount: `[key int64,] sum float64, count int64`.

Type-checker rules (violations throw `std::invalid_argument` before generation or lowering):

1. Column references are positions in the child's output and must be in range.
2. Only Int64 and Float64 columns may appear anywhere in the plan.
3. A literal's type equals its column's type.
4. Join keys and the group column are Int64; `sum_col` is Float64.
5. The root is a `SumCount`; `SumCount` appears only at the root.
6. `Filter.conjuncts` is non-empty.

Pruning rule, identical in both engines: zone-map pruning and the sparse-index start row use
only the **first conjunct** of a `Filter` whose child is a `Scan`, translated to the table's
column index. Other conjuncts are evaluated per row.

Lowering to the vectorized engine:

- Scan + Filter → `ScanOperator(table, first conjunct translated, 4096, cols)` followed by one
  `FilterOperator` per conjunct.
- HashJoin → `HashJoinOperator(probe, build, probe_key, build_key)`.
- Grouped SumCount → `HashAggOperator(child, group_col, sum_col)`.
- Global SumCount → a drain loop in `lower_vec.cpp` summing `sum_col` over live rows.

## 5. Generator (`src/codegen/generator.cpp`)

Produce/consume over the plan. `produce(node)` emits code that makes the node's children
produce tuples; `consume(node, ctx)` emits code handling one tuple, where `ctx` maps output column
positions to in-scope C++ expressions. A `CodeWriter` handles indentation and unique names.

Pipelines break at materialization points (join build, agg). A query is a sequence of pipeline
functions called from `wb_query`, build pipelines first.

| Node | produce | consume |
|---|---|---|
| Scan | For each segment range from the host: dispatch on the segment's width signature to `pN_seg<C...>(seg, lo, hi, state)`, whose body is `for (i = lo; i < hi; ++i) { consume }` | A column is decoded on first use: `int64_t v = base + (int64_t)codes[i];` or `double v = vals[i];` (`float` widened) |
| Filter | `produce(child)` | `if (!(v OP lit)) continue;` per conjunct, in order |
| HashJoin | Build pipeline: `produce(build)` then `jt.finalize()`; then `produce(probe)` in the consuming pipeline | Build side: `jt.append(key, payload...)`. Probe side: `if (auto* b = jt.find(k)) for (j = b->first; j < b->first + b->count; ++j) { build columns = jt.col[c][j]; consume(parent) }` |
| SumCount | `produce(child)`, then emit groups via `out->emit_group` (grouped) or once (global) | Grouped: `auto& e = agg.find_or_insert(k); e.sum += v; ++e.count;` Global: locals `sum += v; ++count;` |

Width specialization: the generator receives, per scanned table, the set of width signatures
(tuple of byte widths of the Int64 columns the pipeline reads) present in its segments. Each
signature becomes one template instantiation; the dispatch switch covers exactly that set and
its default case returns an error code. The signature set is part of the cache key.

Literal emission: doubles as hex floats (`%a`); int64 as `NLL`, with `INT64_MIN` as
`(-9223372036854775807LL - 1)`.

Entry points exported by every generated library:

```cpp
extern "C" int      wb_query(const wb_rt::QueryInput* in, wb_rt::QueryOutput* out);  // 0 = ok
extern "C" uint32_t wb_abi_version();
```

No STL inside generated code; all state lives in runtime-header structs.

## 6. Runtime header (`src/codegen/runtime.h`)

- `kAbiVersion` (bumped on any struct change).
- `ColumnRef { const void* data; int64_t base; uint8_t width; uint8_t is_f32; }`.
- `SegmentRef { const ColumnRef* cols; uint32_t lo, hi; uint32_t width_sig; }`: row range
  already pruned by the host.
- `ScanInput { const SegmentRef* segs; uint32_t num_segs; }`.
- `QueryInput { const ScanInput* scans; uint32_t num_scans; }` (scan order = generator's
  pre-order numbering).
- `QueryOutput { void* ctx; void (*emit_group)(void* ctx, int64_t key, double sum, int64_t count); }`.
- `hash64(int64_t)` = the multiplicative hash used by `AggHashTable` and `HashJoinOperator`.
- `AggTable`: same algorithm as `AggHashTable`. 24-byte slots, INT64_MIN sentinel, load factor
  ≤ 0.5, power-of-two capacity, low-bit index, initial capacity equal to the vectorized
  operator's.
- `JoinTable`: same algorithm as `HashJoinOperator`. Append rows, then `finalize()` stable-sorts
  rows by key and builds a `(key, first, count)` index with top-bit hashing and load factor
  ≤ 0.5.
- Memory comes from `malloc`/`free` inside the runtime structs; the host never frees runtime
  memory.

A test asserts that `wb_rt::hash64` equals the engine's hash for a sample of keys.

## 7. Compiler and cache (`src/codegen/compiler.cpp`)

1. `key = fnv1a64(source + compiler path + flags + contents of runtime.h)`.
2. In-memory map `key → loaded library`. Hit: return the function pointers.
3. Disk: `$WB_CODEGEN_CACHE` (default `.wb_cache/`, created with mode 0700) holds
   `<key>.cpp` and `<key>.dylib`. If the `.dylib` exists, load it.
4. Miss: write `<key>.cpp`, `posix_spawn` `$WB_CXX` (default `clang++`) with
   `-std=c++20 -O<n> -march=native -fPIC -shared -I$WB_SRC_DIR $WB_CODEGEN_EXTRA_FLAGS
   -o <key>.tmp.dylib <key>.cpp`, capture stderr, then rename to `<key>.dylib`.
5. `dlopen`, `dlsym` both entry points, check `wb_abi_version() == kAbiVersion`.
6. Libraries stay loaded for the process lifetime (no `dlclose`).

`CompileStats { double generate_ms, compile_ms, load_ms; bool cache_hit; }` is returned with
every compiled query; `exec_ms` is measured by the caller.

Errors: all throw `CodegenError` with context:

- compiler missing or non-zero exit: full stderr plus the path of the kept `.cpp`;
- `dlopen` or `dlsym` failure: the `dlerror()` text;
- ABI mismatch: both version numbers;
- `wb_query` returning non-zero: a width signature missing from the compiled set (stale cache).

There is never a silent fallback to the vectorized engine.

Trust: source is produced only from in-process plans; no external text reaches the generator.

## 8. Benchmark (`wb_codegen`)

Data:

- `data/readings.csv` (10M rows, the `wb_bench` dataset) for Q1–Q4.
- `make_orders(2M)` and `make_customers(100k)` (uniform keys) for Q5.
- `--orders`, `--customers` and `--iters` flags.

| ID | Plan | Prediction | Reason |
|---|---|---|---|
| Q1 | `sum_count(filter(scan(readings,{ts,value}), ts > mid), ∅, value)` | ≈ equal | memory-bound scan in both |
| Q2 | `… filter(value > 50) …` | vectorized wins or ties | 50 % selective branch mispredicts in codegen; vectorized is branchless |
| Q3 | `… filter(value > 50 AND device_id < 5000) …` | codegen wins | no second selection vector; lazy column load |
| Q4 | `sum_count(scan(readings,{device_id,value}), device_id, value)` | codegen slightly ahead | both probe-bound; codegen avoids batch overhead |
| Q5 | `sum_count(join(filter(scan(orders), amount > 50), scan(customers), 0, 0), region, amount)` | codegen clearly ahead | vectorized join materializes every output row into chunks before aggregating |

For each query the benchmark reports:

- vectorized best and median exec time;
- for codegen at `-O1` and `-O3`: cold `generate_ms`, `compile_ms`, `load_ms` (cache bypassed),
  and warm exec best and median;
- speedup = vectorized / codegen warm best;
- break-even runs = `compile_ms / (vec_ms − cg_ms)`, or ∞ when codegen is not faster;
- allocations per run (operator-new counter, as in `wb_bench`);
- `match` = whether the two QueryResults are equal.

Output goes to stdout and `bench/results/codegen.txt`. A short results section is appended to
this spec after the first run, covering each prediction.

## 9. Testing (`tests/test_codegen.cpp`, existing `TEST` framework)

Differential: each plan runs on both engines and the `QueryResult`s must be **exactly equal**,
because both engines accumulate in the same row order. Cases:

- each of Q1–Q5 on small generated tables;
- an empty table, every segment pruned, and a single row;
- keys at INT64_MIN and INT64_MAX, negative keys, 8-byte FOR columns, and segments with mixed
  width signatures;
- a `narrow_float` column;
- a filter on the sort key (sparse-index path) with Gt, Ge, Eq, Lt and Ne;
- a join with duplicate keys on both sides, a join with no matches, and an empty build side.

Plan checker: rejects a String column, a literal of the wrong type, a non-root or missing
`SumCount`, an out-of-range column, and empty conjuncts.

Compiler:

- the same plan compiled twice hits the cache;
- `WB_CXX=/nonexistent` throws a `CodegenError` naming the compiler;
- a changed runtime-header hash changes the key;
- `wb_rt::hash64` matches the engine's hash.

Tests use a fresh temporary cache directory per run. For sanitizer runs,
`WB_CODEGEN_EXTRA_FLAGS=-fsanitize=address,undefined` builds the generated libraries with the
same sanitizers as the host.

## 10. Risks

- **Compile latency dominates small queries.** That is a result to report, not a bug; the
  break-even column makes it visible.
- **Exact-equality tests are brittle if one engine changes accumulation order.** Accepted on
  purpose: any order change should be a deliberate, visible decision.
- **The runtime tables duplicate engine algorithms.** The hash test and differential tests catch
  drift; the duplication is limited to `runtime.h`.
- **macOS-specific `.dylib` naming.** The compiler module selects the extension per platform
  (`.so` on Linux).

## 11. Results (2026-10-21)

Setup:

- Apple M3 Pro, Apple clang 21, single-threaded, best of 7 runs.
- Data: readings 10M rows (153 segments); orders 2M; customers 100k.
- Raw output: `bench/results/codegen.txt`.
- Every query matched bit for bit across all three engines.

| Query | vec ms | cg-O1 ms | cg-O3 ms | cg-O3 speedup | Prediction | Verdict |
|---|---|---|---|---|---|---|
| Q1 ts > mid | 5.2 | 4.4 | 5.3 | 0.99× (1.0–1.5× across runs) | ≈ equal | **confirmed** |
| Q2 value > 50 | 6.8 | 12.8 | 13.7 | 0.49× | vectorized wins or ties | **confirmed**, for a different reason |
| Q3 value > 50 AND device_id < 5000 | 7.2 | 33.7 | 35.3 | 0.20× | codegen wins | **refuted** |
| Q4 group by device_id | 17.7 | 20.2 | 17.9 | 0.99× | codegen slightly ahead | **refuted** (tie at -O3) |
| Q5 join + group by region | 23.1 | 13.4 | 12.4 | 1.86× | codegen clearly ahead | **confirmed** |

One-off cost per cold query:

- generate under 0.1 ms;
- compile 45–93 ms;
- load 155–230 ms.

Loading costs more than compiling. The first `dlopen` of a newly written `.dylib` takes about
155 ms; re-opening the same cached file in a later process takes 0.25–0.87 ms (measured). That is
consistent with macOS checking each new library on first load, but the cause is not confirmed. Break-even (runs needed to
repay generate + compile + load) is 23–24 runs for Q5 and 263 runs for Q1 at -O1; it never
arrives for Q2–Q4.

**Q1, roughly equal.** Pruning leaves about 5M rows in both engines, and both run one predictable
compare per row (ts is sorted) followed by a serial floating-point add. Run-to-run noise
(±1 ms) is larger than the gap between the engines.

**Q2, vectorized wins: the cause is if-conversion, not misprediction.** In the assembly of the
generated loop, clang removed the branch:

```
fcmp; cinc; fcsel
```

Every one of the 10M rows therefore runs an `fadd` that feeds the next row, a serial dependency
chain at about 4 cycles per add (≈ 11 ms). The vectorized engine first writes a branchless
selection vector, then adds only the 5M survivors.

**Q3, the largest miss.** The generated loop keeps a real `b.le` on `value > 50`, about 50%
unpredictable. The reason is HyPer-style lazy loading: `device_id` is decoded only after the first
predicate passes, so the compiler will not hoist that load above the branch, and the loop cannot be
if-converted. A throwaway probe isolated the first predicate as the cost:

| Variant | vec ms | cg ms |
|---|---|---|
| Q3 as written | 7.9 | 37.0 |
| first predicate always true | 12.1 | 13.7 |
| second predicate always true | 8.8 | 35.3 |
| both always true | 13.8 | 13.7 |

The vectorized engine's branchless selection vectors pay off exactly where Kersten et al. found
vectorization wins: selective predicates on unsorted data.

**Q4, a tie.** Both engines are bound by hash-table probes (the same AggTable algorithm, 10k groups,
Zipf-skewed keys). Codegen's per-row decode and update cost about the same as the vectorized
engine's batched loop over borrowed spans. -O1 is 14% slower than -O3 here.

**Q5, codegen wins 1.86×.** The vectorized HashJoinOperator materializes every output row into
owned chunks before aggregating: 31k allocations per run versus 37. The generated code keeps the
probe value in a register and reads only the build column it needs (region) straight from the join
table into the agg update.

**What to try next.** Each item comes with the measurement that would confirm or refute it.

- An eager-load mode in the generator, emitting branch-free predicate evaluation, aimed at Q2 and
  Q3. Accepting reassociated float sums would allow SIMD reductions, but breaks the bit-identical
  guarantee.
- Relaxing the exact-order requirement for the comparison, to measure what it costs.
- Code-signing or pre-warming generated libraries, to cut `load_ms`.
