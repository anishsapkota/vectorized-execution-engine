# Aggregates: COUNT, SUM, MIN, MAX, AVG, MEDIAN in both engines

Date: 2026-10-21
Status: design approved, spec under review
Builds on: `2026-10-21-codegen-vs-vectorized-design.md` (the "codegen spec")

## 1. Goal

Replace the fixed `SumCount` aggregate with general aggregates, running on the vectorized engine
and on generated code, with results that stay **bit-identical** between the engines and exactly
equal to a reference interpreter. Extend `wb_codegen` with four aggregate-heavy queries
(Q6–Q9) and report how the two execution models compare on them.

Success criteria:

- Every aggregate, on both input types, grouped and global, matches the reference interpreter
  (vectorized engine) and the vectorized engine (generated code) exactly.
- Q1–Q5 results are unchanged. `wb_bench` checksums are unchanged except the group-by total,
  whose last digits change with group emission order (per-group values are unchanged; see §10).
- `wb_codegen` reports Q6–Q9 with `match = yes`; the spec gains a results section covering the
  predictions in section 7.

Non-goals: approximate quantiles, other percentiles, DISTINCT aggregates, NULL inputs (the
storage layer has none), aggregates below the plan root, parallel aggregation
(`parallel_agg` and `AggHashTable` are unchanged).

## 2. Plan IR

`plan::SumCount` is replaced by:

```cpp
enum class AggFn : uint8_t { Count, Sum, Min, Max, Avg, Median };
struct AggExpr   { AggFn fn; std::optional<std::size_t> col; };  // col empty only for COUNT(*)
struct Aggregate { NodePtr child; std::optional<std::size_t> group_col; std::vector<AggExpr> aggs; };

NodePtr aggregate(NodePtr child, std::optional<std::size_t> group_col, std::vector<AggExpr> aggs);
NodePtr sum_count(NodePtr child, std::optional<std::size_t> group_col, std::size_t sum_col);
        // shorthand for aggregate(child, group_col, {{Sum, sum_col}, {Count, {}}})
```

Output columns: `[group key Int64,]` then one column per `AggExpr`, in order.

`to_string` form: `Aggregate(group=0,[min(1),avg(1),count], <child>)`; `group=none` when global.

Type checker rules (violations throw `std::invalid_argument`), replacing the SumCount rules:

1. `aggs` is non-empty.
2. `Count` has no `col`; every other function has one, in range.
3. Aggregated columns and the group column are Int64 or Float64 (already the only legal plan
   types); the group column is Int64.
4. `Aggregate` is the root and appears only there.

Result types:

| Function | Int64 input | Float64 input |
|---|---|---|
| Count | Int64 | Int64 |
| Sum | Int64: accumulated in `__int128`; a final value outside int64 is an error | Float64: sequential `double` sum |
| Min, Max | Int64 | Float64 |
| Avg | Float64: `static_cast<double>(sum128) / static_cast<double>(count)` | Float64: `sum / static_cast<double>(count)` |
| Median | Float64 | Float64 |

## 3. Semantics both engines must implement identically

- **Row order.** Each group sees its rows in scan order (segments in table order, rows
  ascending, join output probe-row-major with matches in build insertion order).
- **SUM.** The state starts at `+0.0` (Float64) or `0` (`__int128`) and **every** row adds,
  including the group's first row. (`0.0 + -0.0 == +0.0`; both engines therefore return `+0.0`
  for a sum of `-0.0` values.)
- **MIN / MAX.** The state starts at the group's first value. Every later value updates with
  `if (v < m || v != v) m = v;` for MIN and `if (v > m || v != v) m = v;` for MAX. Once
  any input is NaN the result is NaN. Int64 uses the same expressions (`v != v` is false).
- **MEDIAN.** Exact. For a group with `n` values sorted ascending as `x[0..n-1]`:
  - `n` odd: `static_cast<double>(x[n/2])`;
  - `n` even: `(static_cast<double>(x[n/2 - 1]) + static_cast<double>(x[n/2])) / 2.0`.

  Float64 values are ordered by one **total** order, used by both engines' selection and by the
  reference sort: numbers ascending with `-0.0` before `+0.0`, then every NaN, NaNs ordered by
  bit pattern. Under `<` alone signed zeros and NaN payloads compare equal, so which of them
  lands in the middle would depend on the selection algorithm (an even-count median of
  `{-0, -0, +0, +0}` could be `-0` or `+0`). With the total order the result is the same for any
  correct selection algorithm, so the engines may use different ones.
- **SUM(Int64) overflow.** If a group's final `__int128` sum is outside
  `[INT64_MIN, INT64_MAX]`, the query throws `std::overflow_error` whose message contains the
  group key (or `global`). Intermediate values may exceed int64. AVG(Int64) never overflows.
- **Empty input.** Grouped: zero rows. Global: exactly one row; COUNT = 0 and SUM, MIN, MAX,
  AVG and MEDIAN are **NULL**. (This changes the previous global SUM on empty input from `0.0`
  to NULL.)

## 4. Results (`src/plan/result.h`)

```cpp
struct ResultColumn {
    DataType             type;     // Int64 or Float64
    std::vector<int64_t> ints;     // used when type == Int64
    std::vector<double>  doubles;  // used when type == Float64
    std::vector<uint8_t> valid;    // 0 = NULL; same length as the value vector
};

struct QueryResult {
    bool                      grouped = false;
    std::vector<ResultColumn> columns;  // key column first when grouped
    std::size_t num_rows() const;
    void        canonicalize();                    // sort rows by key (grouped only)
    bool        operator==(const QueryResult&) const;  // exact; doubles by bit pattern;
                                                       // NULL == NULL regardless of stored bytes
    std::uint64_t checksum() const;
    std::string   diff(const QueryResult&) const;
    // Test accessors
    int64_t int_at(std::size_t col, std::size_t row) const;
    double  double_at(std::size_t col, std::size_t row) const;
    bool    is_null(std::size_t col, std::size_t row) const;
};
```

## 5. Vectorized engine

New in `src/exec/`:

- `agg_spec.h`: `enum class AggKind { Count, Sum, Min, Max, Avg, Median };
  struct AggSpec { AggKind kind; std::size_t col; DataType type; };` (`col`/`type` ignored for
  Count). No dependency on `plan/`.
- `group_index.h/.cpp`: `GroupIndex` maps int64 key → dense `uint32_t` group id, in first-seen
  order. Same hash (`key * 0x9E3779B97F4A7C15`, low bits), linear probing, initial capacity 1024,
  load factor ≤ 0.5 (grow when `size*2 >= capacity`), INT64_MIN held outside the slot array.
  API: `uint32_t find_or_insert(int64_t key)`, `uint32_t size() const`,
  `int64_t key_of(uint32_t gid) const`.

`HashAggOperator` becomes
`HashAggOperator(std::unique_ptr<Operator> child, std::optional<std::size_t> group_col,
std::vector<AggSpec> aggs)`. The old `(child, group_col, sum_col)` constructor remains and
forwards to `{Sum Float64, Count}`.

- Per batch: compute `gids` for the live rows (global: all `0`, group created on first row);
  append state for new groups; then one loop per aggregate over `(gids, values)`, dispatching on
  kind and type once per batch.
- State: struct of arrays, one array per aggregate (`vector<double>`, `vector<__int128>`,
  `vector<int64_t>` or `vector<double>` for MIN/MAX) plus one shared `counts` array.
- MEDIAN: per aggregate, `vector<uint32_t> gids` and `vector<T> values` appended per row. At end
  of input: counting sort by gid using `counts`, then `std::nth_element` per group slice with
  the comparator `less_nan_last`, and for even `n` the maximum of the left part.
- Output batches: `[key,]` then one `Int64Vector`/`Float64Vector` per aggregate; one row per
  existing group. SUM(Int64) overflow is checked while producing output.

`plan::run_vectorized` lowers every `Aggregate` to `HashAggOperator`, and for a global aggregate
with no output row builds the NULL row described in section 3.

## 6. Compiled engine

`runtime.h` (`kAbiVersion = 3`):

- `GroupTable`: same algorithm and API as `GroupIndex` (`find_or_insert`, `size`, `key_of`,
  `init`, `release`). Replaces `AggTable`.
- `template <class Row> struct StateRows { Row* data; uint32_t size, capacity; Row& push(const
  Row&); void release(); }`: malloc-backed, doubling.
- `template <class T> struct MedianBuffer { void append(uint32_t gid, T v);
  void finalize(const int64_t* counts, uint32_t groups, double* out); void release(); }`:
  counting sort by gid, then the runtime's own quickselect with NaN ordered last.
- `QueryOutput { void* ctx; void (*emit_row)(void* ctx, const uint64_t* values, uint32_t n); }`:
  one 64-bit pattern per output column (int64 as is, double via bit cast).

Generator, per query:

- Emits `struct Row { int64_t count; <one field per Sum/Min/Max/Avg, typed per section 2> }`.
  Avg shares the Sum representation; Count reads `count`.
- Grouped consume, per tuple: `g = st.groups.find_or_insert(k)`; when `g == st.rows.size`, push a
  `Row` with `count = 0`, sums `0`, and MIN/MAX set to the current row's values; then one update
  per aggregate per section 3, and `st.medN.append(g, v)` per MEDIAN.
- Global consume: state in locals per segment (prologue/epilogue, as today), MIN/MAX initialised
  under `if (acc_count == 0)`; MEDIAN appends gid 0.
- After all pipelines: finalize every `MedianBuffer`, then emit one row per gid in gid order. A
  SUM(Int64) outside int64 makes `wb_query` return 3 (the host throws `std::overflow_error` naming
  the key). A global query with no rows emits nothing; the host builds the NULL row.

Host: `execute` decodes `emit_row` values by the plan's output types, sets `valid = 1`, and
handles the empty global case as in section 3.

## 7. Benchmark (`wb_codegen`)

| ID | Plan | Prediction | Reason |
|---|---|---|---|
| Q6 | `aggregate(scan(readings,{0,2}), 0, {min(1), max(1), avg(1), count})` | codegen wins | one fused pass and one state record per row vs four per-aggregate loops per batch |
| Q7 | `aggregate(scan(readings,{0,2}), 0, {median(1)})` | ≈ equal | dominated by end-of-query selection, same complexity in both |
| Q8 | `aggregate(filter(scan(readings,{1,2}), ts > mid), ∅, {median(1)})` | ≈ equal | one ~5M-value selection dominates |
| Q9 | `aggregate(join(filter(scan(orders,{0,1}), amount > 50), scan(customers,{0,1}), 0, 0), 3, {avg(1), median(1)})` | codegen wins | join output materialization (as Q5) |

Same columns and rules as the codegen spec section 8. Results are appended to this spec.

## 8. Testing

- **Reference interpreter** (`tests/test_plan.cpp`) extended to every aggregate: MEDIAN by a full
  sort with NaN last, SUM(Int64) in `__int128`. The vectorized engine must equal it exactly.
- **Differential** (`tests/test_codegen.cpp`): generated code must equal the vectorized engine
  exactly for every aggregate × {Int64, Float64} × {grouped, global}, plus:
  - NaN as the first value, a middle value, and every value of a group (MIN/MAX/MEDIAN);
  - `-0.0` inputs to SUM;
  - odd and even median counts, single-row groups;
  - median over INT64_MIN and INT64_MAX;
  - INT64_MIN group key;
  - empty global (COUNT 0, others NULL) and empty grouped (no rows);
  - SUM(Int64) whose running sum leaves int64 but whose final sum fits (correct value);
  - SUM(Int64) overflow (`std::overflow_error` from both engines, message names the key);
  - aggregates above a join.
- **Plan checker:** rejects Count with a column, other functions without one, an out-of-range
  column, empty `aggs`, a Float64 group key, an aggregate below the root.
- **Runtime:** `GroupTable` assigns the same gids in the same order as `GroupIndex` over a random
  key stream; `MedianBuffer::finalize` equals a sorted reference over random groups (with NaN).
- **Regression:** Q1–Q5 match; `wb_bench` checksums equal `bench/results/after-f64.txt` except
  the group-by total (emission order, §10); per-group sums are pinned by
  `vec_group_sums_match_agg_hash_table`.
- **Sanitizers:** the ASan+UBSan run including generated libraries.

## 9. Risks

- `__int128` is a GCC/Clang extension. Accepted: both engines are built only with clang here.
- MEDIAN keeps every value: 8–12 B per input row per MEDIAN aggregate (gid + value). Accepted
  and stated in the results; an approximate median is a non-goal.
- Changing global SUM on empty input from `0.0` to NULL changes existing results; the affected
  tests are updated deliberately.

## 10. Results (2026-10-21)

Setup:

- Apple M3 Pro, single-threaded, best of 7 runs.
- Data: readings 10M rows; orders 2M rows; customers 100k rows.
- Raw output: `bench/results/codegen.txt`.
- All 18 compiled rows (9 queries × `-O1`/`-O3`) matched the vectorized result bit for bit.

| Query | vec ms | cg-O3 ms | speedup | Prediction | Verdict |
|---|---|---|---|---|---|
| Q6 min/max/avg/count by device | 38.1 | 32.5 | 1.17× | codegen wins | **confirmed**, modestly |
| Q7 median by device | 149.3 | 136.4 | 1.10× | ≈ equal | **confirmed** |
| Q8 global median, ts > mid | 67.3 | 64.1 | 1.05× | ≈ equal | **confirmed** |
| Q9 join + avg/median by region | 33.7 | 23.1 | 1.46× | codegen wins | **confirmed** |

**Q6.** The vectorized engine makes four passes per batch over (gid, value): one per
aggregate after the gid pass. The generated loop updates one `Row` per tuple. The gain is
smaller than the join's (Q5/Q9), most likely because both engines pay the same group-table probe
per row (inferred, not profiled).

**Q7 and Q8.** Most of the time is MEDIAN-specific. Q7 takes about 136–149 ms against about
26 ms for a grouped SUM/COUNT over the same rows (Q4), so roughly 80% goes to collecting values
and the finalize step (inferred from that difference, not profiled). That
step is a counting sort plus a per-group selection, with the same complexity in both engines
(`std::nth_element` in one, the runtime's three-way quickselect in the other), so the execution
model barely matters.

MEDIAN memory for 10M rows (computed, not measured):

- payload: 4 B gid plus 8 B value collected per input row, plus 8 B per row for the bucketed
  copy at finalize, so 20 B per row;
- buffers grow by doubling, so capacity rounds 10M up to about 16.8M, and old and new buffers
  coexist during the last grow. The realistic peak is about 28–30 B per row, roughly
  280–300 MB transient for Q7.

**Q9.** The same cause as Q5: the vectorized join materializes every output row into chunks
(31k allocations per run, against 72) before the aggregate sees it.

**Effect on the earlier queries** (codegen spec §11):

- **Q1–Q3:** unchanged. The first cut of the generalized operator ran global aggregates through
  per-row group-id arrays and was 2× slower (Q1 vec 12.7 ms). Global aggregates now accumulate
  in a local per batch, as the old drain loop did.
- **Q4:** now about 25–28 ms in **both** engines (across `-O1`/`-O3`), against about 17.7 ms
  before. Both replaced a
  key → {sum, count} table with dense group ids plus separate per-group state: one more
  dependent memory access per row. Q4 is still a tie (cg-O3 1.05×).
- **`wb_bench` "groupby device 1T":** 16.9 → 25.7–26.6 ms (two runs), for the same reason. A
  fused fast path for SUM/COUNT-only lists, or state stored inline in the group slot, would
  likely recover it; that is a follow-up, not part of this change. Per-group
  sums are bit-identical to the old path (test `vec_group_sums_match_agg_hash_table`). The
  printed total over all groups changed in the last digits because groups are now emitted in
  first-seen order.
