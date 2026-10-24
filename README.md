A simple vectorized execution engine to understand how pull based vectorized execution model works and how it differs from traditional row based volcano iterator model.

It has grown into a small lab for comparing execution models on the same data:

- a **vectorized pull engine**: columnar segments, batches with selection vectors, scan / filter /
  hash aggregate / hash join operators;
- a **compiled engine**, HyPer-style: the same logical plan is turned into C++ (produce/consume),
  compiled with clang into a shared library and loaded with `dlopen`;
- a **simulated Redshift cluster** (weekendRedshift): slices, DISTSTYLE and DS_* join strategies.

The two engines must return **bit-identical** results; the tests check it on every query shape.

## Build and run

Requires a C++20 compiler (Apple clang or clang) and, for the compiled engine at run time,
`clang++` on the PATH. The lint tooling needs Homebrew LLVM (`brew install llvm`).

    make            # all binaries
    make test       # unit, reference and differential tests (wb_test)
    make bench      # generates data/readings.csv (10M rows) if missing, runs wb_bench

| Binary | What it does |
|---|---|
| `wb_gen --rows N --out data/readings.csv` | Generates the `readings(device_id, ts, value)` CSV |
| `wb_bench [--csv path] [--iters n] [--narrow-float]` | Load time, column footprint, allocations and scan/filter/group-by latency |
| `wb_codegen [--csv path] [--iters n]` | The experiment: Q1–Q9 on the vectorized engine and generated code at `-O1`/`-O3`, including compile and load cost |
| `wb_mem <q4\|q7\|q8\|q7f> <vec\|cg> [--csv path]` | Peak memory of one query on one engine (one query per process) |
| `wb_dist --experiment strategies\|skew\|crossover\|all` | weekendRedshift join experiments |
| `wb` | Scratch driver (`src/main.cpp`) |

Generated libraries are cached in `.wb_cache/` (override with `WB_CODEGEN_CACHE`; compiler with
`WB_CXX`).

### Options

    make ALLOCATOR=jemalloc        # link against jemalloc (brew install jemalloc); make clean first
    bench/run_allocators.sh        # system malloc vs jemalloc at 10M and 100M rows

A 100M-row dataset: `./wb_gen --rows 100000000 --out data/readings-100m.csv`, then pass
`--csv data/readings-100m.csv` to any tool. Generated CSVs are git-ignored.

### Lint

    make lint            # report unused #includes (clang-tidy misc-include-cleaner, see .clang-tidy)
    make fix-includes    # remove them; rebuilds and runs the tests, restores everything on failure
    make install-hooks   # pre-commit hook: clean the staged C++ files on every commit

## Layout

| Path | Contents |
|---|---|
| `src/storage/` | Columns (frame-of-reference `Int64Vector`, `Float64Vector` with optional float storage), segments, zone maps, CSV loader |
| `src/exec/` | `Batch`, scan, filter, `HashAggOperator` (COUNT/SUM/MIN/MAX/AVG/MEDIAN), `HashJoinOperator`, parallel group-by |
| `src/plan/` | Logical plan IR (Scan / Filter / HashJoin / Aggregate), type checker, `QueryResult`, lowering to the vectorized engine |
| `src/codegen/` | Generator, clang compiler + cache, host binding, and `runtime.h` compiled into every generated library |
| `src/dist/` | weekendRedshift: cluster, distribution, exchange, distributed join, planner |
| `src/tools/` | The benchmark and data tools above |
| `tests/` | `wb_test` suites |
| `docs/` | `architecture.html`, `system-visual.html`; design specs with measured results in `docs/superpowers/specs/` |
| `bench/results/` | Recorded benchmark output |

## Results in brief

Apple M3 Pro, single-threaded, 10M rows unless noted. Details and the reasoning behind each number
are in the specs' results sections.

| Query | Vectorized | Compiled (`-O3`) | Note |
|---|---|---|---|
| Q3 `value > 50 AND device_id < 5000` | 7.2 ms | 33 ms | branch on an unpredictable predicate; vectorized is branchless |
| Q4 `GROUP BY device` SUM/COUNT | 28.8 ms | 12.4 ms | inline group state helps the fused per-row loop |
| Q5 join + group by | 11.5 ms | 12.4 ms | even, once the join stopped materializing per batch |
| Q7 MEDIAN by device | 159 ms | 131 ms | dominated by the final selection |

- **MEDIAN memory:** about 89 MiB for 10M rows (8–9 B per row), down from 382 MiB.
- **Compile cost:** first-time compile plus load is about 200–300 ms per query (compile 45–93 ms,
  first `dlopen` 155–230 ms); a cached library loads in under 1 ms.
- **Allocators:** jemalloc buys no speed here and uses 7–12% more memory.
- **Specs:** `codegen-vs-vectorized`, `aggregates` and `footprint-fixes` in
  `docs/superpowers/specs/`.

## weekendRedshift: slices, distribution styles and join data movement

A simulated Redshift cluster on top of the engine: N nodes x S slices, one thread per slice,
tables spread by DISTSTYLE (EVEN / KEY / ALL), and joins planned as Redshift's DS_* strategies.

    make test
    ./wb_dist --experiment strategies|skew|crossover|all

The three pieces, each specified in its header:

1. `distribute_table` (`src/dist/distribute.h`): the loader for EVEN / KEY / ALL.
2. `HashJoinOperator` (`src/exec/hash_join.h`): inner equi-join on an int64 key.
3. `is_valid_strategy` + `choose_join_strategy` (`src/dist/planner.h`): the planner rule.

Infrastructure: `Cluster` (`dist/cluster.h`), `Exchange` with redistribute/broadcast
(`dist/exchange.h`), `run_stream` (`dist/stream.h`), `run_distributed_join` (`dist/dist_join.h`).
