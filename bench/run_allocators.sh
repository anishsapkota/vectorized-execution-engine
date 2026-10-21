#!/bin/sh
# Runs the benchmark suite once per malloc implementation and writes
# bench/results/alloc/<tool>-<rows>-<allocator>.txt.
#
#   bench/run_allocators.sh            # system and jemalloc (needs: brew install jemalloc)
#
# Each allocator gets a clean build and a full test run first. For jemalloc, the memory probe is
# also run with MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0 (freed pages returned at once), since
# jemalloc's default keeps freed pages for ~10 s and that shows up in peak footprint.
set -eu
cd "$(dirname "$0")/.."
OUT=bench/results/alloc
mkdir -p "$OUT"

DATA_10M=data/readings.csv
DATA_100M=data/readings-100m.csv
[ -f "$DATA_10M" ] || ./wb_gen --rows 10000000 --out "$DATA_10M"
[ -f "$DATA_100M" ] || ./wb_gen --rows 100000000 --out "$DATA_100M"

run_mem() {  # <rows-label> <csv> <suffix> [MALLOC_CONF]
    : > "$OUT/mem-$1-$3.txt"
    for q in q4 q7 q8 q7f; do
        for e in vec cg; do
            MALLOC_CONF="${4:-}" WB_CODEGEN_CACHE="$(mktemp -d)" ./wb_mem "$q" "$e" --csv "$2" >> "$OUT/mem-$1-$3.txt"
        done
    done
}

for alloc in system jemalloc; do
    make clean > /dev/null
    make -j8 all wb_test ALLOCATOR="$alloc" > /dev/null
    ./wb_test | tail -1 | tee "$OUT/tests-$alloc.txt"
    for size in 10m 100m; do
        csv=$DATA_10M
        [ "$size" = 100m ] && csv=$DATA_100M
        WB_CODEGEN_CACHE="$(mktemp -d)" ./wb_codegen --csv "$csv" --iters 5 > "$OUT/codegen-$size-$alloc.txt"
        ./wb_bench --csv "$csv" --iters 5 > "$OUT/bench-$size-$alloc.txt"
        run_mem "$size" "$csv" "$alloc"
        if [ "$alloc" = jemalloc ]; then
            run_mem "$size" "$csv" jemalloc-decay0 "dirty_decay_ms:0,muzzy_decay_ms:0"
        fi
    done
done

make clean > /dev/null
make -j8 all wb_test > /dev/null  # leave the default (system) build in place
echo "done: $OUT"
