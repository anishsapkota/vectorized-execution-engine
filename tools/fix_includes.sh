#!/bin/sh
# Removes unused #includes (clang-tidy misc-include-cleaner, configured in .clang-tidy).
#
#   tools/fix_includes.sh [--check] [--test] [files...]
#
#   --check   report unused includes and exit 1 if there are any; change nothing
#   --test    after fixing and rebuilding, also run the test suite
#   files     .cpp/.h files to process (default: every .cpp/.h under src/ and tests/)
#
# Fixing is all-or-nothing: every touched file is backed up first, and if the rebuild (or the tests
# with --test) fail, all files are restored and the script exits 1. A failure usually means an
# umbrella header was misreported as unused; add it to IgnoreHeaders in .clang-tidy.
#
# Environment: CLANG_TIDY / CLANG_APPLY_REPLACEMENTS override the tools (default: Homebrew LLVM);
# FIX_INCLUDES_CONFIG overrides the .clang-tidy file.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

LLVM_BIN=$(brew --prefix llvm 2>/dev/null)/bin
TIDY=${CLANG_TIDY:-$LLVM_BIN/clang-tidy}
APPLY=${CLANG_APPLY_REPLACEMENTS:-$LLVM_BIN/clang-apply-replacements}
CONFIG=${FIX_INCLUDES_CONFIG:-.clang-tidy}
[ -x "$TIDY" ] || { echo "fix_includes: clang-tidy not found at $TIDY (brew install llvm)" >&2; exit 2; }
[ -x "$APPLY" ] || { echo "fix_includes: clang-apply-replacements not found at $APPLY" >&2; exit 2; }
# A config clang-tidy cannot parse would otherwise look like "no unused includes".
"$TIDY" --config-file="$CONFIG" --dump-config > /dev/null 2>&1 || {
    echo "fix_includes: clang-tidy cannot read $CONFIG" >&2
    exit 2
}

check=0
run_tests=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check) check=1; shift ;;
        --test) run_tests=1; shift ;;
        --) shift; break ;;
        -*) echo "fix_includes: unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
if [ $# -gt 0 ]; then
    files="$*"
else
    files=$(find src tests -name '*.cpp' -o -name '*.h' | sort)
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Headers are analyzed as their own C++ translation unit.
tidy() {  # <file> [extra clang-tidy args...]
    f=$1
    shift
    case "$f" in
        *.h) "$TIDY" --quiet --config-file="$CONFIG" --extra-arg-before=-xc++ "$@" "$f" ;;
        *) "$TIDY" --quiet --config-file="$CONFIG" "$@" "$f" ;;
    esac
}

if [ "$check" -eq 1 ]; then
    for f in $files; do
        tidy "$f" 2> /dev/null | grep "misc-include-cleaner" >> "$WORK/report" || true
    done
    if [ -s "$WORK/report" ]; then
        cat "$WORK/report"
        echo "fix_includes: $(wc -l < "$WORK/report" | tr -d ' ') unused include(s); run 'make fix-includes'" >&2
        exit 1
    fi
    echo "fix_includes: no unused includes"
    exit 0
fi

# Fix: export each file's fixes, then apply them all at once (deduplicated).
mkdir -p "$WORK/fixes" "$WORK/backup"
i=0
for f in $files; do
    mkdir -p "$WORK/backup/$(dirname "$f")"
    cp -p "$f" "$WORK/backup/$f"
    i=$((i + 1))
    tidy "$f" --export-fixes="$WORK/fixes/$i.yaml" > /dev/null 2>&1 || true
done
"$APPLY" "$WORK/fixes"

changed=""
for f in $files; do
    cmp -s "$f" "$WORK/backup/$f" || changed="$changed $f"
done
if [ -z "$changed" ]; then
    echo "fix_includes: no unused includes"
    exit 0
fi

restore() {
    for f in $changed; do
        cp -p "$WORK/backup/$f" "$f"
    done
    echo "fix_includes: $1; restored:$changed" >&2
    echo "fix_includes: if a removed header was needed, add it to IgnoreHeaders in $CONFIG" >&2
    exit 1
}

make -j8 all wb_test > "$WORK/build.log" 2>&1 || { grep -E "error" "$WORK/build.log" | head -5 >&2; restore "build failed"; }
if [ "$run_tests" -eq 1 ]; then
    ./wb_test > "$WORK/test.log" 2>&1 || { grep -E "^FAIL" -A1 "$WORK/test.log" >&2; restore "tests failed"; }
fi
echo "fix_includes: removed unused includes from:$changed"
