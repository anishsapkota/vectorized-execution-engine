#!/bin/sh
# Tests tools/fix_includes.sh and the pre-commit hook on a scratch copy of the repository.
#
#   tools/test_fix_includes.sh
#
# Exits 0 when every check passes; prints PASS/FAIL per check.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
failures=0

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; failures=$((failures + 1)); }

rsync -a --exclude '*.o' --exclude '*.d' "$ROOT/src" "$ROOT/tests" "$ROOT/tools" "$ROOT/Makefile" \
    "$ROOT/compile_flags.txt" "$ROOT/.clang-tidy" "$WORK/"
cd "$WORK" || exit 1
make -j8 all wb_test > /dev/null 2>&1 || { echo "scratch copy does not build"; exit 1; }

# 1. --check reports a planted unused include and changes nothing.
sed -i '' 's|#include "storage/table.h"|#include "storage/table.h"\n\n#include <map>|' src/storage/table.cpp
before=$(cksum < src/storage/table.cpp)
if tools/fix_includes.sh --check src/storage/table.cpp > check.log 2>&1; then
    fail "check flags an unused <map>"
elif grep -q "<map>\|map" check.log && [ "$(cksum < src/storage/table.cpp)" = "$before" ]; then
    pass "check flags an unused <map> and leaves the file alone"
else
    fail "check output or file contents unexpected"
fi

# 2. The fix removes it and the tree still builds.
if tools/fix_includes.sh src/storage/table.cpp > fix.log 2>&1 && ! grep -q "#include <map>" src/storage/table.cpp; then
    pass "fix removes the unused <map> and the build passes"
else
    fail "fix did not remove <map> (see fix.log)"; cat fix.log
fi

# 3. A removal that breaks the build is rolled back. Without the IgnoreHeaders entry, the include
#    cleaner wrongly drops <mach/mach.h> (an umbrella header) from mem_probe.cpp.
grep -v -e 'IgnoreHeaders' -e "value: 'mach" .clang-tidy > no-ignore.clang-tidy
before=$(cksum < src/tools/mem_probe.cpp)
if FIX_INCLUDES_CONFIG=no-ignore.clang-tidy tools/fix_includes.sh src/tools/mem_probe.cpp > rollback.log 2>&1; then
    fail "fix reports success although the build breaks"
elif [ "$(cksum < src/tools/mem_probe.cpp)" = "$before" ] && grep -qi "restored" rollback.log; then
    pass "a build-breaking removal is rolled back"
else
    fail "rollback did not restore mem_probe.cpp (see rollback.log)"; cat rollback.log
fi

# 4. Hook: refuses a staged file that also has unstaged edits; cleans a fully staged file.
git init -q . && git add -A > /dev/null && git -c user.email=t@t -c user.name=t commit -qm init
git config core.hooksPath tools/git-hooks
sed -i '' 's|#include "storage/table.h"|#include "storage/table.h"\n\n#include <set>|' src/storage/table.cpp
git add src/storage/table.cpp
echo "// unstaged edit" >> src/storage/table.cpp
if git -c user.email=t@t -c user.name=t commit -qm partial > hook1.log 2>&1; then
    fail "hook refuses a partially staged file"
else
    pass "hook refuses a partially staged file"
fi
git checkout -q -- src/storage/table.cpp  # drop the unstaged edit; <set> stays staged
if git -c user.email=t@t -c user.name=t commit -qm clean > hook2.log 2>&1 &&
    ! git show HEAD:src/storage/table.cpp | grep -q "#include <set>"; then
    pass "hook removes the unused <set> from the commit"
else
    fail "hook did not clean the commit (see hook2.log)"; cat hook2.log
fi

[ "$failures" -eq 0 ] && echo "all fix_includes checks passed"
exit "$failures"
