#!/usr/bin/env bash
# Runs the test suite in bounded slices.
#
# The suite splits cleanly in two. The in-process tests finish in well under a
# minute. The rest launch csvplot.py as a subprocess and cost roughly 3.7
# seconds each, almost entirely spent importing seaborn; there are enough of
# them that a single invocation runs for several minutes on one core, which is
# awkward under CI step timeouts and interactive patience alike.
#
# This script runs the fast subset first, so mistakes surface quickly, then
# works through the subprocess tests in batches. Each batch is a separate
# pytest invocation with its own bounded runtime, and a failing batch does not
# stop the remaining ones -- the exit status reflects whether anything failed.
#
# Usage:
#   ./run-tests.sh                 fast subset, then slow tests in batches of 25
#   ./run-tests.sh --batch 10      smaller batches
#   ./run-tests.sh --fast          in-process tests only
#   ./run-tests.sh -k pattern      remaining arguments pass through to pytest
set -uo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
export MPLBACKEND=${MPLBACKEND:-Agg}

BATCH=12
FAST_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --batch) BATCH=$2; shift 2 ;;
        --fast)  FAST_ONLY=1; shift ;;
        *) break ;;
    esac
done

# pytest-xdist helps in proportion to available cores; on a single core it adds
# only overhead, so it is requested but left to xdist to decide the count.
JOBS=()
if [ "$("$PY" -c 'import os; print(os.cpu_count() or 1)')" -gt 1 ] \
   && "$PY" -c "import xdist" 2>/dev/null; then
    JOBS=(-n auto)
fi

FAILED=0

# pytest exit code 5 means "nothing was selected", which is an expected
# outcome when a -k filter matches only subprocess tests. Treat it as neutral
# so a clean run is not reported as a failure.
run_pytest() {
    "$PY" -m pytest -q "$@"
    local rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
        FAILED=1
    fi
}

echo "== in-process tests =="
run_pytest -m "not slow" "${JOBS[@]}" "$@"

if [ "$FAST_ONLY" -eq 1 ]; then
    exit "$FAILED"
fi

# Node IDs are collected up front so the batches are stable and each pytest
# invocation is handed an explicit, bounded list rather than a filter whose
# cost is unknown until it runs.
IDS=$("$PY" -m pytest --collect-only -q -m slow "$@" 2>/dev/null | grep '::' || true)
if [ -z "$IDS" ]; then
    echo "== no subprocess tests selected =="
    exit "$FAILED"
fi

TOTAL=$(printf '%s\n' "$IDS" | wc -l | tr -d ' ')
BATCH_DIR=$(mktemp -d)
trap 'rm -rf "$BATCH_DIR"' EXIT
printf '%s\n' "$IDS" | split -l "$BATCH" - "$BATCH_DIR/batch_"

N=0
for f in "$BATCH_DIR"/batch_*; do
    N=$((N + 1))
done

I=0
for f in "$BATCH_DIR"/batch_*; do
    I=$((I + 1))
    echo "== subprocess tests, batch $I of $N ($TOTAL total, ~$((BATCH * 4))s each) =="
    # Read one node ID per line into an array. Parametrised IDs can contain
    # spaces (the parameter value is part of the ID), so they must never be
    # passed through word splitting.
    BATCH_IDS=()
    while IFS= read -r line; do
        [ -n "$line" ] && BATCH_IDS+=("$line")
    done < "$f"
    run_pytest "${JOBS[@]}" "${BATCH_IDS[@]}"
done

if [ "$FAILED" -eq 0 ]; then
    echo "== all $TOTAL subprocess tests and the in-process subset passed =="
else
    echo "== failures reported above =="
fi
exit "$FAILED"
