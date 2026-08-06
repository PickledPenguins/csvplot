#!/usr/bin/env bash
# Runs the test suite under coverage and prints a line/branch report.
#
# Most end-to-end tests invoke csvplot.py as a separate process. coverage.py
# does not follow subprocesses on its own, so a plain "pytest --cov" run
# understates coverage by roughly twenty points. This script installs the
# documented startup hook (a sitecustomize module on PYTHONPATH that calls
# coverage.process_startup) so every subprocess is measured, then combines
# the per-process data files into a single report.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
HOOK_DIR=$(mktemp -d)
trap 'rm -rf "$HOOK_DIR"' EXIT

cat > "$HOOK_DIR/sitecustomize.py" <<'PYEOF'
import coverage
coverage.process_startup()
PYEOF

rm -f .coverage .coverage.*

# COVERAGE_PROCESS_START must point at the config; its presence is what makes
# the hook activate in each subprocess. COVERAGE_FILE must be absolute: the
# end-to-end tests run csvplot.py with the working directory set to a pytest
# temporary directory, and a relative data-file path would scatter one
# unreachable data file per test there instead of collecting them here.
export COVERAGE_FILE="$PWD/.coverage"
export COVERAGE_PROCESS_START="$PWD/pyproject.toml"
export PYTHONPATH="$HOOK_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=${MPLBACKEND:-Agg}

# The suite is dominated by subprocess start-up, so it scales nearly linearly
# with worker count. pytest-xdist is used when installed and skipped silently
# when it is not, keeping the script usable with a bare pytest.
JOBS=()
if "$PY" -c "import xdist" 2>/dev/null; then
    JOBS=(-n auto)
fi

"$PY" -m coverage run -m pytest "${JOBS[@]}" "$@"

"$PY" -m coverage combine
"$PY" -m coverage report --show-missing --skip-covered
