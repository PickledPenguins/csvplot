#!/usr/bin/env bash
# Installs csvplot locally. No PyPI/internet access is assumed to be
# available -- if dependencies are already importable (common on HPC nodes
# via module systems/conda), this only installs the csvplot command itself.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
DEPS=(pandas numpy matplotlib seaborn yaml)  # module names, not pip names
MISSING=()

for m in "${DEPS[@]}"; do
    "$PY" -c "import $m" 2>/dev/null || MISSING+=("$m")
done

PIP_FLAGS=(--user)
"$PY" -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null && PIP_FLAGS=()
"$PY" -m pip install --dry-run --user pip >/dev/null 2>&1 || PIP_FLAGS+=(--break-system-packages)

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Missing dependencies: ${MISSING[*]}"
    echo "Attempting install via pip (flags: ${PIP_FLAGS[*]:-none})..."
    if ! "$PY" -m pip install "${PIP_FLAGS[@]}" pandas numpy matplotlib seaborn pyyaml; then
        echo
        echo "pip install failed -- likely no network access here."
        echo "Options:"
        echo "  1. On a machine with internet: 'pip download -d wheelhouse pandas numpy matplotlib seaborn pyyaml',"
        echo "     copy the wheelhouse/ dir here, then re-run:"
        echo "     $PY -m pip install ${PIP_FLAGS[*]} --no-index --find-links=wheelhouse pandas numpy matplotlib seaborn pyyaml"
        echo "  2. If your environment provides these via modules/conda, 'module load' or"
        echo "     'conda activate' the right environment first, then re-run this script."
        exit 1
    fi
else
    echo "All dependencies already available -- skipping their install."
fi

echo "Installing csvplot (local, editable, no network needed)..."
"$PY" -m pip install "${PIP_FLAGS[@]}" -e .

echo
echo "Done. Run 'csvplot --version' to confirm (or 'python3 csvplot.py --version'"
echo "if it's not on PATH yet -- check the pip install output above for the bin dir)."
