"""Shared fixtures for the csvplot test suite."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CSVPLOT = str(ROOT / "csvplot.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def run_cli(tmp_path):
    """Run csvplot.py as a subprocess with cwd=tmp_path. Returns CompletedProcess."""
    def _run(*args, stdin=None):
        return subprocess.run(
            [sys.executable, CSVPLOT, *args],
            cwd=tmp_path, input=stdin, capture_output=True, text=True, timeout=30,
        )
    return _run


@pytest.fixture
def m():
    """Import csvplot.py as a module for direct unit-level access."""
    import csvplot
    return csvplot


# ── CSV fixtures ──────────────────────────────────────────────────────────
# Each writes into tmp_path and returns the Path; kept small and purpose-built
# rather than one mega-fixture, so each test's data intent is obvious at a glance.

@pytest.fixture
def csv_numeric(tmp_path):
    """4 numeric columns (a,b,c,d), 60 rows, values are simple multiples of i."""
    p = tmp_path / "numeric.csv"
    rows = ["a,b,c,d"] + [f"{i},{i*2},{i*3.0},{i*0.5}" for i in range(60)]
    p.write_text("\n".join(rows) + "\n")
    return p


@pytest.fixture
def csv_grouped(tmp_path):
    """treatment,biomarker — 3 groups x 2 rows, for --group-by tests."""
    p = tmp_path / "grouped.csv"
    p.write_text("treatment,biomarker\nA,1.0\nA,2.0\nB,5.0\nB,6.0\nC,10.0\nC,11.0\n")
    return p


@pytest.fixture
def csv_hue_nan(tmp_path):
    """x,y,grp — one NaN in the hue column."""
    p = tmp_path / "hue_nan.csv"
    p.write_text("x,y,grp\n1,2,A\n2,3,A\n3,5,\n4,4,B\n5,6,B\n6,7,B\n")
    return p


@pytest.fixture
def csv_timeseries(tmp_path):
    """t,value — 30 rows for transform-chain tests (rolling/ewm/diff/etc.)."""
    p = tmp_path / "series.csv"
    rows = ["t,value"] + [f"{i},{(i % 7) * 2 + 1}" for i in range(30)]
    p.write_text("\n".join(rows) + "\n")
    return p


@pytest.fixture
def csv_single_row(tmp_path):
    p = tmp_path / "single.csv"
    p.write_text("a,b\n1,2\n")
    return p


@pytest.fixture
def csv_all_nan_col(tmp_path):
    p = tmp_path / "allnan.csv"
    p.write_text("a,b\n1,\n2,\n3,\n")
    return p


@pytest.fixture
def csv_constant_col(tmp_path):
    """b is constant — exercises normalize/zscore's divide-by-zero-range path."""
    p = tmp_path / "constant.csv"
    p.write_text("a,b\n1,5\n2,5\n3,5\n4,5\n")
    return p


@pytest.fixture
def csv_dup_columns(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("a,a,b\n1,2,3\n4,5,6\n")
    return p


@pytest.fixture
def csv_with_strings(tmp_path):
    """Numeric column with a stray non-numeric value to exercise coercion warnings."""
    p = tmp_path / "strings.csv"
    p.write_text("a,b\n1,x\n2,3\n3,4\n")
    return p


@pytest.fixture
def csv_empty(tmp_path):
    """Header only, zero data rows."""
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n")
    return p
