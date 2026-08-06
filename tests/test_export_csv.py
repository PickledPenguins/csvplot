"""
CSV export tests.

--export-csv exists so one run can feed another through a pipe, which only
holds if its output is accepted as input unchanged. These tests check the
emitted shape and then chain two real processes to confirm the round trip.
"""
import io
import subprocess
import sys

import pandas as pd
import pytest

from conftest import CSVPLOT


def _chain(first, second, tmp_path):
    """Run `csvplot <first> | csvplot <second>` and return the second result."""
    upstream = subprocess.run([sys.executable, CSVPLOT, *first], cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)
    assert upstream.returncode == 0, upstream.stderr
    return subprocess.run([sys.executable, CSVPLOT, *second], cwd=tmp_path,
                          input=upstream.stdout, capture_output=True, text=True,
                          timeout=60)


# ── emitted shape ──────────────────────────────────────────────────────────

def test_export_csv_has_no_index_column(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--export-csv")
    assert r.returncode == 0, r.stderr
    header = r.stdout.splitlines()[0].split(",")
    assert header == list(pd.read_csv(csv_numeric).columns)


def test_export_csv_omits_internal_index_column(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--export-csv")
    assert r.returncode == 0, r.stderr
    assert "_idx" not in r.stdout.splitlines()[0]


def test_export_csv_reflects_transforms(run_cli, tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("v\n1\n2\n3\n4\n10\n")
    r = run_cli(str(p), "--y", "v", "--cumsum", "--export-csv")
    assert r.returncode == 0, r.stderr
    assert list(pd.read_csv(io.StringIO(r.stdout))["v"]) == [1, 3, 6, 10, 20]


def test_export_csv_labels_rows_by_source_for_multiple_files(run_cli, tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("v\n1\n2\n")
    b.write_text("v\n3\n4\n")
    r = run_cli("--files", str(a), str(b), "--y", "v", "--export-csv")
    assert r.returncode == 0, r.stderr
    out = pd.read_csv(io.StringIO(r.stdout))
    assert set(out["source"]) == {str(a), str(b)}
    assert len(out) == 4


def test_export_csv_omits_source_column_for_one_file(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--export-csv")
    assert "source" not in r.stdout.splitlines()[0].split(",")


def test_export_csv_conflicts_with_export_json(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--export-csv", "--export-json")
    assert r.returncode == 1
    assert "mutually exclusive" in r.stderr


# ── round trip through a pipe ──────────────────────────────────────────────

def test_export_csv_is_accepted_as_input(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("v\n1\n2\n3\n4\n10\n")
    r = _chain([str(p), "--y", "v", "--export-csv"],
               ["-", "--y", "v", "--export-csv"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert list(pd.read_csv(io.StringIO(r.stdout))["v"]) == [1, 2, 3, 4, 10]


def test_transform_then_filter_across_a_pipe(tmp_path):
    """
    The reason the mode exists: --filter runs before transforms, so filtering
    on a transformed value needs a second pass.
    """
    p = tmp_path / "d.csv"
    p.write_text("v\n1\n2\n3\n4\n10\n")
    r = _chain([str(p), "--y", "v", "--cumsum", "--export-csv"],
               ["-", "--y", "v", "--filter", "v > 5", "--export-csv"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert list(pd.read_csv(io.StringIO(r.stdout))["v"]) == [6, 10, 20]


def test_missing_values_survive_the_round_trip(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("i,v\n0,1\n1,\n2,3\n")
    r = _chain([str(p), "--y", "v", "--export-csv"],
               ["-", "--y", "v", "--export-json"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert '"v": [1.0, null, 3.0]' in r.stdout


def test_piped_output_can_be_plotted(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("v\n1\n2\n3\n4\n10\n")
    r = _chain([str(p), "--y", "v", "--smooth", "1", "--export-csv"],
               ["-", "--y", "v", "--output", "o.png", "--force"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.png").exists()


# ── optional dependency guard ──────────────────────────────────────────────

@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("statsmodels") is not None,
    reason="statsmodels is installed, so the guard cannot fire")
def test_robust_regplot_without_statsmodels_fails_cleanly(run_cli, csv_numeric, tmp_path):
    """seaborn raises a bare RuntimeError at draw time; this must not escape."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("plot:\n  mode: regplot\n  robust: true\n")
    r = run_cli(str(csv_numeric), "--x", "a", "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "statsmodels" in r.stderr
