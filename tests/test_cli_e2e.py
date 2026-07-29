"""
End-to-end tests that invoke csvplot.py as a real subprocess — the only
layer that catches issues mocking can hide (argparse wiring, matplotlib
backend interaction, real file I/O, actual seaborn rendering).
"""
import pytest


def ok(r):
    assert r.returncode == 0, f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}"
    assert "Traceback" not in r.stderr


def fails_cleanly(r):
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "csvplot error:" in r.stderr


# ── every plot mode renders without error ──────────────────────────────────

@pytest.mark.parametrize("mode_args", [
    [],  # default line mode
    ["--scatter"],
    ["--hist"],
    ["--kde"],
    ["--hist", "--kde"],
    ["--box", "--x", "a"],
    ["--violin", "--x", "a"],
    ["--bar", "--x", "a"],
    ["--swarm", "--x", "a"],
    ["--regplot"],
])
def test_every_mode_renders(run_cli, csv_numeric, mode_args, tmp_path):
    r = run_cli(str(csv_numeric), "--y", "b", *mode_args, "--output", "o.png", "--force")
    ok(r)
    assert (tmp_path / "o.png").exists()


def test_pairplot_renders(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "c", "--pairplot", "--output", "o.png", "--force")
    ok(r)


def test_corr_renders(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "c", "--corr", "--output", "o.png", "--force")
    ok(r)


def test_subplots_multi_series(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "c", "--subplots", "--output", "o.png", "--force")
    ok(r)


def test_dual_y(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "a", "--dual-y", "b", "--output", "o.png", "--force")
    ok(r)


def test_group_by_bar(run_cli, csv_grouped):
    r = run_cli(str(csv_grouped), "--y", "biomarker", "--group-by", "treatment",
                "--agg", "mean", "--bar", "--output", "o.png", "--force")
    ok(r)


# ── text-output modes ────────────────────────────────────────────────────

def test_stats_mode(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "--stats")
    ok(r)
    assert "descriptive statistics" in r.stderr


def test_export_json_mode(run_cli, csv_numeric):
    import json
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "--export-json")
    ok(r)
    data = json.loads(r.stdout)
    assert "a" in list(data.values())[0]


def test_percentile_mode(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "--percentile", "25", "50", "75")
    ok(r)


# ── regression repros: findings from the last review pass ──────────────────

def test_repro_group_by_violin_uses_grouping_column(run_cli, csv_grouped):
    """1.1: box/violin/swarm previously ignored a --group-by-derived x."""
    r = run_cli(str(csv_grouped), "--y", "biomarker", "--group-by", "treatment",
                "--agg", "mean", "--violin", "--output", "o.png", "--force")
    ok(r)


def test_repro_hist_subplots_no_explicit_y(run_cli, csv_numeric):
    """1.2: --hist --subplots without --y/--ys used to silently collapse to one panel."""
    r = run_cli(str(csv_numeric), "--hist", "--subplots", "--output", "o.png", "--force")
    ok(r)


def test_repro_empty_annotations_yaml_dies_cleanly(run_cli, csv_numeric, tmp_path):
    """1.3: an empty 'annotations:' YAML key parses to None and used to
    crash with an unhandled TypeError instead of a clean error."""
    yaml_path = tmp_path / "empty_ann.yaml"
    yaml_path.write_text("annotations:\n")
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(yaml_path),
                "--output", "o.png", "--force")
    fails_cleanly(r)


def test_repro_bins_zero_rejected(run_cli, csv_numeric):
    """1.4: --bins 0 used to be silently ignored instead of rejected."""
    r = run_cli(str(csv_numeric), "--y", "b", "--hist", "--bins", "0",
                "--output", "o.png", "--force")
    fails_cleanly(r)
    assert "--bins" in r.stderr


def test_repro_bins_negative_rejected_cleanly(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--hist", "--bins", "-5",
                "--output", "o.png", "--force")
    fails_cleanly(r)


def test_repro_clip_nan_rejected(run_cli, csv_numeric):
    """1.7: --clip used to silently accept 'nan' as a bound."""
    r = run_cli(str(csv_numeric), "--y", "b", "--clip", "nan", "5",
                "--output", "o.png", "--force")
    fails_cleanly(r)
    assert "finite" in r.stderr


def test_repro_dpi_too_large_rejected(run_cli, csv_numeric):
    """3.2: no prior upper bound on --dpi."""
    r = run_cli(str(csv_numeric), "--y", "b", "--dpi", "50000",
                "--output", "o.png", "--force")
    fails_cleanly(r)


def test_repro_regplot_hue_nan_warns(run_cli, csv_hue_nan):
    """1.6: NaN hue rows used to be silently dropped with no warning."""
    r = run_cli(str(csv_hue_nan), "--x", "x", "--y", "y", "--hue", "grp",
                "--regplot", "--output", "o.png", "--force")
    ok(r)
    assert "NaN" in r.stderr


def test_repro_head_bypasses_max_rows_when_sole_row_op(run_cli, csv_numeric):
    """3.1: --head alone should preview a file bigger than --max-rows,
    instead of failing the row-count guard before --head gets a chance."""
    r = run_cli(str(csv_numeric), "--y", "b", "--max-rows", "10", "--head", "5",
                "--output", "o.png", "--force")
    ok(r)


def test_repro_max_rows_alone_still_enforced(run_cli, csv_numeric):
    """Regression guard: the --head bypass must not weaken --max-rows when
    --head is NOT used."""
    r = run_cli(str(csv_numeric), "--y", "b", "--max-rows", "10", "--output", "o.png", "--force")
    fails_cleanly(r)
    assert "max-rows" in r.stderr


def test_repro_head_plus_tail_does_not_bypass_max_rows(run_cli, csv_numeric):
    """The head-bypass optimization only applies when --head is the SOLE
    row-op; combined with --tail it must still respect --max-rows."""
    r = run_cli(str(csv_numeric), "--y", "b", "--max-rows", "10", "--head", "5",
                "--tail", "3", "--output", "o.png", "--force")
    fails_cleanly(r)


def test_repro_valid_annotations_and_fill_between_still_work(run_cli, csv_numeric, tmp_path):
    """Regression guard for the 1.3 fix: well-formed annotations/fill_between
    must not be rejected by the new type guard."""
    yaml_path = tmp_path / "ann.yaml"
    yaml_path.write_text(
        "annotations:\n  - x: 10\n    y: 20\n    text: peak\n"
        "fill_between:\n  - xmin: 5\n    xmax: 15\n    alpha: 0.2\n"
    )
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(yaml_path),
                "--output", "o.png", "--force")
    ok(r)


# ── multi-file overlay ──────────────────────────────────────────────────────

def test_multi_file_overlay(run_cli, tmp_path):
    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    f1.write_text("t,v\n0,1\n1,2\n2,3\n")
    f2.write_text("t,v\n0,4\n1,5\n2,6\n")
    r = run_cli("--files", str(f1), str(f2), "--x", "t", "--y", "v",
                "--output", "o.png", "--force")
    ok(r)


def test_duplicate_file_paths_rejected(run_cli, csv_numeric):
    r = run_cli("--files", str(csv_numeric), str(csv_numeric), "--y", "b",
                "--output", "o.png", "--force")
    fails_cleanly(r)


# ── output handling ──────────────────────────────────────────────────────

def test_output_already_exists_requires_force(run_cli, csv_numeric, tmp_path):
    out = tmp_path / "o.png"
    out.write_text("placeholder")
    r = run_cli(str(csv_numeric), "--y", "b", "--output", str(out))
    fails_cleanly(r)
    assert "already exists" in r.stderr


def test_output_directory_missing_dies_cleanly(run_cli, csv_numeric, tmp_path):
    r = run_cli(str(csv_numeric), "--y", "b", "--output", str(tmp_path / "no_such_dir" / "o.png"))
    fails_cleanly(r)


# ── stdin input ──────────────────────────────────────────────────────────

def test_stdin_input(run_cli, tmp_path):
    csv_text = "a,b\n1,2\n3,4\n5,6\n"
    r = run_cli("-", "--y", "b", "--output", "o.png", "--force", stdin=csv_text)
    ok(r)


# ── argument-error paths produce clean messages, not tracebacks ───────────

def test_missing_file_dies_cleanly(run_cli, tmp_path):
    r = run_cli(str(tmp_path / "nope.csv"), "--y", "b")
    fails_cleanly(r)
    assert "not found" in r.stderr


def test_missing_column_dies_cleanly(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "nonexistent_col")
    fails_cleanly(r)


def test_no_y_column_dies_cleanly(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric))
    fails_cleanly(r)


def test_conflicting_modes_die_cleanly(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--box", "--bar")
    fails_cleanly(r)


# ── --version, --headers, --info ─────────────────────────────────────────

def test_version_flag_prints_version_and_exits(run_cli, m):
    r = run_cli("--version")
    assert r.returncode == 0
    assert m.__version__ in r.stdout


def test_headers_lists_columns_no_y_required(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--headers")
    ok(r)
    assert r.stdout.split() == ["a", "b", "c", "d"]


def test_headers_multi_file_labels_each(run_cli, tmp_path):
    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    f1.write_text("a,b\n1,2\n")
    f2.write_text("x,y\n1,2\n")
    r = run_cli("--files", str(f1), str(f2), "--headers")
    ok(r)
    assert str(f1) in r.stdout and str(f2) in r.stdout
    assert "a" in r.stdout and "x" in r.stdout


def test_info_no_y_required(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--info")
    ok(r)
    assert "60 rows" in r.stdout
    assert "4 columns" in r.stdout
    for col in ("a", "b", "c", "d"):
        assert col in r.stdout


def test_info_shows_non_null_counts_and_dtype(run_cli, tmp_path):
    p = tmp_path / "nulls.csv"
    p.write_text("a,b\n1,\n2,2\n3,\n")
    r = run_cli(str(p), "--info")
    ok(r)
    assert "1/3" in r.stdout  # b has 1 non-null out of 3
    assert "int64" in r.stdout


def test_headers_and_info_do_not_require_valid_plot_args(run_cli, csv_numeric):
    """These must work even with none of the usual --y/--x context."""
    r1 = run_cli(str(csv_numeric), "--headers")
    r2 = run_cli(str(csv_numeric), "--info")
    assert r1.returncode == 0 and r2.returncode == 0
