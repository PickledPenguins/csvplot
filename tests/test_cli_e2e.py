"""
End-to-end tests that invoke csvplot.py as a real subprocess — the only
layer that catches issues mocking can hide (argparse wiring, matplotlib
backend interaction, real file I/O, actual seaborn rendering).
"""
import subprocess
import sys

import pytest

from conftest import CSVPLOT


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
    # --title is included here rather than as a separate case: pairplot builds
    # its own PairGrid, so the suptitle goes on a different figure object than
    # every other mode, and that branch is otherwise unexercised.
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "c", "--pairplot",
                "--title", "pair grid", "--output", "o.png", "--force")
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


# ── the internal row-index sentinel stays out of user-facing output ────────

def test_export_json_omits_internal_index_column(run_cli, csv_numeric):
    import json
    r = run_cli(str(csv_numeric), "--y", "b", "--export-json")
    ok(r)
    payload = next(iter(json.loads(r.stdout).values()))
    assert "_idx" not in payload
    assert "b" in payload


def test_percentile_without_y_omits_internal_index_column(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--percentile", "50")
    ok(r)
    assert "_idx" not in r.stdout


def test_stats_without_y_omits_internal_index_column(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--stats")
    ok(r)
    assert "_idx" not in r.stdout
    assert "a" in r.stdout


def test_stats_without_y_covers_all_numeric_columns(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--stats")
    ok(r)
    for col in ("a", "b", "c", "d"):
        assert col in r.stdout


# ── text-only modes never write a figure, so --output must not gate them ───

@pytest.mark.parametrize("mode", [
    ["--headers"], ["--info"], ["--stats"],
    ["--export-json", "--y", "b"],  # --export-json defines its transform target via --y
    ["--export-csv", "--y", "b"],   # likewise
    ["--percentile", "50"],
])
def test_text_modes_ignore_existing_output_path(run_cli, csv_numeric, tmp_path, mode):
    existing = tmp_path / "taken.png"
    existing.write_bytes(b"")
    r = run_cli(str(csv_numeric), *mode, "--output", str(existing))
    ok(r)
    assert existing.read_bytes() == b""  # left untouched


# ── group aggregation: --top / --top-by ───────────────────────────────────

def test_top_keeps_only_the_highest_groups(run_cli, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("g,v\na,1\nb,5\nc,9\nd,3\n")
    r = run_cli(str(p), "--y", "v", "--group-by", "g", "--agg", "mean",
                "--top", "2", "--bar", "--export-json")
    ok(r)
    import json
    groups = next(iter(json.loads(r.stdout).values()))["g"]
    assert sorted(groups) == ["b", "c"]


def test_top_by_ranks_on_the_named_column(run_cli, tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("g,v,w\na,1,100\nb,5,2\nc,9,1\n")
    r = run_cli(str(p), "--ys", "v", "w", "--group-by", "g", "--agg", "mean",
                "--top", "1", "--top-by", "w", "--bar", "--export-json")
    ok(r)
    import json
    assert next(iter(json.loads(r.stdout).values()))["g"] == ["a"]


# ── --fit linear ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", [[], ["--scatter"]])
def test_linear_fit_renders_on_line_and_scatter(run_cli, csv_numeric, mode, tmp_path):
    r = run_cli(str(csv_numeric), "--x", "a", "--y", "b", "--fit", "linear",
                *mode, "--output", "o.png", "--force")
    ok(r)
    assert (tmp_path / "o.png").exists()


# ── categorical modes comparing several columns without an --x ────────────
# These take the melt path, which reshapes wide columns into one categorical
# axis; it is skipped when --x or --subplots is given.

@pytest.mark.parametrize("mode", ["--box", "--violin", "--bar", "--swarm"])
def test_categorical_modes_compare_multiple_columns(run_cli, csv_numeric, mode, tmp_path):
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "c", mode,
                "--output", "o.png", "--force")
    ok(r)
    assert (tmp_path / "o.png").exists()


# ── pairplot and corr guards ──────────────────────────────────────────────

def test_pairplot_caps_the_panel_count(run_cli, tmp_path):
    header = ",".join(f"c{i}" for i in range(9))
    rows = [",".join(str(i + j) for j in range(9)) for i in range(20)]
    p = tmp_path / "wide.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n")
    r = run_cli(str(p), "--pairplot", "--output", "o.png", "--force")
    ok(r)
    assert "capping at 6" in r.stderr


def test_corr_rejects_non_numeric_columns(run_cli, tmp_path):
    p = tmp_path / "mixed.csv"
    p.write_text("a,label\n1,x\n2,y\n3,z\n")
    r = run_cli(str(p), "--ys", "a", "label", "--corr", "--output", "o.png", "--force")
    fails_cleanly(r)
    assert "non-numeric" in r.stderr


@pytest.mark.parametrize("mode", ["--corr", "--pairplot"])
def test_multi_file_modes_warn_that_extra_files_are_ignored(run_cli, csv_numeric, tmp_path, mode):
    second = tmp_path / "second.csv"
    second.write_text(csv_numeric.read_text())
    r = run_cli("--files", str(csv_numeric), str(second), "--ys", "a", "b", "c",
                mode, "--output", "o.png", "--force")
    ok(r)
    assert "additional file(s) ignored" in r.stderr


def test_stats_without_numeric_columns_fails_cleanly(run_cli, tmp_path):
    p = tmp_path / "text.csv"
    p.write_text("label,note\nx,one\ny,two\n")
    r = run_cli(str(p), "--stats")
    fails_cleanly(r)


def test_percentile_without_numeric_columns_fails_cleanly(run_cli, tmp_path):
    p = tmp_path / "text.csv"
    p.write_text("label,note\nx,one\ny,two\n")
    r = run_cli(str(p), "--percentile", "50")
    fails_cleanly(r)


# ── dual-y extras ─────────────────────────────────────────────────────────

def test_dual_y_warns_on_a_misleading_scale_ratio(run_cli, tmp_path):
    p = tmp_path / "scale.csv"
    rows = [f"{i},{i * 1e6}" for i in range(10)]
    p.write_text("small,big\n" + "\n".join(rows) + "\n")
    r = run_cli(str(p), "--y", "small", "--dual-y", "big",
                "--output", "o.png", "--force")
    ok(r)
    assert "dual-y scale ratio" in r.stderr


# ── malformed input ───────────────────────────────────────────────────────

def test_ragged_rows_report_a_parse_error(run_cli, tmp_path):
    p = tmp_path / "ragged.csv"
    p.write_text("a,b\n1,2\n3,4,5,6,7\n")
    r = run_cli(str(p), "--y", "a", "--output", "o.png", "--force")
    fails_cleanly(r)


# ── YAML-only rendering options ───────────────────────────────────────────

def _yaml(tmp_path, text, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_annotation_with_an_arrow_renders(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path, "annotations:\n  - x: 10\n    y: 20\n    text: peak\n"
                          "    arrow: true\n    color: red\n")
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    ok(r)


def test_annotation_with_invalid_styling_fails_cleanly(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path, "annotations:\n  - x: 10\n    y: 20\n    text: peak\n"
                          "    not_a_matplotlib_kwarg: 1\n")
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    fails_cleanly(r)
    assert "annotation" in r.stderr


def test_region_with_invalid_styling_fails_cleanly(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path, "regions:\n  - xmin: 1\n    xmax: 5\n"
                          "    not_a_matplotlib_kwarg: 1\n")
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    fails_cleanly(r)
    assert "fill_between" in r.stderr


def test_scatter_point_size_is_accepted(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path, "plot:\n  mode: scatter\n  s: 80\n")
    r = run_cli(str(csv_numeric), "--x", "a", "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    ok(r)


def test_right_axis_styling_and_limits_render(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path,
                "right_axis:\n  columns: [c]\n  label: Reference\n"
                "  lim: [0, 500]\n  colors: [crimson]\n  markers: [o]\n")
    r = run_cli(str(csv_numeric), "--x", "a", "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    ok(r)


def test_legend_outside_and_theme_extras_render(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path,
                "appearance:\n  font_scale: 1.2\n  legend:\n    outside: true\n"
                "    ncol: 2\n    title: Channels\n    fontsize: 9\n"
                "  rcParameters:\n    axes.titlepad: 12\n")
    r = run_cli(str(csv_numeric), "--ys", "a", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    ok(r)


def test_transparent_output_is_written(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path, "output:\n  transparent: true\n")
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    ok(r)
    assert (tmp_path / "o.png").stat().st_size > 0


def test_invalid_theme_setting_fails_cleanly(run_cli, csv_numeric, tmp_path):
    cfg = _yaml(tmp_path, "appearance:\n  rcParameters:\n    not.a.real.rcparam: 3\n")
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(cfg),
                "--output", "o.png", "--force")
    fails_cleanly(r)


# ── output path handling ──────────────────────────────────────────────────

def test_extensionless_output_defaults_to_png(run_cli, csv_numeric, tmp_path):
    r = run_cli(str(csv_numeric), "--y", "b", "--output", "figure", "--force")
    ok(r)
    assert (tmp_path / "figure").stat().st_size > 0


def test_no_temporary_file_is_left_behind(run_cli, csv_numeric, tmp_path):
    r = run_cli(str(csv_numeric), "--y", "b", "--output", "o.png", "--force")
    ok(r)
    assert not list(tmp_path.glob("*.tmp"))


# ── failure paths that only surface at the CLI boundary ───────────────────

def test_percentile_on_non_numeric_column_fails_cleanly(run_cli, tmp_path):
    p = tmp_path / "text.csv"
    p.write_text("g,h\na,x\nb,y\nc,z\n")
    r = run_cli(str(p), "--percentile", "50", "--y", "g")
    fails_cleanly(r)
    assert "--percentile requires numeric columns" in r.stderr


def test_hist_without_numeric_columns_fails_cleanly(run_cli, tmp_path):
    p = tmp_path / "text.csv"
    p.write_text("g,h\na,x\nb,y\n")
    r = run_cli(str(p), "--hist", "--output", "o.png", "--force")
    fails_cleanly(r)
    assert "no columns to plot for --hist" in r.stderr


def test_corr_on_constant_columns_warns_and_titles(run_cli, tmp_path):
    """A constant column has no variance, so every correlation is NaN."""
    p = tmp_path / "const.csv"
    p.write_text("a,b\n1,5\n1,5\n1,5\n")
    r = run_cli(str(p), "--ys", "a", "b", "--corr", "--title", "Constant",
                "--output", "o.png", "--force")
    ok(r)
    assert "all-NaN" in r.stderr
    assert (tmp_path / "o.png").exists()


def test_directory_given_as_input_fails_cleanly(run_cli, tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    r = run_cli(str(d), "--headers")
    fails_cleanly(r)
    assert "could not load" in r.stderr


def test_closed_output_pipe_exits_quietly(tmp_path):
    """
    Piping into something that stops reading early (head, less) is routine.
    The tool must exit without a traceback rather than dumping BrokenPipeError.
    """
    p = tmp_path / "big.csv"
    p.write_text("a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(30000)))
    proc = subprocess.Popen(
        [sys.executable, CSVPLOT, str(p), "--y", "b", "--export-json"],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    proc.stdout.read(50)
    proc.stdout.close()          # consumer goes away mid-write
    stderr = proc.communicate()[1]
    assert proc.returncode == 0
    assert "Traceback" not in stderr


# ── --hue renders in every mode ────────────────────────────────────────────
# seaborn builds its own legend entry per hue level and supplies the label
# itself. Passing an explicit label alongside hue -- even None -- collided
# with that and raised a TypeError from inside matplotlib, so every mode is
# rendered here rather than only the ones that happened to be covered.

@pytest.mark.parametrize("mode", [
    [], ["--scatter"], ["--regplot"], ["--box"], ["--violin"],
    ["--bar"], ["--swarm"], ["--hist"], ["--kde"],
])
def test_hue_renders_in_every_mode(run_cli, tmp_path, mode):
    p = tmp_path / "hue.csv"
    p.write_text("x,y,g\n1,10,a\n2,20,a\n3,15,b\n4,25,b\n")
    r = run_cli(str(p), "--x", "x", "--y", "y", "--hue", "g", *mode,
                "--output", "o.png", "--force")
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr
    assert (tmp_path / "o.png").exists()
