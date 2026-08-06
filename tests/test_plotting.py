"""
Tests for the plotting layer: _plot_single's per-mode dispatch to seaborn,
and _build_figure's panel-count/axes-sizing logic. Seaborn/matplotlib calls
are mocked so these run fast and assert on the exact kwargs passed, rather
than rendering real figures (that's covered separately in test_cli_e2e.py).
"""
import types
from unittest.mock import patch

import pandas as pd
import pytest


def base_args(**kw):
    """A complete, valid args namespace via the real parser, with overrides."""
    defaults = dict(x=None, hue=None, color=None, marker=None, linewidth=1.5,
                     alpha=1.0, kde=False, ci=95, palette=None, bins=None)
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


@pytest.fixture
def df():
    return pd.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 5.0, 10.0]})


# ── mode dispatch: correct seaborn function + key kwargs ──────────────────

def test_line_mode_calls_lineplot(m, df):
    with patch("csvplot.sns.lineplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "line", object(), base_args())
        assert f.called
        assert f.call_args.kwargs["x"] == "cat"
        assert f.call_args.kwargs["y"] == "val"


def test_scatter_mode_calls_scatterplot(m, df):
    with patch("csvplot.sns.scatterplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "scatter", object(), base_args())
        assert f.called


def test_hist_mode_calls_histplot_with_bins(m, df):
    with patch("csvplot.sns.histplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "hist", object(), base_args(bins=20))
        assert f.call_args.kwargs["bins"] == 20


def test_hist_mode_omits_bins_when_none(m, df):
    with patch("csvplot.sns.histplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "hist", object(), base_args(bins=None))
        assert "bins" not in f.call_args.kwargs


def test_kde_mode_calls_kdeplot(m, df):
    with patch("csvplot.sns.kdeplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "kde", object(), base_args())
        assert f.called


def test_unknown_mode_dies(m, df):
    with pytest.raises(SystemExit):
        m._plot_single(df, "cat", "val", "lbl", "not_a_real_mode", object(), base_args())


# ── regression: box/violin/swarm respect a --group-by-derived x_col ───────

@pytest.mark.parametrize("mode,fn", [
    ("box", "sns.boxplot"), ("violin", "sns.violinplot"),
    ("bar", "sns.barplot"), ("swarm", "sns.swarmplot"),
])
def test_categorical_modes_use_real_x_col_from_group_by(m, df, mode, fn):
    """--group-by sets x_col without setting args.x; box/violin/bar/swarm must
    still use it (regression test for the original --group-by --box bug)."""
    with patch(f"csvplot.{fn}") as f:
        m._plot_single(df, "cat", "val", "lbl", mode, object(), base_args(x=None))
        assert f.call_args.kwargs["x"] == "cat"


@pytest.mark.parametrize("mode,fn", [
    ("box", "sns.boxplot"), ("violin", "sns.violinplot"),
    ("bar", "sns.barplot"), ("swarm", "sns.swarmplot"),
])
def test_categorical_modes_omit_x_for_idx_sentinel(m, df, mode, fn):
    """No --x and no --group-by: x_col is the internal sentinel and must
    NOT be passed as a grouping column (regression: no-op case must not break).
    bar joined this list in v0.18 — it used to pass the sentinel straight
    through, plotting one bar per row instead of a single aggregate bar."""
    with patch(f"csvplot.{fn}") as f:
        m._plot_single(df, m._IDX_COL, "val", "lbl", mode, object(), base_args(x=None))
        assert f.call_args.kwargs["x"] is None


# ── color/style kwarg resolution ───────────────────────────────────────────

def test_series_override_color_reaches_lineplot(m, df):
    with patch("csvplot.sns.lineplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "line", object(), base_args(),
                        so={"color": "crimson"})
        assert f.call_args.kwargs["color"] == "crimson"


def test_global_color_used_when_no_hue(m, df):
    with patch("csvplot.sns.lineplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "line", object(), base_args(color="green"))
        assert f.call_args.kwargs["color"] == "green"


def test_global_color_suppressed_when_hue_active(m, df):
    with patch("csvplot.sns.lineplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "line", object(),
                        base_args(color="green", hue="cat"))
        assert "color" not in f.call_args.kwargs


def test_marker_from_series_override(m, df):
    with patch("csvplot.sns.lineplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "line", object(), base_args(),
                        so={"marker": "^"})
        assert f.call_args.kwargs["marker"] == "^"


def test_linestyle_only_set_when_explicit_override_present(m, df):
    with patch("csvplot.sns.lineplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "line", object(), base_args())
        assert "linestyle" not in f.call_args.kwargs
    with patch("csvplot.sns.lineplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "line", object(), base_args(),
                        so={"linestyle": ":"})
        assert f.call_args.kwargs["linestyle"] == ":"


# ── regplot: hue branch and NaN warning ────────────────────────────────────

def test_regplot_no_hue_single_call(m, df):
    with patch("csvplot.sns.regplot") as f:
        m._plot_single(df, "cat", "val", "lbl", "regplot", object(), base_args(hue=None))
        assert f.call_count == 1


def test_regplot_with_hue_one_call_per_group(m):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "grp": ["A", "A", "B", "B"]})
    with patch("csvplot.sns.regplot") as f:
        m._plot_single(df, "x", "y", "lbl", "regplot", object(), base_args(hue="grp"))
        assert f.call_count == 2  # one per hue group


def test_regplot_hue_too_many_categories_dies(m):
    df = pd.DataFrame({"x": range(25), "y": range(25), "grp": [str(i) for i in range(25)]})
    with pytest.raises(SystemExit):
        m._plot_single(df, "x", "y", "lbl", "regplot", object(), base_args(hue="grp"))


def test_regplot_hue_nan_warns(m, capsys):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "grp": ["A", "A", None, "B"]})
    with patch("csvplot.sns.regplot"):
        m._plot_single(df, "x", "y", "lbl", "regplot", object(), base_args(hue="grp"))
    assert "NaN" in capsys.readouterr().err


def test_regplot_hue_no_nan_no_warning(m, capsys):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "grp": ["A", "A", "B", "B"]})
    with patch("csvplot.sns.regplot"):
        m._plot_single(df, "x", "y", "lbl", "regplot", object(), base_args(hue="grp"))
    assert capsys.readouterr().err == ""


# ── _build_figure: panel-count sizing (regression for the --subplots bug) ──

def full_args(**kw):
    """Complete args for _build_figure via the real parser, with overrides."""
    p = __import__("csvplot").build_parser()
    ns = p.parse_args(["data.csv"])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _with_idx(df, m):
    """Add the internal row-index sentinel column, normally added by
    _run_pipeline — needed here since these tests call _build_figure directly."""
    return df.assign(**{m._IDX_COL: df.index})


def test_panel_count_matches_explicit_ycols_with_subplots(m, tmp_path):
    df = _with_idx(pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}), m)
    frames = {"f": df}
    args = full_args(subplots=True, output=str(tmp_path / "o.png"), force=True)
    with patch("csvplot.plt.subplots", wraps=__import__("matplotlib.pyplot", fromlist=["subplots"]).subplots) as sp:
        m._build_figure(frames, args, ["a", "b", "c"], m._IDX_COL, {})
    assert sp.call_args_list[0].args[:2] == (3, 1)


def test_panel_count_matches_implicit_hist_columns_with_subplots(m, tmp_path):
    """Regression for the original bug: --hist --subplots with no --y/--ys
    must size panels from the columns that will actually be drawn."""
    df = _with_idx(pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}), m)
    frames = {"f": df}
    args = full_args(hist=True, subplots=True, output=str(tmp_path / "o.png"), force=True)
    with patch("csvplot.plt.subplots", wraps=__import__("matplotlib.pyplot", fromlist=["subplots"]).subplots) as sp:
        m._build_figure(frames, args, [], m._IDX_COL, {})
    assert sp.call_args_list[0].args[:2] == (4, 1)


def test_panel_count_single_panel_without_subplots(m, tmp_path):
    df = _with_idx(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), m)
    frames = {"f": df}
    args = full_args(subplots=False, output=str(tmp_path / "o.png"), force=True)
    with patch("csvplot.plt.subplots", wraps=__import__("matplotlib.pyplot", fromlist=["subplots"]).subplots) as sp:
        m._build_figure(frames, args, ["a", "b"], m._IDX_COL, {})
    # single-panel branch calls plt.subplots(figsize=...) with no positional nrows/ncols
    assert sp.call_args_list[0].args == ()


def test_panel_height_floor_applied_for_many_panels(m, tmp_path):
    df = _with_idx(pd.DataFrame({f"c{i}": [1, 2] for i in range(6)}), m)
    frames = {"f": df}
    args = full_args(subplots=True, figsize="10,2", output=str(tmp_path / "o.png"), force=True)
    with patch("csvplot.plt.subplots", wraps=__import__("matplotlib.pyplot", fromlist=["subplots"]).subplots) as sp:
        m._build_figure(frames, args, list(df.columns), m._IDX_COL, {})
    figsize_used = sp.call_args_list[0].kwargs["figsize"]
    assert figsize_used[1] >= 6 * 2.0  # floor kicked in over the requested height=2


def test_panel_height_floor_does_not_shrink_large_figsize(m, tmp_path):
    df = _with_idx(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), m)
    frames = {"f": df}
    args = full_args(subplots=True, figsize="10,50", output=str(tmp_path / "o.png"), force=True)
    with patch("csvplot.plt.subplots", wraps=__import__("matplotlib.pyplot", fromlist=["subplots"]).subplots) as sp:
        m._build_figure(frames, args, ["a", "b"], m._IDX_COL, {})
    figsize_used = sp.call_args_list[0].kwargs["figsize"]
    assert figsize_used[1] == 50.0  # explicit user figsize preserved, not shrunk


# ── _build_figure: dual-y color consistency across panels ─────────────────

def test_dual_y_series_gets_same_default_color_on_every_panel(m, tmp_path):
    """Regression: color_idx used to leak across panels, giving the same
    dual-y series a different default color on each subplot panel."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "ref": [5, 6]})
    frames = {"f": df}
    args = full_args(subplots=True, dual_y=["ref"], output=str(tmp_path / "o.png"), force=True)
    lineplot_calls = []
    with patch("csvplot.sns.lineplot", side_effect=lambda **kw: lineplot_calls.append(kw)):
        m._build_figure(frames, args, ["a", "b"], m._IDX_COL, {})
    dual_y_calls = [c for c in lineplot_calls if c.get("y") == "ref"]
    assert len(dual_y_calls) == 2  # one per panel
    assert dual_y_calls[0]["color"] == dual_y_calls[1]["color"]


# ── _overlay_linear_fit ───────────────────────────────────────────────────
# The --fit linear overlay. Uses a real Axes because the assertions are about
# the line that ends up on it, not about a call signature.

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture
def real_ax():
    fig, a = plt.subplots()
    yield a
    plt.close(fig)


def _fit_args(m, **kw):
    return base_args(**{"linewidth": 1.5, "alpha": 1.0, **kw})


def test_linear_fit_draws_a_line_with_the_expected_slope(m, real_ax):
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 3.0, 5.0, 7.0]})
    m._overlay_linear_fit(df, "x", "y", real_ax, _fit_args(m))
    assert len(real_ax.lines) == 1
    xs, ys = real_ax.lines[0].get_data()
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    assert slope == pytest.approx(2.0)


def test_linear_fit_labels_the_line_for_the_legend(m, real_ax):
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    m._overlay_linear_fit(df, "x", "y", real_ax, _fit_args(m))
    assert real_ax.lines[0].get_label() == "y fit"


def test_linear_fit_draws_left_to_right_on_unsorted_input(m, real_ax):
    """The fit line is drawn over sorted x so it renders as one clean stroke."""
    df = pd.DataFrame({"x": [3.0, 0.0, 2.0, 1.0], "y": [7.0, 1.0, 5.0, 3.0]})
    m._overlay_linear_fit(df, "x", "y", real_ax, _fit_args(m))
    xs = real_ax.lines[0].get_data()[0]
    assert list(xs) == sorted(xs)


def test_linear_fit_ignores_rows_with_missing_values(m, real_ax):
    df = pd.DataFrame({"x": [0.0, 1.0, None, 3.0], "y": [1.0, 3.0, 99.0, 7.0]})
    m._overlay_linear_fit(df, "x", "y", real_ax, _fit_args(m))
    xs, ys = real_ax.lines[0].get_data()
    assert len(xs) == 3
    assert (ys[-1] - ys[0]) / (xs[-1] - xs[0]) == pytest.approx(2.0)


def test_linear_fit_coerces_non_numeric_values(m, real_ax):
    df = pd.DataFrame({"x": ["0", "1", "junk"], "y": ["1", "3", "5"]})
    m._overlay_linear_fit(df, "x", "y", real_ax, _fit_args(m))
    assert len(real_ax.lines) == 1


@pytest.mark.parametrize("df_kw", [
    {"x": [1.0], "y": [2.0]},                       # single point
    {"x": [1.0, None], "y": [2.0, 3.0]},             # one usable pair
    {"x": [float("nan")] * 3, "y": [1.0, 2.0, 3.0]},  # nothing usable
])
def test_linear_fit_skipped_when_fewer_than_two_points(m, real_ax, df_kw):
    m._overlay_linear_fit(pd.DataFrame(df_kw), "x", "y", real_ax, _fit_args(m))
    assert len(real_ax.lines) == 0


def test_linear_fit_on_constant_x_does_not_raise(m, real_ax, capsys):
    """Degenerate x makes the fit rank-deficient; it must warn, not crash."""
    df = pd.DataFrame({"x": [2.0, 2.0, 2.0], "y": [1.0, 2.0, 3.0]})
    m._overlay_linear_fit(df, "x", "y", real_ax, _fit_args(m))
    if len(real_ax.lines) == 0:
        assert "linear fit" in capsys.readouterr().err
