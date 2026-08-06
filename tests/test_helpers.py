"""Unit tests for csvplot's small, pure helper functions."""
import math
import types

import pandas as pd
import pytest


# ── _categorical_x ────────────────────────────────────────────────────────

def test_categorical_x_sentinel_becomes_none(m):
    assert m._categorical_x(m._IDX_COL) is None


def test_categorical_x_real_column_passes_through(m):
    assert m._categorical_x("treatment") == "treatment"


def test_categorical_x_empty_string_is_not_sentinel(m):
    # edge case: an empty-string column name is falsy but not the sentinel,
    # and should still pass through rather than being treated as "no column"
    assert m._categorical_x("") == ""


# ── _resolve_plot_columns ─────────────────────────────────────────────────

def test_resolve_plot_columns_explicit_ycols_always_win(m):
    df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    assert m._resolve_plot_columns(df, ["x"], True) == ["x"]
    assert m._resolve_plot_columns(df, ["x"], False) == ["x"]


def test_resolve_plot_columns_implicit_when_allowed(m):
    """Used by --hist/--kde (mode in {'hist','kde'}) and by --corr/--pairplot
    (always allow_implicit=True) when no --y/--ys was given."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "s": ["a", "b"]})
    cols = m._resolve_plot_columns(df, [], True)
    assert set(cols) == {"x", "y"}  # string column excluded, no ycols needed


def test_resolve_plot_columns_no_implicit_when_disallowed(m):
    """Used by line/scatter/box/violin/bar/swarm/regplot, which require
    explicit --y/--ys rather than falling back to all numeric columns."""
    df = pd.DataFrame({"x": [1], "y": [2]})
    assert m._resolve_plot_columns(df, [], False) == []


def test_resolve_plot_columns_excludes_idx_sentinel(m):
    """Regression: the internal row-index column must never be treated as
    a real data series for hist/kde/corr/pairplot's implicit-column fallback."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], m._IDX_COL: [0, 1]})
    cols = m._resolve_plot_columns(df, [], True)
    assert m._IDX_COL not in cols
    assert set(cols) == {"a", "b"}


# ── _style_val / _series_style ────────────────────────────────────────────

def test_style_val_returns_indexed_value(m):
    assert m._style_val({"y_colors": ["red", "blue"]}, "y_colors", 1, "fallback") == "blue"


def test_style_val_falls_back_when_missing_or_out_of_range(m):
    assert m._style_val({}, "y_colors", 0, "fb") == "fb"
    assert m._style_val({"y_colors": ["red"]}, "y_colors", 5, "fb") == "fb"


def test_style_val_falls_back_on_explicit_none_entry(m):
    """A YAML list like [red, null] should fall back for the null slot,
    not propagate None as if it were a deliberate override."""
    assert m._style_val({"y_colors": ["red", None]}, "y_colors", 1, "fb") == "fb"


def test_series_style_only_includes_explicit_overrides(m):
    so = m._series_style({"y_colors": ["red"], "y_alphas": [0.5]}, "y_", 0)
    assert so == {"color": "red", "alpha": 0.5}


def test_series_style_empty_when_nothing_set(m):
    assert m._series_style({}, "y_", 0) == {}


def test_series_style_prefix_isolates_left_and_right_axis(m):
    yaml_x = {"y_colors": ["left"], "dual_y_colors": ["right"]}
    assert m._series_style(yaml_x, "y_", 0) == {"color": "left"}
    assert m._series_style(yaml_x, "dual_y_", 0) == {"color": "right"}


# ── _color_override ───────────────────────────────────────────────────────

def _args(**kw):
    defaults = {"color": None}
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def test_color_override_series_override_wins(m):
    assert m._color_override({"color": "green"}, _args(color="purple"), None) == {"color": "green"}


def test_color_override_falls_back_to_global_when_no_hue(m):
    assert m._color_override({}, _args(color="purple"), None) == {"color": "purple"}


def test_color_override_hue_suppresses_global_color(m):
    assert m._color_override({}, _args(color="purple"), "grp") == {}


def test_color_override_nothing_set_is_empty(m):
    assert m._color_override({}, _args(), None) == {}


# ── parse_figsize ─────────────────────────────────────────────────────────

def test_parse_figsize_valid(m):
    assert m.parse_figsize("10,6") == (10.0, 6.0)
    assert m.parse_figsize("10.5,6.25") == (10.5, 6.25)


@pytest.mark.parametrize("bad", ["10", "10,6,3", "a,b", "0,6", "-1,6", "10,-1"])
def test_parse_figsize_invalid_dies(m, bad):
    with pytest.raises(SystemExit):
        m.parse_figsize(bad)


def test_parse_figsize_rejects_nonfinite(m):
    with pytest.raises(SystemExit):
        m.parse_figsize("nan,6")
    with pytest.raises(SystemExit):
        m.parse_figsize("inf,6")


# ── _active_mode ──────────────────────────────────────────────────────────

def _mode_args(**kw):
    base = dict(scatter=False, hist=False, kde=False, box=False, violin=False,
                bar=False, swarm=False, regplot=False, corr=False, pairplot=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_active_mode_default_is_line(m):
    assert m._active_mode(_mode_args()) == "line"


def test_active_mode_kde_alone(m):
    assert m._active_mode(_mode_args(kde=True)) == "kde"


@pytest.mark.parametrize("mode", ["scatter", "hist", "box", "violin", "bar", "swarm", "regplot", "corr"])
def test_active_mode_explicit_flags(m, mode):
    assert m._active_mode(_mode_args(**{mode: True})) == mode


def test_active_mode_hist_and_kde_together_is_hist(m):
    # hist+kde is the one allowed dual-flag combo; hist takes precedence
    # in _active_mode (kde becomes an overlay inside the hist branch)
    assert m._active_mode(_mode_args(hist=True, kde=True)) == "hist"


# ── parse_units ───────────────────────────────────────────────────────────

def test_parse_units_basic_conversions(m):
    s = pd.Series(["10ms", "-3us", "5kb", "1.5e3ms"])
    out = m.parse_units(s)
    assert out.tolist() == pytest.approx([0.01, -3e-6, 5000, 1.5])


def test_parse_units_unknown_suffix_becomes_nan(m):
    # threshold=0.5 isolates the per-value "unknown suffix -> NaN" behavior
    # from the whole-column revert-on-low-conversion-rate behavior tested below
    s = pd.Series(["10ms", "5km"])  # "km" is not a recognized unit
    out = m.parse_units(s, threshold=0.5)
    assert out.iloc[0] == pytest.approx(0.01)
    assert math.isnan(out.iloc[1])


def test_parse_units_no_suffix_column_returned_unchanged(m):
    s = pd.Series(["1.0", "2.1", "3.5"])
    out = m.parse_units(s, require_suffix=True)
    assert out is s  # require_suffix short-circuits; identity, not just equality


def test_parse_units_bare_numbers_allowed_when_require_suffix_false(m):
    s = pd.Series(["1.0", "2.5"])
    out = m.parse_units(s, require_suffix=False)
    assert out.tolist() == pytest.approx([1.0, 2.5])


def test_parse_units_threshold_reverts_on_low_conversion_rate(m):
    # only 1 of 4 values has a recognized suffix; below default threshold=1.0
    # the whole column should revert to original rather than partially convert
    s = pd.Series(["10ms", "abc", "def", "ghi"])
    out = m.parse_units(s, threshold=1.0)
    assert out is s


def test_parse_units_all_nan_column(m):
    s = pd.Series([None, None], dtype=object)
    out = m.parse_units(s)
    assert out.isna().all()


# ── gaussian_smooth ───────────────────────────────────────────────────────

def test_gaussian_smooth_rejects_nonpositive_sigma(m):
    with pytest.raises(ValueError):
        m.gaussian_smooth([1, 2, 3], 0)
    with pytest.raises(ValueError):
        m.gaussian_smooth([1, 2, 3], -1)


def test_gaussian_smooth_all_nan_returns_unchanged(m):
    import numpy as np
    out = m.gaussian_smooth([float("nan")] * 5, 1.0)
    assert np.isnan(out).all()


def test_gaussian_smooth_single_element(m):
    out = m.gaussian_smooth([5.0], 1.0)
    assert list(out) == [5.0]


def test_gaussian_smooth_preserves_nan_positions(m):
    import numpy as np
    vals = [1.0, float("nan"), 3.0, 4.0, float("nan"), 6.0]
    out = m.gaussian_smooth(vals, 1.0)
    assert np.isnan(out[1]) and np.isnan(out[4])
    assert not np.isnan(out[0]) and not np.isnan(out[2])


def test_gaussian_smooth_constant_series_stays_constant(m):
    out = m.gaussian_smooth([5.0] * 10, 2.0)
    assert all(abs(v - 5.0) < 1e-9 for v in out)


# ── _drop_infinities ──────────────────────────────────────────────────────

def test_drop_infinities_replaces_inf_and_reports_change(m):
    s = pd.Series([1.0, float("inf"), 3.0, float("-inf")])
    out, changed = m._drop_infinities(s, "col")
    assert changed is True
    assert out.tolist()[0] == 1.0 and out.tolist()[2] == 3.0
    assert math.isnan(out.tolist()[1]) and math.isnan(out.tolist()[3])


def test_drop_infinities_no_change_when_all_finite(m):
    s = pd.Series([1.0, 2.0, 3.0])
    out, changed = m._drop_infinities(s, "col")
    assert changed is False
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_drop_infinities_warning_message_content(m, capsys):
    s = pd.Series([float("inf"), float("inf"), 1.0])
    m._drop_infinities(s, "mycol")
    err = capsys.readouterr().err
    assert "mycol" in err and "2 value(s)" in err and "infinite" in err

def test_yaml_depth_budget_stops_alias_amplification(m):
    """Regression: a small aliased YAML structure must not cause an
    unbounded/very slow recursive walk in _yaml_depth."""
    import yaml
    import time
    lines = ['a0: &a0 ["x"]']
    for i in range(1, 8):
        refs = ",".join([f"*a{i-1}"] * 15)
        lines.append(f"a{i}: &a{i} [{refs}]")
    parsed = yaml.safe_load("\n".join(lines))
    t0 = time.time()
    d = m._yaml_depth(parsed)
    assert time.time() - t0 < 2.0
    assert d > m._MAX_YAML_DEPTH


# ── _json_safe ────────────────────────────────────────────────────────────
# --export-json must emit strict JSON. json.dumps would otherwise write the
# non-standard NaN/Infinity literals, and pd.NA/numpy scalars/Decimal are not
# serialisable at all.

def test_json_safe_pd_na_becomes_null(m):
    assert m._json_safe(pd.NA) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_json_safe_non_finite_float_becomes_null(m, value):
    assert m._json_safe(value) is None


def test_json_safe_finite_float_passes_through(m):
    assert m._json_safe(1.5) == 1.5


def test_json_safe_numpy_scalar_becomes_python_scalar(m):
    import numpy as np
    result = m._json_safe(np.int64(7))
    assert result == 7 and not isinstance(result, np.generic)


def test_json_safe_non_finite_numpy_scalar_becomes_null(m):
    import numpy as np
    assert m._json_safe(np.float64("inf")) is None


def test_json_safe_decimal_becomes_float(m):
    from decimal import Decimal
    result = m._json_safe(Decimal("2.50"))
    assert result == 2.5 and isinstance(result, float)


def test_json_safe_non_finite_decimal_becomes_null(m):
    from decimal import Decimal
    assert m._json_safe(Decimal("Infinity")) is None


def test_json_safe_leaves_strings_alone(m):
    assert m._json_safe("text") == "text"


# ── _series_range ─────────────────────────────────────────────────────────
# Feeds the dual-y scale-ratio heuristic, which must never be the thing that
# fails a plot; every failure path returns 0.0.

def test_series_range_spans_all_frames(m):
    frames = {"a": pd.DataFrame({"v": [1.0, 3.0]}), "b": pd.DataFrame({"v": [10.0]})}
    assert m._series_range(frames, "v") == 9.0


def test_series_range_all_nan_column_is_zero(m):
    frames = {"a": pd.DataFrame({"v": [float("nan"), float("nan")]})}
    assert m._series_range(frames, "v") == 0.0


def test_series_range_missing_column_is_zero(m):
    assert m._series_range({"a": pd.DataFrame({"v": [1.0]})}, "absent") == 0.0


def test_series_range_non_numeric_column_is_zero(m):
    assert m._series_range({"a": pd.DataFrame({"v": ["x", "y"]})}, "v") == 0.0


# ── parse_units: per-value conversion edges ───────────────────────────────

def test_parse_units_preserves_missing_values(m):
    out = m.parse_units(pd.Series(["10ms", None, "20ms"]))
    assert out.isna().sum() == 1
    assert out.iloc[0] == pytest.approx(0.01)


def test_parse_units_overflowing_exponent_becomes_nan(m):
    """1e309 overflows to inf, which must be normalised to NaN, not plotted."""
    out = m.parse_units(pd.Series(["1e309ms", "1ms"]), threshold=0.0)
    assert out.isna().iloc[0]
