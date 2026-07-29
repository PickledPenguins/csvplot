"""
Tests for _validate_args (CLI argument ranges/exclusions) and _validate_extras
(YAML-only extra key validation). Uses the real argparse parser so every args
object has every attribute the validator touches — no hand-rolled mocks.
"""
import pytest


def parse(m, *cli):
    """Parse CLI args through the real parser; a minimal valid file positional
    is prepended so tests only need to specify the flags under test."""
    return m.build_parser().parse_args(["data.csv", *cli])


def assert_dies(m, *cli):
    with pytest.raises(SystemExit):
        m._validate_args(parse(m, *cli))


def assert_ok(m, *cli):
    m._validate_args(parse(m, *cli))  # raises if it dies


# ── numeric lower-bound guards (must be >= 1) ─────────────────────────────

@pytest.mark.parametrize("flag", ["--head", "--tail", "--sample", "--every",
                                   "--rolling", "--ewm", "--dpi", "--max-rows", "--bins"])
def test_positive_int_flags_reject_zero_and_negative(m, flag):
    assert_dies(m, flag, "0")
    assert_dies(m, flag, "-1")


@pytest.mark.parametrize("flag", ["--head", "--tail", "--sample", "--every",
                                   "--rolling", "--ewm", "--bins"])
def test_positive_int_flags_accept_one(m, flag):
    assert_ok(m, flag, "1")


def test_top_requires_group_by_so_bare_top_needs_it_too(m):
    # --top alone (without --group-by) should die regardless of the value
    assert_dies(m, "--top", "5")


# ── --dpi upper bound ──────────────────────────────────────────────────────

def test_dpi_upper_bound_rejected(m):
    assert_dies(m, "--dpi", "2001")
    assert_dies(m, "--dpi", "50000")


def test_dpi_at_and_under_bound_ok(m):
    assert_ok(m, "--dpi", "2000")
    assert_ok(m, "--dpi", "300")


# ── --clip / --hline / --vline finiteness ─────────────────────────────────

def test_clip_rejects_nan_and_inf(m):
    assert_dies(m, "--clip", "nan", "5")
    assert_dies(m, "--clip", "0", "inf")
    assert_dies(m, "--clip", "-inf", "5")


def test_clip_rejects_lo_gte_hi(m):
    assert_dies(m, "--clip", "5", "5")
    assert_dies(m, "--clip", "5", "1")


def test_clip_valid_range_ok(m):
    assert_ok(m, "--clip", "0", "10")


def test_hline_rejects_nonfinite(m):
    assert_dies(m, "--hline", "nan")
    assert_dies(m, "--hline", "1", "inf")


def test_vline_rejects_nonfinite(m):
    assert_dies(m, "--vline", "nan")


def test_hline_vline_finite_values_ok(m):
    assert_ok(m, "--hline", "1", "2", "3")
    assert_ok(m, "--vline", "0")


# ── range checks: alpha, ci, percentile, unit-parse-threshold ─────────────

@pytest.mark.parametrize("bad", ["-0.1", "1.1"])
def test_alpha_out_of_range(m, bad):
    assert_dies(m, "--alpha", bad)


def test_alpha_boundary_values_ok(m):
    assert_ok(m, "--alpha", "0.0")
    assert_ok(m, "--alpha", "1.0")


@pytest.mark.parametrize("bad", ["-1", "101"])
def test_ci_out_of_range(m, bad):
    assert_dies(m, "--ci", bad)


def test_ci_boundary_ok(m):
    assert_ok(m, "--ci", "0")
    assert_ok(m, "--ci", "100")


def test_percentile_out_of_range(m):
    assert_dies(m, "--percentile", "50", "101")
    assert_dies(m, "--percentile", "-1")


def test_percentile_in_range_ok(m):
    assert_ok(m, "--percentile", "0", "50", "100")


@pytest.mark.parametrize("bad", ["-0.1", "1.5"])
def test_unit_parse_threshold_out_of_range(m, bad):
    assert_dies(m, "--unit-parse-threshold", bad)


# ── other scalar guards ────────────────────────────────────────────────────

def test_linewidth_must_be_positive(m):
    assert_dies(m, "--linewidth", "0")
    assert_dies(m, "--linewidth", "-1")
    assert_ok(m, "--linewidth", "0.1")


def test_interval_must_be_positive(m):
    assert_dies(m, "--interval", "0")


def test_smooth_must_be_positive_when_given(m):
    assert_dies(m, "--smooth", "0")
    assert_dies(m, "--smooth", "-1")
    assert_ok(m, "--smooth", "0.5")


def test_normalize_and_zscore_mutually_exclusive(m):
    assert_dies(m, "--normalize", "--zscore")


# ── --group-by dependent-flag wiring ──────────────────────────────────────

def test_group_by_requires_agg(m):
    assert_dies(m, "--group-by", "cat")


def test_agg_requires_group_by(m):
    assert_dies(m, "--agg", "mean")


def test_top_by_requires_group_by(m):
    assert_dies(m, "--top-by", "cat")


def test_group_by_with_agg_ok(m):
    assert_ok(m, "--group-by", "cat", "--agg", "mean")


# ── --dual-y mode incompatibility ─────────────────────────────────────────

@pytest.mark.parametrize("incompatible", ["--hist", "--corr", "--pairplot"])
def test_dual_y_incompatible_modes(m, incompatible):
    assert_dies(m, "--dual-y", "col", incompatible)


def test_dual_y_with_line_mode_ok(m):
    assert_ok(m, "--dual-y", "col")


# ── mode exclusivity ───────────────────────────────────────────────────────

def test_two_incompatible_modes_dies(m):
    assert_dies(m, "--box", "--bar")


def test_hist_and_kde_together_is_the_one_allowed_combo(m):
    assert_ok(m, "--hist", "--kde")


def test_single_mode_ok(m):
    assert_ok(m, "--scatter")


# ── --filter AST allow-list ────────────────────────────────────────────────

def test_filter_simple_comparison_ok(m):
    assert_ok(m, "--filter", "voltage > 0 and t < 10")


@pytest.mark.parametrize("expr", [
    "os.system('ls')",
    "__import__('os').system('ls')",
    "a.b",
    "f(1)",
    "a[0]",
    "lambda x: x",
])
def test_filter_rejects_unsafe_constructs(m, expr):
    assert_dies(m, "--filter", expr)


def test_filter_rejects_syntax_errors(m):
    assert_dies(m, "--filter", "a >")


# ── _validate_extras: annotations / fill_between ──────────────────────────

def test_extras_annotations_none_dies_cleanly(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"annotations": None})


def test_extras_fill_between_scalar_dies_cleanly(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"fill_between": 5})


def test_extras_annotations_valid_list_ok(m):
    m._validate_extras({"annotations": [{"x": 1, "y": 2, "text": "hi"}]})


def test_extras_annotations_missing_required_key_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"annotations": [{"x": 1, "y": 2}]})  # missing 'text'


def test_extras_annotations_non_numeric_coord_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"annotations": [{"x": "bad", "y": 2, "text": "hi"}]})


def test_extras_annotations_not_a_dict_element_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"annotations": ["not a dict"]})


def test_extras_fill_between_valid_list_ok(m):
    m._validate_extras({"fill_between": [{"xmin": 0, "xmax": 1}]})


def test_extras_fill_between_missing_key_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"fill_between": [{"xmin": 0}]})


def test_extras_annotations_too_many_entries_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"annotations": [{"x": 1, "y": 1, "text": "x"}] * 1001})


# ── _validate_extras: xlim/ylim/dual_y_lim ────────────────────────────────

@pytest.mark.parametrize("key", ["xlim", "ylim", "dual_y_lim"])
def test_extras_lim_wrong_length_dies(m, key):
    with pytest.raises(SystemExit):
        m._validate_extras({key: [0]})
    with pytest.raises(SystemExit):
        m._validate_extras({key: [0, 1, 2]})


@pytest.mark.parametrize("key", ["xlim", "ylim", "dual_y_lim"])
def test_extras_lim_non_numeric_dies(m, key):
    with pytest.raises(SystemExit):
        m._validate_extras({key: ["a", "b"]})


@pytest.mark.parametrize("key", ["xlim", "ylim", "dual_y_lim"])
def test_extras_lim_valid_ok(m, key):
    m._validate_extras({key: [0, 10]})


# ── _validate_extras: choice-restricted keys ──────────────────────────────

def test_extras_hist_stat_invalid_choice_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"hist_stat": "not_a_real_stat"})


def test_extras_hist_stat_valid_choice_ok(m):
    m._validate_extras({"hist_stat": "density"})


def test_extras_violin_inner_invalid_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"violin_inner": "nonsense"})


# ── _validate_extras: hline_style / vline_style ───────────────────────────

def test_extras_line_style_invalid_key_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"hline_style": {"not_a_valid_key": 1}})


def test_extras_line_style_non_dict_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"hline_style": "red"})


def test_extras_line_style_valid_ok(m):
    m._validate_extras({"hline_style": {"color": "red", "linewidth": 2, "alpha": 0.5}})


def test_extras_line_style_non_numeric_linewidth_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"hline_style": {"linewidth": "thick"}})


# ── _validate_extras: corr heatmap options ────────────────────────────────

def test_extras_corr_vmin_vmax_non_numeric_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"corr_vmin": "low"})


def test_extras_corr_annot_non_bool_dies(m):
    with pytest.raises(SystemExit):
        m._validate_extras({"corr_annot": "yes"})


def test_extras_corr_valid_ok(m):
    m._validate_extras({"corr_vmin": -1, "corr_vmax": 1, "corr_annot": True})


# ── _get_section ───────────────────────────────────────────────────────────

def test_get_section_missing_key_returns_empty_dict(m):
    assert m._get_section({}, "x_axis") == {}


def test_get_section_non_mapping_dies(m):
    with pytest.raises(SystemExit):
        m._get_section({"x_axis": "not a dict"}, "x_axis")


def test_get_section_returns_the_mapping(m):
    assert m._get_section({"x_axis": {"label": "Time"}}, "x_axis") == {"label": "Time"}
