"""
Tests for the row-operations and column-transform pipeline
(apply_row_ops, apply_transforms). Covers each operation individually, the
documented fixed execution order, and interaction/chaining effects between
transforms — the area flagged for deeper review after the last pass.
"""
import math

import numpy as np
import pandas as pd
import pytest


def parse(m, *cli):
    return m.build_parser().parse_args(["data.csv", *cli])


def seq_df(n=10):
    """0..n-1 in column 'v', for order-sensitive row-op tests."""
    return pd.DataFrame({"v": list(range(n))})


# ── apply_row_ops: individual operations ──────────────────────────────────

def test_head(m):
    out = m.apply_row_ops(seq_df(10), parse(m, "--head", "3"))
    assert out["v"].tolist() == [0, 1, 2]


def test_tail(m):
    out = m.apply_row_ops(seq_df(10), parse(m, "--tail", "3"))
    assert out["v"].tolist() == [7, 8, 9]


def test_filter(m):
    out = m.apply_row_ops(seq_df(10), parse(m, "--filter", "v > 5"))
    assert out["v"].tolist() == [6, 7, 8, 9]


def test_filter_invalid_column_dies(m):
    with pytest.raises(SystemExit):
        m.apply_row_ops(seq_df(10), parse(m, "--filter", "nonexistent_col > 5"))


def test_sort(m):
    df = pd.DataFrame({"v": [3, 1, 2]})
    out = m.apply_row_ops(df, parse(m, "--sort", "v"))
    assert out["v"].tolist() == [1, 2, 3]


def test_sort_missing_column_dies(m):
    with pytest.raises(SystemExit):
        m.apply_row_ops(seq_df(5), parse(m, "--sort", "nope"))


def test_sample_size_and_determinism_with_seed(m):
    out1 = m.apply_row_ops(seq_df(20), parse(m, "--sample", "5", "--seed", "42"))
    out2 = m.apply_row_ops(seq_df(20), parse(m, "--sample", "5", "--seed", "42"))
    assert len(out1) == 5
    assert out1["v"].tolist() == out2["v"].tolist()


def test_sample_larger_than_df_does_not_error(m):
    out = m.apply_row_ops(seq_df(5), parse(m, "--sample", "100"))
    assert len(out) == 5  # min(requested, available)


def test_every(m):
    out = m.apply_row_ops(seq_df(10), parse(m, "--every", "3"))
    assert out["v"].tolist() == [0, 3, 6, 9]


def test_no_row_ops_returns_equivalent_frame(m):
    df = seq_df(5)
    out = m.apply_row_ops(df, parse(m))
    assert out["v"].tolist() == df["v"].tolist()


def test_index_reset_only_when_modified(m):
    # after --tail, index should be reset to 0-based rather than keeping
    # the original 7,8,9 positional index
    out = m.apply_row_ops(seq_df(10), parse(m, "--tail", "3"))
    assert out.index.tolist() == [0, 1, 2]


# ── apply_row_ops: documented fixed order (head -> tail -> filter -> sort -> sample -> every) ──

def test_order_head_then_tail(m):
    # head(5) of 0..9 -> [0,1,2,3,4]; tail(2) of that -> [3,4]
    # if order were reversed (tail first), result would be [8,9] instead
    out = m.apply_row_ops(seq_df(10), parse(m, "--head", "5", "--tail", "2"))
    assert out["v"].tolist() == [3, 4]


def test_order_filter_before_sort(m):
    df = pd.DataFrame({"v": [5, 1, 8, 2, 9, 3]})
    out = m.apply_row_ops(df, parse(m, "--filter", "v > 2", "--sort", "v"))
    assert out["v"].tolist() == [3, 5, 8, 9]


def test_order_sort_before_every(m):
    # sort ascending then take every 2nd row
    df = pd.DataFrame({"v": [5, 1, 4, 2, 3]})
    out = m.apply_row_ops(df, parse(m, "--sort", "v", "--every", "2"))
    assert out["v"].tolist() == [1, 3, 5]


def test_order_head_before_filter(m):
    # head(4) of 0..9 -> [0,1,2,3]; filter v>1 on that -> [2,3]
    # if filter ran first, head(4) of the filtered [2..9] would be [2,3,4,5]
    out = m.apply_row_ops(seq_df(10), parse(m, "--head", "4", "--filter", "v > 1"))
    assert out["v"].tolist() == [2, 3]


# ── apply_transforms: individual transforms ───────────────────────────────

def test_cumsum(m):
    df = pd.DataFrame({"v": [1, 2, 3, 4]})
    out = m.apply_transforms(df, parse(m, "--cumsum"), ["v"])
    assert out["v"].tolist() == [1, 3, 6, 10]


def test_cummax(m):
    df = pd.DataFrame({"v": [1, 3, 2, 5, 4]})
    out = m.apply_transforms(df, parse(m, "--cummax"), ["v"])
    assert out["v"].tolist() == [1, 3, 3, 5, 5]


def test_cummin(m):
    df = pd.DataFrame({"v": [5, 3, 4, 1, 2]})
    out = m.apply_transforms(df, parse(m, "--cummin"), ["v"])
    assert out["v"].tolist() == [5, 3, 3, 1, 1]


def test_diff(m):
    df = pd.DataFrame({"v": [1, 3, 6, 10]})
    out = m.apply_transforms(df, parse(m, "--diff"), ["v"])
    assert math.isnan(out["v"].iloc[0])
    assert out["v"].iloc[1:].tolist() == [2, 3, 4]


def test_rate_is_percent_change(m):
    df = pd.DataFrame({"v": [10.0, 15.0, 12.0]})
    out = m.apply_transforms(df, parse(m, "--rate"), ["v"])
    assert math.isnan(out["v"].iloc[0])
    assert out["v"].iloc[1] == pytest.approx(0.5)   # (15-10)/10
    assert out["v"].iloc[2] == pytest.approx(-0.2)  # (12-15)/15


def test_rate_division_by_zero_becomes_nan_not_inf(m):
    df = pd.DataFrame({"v": [0.0, 5.0, 0.0]})
    out = m.apply_transforms(df, parse(m, "--rate"), ["v"])
    # (5-0)/0 = inf -> must be NaN, not +/-inf
    assert not np.isinf(out["v"]).any()
    assert math.isnan(out["v"].iloc[1])


def test_ewm(m):
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    out = m.apply_transforms(df, parse(m, "--ewm", "2"), ["v"])
    assert out["v"].iloc[0] == pytest.approx(1.0)  # min_periods=1, first value unchanged
    assert not out["v"].isna().any()


def test_rolling_no_leading_nan(m):
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = m.apply_transforms(df, parse(m, "--rolling", "3"), ["v"])
    assert not out["v"].isna().any()  # min_periods=1
    assert out["v"].iloc[0] == pytest.approx(1.0)
    assert out["v"].iloc[2] == pytest.approx(2.0)  # mean(1,2,3)


def test_smooth_calls_gaussian_smooth(m):
    df = pd.DataFrame({"v": [1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0]})
    raw_std = df["v"].std()  # captured before the call: apply_transforms mutates df in place
    out = m.apply_transforms(df, parse(m, "--smooth", "1.0"), ["v"])
    # smoothing should reduce the peak-to-peak variance vs raw oscillation
    assert out["v"].std() < raw_std


def test_clip(m):
    df = pd.DataFrame({"v": [-5.0, 0.0, 5.0, 15.0]})
    out = m.apply_transforms(df, parse(m, "--clip", "0", "10"), ["v"])
    assert out["v"].tolist() == [0.0, 0.0, 5.0, 10.0]


def test_baseline_subtracts_first_nonnan(m):
    df = pd.DataFrame({"v": [10.0, 12.0, 15.0]})
    out = m.apply_transforms(df, parse(m, "--baseline"), ["v"])
    assert out["v"].tolist() == [0.0, 2.0, 5.0]


def test_baseline_skips_leading_nan(m):
    df = pd.DataFrame({"v": [float("nan"), 10.0, 12.0]})
    out = m.apply_transforms(df, parse(m, "--baseline"), ["v"])
    assert math.isnan(out["v"].iloc[0])
    assert out["v"].iloc[1] == pytest.approx(0.0)
    assert out["v"].iloc[2] == pytest.approx(2.0)


def test_baseline_all_nan_uses_zero(m):
    df = pd.DataFrame({"v": [float("nan"), float("nan")]})
    out = m.apply_transforms(df, parse(m, "--baseline"), ["v"])
    assert out["v"].isna().all()  # nan - 0.0 stays nan


def test_normalize_min_max_scaling(m):
    df = pd.DataFrame({"v": [0.0, 5.0, 10.0]})
    out = m.apply_transforms(df, parse(m, "--normalize"), ["v"])
    assert out["v"].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_constant_column_maps_to_zero(m, capsys):
    df = pd.DataFrame({"v": [5.0, 5.0, 5.0]})
    out = m.apply_transforms(df, parse(m, "--normalize"), ["v"])
    assert out["v"].tolist() == [0.0, 0.0, 0.0]


def test_normalize_all_nan_warns_and_stays_nan(m, capsys):
    df = pd.DataFrame({"v": [float("nan"), float("nan")]})
    out = m.apply_transforms(df, parse(m, "--normalize"), ["v"])
    assert out["v"].isna().all()
    assert "normalize skipped" in capsys.readouterr().err


def test_zscore_basic(m):
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = m.apply_transforms(df, parse(m, "--zscore"), ["v"])
    assert out["v"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["v"].std() == pytest.approx(1.0)


def test_zscore_constant_column_maps_to_zero_with_warning(m, capsys):
    df = pd.DataFrame({"v": [5.0, 5.0, 5.0]})
    out = m.apply_transforms(df, parse(m, "--zscore"), ["v"])
    assert out["v"].tolist() == [0.0, 0.0, 0.0]
    assert "z-score maps to 0" in capsys.readouterr().err


def test_coercion_failure_warns(m, capsys):
    df = pd.DataFrame({"v": ["1", "x", "3"]})
    m.apply_transforms(df, parse(m, "--cumsum"), ["v"])
    assert "could not be coerced" in capsys.readouterr().err


def test_missing_column_is_skipped_not_erroring(m):
    df = pd.DataFrame({"v": [1, 2, 3]})
    out = m.apply_transforms(df, parse(m, "--cumsum"), ["v", "nonexistent"])
    assert out["v"].tolist() == [1, 3, 6]  # only 'v' transformed; no KeyError


# ── apply_transforms: chaining / interaction effects ──────────────────────

def test_cumsum_then_normalize_operates_on_cumsum_result(m):
    # transforms chain: normalize should scale the CUMSUM output, not the
    # original column — verifies 's' is correctly threaded between steps
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    out = m.apply_transforms(df, parse(m, "--cumsum", "--normalize"), ["v"])
    # cumsum -> [1,3,6,10]; normalize -> [(1-1)/9, (3-1)/9, (6-1)/9, (10-1)/9]
    assert out["v"].tolist() == pytest.approx([0.0, 2 / 9, 5 / 9, 1.0])


def test_baseline_then_normalize(m):
    df = pd.DataFrame({"v": [10.0, 15.0, 20.0]})
    out = m.apply_transforms(df, parse(m, "--baseline", "--normalize"), ["v"])
    # baseline -> [0,5,10]; normalize -> [0, 0.5, 1.0]
    assert out["v"].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_clip_then_baseline(m):
    df = pd.DataFrame({"v": [-10.0, 0.0, 5.0, 100.0]})
    out = m.apply_transforms(df, parse(m, "--clip", "0", "10", "--baseline"), ["v"])
    # clip -> [0,0,5,10]; baseline subtracts first non-nan (0) -> unchanged
    assert out["v"].tolist() == pytest.approx([0.0, 0.0, 5.0, 10.0])


def test_diff_then_clip(m):
    df = pd.DataFrame({"v": [1.0, 10.0, 2.0, 20.0]})
    out = m.apply_transforms(df, parse(m, "--diff", "--clip", "-3", "3"), ["v"])
    # diff -> [nan, 9, -8, 18]; clip to [-3,3] -> [nan, 3, -3, 3]
    assert math.isnan(out["v"].iloc[0])
    assert out["v"].iloc[1:].tolist() == pytest.approx([3.0, -3.0, 3.0])


def test_rolling_then_zscore(m):
    df = pd.DataFrame({"v": [1.0, 100.0, 2.0, 100.0, 3.0, 100.0]})
    out = m.apply_transforms(df, parse(m, "--rolling", "2", "--zscore"), ["v"])
    assert out["v"].mean() == pytest.approx(0.0, abs=1e-9)
    assert out["v"].std() == pytest.approx(1.0)


def test_full_chain_does_not_raise(m):
    """All chainable transforms enabled at once on well-behaved data must
    not raise — establishes a baseline for the full combinatorial space."""
    df = pd.DataFrame({"v": [float(i % 5 + 1) for i in range(30)]})
    args = parse(m, "--cumsum", "--rolling", "3", "--smooth", "1.0",
                 "--clip", "-1000", "1000", "--baseline", "--normalize")
    out = m.apply_transforms(df, args, ["v"])
    assert len(out) == 30
    assert out["v"].max() <= 1.0 + 1e-9
    assert out["v"].min() >= 0.0 - 1e-9


# ── mutation behavior (shared-state check) ────────────────────────────────

def test_apply_transforms_mutates_input_df_in_place(m):
    """
    Documents actual behavior: apply_transforms both mutates its `df`
    argument's columns AND returns the same object. Callers that still hold
    a reference to the pre-transform frame will see it change too. This is
    safe in the current pipeline (call sites don't reuse the pre-transform
    reference) but is a sharp edge for future refactors — this test exists
    so it fails loudly if that assumption ever breaks silently.
    """
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    out = m.apply_transforms(df, parse(m, "--cumsum"), ["v"])
    assert out is df
    assert df["v"].tolist() == [1.0, 3.0, 6.0]  # original object itself changed
