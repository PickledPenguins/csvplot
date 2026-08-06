"""
Option ordering and combination tests.

Three properties are checked here that the per-option tests do not reach:

1. Command-line order is irrelevant. The pipeline order is fixed in code, and
   the documentation states plainly that `--diff --normalize` and
   `--normalize --diff` are the same run. That claim needs a test.
2. Row operations compose in the documented order for every subset, not just
   the four pairs that were spot-checked. The expected result is computed by
   an independent reference implementation rather than by restating the
   tool's own output.
3. Transforms chain without raising for every pair, and rejection of a bad
   value survives being surrounded by valid options.
"""
import itertools

import pandas as pd
import pytest


def parse(m, *argv):
    return m.build_parser().parse_args(["data.csv", "--y", "v", *argv])


def frame(n=12):
    return pd.DataFrame({"v": [float((i * 7) % 11 + 1) for i in range(n)],
                         "w": [float(i) for i in range(n)]})


# ── 1. command-line order does not change the result ───────────────────────

# Each entry is a list of complete flag groups. Permuting the groups must not
# change the outcome; splitting a flag from its value would not be a valid
# command line, so groups move as units.
ORDER_CASES = [
    [["--diff"], ["--normalize"]],
    [["--cumsum"], ["--rolling", "3"], ["--baseline"]],
    [["--clip", "2", "8"], ["--zscore"], ["--smooth", "1"]],
    [["--rate"], ["--cummax"], ["--ewm", "3"]],
]


@pytest.mark.parametrize("groups", ORDER_CASES, ids=lambda g: "+".join(x[0] for x in g))
def test_transform_result_is_independent_of_flag_order(m, groups):
    results = []
    for perm in itertools.permutations(groups):
        argv = [tok for group in perm for tok in group]
        out = m.apply_transforms(frame(), parse(m, *argv), ["v"])
        results.append(out["v"].tolist())
    first = results[0]
    for other in results[1:]:
        assert other == pytest.approx(first, nan_ok=True)


ROW_ORDER_CASES = [
    [["--head", "8"], ["--tail", "4"]],
    [["--filter", "v > 3"], ["--sort", "v"], ["--every", "2"]],
    [["--head", "10"], ["--sort", "v"], ["--tail", "3"]],
]


@pytest.mark.parametrize("groups", ROW_ORDER_CASES, ids=lambda g: "+".join(x[0] for x in g))
def test_row_op_result_is_independent_of_flag_order(m, groups):
    results = []
    for perm in itertools.permutations(groups):
        argv = [tok for group in perm for tok in group]
        out = m.apply_row_ops(frame(), parse(m, *argv, "--seed", "1"))
        results.append(out["v"].tolist())
    for other in results[1:]:
        assert other == results[0]


# ── 2. every subset of row operations follows the documented order ─────────

ROW_OPS = {
    "head":   (["--head", "9"],        lambda df: df.head(9)),
    "tail":   (["--tail", "6"],        lambda df: df.tail(6)),
    "filter": (["--filter", "v > 3"],  lambda df: df.query("v > 3")),
    "sort":   (["--sort", "v"],        lambda df: df.sort_values("v")),
    "sample": (["--sample", "4"],      lambda df: df.sample(min(4, len(df)), random_state=1)),
    "every":  (["--every", "2"],       lambda df: df.iloc[::2]),
}
ORDER = ["head", "tail", "filter", "sort", "sample", "every"]


def _reference(df, chosen):
    """Independent implementation of the documented head->...->every order."""
    for name in ORDER:                      # fixed order, not the caller's order
        if name in chosen:
            df = ROW_OPS[name][1](df)
    return df.reset_index(drop=True) if chosen else df


@pytest.mark.parametrize("size", range(len(ORDER) + 1))
def test_every_row_op_subset_matches_the_documented_order(m, size):
    """
    All 2**6 subsets, grouped by size so a failure names how many options were
    combined. Flags are passed in reverse of the pipeline order to confirm the
    result follows the pipeline, not the command line.
    """
    df = frame(14)
    for chosen in itertools.combinations(ORDER, size):
        argv = [tok for name in reversed(chosen) for tok in ROW_OPS[name][0]]
        out = m.apply_row_ops(df, parse(m, *argv, "--seed", "1"))
        expected = _reference(df, set(chosen))
        assert out["v"].tolist() == expected["v"].tolist(), f"subset={chosen}"


# ── 3. transforms chain in pairs without raising ───────────────────────────

TRANSFORMS = [
    ["--cumsum"], ["--cummax"], ["--cummin"], ["--diff"], ["--rate"],
    ["--ewm", "3"], ["--rolling", "3"], ["--smooth", "1"],
    ["--clip", "2", "8"], ["--baseline"], ["--normalize"], ["--zscore"],
]


@pytest.mark.parametrize("first", range(len(TRANSFORMS)))
def test_transform_pairs_chain_cleanly(m, first):
    """
    Every ordered pair of transforms. --normalize and --zscore are mutually
    exclusive and rejected by validation, so that one pair is skipped.
    """
    df = frame(20)
    a = TRANSFORMS[first]
    for b in TRANSFORMS:
        if a is b:
            continue
        if {a[0], b[0]} == {"--normalize", "--zscore"}:
            continue
        out = m.apply_transforms(df, parse(m, *a, *b), ["v"])
        assert len(out) == len(df), f"{a[0]} + {b[0]} changed row count"
        assert "v" in out.columns


def test_all_transforms_at_once(m):
    """Every chainable transform enabled together, bar the exclusive pair."""
    argv = [tok for t in TRANSFORMS if t[0] != "--zscore" for tok in t]
    out = m.apply_transforms(frame(30), parse(m, *argv), ["v"])
    assert len(out) == 30


def test_row_ops_and_transforms_combine(m):
    """Row operations run first, so transforms see the reduced frame."""
    df = frame(20)
    args = parse(m, "--head", "10", "--every", "2", "--cumsum", "--seed", "1")
    rows = m.apply_row_ops(df, args)
    # captured before the call: apply_transforms writes into the frame it is
    # given rather than a copy, so reading it afterwards sees the result
    expected = rows["v"].cumsum().tolist()
    out = m.apply_transforms(rows, args, ["v"])
    assert len(out) == 5
    assert out["v"].tolist() == expected


def test_apply_transforms_writes_into_the_frame_it_is_given(m):
    """
    Documents an aliasing property callers must know: the input frame is
    modified, not copied. Every caller reassigns the return value, so this is
    not a defect, but a frame kept for comparison must be copied first.
    """
    df = frame(6)
    before = df["v"].tolist()
    out = m.apply_transforms(df, parse(m, "--cumsum"), ["v"])
    assert out["v"].tolist() == df["v"].tolist()
    assert df["v"].tolist() != before


# ── 4. a bad value is still rejected when surrounded by valid options ──────

VALID_CONTEXT = ["--smooth", "1", "--sort", "v", "--head", "10",
                 "--title", "t", "--alpha", "0.5", "--grid"]

BAD_VALUES = [
    ["--bins", "0"], ["--dpi", "-1"], ["--alpha", "5"], ["--ci", "150"],
    ["--rolling", "0"], ["--ewm", "0"], ["--smooth", "-1"], ["--every", "0"],
    ["--head", "-1"], ["--sample", "0"], ["--figsize", "0x0"],
    ["--clip", "9", "1"], ["--percentile", "101"],
]


@pytest.mark.parametrize("bad", BAD_VALUES, ids=lambda b: b[0].lstrip("-"))
def test_invalid_value_is_rejected_among_valid_options(m, run_cli, csv_numeric, bad):
    """
    Validation must not be order- or neighbour-sensitive: the bad option is
    placed after a run of valid ones, and the run must still be refused with a
    clean message rather than a traceback.
    """
    r = run_cli(str(csv_numeric), "--y", "b", *VALID_CONTEXT, *bad,
                "--output", "o.png", "--force")
    assert r.returncode == 1, f"{bad} was accepted"
    assert "Traceback" not in r.stderr
    assert r.stderr.strip()
