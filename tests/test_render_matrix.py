"""
Render matrix: every plot mode against every modifier, drawn for real.

The mode-dispatch tests in test_plotting.py mock seaborn, so they verify that
a call is made with the expected keywords -- not that the plotting backend
accepts them. That distinction hid a crash where `--hue` on a line or scatter
plot raised a TypeError from inside matplotlib, because an explicit label was
passed alongside hue.

These tests close that gap by rendering with a real backend. They run
in-process rather than through the CLI: a subprocess costs well over a second
apiece and this matrix is large. The contract asserted is that a run either
succeeds or exits cleanly through die() -- no other exception may escape,
whatever the backend does internally.
"""
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODES = {
    "line": [], "scatter": ["--scatter"], "regplot": ["--regplot"],
    "hist": ["--hist"], "kde": ["--kde"], "box": ["--box"],
    "violin": ["--violin"], "bar": ["--bar"], "swarm": ["--swarm"],
    "corr": ["--corr"], "pairplot": ["--pairplot"],
}

MODIFIERS = {
    "plain": ["--x", "x", "--y", "y"],
    "hue": ["--x", "x", "--y", "y", "--hue", "g"],
    "dual-y": ["--x", "x", "--y", "y", "--dual-y", "z"],
    "multi-ys": ["--x", "x", "--ys", "y", "z"],
    "subplots": ["--x", "x", "--ys", "y", "z", "--subplots"],
    "fit": ["--x", "x", "--y", "y", "--fit", "linear"],
    "group-by": ["--y", "y", "--group-by", "g", "--agg", "mean"],
    "styled": ["--x", "x", "--y", "y", "--alpha", "0.3", "--marker", "o",
               "--linewidth", "3", "--grid", "--legend"],
}

# Degenerate frames. Each has previously broken, or plausibly could break,
# something inside the plotting backend.
SHAPES = {
    "normal":     [(i, i * 7 % 13 + 1, i * 3 % 5 + 100, "ab"[i % 2]) for i in range(24)],
    "single_row": [(1, 5, 9, "a")],
    "constant_y": [(i, 7, 9, "ab"[i % 2]) for i in range(10)],
    "all_nan_y":  [(i, None, 9, "ab"[i % 2]) for i in range(10)],
    "negatives":  [(i, i - 5, 9, "ab"[i % 2]) for i in range(10)],
    "one_x":      [(1, i + 1, 9, "ab"[i % 2]) for i in range(10)],
}


def _write(tmp_path, rows):
    lines = ["x,y,z,g"]
    lines += [",".join("" if v is None else str(v) for v in r) for r in rows]
    p = tmp_path / "d.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def _render(m, monkeypatch, csv_path, argv, out):
    """
    Drive a full run in-process. Returns True if it completed, False if it
    exited cleanly through die(); any other exception propagates and fails
    the test.
    """
    monkeypatch.setattr(sys, "argv",
                        ["csvplot", str(csv_path), *argv, "--output", str(out), "--force"])
    try:
        m.main()
        return True
    except SystemExit:
        return False
    finally:
        plt.close("all")


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("modifier", sorted(MODIFIERS))
def test_mode_and_modifier_render_or_fail_cleanly(m, tmp_path, monkeypatch, mode, modifier):
    csv_path = _write(tmp_path, SHAPES["normal"])
    _render(m, monkeypatch, csv_path,
            [*MODIFIERS[modifier], *MODES[mode]], tmp_path / "o.png")


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("mode", sorted(MODES))
def test_degenerate_data_renders_or_fails_cleanly(m, tmp_path, monkeypatch, shape, mode):
    csv_path = _write(tmp_path, SHAPES[shape])
    _render(m, monkeypatch, csv_path,
            ["--x", "x", "--y", "y", "--hue", "g", *MODES[mode]], tmp_path / "o.png")


def test_hue_on_a_line_plot_actually_draws(m, tmp_path, monkeypatch):
    """
    The specific regression: seaborn supplies its own per-level label when hue
    is set, and an explicit label alongside it raised a TypeError.
    """
    csv_path = _write(tmp_path, SHAPES["normal"])
    out = tmp_path / "o.png"
    assert _render(m, monkeypatch, csv_path, ["--x", "x", "--y", "y", "--hue", "g"], out)
    assert out.exists()


def test_all_missing_column_with_hue_reports_cleanly(m, tmp_path, monkeypatch, capsys):
    """
    An entirely missing column drove seaborn's boxplot into an internal
    UnboundLocalError. Backend failures of any type must surface as a named
    diagnostic rather than a traceback.
    """
    csv_path = _write(tmp_path, SHAPES["all_nan_y"])
    assert not _render(m, monkeypatch, csv_path,
                       ["--x", "x", "--y", "y", "--hue", "g", "--box"], tmp_path / "o.png")
    err = capsys.readouterr().err
    assert "could not render" in err and "'y'" in err
