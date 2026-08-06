"""
Axis-decoration tests: reference lines, log scaling, grid, axis limits and
tick rotation.

These exercise _decorate_axes against a real matplotlib Axes rather than
asserting only on process exit status, so the effect of each option is
actually observed. A small set of end-to-end renders backs them up, since
the unit layer cannot catch argparse wiring mistakes.
"""
import matplotlib
import pytest

matplotlib.use("Agg")  # no display available in CI; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture
def ax():
    """A real Axes carrying strictly positive data, closed after each test."""
    fig, a = plt.subplots()
    a.plot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    yield a
    plt.close(fig)


@pytest.fixture
def ax_with_zero():
    """A real Axes whose data starts at zero, so log scaling must be refused."""
    fig, a = plt.subplots()
    a.plot([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    yield a
    plt.close(fig)


def _args(m, *argv):
    """Build a parsed argument namespace from real CLI tokens."""
    return m.build_parser().parse_args(["data.csv", "--y", "b", *argv])


# ── reference lines ────────────────────────────────────────────────────────

def test_hline_draws_one_line_per_value(m, ax):
    before = len(ax.lines)
    m._decorate_axes(ax, _args(m, "--hline", "1.0", "2.5"))
    assert len(ax.lines) == before + 2


def test_vline_draws_one_line_per_value(m, ax):
    before = len(ax.lines)
    m._decorate_axes(ax, _args(m, "--vline", "0.5"))
    assert len(ax.lines) == before + 1


def test_hline_style_overrides_defaults(m, ax):
    m._decorate_axes(ax, _args(m, "--hline", "1.0"),
                     {"hline_style": {"color": "red", "linewidth": 3.0}})
    line = ax.lines[-1]
    assert line.get_linewidth() == 3.0
    assert matplotlib.colors.to_hex(line.get_color()) == "#ff0000"


# ── log scaling ────────────────────────────────────────────────────────────

def test_log_x_applied_when_data_is_positive(m, ax):
    m._decorate_axes(ax, _args(m, "--log-x"))
    assert ax.get_xscale() == "log"


def test_log_y_applied_when_data_is_positive(m, ax):
    m._decorate_axes(ax, _args(m, "--log-y"))
    assert ax.get_yscale() == "log"


def test_log_x_skipped_and_warned_on_non_positive_data(m, ax_with_zero, capsys):
    m._decorate_axes(ax_with_zero, _args(m, "--log-x"))
    assert ax_with_zero.get_xscale() == "linear"
    assert "--log-x skipped" in capsys.readouterr().err


def test_log_y_skipped_and_warned_on_non_positive_data(m, ax_with_zero, capsys):
    m._decorate_axes(ax_with_zero, _args(m, "--log-y"))
    assert ax_with_zero.get_yscale() == "linear"
    assert "--log-y skipped" in capsys.readouterr().err


# ── grid ───────────────────────────────────────────────────────────────────

def test_grid_enables_gridlines(m, ax):
    m._decorate_axes(ax, _args(m, "--grid"))
    assert ax.xaxis.get_gridlines()[0].get_visible()


def test_grid_style_overrides_apply(m, ax):
    m._decorate_axes(ax, _args(m, "--grid"),
                     {"grid_alpha": 0.25, "grid_linestyle": ":", "grid_color": "red"})
    gridline = ax.xaxis.get_gridlines()[0]
    assert gridline.get_alpha() == 0.25
    assert matplotlib.colors.to_hex(gridline.get_color()) == "#ff0000"


def test_grid_absent_by_default(m, ax):
    m._decorate_axes(ax, _args(m))
    assert not ax.xaxis.get_gridlines()[0].get_visible()


# ── limits and tick rotation (YAML-only options) ───────────────────────────

def test_axis_limits_applied(m, ax):
    m._decorate_axes(ax, _args(m), {"xlim": [0, 10], "ylim": [-5, 5]})
    assert ax.get_xlim() == (0.0, 10.0)
    assert ax.get_ylim() == (-5.0, 5.0)


def test_tick_rotation_applied(m, ax):
    m._decorate_axes(ax, _args(m), {"xtick_rotation": 30, "ytick_rotation": 45})
    assert ax.get_xticklabels()[0].get_rotation() == 30
    assert ax.get_yticklabels()[0].get_rotation() == 45


def test_zero_rotation_is_not_treated_as_absent(m, ax):
    """0 is a meaningful rotation; a truthiness check would silently drop it."""
    fig, other = plt.subplots()
    try:
        other.plot([1, 2], [1, 2])
        other.tick_params(axis="x", labelrotation=60)
        m._decorate_axes(other, _args(m), {"xtick_rotation": 0})
        assert other.get_xticklabels()[0].get_rotation() == 0
    finally:
        plt.close(fig)


# ── end-to-end backing ─────────────────────────────────────────────────────

@pytest.mark.parametrize("flags", [
    ["--hline", "5"],
    ["--vline", "10"],
    ["--grid"],
    ["--log-y"],
    ["--hline", "5", "--vline", "10", "--grid", "--log-x", "--log-y"],
])
def test_decoration_flags_render_end_to_end(run_cli, csv_numeric, flags, tmp_path):
    r = run_cli(str(csv_numeric), "--y", "b", *flags, "--output", "o.png", "--force")
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr
    assert (tmp_path / "o.png").exists()
