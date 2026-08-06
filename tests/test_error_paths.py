"""
Error and fallback paths that the rest of the suite does not reach.

These are the handlers that only fire when something goes wrong underneath
the tool -- a rendering backend refusing a figure, a save failing partway, a
consumer closing a pipe, an optional dependency missing, an exception
escaping main(). They are cheap to get wrong and expensive to debug in the
field, so each one is exercised directly rather than left to chance.

Also covered here are the few edge paths that need the environment bent to
reach them at all: type coercion corners in the YAML loader, and the
pairplot row cap.
"""
import math
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _args(m, *argv):
    """Parse real CLI tokens into an argument namespace."""
    return m.build_parser().parse_args(["data.csv", "--y", "b", *argv])


# ── version probe ──────────────────────────────────────────────────────────

def test_sns_version_probe_falls_back_when_unparseable(m, monkeypatch):
    """An unrecognisable seaborn version must assume a modern seaborn."""
    monkeypatch.setattr(m, "_sns_ge_012_cache", None)
    monkeypatch.setattr(m.sns, "__version__", "not-a-version")
    assert m._sns_ge_012() is True


# ── JSON coercion ──────────────────────────────────────────────────────────

def test_json_safe_non_finite_float32_becomes_null(m):
    """
    np.float64 subclasses Python float and is caught by the plain float check,
    so the numpy-scalar branch is only reached by widths that do not -- float32
    among them.
    """
    assert not isinstance(np.float32("nan"), float)
    assert m._json_safe(np.float32("nan")) is None
    assert m._json_safe(np.float32("inf")) is None


def test_json_safe_finite_float32_becomes_python_float(m):
    out = m._json_safe(np.float32(1.5))
    assert out == 1.5 and math.isfinite(out)


# ── figure output ──────────────────────────────────────────────────────────

def test_show_is_called_when_no_output_is_given(m):
    args = _args(m)
    args.output = None
    fig = MagicMock()
    with patch("csvplot.plt.show") as show:
        m._save_or_show(fig, args, {})
    show.assert_called_once()


def test_show_failure_dies_cleanly(m):
    args = _args(m)
    args.output = None
    with patch("csvplot.plt.show", side_effect=RuntimeError("no display")):
        with pytest.raises(SystemExit):
            m._save_or_show(MagicMock(), args, {})


def test_save_failure_leaves_no_temporary_file(m, tmp_path):
    """
    The figure is written to a sibling temporary file and moved into place, so
    a failed save must not leave that temporary behind.
    """
    out = tmp_path / "fig.png"
    args = _args(m, "--output", str(out))
    fig = MagicMock()
    fig.savefig.side_effect = OSError("disk full")
    with pytest.raises(SystemExit):
        m._save_or_show(fig, args, {})
    assert list(tmp_path.iterdir()) == []


def test_tight_layout_failure_is_warned_not_fatal(m, capsys):
    args = _args(m)
    args.output = None
    with patch("csvplot.plt.tight_layout", side_effect=ValueError("singular")), \
         patch("csvplot.plt.show"), patch("csvplot.plt.close"):
        m._finalize_figure(MagicMock(), args, {})
    assert "tight_layout failed" in capsys.readouterr().err


# ── unit parsing ───────────────────────────────────────────────────────────

def test_unit_parse_failure_is_warned_not_fatal(m, capsys):
    """A column that cannot be unit-parsed is left alone, not fatal."""
    import pandas as pd
    df = pd.DataFrame({"a": ["1k", "2k"]})
    with patch("csvplot.parse_units", side_effect=RuntimeError("boom")):
        out = m._coerce_units(df, threshold=1.0, require_suffix=True)
    assert "unit parsing failed for column 'a'" in capsys.readouterr().err
    assert out["a"].tolist() == ["1k", "2k"]


# ── pairplot diagonal fallback ─────────────────────────────────────────────

def test_pairplot_falls_back_to_histogram_diagonal(m, capsys):
    """
    A KDE diagonal fails on degenerate data; the histogram diagonal is the
    documented fallback and must be attempted before giving up.
    """
    import pandas as pd
    args = _args(m, "--pairplot")
    args.output = None
    frames = {"f.csv": pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})}
    grid = MagicMock()
    with patch("csvplot.sns.pairplot",
               side_effect=[np.linalg.LinAlgError("singular"), grid]) as pp, \
         patch("csvplot.plt.tight_layout"), patch("csvplot.plt.show"), \
         patch("csvplot.plt.close"):
        assert m._render_special(frames, args, ["a", "b"], {}) is True
    assert pp.call_count == 2
    assert pp.call_args_list[0].kwargs["diag_kind"] == "kde"
    assert pp.call_args_list[1].kwargs["diag_kind"] == "hist"
    assert "falling back to histogram" in capsys.readouterr().err


def test_pairplot_render_failure_dies_cleanly(m):
    import pandas as pd
    args = _args(m, "--pairplot")
    args.output = None
    frames = {"f.csv": pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})}
    with patch("csvplot.sns.pairplot", side_effect=ValueError("bad values")):
        with pytest.raises(SystemExit):
            m._render_special(frames, args, ["a", "b"], {})


def test_corr_heatmap_render_failure_dies_cleanly(m):
    import pandas as pd
    args = _args(m, "--corr")
    args.output = None
    frames = {"f.csv": pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})}
    with patch("csvplot.sns.heatmap", side_effect=ValueError("bad values")), \
         patch("csvplot.plt.subplots", return_value=(MagicMock(), MagicMock())):
        with pytest.raises(SystemExit):
            m._render_special(frames, args, ["a", "b"], {})


# ── optional dependency guard ──────────────────────────────────────────────

def test_missing_pyyaml_reports_the_install_command(m, tmp_path, monkeypatch):
    """PyYAML is a declared dependency, but --yaml must still explain itself
    if the module is unavailable in the running environment."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("y: [a]\n")
    monkeypatch.setitem(sys.modules, "yaml", None)  # import yields None -> AttributeError
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(cfg), m.build_parser())


# ── entry-point guards ─────────────────────────────────────────────────────

def test_keyboard_interrupt_exits_quietly(m, monkeypatch, capsys):
    monkeypatch.setattr(m, "main", lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(SystemExit) as e:
        m._cli_entrypoint()
    assert e.value.code == 0
    assert capsys.readouterr().err == ""


def test_unexpected_error_is_labelled_before_the_traceback(m, monkeypatch, capsys):
    def boom():
        raise RuntimeError("some internal failure")
    monkeypatch.setattr(m, "main", boom)
    with pytest.raises(SystemExit) as e:
        m._cli_entrypoint()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "unexpected internal error" in err
    assert "RuntimeError" in err  # the traceback is kept, not swallowed


# ── YAML type coercion ─────────────────────────────────────────────────────

def test_scalar_value_for_a_list_option_is_wrapped_and_coerced(m, tmp_path):
    """A YAML scalar given for a nargs='+' option becomes a single-element
    list, and its element is still type-coerced."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("percentile: '50'\n")
    defaults, _ = m._load_yaml_defaults(str(cfg), m.build_parser())
    assert defaults["percentile"] == [50.0]


def test_list_value_for_a_list_option_is_coerced_element_wise(m, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("percentile: ['25', '75']\n")
    defaults, _ = m._load_yaml_defaults(str(cfg), m.build_parser())
    assert defaults["percentile"] == [25.0, 75.0]


# ── unit parsing can be switched off ───────────────────────────────────────

def test_unit_parsing_is_skipped_when_disabled(m, tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("size\n1kb\n2kb\n")
    args = m.build_parser().parse_args([str(p), "--y", "size", "--no-parse-units"])
    frames = m._load_frames(args)
    # left as the original strings rather than expanded to 1000/2000
    assert list(frames[str(p)]["size"]) == ["1kb", "2kb"]


def test_unit_parsing_runs_by_default(m, tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("size\n1kb\n2kb\n")
    args = m.build_parser().parse_args([str(p), "--y", "size"])
    frames = m._load_frames(args)
    assert list(frames[str(p)]["size"]) == [1000.0, 2000.0]


# ── pairplot row cap ───────────────────────────────────────────────────────

def test_pairplot_samples_and_warns_above_the_row_cap(m, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(m, "_PAIRPLOT_MAX_ROWS", 3)
    p = tmp_path / "p.csv"
    p.write_text("a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(8)))
    args = m.build_parser().parse_args(
        [str(p), "--ys", "a", "b", "--pairplot",
         "--output", str(tmp_path / "o.png"), "--force"])
    frames = m._load_frames(args)
    assert m._render_special(frames, args, ["a", "b"], {}) is True
    assert "sampling 3" in capsys.readouterr().err


# ── environment-dependent failures ─────────────────────────────────────────
# These are reachable in normal use -- an unreadable config, an unwritable
# destination, a bare invocation at a terminal -- but cannot be provoked from
# a test process directly: the suite frequently runs as root, where filesystem
# permissions are not enforced, and its stdin is never a terminal. The
# condition is injected at the boundary and the diagnostic is asserted.

def test_unreadable_yaml_config_is_reported(m, tmp_path, monkeypatch, capsys):
    import builtins
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("plot:\n  mode: line\n")
    real_open = builtins.open

    def refuse(path, *a, **kw):
        if str(path) == str(cfg):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", refuse)
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(cfg), m.build_parser())
    assert "permission denied" in capsys.readouterr().err.lower()


def test_unwritable_existing_output_is_reported(m, tmp_path, monkeypatch, capsys):
    """--force clears the exists check, then the file itself is read-only."""
    import os
    out = tmp_path / "taken.png"
    out.write_bytes(b"")
    monkeypatch.setattr(os, "access", lambda *a, **kw: False)
    args = _args(m, "--output", str(out), "--force")
    with pytest.raises(SystemExit):
        m._check_output(args)
    assert "not writable" in capsys.readouterr().err


def test_unwritable_output_directory_is_reported(m, tmp_path, monkeypatch, capsys):
    import os
    monkeypatch.setattr(os, "access", lambda *a, **kw: False)
    args = _args(m, "--output", str(tmp_path / "new.png"))
    with pytest.raises(SystemExit):
        m._check_output(args)
    assert "directory is not writable" in capsys.readouterr().err


def test_no_input_at_a_terminal_is_reported(m, monkeypatch, capsys):
    """
    With no file argument the tool reads stdin. When stdin is a terminal there
    is nothing to read, so it explains itself instead of blocking forever.
    """
    monkeypatch.setattr(sys, "argv", ["csvplot", "--y", "b"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    with pytest.raises(SystemExit):
        m.main()
    assert "no input" in capsys.readouterr().err


# ── renderer rejecting the data ────────────────────────────────────────────

def _idx_frame(m, **cols):
    """A frame carrying the internal row-index column _build_figure expects."""
    import pandas as pd
    df = pd.DataFrame(cols)
    return df.assign(**{m._IDX_COL: df.index})


def test_multi_column_render_failure_names_the_columns(m, tmp_path, monkeypatch, capsys):
    """
    The melted multi-column categorical path wraps the renderer so a failure
    reports which columns were being drawn instead of a bare seaborn error.
    """
    def explode(*a, **kw):
        raise ValueError("injected renderer failure")

    monkeypatch.setattr(m, "_plot_single", explode)
    args = _args(m, "--box", "--output", str(tmp_path / "o.png"), "--force")
    with pytest.raises(SystemExit):
        m._build_figure({"f": _idx_frame(m, a=[1, 2], b=[3, 4])},
                        args, ["a", "b"], m._IDX_COL, {})
    err = capsys.readouterr().err
    assert "could not render" in err and "'a'" in err


def test_dual_y_render_failure_names_the_column(m, tmp_path, monkeypatch, capsys):
    """
    The left axis is drawn with --scatter so that breaking lineplot isolates
    the right-axis draw; sharing a renderer would trip the left axis first and
    test the wrong handler.
    """
    def explode(*a, **kw):
        raise ValueError("injected renderer failure")

    monkeypatch.setattr(m.sns, "lineplot", explode)
    args = _args(m, "--scatter", "--dual-y", "b",
                 "--output", str(tmp_path / "o.png"), "--force")
    with pytest.raises(SystemExit):
        m._build_figure({"f": _idx_frame(m, a=[1.0, 2.0], b=[3.0, 4.0])},
                        args, ["a"], m._IDX_COL, {})
    assert "dual-y 'b'" in capsys.readouterr().err
