"""
Tests for the YAML configuration surface: _flatten_yaml_schema (nested
user-facing schema -> flat internal keys) and _load_yaml_defaults (full
load + coerce + validate pipeline). Flagged as a deep-review area since
it's the most complex, least-typed part of the tool.
"""
import pytest


# ── _flatten_yaml_schema: per-section mapping ─────────────────────────────

def test_flatten_data_section(m):
    raw = {"data": {"file": "d.csv", "x": "t", "hue": "grp", "rolling": 5}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["file"] == "d.csv"
    assert flat["x"] == "t"
    assert flat["hue"] == "grp"
    assert flat["rolling"] == 5


def test_flatten_data_y_becomes_ys(m):
    flat = m._flatten_yaml_schema({"data": {"y": ["a", "b"]}})
    assert flat["ys"] == ["a", "b"]
    assert "y" not in flat


def test_flatten_legacy_flat_y_scalar_becomes_ys_list(m):
    # no 'data:' wrapper — legacy flat schema; scalar y must still become a list
    # since 'ys' is the nargs='+' CLI dest, not the scalar '--y' dest
    flat = m._flatten_yaml_schema({"y": "single_col"})
    assert flat["ys"] == ["single_col"]


def test_flatten_legacy_flat_ys_passthrough(m):
    flat = m._flatten_yaml_schema({"ys": ["a", "b"]})
    assert flat["ys"] == ["a", "b"]


def test_flatten_legacy_y_ignored_when_ys_also_present(m):
    flat = m._flatten_yaml_schema({"y": "a", "ys": ["b", "c"]})
    assert flat["ys"] == ["b", "c"]


def test_flatten_x_axis_section(m):
    raw = {"x_axis": {"label": "Time", "lim": [0, 10], "log": True,
                       "rotation": 45, "vline": [1, 2], "vline_style": {"color": "red"}}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["xlabel"] == "Time"
    assert flat["xlim"] == [0, 10]
    assert flat["log_x"] is True
    assert flat["xtick_rotation"] == 45
    assert flat["vline"] == [1, 2]
    assert flat["vline_style"] == {"color": "red"}


def test_flatten_x_axis_log_false_omits_flag(m):
    flat = m._flatten_yaml_schema({"x_axis": {"log": False}})
    assert "log_x" not in flat


def test_flatten_y_axis_section_including_style_lists(m):
    raw = {"y_axis": {"label": "Val", "lim": [-1, 1], "log": True, "rotation": 30,
                       "hline": [0], "hline_style": {"color": "grey"},
                       "colors": ["red", "blue"], "linestyles": ["-", "--"],
                       "linewidths": [1, 2], "alphas": [1.0, 0.5], "markers": ["o", None]}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["ylabel"] == "Val"
    assert flat["ylim"] == [-1, 1]
    assert flat["log_y"] is True
    assert flat["ytick_rotation"] == 30
    assert flat["hline"] == [0]
    assert flat["hline_style"] == {"color": "grey"}
    assert flat["y_colors"] == ["red", "blue"]
    assert flat["y_linestyles"] == ["-", "--"]
    assert flat["y_linewidths"] == [1, 2]
    assert flat["y_alphas"] == [1.0, 0.5]
    assert flat["y_markers"] == ["o", None]


def test_flatten_right_axis_section(m):
    raw = {"right_axis": {"columns": ["ref"], "label": "Reference", "lim": [0, 100],
                           "colors": ["crimson"], "linestyles": [":"]}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["dual_y"] == ["ref"]
    assert flat["dual_y_label"] == "Reference"
    assert flat["dual_y_lim"] == [0, 100]
    assert flat["dual_y_colors"] == ["crimson"]
    assert flat["dual_y_linestyles"] == [":"]


def test_flatten_plot_mode_becomes_boolean_flag(m):
    flat = m._flatten_yaml_schema({"plot": {"mode": "hist"}})
    assert flat["hist"] is True


def test_flatten_plot_mode_line_needs_no_flag(m):
    flat = m._flatten_yaml_schema({"plot": {"mode": "line"}})
    assert "line" not in flat
    assert not any(flat.get(k) for k in ["scatter", "hist", "kde", "box", "violin",
                                          "bar", "swarm", "regplot", "corr", "pairplot"])


def test_flatten_plot_mode_invalid_dies(m):
    with pytest.raises(SystemExit):
        m._flatten_yaml_schema({"plot": {"mode": "not_a_mode"}})


def test_flatten_plot_figsize_list_to_string(m):
    flat = m._flatten_yaml_schema({"plot": {"figsize": [12, 8]}})
    assert flat["figsize"] == "12,8"


def test_flatten_plot_figsize_wrong_shape_dies(m):
    with pytest.raises(SystemExit):
        m._flatten_yaml_schema({"plot": {"figsize": [12, 8, 4]}})


def test_flatten_plot_mode_specific_options_prefixed(m):
    raw = {"plot": {"mode": "hist", "bins": 30, "stat": "density", "multiple": "stack"}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["bins"] == 30
    assert flat["hist_stat"] == "density"
    assert flat["hist_multiple"] == "stack"


def test_flatten_plot_corr_options_prefixed(m):
    raw = {"plot": {"cmap": "viridis", "vmin": -1, "vmax": 1, "annot": False, "fmt": ".1f"}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["corr_cmap"] == "viridis"
    assert flat["corr_vmin"] == -1
    assert flat["corr_vmax"] == 1
    assert flat["corr_annot"] is False
    assert flat["corr_fmt"] == ".1f"


def test_flatten_appearance_section(m):
    raw = {"appearance": {"style": "whitegrid", "context": "paper", "palette": "muted",
                           "color": "blue", "linewidth": 2.0, "marker": "o", "alpha": 0.8}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["style"] == "whitegrid"
    assert flat["context"] == "paper"
    assert flat["palette"] == "muted"
    assert flat["color"] == "blue"
    assert flat["linewidth"] == 2.0


def test_flatten_appearance_rc_merges_title_weight_and_label_size(m):
    raw = {"appearance": {"title_weight": "bold", "label_size": 14,
                           "rcParameters": {"legend.frameon": False}}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["rc"]["axes.titleweight"] == "bold"
    assert flat["rc"]["axes.labelsize"] == 14
    assert flat["rc"]["legend.frameon"] is False


def test_flatten_appearance_grid_implies_grid_flag(m):
    raw = {"appearance": {"grid": {"alpha": 0.3, "linestyle": ":", "color": "gray"}}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["grid"] is True
    assert flat["grid_alpha"] == 0.3
    assert flat["grid_linestyle"] == ":"
    assert flat["grid_color"] == "gray"


def test_flatten_appearance_legend_implies_legend_flag(m):
    raw = {"appearance": {"legend": {"location": "upper right", "ncol": 2,
                                      "title": "Ch", "fontsize": 9, "outside": True}}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["legend"] is True
    assert flat["legend_loc"] == "upper right"
    assert flat["legend_ncol"] == 2
    assert flat["legend_title"] == "Ch"
    assert flat["legend_fontsize"] == 9
    assert flat["legend_outside"] is True


def test_flatten_root_level_title_annotations_regions(m):
    raw = {"title": "Fig 1",
           "annotations": [{"x": 1, "y": 1, "text": "pt"}],
           "regions": [{"xmin": 0, "xmax": 1}]}
    flat = m._flatten_yaml_schema(raw)
    assert flat["title"] == "Fig 1"
    assert flat["annotations"] == [{"x": 1, "y": 1, "text": "pt"}]
    assert flat["fill_between"] == [{"xmin": 0, "xmax": 1}]  # 'regions' renamed internally


def test_flatten_output_section(m):
    raw = {"output": {"path": "fig.png", "dpi": 300, "transparent": True}}
    flat = m._flatten_yaml_schema(raw)
    assert flat["output"] == "fig.png"
    assert flat["dpi"] == 300
    assert flat["transparent"] is True


def test_flatten_unknown_top_level_key_passes_through(m):
    flat = m._flatten_yaml_schema({"max_rows": 500000, "force": True})
    assert flat["max_rows"] == 500000
    assert flat["force"] is True


def test_flatten_non_dict_root_passed_through_unchanged(m):
    # caller (_load_yaml_defaults) is responsible for the type error message;
    # _flatten_yaml_schema itself must not raise on a non-dict root
    assert m._flatten_yaml_schema(["not", "a", "dict"]) == ["not", "a", "dict"]


def test_flatten_wrong_type_section_dies(m):
    with pytest.raises(SystemExit):
        m._flatten_yaml_schema({"x_axis": "not a mapping"})


# ── _load_yaml_defaults: full pipeline via real files ─────────────────────

def _write_yaml(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


def test_load_yaml_defaults_basic(m, tmp_path):
    p = _write_yaml(tmp_path, "data:\n  x: t\n  y: [a, b]\nplot:\n  mode: hist\n  bins: 20\n")
    parser = m.build_parser()
    defaults, extras = m._load_yaml_defaults(str(p), parser)
    assert defaults["x"] == "t"
    assert defaults["ys"] == ["a", "b"]
    assert defaults["hist"] is True
    assert defaults["bins"] == 20


def test_load_yaml_defaults_type_coercion_string_to_int(m, tmp_path):
    p = _write_yaml(tmp_path, "rolling: '5'\n")
    defaults, _ = m._load_yaml_defaults(str(p), m.build_parser())
    assert defaults["rolling"] == 5
    assert isinstance(defaults["rolling"], int)


def test_load_yaml_defaults_type_coercion_failure_dies(m, tmp_path):
    p = _write_yaml(tmp_path, "rolling: 'not_a_number'\n")
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(p), m.build_parser())


def test_load_yaml_defaults_scalar_to_list_for_nargs_plus(m, tmp_path):
    p = _write_yaml(tmp_path, "hline: 5\n")  # scalar, but --hline is nargs='+'
    defaults, _ = m._load_yaml_defaults(str(p), m.build_parser())
    assert defaults["hline"] == [5.0]


def test_load_yaml_defaults_unknown_key_warns_and_dropped(m, tmp_path, capsys):
    p = _write_yaml(tmp_path, "totally_made_up_key: 1\nstyle: whitegrid\n")
    defaults, _ = m._load_yaml_defaults(str(p), m.build_parser())
    assert "totally_made_up_key" not in defaults
    assert defaults["style"] == "whitegrid"
    assert "unknown YAML key" in capsys.readouterr().err


def test_load_yaml_defaults_invalid_choice_dies(m, tmp_path):
    p = _write_yaml(tmp_path, "style: not_a_real_style\n")
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(p), m.build_parser())


def test_load_yaml_defaults_splits_extras_from_defaults(m, tmp_path):
    p = _write_yaml(tmp_path, "style: whitegrid\ny_axis:\n  colors: [red, blue]\n")
    defaults, extras = m._load_yaml_defaults(str(p), m.build_parser())
    assert defaults["style"] == "whitegrid"
    assert "y_colors" not in defaults
    assert extras["y_colors"] == ["red", "blue"]


def test_load_yaml_defaults_nested_yaml_self_reference_warns(m, tmp_path, capsys):
    p = _write_yaml(tmp_path, "yaml: other.yaml\nstyle: whitegrid\n")
    defaults, _ = m._load_yaml_defaults(str(p), m.build_parser())
    assert "yaml" not in defaults
    assert defaults["style"] == "whitegrid"
    assert "nested YAML loading is not supported" in capsys.readouterr().err


def test_load_yaml_defaults_file_not_found_dies(m, tmp_path):
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(tmp_path / "nope.yaml"), m.build_parser())


def test_load_yaml_defaults_empty_file_returns_empty(m, tmp_path):
    p = _write_yaml(tmp_path, "")
    defaults, extras = m._load_yaml_defaults(str(p), m.build_parser())
    assert defaults == {}
    assert extras == {}


def test_load_yaml_defaults_root_not_a_mapping_dies(m, tmp_path):
    p = _write_yaml(tmp_path, "- item1\n- item2\n")
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(p), m.build_parser())


def test_load_yaml_defaults_hyphenated_keys_normalized(m, tmp_path):
    p = _write_yaml(tmp_path, "log-x: true\n")
    defaults, _ = m._load_yaml_defaults(str(p), m.build_parser())
    assert defaults["log_x"] is True


def test_load_yaml_defaults_oversized_file_dies(m, tmp_path, monkeypatch):
    p = _write_yaml(tmp_path, "style: whitegrid\n")
    monkeypatch.setattr(m, "_MAX_YAML_SIZE_MB", -1)  # force the size guard to trip
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(p), m.build_parser())


def test_yaml_depth_within_limit(m):
    nested = {}
    cur = nested
    for _ in range(5):
        cur["k"] = {}
        cur = cur["k"]
    assert m._yaml_depth(nested) <= m._MAX_YAML_DEPTH


def test_yaml_depth_exceeds_limit_dies(m, tmp_path):
    # build a YAML file nested deeper than _MAX_YAML_DEPTH
    depth = m._MAX_YAML_DEPTH + 5
    text = "a:\n" + "".join(f"{'  ' * (i+1)}b:\n" for i in range(depth))
    text += "  " * (depth + 1) + "val: 1\n"
    p = _write_yaml(tmp_path, text)
    with pytest.raises(SystemExit):
        m._load_yaml_defaults(str(p), m.build_parser())


# ── end-to-end: YAML through the CLI, precedence rules ─────────────────────

def test_yaml_cli_flag_overrides_yaml_value(m, tmp_path, run_cli, csv_numeric):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("appearance:\n  style: whitegrid\n")
    r = run_cli(str(csv_numeric), "--y", "b", "--yaml", str(yaml_path),
                "--style", "dark", "--output", "o.png", "--force")
    assert r.returncode == 0  # CLI --style should win without erroring


def test_yaml_full_example_from_docstring_loads_cleanly(m, tmp_path, run_cli, csv_numeric):
    """The example config embedded in _load_yaml_defaults' own docstring
    must actually work end-to-end — otherwise the documentation is lying."""
    yaml_text = """
data:
  x: a
  y: [b, c]
y_axis:
  label: "Amplitude"
  colors: [steelblue, darkorange]
  linestyles: ["-", "--"]
x_axis:
  label: "Time (s)"
plot:
  mode: line
  figsize: [10, 6]
appearance:
  style: whitegrid
  context: paper
title: "Test Figure"
output:
  dpi: 100
"""
    yaml_path = tmp_path / "full.yaml"
    yaml_path.write_text(yaml_text)
    r = run_cli(str(csv_numeric), "--yaml", str(yaml_path), "--output", "o.png", "--force")
    assert r.returncode == 0, r.stderr
