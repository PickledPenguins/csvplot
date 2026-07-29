#!/usr/bin/env python3
"""
csvplot — Research-grade CSV plotting tool powered by seaborn.

All plots share a seaborn theme (--style, --context, --palette) giving a
consistent, publication-ready look without per-figure boilerplate.

Common patterns
────────────────────────────────────────────────────────────────────────────
  Time series / transforms:
    csvplot data.csv --x time --y voltage --rolling 10
    csvplot data.csv --x time --ys ch1 ch2 ch3 --subplots --smooth 2.0
    csvplot data.csv --x t --y v --baseline --normalize

  Distributions:
    csvplot data.csv --y score --hist --kde --bins 30
    csvplot data.csv --y score --box --x condition --hue group
    csvplot data.csv --ys a b c --violin               # compare columns

  Scatter / regression:
    csvplot data.csv --x weight --y height --scatter --fit linear
    csvplot data.csv --x dose --y response --regplot --ci 95

  Group summary:
    csvplot data.csv --y value --group-by treatment --agg mean --bar
    csvplot data.csv --y val --group-by category --agg median --top 5 --bar

  Correlation / overview:
    csvplot data.csv --ys c1 c2 c3 c4 --corr --corr-method spearman
    csvplot data.csv --ys c1 c2 c3 --pairplot --hue group

  Multi-file overlay:
    csvplot --files ctrl.csv trt.csv --x t --y signal --style whitegrid

  Publication export:
    csvplot data.csv --x time --y amp --style whitegrid --context paper \\
      --palette colorblind --dpi 300 --output figure.png
────────────────────────────────────────────────────────────────────────────
Requires: seaborn ≥ 0.13, matplotlib ≥ 3.7.3, pandas ≥ 2.1.1, numpy
"""
__version__ = "0.19"

import argparse
import ast
import math
import sys
import json
import os
import re
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Module-level constants ────────────────────────────────────────────────────

# Internal sentinel column name used when no --x is given; defined as a
# constant so the assignment site and the label-substitution site stay in sync.
_IDX_COL = "_idx"

# np.RankWarning moved to np.exceptions.RankWarning in NumPy 2.0.
# Build a compat alias that works on both 1.x and 2.x at import time.
_RankWarning = getattr(
    getattr(np, "exceptions", None), "RankWarning",
    getattr(np, "RankWarning", UserWarning),
)

UNIT_MULTIPLIERS = {
    "": 1, "b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
    "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1,
}

# These modes treat x as a categorical grouping axis, not a numeric axis.
_CAT_MODES = frozenset({"box", "violin", "bar", "swarm"})

# seaborn ≥ 0.12 replaced the deprecated 'ci' kwarg in barplot with 'errorbar'.
# packaging is a transitive dependency of matplotlib, so this is always available.
try:
    from packaging.version import Version
    _SNS_GE_012 = Version(sns.__version__) >= Version("0.12")
except Exception:
    _SNS_GE_012 = True  # assume modern seaborn if the version string is unparseable

# Keys accepted in YAML only — deliberately absent from the CLI to keep it concise.
# They provide per-series styling, fine-grained axes/legend control, plot-mode
# parameters, and post-plot overlays that would be unwieldy as CLI flags.
# Per-series style attributes shared by the left (y_*) and dual-y (dual_y_*) axes;
# single source of truth for YAML_EXTRA_KEYS, schema flattening, and style lookup.
_STYLE_ATTRS = (("color", "colors"), ("linestyle", "linestyles"),
                ("linewidth", "linewidths"), ("alpha", "alphas"), ("marker", "markers"))

YAML_EXTRA_KEYS = frozenset({
    # per-series styling — primary (left) y-axis, indexed to match --ys order
    *(f"y_{suffix}" for _, suffix in _STYLE_ATTRS),
    # per-series styling — dual-y (right) axis, indexed to match --dual-y order
    *(f"dual_y_{suffix}" for _, suffix in _STYLE_ATTRS),
    # seaborn theme extensions
    "font_scale", "rc",
    # axes limits and tick formatting
    "xlim", "ylim", "dual_y_lim", "xtick_rotation", "ytick_rotation",
    # legend refinement
    "legend_loc", "legend_ncol", "legend_title", "legend_fontsize", "legend_outside",
    # grid style (requires --grid; overrides seaborn defaults)
    "grid_alpha", "grid_linestyle", "grid_color",
    # figure output
    "transparent",
    # histogram (sns.histplot) extras
    "hist_stat", "hist_multiple",
    # KDE (sns.kdeplot) extras
    "kde_bw_adjust", "kde_cut",
    # scatter extras
    "scatter_s",
    # violin extras
    "violin_inner",
    # regression extras
    "regplot_order", "regplot_robust",
    # post-plot text annotations
    "annotations",
    # post-plot shaded vertical regions
    "fill_between",
    # reference-line (--hline/--vline) styling overrides
    "hline_style", "vline_style",
    # --corr heatmap styling overrides
    "corr_cmap", "corr_vmin", "corr_vmax", "corr_annot", "corr_fmt",
})


# ── Unit coercion ─────────────────────────────────────────────────────────────

# shared by parse_units's two passes: value-conversion and suffix-detection
_UNIT_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-z]+)?$")


def parse_units(series, threshold=1.0, require_suffix=True):
    """
    Convert a string Series with SI/data-rate unit suffixes to float.
    e.g. "10ms" → 0.01,  "-3us" → -3e-6,  "5kb" → 5000,  "1.5e3ms" → 1.5.

    Supports scientific notation (1e6, 1.5e-3, 2E9), leading sign (+10ms),
    and trailing decimal (10.).  Unknown unit suffixes (e.g. "km", "foobar")
    return NaN rather than silently stripping the unit and returning the bare
    number, which would produce silently wrong magnitudes.

    Safety guards (both configurable via --unit-parse-* / YAML data.*):
      - require_suffix: if no value in the column has a recognized unit
        suffix, return the Series unchanged. Without this, bare numeric
        strings (versions, IDs, codes — "1.0", "2.1") would be coerced to
        floats even though no unit is involved.
      - threshold: fraction of non-null values that must convert successfully
        for the result to be used; otherwise the original Series is returned.
        Default 1.0 requires every value to convert.

    ±inf (e.g. from 1e309) are replaced with NaN after conversion.
    """
    def _convert(v):
        if pd.isna(v):
            return np.nan
        raw = str(v).strip().lower()
        # full pattern covering: integers, decimals, leading-dot, scientific
        # notation, leading sign, optional unit suffix; anchored to block
        # partial matches like "1.2.3kb"
        m = _UNIT_RE.match(raw)
        if not m:
            return pd.to_numeric(v, errors="coerce")
        try:
            val = float(m.group(1))
        except ValueError:
            return np.nan
        unit = m.group(2) or ""
        if unit and unit not in UNIT_MULTIPLIERS:
            # unknown suffix — return NaN rather than silently dropping the unit
            return np.nan
        return val * UNIT_MULTIPLIERS.get(unit, 1)

    if require_suffix:
        # vectorized: extract the unit-suffix group across the whole column at
        # once and check membership, instead of a per-cell apply(). Columns
        # with no unit suffixes (the common case — versions/IDs/codes) exit
        # here without ever running the per-cell _convert pass below.
        units = series.astype(str).str.strip().str.lower().str.extract(_UNIT_RE)[1]
        if not units.isin(UNIT_MULTIPLIERS).any():
            return series

    converted  = series.apply(_convert)
    # ±inf can arise from extreme exponents (e.g. "1e309"); normalise to NaN
    converted  = converted.replace([np.inf, -np.inf], np.nan)
    orig_valid = series.notna().sum()
    if orig_valid > 0 and converted.notna().sum() / orig_valid < threshold:
        return series
    return converted


# ── Gaussian smoothing ────────────────────────────────────────────────────────

def gaussian_smooth(values, sigma):
    """
    Gaussian kernel smoothing using pure-NumPy convolution (no scipy needed).

    NaN gaps are linearly interpolated before convolution to eliminate
    boundary artifacts, then their positions are restored in the output.
    'sigma' is the kernel std-dev measured in rows (samples).
    Kernel radius = 3σ, covering 99.7 % of the Gaussian distribution.

    Note: reflect-padding reduces but does not eliminate edge artifacts for
    arrays shorter than ~6σ samples; samples near the boundaries of short
    series should be interpreted with caution.
    """
    if sigma <= 0:
        # CLI validation already blocks this, but guard at the function
        # boundary as well for direct programmatic use
        raise ValueError(f"gaussian_smooth: sigma must be > 0, got {sigma!r}")
    arr      = np.asarray(values, dtype=float)
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr
    # numpy reflect mode requires ≥ 2 elements; single-element input cannot
    # be reflected and needs no smoothing anyway
    if len(arr) < 2:
        return arr.copy()
    if nan_mask.any():
        idx = np.arange(len(arr))
        arr = np.interp(idx, idx[~nan_mask], arr[~nan_mask])
    radius = max(1, round(3 * sigma))
    x      = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # reflect-padding reduces boundary distortion; short arrays may still show artifacts
    smoothed = np.convolve(np.pad(arr, radius, mode="reflect"), kernel, mode="valid")
    smoothed[nan_mask] = np.nan
    return smoothed


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser():
    """Return a fully documented, grouped ArgumentParser."""
    p = argparse.ArgumentParser(
        prog="csvplot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # ── Input ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Input")
    g.add_argument("file", nargs="?", default="-",
                   help="CSV file to plot; '-' reads from stdin (default: stdin)")
    g.add_argument("--files", nargs="+", metavar="FILE",
                   help="Load multiple files and overlay them on the same plot")
    g.add_argument("--sep", default=",", metavar="SEP",
                   help="Column delimiter (default: ','; use '\\t' for TSV)")
    g.add_argument("--no-parse-units", action="store_true",
                   help="Disable automatic unit-suffix coercion (e.g. '10ms' → 0.01)")
    g.add_argument("--unit-parse-threshold", type=float, default=1.0, metavar="F",
                   help="Fraction (0.0-1.0) of values in a column that must parse "
                        "as numbers/units for auto unit-parsing to apply (default: 1.0)")
    g.add_argument("--unit-parse-allow-bare-numbers", action="store_true",
                   help="Allow auto unit-parsing to convert columns with no recognized "
                        "unit suffix at all, e.g. version/ID strings (off by default)")
    g.add_argument("--headers", action="store_true",
                   help="Print each file's column names, one per line, and exit — "
                        "no --y/--ys required")
    g.add_argument("--info", action="store_true",
                   help="Print each file's shape, column names, dtypes, and non-null "
                        "counts, and exit — no --y/--ys required")
    g.add_argument("--max-rows", type=int, default=2_000_000, metavar="N",
                   help="Abort if a loaded file exceeds N rows (default: 2,000,000). "
                        "For stdin input (-), this is the only memory guard (no file-size "
                        "pre-check is possible). Use --head/--sample to subset large inputs.")
    g.add_argument("--yaml", metavar="PATH",
                   help="Load all options from a YAML config file. CLI flags take precedence "
                        "over YAML values; YAML values override built-in defaults. "
                        "Keys match CLI flag names with hyphens or underscores. "
                        "Example: 'style: whitegrid' equals --style whitegrid")

    # ── Column selection ───────────────────────────────────────────────────
    g = p.add_argument_group("Column selection")
    g.add_argument("--x",   metavar="COL",
                   help="X-axis column; integer row index is used when omitted")
    g.add_argument("--y",   metavar="COL", help="Single Y-axis column")
    g.add_argument("--ys",  nargs="+", metavar="COL",
                   help="One or more Y-axis columns; overrides --y")
    g.add_argument("--hue", metavar="COL",
                   help="Column for seaborn colour/hue grouping")
    g.add_argument("--dual-y", nargs="+", metavar="COL",
                   help="Column(s) plotted on a second RIGHT y-axis for a different scale. "
                        "Series are drawn dashed to distinguish them from left-axis series. "
                        "Not supported with --hist, --corr, or --pairplot")
    g.add_argument("--dual-y-label", metavar="TEXT",
                   help="Label for the right y-axis (default: comma-joined column names)")

    # ── Output ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("Output")
    g.add_argument("--output", metavar="PATH",
                   help="Save figure to PATH; format inferred from extension (png/pdf/svg …)")
    g.add_argument("--dpi", type=int, default=100, metavar="N",
                   help="Figure resolution in DPI for saved files (default: 100; use 300 for print)")
    g.add_argument("--export-json", action="store_true",
                   help="Print post-transform data as JSON then exit without plotting")
    g.add_argument("--force",  action="store_true",
                   help="Overwrite --output file if it already exists")

    # ── Row operations ─────────────────────────────────────────────────────
    g = p.add_argument_group(
        "Row operations  [stable execution order: head → tail → filter → sort → sample → every]")
    g.add_argument("--head",   type=int, metavar="N", help="Keep the first N rows")
    g.add_argument("--tail",   type=int, metavar="N", help="Keep the last N rows")
    g.add_argument("--filter", metavar="EXPR",
                   help="pandas query expression, e.g. --filter 'voltage > 0 and t < 10'. "
                        "Evaluated via pandas eval — avoid untrusted input in shared environments")
    g.add_argument("--sort",   metavar="COL",
                   help="Sort rows ascending by COL before plotting")
    g.add_argument("--sample", type=int, metavar="N",
                   help="Randomly subsample N rows (pair with --seed for reproducibility)")
    g.add_argument("--seed",   type=int, metavar="N",
                   help="Random seed for --sample; makes subsampling deterministic")
    g.add_argument("--every",  type=int, metavar="N",
                   help="Keep every Nth row — deterministic thinning for dense series")
    g.add_argument("--stats",  action="store_true",
                   help="Print describe() statistics for y columns then exit (no plot). "
                        "NOTE: row-ops (--head/--tail/--filter/etc.) and transforms are "
                        "applied before statistics are computed.")

    # ── Group aggregation ──────────────────────────────────────────────────
    g = p.add_argument_group("Group aggregation")
    g.add_argument("--group-by", metavar="COL",
                   help="Group rows by COL; typically paired with --agg and a categorical plot")
    g.add_argument("--agg", choices=["mean", "min", "max", "sum", "median"],
                   help="Aggregation function per group; requires --group-by")
    g.add_argument("--top", type=int, metavar="N",
                   help="After aggregation keep the N groups with the highest y value")
    g.add_argument("--top-by", metavar="COL",
                   help="Rank --top groups by COL instead of the first --y/--ys column")

    # ── Column transforms ──────────────────────────────────────────────────
    g = p.add_argument_group("Column transforms  [applied in listed order; each chains onto the previous]")
    g.add_argument("--cumsum",    action="store_true", help="Cumulative sum")
    g.add_argument("--cummax",    action="store_true", help="Running maximum")
    g.add_argument("--cummin",    action="store_true", help="Running minimum")
    g.add_argument("--diff",      action="store_true",
                   help="Absolute row-to-row difference; first element becomes NaN")
    g.add_argument("--rate",      action="store_true",
                   help="Relative percent change between rows (use --diff for absolute change)")
    g.add_argument("--ewm",       type=int, metavar="SPAN",
                   help="Exponential weighted moving average with the given span")
    g.add_argument("--rolling",   type=int, metavar="N",
                   help="Rolling mean over N rows (min_periods=1 — no leading NaN)")
    g.add_argument("--smooth",    type=float, metavar="SIGMA",
                   help="Gaussian kernel smoothing; SIGMA is the std-dev in row units")
    g.add_argument("--clip",      nargs=2, type=float, metavar=("LO", "HI"),
                   help="Clip y values to [LO, HI] before plotting")
    g.add_argument("--baseline",  action="store_true",
                   help="Subtract the first non-NaN value so each series starts at zero")
    g.add_argument("--normalize", action="store_true",
                   help="Min-max scale each y column to [0, 1]; constant columns → 0")
    g.add_argument("--zscore",    action="store_true",
                   help="Z-score each y column: (x − μ) / σ  (zero mean, unit variance)")

    # ── Plot mode ──────────────────────────────────────────────────────────
    g = p.add_argument_group("Plot mode  [default: line]")
    g.add_argument("--scatter", action="store_true",
                   help="Scatter plot (sns.scatterplot) — good for two continuous variables")
    g.add_argument("--hist",    action="store_true",
                   help="Histogram (sns.histplot); combine with --kde to overlay a density curve")
    g.add_argument("--kde",     action="store_true",
                   help="Kernel density estimate (sns.kdeplot); usable alone or combined with --hist")
    g.add_argument("--box",     action="store_true",
                   help="Box plot (sns.boxplot); use --x for grouping column")
    g.add_argument("--violin",  action="store_true",
                   help="Violin plot (sns.violinplot); shows full distribution shape")
    g.add_argument("--bar",     action="store_true",
                   help="Bar chart with optional error bands (sns.barplot); see --ci")
    g.add_argument("--swarm",   action="store_true",
                   help="Swarm plot — shows every individual point; best for small N")
    g.add_argument("--regplot", action="store_true",
                   help="Scatter + OLS regression line + CI band (sns.regplot / per-hue loop)")
    g.add_argument("--pairplot", action="store_true",
                   help="Pair-wise scatter matrix for all y columns (sns.pairplot); ignores --x")
    g.add_argument("--corr",    action="store_true",
                   help="Correlation heatmap (sns.heatmap); see --corr-method")
    g.add_argument("--corr-method", default="pearson",
                   choices=["pearson", "spearman", "kendall"],
                   help="Correlation coefficient (default: pearson)")

    # ── Regression / fit ───────────────────────────────────────────────────
    g = p.add_argument_group("Regression / fit")
    g.add_argument("--fit", choices=["linear"],
                   help="Overlay a dashed OLS fit line on line or scatter plots")
    g.add_argument("--ci", type=int, default=95, metavar="PCT",
                   help="Confidence interval %% for --regplot and --bar bands (0 = off, default: 95)")

    # ── Computed statistics ────────────────────────────────────────────────
    g = p.add_argument_group("Computed statistics  [text output, no plot]")
    g.add_argument("--percentile", nargs="+", type=float, metavar="P",
                   help="Compute percentiles in 0–100 range, e.g. --percentile 25 50 75")

    # ── Live mode (not yet implemented) ────────────────────────────────────
    g = p.add_argument_group("Live mode  [not yet implemented]")
    g.add_argument("--watch",    action="store_true",
                   help="[NOT YET IMPLEMENTED] Re-read and redraw the file every --interval seconds")
    g.add_argument("--interval", type=float, default=1.0, metavar="SEC",
                   help="Polling interval for --watch in seconds (default: 1.0)")
    g.add_argument("--alert",    nargs="+", metavar="EXPR",
                   help="[NOT YET IMPLEMENTED] Print a warning when a column threshold expression fires")

    # ── Axes / layout ──────────────────────────────────────────────────────
    g = p.add_argument_group("Axes / layout")
    g.add_argument("--title",    metavar="TEXT", help="Figure suptitle")
    g.add_argument("--xlabel",   metavar="TEXT",
                   help="X-axis label override (defaults to column name)")
    g.add_argument("--ylabel",   metavar="TEXT",
                   help="Y-axis label override (applied to all panels when using --subplots)")
    g.add_argument("--hline",    nargs="+", type=float, metavar="Y",
                   help="Dashed horizontal reference line(s) at the given y value(s)")
    g.add_argument("--vline",    nargs="+", type=float, metavar="X",
                   help="Dashed vertical reference line(s) at the given x value(s)")
    g.add_argument("--log-x",    action="store_true", help="Log scale on x-axis")
    g.add_argument("--log-y",    action="store_true", help="Log scale on y-axis")
    g.add_argument("--grid",     action="store_true",
                   help="Force gridlines; useful with --style white or --style ticks")
    g.add_argument("--subplots", action="store_true",
                   help="One subplot panel per y column sharing the same x-axis")
    g.add_argument("--figsize",  default="10,6", metavar="W,H",
                   help="Figure size in inches as 'width,height' (default: 10,6)")
    g.add_argument("--legend",   action="store_true",
                   help="Force legend display even for single-series plots")
    g.add_argument("--bins",     type=int, metavar="N",
                   help="Histogram bin count; seaborn auto-selects when omitted")

    # ── Seaborn theme ──────────────────────────────────────────────────────
    g = p.add_argument_group("Seaborn theme  [applied globally to all plots]")
    g.add_argument("--style",   default="darkgrid",
                   choices=["darkgrid", "whitegrid", "dark", "white", "ticks"],
                   help="Axes background style (default: darkgrid; 'whitegrid'/'ticks' for papers)")
    g.add_argument("--context", default="notebook",
                   choices=["paper", "notebook", "talk", "poster"],
                   help="Font/line scaling context (default: notebook; 'paper' for publications)")
    g.add_argument("--palette",  metavar="NAME",
                   help="Seaborn colour palette — e.g. muted, deep, pastel, colorblind, tab10, husl")

    # ── Line / marker style ────────────────────────────────────────────────
    g = p.add_argument_group("Line / marker style")
    g.add_argument("--color",     metavar="C",
                   help="Single colour for all series (hex or name); ignored when --hue is active")
    g.add_argument("--linewidth", type=float, default=1.5, metavar="W",
                   help="Line width in points (default: 1.5)")
    g.add_argument("--marker",    metavar="M",
                   help="Matplotlib marker: 'o' circle, 's' square, '^' triangle, etc.")
    g.add_argument("--alpha",     type=float, default=1.0, metavar="A",
                   help="Transparency 0.0 (invisible) – 1.0 (opaque) for all elements (default: 1.0)")

    return p


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _json_safe(v):
    """
    Normalise a single value for JSON serialisation.
    Converts non-finite floats, pd.NA, numpy scalars, and Decimal to
    JSON-compatible Python types. Returns None for any non-finite/missing value.
    json.dumps(default=str) would silently emit '<NA>' for pd.NA and
    'NaN'/'Infinity' (non-standard) for non-finite floats — this prevents that.
    """
    if v is pd.NA:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, np.generic):
        pv = v.item()
        if isinstance(pv, float) and not math.isfinite(pv):
            return None
        return pv
    # decimal.Decimal is not JSON-serialisable; convert to float
    try:
        from decimal import Decimal
        if isinstance(v, Decimal):
            fv = float(v)
            return None if not math.isfinite(fv) else fv
    except ImportError:
        pass
    return v


def die(msg):
    """Print a formatted error to stderr and exit with code 1."""
    print(f"csvplot error: {msg}", file=sys.stderr)
    sys.exit(1)


def _warn(msg):
    """Print a formatted warning to stderr; execution continues."""
    print(f"csvplot warning: {msg}", file=sys.stderr)


# Hard caps on input size to avoid OOM on malformed or oversized files.
# _MAX_FILE_SIZE_MB is a fast pre-check on on-disk size; for compressed inputs
# (.gz/.zip/etc.) the post-load row/column caps below are the real backstop,
# since decompressed size can far exceed the file's on-disk size.
# Row count is also operator-overridable via --max-rows.
_MAX_FILE_SIZE_MB = 500
_MAX_COLS = 1000

# pairplot renders every row in every panel; beyond a few thousand points,
# individual markers become indistinguishable from overplotting anyway, so
# sample down rather than rendering the full dataset.
_PAIRPLOT_MAX_ROWS = 5000

# Hard caps on YAML config size/shape; configs are small hand-written files,
# so generous limits here mainly catch accidental or malformed input.
_MAX_YAML_SIZE_MB = 10
_MAX_YAML_DEPTH = 20
_MAX_YAML_LIST_LEN = 1000

# YAML anchors/aliases share one parsed object across every reference to it
# rather than copying it, so a small, legitimately-shaped file (a handful of
# anchors, each referenced a dozen times, chained a dozen levels deep) can
# represent an astronomically large *logical* structure — the classic
# "billion laughs" pattern. A naive recursive walk re-visits the same shared
# object once per reference, so cost grows with the fully-expanded size, not
# the file's actual byte size or object-graph depth (both of which stay
# small and pass their own caps above). This budget bounds total node visits
# directly, so _yaml_depth() fails fast regardless of how the blowup is
# shaped, rather than trying to specifically detect reference sharing.
_MAX_YAML_NODES = 100_000


def load_dataframe(file, sep, max_rows, head_hint=None):
    """
    Load a delimited file; provide actionable error messages on all failure paths.
    head_hint, when given, reads only its first N rows directly (bypassing the
    --max-rows size guard entirely) — used only when --head is the sole row-op,
    so a small --head preview of a file bigger than --max-rows still works
    instead of failing the row-count check before --head gets a chance to apply.
    """
    if file != "-" and head_hint is None:
        try:
            size_mb = Path(file).stat().st_size / 1e6
        except OSError:
            size_mb = 0  # let pd.read_csv raise the real error below
        if size_mb > _MAX_FILE_SIZE_MB:
            die(f"{file!r} is {size_mb:.0f} MB, exceeding the {_MAX_FILE_SIZE_MB} MB limit")
    try:
        df = pd.read_csv(sys.stdin if file == "-" else file, sep=sep,
                          nrows=head_hint if head_hint is not None else max_rows + 1)
    except FileNotFoundError:     die(f"file not found: {file}")
    except PermissionError:       die(f"permission denied: {file}")
    except pd.errors.ParserError as e: die(f"CSV parse error in {file!r}: {e}")
    except Exception as e:        die(f"could not load {file!r}: {e}")
    if len(df.columns) > _MAX_COLS:
        die(f"{file!r} has {len(df.columns)} columns, exceeding the {_MAX_COLS} limit")
    if head_hint is None and len(df) > max_rows:
        die(f"{file!r} has more than {max_rows} rows (--max-rows limit); "
            f"use --head/--sample to subset, or raise --max-rows")
    return df


# ── YAML config loader ────────────────────────────────────────────────────────

# ── YAML schema flattening ─────────────────────────────────────────────────

# Canonical, ordered list of the 10 plot-mode CLI flags (matches the
# argparse "Plot mode" group 1:1) — single source of truth for the YAML
# plot.mode validation set below, the CLI mutual-exclusivity check in
# _validate_args, and the reachable subset in _active_mode().
_PLOT_MODE_NAMES = ("scatter", "hist", "kde", "box", "violin", "bar",
                     "swarm", "regplot", "corr", "pairplot")

# pairplot never reaches _active_mode() — main() dispatches it to
# _render_special() before _build_figure() is ever called — so it's
# excluded here (kde is handled separately, after the loop below). corr has
# the same dispatch in practice but stays in this list: the test suite
# exercises _active_mode() directly with corr=True and expects it back.
_FIGURE_MODES = tuple(m for m in _PLOT_MODE_NAMES if m not in ("kde", "pairplot"))

# valid values for YAML's plot.mode key; "line" is the implicit CLI default
# but must be spelled out explicitly to select it via YAML.
_MODE_FLAGS = frozenset(_PLOT_MODE_NAMES) | {"line"}

# plot.<key> → internal flat key, for keys that map 1:1 by name
_PLOT_PASSTHROUGH = frozenset({
    "bins", "stat", "multiple", "bw_adjust", "cut", "s", "order", "robust",
    "inner", "method",
})


def _get_section(raw, key):
    """
    Return raw[key] as a dict, or {} if absent.
    Die with a clear message if the key exists but is not a mapping.
    Eliminates the repeated isinstance(x, dict) guard in _flatten_yaml_schema.
    """
    v = raw.get(key, {})
    if not isinstance(v, dict):
        die(f"YAML: {key!r} must be a mapping, got {type(v).__name__!r}")
    return v


def _flatten_yaml_schema(raw):
    """
    Convert the nested user-facing YAML schema into the flat internal key
    structure consumed by _load_yaml_defaults().  This is the ONLY place
    that needs to change if the user-facing schema changes again.

    Nested schema → flat internal keys (examples):
        data.file/x/y/hue/...     → file/x/ys/hue/...        (CLI dests)
        data.percentile/top_by    → percentile/top_by
        x_axis.label/lim/log/...  → xlabel/xlim/log_x/...
        x_axis.vline/vline_style  → vline/vline_style
        y_axis.label/lim/...      → ylabel/ylim/log_y/...
        y_axis.colors/...         → y_colors/...             (per-series lists)
        y_axis.hline/hline_style  → hline/hline_style
        right_axis.columns/...    → dual_y/dual_y_label/...
        right_axis.colors/...     → dual_y_colors/...
        plot.mode: hist           → hist: true               (CLI boolean flag)
        plot.figsize: [10,6]      → figsize: "10,6"           (CLI string format)
        plot.bins/stat/...        → bins/hist_stat/...        (mode-prefixed)
        plot.cmap/vmin/vmax/...   → corr_cmap/corr_vmin/...   (--corr heatmap)
        appearance.style/...      → style/context/palette/font_scale
        appearance.color/...      → color/linewidth/marker/alpha (single-series defaults)
        appearance.grid.*         → grid_alpha/grid_linestyle/grid_color
        appearance.legend.*       → legend_loc/legend_ncol/...
        appearance.rcParameters   → rc
        title                     → title
        annotations/regions       → annotations/fill_between
        output.path/dpi/...       → output/dpi/transparent

    Any other top-level key matching a CLI argument's dest name (e.g. a flat
    `max_rows: 500000` or `force: true`) passes through unchanged — every CLI
    flag is configurable via YAML even if not listed above.
    """
    if not isinstance(raw, dict):
        return raw  # let the caller's type check handle the error

    flat = {}

    def _take(src, key, dst=None):
        """Copy src[key] to flat[dst or key] if present."""
        if key in src:
            flat[dst or key] = src[key]

    # ── data: file, columns, hue, transforms, filters, aggregation ────────
    data = _get_section(raw, "data")
    for key in ("file", "files", "x", "hue", "sep", "filter", "sort", "head",
                "tail", "sample", "seed", "every", "rolling", "ewm", "smooth",
                "clip", "cumsum", "cummax", "cummin", "diff", "rate",
                "baseline", "normalize", "zscore", "group_by", "agg", "top", "top_by",
                "percentile", "no_parse_units", "unit_parse_threshold",
                "unit_parse_allow_bare_numbers"):
        _take(data, key)
    if "y" in data:
        flat["ys"] = data["y"]  # 'y' (singular, user-facing) → 'ys' (CLI dest)

    # legacy flat-style YAML (no 'data:' wrapper) may have 'y'/'ys' at root —
    # apply the same singular→plural rename so a bare list under 'y' doesn't
    # get passed straight through to the --y (singular, non-list) CLI dest
    if "data" not in raw:
        if "y" in raw and "ys" not in raw:
            flat["ys"] = raw["y"] if isinstance(raw["y"], list) else [raw["y"]]
        elif "ys" in raw:
            flat["ys"] = raw["ys"]

    # ── x_axis: label, range, scale, ticks ─────────────────────────────────
    x_axis = _get_section(raw, "x_axis")
    for key, dst in [("label", "xlabel"), ("lim", "xlim"), ("rotation", "xtick_rotation"),
                     ("vline", "vline"), ("vline_style", "vline_style")]:
        _take(x_axis, key, dst)
    if x_axis.get("log"):  flat["log_x"] = True

    # ── y_axis: label, range, scale, ticks, per-series styling ────────────
    y_axis = _get_section(raw, "y_axis")
    _take(y_axis, "label", "ylabel")
    _take(y_axis, "lim",   "ylim")
    if y_axis.get("log"):  flat["log_y"] = True
    _take(y_axis, "rotation", "ytick_rotation")
    _take(y_axis, "hline")
    _take(y_axis, "hline_style")
    for _, suffix in _STYLE_ATTRS:
        _take(y_axis, suffix, f"y_{suffix}")

    # ── right_axis: same shape as y_axis, plus 'columns' and 'label' ──────
    right_axis = _get_section(raw, "right_axis")
    _take(right_axis, "columns", "dual_y")
    _take(right_axis, "label",   "dual_y_label")
    for _, suffix in _STYLE_ATTRS:
        _take(right_axis, suffix, f"dual_y_{suffix}")
    _take(right_axis, "lim",   "dual_y_lim")

    # ── plot: mode (→ boolean flag), figsize, mode-specific options ───────
    plot = _get_section(raw, "plot")
    mode = plot.get("mode")
    if mode is not None:
        if mode not in _MODE_FLAGS:
            die(f"YAML: plot.mode must be one of {sorted(_MODE_FLAGS)}, got {mode!r}")
        if mode != "line":  # 'line' is the implicit default; no flag needed
            flat[mode] = True
    if "figsize" in plot:
        fs = plot["figsize"]
        if not (isinstance(fs, list) and len(fs) == 2):
            die(f"YAML: plot.figsize must be a two-element list, e.g. [10, 6], got {fs!r}")
        flat["figsize"] = f"{fs[0]},{fs[1]}"  # CLI internal format is 'W,H'
    _take(plot, "fit")
    _take(plot, "ci")
    _take(plot, "corr_method", "corr_method")
    # mode-specific options are prefixed by mode name in the internal schema
    # (hist_stat, kde_bw_adjust, violin_inner, etc.) except a few that are
    # mode-agnostic (bins, scatter_s, regplot_order/robust)
    _PREFIX_MAP = {
        "stat": "hist_stat", "multiple": "hist_multiple",
        "bw_adjust": "kde_bw_adjust", "cut": "kde_cut",
        "inner": "violin_inner",
        "order": "regplot_order", "robust": "regplot_robust",
        "s": "scatter_s",
        # corr-specific: only meaningful under plot.mode: corr
        "cmap": "corr_cmap", "vmin": "corr_vmin", "vmax": "corr_vmax",
        "annot": "corr_annot", "fmt": "corr_fmt",
    }
    for key, val in plot.items():
        if key in _PREFIX_MAP:
            flat[_PREFIX_MAP[key]] = val
        elif key == "bins":
            flat["bins"] = val

    # ── appearance: theme, grid, legend, rcParameters ─────────────────────
    appearance = _get_section(raw, "appearance")
    for key in ("style", "context", "palette", "font_scale",
                "color", "linewidth", "marker", "alpha"):
        _take(appearance, key)
    # title_weight / label_size are user-facing names for common rcParams;
    # merge into the rc dict alongside any explicit rcParameters
    rc = _get_section(appearance, "rcParameters")
    if "title_weight" in appearance:
        rc["axes.titleweight"] = appearance["title_weight"]
    if "label_size" in appearance:
        rc["axes.labelsize"] = appearance["label_size"]
    if rc:
        flat["rc"] = rc

    grid = _get_section(appearance, "grid")
    for key, dst in [("alpha", "grid_alpha"), ("linestyle", "grid_linestyle"),
                     ("color", "grid_color")]:
        _take(grid, key, dst)
    if grid:
        flat["grid"] = True  # presence of grid settings implies --grid

    legend = _get_section(appearance, "legend")
    for key, dst in [("location", "legend_loc"), ("ncol", "legend_ncol"),
                     ("title", "legend_title"), ("fontsize", "legend_fontsize"),
                     ("outside", "legend_outside")]:
        _take(legend, key, dst)
    if legend:
        flat["legend"] = True  # presence of legend settings implies --legend

    # ── root-level: title, annotations, regions ────────────────────────────
    _take(raw, "title")
    _take(raw, "annotations")
    if "regions" in raw:
        flat["fill_between"] = raw["regions"]  # internal name retained for axvspan

    # ── output: path, dpi, transparent ─────────────────────────────────────
    output = _get_section(raw, "output")
    _take(output, "path", "output")
    _take(output, "dpi")
    _take(output, "transparent")

    # any top-level key not part of the nested schema passes through as-is —
    # supports flat legacy keys for backward compatibility / power users.
    # 'y' is excluded: it's handled above via the singular→'ys' rename, and
    # passing it through unchanged would conflict with the --y (scalar) dest.
    _SCHEMA_GROUPS = {"data", "x_axis", "y_axis", "right_axis", "plot",
                      "appearance", "annotations", "regions", "output", "title", "y"}
    for k, v in raw.items():
        if k not in _SCHEMA_GROUPS and k not in flat:
            flat[k] = v

    return flat


def _yaml_depth(obj, _d=0, _budget=None):
    """
    Max nesting depth of a parsed YAML structure; bails early past the depth
    cap, and past a total-node-visit budget (see _MAX_YAML_NODES) so a
    YAML anchor/alias amplification file can't make this safety check itself
    take an extremely long time to run.
    """
    if _budget is None:
        _budget = [_MAX_YAML_NODES]
    _budget[0] -= 1
    if _budget[0] <= 0:
        return _MAX_YAML_DEPTH + 1  # exhausted budget — treat as "too deep", fail closed
    if _d > _MAX_YAML_DEPTH:
        return _d
    if isinstance(obj, dict):
        return max((_yaml_depth(v, _d + 1, _budget) for v in obj.values()), default=_d)
    if isinstance(obj, list):
        return max((_yaml_depth(v, _d + 1, _budget) for v in obj), default=_d)
    return _d


def _load_yaml_defaults(path, parser):
    """
    Load a YAML config file and return (defaults, extras).

    defaults  — dict passed to parser.set_defaults(); keys map 1-to-1 to CLI args.
    extras    — dict of YAML-only advanced settings (YAML_EXTRA_KEYS); applied
                directly in main() at the appropriate stage.

    The user-facing YAML schema is nested by subject (data, x_axis, y_axis,
    right_axis, plot, appearance, output) and is flattened internally by
    _flatten_yaml_schema() before the validation below runs. Flat top-level
    keys not part of the nested schema pass through unchanged for power users.

    Precedence once set_defaults(**defaults) is called:
        explicit CLI flag  >  YAML value  >  argparse built-in default

    Example csvplot.yaml
    ─────────────────────
        data:
          file: data.csv
          x: time
          y: [signal_a, signal_b]
          hue: group
          rolling: 10

        y_axis:
          label: "Amplitude"
          lim: [-2.0, 2.0]
          colors:     [steelblue, darkorange]
          linestyles: ["-", "--"]
          linewidths: [2.0, 1.2]
          alphas:     [1.0, 0.8]
          markers:    [o, null]

        right_axis:
          columns: [reference]
          label: "Reference"
          colors:     [crimson]
          linestyles: [":"]

        x_axis:
          label: "Time (s)"
          lim: [0, 10]
          rotation: 30

        plot:
          mode: hist          # line | scatter | hist | kde | box | violin |
                               # bar | swarm | regplot | pairplot | corr
          bins: 30
          stat: density        # histplot 'stat' kwarg
          multiple: layer       # histplot 'multiple' kwarg
          figsize: [10, 6]

        appearance:
          style: whitegrid
          context: paper
          palette: colorblind
          title_weight: bold
          label_size: 12
          grid:
            alpha: 0.3
            linestyle: ":"
            color: gray
          legend:
            location: upper right
            ncol: 2
            title: "Channels"
            fontsize: 9
            outside: true
          rcParameters:          # arbitrary matplotlib rcParams passthrough;
            axes.titlepad: 12     # invalid keys/values raise matplotlib's
            legend.frameon: false # own errors

        title: "Experiment 3"

        annotations:
          - x: 3.14
            y: 1.0
            text: "π"
            color: red
            arrow: true

        regions:
          - xmin: 0.0
            xmax: 1.0
            color: blue
            alpha: 0.08
            label: "Phase 1"

        output:
          path: figure.png
          dpi: 300
          transparent: true
    """
    try:
        import yaml
    except ImportError:
        die("--yaml requires PyYAML:  pip install pyyaml")
    try:
        with open(path, encoding="utf-8") as fh:
            size_mb = os.fstat(fh.fileno()).st_size / 1e6
            if size_mb > _MAX_YAML_SIZE_MB:
                die(f"{path!r} is {size_mb:.1f} MB, exceeding the {_MAX_YAML_SIZE_MB} MB limit")
            # Note: yaml.safe_load shares (not copies) aliased objects, so this
            # call itself doesn't blow up in memory. The real risk was a naive
            # recursive walk re-visiting a shared object once per reference —
            # that's what _yaml_depth() below used to do; it's now bounded by
            # _MAX_YAML_NODES (see its docstring).
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        die(f"YAML file not found: {path}")
    except PermissionError:
        die(f"permission denied reading YAML: {path}")
    except Exception as e:
        die(f"YAML parse error in {path!r}: {e}")

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        # a YAML file starting with '- item' produces a list, not a mapping
        die(f"YAML: root must be a key-value mapping, "
            f"got {type(raw).__name__!r}; check for accidental leading '-' list syntax")
    if _yaml_depth(raw) > _MAX_YAML_DEPTH:
        die(f"YAML: nesting depth exceeds {_MAX_YAML_DEPTH} levels")

    # convert the nested user-facing schema (data/x_axis/y_axis/.../output)
    # into the flat internal key structure the rest of this function expects
    raw = _flatten_yaml_schema(raw)

    # normalise keys: "log-x" → "log_x" to match argparse dest names
    cfg = {k.replace("-", "_"): v for k, v in raw.items()}
    # strip self-reference: a YAML file containing 'yaml: other.yaml' would be
    # silently ignored (no recursive loading); warn so the author knows.
    nested_yaml = cfg.pop("yaml", None)
    if nested_yaml:
        _warn(f"YAML key 'yaml: {nested_yaml}' in {path!r} is ignored; "
              f"nested YAML loading is not supported")

    # split into argparse-compatible defaults and YAML-only extras
    extras   = {k: v for k, v in cfg.items() if k in YAML_EXTRA_KEYS}
    defaults = {k: v for k, v in cfg.items() if k not in YAML_EXTRA_KEYS}

    # validate argparse-compatible defaults against the live parser
    known   = {a.dest for a in parser._actions}
    choices = {a.dest: a.choices for a in parser._actions if a.choices}
    nargs_p = {a.dest for a in parser._actions if a.nargs in ("+", "*")}

    for k in list(defaults):
        if k not in known:
            _warn(f"unknown YAML key '{k}' ignored")
            defaults.pop(k)
        elif k in choices and defaults[k] not in choices[k]:
            die(f"YAML: invalid value {defaults[k]!r} for '{k}', must be one of {choices[k]}")

    # coerce scalars to lists for nargs='+'/nargs='*' args
    for k in nargs_p:
        if k in defaults and not isinstance(defaults[k], list):
            defaults[k] = [defaults[k]]

    # argparse only coerces types for CLI-supplied values, NOT for set_defaults() values.
    # A YAML entry like 'rolling: "5"' (str) would stay as str and break int comparisons
    # in _validate_args (e.g. val < 1 raises TypeError).  Coerce explicitly here.
    # nargs='+' args are also coerced element-wise (scalar coercion loop above skips them
    # to avoid double-wrapping, but string elements still need type conversion).
    type_map = {a.dest: a.type for a in parser._actions if a.type}
    for k in list(defaults):
        if k not in type_map or defaults[k] is None:
            continue
        try:
            val = defaults[k]
            if k in nargs_p:
                # element-wise coercion for list-valued nargs='+' args
                # e.g. YAML percentile: ["25","50"] -> [25.0, 50.0]
                if isinstance(val, list):
                    defaults[k] = [type_map[k](v) for v in val]
            elif isinstance(val, list):
                # nargs=2 (e.g. --clip) arrives as a list; coerce each element
                defaults[k] = [type_map[k](v) for v in val]
            else:
                defaults[k] = type_map[k](val)
        except (ValueError, TypeError) as e:
            die(f"YAML: cannot coerce '{k}: {defaults[k]!r}' "
                f"to {type_map[k].__name__}: {e}")

    _validate_extras(extras)
    return defaults, extras


def _validate_extras(extras):
    """
    Validate all YAML-only extra keys (those in YAML_EXTRA_KEYS).
    Separated from _load_yaml_defaults so the three validation layers are clear:
      _flatten_yaml_schema   → structural translation
      _load_yaml_defaults    → type coercion + split
      _validate_extras       → semantic validation of extras values
    """
    _EXTRA_CHOICES = {
        "hist_stat":     {"count", "frequency", "density", "probability", "percent"},
        "hist_multiple": {"layer", "stack", "dodge", "fill"},
        "violin_inner":  {"box", "quartile", "point", "stick"},
    }
    for k, valid in _EXTRA_CHOICES.items():
        if k in extras and extras[k] not in valid:
            die(f"YAML: invalid value {extras[k]!r} for '{k}', must be one of {sorted(valid)}")
    for k in ("xlim", "ylim", "dual_y_lim"):
        if k in extras:
            v = extras[k]
            if not isinstance(v, list) or len(v) != 2:
                die(f"YAML: '{k}' must be a two-element list, e.g. {k}: [0, 10]")
            if not all(isinstance(n, (int, float)) for n in v):
                die(f"YAML: '{k}' elements must be numeric, got {v!r}")

    # guard against a scalar/null value where a list is expected (e.g. an empty
    # 'annotations:' YAML key parses to None); len()/enumerate() would otherwise
    # raise an unhandled TypeError instead of a clean die().
    for k in ("annotations", "fill_between"):
        if k in extras and not isinstance(extras[k], list):
            die(f"YAML: '{k}' must be a list, got {type(extras[k]).__name__!r}")

    if len(extras.get("annotations", [])) > _MAX_YAML_LIST_LEN:
        die(f"YAML: 'annotations' has more than {_MAX_YAML_LIST_LEN} entries")
    for i, ann in enumerate(extras.get("annotations", [])):
        if not isinstance(ann, dict):
            die(f"YAML: annotations[{i}] must be a mapping, got {type(ann).__name__!r}")
        missing = {"x", "y", "text"} - ann.keys()
        if missing:
            die(f"YAML: annotations[{i}] is missing required key(s): {sorted(missing)}")
        for coord in ("x", "y"):
            if not isinstance(ann[coord], (int, float)):
                die(f"YAML: annotations[{i}]['{coord}'] must be numeric, got {ann[coord]!r}")

    if len(extras.get("fill_between", [])) > _MAX_YAML_LIST_LEN:
        die(f"YAML: 'fill_between' has more than {_MAX_YAML_LIST_LEN} entries")
    for i, region in enumerate(extras.get("fill_between", [])):
        if not isinstance(region, dict):
            die(f"YAML: fill_between[{i}] must be a mapping, got {type(region).__name__!r}")
        missing = {"xmin", "xmax"} - region.keys()
        if missing:
            die(f"YAML: fill_between[{i}] is missing required key(s): {sorted(missing)}")
        for coord in ("xmin", "xmax"):
            if not isinstance(region[coord], (int, float)):
                die(f"YAML: fill_between[{i}]['{coord}'] must be numeric, "
                    f"got {region[coord]!r}")

    _LINE_STYLE_KEYS = {"color", "linestyle", "linewidth", "alpha"}
    for k in ("hline_style", "vline_style"):
        if k in extras:
            style = extras[k]
            if not isinstance(style, dict) or not set(style) <= _LINE_STYLE_KEYS:
                die(f"YAML: '{k}' must be a mapping using only {sorted(_LINE_STYLE_KEYS)}")
            for nk in ("linewidth", "alpha"):
                if nk in style and not isinstance(style[nk], (int, float)):
                    die(f"YAML: '{k}.{nk}' must be numeric, got {style[nk]!r}")

    for k in ("corr_vmin", "corr_vmax"):
        if k in extras and not isinstance(extras[k], (int, float)):
            die(f"YAML: '{k}' must be numeric, got {extras[k]!r}")
    if "corr_annot" in extras and not isinstance(extras["corr_annot"], bool):
        die(f"YAML: 'corr_annot' must be true/false, got {extras['corr_annot']!r}")


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_args(args):
    """
    Check argument ranges and mutual exclusions before any file I/O.
    All guards run up-front so invalid inputs fail fast with clear messages.
    """
    for name, val in [("head", args.head), ("tail", args.tail), ("sample", args.sample),
                      ("every", args.every), ("rolling", args.rolling), ("ewm", args.ewm),
                      ("top", args.top), ("dpi", args.dpi), ("max_rows", args.max_rows),
                      ("bins", args.bins)]:
        if val is not None and val < 1:
            die(f"--{name} must be >= 1")
    if args.dpi > 2000: die("--dpi must be <= 2000")
    if not 0.0 <= args.alpha <= 1.0: die("--alpha must be between 0.0 and 1.0")
    if not 0.0 <= args.unit_parse_threshold <= 1.0:
        die("--unit-parse-threshold must be between 0.0 and 1.0")
    if args.linewidth <= 0:          die("--linewidth must be > 0")
    if args.interval  <= 0:          die("--interval must be > 0")
    if args.smooth is not None and args.smooth <= 0: die("--smooth sigma must be > 0")
    if not 0 <= args.ci <= 100:      die("--ci must be between 0 and 100")
    if args.percentile and any(not 0.0 <= p <= 100.0 for p in args.percentile):
        die("--percentile values must be in range [0, 100]")
    if args.clip and not all(math.isfinite(v) for v in args.clip):
        die("--clip values must be finite numbers")
    if args.clip and args.clip[0] >= args.clip[1]:
        die("--clip: LO must be strictly less than HI")
    if args.hline and not all(math.isfinite(v) for v in args.hline):
        die("--hline values must be finite numbers")
    if args.vline and not all(math.isfinite(v) for v in args.vline):
        die("--vline values must be finite numbers")
    if args.normalize and args.zscore:
        die("--normalize and --zscore are mutually exclusive")

    # validate --group-by dependent flags
    if args.top and not args.group_by:
        die("--top requires --group-by")
    if args.top_by and not args.group_by:
        die("--top-by requires --group-by")
    if args.agg and not args.group_by:
        die("--agg requires --group-by")
    if args.group_by and not args.agg:
        die("--group-by requires --agg (e.g. --agg mean)")

    # validate --filter expression once here, not on every apply_row_ops call
    if args.filter:
        _validate_filter_expr(args.filter)

    # figsize validated here (fail-fast before any file I/O); the return value
    # is used again in main() when plt.subplots() is called.
    parse_figsize(args.figsize)

    if args.dual_y and (args.hist or args.corr or args.pairplot):
        die("--dual-y is not supported with --hist, --corr, or --pairplot")

    # at most one primary plot mode active at a time; --hist + --kde is the one allowed combo
    active = [m for m in _PLOT_MODE_NAMES if getattr(args, m.replace("-", "_"), False)]
    if len(active) > 1 and set(active) != {"hist", "kde"}:
        die(f"only one plot mode at a time (except --hist + --kde); got: {active}")


# ── Data pipeline ─────────────────────────────────────────────────────────────

# --filter is evaluated via df.query()/pd.eval(), which can execute arbitrary
# attribute access and function calls (e.g. "@pd.io.common.os.system(...)").
# Allow-list the AST node types a column-comparison expression actually needs;
# Call/Attribute/Subscript/Lambda/etc. are rejected outright. Pandas' '@var'
# syntax isn't valid standalone Python either, so ast.parse already rejects it.
_SAFE_FILTER_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.In, ast.NotIn, ast.List, ast.Tuple,
    # Note: bitwise ops (BitAnd/BitOr/BitXor/LShift/RShift) are intentionally
    # excluded — pandas query() engine (numexpr backend) does not support them
    # reliably, so allowing them in the AST would pass validation but fail at
    # df.query() with a cryptic engine error rather than a clean message.
)


def _validate_filter_expr(expr):
    """Reject --filter expressions containing calls, attribute access, or
    other constructs that could execute arbitrary code via df.query()."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        die(f"invalid --filter expression: {e}")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_FILTER_NODES):
            die(f"--filter: disallowed expression element "
                f"{type(node).__name__!r} in {expr!r}")


def apply_row_ops(df, args):
    """
    Apply row-level subset operations in fixed order:
    head → tail → filter → sort → sample → every.
    Index is reset afterward (only when needed) so iloc stays consistent after slicing.
    """
    modified = False
    if args.head   is not None: df = df.head(args.head);   modified = True
    if args.tail   is not None: df = df.tail(args.tail);   modified = True
    if args.filter:
        # expression already validated by _validate_filter_expr in _validate_args
        try:
            df = df.query(args.filter); modified = True
        except Exception as e:
            die(f"invalid --filter expression: {e}")
    if args.sort:
        if args.sort not in df.columns: die(f"--sort column not found: {args.sort!r}")
        df = df.sort_values(args.sort); modified = True
    if args.sample is not None:
        df = df.sample(min(args.sample, len(df)), random_state=args.seed); modified = True
    if args.every  is not None:
        df = df.iloc[::args.every]; modified = True
    return df.reset_index(drop=True) if modified else df


def _drop_infinities(s, col_name):
    """
    Replace ±inf with NaN, warning if any were found. ±inf (e.g. from
    numeric overflow like "1e309") crashes deep inside matplotlib's axis
    tick locator rather than failing gracefully, so this runs wherever
    numeric data reaches a plot — not just apply_transforms' --rate path.
    Returns (cleaned_series, whether anything changed).
    """
    mask = np.isinf(s)
    if not mask.any():
        return s, False
    n_inf = int(mask.sum())
    _warn(f"column '{col_name}' — {n_inf} value(s) were ±infinite "
          f"(numeric overflow?) and became NaN")
    return s.replace([np.inf, -np.inf], np.nan), True


def apply_transforms(df, args, cols):
    """
    Apply column-level numeric transforms in a fixed, chainable order.
    Each step reassigns 's' so the next transform chains onto the result,
    not the original column values.
    """
    for c in cols:
        if c not in df:
            continue
        s = pd.to_numeric(df[c], errors="coerce")

        # warn when coercion silently destroyed data (e.g. typo in column name
        # that passes existence checks but contains strings)
        coerced_nan = s.isna().sum() - df[c].isna().sum()
        if coerced_nan > 0:
            _warn(f"column '{c}' — {coerced_nan} value(s) "
                  f"could not be coerced to numeric and became NaN")

        s, had_inf = _drop_infinities(s, c)
        if had_inf:
            df[c] = s

        if args.cumsum:    s = s.cumsum();                                     df[c] = s
        if args.cummax:    s = s.cummax();                                     df[c] = s
        if args.cummin:    s = s.cummin();                                     df[c] = s
        if args.diff:      s = s.diff();                                       df[c] = s
        if args.rate:
            # pct_change() yields ±Inf on division-by-zero; treat as undefined, not Inf
            s = s.pct_change().replace([np.inf, -np.inf], np.nan)
            df[c] = s
        if args.ewm:       s = s.ewm(span=args.ewm, min_periods=1).mean();    df[c] = s
        if args.rolling:   s = s.rolling(args.rolling, min_periods=1).mean(); df[c] = s
        if args.smooth:
            s = pd.Series(gaussian_smooth(s.values, args.smooth), index=s.index); df[c] = s
        if args.clip:      s = s.clip(*args.clip);                             df[c] = s
        if args.baseline:
            first = s.dropna().iloc[0] if s.notna().any() else 0.0
            s = s - first;                                                     df[c] = s
        if args.normalize:
            mn, mx = s.min(), s.max()
            if s.isna().all():
                _warn(f"column '{c}' is all-NaN after transforms; normalize skipped")
                df[c] = s  # keep as numeric NaN, not the original (possibly non-numeric) data
            else:
                # constant column has undefined range — map to 0 by convention
                df[c] = (s - mn) / (mx - mn) if mx != mn else pd.Series(0.0, index=s.index)
        elif args.zscore:
            std = s.std()
            if pd.isna(std) or std == 0:
                # std is NaN (all-NaN/single-value) or zero (constant series); either way
                # every value equals the mean, so z=0 is well-defined — same 0-by-convention
                # as the constant-column case in --normalize, rather than NaN
                _warn(f"column '{c}' has std={std!r}; z-score maps to 0")
                df[c] = pd.Series(0.0, index=s.index)
            else:
                df[c] = (s - s.mean()) / std
    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────

def parse_figsize(raw):
    """Parse a 'W,H' string into a (float, float) tuple; die on bad or non-positive input."""
    try:
        w, h = raw.split(",")
        w, h = float(w), float(h)
    except (ValueError, AttributeError):
        die(f"--figsize must be 'W,H' (e.g. '10,6'), got: {raw!r}")
    if not (math.isfinite(w) and math.isfinite(h)):
        die(f"--figsize dimensions must be finite numbers, got: {raw!r}")
    if w <= 0 or h <= 0:
        die(f"--figsize dimensions must be positive, got: {raw!r}")
    return w, h


def _active_mode(args):
    """
    Return the active plot mode name (first match wins).
    --kde alone is its own mode; when combined with --hist it becomes an
    overlay (histplot(kde=True)) rather than a standalone kdeplot. pairplot
    is excluded — main() always dispatches it to _render_special() before
    _build_figure() (and this function) is ever reached.
    """
    for mode in _FIGURE_MODES:
        if getattr(args, mode.replace("-", "_"), False):
            return mode
    return "kde" if args.kde else "line"


def _resolve_plot_columns(df, ycols, allow_implicit):
    """
    Return the y-columns to actually plot: explicit --y/--ys, or (when
    allow_implicit) every numeric column except the internal _idx sentinel.
    Shared by _build_figure's hist/kde panel sizing and per-frame plotting,
    and by _render_special's corr/pairplot (which always allow it) — one
    place to exclude _idx rather than three slightly different copies of
    the same "ycols or implicit numeric columns" logic.
    """
    if ycols or not allow_implicit:
        return ycols or []
    return [c for c in df.select_dtypes(include="number").columns if c != _IDX_COL]


def _categorical_x(x_col):
    """
    Return x_col for categorical modes (box/violin/swarm), or None if x_col is
    just the internal row-index sentinel (_IDX_COL) rather than a real column —
    grouping by row number would produce one degenerate box/violin per row.
    Checking x_col itself (not args.x) means a --group-by-derived x_col is
    correctly treated as a real grouping column even though --x wasn't set.
    """
    return x_col if x_col != _IDX_COL else None


def _style_val(yaml_x, key, idx, fallback):
    """
    Return yaml_x[key][idx] if it exists and is not None, else fallback.
    Used by both the left-axis per-series loop and the dual-y series loop
    to look up YAML per-series style overrides (colors, linestyles, etc.).
    """
    lst = yaml_x.get(key, [])
    return lst[idx] if idx < len(lst) and lst[idx] is not None else fallback


def _series_style(yaml_x, prefix, idx):
    """
    Return this series' explicit style overrides only (color/linestyle/linewidth/
    alpha/marker) — no fallback baked in, so left-axis and dual-y-axis callers can
    each apply their own default. prefix is 'y_' (left axis) or 'dual_y_' (right axis).
    """
    styles = {}
    for attr, suffix in _STYLE_ATTRS:
        val = _style_val(yaml_x, f"{prefix}{suffix}", idx, None)
        if val is not None:
            styles[attr] = val
    return styles


def _color_override(so, args, hue):
    """
    Return {'color': ...} if an explicit per-series or global --color should be
    applied, else {} (letting seaborn's hue-based coloring take over). Shared by
    _plot_single's line_kw and dist_kw so the same rule can't drift between them.
    """
    color = so.get("color") or (args.color if not hue else None)
    return {"color": color} if color else {}


@dataclass
class _PlotContext:
    """
    Bundles the state every per-mode plot handler needs. Avoids repeating an
    8+ parameter signature across every _plot_<mode> function below, and
    keeps _plot_single's own signature unchanged for existing callers.
    """
    df: object
    x_col: str
    y_col: str
    label: str
    ax: object
    args: object
    hue: object
    palette: object
    so: dict
    px: dict
    line_kw: dict
    dist_kw: dict


def _plot_line(ctx):
    sns.lineplot(data=ctx.df, x=ctx.x_col, y=ctx.y_col, hue=ctx.hue,
                 label=ctx.label if not ctx.hue else None,
                 palette=ctx.palette, ax=ctx.ax, **ctx.line_kw)


def _plot_scatter(ctx):
    scatter_kw = {**ctx.dist_kw}
    if "scatter_s" in ctx.px:
        scatter_kw["s"] = ctx.px["scatter_s"]
    sns.scatterplot(data=ctx.df, x=ctx.x_col, y=ctx.y_col, hue=ctx.hue,
                    label=ctx.label if not ctx.hue else None,
                    palette=ctx.palette, ax=ctx.ax, **scatter_kw)


def _plot_regplot(ctx):
    args, hue, px = ctx.args, ctx.hue, ctx.px
    reg_kw = {
        "ci":     args.ci if args.ci > 0 else None,
        "order":  px.get("regplot_order", 1),
        "robust": px.get("regplot_robust", False),
        "scatter_kws": {"alpha": args.alpha * 0.5},
        "line_kws":    {"linewidth": args.linewidth},
    }
    if not hue:
        sns.regplot(data=ctx.df, x=ctx.x_col, y=ctx.y_col, ax=ctx.ax, **reg_kw)
        return
    n_hue = ctx.df[hue].nunique()
    if n_hue > 20:
        die(f"--regplot --hue {hue!r} has {n_hue} unique values; "
            f"per-group regression requires a low-cardinality "
            f"categorical column (≤ 20)")
    if ctx.df[hue].isna().any():
        _warn(f"--hue {hue!r} has NaN values; those rows are excluded from --regplot")
    colors = sns.color_palette(ctx.palette, n_colors=n_hue)
    for j, (hval, grp) in enumerate(ctx.df.groupby(hue)):
        sns.regplot(data=grp, x=ctx.x_col, y=ctx.y_col, ax=ctx.ax,
                    label=str(hval), color=colors[j], **reg_kw)


def _plot_hist(ctx):
    bins_kw = {"bins": ctx.args.bins} if ctx.args.bins is not None else {}
    sns.histplot(data=ctx.df, x=ctx.y_col, hue=ctx.hue,
                 kde=ctx.args.kde,
                 stat=ctx.px.get("hist_stat", "count"),
                 multiple=ctx.px.get("hist_multiple", "layer"),
                 palette=ctx.palette, ax=ctx.ax, **bins_kw, **ctx.dist_kw)


def _plot_kde(ctx):
    so, args, px = ctx.so, ctx.args, ctx.px
    sns.kdeplot(data=ctx.df, x=ctx.y_col, hue=ctx.hue, fill=True,
                alpha=min(so.get("alpha", args.alpha), 0.4),
                linewidth=so.get("linewidth", args.linewidth),
                bw_adjust=px.get("kde_bw_adjust", 1.0),
                cut=px.get("kde_cut", 3),
                palette=ctx.palette, ax=ctx.ax)


def _plot_box(ctx):
    sns.boxplot(data=ctx.df, x=_categorical_x(ctx.x_col), y=ctx.y_col,
                hue=ctx.hue, palette=ctx.palette, ax=ctx.ax)


def _plot_violin(ctx):
    sns.violinplot(data=ctx.df, x=_categorical_x(ctx.x_col), y=ctx.y_col,
                   hue=ctx.hue, palette=ctx.palette, ax=ctx.ax,
                   inner=ctx.px.get("violin_inner", "box"))


def _plot_bar(ctx):
    args = ctx.args
    ci_kw = ({"errorbar": ("ci", args.ci) if args.ci > 0 else None}
             if _SNS_GE_012 else {"ci": args.ci if args.ci > 0 else None})
    sns.barplot(data=ctx.df, x=_categorical_x(ctx.x_col), y=ctx.y_col, hue=ctx.hue,
                palette=ctx.palette, ax=ctx.ax, **ci_kw)


def _plot_swarm(ctx):
    sns.swarmplot(data=ctx.df, x=_categorical_x(ctx.x_col), y=ctx.y_col,
                  hue=ctx.hue, palette=ctx.palette, ax=ctx.ax, **ctx.dist_kw)


# mode name -> handler; adding a mode means adding one function and one
# entry here, not a new branch in a growing if/elif chain
_MODE_HANDLERS = {
    "line": _plot_line, "scatter": _plot_scatter, "regplot": _plot_regplot,
    "hist": _plot_hist, "kde": _plot_kde, "box": _plot_box,
    "violin": _plot_violin, "bar": _plot_bar, "swarm": _plot_swarm,
}


def _plot_single(df, x_col, y_col, label, mode, ax, args, so=None, px=None):
    """
    Render one (x, y) series by dispatching to _MODE_HANDLERS[mode].

    so (series_overrides): per-series style dict from yaml_extra y_colors/y_linestyles/…
    px (plot_extras):      full yaml_extra dict for mode-specific params
                           (hist_stat, violin_inner, scatter_s, etc.)
    """
    so = so or {}
    px = px or {}
    hue     = args.hue
    palette = args.palette

    # merge per-series overrides with global args; overrides win
    color_kw = _color_override(so, args, hue)  # shared by line_kw and dist_kw below
    line_kw = {
        "linewidth": so.get("linewidth", args.linewidth),
        "alpha":     so.get("alpha",     args.alpha),
        **color_kw,
        **({"marker":   so.get("marker") or args.marker}
           if (so.get("marker") or args.marker) else {}),
        **({"linestyle": so["linestyle"]} if "linestyle" in so else {}),
    }
    dist_kw = {
        "alpha": so.get("alpha", args.alpha),
        **color_kw,
    }

    handler = _MODE_HANDLERS.get(mode)
    if handler is None:
        die(f"unknown plot mode: {mode!r}")
    handler(_PlotContext(df=df, x_col=x_col, y_col=y_col, label=label, ax=ax, args=args,
                         hue=hue, palette=palette, so=so, px=px,
                         line_kw=line_kw, dist_kw=dist_kw))


def _overlay_linear_fit(df, x_col, y_col, ax, args):
    """
    Overlay a dashed OLS line on an existing axes.
    Silently skips when fewer than 2 valid numeric (x, y) pairs exist.
    Catches numpy.RankWarning for near-constant or degenerate X data.
    """
    xv   = pd.to_numeric(df[x_col], errors="coerce").values
    yv   = pd.to_numeric(df[y_col],  errors="coerce").values
    mask = ~(np.isnan(xv) | np.isnan(yv))
    if mask.sum() < 2:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", _RankWarning)
            m, b = np.polyfit(xv[mask], yv[mask], 1)
    except (_RankWarning, np.linalg.LinAlgError) as e:
        _warn(f"linear fit for '{y_col}' skipped — {e}")
        return
    xs = np.sort(xv[mask])  # sort so the fit line draws cleanly left-to-right
    ax.plot(xs, m * xs + b, "--",
            linewidth=args.linewidth, alpha=args.alpha, label=f"{y_col} fit")


def _decorate_axes(ax, args, yaml_x=None):
    """
    Apply reference lines, log scale, grid style, axis limits, and tick
    formatting. yaml_x provides optional fine-grained overrides for each.
    """
    yaml_x = yaml_x or {}
    if args.hline:
        style = {"linestyle": "--", "linewidth": 0.8, "color": "grey", "alpha": 0.6,
                  **yaml_x.get("hline_style", {})}
        for yv in args.hline:
            ax.axhline(yv, **style)
    if args.vline:
        style = {"linestyle": "--", "linewidth": 0.8, "color": "grey", "alpha": 0.6,
                  **yaml_x.get("vline_style", {})}
        for xv in args.vline:
            ax.axvline(xv, **style)
    if args.log_x:
        x0, _ = ax.dataLim.intervalx
        if x0 <= 0:
            _warn("--log-x skipped — x data includes zero or "
                  "negative values, which cannot be shown on a log scale")
        else:
            ax.set_xscale("log")
    if args.log_y:
        y0, _ = ax.dataLim.intervaly
        if y0 <= 0:
            _warn("--log-y skipped — y data includes zero or "
                  "negative values, which cannot be shown on a log scale")
        else:
            ax.set_yscale("log")
    if args.grid:
        grid_kw = {}
        if "grid_alpha"     in yaml_x: grid_kw["alpha"]     = yaml_x["grid_alpha"]
        if "grid_linestyle" in yaml_x: grid_kw["linestyle"] = yaml_x["grid_linestyle"]
        if "grid_color"     in yaml_x: grid_kw["color"]     = yaml_x["grid_color"]
        ax.grid(True, **grid_kw)
    if yaml_x.get("xlim"): ax.set_xlim(yaml_x["xlim"])
    if yaml_x.get("ylim"): ax.set_ylim(yaml_x["ylim"])
    if yaml_x.get("xtick_rotation") is not None:
        ax.tick_params(axis="x", labelrotation=yaml_x["xtick_rotation"])
    if yaml_x.get("ytick_rotation") is not None:
        ax.tick_params(axis="y", labelrotation=yaml_x["ytick_rotation"])


def _save_or_show(fig, args, yaml_x=None):
    """Save to --output or show interactively; wrap both in explicit exception handling."""
    yaml_x = yaml_x or {}
    if args.output:
        out_path = Path(args.output)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        # derive format from out_path suffix, NOT tmp_path (which ends in .tmp).
        # default "png" so extension-less output names don't cause matplotlib to
        # infer format from ".tmp" and raise ValueError: Format 'tmp' not supported.
        fmt = out_path.suffix[1:] or "png"
        try:
            fig.savefig(tmp_path, format=fmt, dpi=args.dpi, bbox_inches="tight",
                        transparent=yaml_x.get("transparent", False))
            os.replace(tmp_path, out_path)  # atomic — avoids partial/corrupted output
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            die(f"could not save figure to {args.output!r}: {e}")
        print(f"saved: {args.output}", file=sys.stderr)
    else:
        try:
            plt.show()
        except Exception as e:
            die(f"could not display figure: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _coerce_units(df, threshold, require_suffix):
    """
    Coerce string columns that contain SI/data-rate unit suffixes to float.
    Operates on a copy so the original frame is never mutated in-place.
    """
    df = df.copy()
    for c in df.columns:
        # is_string_dtype covers both legacy object and pandas 3.x StringDtype
        if pd.api.types.is_string_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            try:
                df[c] = parse_units(df[c], threshold=threshold,
                                    require_suffix=require_suffix)
            except Exception as e:
                _warn(f"unit parsing failed for column '{c}': {e}")
    return df


def _finalize_figure(fig, args, yaml_x, close_all=False):
    """
    Shared finalisation: tight_layout (non-fatal), save or show, then close.
    close_all=True is used for the pairplot path where plt created the figure
    internally (g.figure) and plt.close("all") is the appropriate cleanup.
    """
    try:
        plt.tight_layout()
    except Exception as e:
        _warn(f"tight_layout failed ({e}); output may have clipping")
    _save_or_show(fig, args, yaml_x)
    plt.close("all" if close_all else fig)


def _series_range(frames, col):
    """
    Return the numeric range (max - min) of `col` across all frames, ignoring NaN.
    Returns 0.0 if the column is all-NaN in every frame or cannot be coerced to numeric.
    Used by the dual-y scale-ratio heuristic; must never raise.
    Uses nanmin/nanmax to avoid NaN propagation from min()/max() over numpy scalars.
    """
    try:
        vals = pd.concat(
            pd.to_numeric(df[col], errors="coerce") for df in frames.values()
        ).dropna()
        if vals.empty:
            return 0.0
        return float(vals.max() - vals.min())
    except Exception:
        return 0.0


def _parse_args():
    """Two-pass argument parsing: first extract --yaml, then full parse with YAML defaults."""
    p = build_parser()
    yaml_x = {}
    pre, _ = p.parse_known_args()
    if pre.yaml:
        yaml_defs, yaml_x = _load_yaml_defaults(pre.yaml, p)
        p.set_defaults(**yaml_defs)
    args = p.parse_args()
    if args.watch or args.alert:
        die("--watch and --alert are not yet implemented")
    return args, yaml_x


def _check_output(args):
    """Validate output path early — before expensive rendering."""
    if not args.output:
        return
    out_path = Path(args.output)
    if not out_path.parent.exists():
        die(f"output directory does not exist: {out_path.parent}")
    if out_path.exists():
        if not args.force:
            die(f"output file already exists: {out_path} (use --force to overwrite)")
        if not os.access(out_path, os.W_OK):
            die(f"output file is not writable: {out_path}")
    else:
        if not os.access(out_path.parent, os.W_OK):
            die(f"output directory is not writable: {out_path.parent}")


def _apply_theme(args, yaml_x):
    """Apply seaborn theme; font_scale and rc are yaml-only extensions."""
    try:
        sns.set_theme(style=args.style, context=args.context,
                      palette=args.palette or "muted",
                      font_scale=yaml_x.get("font_scale", 1.0),
                      rc=yaml_x.get("rc") or {})
    except Exception as e:
        die(f"invalid theme settings (check --style/--context/--palette): {e}")


def _load_frames(args):
    """Load all input files, coerce units, and return validated frames dict."""
    if args.files and args.file and args.file != "-":
        _warn(f"--file {args.file!r} ignored because --files was also specified")
    sources = args.files if args.files else ([args.file] if args.file != "-" else ["-"])
    real_sources = [os.path.realpath(s) for s in sources if s != "-"]
    if len(set(real_sources)) != len(real_sources):
        die(f"duplicate input file specified (paths resolve to the same file): {sources}")
    # only safe to read exactly --head rows when head is the SOLE row-op — the
    # documented order is head -> tail -> filter -> sort -> sample -> every, so
    # any of the others needs the full (up to --max-rows) file to apply correctly
    solo_head = (args.head if (args.head is not None and args.tail is None and not args.filter
                                and not args.sort and args.sample is None and args.every is None)
                 else None)
    frames = {src: load_dataframe(src, args.sep, args.max_rows, head_hint=solo_head)
              for src in sources}
    if not args.no_parse_units:
        frames = {s: _coerce_units(df,
                                   threshold=args.unit_parse_threshold,
                                   require_suffix=not args.unit_parse_allow_bare_numbers)
                  for s, df in frames.items()}
    empty_sources = [s for s, df in frames.items() if df.empty]
    if empty_sources:
        die(f"empty dataset: {empty_sources if len(empty_sources) > 1 else empty_sources[0]}")
    return frames


def _resolve_columns(frames, args):
    """Resolve and validate ycols against every frame; return ycols list."""
    ycols = args.ys if args.ys else ([args.y] if args.y else [])
    if args.ys and args.y:
        _warn(f"--y {args.y!r} ignored because --ys was also specified")
    ycols = list(dict.fromkeys(ycols))  # deduplicate, preserve order

    y_optional = (args.hist or args.corr or args.percentile or args.pairplot or args.kde)
    if not ycols:
        if args.group_by:
            die("--group-by requires at least one y column via --y or --ys")
        if not y_optional:
            die("specify at least one y column with --y or --ys")

    for src, df in frames.items():
        tag = f" (file: {src!r})" if len(frames) > 1 else ""
        if args.x      and args.x      not in df.columns: die(f"--x column not found: {args.x!r}{tag}")
        if args.hue    and args.hue    not in df.columns: die(f"--hue column not found: {args.hue!r}{tag}")
        if args.group_by and args.group_by not in df.columns:
            die(f"--group-by column not found: {args.group_by!r}{tag}")
        for c in ycols:
            if c not in df.columns: die(f"--y column not found: {c!r}{tag}")
            if args.group_by and not pd.api.types.is_numeric_dtype(df[c]):
                die(f"--group-by requires numeric y columns for aggregation; "
                    f"{c!r} is not numeric{tag}")
        for c in (args.dual_y or []):
            if c not in df.columns: die(f"--dual-y column not found: {c!r}{tag}")

    if args.dual_y:
        overlap = set(ycols) & set(args.dual_y)
        if overlap:
            die(f"column(s) appear in both --ys and --dual-y: {sorted(overlap)}")
        if args.group_by:
            die("--dual-y cannot be combined with --group-by: aggregation drops "
                "non-y columns. Aggregate first, then plot with --dual-y.")

    if args.top_by and args.top_by not in ycols:
        die(f"--top-by must be one of --y/--ys: {args.top_by!r} not in {ycols}")

    return ycols


def _run_pipeline(frames, args, ycols):
    """
    Apply row ops, per-group or global transforms, group aggregation, and
    x-column resolution.  Returns (frames, x_col) ready for plotting.
    """
    frames = {s: apply_row_ops(df, args) for s, df in frames.items()}
    if args.group_by:
        # rolling/diff/rate/etc. must operate within each group's row order, not
        # across group boundaries. Manual iterate+concat avoids pandas 3.x dropping
        # the group column from groupby().apply() results.
        frames = {
            s: pd.concat(
                apply_transforms(g.copy(), args, ycols)
                for _, g in df.groupby(args.group_by, group_keys=False, sort=False)
            ).sort_index()
            for s, df in frames.items()
        }
    else:
        frames = {s: apply_transforms(df, args, ycols) for s, df in frames.items()}

    if args.group_by:
        frames = {
            s: df.groupby(args.group_by)[ycols].agg(args.agg).reset_index()
            for s, df in frames.items()
        }
        if args.top:
            top_col = args.top_by or ycols[0]
            frames = {s: df.nlargest(args.top, top_col) for s, df in frames.items()}

    local_x = args.x or (args.group_by if args.group_by else None)
    if not local_x:
        for s, df in frames.items():
            if _IDX_COL in df.columns:
                _warn(f"column {_IDX_COL!r} will be overwritten by the internal row-index sentinel in {s!r}")
        frames = {s: df.assign(**{_IDX_COL: df.index}) for s, df in frames.items()}
        x_col = _IDX_COL
    else:
        x_col = local_x

    return frames, x_col


def _print_headers(frames):
    """Print each file's column names, one per line — no --y/--ys required."""
    multi = len(frames) > 1
    for src, df in frames.items():
        if multi:
            print(f"== {src} ==")
        for col in df.columns:
            print(col)


def _print_info(frames):
    """
    Print each file's shape, column names, dtypes, non-null counts, and an
    example value per column — no --y/--ys required. Meant as a first look
    at an unfamiliar file before deciding what to plot.
    """
    for src, df in frames.items():
        print(f"== {src} ==" if len(frames) > 1 else f"{src}")
        n_rows, n_cols = len(df), len(df.columns)
        print(f"{n_rows} row{'s' if n_rows != 1 else ''}, "
              f"{n_cols} column{'s' if n_cols != 1 else ''}\n")
        non_null_strs = {c: f"{int(df[c].notna().sum())}/{n_rows}" for c in df.columns}
        name_w  = max([len("column")]   + [len(c) for c in df.columns])
        dtype_w = max([len("dtype")]    + [len(str(df[c].dtype)) for c in df.columns])
        nn_w    = max([len("non-null")] + [len(s) for s in non_null_strs.values()])
        print(f"  {'column':<{name_w}}  {'dtype':<{dtype_w}}  {'non-null':<{nn_w}}  example")
        for col in df.columns:
            example = df[col].dropna().iloc[0] if df[col].notna().any() else ""
            print(f"  {col:<{name_w}}  {str(df[col].dtype):<{dtype_w}}  "
                  f"{non_null_strs[col]:<{nn_w}}  {example}")
        print()


def _text_output(frames, args, ycols):
    """
    Handle all text-only output modes (--stats, --export-json, --percentile).
    Returns True if a text mode was active and the caller should return early.
    """
    if args.stats:
        for src, df in frames.items():
            cols = ycols or df.select_dtypes(include="number").columns.tolist()
            print(f"=== {src} ===" if len(frames) > 1 else "=== descriptive statistics ===",
                  file=sys.stderr)
            print(df[cols].describe().to_string())
        return True

    if args.export_json:
        out = {s: {c: [_json_safe(v) for v in vals]
                   for c, vals in df.to_dict(orient="list").items()}
               for s, df in frames.items()}
        print(json.dumps(out))
        return True

    if args.percentile:
        q = [p / 100.0 for p in args.percentile]
        for s, df in frames.items():
            cols = ycols or df.select_dtypes(include="number").columns.tolist()
            if not cols:
                die("--percentile: no numeric columns found; use --y to specify")
            try:
                print(df[cols].quantile(q).to_string())
            except TypeError as e:
                die(f"--percentile requires numeric columns: {e}")
        return True

    return False


def _render_special(frames, args, ycols, yaml_x):
    """
    Handle plot modes that create their own figure (pairplot, corr).
    Returns True if the mode was active and the caller should return early.
    """
    if args.pairplot:
        if len(frames) > 1:
            _warn(f"--pairplot uses only the first file; {len(frames)-1} additional file(s) ignored")
        df   = next(iter(frames.values()))
        cols = _resolve_plot_columns(df, ycols, True)
        if len(cols) < 2: die("--pairplot requires at least 2 numeric columns")
        if len(cols) > 6:
            _warn(f"pairplot with {len(cols)} columns produces {len(cols)**2} panels and will be slow; "
                  f"capping at 6. Use --ys to select specific columns.")
            cols = cols[:6]
        pcols = cols + ([args.hue] if args.hue and args.hue not in cols else [])
        df_pp = df[pcols].copy()
        for c in cols:
            df_pp[c], _ = _drop_infinities(pd.to_numeric(df_pp[c], errors="coerce"), c)
        if len(df_pp) > _PAIRPLOT_MAX_ROWS:
            _warn(f"pairplot on {len(df_pp)} rows; sampling {_PAIRPLOT_MAX_ROWS} for rendering")
            df_pp = df_pp.sample(_PAIRPLOT_MAX_ROWS, random_state=args.seed)
        try:
            try:
                g = sns.pairplot(df_pp, hue=args.hue, diag_kind="kde",
                                 plot_kws={"alpha": args.alpha})
            except Exception as e:
                _warn(f"pairplot KDE diagonal failed ({e}); falling back to histogram")
                g = sns.pairplot(df_pp, hue=args.hue, diag_kind="hist",
                                 plot_kws={"alpha": args.alpha})
        except (ValueError, OverflowError) as e:
            die(f"could not render pairplot: {e} (often caused by "
                f"extremely large or non-finite values)")
        if args.title:
            g.figure.suptitle(args.title, y=1.02)
        _finalize_figure(g.figure, args, yaml_x, close_all=True)
        return True

    if args.corr:
        if len(frames) > 1:
            _warn(f"--corr uses only the first file; {len(frames)-1} additional file(s) ignored")
        df   = next(iter(frames.values()))
        cols = _resolve_plot_columns(df, ycols, True)
        if len(cols) < 2: die("--corr requires at least 2 numeric columns")
        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            die(f"--corr requires numeric columns; non-numeric: {non_numeric}")
        df_corr = df[cols].copy()
        for c in cols:
            df_corr[c], _ = _drop_infinities(df_corr[c], c)
        corr = df_corr.corr(method=args.corr_method)
        if corr.isna().all(axis=None):
            _warn("correlation matrix is all-NaN (constant or all-NaN columns?)")
        fig, ax = plt.subplots(figsize=parse_figsize(args.figsize))
        try:
            sns.heatmap(corr, annot=yaml_x.get("corr_annot", True),
                        fmt=yaml_x.get("corr_fmt", ".2f"), cmap=yaml_x.get("corr_cmap", "coolwarm"),
                        vmin=yaml_x.get("corr_vmin", -1), vmax=yaml_x.get("corr_vmax", 1),
                        square=True, linewidths=0.5, ax=ax)
        except (ValueError, OverflowError) as e:
            die(f"could not render correlation heatmap: {e}")
        print(corr.to_string())
        if args.title:
            fig.suptitle(args.title, fontsize="large")
        _finalize_figure(fig, args, yaml_x)
        return True

    return False


def _build_figure(frames, args, ycols, x_col, yaml_x):
    """
    Build the standard figure: axes setup, per-series plotting, dual-y,
    axis decoration, annotations, title, and finalization.
    """
    mode    = _active_mode(args)
    figsize = parse_figsize(args.figsize)
    # panel count must reflect what will actually be plotted, including the
    # implicit-numeric-columns fallback for --hist/--kde with no --y/--ys given —
    # otherwise --subplots silently collapses to one panel. Uses the first frame;
    # _resolve_plot_columns() is called again per-frame in the loop below, so
    # --files with heterogeneous columns still plot correctly per file.
    panel_cols = _resolve_plot_columns(next(iter(frames.values())), ycols, mode in {"hist", "kde"})
    n_panels = len(panel_cols) if args.subplots and len(panel_cols) > 1 else 1

    if n_panels > 1:
        figsize = (figsize[0], max(figsize[1], n_panels * 2.0))  # floor: keep stacked panels from getting too cramped
        fig, ax_arr = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
        axs = list(ax_arr)
    else:
        fig, ax0 = plt.subplots(figsize=figsize)
        axs = [ax0] * max(1, len(ycols))

    # ── per-series plotting ───────────────────────────────────────────────
    stems = [Path(s).stem for s in frames]
    prefixes = ({s: Path(s).stem + "/"      for s in frames} if len(set(stems)) == len(stems)
                else {s: Path(s).as_posix() + "/" for s in frames})

    for src, df in frames.items():
        prefix = prefixes[src] if len(frames) > 1 else ""
        cols_to_plot = _resolve_plot_columns(df, ycols, mode in {"hist", "kde"})
        if not cols_to_plot:
            die(f"no columns to plot for --{mode}")

        if mode in _CAT_MODES and len(cols_to_plot) > 1 and not args.x and not args.subplots:
            id_vars = [c for c in df.columns if c not in cols_to_plot and c != _IDX_COL]
            df_long = df.melt(id_vars=id_vars or None, value_vars=cols_to_plot,
                              var_name="_variable", value_name="_value")
            try:
                _plot_single(df_long, "_variable", "_value", prefix + "data", mode, axs[0], args,
                             px=yaml_x)
            except (ValueError, OverflowError) as e:
                die(f"could not render {cols_to_plot}: {e} (often caused by "
                    f"extremely large or non-finite values)")
        else:
            for i, c in enumerate(cols_to_plot):
                ax_i = axs[min(i, len(axs) - 1)]
                so = _series_style(yaml_x, "y_", i)
                try:
                    _plot_single(df, x_col, c, prefix + c, mode, ax_i, args, so=so, px=yaml_x)
                except (ValueError, OverflowError) as e:
                    die(f"could not render '{c}': {e} (often caused by "
                        f"extremely large or non-finite values)")
                if args.fit == "linear" and mode in {"line", "scatter"}:
                    _overlay_linear_fit(df, x_col, c, ax_i, args)

    unique_axs = list(dict.fromkeys(axs))

    # ── dual-y right axis ─────────────────────────────────────────────────
    ax_r_map = {}
    if args.dual_y:
        n_left   = len(ycols)
        full_pal = sns.color_palette(args.palette or "muted",
                                     n_colors=n_left + len(args.dual_y) * len(frames))
        try:
            lr = max((_series_range(frames, c) for c in ycols), default=0)
            rr = max((_series_range(frames, c) for c in args.dual_y), default=0)
            if lr > 0 and rr > 0 and max(lr, rr) / min(lr, rr) > 1000:
                _warn(f"dual-y scale ratio ≈ {max(lr, rr) / min(lr, rr):.0f}× — axes may be "
                      f"visually misleading; consider --normalize or --zscore before plotting")
        except Exception:
            pass
        for ax in unique_axs:
            ax_r = ax.twinx()
            ax_r_map[id(ax)] = ax_r
            color_idx = 0  # reset per panel so the same series keeps the same default color on every panel
            for src, df in frames.items():
                for j, c in enumerate(args.dual_y):
                    so        = _series_style(yaml_x, "dual_y_", j)
                    color     = so.get("color", full_pal[(n_left + color_idx) % len(full_pal)])
                    linestyle = so.get("linestyle", "--")
                    linewidth = so.get("linewidth", args.linewidth)
                    alpha     = so.get("alpha", args.alpha)
                    marker    = so.get("marker")
                    color_idx += 1
                    kw = {"linestyle": linestyle, "linewidth": linewidth,
                          "alpha": alpha, "color": color}
                    if marker:
                        kw["marker"] = marker
                    try:
                        sns.lineplot(data=df, x=x_col, y=c, ax=ax_r, label=f"{c} →", **kw)
                    except (ValueError, OverflowError) as e:
                        die(f"could not render dual-y '{c}': {e} (often caused by "
                            f"extremely large or non-finite values)")
            ax_r.set_ylabel(args.dual_y_label or ", ".join(args.dual_y))
            if yaml_x.get("dual_y_lim"):
                ax_r.set_ylim(yaml_x["dual_y_lim"])

    # ── axes decoration ───────────────────────────────────────────────────
    lgd_kw = {"loc": yaml_x.get("legend_loc", "best")}
    if "legend_ncol"     in yaml_x: lgd_kw["ncol"]     = yaml_x["legend_ncol"]
    if "legend_title"    in yaml_x: lgd_kw["title"]    = yaml_x["legend_title"]
    if "legend_fontsize" in yaml_x: lgd_kw["fontsize"] = yaml_x["legend_fontsize"]
    if yaml_x.get("legend_outside"):
        lgd_kw.update({"bbox_to_anchor": (1.05, 1), "loc": "upper left", "borderaxespad": 0})

    for i, ax in enumerate(unique_axs):
        _decorate_axes(ax, args, yaml_x)
        if n_panels > 1 and i < len(ycols):
            ax.set_ylabel(args.ylabel or ycols[i])
        elif args.ylabel:
            ax.set_ylabel(args.ylabel)
        if id(ax) in ax_r_map:
            ax_r = ax_r_map[id(ax)]
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_r.get_legend_handles_labels()
            if h1 or h2:
                ax.legend(h1 + h2, l1 + l2, **lgd_kw)
        elif args.legend or (len(ycols) > 1 and n_panels == 1):
            ax.legend(**lgd_kw)

    if args.xlabel:
        unique_axs[-1].set_xlabel(args.xlabel)
    elif x_col == _IDX_COL:
        unique_axs[-1].set_xlabel("index")

    if args.title:
        fig.suptitle(args.title, fontsize="large")

    # ── post-plot overlays (yaml-only) ────────────────────────────────────
    for i, ann in enumerate(yaml_x.get("annotations", [])):
        off = ann.get("xytext_offset", [0, 0.15])
        kw  = {k: v for k, v in ann.items()
               if k not in {"x", "y", "text", "arrow", "xytext_offset"}}
        for ax in unique_axs:
            try:
                if ann.get("arrow"):
                    ax.annotate(ann["text"], xy=(ann["x"], ann["y"]),
                                xytext=(ann["x"] + off[0], ann["y"] + off[1]),
                                arrowprops={"arrowstyle": "->"}, **kw)
                else:
                    ax.text(ann["x"], ann["y"], ann["text"], **kw)
            except Exception as e:
                die(f"YAML annotation[{i}]: invalid matplotlib kwargs: {e}")

    for i, region in enumerate(yaml_x.get("fill_between", [])):
        fb_kw = {k: v for k, v in region.items() if k not in {"xmin", "xmax"}}
        for ax in unique_axs:
            try:
                ax.axvspan(region.get("xmin", 0), region.get("xmax", 1), **fb_kw)
            except Exception as e:
                die(f"YAML fill_between[{i}]: invalid matplotlib kwargs: {e}")

    _finalize_figure(fig, args, yaml_x)


def main():
    """
    Orchestrate the full pipeline:
    parse → validate → load → resolve → transform → output/render.
    """
    args, yaml_x = _parse_args()
    _validate_args(args)

    if not args.files and args.file == "-" and sys.stdin.isatty():
        die("no input: provide a CSV file path or pipe data to stdin\n"
            "       run 'csvplot --help' for usage")

    _check_output(args)
    _apply_theme(args, yaml_x)

    frames = _load_frames(args)

    if args.headers:
        _print_headers(frames)
        return
    if args.info:
        _print_info(frames)
        return

    ycols  = _resolve_columns(frames, args)
    frames, x_col = _run_pipeline(frames, args, ycols)

    if _text_output(frames, args, ycols):
        return
    if _render_special(frames, args, ycols, yaml_x):
        return
    _build_figure(frames, args, ycols, x_col, yaml_x)


def _cli_entrypoint():
    """
    Shared entry point for both direct script execution (`python csvplot.py`)
    and the installed console script (`csvplot`, via pyproject.toml's
    [project.scripts]) — both need the same top-level error handling, which
    only runs automatically for the former if it lives in `if __name__ ==
    "__main__"` instead of here.
    """
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except BrokenPipeError:
        # normal when stdout is piped to head/grep and the receiving end closes early
        sys.stderr.close()
        sys.exit(0)
    except Exception:
        # last-resort guard: keep the traceback (needed to debug whatever wasn't
        # caught above) but label it clearly instead of a bare stack dump
        print("csvplot: unexpected internal error, traceback follows:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _cli_entrypoint()
