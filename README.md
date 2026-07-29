# csvplot

A single-file, dependency-light command-line tool for turning a CSV into a
publication-ready seaborn plot — without writing a script. Pipe in data, get
a `.png` out, no Jupyter required.

```
csvplot data.csv --x time --y voltage --rolling 10 --output fig.png
```

**Version:** 0.19 · **Requires:** Python 3.9+, seaborn ≥ 0.13, matplotlib ≥ 3.7.3, pandas ≥ 2.1.1, numpy, PyYAML

---

## Table of contents

- [Why csvplot](#why-csvplot)
- [Installation](#installation)
- [Quick start](#quick-start)
- [User guide](#user-guide)
  - [Input](#input)
  - [Selecting columns](#selecting-columns)
  - [Row operations](#row-operations)
  - [Column transforms](#column-transforms)
  - [Grouping and aggregation](#grouping-and-aggregation)
  - [Plot modes](#plot-modes)
  - [Dual y-axis](#dual-y-axis)
  - [Styling](#styling)
  - [Text-output modes](#text-output-modes-no-plot)
  - [Output handling](#output-handling)
  - [Multi-file overlay](#multi-file-overlay)
- [YAML configuration](#yaml-configuration)
  - [Full example](#full-example)
  - [Schema reference](#schema-reference)
  - [Depth/size limits](#depthsize-limits-on-yaml-files)
- [Full CLI reference](#full-cli-reference)
- [Limits and guardrails](#limits-and-guardrails)
- [Known limitations](#known-limitations)
- [For developers](#for-developers)
  - [Architecture](#architecture)
  - [Adding a new plot mode](#adding-a-new-plot-mode)
  - [Adding a new transform](#adding-a-new-transform)
  - [Adding a new YAML option](#adding-a-new-yaml-option)
  - [Testing](#testing)
  - [Conventions](#conventions)
- [Changelog](#changelog)

---

## Why csvplot

Most "just plot this CSV" needs don't justify opening a notebook. csvplot
covers the common cases from the command line: time series with smoothing,
distributions, group comparisons, correlation matrices, regression overlays,
multi-file overlays — with a consistent, themed look and no repeated
boilerplate.

- **One file, no project setup.** Copy `csvplot.py` anywhere Python and the
  four dependencies exist.
- **A full data pipeline before plotting**: row filtering/sampling → column
  transforms (rolling mean, z-score, cumulative sum, …) → grouping/aggregation
  → render. Each stage is optional and composable.
- **10 plot modes** behind one consistent flag set, plus 3 text-output modes
  (`--stats`, `--export-json`, `--percentile`) for when you want numbers, not pixels.
- **YAML config** for anything that gets unwieldy as flags — every CLI option
  is also a YAML key, with a friendlier nested schema on top.
- **Fails loudly and clearly.** Bad input produces a one-line
  `csvplot error: ...` message, not a stack trace.

---

## Installation

**With internet access** — installs csvplot and its dependencies:

```bash
./install.sh
```

`install.sh` checks whether pandas/numpy/matplotlib/seaborn/pyyaml are
already importable (common on clusters with a module system or conda
environment) and only pip-installs what's actually missing, using `--user`
or `--break-system-packages` as your Python environment requires. It then
installs `csvplot` itself locally (`pip install -e .`, no network needed
for that part) so the `csvplot` command is on your `PATH`.

**No internet access** (e.g. an air-gapped cluster node) — if the
dependencies aren't already available via your environment/modules,
download their wheels on a machine that does have access first:

```bash
pip download -d wheelhouse pandas numpy matplotlib seaborn pyyaml
# copy wheelhouse/ alongside csvplot.py, then:
pip install --user --no-index --find-links=wheelhouse pandas numpy matplotlib seaborn pyyaml
./install.sh   # installs just csvplot itself, deps are already satisfied
```

**Without installing at all** — the script still works standalone as long
as the dependencies are importable:

```bash
python3 csvplot.py --help
```

Put it on your `PATH` (e.g. `~/bin/csvplot`) if you'd rather skip `install.sh`
and just run it directly.

---

## Quick start

```bash
# First look at an unfamiliar file: what columns exist, what shape is it
csvplot data.csv --headers
csvplot data.csv --info

# Time series with a rolling average
csvplot data.csv --x time --y voltage --rolling 10

# Three channels, one subplot each, smoothed
csvplot data.csv --x time --ys ch1 ch2 ch3 --subplots --smooth 2.0

# Distribution with histogram + density overlay
csvplot data.csv --y score --hist --kde --bins 30

# Scatter with a linear fit
csvplot data.csv --x weight --y height --scatter --fit linear

# Group means as a bar chart
csvplot data.csv --y value --group-by treatment --agg mean --bar

# Correlation matrix across every numeric column
csvplot data.csv --corr

# Overlay two files on one axis
csvplot --files ctrl.csv trt.csv --x t --y signal --style whitegrid

# Publication export
csvplot data.csv --x time --y amp --style whitegrid --context paper \
  --palette colorblind --dpi 300 --output figure.png
```

Reading from stdin works anywhere a filename would go — the `file` argument
defaults to `-`:

```bash
curl -s https://example.com/data.csv | csvplot --y value --hist
```

---

## User guide

### Input

| Flag | Meaning |
|---|---|
| `file` (positional) | CSV path, or `-`/omitted for stdin |
| `--files F1 F2 ...` | Load several files and overlay them on one plot |
| `--sep` | Delimiter (default `,`; use `--sep '\t'` for TSV) |
| `--max-rows` | Abort if a file exceeds this many rows (default 2,000,000) |
| `--no-parse-units` | Disable automatic unit-suffix parsing |
| `--headers` | Print each file's column names, one per line, and exit — no `--y`/`--ys` needed |
| `--info` | Print each file's shape, columns, dtypes, non-null counts, and an example value, and exit — no `--y`/`--ys` needed |
| `--version` | Print the version and exit |

**Automatic unit parsing:** a string column where values look like `10ms`,
`5kb`, `1.5e3us` is auto-converted to numbers (`0.01`, `5000`, `1.5`) unless
`--no-parse-units` is set. Controlled by `--unit-parse-threshold` (what
fraction of values must parse for the column to convert — default requires
100%) and `--unit-parse-allow-bare-numbers` (allow conversion even when no
value has a recognized suffix, off by default since it's easy to misfire on
ID/version-like columns).

### Selecting columns

```bash
csvplot data.csv --x time --y voltage            # one series
csvplot data.csv --x time --ys ch1 ch2 ch3        # several, overlaid
csvplot data.csv --y voltage --hue device         # colored by category
```

`--x` defaults to the row index when omitted. `--ys` overrides `--y` if both
are given. Most distribution/statistics modes (`--hist`, `--kde`, `--corr`,
`--pairplot`, `--percentile`) work without any `--y`/`--ys` at all — they fall
back to every numeric column in the file.

### Row operations

Applied in this **fixed order**, regardless of flag order on the command line:

```
--head → --tail → --filter → --sort → --sample → --every
```

| Flag | Meaning |
|---|---|
| `--head N` / `--tail N` | Keep first/last N rows |
| `--filter EXPR` | pandas query expression, e.g. `--filter "voltage > 0 and t < 10"` |
| `--sort COL` | Sort ascending by COL |
| `--sample N` / `--seed N` | Random subsample; `--seed` makes it reproducible |
| `--every N` | Keep every Nth row |

`--filter` is evaluated through an AST allow-list (comparisons, boolean/
arithmetic operators, literals) — attribute access, function calls, and
subscripting are rejected before pandas ever sees the expression, so
`--filter "__import__('os').system('x')"` fails validation rather than
running.

### Column transforms

Applied to each `--y`/`--ys` column, in this **fixed, chainable order** — each
step's output feeds the next:

```
cumsum → cummax → cummin → diff → rate → ewm → rolling → smooth → clip → baseline → normalize/zscore
```

| Flag | Meaning |
|---|---|
| `--cumsum` / `--cummax` / `--cummin` | Cumulative sum / running max / running min |
| `--diff` | Row-to-row difference (first value → NaN) |
| `--rate` | Percent change between rows (division-by-zero → NaN, never ±Inf) |
| `--ewm SPAN` | Exponential weighted moving average |
| `--rolling N` | Rolling mean over N rows (`min_periods=1` — no leading NaN) |
| `--smooth SIGMA` | Gaussian kernel smoothing |
| `--clip LO HI` | Clip values to `[LO, HI]` |
| `--baseline` | Subtract the first non-NaN value (series starts at 0) |
| `--normalize` | Min-max scale to `[0, 1]`; a constant column maps to 0 |
| `--zscore` | `(x − mean) / std`; a constant column maps to 0 (mutually exclusive with `--normalize`) |

```bash
# baseline-then-normalize: each series starts at 0 and tops out at 1
csvplot data.csv --x t --y signal --baseline --normalize
```

Non-numeric values in a `--y` column become NaN with a warning rather than
failing the whole plot; the same applies to `±Inf` from numeric overflow
(e.g. a stray `1e309` in the CSV) — both are neutralized before rendering,
since matplotlib can't draw either.

### Grouping and aggregation

```bash
csvplot data.csv --y value --group-by treatment --agg mean --bar
csvplot data.csv --y value --group-by category --agg median --top 5 --bar
```

`--group-by COL` groups rows by `COL`; `--agg {mean,min,max,sum,median}`
picks the aggregation (required together). `--top N` keeps only the N
highest-value groups after aggregation (`--top-by COL` ranks by a different
column than the first `--y`/`--ys`). All four plot categorical modes
(`--box`, `--violin`, `--bar`, `--swarm`) correctly use the group column as
the x-axis grouping — this used to only work with `--bar` (see
[Changelog](#changelog), v0.14).

### Plot modes

One mode flag at a time (`--hist` + `--kde` is the one allowed combination —
histogram with a density curve overlaid). Default is a line plot.

| Flag | Renders | Notes |
|---|---|---|
| *(none)* | Line (`sns.lineplot`) | Default |
| `--scatter` | Scatter (`sns.scatterplot`) | Add `--fit linear` for an OLS overlay |
| `--hist` | Histogram (`sns.histplot`) | `--bins N`, combine with `--kde` |
| `--kde` | Density curve (`sns.kdeplot`) | Usable alone or with `--hist` |
| `--box` | Box plot | `--x` or `--group-by` sets the grouping column |
| `--violin` | Violin plot | Same grouping rules as `--box` |
| `--bar` | Bar chart with error bands | `--ci` controls the band (0 = off) |
| `--swarm` | Every point individually | Best for small N |
| `--regplot` | Scatter + OLS fit + CI band | `--hue` fits one line per group (≤ 20 groups) |
| `--pairplot` | Pairwise scatter matrix | Ignores `--x`; capped at 6 columns (use `--ys` to pick which); with `--files`, only the first file is used |
| `--corr` | Correlation heatmap | `--corr-method {pearson,spearman,kendall}`; with `--files`, only the first file is used |

```bash
csvplot data.csv --y score --box --x condition --hue group
csvplot data.csv --ys a b c --violin                  # compare columns directly
csvplot data.csv --ys c1 c2 c3 c4 --corr --corr-method spearman
csvplot data.csv --ys c1 c2 c3 --pairplot --hue group
```

`--subplots` gives each y-column (explicit or implicit) its own panel
sharing the x-axis, instead of overlaying them.

### Dual y-axis

```bash
csvplot data.csv --y pressure --dual-y temperature --dual-y-label "Temp (°C)"
```

`--dual-y COL [COL ...]` plots column(s) on a second, right-hand axis
(drawn dashed) for a different scale — not supported with `--hist`, `--corr`,
or `--pairplot`, and not combinable with `--group-by`.

### Styling

| Flag | Meaning |
|---|---|
| `--style` | `darkgrid` (default) / `whitegrid` / `dark` / `white` / `ticks` |
| `--context` | `notebook` (default) / `paper` / `talk` / `poster` |
| `--palette` | Any seaborn palette name — `muted`, `deep`, `colorblind`, `tab10`, `husl`, … |
| `--color` | Single color for all series (ignored once `--hue` is active) |
| `--linewidth`, `--marker`, `--alpha` | Self-explanatory; apply globally |
| `--title`, `--xlabel`, `--ylabel` | Text overrides |
| `--hline Y [Y ...]`, `--vline X [X ...]` | Dashed reference lines |
| `--log-x`, `--log-y`, `--grid` | Axis scale / gridlines |
| `--figsize W,H` | Inches (default `10,6`) |
| `--legend` | Force a legend even for a single series |

Per-series colors/linestyles/markers beyond what CLI flags allow — see
[YAML configuration](#yaml-configuration).

### Text-output modes (no plot)

```bash
csvplot data.csv --ys a b c --stats            # describe() table to stdout
csvplot data.csv --ys a b c --export-json       # post-transform data as JSON
csvplot data.csv --ys a b --percentile 25 50 75 # arbitrary percentiles
```

Row operations and column transforms still apply before these compute —
`--stats` after `--rolling 10` describes the smoothed series, not the raw one.

### Output handling

```bash
csvplot data.csv --y v --output fig.png --dpi 300
csvplot data.csv --y v --output fig.pdf            # format from extension
csvplot data.csv --y v --output fig.png --force     # overwrite existing
```

Without `--output`, the figure opens in a matplotlib viewer window instead.
Saves are atomic (written to a temp file, then renamed) so a crash mid-save
never leaves a corrupt output file. `--dpi` is capped at 2000 — anything
higher is almost always a typo, and the combination with a large `--figsize`
can otherwise ask for a multi-gigabyte image buffer.

### Multi-file overlay

```bash
csvplot --files run1.csv run2.csv run3.csv --x t --y signal --hue file
```

Each file becomes its own series (labelled by filename, or full path if
filenames collide). Duplicate paths (even via different relative spellings)
are rejected up front rather than silently double-plotting one file.

---

## YAML configuration

Anything expressible as flags is also expressible as YAML — useful once a
plot's options stop fitting comfortably on one command line, or you want to
version-control a figure's exact settings.

```bash
csvplot data.csv --yaml config.yaml
csvplot data.csv --yaml config.yaml --style dark   # CLI still overrides YAML
```

**Precedence:** CLI flags > YAML values > built-in defaults. You can mix —
keep the bulk of a plot's config in YAML and override one thing (like
`--style`) per invocation.

### Full example

```yaml
data:
  x: time
  y: [ch1, ch2]
  hue: device
  rolling: 10

x_axis:
  label: "Time (s)"

y_axis:
  label: "Amplitude"
  colors: [steelblue, darkorange]
  linestyles: ["-", "--"]
  hline: [0]
  hline_style: {color: gray, linestyle: ":"}

right_axis:
  columns: [temperature]
  label: "Temp (°C)"

plot:
  mode: line
  figsize: [10, 6]

appearance:
  style: whitegrid
  context: paper
  grid:
    alpha: 0.3

title: "Sensor Reading"

annotations:
  - {x: 12.0, y: 0.8, text: "event"}

regions:
  - {xmin: 5, xmax: 8, color: orange, alpha: 0.2}

output:
  dpi: 300
  path: figure.png
```

### Schema reference

The nested schema below maps to internal flag-equivalent keys; a flat
legacy key at the root (e.g. `max_rows: 500000`) also works for anything not
listed, since every CLI flag is configurable via YAML.

| Section | Keys | Maps to |
|---|---|---|
| `data` | `file`, `files`, `x`, `y`/`ys`, `hue`, `sep`, `filter`, `sort`, `head`, `tail`, `sample`, `seed`, `every`, `rolling`, `ewm`, `smooth`, `clip`, `cumsum`/`cummax`/`cummin`/`diff`/`rate`/`baseline`/`normalize`/`zscore`, `group_by`, `agg`, `top`, `top_by`, `percentile` | Same-named CLI flags |
| `x_axis` | `label`, `lim`, `log`, `rotation`, `vline`, `vline_style` | `--xlabel`, `--log-x`, etc. |
| `y_axis` | `label`, `lim`, `log`, `rotation`, `hline`, `hline_style`, `colors`, `linestyles`, `linewidths`, `alphas`, `markers` | `--ylabel`, `--log-y`, per-series style lists (indexed to match `--ys` order) |
| `right_axis` | Same shape as `y_axis`, plus `columns` | `--dual-y` and its styling |
| `plot` | `mode`, `figsize`, plus mode-specific: `bins`/`stat`/`multiple` (hist), `bw_adjust`/`cut` (kde), `inner` (violin), `order`/`robust` (regplot), `s` (scatter), `cmap`/`vmin`/`vmax`/`annot`/`fmt` (corr) | `--hist`/etc. as a boolean flag, `--figsize`, `--bins`, … |
| `appearance` | `style`, `context`, `palette`, `color`, `linewidth`, `marker`, `alpha`, `font_scale`, `title_weight`, `label_size`, `rcParameters`, `grid.*`, `legend.*` | Theme flags; `grid`/`legend` sub-keys imply `--grid`/`--legend` |
| *(root)* | `title`, `annotations`, `regions` | `--title`; post-plot annotations/shaded regions |
| `output` | `path`, `dpi`, `transparent` | `--output`, `--dpi` |

**Annotations** (`{x, y, text}` required; `arrow: true` draws an arrow instead
of plain text) and **regions** (`{xmin, xmax}` required) accept any other key
as a matplotlib keyword — invalid ones fail with a clear per-entry error
rather than a matplotlib traceback.

**Per-series style lists** (`y_axis.colors`, `right_axis.linestyles`, etc.)
are indexed positionally against `--ys`/`--dual-y` — the first color goes
with the first column, and so on; a shorter list just leaves later series on
the automatic palette.

### Depth/size limits on YAML files

A YAML file is capped at 10 MB, 20 levels of nesting, and 1000 entries in
any single list (`annotations`, `fill_between`) — generous for any
hand-written config, and a guard against a malformed file accidentally
producing something huge. The nesting-depth check itself is also bounded
against anchor/alias amplification (`&anchor`/`*alias` reference sharing) by
a total-node-visit budget (see the v0.16 changelog entry) — a small file
using chained anchors can no longer make the depth check hang.

---

## Full CLI reference

Grouped exactly as `csvplot --help` presents them — run that for the
canonical, always-in-sync version; this is the same thing for browsing.

**Input** — `--files`, `--sep`, `--no-parse-units`, `--unit-parse-threshold`,
`--unit-parse-allow-bare-numbers`, `--headers`, `--info`, `--max-rows`, `--yaml`

**Column selection** — `--x`, `--y`, `--ys`, `--hue`, `--dual-y`, `--dual-y-label`

**Output** — `--output`, `--dpi`, `--export-json`, `--force`

**Row operations** *(head → tail → filter → sort → sample → every)* —
`--head`, `--tail`, `--filter`, `--sort`, `--sample`, `--seed`, `--every`, `--stats`

**Group aggregation** — `--group-by`, `--agg`, `--top`, `--top-by`

**Column transforms** *(applied in listed order)* — `--cumsum`, `--cummax`,
`--cummin`, `--diff`, `--rate`, `--ewm`, `--rolling`, `--smooth`, `--clip`,
`--baseline`, `--normalize`, `--zscore`

**Plot mode** *(default: line)* — `--scatter`, `--hist`, `--kde`, `--box`,
`--violin`, `--bar`, `--swarm`, `--regplot`, `--pairplot`, `--corr`, `--corr-method`

**Regression / fit** — `--fit`, `--ci`

**Computed statistics** *(text output, no plot)* — `--percentile`

**Live mode** *(flags parse but are not yet implemented)* — `--watch`,
`--interval`, `--alert`

**Axes / layout** — `--title`, `--xlabel`, `--ylabel`, `--hline`, `--vline`,
`--log-x`, `--log-y`, `--grid`, `--subplots`, `--figsize`, `--legend`, `--bins`

**Seaborn theme** — `--style`, `--context`, `--palette`

**Line / marker style** — `--color`, `--linewidth`, `--marker`, `--alpha`

Every flag has a one-line description in `--help`; the groupings and
descriptions there are the source of truth for exact defaults and choices.

---

## Limits and guardrails

| Guardrail | Default | Purpose |
|---|---|---|
| `--max-rows` | 2,000,000 | Memory guard on load |
| Max columns | 1,000 | Same |
| Max input file size | 500 MB | Same (skipped when `--head` alone can bound the read — see below) |
| Max `--dpi` | 2,000 | Blocks accidental multi-GB image buffers |
| Max YAML file size | 10 MB | |
| Max YAML nesting depth | 20 | |
| Max YAML list length (`annotations`/`fill_between`) | 1,000 | |
| `--regplot --hue` cardinality | 20 groups | Per-group regression above this is almost never intentional |
| `--pairplot` columns | 6 (auto-capped, use `--ys` to choose) | Panel count grows with the square of column count |

`--head N` on its own (no other row-op) reads only the first N rows off
disk, bypassing the file-size/row-count guard entirely — so previewing the
first few rows of a file bigger than `--max-rows` still works. Combine
`--head` with any other row-op (`--tail`, `--filter`, `--sort`, `--sample`,
`--every`) and the full guarded read path is used instead, since those need
to see more than the first N rows to work correctly.

---

## Known limitations

Documented deliberately, so they read as "not yet done" rather than bugs:

- **No native date/time parsing.** `--x` on a timestamp column plots it as a
  categorical/string axis. Works out visually for already-sorted ISO 8601
  timestamps (lexicographic order happens to match chronological order);
  other formats need pre-sorting or a numeric time column.
- **`--watch` and `--alert` are unimplemented.** Both flags parse and are
  documented, but currently just fail with a clear "not yet implemented"
  message — reserved for a live-redraw mode.
- **Annotations/shaded regions apply to every `--subplots` panel** — no
  per-panel targeting yet.
- **`--figsize` height has a floor but no cap when using `--subplots`** —
  many panels get a minimum height so they don't become illegibly cramped,
  but an explicit large `--figsize` is never shrunk.

---

## For developers

### Architecture

Single file, linear pipeline, no classes for the data flow (small `_PlotContext`
dataclass is the one exception — see below):

```
main()
 ├─ build_parser()                    argparse setup, all flags/groups/help text
 ├─ _parse_args()                     two-pass parse: detect --yaml, inject its
 │                                     defaults via set_defaults(), parse again
 ├─ _validate_args() / _validate_extras()   fail fast, before any file I/O
 ├─ _check_output()                   validate --output path early
 ├─ _apply_theme()                    apply seaborn style/context/palette
 ├─ _load_frames()                    read file(s), unit-parsing, dedup check
 ├─ --headers / --info?               _print_headers() / _print_info(), then return
 ├─ _resolve_columns()                validate x/y/hue/group_by/dual_y exist
 ├─ _run_pipeline()                   apply_row_ops → apply_transforms →
 │                                     group/aggregate → resolve x_col
 └─ dispatch to exactly one of:
     ├─ _text_output()                --stats / --export-json / --percentile
     ├─ _render_special()             --corr / --pairplot
     └─ _build_figure()               everything else → _plot_single() per series
```

YAML config loading is its own three-stage pipeline, kept deliberately
separate so each stage has one job (see `_flatten_yaml_schema`'s docstring
for the full nested→flat key mapping):

```
_flatten_yaml_schema()   nested user schema → flat internal keys
_load_yaml_defaults()    type-coerce against argparse's types, split
                         CLI-known keys (defaults) from YAML-only keys (extras)
_validate_extras()       semantic validation of the YAML-only keys
```

### Adding a new plot mode

`_plot_single` dispatches through a `mode -> handler` dict
(`_MODE_HANDLERS`) instead of a growing if/elif chain. Each handler is a
small function taking one `_PlotContext` (bundles `df`, `x_col`, `y_col`,
`ax`, `args`, `hue`, `palette`, `so`, `px`, `line_kw`, `dist_kw` — the common
state every mode needs, so handler signatures don't grow every time a new
piece of shared state is needed):

```python
def _plot_myMode(ctx):
    sns.something(data=ctx.df, x=ctx.x_col, y=ctx.y_col,
                   hue=ctx.hue, palette=ctx.palette, ax=ctx.ax, **ctx.line_kw)

_MODE_HANDLERS["mymode"] = _plot_myMode
```

Then: add the `--my-mode` flag in `build_parser()`'s "Plot mode" group, and
add `"mymode"` in the three places that independently track mode names —
`_active_mode()`'s checked list, the mutual-exclusivity check near the top
of `_validate_args`, and `_MODE_FLAGS` (YAML `plot.mode` validation). These
three aren't unified behind one shared constant yet; missing one of them
won't break anything visibly, it'll just make the new mode behave
inconsistently (e.g. accepted on the CLI but rejected from YAML). If the
mode is categorical (needs `_categorical_x` handling for `--group-by`
support like box/violin/bar/swarm), add it to `_CAT_MODES` too.

### Adding a new transform

`apply_transforms` is one function, one `for c in cols:` loop, transforms
applied as a fixed `if args.x: s = ...; df[c] = s` sequence — each line
reassigns `s` so the next transform in the chain sees the previous one's
output. Add the new `if` in the position that makes sense in the chain
(document it in the "Column transforms" argument group's description if the
overall order changes), and add the flag in `build_parser()`.

### Adding a new YAML option

If it's a new CLI flag: nothing extra needed — any flag not explicitly
handled in `_flatten_yaml_schema` still passes through automatically for a
matching flat root-level key (e.g. `some_new_flag: true` in YAML works the
moment `--some-new-flag` exists as a CLI flag).

If it needs a friendlier nested location (like `plot.some_new_option`
instead of a flat `some_new_option`): add the mapping in
`_flatten_yaml_schema`, add the key to `YAML_EXTRA_KEYS` if it's YAML-only
(no CLI equivalent), and add any semantic validation (ranges, required
sub-keys, allowed values) to `_validate_extras`.

### Testing

```bash
pip install pytest
pytest tests/                 # full suite, ~2 minutes (subprocess-heavy)
pytest tests/test_helpers.py  # fast unit tests only, ~2 seconds
pytest tests/ -k regression    # anything with "regression" in the name
```

| File | Tests | Covers |
|---|---|---|
| `test_helpers.py` | 51 | Pure functions: column resolution, style lookup, unit/figsize parsing, gaussian smoothing, infinity handling |
| `test_validation.py` | 89 | Every `_validate_args`/`_validate_extras` guard |
| `test_transforms.py` | 42 | Each row-op/transform individually, the fixed execution order, and chaining effects |
| `test_yaml_config.py` | 42 | Schema flattening per section, the full load pipeline, CLI/YAML precedence |
| `test_plotting.py` | 30 | Mode dispatch (mocked seaborn calls, asserts exact kwargs), panel-count sizing, dual-y color consistency |
| `test_cli_e2e.py` | 45 | Real subprocess runs — every plot mode, both text-output modes, known-bug repros |
| `test_fuzz.py` | 30 | Adversarial/malformed input: duplicate/unicode/very-long column names, encoding oddities, degenerate data shapes, extreme numeric values |

`conftest.py` holds shared fixtures: a `run_cli` subprocess runner, an `m`
fixture for direct module-level access, and a handful of purpose-built CSV
fixtures (`csv_numeric`, `csv_grouped`, `csv_hue_nan`, etc.) — prefer adding a
small new fixture over inlining a CSV string in every test that needs
similar shape.

Before submitting any change: `python3 -m py_compile csvplot.py`,
`python3 -m pyflakes csvplot.py`, then the full suite. All three are zero
tolerance — a pyflakes warning previously caught a genuinely dead variable
that manual review had missed.

### Conventions

- **Fail with `die(msg)`** (prints `csvplot error: {msg}` to stderr, exits 1)
  for anything the user did wrong. Reserve exceptions for genuinely
  unexpected internal failures — those are caught once, at the top of
  `__main__`, and printed with a labeled traceback rather than a bare one.
- **Warn with `_warn(msg)`** (prints `csvplot warning: {msg}` to stderr,
  keeps going) for silent-data-loss situations: coercion failures, dropped
  NaN rows, a column about to be overwritten.
- **Comment density**: doc-comment on every function's *purpose*; inline
  comments only where a line does something non-obvious (a tricky
  transformation, a why-not-the-obvious-approach note) — not restating what
  the code already says.
- **Versioning**: smallest possible increment per change batch, never
  crossing the next decimal boundary.
- **SLOC**: prefer a tight, well-named one-liner over a verbose multi-line
  equivalent, but not at the cost of a line doing three unrelated things.

---

## Changelog

### v0.19
- Fixed: the three separate mode-name lists (`_MODE_FLAGS`, `_validate_args`'s
  exclusivity check, `_active_mode()`) now derive from one canonical
  `_PLOT_MODE_NAMES` tuple instead of three hand-maintained copies, with
  each list's original reachable-mode set preserved exactly — `pairplot`
  still excluded from `_active_mode()` (`main()` always routes it to
  `_render_special()` first), `corr` still included there (it's exercised
  directly by the test suite independent of `main()`'s dispatch order).
  Also fixed a stale comment on the old `_MODE_FLAGS` definition that
  described unrelated nargs-coercion logic living elsewhere in the file.

### v0.18
- Fixed: `--bar` with no `--x`/`--group-by` plotted one bar per row instead
  of a single aggregate bar — it was the one categorical mode not routed
  through `_categorical_x()`, unlike its box/violin/swarm siblings. Test
  suite updated to match: `bar` folded into the existing parametrized
  box/violin/swarm regression tests instead of a separate one-off case,
  since it now shares their exact behavior (329 tests total).

### v0.17
- Fixed: dependency floor claims (docstring, README badge, `pyproject.toml`)
  understated what's actually needed on Python 3.12 — verified minimum is
  now pandas ≥ 2.1.1, matplotlib ≥ 3.7.3, seaborn ≥ 0.13 (seaborn < 0.13
  crashes on all-NaN y-data, e.g. `--diff` on a single row).
- Fixed: two README sections and a code comment still described the YAML
  anchor/alias risk as an unguarded "accepted gap" — v0.16's
  `_MAX_YAML_NODES` budget already closed the specific failure mode
  (the depth check hanging); docs now reflect that.
- Fixed: "Full CLI reference" Input group was missing `--headers`/`--info`.
- Fixed: stale per-file counts in the Testing table (`test_helpers.py`
  50→51, `test_cli_e2e.py` 39→45); total remains 328.
- Fixed: intro bullet said "2 text-output modes," missing `--percentile`
  (there are 3).
- Fixed: architecture diagram omitted `_check_output()` / `_apply_theme()`
  and the `--headers`/`--info` early-return added in v0.16.

### v0.16
- Fixed: `_yaml_depth()` (the safety check meant to reject oversized YAML
  configs) could itself hang for a very long time on a tiny, innocuous-
  looking file using chained anchors/aliases — YAML aliases share one
  parsed object across every reference rather than copying it, so a
  ~500-byte file could represent billions of logical nodes, and the naive
  recursive depth check re-walked the shared object once per reference.
  Now bounded by a total-node-visit budget, so it fails fast regardless of
  how the amplification is shaped.
- Added `--version`.
- Added `--headers` (print each file's column names) and `--info` (print
  shape, dtypes, non-null counts, and an example value per column) — both
  work without `--y`/`--ys`, as a first look at an unfamiliar file.
- Added `pyproject.toml` and `install.sh` for local, offline-capable
  installation (checks for already-available dependencies before trying
  pip, supports a pre-downloaded wheelhouse for air-gapped installs).
- Refactored the CLI entry point into `_cli_entrypoint()` so both direct
  script execution and the installed console command get the same
  top-level error handling.

### v0.15
- Added the full `tests/` pytest suite (321 tests).
- Refactored `_plot_single`'s mode dispatch from an if/elif chain to a
  `_MODE_HANDLERS` dict of small per-mode functions taking a `_PlotContext`.
- Fixed: extremely large or overflowing numeric values (e.g. `1e309` →
  `inf`) crashed matplotlib's axis-tick locator in line/corr/pairplot
  rendering; now neutralized to NaN with a warning, and any remaining
  render failure produces a clean error instead of a traceback.
- Fixed: `--corr` without explicit `--ys` could include the internal
  row-index column in the correlation matrix.

### v0.14
- Fixed: `--box`/`--violin`/`--swarm` ignored a `--group-by`-derived x
  column (only `--bar` respected it) — the highest-severity finding from
  the v0.13 audit.
- Fixed: `--hist --subplots` with no explicit `--y`/`--ys` silently
  collapsed to a single panel.
- Fixed: an empty `annotations:` YAML key crashed with an unhandled
  `TypeError` instead of a clean error.
- Fixed: `--bins 0` was silently ignored instead of rejected; negative
  `--bins` crashed instead of failing cleanly.
- Added: `--dpi` upper bound (2000), finiteness checks for `--clip`/
  `--hline`/`--vline`.
- Added: a top-level exception handler so any unexpected failure prints a
  labeled traceback instead of a bare one.
- Added: `--head` alone now bypasses the `--max-rows` guard, so previewing
  a large file's first rows works instead of failing the size check.
- Fixed: a dual-y series could get a different default color on each
  `--subplots` panel.
- Fixed: NaN values in a `--regplot --hue` column were silently dropped
  with no warning.

### v0.2 – v0.13
Rolled up rather than itemized per version — these predate this changelog
and weren't individually tracked at the time, so the fixes below are known
to have landed somewhere in this range, not attributable to a specific
version number within it:
- Added: an AST allow-list restricting `--filter` expressions to safe
  comparison/arithmetic operations, rejecting attribute access, function
  calls, and subscripting.
- Changed: YAML loading switched to `yaml.safe_load`.
- Fixed: NumPy 2.x compatibility issues.
- Added: atomic `--output` saves (write to a temp file, then rename).
- Fixed: `pd.NA` values crashing JSON export; sanitized before serialization.
- Refactored: a 466-line `main()` split into the pipeline functions
  described in [Architecture](#architecture).
