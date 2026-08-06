# csvplot

A single-file, dependency-light command-line tool for turning a CSV into a
publication-ready seaborn plot — without writing a script. Pipe in data, get
a `.png` out, no Jupyter required.

```
csvplot data.csv --x time --y voltage --rolling 10 --output fig.png
```

**Version:** 0.26 · **Requires:** Python 3.9+, seaborn ≥ 0.13, matplotlib ≥ 3.7.3, pandas ≥ 2.1.1, numpy, PyYAML

---

## Table of contents

- [Why csvplot](#why-csvplot)
- [Requirements](#requirements)
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
- [Option semantics](#option-semantics)
  - [Execution order](#execution-order)
  - [Row operations in detail](#row-operations-in-detail)
  - [Column transforms in detail](#column-transforms-in-detail)
  - [Choosing a smoothing filter](#choosing-a-smoothing-filter)
  - [Chaining runs](#chaining-runs)
  - [Aggregation and ranking](#aggregation-and-ranking)
  - [Unit-suffixed values](#unit-suffixed-values)
  - [What each plot mode draws](#what-each-plot-mode-draws)
  - [Recipes](#recipes)
- [YAML configuration](#yaml-configuration)
  - [Full example](#full-example)
  - [Schema reference](#schema-reference)
  - [Depth/size limits](#depthsize-limits-on-yaml-files)
- [Full CLI reference](#full-cli-reference)
- [Limits and guardrails](#limits-and-guardrails)
  - [Precedence and overrides](#precedence-and-overrides)
  - [Every warning the tool can emit](#every-warning-the-tool-can-emit)
  - [Reading bounded input](#reading-bounded-input)
- [Startup cost](#startup-cost)
- [Known limitations](#known-limitations)
  - [Assumptions](#assumptions)
  - [Future work](#future-work)
- [For developers](#for-developers)
  - [Project structure](#project-structure)
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
  (`--stats`, `--export-json`, `--export-csv`, `--percentile`) for when numbers are wanted
  rather than pixels.
- **YAML config** for anything that gets unwieldy as flags — every CLI option
  is also a YAML key, with a friendlier nested schema on top.
- **Fails loudly and clearly.** Bad input produces a one-line
  `csvplot error: ...` message, not a stack trace.

---

## Requirements

**Python 3.9 or later.**

### Required

Every feature depends on these; the tool will not start without them.

| Package | Minimum | Needed for |
|---|---|---|
| `pandas` | 2.1.1 | Reading input, every row operation and transform |
| `numpy` | any | Numeric handling and the Gaussian smoothing kernel |
| `matplotlib` | 3.7.3 | All rendering and file output |
| `seaborn` | 0.13 | Every plot mode |

`pip install csvplot` (or `install.sh`) pulls all four in.

### Optional

Two features, and only these two, need something beyond the list above. Both
report the missing package by name and exit cleanly rather than failing part
way through.

| Package | Required by | Install | Without it |
|---|---|---|---|
| `PyYAML` | `--yaml` only | `pip install pyyaml` | `--yaml` reports `--yaml requires PyYAML`. Every other option is unaffected |
| `statsmodels` | YAML `robust: true` under `--regplot` only | `pip install ".[stats]"` | Reported during validation, before any work starts. Ordinary and polynomial regressions do not need it |

`PyYAML` is listed as a declared dependency so a normal install has it
already; it is separated out here because a hand-copied `csvplot.py` on a
machine without it still works for everything except `--yaml`.

### Explicitly not required

- **`scipy`** — seaborn declares it as an optional extra, and none of the
  modes csvplot exposes reach the code paths that use it. `--kde`,
  `--hist --kde`, `--regplot --ci` and `--pairplot` all produce identical
  output with scipy absent.
- **A display.** matplotlib selects a non-interactive backend automatically
  when no display is present, so `--output` works over SSH and in
  containers. Set `MPLBACKEND=Agg` to force it.
- **Network access.** Nothing is fetched at runtime. See
  [Installation](#installation) for an offline install.

### Development

`pytest`, `pytest-cov`, `pytest-xdist` and `pyflakes`, all installed by
`pip install -e ".[dev]"`. See [Testing](#testing).

---

## Installation

**With internet access** — installs csvplot and its dependencies:

```bash
./install.sh
```

`install.sh` checks whether pandas/numpy/matplotlib/seaborn/pyyaml are
already importable (common on clusters with a module system or conda
environment) and only pip-installs what's actually missing, using `--user`
or `--break-system-packages` as the Python environment requires. It then
installs `csvplot` itself locally (`pip install -e .`, no network needed
for that part) so the `csvplot` command lands on `PATH`.

**No internet access** (e.g. an air-gapped cluster node) — if the
dependencies aren't already available via the environment or module system,
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

Placing it on `PATH` (e.g. as `~/bin/csvplot`) skips `install.sh` entirely.

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
| `--encoding` | Input text encoding (default `utf-8-sig`) |
| `--max-rows` | Abort if a file exceeds this many rows (default 2,000,000) |
| `--no-parse-units` | Disable automatic unit-suffix parsing |
| `--headers` | Print each file's column names, one per line, and exit — no `--y`/`--ys` needed |
| `--info` | Print each file's shape, columns, dtypes, non-null counts, and an example value, and exit — no `--y`/`--ys` needed |
| `--version` | Print the version and exit |

**Separators:** the literal escapes `\t`, `\n` and `\r` are translated to the
characters they name, so `--sep '\t'` selects a real tab and keeps the fast
parser. Any other multi-character separator is passed through untouched and
treated as a regular expression, so `--sep '\s+'` still means "one or more
whitespace characters".

**Encodings:** input is read as `utf-8-sig` by default — UTF-8, tolerating an
optional byte-order mark. Files that are not valid UTF-8 report the offending
byte and the flag to reach for; `--encoding latin-1` and `--encoding cp1252`
cover most Western-European exports. The setting applies to stdin as well.

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
`--pairplot`, `--percentile`, `--stats`) work without any `--y`/`--ys` at
all — they fall back to every numeric column in the file.

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

Aggregation keeps only the grouping column and the y column(s); every other
column is gone by the time the plot is drawn. Referring to a dropped column
through `--x` or `--hue` is therefore rejected up front, naming the columns
that survive. `--dual-y` is refused with `--group-by` for the same reason.
Where a second dimension is genuinely needed, aggregate first and plot the
resulting file.

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
csvplot data.csv --ys a b c --export-csv        # post-transform data as CSV
csvplot data.csv --ys a b --percentile 25 50 75 # arbitrary percentiles
```

Row operations and column transforms still apply before these compute —
`--stats` after `--rolling 10` describes the smoothed series, not the raw one.

`--stats` and `--percentile` fall back to every numeric column when no
`--y`/`--ys` is given. Both export modes always emit the whole frame and use
`--y`/`--ys` only to decide which columns the transforms apply to, so they
still require one.

`--export-csv` is written to be read back in: no index column, no internal
columns, and missing values as empty fields. That makes `csvplot ... | csvplot
- ...` a working pipeline — see [Chaining runs](#chaining-runs).

None of these modes render a figure, so `--output` is ignored rather than
validated — an output path left in a YAML config never blocks a quick
`--headers` or `--stats` look at the data. The internal row-index column used
when no `--x` is given is likewise never included in their output.

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

## Option semantics

The User guide above describes what each option is *for*. This section
describes what each one *does* — the exact operation applied, the order it
happens in, and how edge cases resolve. Read it when the difference between
two similar options matters, or when a result is not the one expected.

Every example below transforms the same five-row input so results can be
compared directly:

```
v
1
2
3
4
10
```

### Execution order

The pipeline order is fixed in the code. **The order options are typed on the
command line has no effect.** `--diff --normalize` and `--normalize --diff`
produce byte-identical output, because differencing always runs first.

1. **Load** — read the file, apply `--max-rows`/size guards, parse units.
2. **Row operations**, in this order:
   `--head` → `--tail` → `--filter` → `--sort` → `--sample` → `--every`.
3. **Column transforms**, in this order, applied only to `--y`/`--ys` columns:
   `--cumsum` → `--cummax` → `--cummin` → `--diff` → `--rate` → `--ewm` →
   `--rolling` → `--smooth` → `--clip` → `--baseline` →
   `--normalize` *or* `--zscore`.
4. **Aggregation** — `--group-by` with `--agg`.
5. **Ranking** — `--top` / `--top-by`.
6. **Render.**

Four consequences worth internalising:

- **`--filter` only ever sees raw values.** It runs in stage 2, before any
  transform in stage 3. `--filter 'v > 2' --cumsum` filters the original
  column and *then* accumulates. To filter on a smoothed, differenced or
  normalised value, chain a second run — see [Chaining runs](#chaining-runs).
- **Row operations compose in sequence, not in parallel.**
  `--head 3 --tail 2` yields rows 2–3: the first three rows are taken, then
  the last two *of those three*.
- **Transforms chain.** Each step consumes the previous step's output, so
  `--diff --normalize` normalises the differences, not the original values.
- **With `--group-by`, transforms run inside each group**, before
  aggregation. A `--rolling` mean never averages across a group boundary.

### Row operations in detail

| Option | Exact operation | Notes |
|---|---|---|
| `--head N` | First N rows | Alone with no other row op, reads only N rows off disk and bypasses the size guard |
| `--tail N` | Last N rows | Applied after `--head`, so it sees the already-truncated frame |
| `--filter EXPR` | `DataFrame.query(EXPR)` | Pandas query syntax: `v > 2 and g == "a"`. Column names are bare identifiers; string literals need quotes. Expressions are parsed and rejected before evaluation — no attribute access or function calls |
| `--sort COL` | Ascending sort on COL | Uses the pandas default sort, which is not stable — equal keys may be reordered. `COL` need not be a plotted column. Sorts the frame, so it changes what `--head`/`--tail` would have meant had they run after — but they run before |
| `--sample N` | Random N rows without replacement | Uses `--seed` for reproducibility. `N` larger than the frame takes the whole frame rather than erroring |
| `--every N` | Every Nth row starting at the first | Row indices 0, N, 2N, … after the preceding operations. Positional, not value-based |

The index is reset after any row operation, so downstream positional logic
stays consistent.

### Column transforms in detail

Each transform is applied to every `--y`/`--ys` column independently. Before
any of them run, the column is coerced with `pd.to_numeric(errors="coerce")`:
non-numeric entries become NaN and a warning reports how many. Infinities
become NaN with their own warning, because they crash matplotlib's tick
locator rather than failing cleanly.

| Option | Operation | First value | NaN handling |
|---|---|---|---|
| `--cumsum` | Running total | unchanged | Held in place, skipped in the total — the running sum continues across the gap |
| `--cummax` | Running maximum | unchanged | Held in place, skipped in the comparison |
| `--cummin` | Running minimum | unchanged | Held in place, skipped in the comparison |
| `--diff` | `v[i] - v[i-1]` | NaN | One NaN input produces two NaN outputs: its own position and the next |
| `--rate` | `(v[i] - v[i-1]) / v[i-1]` | NaN | Division by zero yields NaN, not infinity |
| `--ewm N` | Exponentially weighted mean, `span=N` | unchanged | Skipped — gaps are filled |
| `--rolling N` | Trailing mean over N rows | unchanged | Skipped — gaps are filled |
| `--smooth σ` | Gaussian kernel convolution | smoothed | **Preserved** — gaps stay gaps |
| `--clip LO HI` | Bound values to [LO, HI] | clipped | unchanged |
| `--baseline` | Subtract the first non-NaN value | 0 | Leading NaNs stay NaN |
| `--normalize` | `(v - min) / (max - min)` → [0, 1] | scaled | ignored in min/max |
| `--zscore` | `(v - mean) / std`, **sample std (ddof = 1)** | scaled | ignored in mean/std |

`--rate` returns a **fraction, not a percentage**: a doubling is `1.0`, not
`100`. A constant column normalises to `0` rather than NaN, and z-scores to
`0` with a warning, since every value equals the mean. `--normalize` and
`--zscore` are mutually exclusive and rejected together.

With missing data, the three cumulative options and `--baseline` hold the
gap in place rather than filling or propagating it. For input
`NaN, 5, NaN, 2, 9`, `--cumsum` gives `NaN, 5, NaN, 7, 16` and `--baseline`
gives `NaN, 0, NaN, -3, 4` — the first *non-missing* value is the one that
becomes zero.

Applied to the five-row input above:

| Option | Result |
|---|---|
| `(raw input)` | 1, 2, 3, 4, 10 |
| `--cumsum` | 1, 3, 6, 10, 20 |
| `--cummax` | 1, 2, 3, 4, 10 |
| `--cummin` | 1, 1, 1, 1, 1 |
| `--diff` | —, 1, 1, 1, 6 |
| `--rate` | —, 1, 0.5, 0.3333, 1.5 |
| `--ewm 3` | 1, 1.667, 2.429, 3.267, 6.742 |
| `--rolling 3` | 1, 1.5, 2, 3, 5.667 |
| `--smooth 1` | 1.727, 2.148, 3.27, 5.084, 6.269 |
| `--clip 2 4` | 2, 2, 3, 4, 4 |
| `--baseline` | 0, 1, 2, 3, 9 |
| `--normalize` | 0, 0.1111, 0.2222, 0.3333, 1 |
| `--zscore` | -0.8485, -0.5657, -0.2828, 0, 1.697 |

(— is a missing value. Reproduce any row with
`csvplot data.csv --y v --export-json <option>`.)

### Choosing a smoothing filter

Three options smooth a series and they are not interchangeable.

**`--rolling N` — trailing moving average.** Each point is the unweighted
mean of itself and the N-1 rows before it. Because `min_periods=1`, the
first N-1 points are means of short partial windows rather than missing
values — the series never loses length, but its leading points are noisier
than the rest. Being trailing, it **lags**: a step change appears smeared
over the following N rows and never before them. Use it when the lag is
acceptable or wanted, such as a rate that should not appear to react before
its cause.

**`--ewm N` — exponentially weighted mean, `span=N`.** Every past point
contributes, with weight decaying geometrically; the decay constant
α = 2/(N+1). Also trailing, so it also lags, but it responds faster than a
rolling mean of the same N and produces no window edge. Use it for live-ish
series where recent values should dominate.

**`--smooth σ` — Gaussian kernel convolution.** Symmetric and **zero-lag**:
each point is a weighted mean of its neighbours on *both* sides, so features
stay where they are. σ is measured in rows, and the kernel is truncated at a
radius of 3σ (99.7% of the distribution). Edges are handled by reflect
padding, which reduces but does not remove boundary distortion — for a series
shorter than roughly 6σ rows, treat the first and last few points with
suspicion. Use it for post-hoc analysis of a complete series, where symmetry
matters more than causality.

Their treatment of missing data differs and this is often the deciding
factor. For input `1, NaN, 3, 4, 10`:

| Option | Result | Behaviour |
|---|---|---|
| `--rolling 3` | 1, 1, 2, 3.5, 5.667 | Gap **filled** — NaN skipped in the window |
| `--ewm 3` | 1, 1, 2.6, 3.462, 7.069 | Gap **filled** — NaN skipped in the weighting |
| `--smooth 1` | 1.727, —, 3.27, 5.084, 6.269 | Gap **preserved** — interpolated internally so neighbours are not distorted, then restored as missing |

If gaps in the data must remain visibly gaps in the plot, `--smooth` is the
only one of the three that keeps them.

### Chaining runs

Because `--filter` runs before every transform, questions of the form "plot
only the points where the *smoothed* value exceeds X" cannot be answered by a
single pass through the pipeline. `--export-csv` emits the fully transformed
frame in the same format the tool accepts as input, so a second pass can be
fed straight from the first:

```bash
csvplot data.csv --y latency --smooth 3 --export-csv \
  | csvplot - --y latency --filter 'latency > 100' --output spikes.png
```

The `-` argument reads stdin. Each stage runs the whole pipeline, so the
second stage's `--filter` sees the first stage's smoothed values as its raw
input. Stages can be chained indefinitely, and any option is available at any
stage:

```bash
# rank groups by a rolling mean, then keep the top few
csvplot metrics.csv --y throughput --rolling 60 --export-csv \
  | csvplot - --y throughput --group-by host --agg max --top 5 --bar \
      --output slowest.png
```

The same pattern applies to filtering on differences, rates, z-scores or
group aggregates. Missing values are written as empty fields and read back as
missing, so gaps survive a round trip intact.

With several input files the export gains a `source` column naming the file
each row came from, which makes the next stage's `--hue source` a natural
way to keep the series distinct:

```bash
csvplot --files run1.csv run2.csv --y latency --smooth 3 --export-csv \
  | csvplot - --y latency --hue source --output compare.png
```

`--export-json` remains available for feeding other tools; it emits the same
frame keyed by source file. The two export modes are mutually exclusive.

### Aggregation and ranking

`--group-by COL --agg FUNC` collapses the frame to one row per distinct value
of COL, applying FUNC to each `--y` column. Transforms have already run,
per group, by this point.

Aggregation **keeps only the grouping column and the y columns.** Every other
column is discarded, which is why `--x` and `--hue` naming a dropped column
are rejected up front, and why `--dual-y` cannot be combined with
`--group-by` at all.

`--top N` keeps the N largest groups after aggregation, ranked by
`--top-by COL` if given, otherwise by the first `--y` column. It is
`nlargest`, so ties resolve by position and there is no ascending variant —
sort the aggregated output in a second pass if the smallest are wanted.

### Unit-suffixed values

Columns whose values carry a recognised suffix are converted to base units
automatically. `10ms` becomes `0.01`, `5kb` becomes `5000`, `1.5e3ms`
becomes `1.5`.

| Suffix | Multiplier | | Suffix | Multiplier |
|---|---|---|---|---|
| `b` | 1 | | `ns` | 1e-9 |
| `kb` | 1e3 | | `us` | 1e-6 |
| `mb` | 1e6 | | `ms` | 1e-3 |
| `gb` | 1e9 | | `s` | 1 |
| `tb` | 1e12 | | | |

Matching is case-insensitive and accepts a leading sign, a decimal point and
scientific notation. Two guards prevent unwanted conversion:

- **A recognised suffix must appear somewhere in the column.** Otherwise the
  column is returned untouched, so version strings and ID codes like `1.0` or
  `2.1` are never silently coerced. Override with
  `--unit-parse-allow-bare-numbers`.
- **A fraction of values must convert**, set by `--unit-parse-threshold`
  (default `1.0`, meaning all of them). Below the threshold the original
  column is kept unchanged.

An **unrecognised** suffix such as `km` converts to NaN rather than dropping
the unit and keeping the bare number, which would silently produce a value a
thousand times off. Disable the whole mechanism with `--no-parse-units`.

### What each plot mode draws

| Option | Underlying call | Options that affect it |
|---|---|---|
| *(default)* | `lineplot` | `--linewidth`, `--alpha`, `--marker` |
| `--scatter` | `scatterplot` | `--alpha`, YAML `scatter_s` |
| `--regplot` | `regplot` | `--ci` (0 disables the band). Scatter points are drawn at **half** the `--alpha` given, so the fitted line stays readable over dense data. YAML `regplot_order` for polynomial degree, `regplot_robust` (**needs the optional `statsmodels` package** — see [Requirements](#requirements)). With `--hue`, one regression per group, capped at 20 groups |
| `--hist` | `histplot` | `--bins`, `--kde` to overlay a density curve, YAML `hist_stat` (`count`, `frequency`, `density`, `probability`, `percent`), `hist_multiple` (`layer`, `stack`, `dodge`, `fill`) |
| `--kde` | `kdeplot`, filled | YAML `kde_bw_adjust` (bandwidth multiplier), `kde_cut` (how far past the data extremes to draw) |
| `--box` | `boxplot` | x treated as categorical |
| `--violin` | `violinplot` | YAML `violin_inner` (`box`, `quartile`, `point`, `stick`) |
| `--bar` | `barplot` | `--ci` for error bars (0 disables) |
| `--swarm` | `swarmplot` | `--alpha` |
| `--fit linear` | Least-squares line overlaid on any mode | Independent of `--regplot`; draws the fit without replacing the plot |
| `--corr` | `heatmap` of the correlation matrix | Numeric columns only, at least two required. `--corr-method` selects `pearson` (default), `spearman` or `kendall` |
| `--pairplot` | `pairplot` | Capped at 6 columns and a row limit; both sample down with a warning |

`--box`, `--violin`, `--bar` and `--swarm` treat x as a categorical axis. With
several `--ys` and no `--x`, the frame is melted so the y columns themselves
become the categories.

### Recipes

| Goal | Options |
|---|---|
| Smooth a noisy series without shifting features | `--smooth 3` |
| Smooth it the way a monitoring dashboard would | `--rolling 30` |
| Show growth rate rather than absolute value | `--rate` |
| Compare series with different magnitudes on one axis | `--normalize` (shape) or `--zscore` (outliers) |
| Compare two series with incomparable units | `--y a --dual-y b` |
| Show change since the start of the run | `--baseline` |
| Plot a large file quickly | `--every 100` or `--sample 5000 --seed 1` |
| Look at the last portion of a run | `--tail 1000` |
| Suppress outliers without deleting rows | `--clip 0 99` |
| Per-category totals, largest first | `--group-by category --agg sum --top 10 --bar` |
| Per-category distribution rather than a total | `--group-by` omitted, `--x category --box` |
| Correlation between all numeric columns | `--corr` (no `--y` needed); `--corr-method spearman` for rank correlation |
| Check what is in a file before plotting | `--info` |
| Distribution of one column | `--hist --bins 50` or `--kde` |
| Distribution split by a label | `--hist --hue label` |
| Trend line with a confidence band | `--regplot --ci 95` |
| Trend line resistant to outliers | `--regplot` with YAML `robust: true` (needs `statsmodels`, see [Requirements](#requirements)) |
| Same metric across several files | `--files a.csv b.csv --y metric` |
| Numbers instead of a plot | `--stats`, `--percentile 50 95 99`, or `--export-json` |
| Feed one run into another | `--export-csv` piped to `csvplot -` |
| Filter on a *transformed* value | Chain two runs — see [Chaining runs](#chaining-runs) |

---

## YAML configuration

Anything expressible as flags is also expressible as YAML — useful once a
plot's options stop fitting comfortably on one command line, or a figure's
exact settings need to be version-controlled.

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
| `appearance.grid` | `alpha`, `linestyle`, `color` | Grid line styling; any of them turns the grid on |
| `appearance.legend` | `location`, `ncol`, `title`, `fontsize`, `outside` | Legend placement and formatting; any of them turns the legend on. `outside: true` moves the legend clear of the axes |
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


**One naming discrepancy to be aware of:** shaded regions are written under
`regions:` but are called `fill_between` internally, and validation messages
use the internal name — a missing bound is reported as
`YAML: fill_between[0] is missing required key(s): ['xmax']`. The index in
that message is the position in the `regions:` list.

### Depth/size limits on YAML files

A YAML file is capped at 10 MB, 20 levels of nesting, and 1,000 entries in
any single list (`annotations`, `regions`) — generous for any hand-written
config, and a guard against a malformed file accidentally producing
something huge.

The nesting-depth check itself is bounded against anchor/alias amplification
(`&anchor`/`*alias` reference sharing) by a budget of **100,000 node visits**.
A small file using chained anchors can expand to an enormous structure, and
without the budget the depth check alone could run for a very long time.
Exhausting the budget is treated as "too deep" and the file is refused, so
the failure is a clean rejection rather than a hang.

---

## Full CLI reference

Grouped exactly as `csvplot --help` presents them — run that for the
canonical, always-in-sync version; this is the same thing for browsing.

**Input** — `--files`, `--sep`, `--encoding`, `--no-parse-units`, `--unit-parse-threshold`,
`--unit-parse-allow-bare-numbers`, `--headers`, `--info`, `--max-rows`, `--yaml`

**Column selection** — `--x`, `--y`, `--ys`, `--hue`, `--dual-y`, `--dual-y-label`

**Output** — `--output`, `--dpi`, `--export-json`, `--export-csv`, `--force`

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
| Max YAML list length (`annotations`/`regions`) | 1,000 | |
| Max YAML total nodes | 100,000 | Bounds parsing work on a hostile or generated config |
| `--regplot --hue` cardinality | 20 groups | Per-group regression above this is almost never intentional |
| `--pairplot` columns | 6 (auto-capped, use `--ys` to choose) | Panel count grows with the square of column count |
| `--pairplot` rows | 5,000 (sampled above this, with a warning) | Every panel redraws every point |

### Precedence and overrides

Where two options cover the same ground, the more specific one wins and the
other is reported as ignored rather than silently dropped:

| Given both | Winner | Behaviour |
|---|---|---|
| a positional file and `--files` | `--files` | The positional file is ignored, with a warning |
| `--y` and `--ys` | `--ys` | `--y` is ignored, with a warning |
| a CLI flag and the same key in YAML | the CLI flag | No warning; this is the documented precedence |
| `--color` and `--hue` | `--hue` | `--color` has no effect once a hue palette is in play |
| `--normalize` and `--zscore` | neither | Rejected as mutually exclusive |
| `--export-json` and `--export-csv` | neither | Rejected as mutually exclusive |

Two more worth knowing:

- **`--corr` and `--pairplot` use only the first input file.** Given `--files
  a.csv b.csv`, the remaining files are ignored with a warning — neither mode
  has a meaningful multi-file form.
- **A column literally named `_idx` is overwritten.** When no `--x` is given,
  an internal row-index column of that name is added. A real column with that
  name is replaced, with a warning. The internal column never appears in
  `--export-csv`, `--export-json`, `--stats`, `--percentile` or `--corr`.

### Every warning the tool can emit

Warnings go to stderr, are prefixed `csvplot warning:`, and never stop the
run. This list is exhaustive.

| Warning | Trigger |
|---|---|
| ±infinite values became NaN | Numeric overflow in a plotted column (e.g. `1e309`); infinities crash matplotlib's tick locator, so they are removed |
| values could not be coerced to numeric and became NaN | A plotted column contains text |
| all-NaN after transforms; normalize skipped | `--normalize` on a column with nothing left to scale |
| `std=0`; z-score maps to 0 | `--zscore` on a constant column |
| unit parsing failed for column | A unit-suffixed column that could not be converted; the column is left as-is |
| `--log-x` / `--log-y` skipped | The data includes zero or negative values, so a log axis is impossible |
| linear fit skipped | `--fit linear` could not solve, e.g. too few finite points |
| `--hue` has NaN values; those rows are excluded from `--regplot` | Missing values in the hue column; the rows are dropped from the regression only |
| `--file` ignored because `--files` was also specified | See precedence above |
| `--y` ignored because `--ys` was also specified | See precedence above |
| column `_idx` will be overwritten | See precedence above |
| `--corr` / `--pairplot` uses only the first file | Several inputs given to a single-frame mode |
| pairplot with N columns produces N² panels and will be slow | Column count near the cap |
| pairplot on N rows; sampling 5,000 for rendering | Row count above the pairplot cap |
| pairplot KDE diagonal failed; falling back to histogram | The density estimate could not be computed for a diagonal panel |
| correlation matrix is all-NaN | Every column is constant or empty |
| dual-y scale ratio ≈ N× | The two axes differ by more than 1,000×, where a shared plot is easy to misread. Suggests `--normalize` or `--zscore` |
| `tight_layout` failed; output may have clipping | matplotlib could not fit the layout; the file is still written |
| YAML `yaml:` key ignored | A config referencing another config — includes are not recursive |
| unknown YAML key ignored | An unrecognised **top-level** key. See the caveat below |

**One silent behaviour to be aware of:** unknown keys at the top level of a
YAML file are reported, but an unrecognised key *inside* a known section is
dropped without a warning. A misspelled style key such as `bw_adjsut` under
`plot:` therefore has no effect and says nothing. Check
[YAML configuration](#yaml-configuration) for the accepted key names.

### Reading bounded input

`--head N` on its own (no other row-op) reads only the first N rows off
disk, bypassing the file-size/row-count guard entirely — so previewing the
first few rows of a file bigger than `--max-rows` still works. Combine
`--head` with any other row-op (`--tail`, `--filter`, `--sort`, `--sample`,
`--every`) and the full guarded read path is used instead, since those need
to see more than the first N rows to work correctly.

---

## Startup cost

matplotlib and seaborn are imported on first use rather than at start-up.
Together they account for almost all of the time a short invocation spends
before it does anything, and most invocations never draw: `--help`,
`--version`, `--headers`, `--info`, `--stats`, `--export-json`, `--export-csv`,
`--percentile` and every argument-validation failure all finish without a
figure existing.

| invocation | seconds |
|---|---|
| `csvplot --version` | 0.32 |
| `csvplot data.csv --headers` | ~0.35 |
| any run that renders a figure | ~1.7 |

This matters most where the tool is used interactively over a slow or shared
filesystem — a login node, a container image on a network mount — where a
mistyped flag costing four seconds to report an error is the difference
between a usable tool and an irritating one.

The mechanism is a small module-level proxy that imports on first attribute
access, so call sites read as ordinary `sns.lineplot(...)` calls and nothing
in the plotting code has to know about it.

---

## Known limitations

Documented deliberately, so they read as "not yet done" rather than bugs:

- **No native date/time parsing.** `--x` on a timestamp column plots it as a
  categorical/string axis. Works out visually for already-sorted ISO 8601
  timestamps (lexicographic order happens to match chronological order);
  other formats need pre-sorting or a numeric time column.
- **Annotations/shaded regions apply to every `--subplots` panel** — no
  per-panel targeting yet.
- **`--figsize` height has a floor but no cap when using `--subplots`** —
  many panels get a minimum height so they don't become illegibly cramped,
  but an explicit large `--figsize` is never shrunk.

### Assumptions

The tool is built around a few assumptions worth stating, since inputs that
violate them will produce a diagnostic rather than a plot:

- Input is a delimited text file with a single header row. There is no
  multi-row header, sheet-selection or fixed-width support.
- A column is either numeric or it is not. Mixed columns are coerced with
  non-numeric entries becoming missing values, and unit suffixes are stripped
  only when a clear majority of a column's values share a recognised suffix.
- Every input file in a multi-file overlay carries the columns named on the
  command line. A file missing one of them is an error, not a silent skip.
- Reading a file is cheap enough to do in full. There is no streaming or
  chunked mode; the guardrails above bound memory by refusing oversized
  input rather than by processing it incrementally.
- Rendering happens once per invocation and the process then exits. Nothing
  is cached between runs.

### Future work

Candidates, in rough order of how often their absence is felt:

- Native date/time parsing for `--x`, removing the pre-sorting workaround
  described above.
- Per-panel targeting for annotations and shaded regions under `--subplots`.
- Live reload — re-rendering when the input file changes. Flags for this
  existed through v0.20 but only ever reported "not yet implemented", and
  were removed in v0.21 rather than left advertised.
- Streaming/chunked reads, which would turn the file-size guardrail from a
  refusal into a strategy.

---

## For developers

### Project structure

```
csvplot.py                 the tool — parser, pipeline, renderers, entry point
install.sh                 offline-friendly installer (see Installation)
run-tests.sh               test suite in bounded batches
run-coverage.sh            test suite under coverage, including subprocesses
pyproject.toml             packaging metadata, pytest and coverage settings
LICENSE                    GNU General Public License v3.0 or later
README.md                  this document
tests/
  conftest.py              shared fixtures: run_cli, module access, CSV samples
  test_helpers.py          pure helper functions
  test_validation.py       argument and YAML-extras guards
  test_transforms.py       row operations and column transforms
  test_yaml_config.py      YAML schema flattening and load pipeline
  test_plotting.py         plot-mode dispatch and figure layout
  test_decorations.py      axis decoration: reference lines, scales, grid
  test_input_options.py    text encodings, separators, size limits
  test_error_paths.py      failure handlers and hard-to-reach edge paths
  test_cli_e2e.py          full runs as a subprocess
  test_fuzz.py             adversarial and malformed input
.github/workflows/ci.yml   static analysis and test matrix
```

Everything the tool does lives in `csvplot.py`. Keeping it to one file is a
deliberate constraint: the tool is frequently copied to a login node or a
container by itself, and a single file with no local imports always arrives
intact. Splitting it is worth revisiting only if the file stops being
reviewable as a whole.

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
     ├─ _text_output()                --stats / --export-json / --export-csv / --percentile
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
pip install -e ".[dev]"
./run-tests.sh                # everything, in bounded slices
./run-tests.sh --fast         # in-process tests only, a few seconds
pytest -m "not slow"          # the same fast subset, directly
pytest -k regression          # anything with "regression" in the name
./run-coverage.sh             # full suite plus a line/branch coverage report
```

**Where the time goes.** The suite splits in two. 330 tests run in-process
and finish in about ten seconds. The other 145 launch `csvplot.py` as a real
subprocess, and a launch is dominated by imports rather than by the test
itself. Because the plotting stack loads lazily (see
[Startup cost](#startup-cost)), a run that never draws — an argument error,
`--headers`, `--stats` — starts in about a third of a second, while a run
that renders pays for seaborn and matplotlib as well.

No single test is pathological; the cost is proportional to how many
subprocess launches actually draw something, and the full pass still takes
several minutes on one core. Tests are marked `slow` automatically based on
whether they use the `run_cli` fixture — the marker is derived, never applied
by hand, so it cannot drift as tests are added.

`run-tests.sh` runs the in-process subset first, so mistakes surface in
seconds, then works through the subprocess tests in batches (`--batch N`,
default 12). Each batch is its own pytest invocation with a bounded runtime,
which keeps any single call short enough for CI step limits and for an
interactive session; a failing batch does not stop the rest. On a
multi-core machine `pytest-xdist` is used automatically and the total drops
roughly in proportion to core count.

| File | Tests | Covers |
|---|---|---|
| `test_helpers.py` | 67 | Pure functions: column resolution, style lookup, unit/figsize parsing, gaussian smoothing, infinity handling, JSON coercion |
| `test_validation.py` | 109 | Every `_validate_args`/`_validate_extras` guard, plus the `--group-by` column-drop rules |
| `test_transforms.py` | 42 | Each row-op/transform individually, the fixed execution order, and chaining effects |
| `test_yaml_config.py` | 46 | Schema flattening per section, the full load pipeline, CLI/YAML precedence |
| `test_plotting.py` | 39 | Mode dispatch (mocked seaborn calls, asserts exact kwargs), panel-count sizing, dual-y color consistency, linear-fit overlay |
| `test_combinations.py` | 42 | Flag-order independence, every subset of row operations against an independent reference, transform pairs, and invalid values surrounded by valid ones |
| `test_render_matrix.py` | 156 | Every mode against every modifier and against degenerate inputs, drawn with a real backend rather than a mocked one; runs in-process to stay affordable |
| `test_decorations.py` | 18 | Reference lines, log-scale application and refusal, grid, axis limits and tick rotation — asserted against a real Axes |
| `test_input_options.py` | 22 | Text encodings, byte-order marks, separator escape handling, size and shape limits |
| `test_export_csv.py` | 11 | CSV export shape, and two chained processes confirming the output is accepted as input |
| `test_error_paths.py` | 25 | Handlers that fire only when something underneath fails: backend refusals, partial saves, unreadable configs, unwritable destinations, a missing optional dependency, an exception escaping `main()` |
| `test_cli_e2e.py` | 93 | Real subprocess runs — every plot mode, every mode with `--hue`, the text-output modes, known-bug repros |
| `test_fuzz.py` | 30 | Adversarial/malformed input: duplicate/unicode/very-long column names, encoding oddities, degenerate data shapes, extreme numeric values |

`conftest.py` holds shared fixtures: a `run_cli` subprocess runner, an `m`
fixture for direct module-level access, and a handful of purpose-built CSV
fixtures (`csv_numeric`, `csv_grouped`, `csv_hue_nan`, etc.) — prefer adding a
small new fixture over inlining a CSV string in every test that needs
similar shape.

**Coverage.** 700 tests, all passing. Current measured coverage of
`csvplot.py` is **99.5% of statements** (1119 of 1125) and **99.6% of
branches** (518 of 520).

Coverage is not the only measure applied. The mode-dispatch tests assert that
seaborn is *called* correctly, which is not the same as seaborn *accepting*
the call — a gap that once hid a crash on `--hue`. `test_render_matrix.py`
covers acceptance by drawing for real, and the option surface has been swept
exhaustively: every mode against every modifier flag, every YAML style key
against every mode, and every mode against a set of degenerate frames.

Two further measures are tracked deliberately:

- **Every one of the 103 `die()` call sites is executed by the suite.** No
  refusal path is unreached, which is the reason error handling can be
  changed with some confidence.
- **All 64 subsets of the six row operations** are checked against an
  independent reference implementation of the documented order, with the
  flags supplied in reverse of that order so the result must follow the
  pipeline rather than the command line.

Measure it with `./run-coverage.sh`, not a bare `pytest --cov`. Because the
end-to-end tests run the tool in subprocesses, coverage.py must be started
inside each one; the script installs the required `sitecustomize` hook, points
`COVERAGE_FILE` at an absolute path (the subprocesses run with their working
directory set to a temporary directory), then combines the per-process data.
A plain `pytest --cov` run misses every subprocess and reports roughly twenty
points lower.

Conditions that a test process cannot produce on its own are injected at the
boundary instead of being left untested. Filesystem permission failures are
one example: the suite commonly runs as root in containers and CI images,
where `chmod` does not actually deny access, so `os.access` is patched and
the resulting diagnostic is asserted. Terminal detection, an unreadable YAML
config, and a rendering backend that rejects the data are handled the same
way.

What remains uncovered is six statements and two partial branches, all
structurally unreachable rather than untested:

- A `float()` conversion guarded by a regex that has already matched a
  numeric literal, so the conversion cannot fail.
- An `ImportError` fallback for `decimal`, a standard-library module.
- A bare `except` around the dual-y scale-ratio warning, which exists so a
  cosmetic advisory can never abort a plot that would otherwise render.

Each is a deliberate defensive handler kept for safety, not a gap.

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

### v0.26
- Added: a complete list of every warning the tool can emit and what triggers
  it, derived from the code so it is exhaustive.
- Added: a precedence table covering the cases where two options overlap —
  a positional file versus `--files`, `--y` versus `--ys`, CLI versus YAML,
  `--color` versus `--hue`, and the two mutually exclusive pairs.
- Added: two previously undocumented limits — the `--pairplot` row cap of
  5,000 (above which rows are sampled) and the YAML parser's budget of
  100,000 node visits, whose value was previously referenced only by pointing
  at an old changelog entry.
- Added: documentation that `--regplot` draws its scatter points at half the
  `--alpha` given, that `--corr` and `--pairplot` use only the first input
  file, that a column named `_idx` is overwritten by the internal row-index
  column, and that the dual-y scale advisory fires above a 1,000x ratio.
- Added: a note that an unrecognised key *inside* a known YAML section is
  dropped silently, unlike an unrecognised top-level key, which is reported.
- Fixed: the YAML limits text named the internal key `fill_between` where a
  user writes `regions`. The user-facing name is now used, with an explicit
  note that validation messages still report the internal one.

### v0.25
- Added: ordering and combination tests. Command-line flag order is
  documented as irrelevant, which had never been asserted; permutations of
  several flag sets are now compared for identical output. All 64 subsets of
  the six row operations are checked against an independent reference
  implementation of the documented order, with flags supplied in reverse of
  that order. Every ordered pair of transforms is chained, and each invalid
  option value is rejected when surrounded by valid ones.
- Added: documentation of an aliasing property callers depend on —
  `apply_transforms` writes into the frame it is given rather than a copy.

### v0.24
- Fixed: a rendering failure inside the plotting backend could escape as a
  traceback. The five render call sites caught only `ValueError` and
  `OverflowError`, but seaborn raises other types from its own draw code — an
  entirely missing column combined with `--box --hue` surfaced as an
  `UnboundLocalError` from seaborn internals. All five now report any backend
  failure as a named diagnostic including the exception type.
- Added: a render matrix covering every plot mode against every modifier flag
  and against degenerate inputs (single row, constant column, entirely
  missing column, negatives, one distinct x). These render with a real
  backend, unlike the mode-dispatch tests which mock seaborn and so verify
  only that a call is made, not that it is accepted. The matrix runs
  in-process to stay fast: 156 cases in well under a minute.
- Added: a Requirements section stating exactly which packages each feature
  needs, which two features have requirements beyond the core four, and what
  is explicitly *not* required.
- Added: a `stats` extra (`pip install ".[stats]"`) so the optional
  `statsmodels` dependency can be installed by name.

### v0.23
- Added: `--export-csv`, which prints the fully transformed frame as CSV in
  the same format the tool accepts as input. Because `--filter` runs before
  every transform, filtering on a transformed value previously required
  exporting JSON and converting it by hand; runs can now be piped directly,
  `csvplot ... --export-csv | csvplot - ...`. The output carries no index
  column and no internal columns, and writes missing values as empty fields
  so gaps survive the round trip. With several input files a `source` column
  names the origin of each row, which the next stage can use as `--hue`.
  Mutually exclusive with `--export-json`.
- Fixed: `--hue` crashed with an unhandled `TypeError` traceback on both line
  and scatter plots — the two most common modes. seaborn constructs one
  legend entry per hue level and supplies its own label, and the explicit
  label passed alongside `hue` collided with it. The label is now omitted
  entirely when `hue` is in play. Present since at least v0.19; the existing
  tests missed it because the mode-dispatch tests mock seaborn and no
  end-to-end test combined `--hue` with a line or scatter plot. Every mode is
  now rendered with `--hue` as a regression test.
- Fixed: YAML `robust: true` under a regplot raised a bare `RuntimeError`
  from seaborn when the optional `statsmodels` package was absent. It is now
  detected during validation and reported like any other missing
  prerequisite, before any work starts.

### v0.22
- Added: an "Option semantics" section documenting exactly what each option
  does to the data rather than what it is for — the fixed pipeline order and
  why command-line flag order is irrelevant, per-option operations with their
  first-value and missing-data behaviour, a worked example running one input
  series through every transform, a comparison of the three smoothing filters
  covering lag and gap handling, the two-stage pattern needed to filter on a
  transformed value, and a task-to-options recipe table.
- Fixed: documentation stated that `--sort` was stable; it uses the pandas
  default sort, which is not.
- Fixed: documentation described the cumulative transforms as propagating
  missing values. They hold a gap in place and continue across it.
- Fixed: `hist_stat` and `violin_inner` value lists in the documentation
  omitted accepted values and listed one that is not accepted.
- Added: documentation now records that YAML `robust: true` under `--regplot`
  requires the optional `statsmodels` package.

### v0.21
- Added: `--encoding` (default `utf-8-sig`) for inputs that are not valid
  UTF-8. A decoding failure previously surfaced as a raw codec message with
  no way to recover; it now names the offending byte and the flag to use.
  The setting applies to stdin as well, which is read through its raw buffer
  so the encoding is honoured there too.
- Fixed: `--sep '\t'` reached pandas as a two-character separator, which is
  interpreted as a regular expression — the documented TSV invocation fell
  back to the slower python parser engine and emitted a `ParserWarning` on
  every run. The escapes `\t`, `\n` and `\r` are now translated to the
  characters they name. Longer separators are still passed through as
  regexes, so `--sep '\s+'` is unaffected.
- Fixed: combining `--group-by` with a `--hue` or `--x` naming a column that
  aggregation drops produced a seaborn error blaming "extremely large or
  non-finite values", which pointed at the wrong thing entirely. Both are now
  rejected up front, naming the columns that survive aggregation — matching
  the guard `--dual-y` already had.
- Fixed: the internal row-index column added when no `--x` is given leaked
  into `--export-json` output and into `--percentile` when no `--y` was
  given, where it appeared as a real data column.
- Fixed: `--headers`, `--info`, `--stats`, `--export-json` and `--percentile`
  never render a figure, but still had their `--output` path validated and
  would abort if a file was already there. Output validation is now skipped
  for text-only runs.
- Changed: `--stats` no longer requires `--y`/`--ys`, falling back to every
  numeric column like `--percentile` already did.
- Changed: matplotlib and seaborn are now imported on first use instead of at
  start-up. Invocations that never draw a figure — `--help`, `--version`, the
  text-output modes and every argument error — start roughly four times
  faster (1.31s to 0.32s measured on the same machine). Call sites are
  unchanged; a module-level proxy performs the import on first attribute
  access. Theme application moved behind the same check, since it is a
  seaborn call and has no bearing on text output.
- Removed: `--watch`, `--alert` and `--interval`. They were advertised in
  `--help` but exited with "not yet implemented" on use. Removing them is
  the honest state of the tool; live-reload remains a candidate for a future
  release rather than a half-present feature.
- Added: test coverage for axis decoration (reference lines, log scaling,
  grid, limits, tick rotation), which had no render-level tests at all, and
  for encoding and separator handling, and for the failure handlers that only
  fire when something underneath the tool breaks. 481 tests total, 99.5%
  statement and 99.6% branch coverage.
- Added: `run-tests.sh`, which runs the in-process tests first and then the
  subprocess tests in bounded batches, and `run-coverage.sh`, which measures
  the subprocess-based tests that a plain `pytest --cov` run silently misses.
  Tests that launch a subprocess are marked `slow` automatically, derived
  from fixture use, so `pytest -m "not slow"` gives a five-second feedback
  loop over roughly two thirds of the suite.
- Added: a CI workflow running static analysis and the suite across Python
  3.9, 3.11 and 3.13.
- Changed: three redundant tests were removed and five `_validate_extras`
  cases moved to sit with their peers. Two duplicated existing coverage
  exactly and a third was fully contained by a newer parametrised test; the
  misplaced cases were what allowed the duplication to go unnoticed.
- Changed: packaging metadata now carries the license, readme, project URLs
  and classifiers; `csvplot.py`, `install.sh` and `run-coverage.sh` are
  committed with their executable bit set.

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
