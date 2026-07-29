"""
Adversarial and malformed-input tests: duplicate/weird column names,
degenerate data shapes, encoding oddities. Goal is a clean 'csvplot error:'
message or successful output for every case — never a raw traceback.
"""


def clean_outcome(r):
    """Either succeeds, or fails with csvplot's own clean error — never a
    raw Python traceback reaching the user."""
    assert "Traceback" not in r.stderr, f"unhandled exception:\n{r.stderr}"
    if r.returncode != 0:
        assert "csvplot error:" in r.stderr


# ── degenerate data shapes ─────────────────────────────────────────────────

def test_duplicate_column_names(run_cli, tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("a,a,b\n1,2,3\n4,5,6\n")
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_all_nan_column(run_cli, csv_all_nan_col):
    r = run_cli(str(csv_all_nan_col), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_all_nan_column_with_normalize(run_cli, csv_all_nan_col):
    r = run_cli(str(csv_all_nan_col), "--y", "b", "--normalize", "--output", "o.png", "--force")
    clean_outcome(r)


def test_single_row(run_cli, csv_single_row):
    r = run_cli(str(csv_single_row), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_single_row_with_rolling(run_cli, csv_single_row):
    r = run_cli(str(csv_single_row), "--y", "b", "--rolling", "5", "--output", "o.png", "--force")
    clean_outcome(r)


def test_single_row_with_diff(run_cli, csv_single_row):
    r = run_cli(str(csv_single_row), "--y", "b", "--diff", "--output", "o.png", "--force")
    clean_outcome(r)


def test_single_row_scatter_with_fit(run_cli, csv_single_row):
    # linear fit needs >= 2 points; single-row must not crash the fit overlay
    r = run_cli(str(csv_single_row), "--x", "a", "--y", "b", "--scatter", "--fit", "linear",
                "--output", "o.png", "--force")
    clean_outcome(r)


def test_empty_dataset_header_only(run_cli, csv_empty):
    r = run_cli(str(csv_empty), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)
    assert r.returncode == 1
    assert "empty dataset" in r.stderr


def test_constant_column_zscore(run_cli, csv_constant_col):
    r = run_cli(str(csv_constant_col), "--y", "b", "--zscore", "--output", "o.png", "--force")
    clean_outcome(r)


def test_two_row_dataframe(run_cli, tmp_path):
    p = tmp_path / "two.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    r = run_cli(str(p), "--y", "b", "--rolling", "10", "--output", "o.png", "--force")
    clean_outcome(r)


# ── weird column names ──────────────────────────────────────────────────────

def test_column_name_matches_internal_sentinel(run_cli, tmp_path):
    """A user column literally named '_idx' should warn, not crash, when
    the internal sentinel would overwrite it."""
    p = tmp_path / "idx_collision.csv"
    p.write_text("_idx,b\n10,1\n20,2\n30,3\n")
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)
    assert r.returncode == 0
    assert "will be overwritten" in r.stderr


def test_column_name_with_spaces(run_cli, tmp_path):
    p = tmp_path / "spaces.csv"
    p.write_text("col a,col b\n1,2\n3,4\n")
    r = run_cli(str(p), "--y", "col b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_column_name_with_unicode(run_cli, tmp_path):
    p = tmp_path / "unicode.csv"
    p.write_text("température,valeur\n1,2\n3,4\n", encoding="utf-8")
    r = run_cli(str(p), "--y", "valeur", "--output", "o.png", "--force")
    clean_outcome(r)


def test_very_long_column_name(run_cli, tmp_path):
    long_name = "x" * 500
    p = tmp_path / "longname.csv"
    p.write_text(f"{long_name},b\n1,2\n3,4\n")
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_single_column_file(run_cli, tmp_path):
    p = tmp_path / "onecol.csv"
    p.write_text("a\n1\n2\n3\n")
    r = run_cli(str(p), "--y", "a", "--output", "o.png", "--force")
    clean_outcome(r)


# ── encoding / line-ending oddities ─────────────────────────────────────────

def test_windows_line_endings(run_cli, tmp_path):
    p = tmp_path / "crlf.csv"
    p.write_bytes(b"a,b\r\n1,2\r\n3,4\r\n")
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_utf8_bom(run_cli, tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbfa,b\n1,2\n3,4\n")
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_non_utf8_encoding(run_cli, tmp_path):
    """Latin-1 encoded file with a non-ASCII byte — must die cleanly,
    not crash with an unhandled UnicodeDecodeError."""
    p = tmp_path / "latin1.csv"
    p.write_bytes("a,b\n1,café\n3,4\n".encode("latin-1"))
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_trailing_whitespace_in_values(run_cli, tmp_path):
    p = tmp_path / "ws.csv"
    p.write_text("a,b\n 1 , 2 \n3,4\n")
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


# ── mixed-type and malformed numeric columns ────────────────────────────────

def test_mixed_numeric_and_string_column(run_cli, csv_with_strings):
    r = run_cli(str(csv_with_strings), "--y", "b", "--cumsum", "--output", "o.png", "--force")
    clean_outcome(r)
    assert "could not be coerced" in r.stderr


def test_scientific_notation_values(run_cli, tmp_path):
    p = tmp_path / "sci.csv"
    p.write_text("a,b\n1,1e10\n2,-1.5e-5\n3,2.5E3\n")
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_extreme_values_near_overflow(run_cli, tmp_path):
    p = tmp_path / "extreme.csv"
    p.write_text("a,b\n1,1e308\n2,1e309\n3,-1e309\n")  # 1e309 overflows to inf
    r = run_cli(str(p), "--y", "b", "--output", "o.png", "--force")
    clean_outcome(r)


def test_extreme_values_pairplot(run_cli, tmp_path):
    """Regression: even after neutralizing actual inf, a legitimately-finite
    value near the float64 overflow boundary can overflow matplotlib's own
    internal margin/tick math — pairplot must still fail cleanly, not crash."""
    p = tmp_path / "extreme_pp.csv"
    p.write_text("a,b,c\n1,1e308,2\n2,-1e309,4\n3,5,6\n")
    r = run_cli(str(p), "--pairplot", "--output", "o.png", "--force")
    clean_outcome(r)


def test_extreme_values_corr(run_cli, tmp_path):
    p = tmp_path / "extreme_corr.csv"
    p.write_text("a,b,c\n1,1e308,2\n2,-1e309,4\n3,5,6\n")
    r = run_cli(str(p), "--corr", "--output", "o.png", "--force")
    clean_outcome(r)


def test_corr_excludes_idx_sentinel_column(run_cli, csv_numeric):
    """Regression: --corr without explicit --ys used to include the internal
    row-index sentinel as if it were a real data column."""
    r = run_cli(str(csv_numeric), "--corr", "--output", "o.png", "--force")
    clean_outcome(r)
    assert "_idx" not in r.stdout


def test_all_empty_string_values(run_cli, tmp_path):
    p = tmp_path / "blank.csv"
    p.write_text("a,b\n1,\n2,\n3,\n")
    r = run_cli(str(p), "--y", "b", "--stats")
    clean_outcome(r)


# ── column count / size edge cases ─────────────────────────────────────────

def test_many_columns_under_cap(run_cli, tmp_path, m):
    n = 50
    header = ",".join(f"c{i}" for i in range(n))
    row1 = ",".join(str(i) for i in range(n))
    row2 = ",".join(str(i * 2) for i in range(n))
    p = tmp_path / "wide.csv"
    p.write_text(f"{header}\n{row1}\n{row2}\n")
    r = run_cli(str(p), "--y", "c0", "--output", "o.png", "--force")
    clean_outcome(r)


def test_columns_over_cap_dies_cleanly(run_cli, tmp_path, m, monkeypatch):
    # exercise the _MAX_COLS guard without generating a 1000+ column file
    import csvplot
    orig = csvplot._MAX_COLS
    try:
        csvplot._MAX_COLS = 3
        header = "a,b,c,d,e"
        p = tmp_path / "toowide.csv"
        p.write_text(f"{header}\n1,2,3,4,5\n")
        with __import__("pytest").raises(SystemExit):
            csvplot.load_dataframe(str(p), ",", 1000)
    finally:
        csvplot._MAX_COLS = orig


# ── filter / query edge cases on real data ─────────────────────────────────

def test_filter_that_excludes_everything(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--filter", "a > 99999",
                "--output", "o.png", "--force")
    clean_outcome(r)


def test_filter_with_no_matching_rows_then_stats(run_cli, csv_numeric):
    r = run_cli(str(csv_numeric), "--y", "b", "--filter", "a > 99999", "--stats")
    clean_outcome(r)
