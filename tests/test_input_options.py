"""
Input decoding tests: --encoding and --sep.

Covers text encodings that are not valid UTF-8, byte-order marks, and the
escape-sequence handling that makes the documented '\\t' TSV invocation use
the fast parser instead of falling back to a regex separator.
"""
import pytest


@pytest.fixture
def csv_latin1(tmp_path):
    """A file that is valid latin-1 but not valid UTF-8."""
    p = tmp_path / "latin1.csv"
    p.write_bytes("label,value\nCaf\u00e9,1\nNa\u00efve,2\n".encode("latin-1"))
    return p


@pytest.fixture
def csv_bom(tmp_path):
    """UTF-8 with a byte-order mark, as written by several spreadsheet tools."""
    p = tmp_path / "bom.csv"
    p.write_bytes("\ufeffa,b\n1,2\n3,4\n".encode("utf-8"))
    return p


# ── --encoding ─────────────────────────────────────────────────────────────

def test_non_utf8_file_reports_actionable_error(run_cli, csv_latin1):
    r = run_cli(str(csv_latin1), "--headers")
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "--encoding" in r.stderr


def test_non_utf8_file_loads_with_explicit_encoding(run_cli, csv_latin1):
    r = run_cli(str(csv_latin1), "--encoding", "latin-1", "--headers")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["label", "value"]


def test_unknown_encoding_fails_cleanly(run_cli, csv_latin1):
    r = run_cli(str(csv_latin1), "--encoding", "not-a-codec", "--headers")
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert "unknown --encoding" in r.stderr


def test_byte_order_mark_is_stripped_from_column_names(run_cli, csv_bom):
    r = run_cli(str(csv_bom), "--headers")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["a", "b"]


def test_byte_order_mark_column_is_selectable(run_cli, csv_bom):
    r = run_cli(str(csv_bom), "--y", "a", "--output", "o.png", "--force")
    assert r.returncode == 0, r.stderr


def test_encoding_applies_to_stdin(run_cli, csv_latin1):
    r = run_cli("-", "--encoding", "latin-1", "--headers",
                stdin=csv_latin1.read_bytes().decode("latin-1"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["label", "value"]


# ── --sep ──────────────────────────────────────────────────────────────────

def test_tab_escape_selects_the_fast_parser(run_cli, tmp_path):
    """
    A literal backslash-t must be translated to a tab. Left untranslated it is
    a two-character separator, which pandas treats as a regex and handles with
    its slower python engine while emitting a ParserWarning.
    """
    p = tmp_path / "t.tsv"
    p.write_text("a\tb\n1\t2\n3\t4\n")
    r = run_cli(str(p), "--sep", "\\t", "--headers")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["a", "b"]
    assert "ParserWarning" not in r.stderr


def test_real_tab_character_still_works(run_cli, tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("a\tb\n1\t2\n")
    r = run_cli(str(p), "--sep", "\t", "--headers")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["a", "b"]


def test_regex_separator_is_left_untranslated(run_cli, tmp_path):
    """Multi-character separators other than the named escapes stay regexes."""
    p = tmp_path / "sp.csv"
    p.write_text("a  b\n1  2\n3  4\n")
    r = run_cli(str(p), "--sep", r"\s+", "--headers")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["a", "b"]


@pytest.mark.parametrize("escape,char", [("\\t", "\t"), ("\\n", "\n"), ("\\r", "\r")])
def test_decode_sep_translates_named_escapes(m, escape, char):
    assert m._decode_sep(escape) == char


@pytest.mark.parametrize("sep", [",", ";", "|", r"\s+", r"\|"])
def test_decode_sep_passes_everything_else_through(m, sep):
    assert m._decode_sep(sep) == sep


# ── size guards ────────────────────────────────────────────────────────────
# The real thresholds are far too large to reach in a test, so the module
# constants are lowered for the duration of each case.

def test_oversized_file_is_refused(m, tmp_path, monkeypatch):
    p = tmp_path / "big.csv"
    p.write_text("a,b\n" + "".join(f"{i},{i}\n" for i in range(500)))
    monkeypatch.setattr(m, "_MAX_FILE_SIZE_MB", 0.0001)
    with pytest.raises(SystemExit):
        m.load_dataframe(str(p), ",", 1000)


def test_size_guard_is_bypassed_for_a_head_preview(m, tmp_path, monkeypatch):
    """--head as the only row-op reads N rows directly, so the guard is skipped."""
    p = tmp_path / "big.csv"
    p.write_text("a,b\n" + "".join(f"{i},{i}\n" for i in range(500)))
    monkeypatch.setattr(m, "_MAX_FILE_SIZE_MB", 0.0001)
    assert len(m.load_dataframe(str(p), ",", 1000, head_hint=5)) == 5


def test_row_limit_is_enforced(m, tmp_path):
    p = tmp_path / "rows.csv"
    p.write_text("a\n" + "".join(f"{i}\n" for i in range(50)))
    with pytest.raises(SystemExit):
        m.load_dataframe(str(p), ",", 10)


def test_column_limit_is_enforced(m, tmp_path, monkeypatch):
    p = tmp_path / "wide.csv"
    p.write_text(",".join(f"c{i}" for i in range(20)) + "\n" +
                 ",".join("1" for _ in range(20)) + "\n")
    monkeypatch.setattr(m, "_MAX_COLS", 5)
    with pytest.raises(SystemExit):
        m.load_dataframe(str(p), ",", 1000)


def test_missing_file_is_reported(m, tmp_path):
    with pytest.raises(SystemExit):
        m.load_dataframe(str(tmp_path / "absent.csv"), ",", 1000)
