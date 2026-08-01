"""
SQL identifier safety for the DuckDB introspection helpers.

`_get_columns_duckdb` and `_table_profile_duckdb` interpolated table and column
names straight into f-string SQL. Column names there come from an uploaded CSV
header, so they are attacker-influenced.

The upload normaliser happens to strip `"` today, which is what kept this from
being live — but db_tools is a generic helper that also runs against DuckDB
files whose table and column names it does not control, and one change to that
normaliser would open it. These tests pin the escaping in place directly rather
than relying on a caller in another module.
"""
import duckdb
import pytest

from tools.db_tools import DBConnection, quote_ident


# ── quote_ident ───────────────────────────────────────────────────────────────


def test_embedded_quotes_are_doubled():
    assert quote_ident('weird"col') == '"weird""col"'


def test_injection_payload_cannot_terminate_the_identifier():
    payload = 'x" FROM t; DROP TABLE t; --'
    quoted = quote_ident(payload)
    # Every inner quote is doubled, so nothing closes the identifier early.
    assert quoted.startswith('"') and quoted.endswith('"')
    assert '""' in quoted
    assert quoted.count('"') % 2 == 0


def test_plain_identifiers_are_just_wrapped():
    assert quote_ident("user_id") == '"user_id"'


def test_accepts_names_a_strict_allowlist_would_reject():
    """These are legitimate outputs of the CSV column normaliser.

    `re.sub(r"[^\\w]", "_", ...)` is Unicode-aware and does not force an
    alphabetic first character, so an ^[a-zA-Z_][a-zA-Z0-9_]*$ allowlist would
    reject real uploads.
    """
    for name in ("2024_revenue", "café", "日本", "_leading"):
        assert quote_ident(name) == f'"{name}"'


def test_rejects_empty_and_nul():
    with pytest.raises(ValueError):
        quote_ident("")
    with pytest.raises(ValueError):
        quote_ident("bad\x00name")


# ── End-to-end against a real DuckDB file ─────────────────────────────────────


@pytest.fixture
def hostile_db(tmp_path):
    """A DuckDB file whose table and column names carry injection payloads."""
    path = str(tmp_path / "hostile.db")
    con = duckdb.connect(path)
    try:
        con.execute(
            'CREATE TABLE "ev""il" ('
            '  "col"" FROM x --" VARCHAR,'
            '  normal INTEGER'
            ')'
        )
        con.execute("""INSERT INTO "ev""il" VALUES ('a', 1), ('b', 2), ('a', 3)""")
        con.execute("CREATE TABLE sentinel (id INTEGER)")
        con.execute("INSERT INTO sentinel VALUES (1)")
    finally:
        con.close()
    return path


def test_columns_are_read_from_a_hostile_table_name(hostile_db):
    db = DBConnection("duckdb", path=hostile_db)
    cols = dict(db._get_columns_duckdb('ev"il'))
    assert 'col" FROM x --' in cols
    assert cols["normal"] == "INTEGER"


def test_profile_handles_hostile_identifiers_without_dropping_the_column(hostile_db):
    """Unquoted interpolation raised, and the bare `except` swallowed it —
    so the column silently vanished from the profile instead of being counted."""
    db = DBConnection("duckdb", path=hostile_db)
    profile = db._table_profile_duckdb('ev"il')

    assert profile is not None
    assert profile["n_rows"] == 3

    hostile_col = profile["columns"]['col" FROM x --']
    assert hostile_col["n_distinct"] == 2
    assert sorted(hostile_col["samples"]) == ["a", "b"]


def test_injected_sql_does_not_execute(hostile_db):
    """The payload must be treated as a name, never as SQL."""
    db = DBConnection("duckdb", path=hostile_db)
    db._table_profile_duckdb('ev"il')

    con = duckdb.connect(hostile_db, read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert "sentinel" in tables


def test_unknown_table_returns_no_columns_rather_than_erroring(hostile_db):
    db = DBConnection("duckdb", path=hostile_db)
    assert db._get_columns_duckdb("no_such_table_'; --") == []


# ── Regression: ordinary schemas still profile correctly ──────────────────────


@pytest.fixture
def plain_db(tmp_path):
    path = str(tmp_path / "plain.db")
    con = duckdb.connect(path)
    try:
        con.execute("CREATE TABLE events (user_id INTEGER, platform VARCHAR)")
        con.execute(
            "INSERT INTO events VALUES (1, 'ios'), (2, 'android'), (3, 'ios')"
        )
    finally:
        con.close()
    return path


def test_plain_schema_columns_and_types(plain_db):
    db = DBConnection("duckdb", path=plain_db)
    assert db._get_columns_duckdb("events") == [
        ("user_id", "INTEGER"),
        ("platform", "VARCHAR"),
    ]


def test_plain_schema_profile(plain_db):
    db = DBConnection("duckdb", path=plain_db)
    profile = db._table_profile_duckdb("events")
    assert profile["n_rows"] == 3
    assert profile["columns"]["platform"]["n_distinct"] == 2
    assert sorted(profile["columns"]["platform"]["samples"]) == ["android", "ios"]
