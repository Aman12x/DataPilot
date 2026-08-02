"""
tests/test_mysql_metadata_binding.py — schema/table names are values, not code.

The MySQL metadata queries interpolated the database and table name straight
into a SQL string literal and guarded it with `_SAFE_IDENT_RE`
(`^[a-zA-Z_][a-zA-Z0-9_]*$`). The Postgres path right above them already bound
the same values with `%s`.

Two problems with the old shape, and the second is the one that actually bit:

  * an allowlist is the weaker mechanism — it only holds while the regex is
    right, whereas a bound parameter is never parsed as SQL at all;
  * it rejects legitimate names. `café`, `日本`, and `2024_revenue` are all valid
    MySQL identifiers, and CSV-derived table names routinely look like the last
    one. Schema introspection simply raised on those databases.

These tests capture the SQL and params handed to the driver rather than talking
to a real MySQL — the question is what gets sent, and a live server would only
add a dependency without answering it any better.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tools.db_tools import DBConnection


@pytest.fixture
def mysql_db(monkeypatch):
    """A MySQL DBConnection whose queries are captured instead of executed."""
    db = DBConnection(
        "mysql",
        host="db.example.com",
        port=3306,
        dbname="analytics",
        user="svc",
        password="pw",
    )
    calls: list[tuple[str, tuple | None]] = []

    def _capture(sql, params=None):
        calls.append((sql, params))
        return pd.DataFrame({"table_name": [], "column_name": [], "data_type": []})

    monkeypatch.setattr(db, "_query_mysql", _capture)
    db.captured = calls  # type: ignore[attr-defined]
    return db


def test_table_listing_binds_the_schema_name(mysql_db):
    mysql_db._get_tables_mysql()
    sql, params = mysql_db.captured[0]
    assert "%s" in sql
    assert params == ("analytics",)
    assert "'analytics'" not in sql, "schema name was interpolated, not bound"


def test_column_listing_binds_both_values(mysql_db):
    mysql_db._get_columns_mysql("orders")
    sql, params = mysql_db.captured[0]
    assert sql.count("%s") == 2
    assert params == ("analytics", "orders")
    assert "'orders'" not in sql


@pytest.mark.parametrize("name", [
    "2024_revenue",          # leading digit — CSV-derived names look like this
    "café",                  # non-ASCII, and legal in MySQL
    "日本",
    "orders' OR '1'='1",     # the case the allowlist existed for
    "orders`; DROP TABLE x;--",
])
def test_names_the_allowlist_rejected_are_now_bound(mysql_db, name):
    """The first three used to raise; the last two are why binding is the fix."""
    mysql_db._get_columns_mysql(name)
    sql, params = mysql_db.captured[0]
    assert params == ("analytics", name)
    # Whatever the name contains, none of it reaches the statement text.
    assert name not in sql


def test_hostile_schema_name_never_reaches_the_statement(monkeypatch):
    db = DBConnection(
        "mysql",
        host="db.example.com", port=3306,
        dbname="analytics'; DROP TABLE users; --",
        user="svc", password="pw",
    )
    calls: list[tuple[str, tuple | None]] = []
    monkeypatch.setattr(
        db, "_query_mysql",
        lambda sql, params=None: (calls.append((sql, params)), pd.DataFrame({"table_name": []}))[1],
    )
    db._get_tables_mysql()
    sql, params = calls[0]
    assert "DROP TABLE" not in sql
    assert params == ("analytics'; DROP TABLE users; --",)


def test_query_mysql_forwards_params_to_the_driver(monkeypatch):
    """Binding at the call site is only real if the driver actually gets them."""
    seen: dict = {}

    class _Conn:
        def close(self):
            pass

    monkeypatch.setitem(
        __import__("sys").modules, "pymysql",
        type("M", (), {
            "connect": staticmethod(lambda **kw: _Conn()),
            "cursors": type("C", (), {"Cursor": object}),
        }),
    )
    monkeypatch.setattr(
        pd, "read_sql",
        lambda sql, conn, params=None: seen.update(sql=sql, params=params) or pd.DataFrame(),
    )

    db = DBConnection(
        "mysql", host="h", port=3306, dbname="analytics", user="u", password="p"
    )
    db._query_mysql("SELECT 1 WHERE x = %s", params=("value",))
    assert seen["params"] == ("value",)
