"""
tests/test_db_tools.py — Unit tests for tools/db_tools.py

Uses the tmp_duckdb fixture (minimal in-memory-equivalent DuckDB).
Does NOT touch the full dau_experiment.db.
"""

import pandas as pd
import pytest

from tools.db_tools import DBConnection


def test_duckdb_connection_succeeds(tmp_duckdb):
    """Connecting to a valid DuckDB file returns success with correct table count."""
    db = DBConnection("duckdb", path=tmp_duckdb)
    result = db.test_connection()
    assert result["success"] is True
    assert result["table_count"] == 1


def test_postgres_connection_fails_gracefully():
    """Bad Postgres credentials return {success: False, error: str} — no exception raised."""
    db = DBConnection("postgres", host="localhost", port=5432,
                      dbname="nonexistent_db", user="nobody", password="wrong")
    result = db.test_connection()
    assert result["success"] is False
    assert isinstance(result["error"], str)
    assert len(result["error"]) > 0


def test_inspect_schema_format(tmp_duckdb):
    """Schema string contains TABLE: markers and -- inline comments."""
    db = DBConnection("duckdb", path=tmp_duckdb)
    schema = db.inspect_schema()
    assert "TABLE:" in schema
    assert "--" in schema          # inline comments from SCHEMA_COMMENTS
    assert "events" in schema      # table name present


def test_query_returns_dataframe(tmp_duckdb):
    """SQL query returns a DataFrame with the expected columns and row count."""
    db = DBConnection("duckdb", path=tmp_duckdb)
    df = db.query("SELECT user_id, dau_flag FROM events ORDER BY user_id")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["user_id", "dau_flag"]
    assert len(df) == 3
