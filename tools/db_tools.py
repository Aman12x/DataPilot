"""
tools/db_tools.py — Unified DB layer: DuckDB + Postgres.

All database interaction in DataPilot goes through DBConnection.
No LangGraph or Streamlit imports. Pure Python.
"""

from __future__ import annotations

import json
import os
from typing import Any

import duckdb
import pandas as pd


# Schema comments: human-readable column descriptions injected into inspect_schema() output.
# These are the canonical annotations for the built-in demo dataset.
SCHEMA_COMMENTS: dict[str, dict[str, str]] = {
    "events": {
        "user_id":        "unique user identifier",
        "date":           "event date",
        "platform":       "'android' | 'ios' | 'web'",
        "user_segment":   "'new' | 'returning' | 'power'",
        "is_new_user":    "1 if within first 7 days since install",
        "dau_flag":       "1 if user was active that day",
        "session_count":  "number of sessions that day",
        "notif_received": "push notifications received that day",
        "notif_opened":   "push notifications opened that day",
        "notif_optout":   "1 if user opted out of notifications that day",
        "d7_retained":    "1 if user was active 7 days after first seen",
        "install_date":   "date the user installed the app",
    },
    "funnel": {
        "user_id":    "unique user identifier",
        "date":       "date of funnel step attempt",
        "step":       "'impression' | 'click' | 'install' | 'd1_retain'",
        "completed":  "1 if the funnel step was completed",
    },
    "experiment": {
        "user_id":         "unique user identifier",
        "variant":         "'control' | 'treatment'",
        "assignment_date": "date the user was assigned to a variant",
        "week":            "experiment week number (1 or 2) for novelty detection",
    },
    "metrics_daily": {
        "date":              "calendar date",
        "platform":          "'android' | 'ios' | 'web'",
        "user_segment":      "'new' | 'returning' | 'power'",
        "dau":               "daily active users",
        "new_users":         "users active for the first time within their first 7 days",
        "retained_users":    "active today AND active in the prior 28-day window",
        "resurrected_users": "active today, NOT in prior 28d, but active before that",
        "churned_users":     "active 28 days ago, not active today",
        "d7_retention_rate": "fraction of users still active 7 days after first seen",
        "notif_optout_rate": "fraction of active users who opted out of notifications",
        "avg_session_count": "mean sessions per active user",
    },
}


class DBConnection:
    """
    Unified database interface for DuckDB and Postgres.

    Usage:
        # DuckDB (built-in demo)
        db = DBConnection("duckdb", path="data/dau_experiment.db")

        # Postgres (user-supplied)
        db = DBConnection("postgres", host="localhost", port=5432,
                          dbname="mydb", user="me", password="secret",
                          sslmode="require")
    """

    def __init__(self, backend: str, **kwargs: Any) -> None:
        if backend not in ("duckdb", "postgres"):
            raise ValueError(f"backend must be 'duckdb' or 'postgres', got '{backend}'")

        self.backend = backend
        self._kwargs = kwargs

        if backend == "duckdb":
            path = kwargs.get("path")
            if not path:
                raise ValueError("DuckDB backend requires 'path' kwarg")
            self._path = path

        elif backend == "postgres":
            required = ("host", "port", "dbname", "user", "password")
            missing = [k for k in required if k not in kwargs]
            if missing:
                raise ValueError(f"Postgres backend missing kwargs: {missing}")

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a DataFrame."""
        if self.backend == "duckdb":
            return self._query_duckdb(sql)
        return self._query_postgres(sql)

    def _query_duckdb(self, sql: str) -> pd.DataFrame:
        con = duckdb.connect(self._path, read_only=True)
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    def _query_postgres(self, sql: str) -> pd.DataFrame:
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError("psycopg2 is required for Postgres connections. "
                              "Install it with: pip install psycopg2-binary") from e

        kw = self._kwargs
        conn = psycopg2.connect(
            host=kw["host"],
            port=kw["port"],
            dbname=kw["dbname"],
            user=kw["user"],
            password=kw["password"],
            sslmode=kw.get("sslmode", "prefer"),
        )
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()

    # ── Schema inspection ──────────────────────────────────────────────────────

    def inspect_schema(self, annotation_path: str | None = None) -> str:
        """
        Return a formatted schema string for all tables.

        Format (per Rule 7):
            TABLE: events
              user_id   STRING   -- unique user identifier
              ...

        Inline comments come from SCHEMA_COMMENTS (built-in dataset) or from
        the annotation_path JSON file (external databases).
        """
        annotations = self._load_annotations(annotation_path)

        if self.backend == "duckdb":
            tables = self._get_tables_duckdb()
            lines = []
            for table in tables:
                lines.append(f"TABLE: {table}")
                cols = self._get_columns_duckdb(table)
                for col_name, col_type in cols:
                    comment = annotations.get(table, {}).get(col_name, "")
                    comment_str = f"  -- {comment}" if comment else ""
                    lines.append(f"  {col_name:<22} {col_type:<10}{comment_str}")
                lines.append("")
            return "\n".join(lines).rstrip()

        else:
            tables = self._get_tables_postgres()
            lines = []
            for table in tables:
                lines.append(f"TABLE: {table}")
                cols = self._get_columns_postgres(table)
                for col_name, col_type in cols:
                    comment = annotations.get(table, {}).get(col_name, "")
                    # For unannotated string columns on external DBs, sample values
                    # so the LLM knows valid enum values and doesn't hallucinate them.
                    if not comment and col_type.lower() in self._POSTGRES_STRING_TYPES:
                        samples = self._sample_distinct_values_postgres(table, col_name)
                        if samples:
                            comment = "SAMPLE VALUES: " + " | ".join(f"'{v}'" for v in samples)
                    comment_str = f"  -- {comment}" if comment else ""
                    lines.append(f"  {col_name:<22} {col_type:<10}{comment_str}")
                lines.append("")
            return "\n".join(lines).rstrip()

    def _load_annotations(self, annotation_path: str | None) -> dict:
        if annotation_path and os.path.exists(annotation_path):
            with open(annotation_path) as f:
                return json.load(f)
        if self.backend == "duckdb":
            return SCHEMA_COMMENTS
        return {}

    def _get_tables_duckdb(self) -> list[str]:
        con = duckdb.connect(self._path, read_only=True)
        try:
            result = con.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        finally:
            con.close()

    def _get_columns_duckdb(self, table: str) -> list[tuple[str, str]]:
        con = duckdb.connect(self._path, read_only=True)
        try:
            result = con.execute(f"PRAGMA table_info('{table}')").fetchall()
            # PRAGMA columns: cid, name, type, notnull, dflt_value, pk
            return [(row[1], row[2]) for row in result]
        finally:
            con.close()

    def _get_tables_postgres(self) -> list[str]:
        df = self._query_postgres(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        return df["table_name"].tolist()

    def _get_columns_postgres(self, table: str) -> list[tuple[str, str]]:
        df = self._query_postgres(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema = 'public' AND table_name = '{table}' "
            f"ORDER BY ordinal_position"
        )
        return list(zip(df["column_name"], df["data_type"]))

    # String-ish Postgres types that may contain categorical values worth sampling.
    _POSTGRES_STRING_TYPES = frozenset({
        "text", "varchar", "character varying", "character", "char",
        "bpchar", "name", "citext",
    })

    def _sample_distinct_values_postgres(
        self,
        table: str,
        col: str,
        max_cardinality: int = 50,
        max_show: int = 10,
    ) -> list[str] | None:
        """
        Return up to `max_show` distinct values for a Postgres column if
        its cardinality is <= max_cardinality.  Returns None on failure or
        when the column looks high-cardinality.
        """
        try:
            # Fetch one more than the limit so we know when to give up
            df = self._query_postgres(
                f"SELECT DISTINCT {col}::TEXT AS v "
                f"FROM {table} "
                f"WHERE {col} IS NOT NULL "
                f"ORDER BY 1 LIMIT {max_cardinality + 1}"
            )
            vals = df["v"].dropna().astype(str).tolist()
            if len(vals) <= max_cardinality:
                return vals[:max_show]
            return None      # too many distinct values — skip annotation
        except Exception:
            return None

    # ── Connection test ────────────────────────────────────────────────────────

    def test_connection(self) -> dict:
        """
        Returns: {success: bool, error: str | None, table_count: int}
        Used by UI before saving connection to session state.
        """
        try:
            if self.backend == "duckdb":
                tables = self._get_tables_duckdb()
            else:
                tables = self._get_tables_postgres()
            return {"success": True, "error": None, "table_count": len(tables)}
        except Exception as e:
            return {"success": False, "error": str(e), "table_count": 0}
