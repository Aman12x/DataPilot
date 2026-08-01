"""
tools/db_tools.py — Unified DB layer: DuckDB + Postgres + MySQL + BigQuery.

All database interaction in DataPilot goes through DBConnection.
No LangGraph or Streamlit imports. Pure Python.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import duckdb
import pandas as pd

# Compiled once at import time — blocks any LLM-generated mutation statement.
# DuckDB already enforces read_only=True at the driver level; this adds a
# defence-in-depth check that also covers external SQL engines.
_MUTATION_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|MERGE|COPY|"
    r"GRANT|REVOKE|CALL|DO|ATTACH|DETACH|EXPORT|LOAD)\b",
    re.IGNORECASE,
)

_SUPPORTED_BACKENDS = frozenset({"duckdb", "postgres", "mysql", "bigquery"})
_DIALECT_LABELS = {
    "duckdb": "DuckDB SQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "bigquery": "BigQuery Standard SQL",
}

# DuckDB table functions that can read arbitrary host files.
_FILE_READ_RE = re.compile(
    r"\b(read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|glob|read_blob|read_text)\s*\(",
    re.IGNORECASE,
)

_MAX_SQL_LIMIT = int(os.getenv("SQL_MAX_ROWS", "50000"))


def _strip_leading_sql_comments(sql: str) -> str:
    """Remove leading line/block comments before checking the first SQL token."""
    rest = sql.lstrip()
    while True:
        if rest.startswith("--"):
            _, sep, tail = rest.partition("\n")
            rest = tail.lstrip() if sep else ""
            continue
        if rest.startswith("/*"):
            end = rest.find("*/", 2)
            if end == -1:
                return ""
            rest = rest[end + 2:].lstrip()
            continue
        return rest


def validate_sql(sql: str) -> None:
    """
    Defence-in-depth checks before executing analyst/LLM SQL.
    Raises ValueError when the statement looks unsafe or out of scope.
    """
    stripped = sql.strip()
    if not stripped:
        raise ValueError("Empty SQL statement")

    if ";" in stripped.rstrip(";"):
        raise ValueError("Multi-statement SQL is not permitted")

    first_sql = _strip_leading_sql_comments(stripped)
    upper = first_sql.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT/WITH queries are permitted")

    if _MUTATION_RE.search(stripped):
        raise ValueError("Mutation or privileged SQL is not permitted")

    if _FILE_READ_RE.search(stripped):
        raise ValueError("File-read SQL functions are not permitted")


def _ensure_limit(sql: str) -> str:
    """Append LIMIT when missing (after validate_sql would have rejected — safety net)."""
    if re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip()} LIMIT {_MAX_SQL_LIMIT}"


_SAFE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _quote_ident(name: str, backend: str = "postgres") -> str:
    if not _SAFE_IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    if backend == "mysql":
        return f"`{name}`"
    if backend == "bigquery":
        return f"`{name}`"
    return f'"{name}"'


def dialect_label(backend: str) -> str:
    return _DIALECT_LABELS.get(backend, "SQL")


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
    Unified database interface for DuckDB, Postgres, MySQL, and BigQuery.

    Usage:
        db = DBConnection("duckdb", path="data/dau_experiment.db")
        db = DBConnection("postgres", host=..., port=5432, dbname=..., user=..., password=...)
        db = DBConnection("mysql", host=..., port=3306, dbname=..., user=..., password=...)
        db = DBConnection("bigquery", project_id=..., dataset=..., credentials_json=...)
    """

    def __init__(self, backend: str, **kwargs: Any) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(_SUPPORTED_BACKENDS)}, got '{backend}'"
            )

        self.backend = backend
        self._kwargs = kwargs

        if backend == "duckdb":
            path = kwargs.get("path")
            if not path:
                raise ValueError("DuckDB backend requires 'path' kwarg")
            self._path = path

        elif backend in ("postgres", "mysql"):
            required = ("host", "port", "dbname", "user", "password")
            missing = [k for k in required if k not in kwargs]
            if missing:
                raise ValueError(f"{backend} backend missing kwargs: {missing}")

        elif backend == "bigquery":
            required = ("project_id", "dataset")
            missing = [k for k in required if not str(kwargs.get(k) or "").strip()]
            if missing:
                raise ValueError(f"BigQuery backend missing kwargs: {missing}")
            if not (kwargs.get("credentials_json") or kwargs.get("credentials_path")):
                raise ValueError(
                    "BigQuery backend requires credentials_json or credentials_path"
                )

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a DataFrame. Only SELECT is permitted."""
        validate_sql(sql)
        sql = _ensure_limit(sql)
        if self.backend == "duckdb":
            return self._query_duckdb(sql)
        if self.backend == "postgres":
            return self._query_postgres(sql)
        if self.backend == "mysql":
            return self._query_mysql(sql)
        return self._query_bigquery(sql)

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

    def _query_mysql(self, sql: str) -> pd.DataFrame:
        try:
            import pymysql
        except ImportError as e:
            raise ImportError(
                "pymysql is required for MySQL connections. "
                "Install it with: pip install pymysql"
            ) from e

        kw = self._kwargs
        sslmode = (kw.get("sslmode") or "prefer").lower()
        ssl_kwargs: dict[str, Any] | None = None
        if sslmode in ("require", "verify-ca", "verify-full"):
            ssl_kwargs = {}

        conn = pymysql.connect(
            host=kw["host"],
            port=int(kw["port"]),
            database=kw["dbname"],
            user=kw["user"],
            password=kw["password"],
            ssl=ssl_kwargs,
            cursorclass=pymysql.cursors.Cursor,
        )
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()

    def _bq_client(self):
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError as e:
            raise ImportError(
                "google-cloud-bigquery is required for BigQuery connections. "
                "Install it with: pip install google-cloud-bigquery db-dtypes"
            ) from e

        kw = self._kwargs
        project_id = kw["project_id"]
        creds = None
        creds_json = kw.get("credentials_json") or ""
        creds_path = kw.get("credentials_path") or ""
        if creds_json:
            info = json.loads(creds_json) if isinstance(creds_json, str) else creds_json
            creds = service_account.Credentials.from_service_account_info(info)
        elif creds_path:
            creds = service_account.Credentials.from_service_account_file(creds_path)
        return bigquery.Client(project=project_id, credentials=creds)

    def _query_bigquery(self, sql: str) -> pd.DataFrame:
        client = self._bq_client()
        job = client.query(sql)
        return job.to_dataframe(create_bqstorage_client=False)

    # ── Schema inspection ──────────────────────────────────────────────────────

    def inspect_schema(
        self,
        annotation_path: str | None = None,
        annotations: dict | None = None,
    ) -> str:
        """
        Return a formatted schema string for all tables.

        Format (per Rule 7):
            TABLE: events
              user_id   STRING   -- unique user identifier
              ...

        Inline comments come from (in priority order):
          1. `annotations` dict passed in-memory (saved connection annotations)
          2. `annotation_path` JSON file
          3. SCHEMA_COMMENTS for the built-in DuckDB demo
        """
        loaded = self._load_annotations(annotation_path, annotations=annotations)

        if self.backend == "duckdb":
            tables = self._get_tables_duckdb()
            lines = []
            for table in tables:
                profile = self._table_profile_duckdb(table)
                row_note = f"  -- {profile['n_rows']:,} rows" if profile else ""
                lines.append(f"TABLE: {table}{row_note}")
                cols = self._get_columns_duckdb(table)
                for col_name, col_type in cols:
                    comment = loaded.get(table, {}).get(col_name, "")
                    if not comment and profile:
                        col_info = profile["columns"].get(col_name, {})
                        parts = []
                        n_distinct = col_info.get("n_distinct")
                        if n_distinct is not None:
                            parts.append(f"{n_distinct:,} distinct")
                        samples = col_info.get("samples")
                        if samples:
                            parts.append("e.g. " + ", ".join(f"'{v}'" for v in samples))
                        comment = "  ".join(parts)
                    comment_str = f"  -- {comment}" if comment else ""
                    lines.append(f"  {col_name:<22} {col_type:<10}{comment_str}")
                lines.append("")
            return "\n".join(lines).rstrip()

        tables = self._get_tables()
        string_types = self._string_types()
        lines = []
        for table in tables:
            lines.append(f"TABLE: {table}")
            cols = self._get_columns(table)
            for col_name, col_type in cols:
                comment = loaded.get(table, {}).get(col_name, "")
                # For unannotated string columns on external DBs, sample values
                # so the LLM knows valid enum values and doesn't hallucinate them.
                if not comment and col_type.lower() in string_types:
                    samples = self._sample_distinct_values(table, col_name)
                    if samples:
                        comment = "SAMPLE VALUES: " + " | ".join(f"'{v}'" for v in samples)
                comment_str = f"  -- {comment}" if comment else ""
                lines.append(f"  {col_name:<22} {col_type:<10}{comment_str}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _load_annotations(
        self,
        annotation_path: str | None,
        annotations: dict | None = None,
    ) -> dict:
        if annotations:
            return annotations
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

    # Number of sample values to show per column in schema context.
    _PROFILE_MAX_SAMPLES = 5
    # Only show samples for columns with at most this many distinct values.
    _PROFILE_SAMPLE_CARDINALITY = 50

    def _table_profile_duckdb(self, table: str) -> dict | None:
        """
        Return a lightweight data profile for a DuckDB table:
          { n_rows: int, columns: { col: { n_distinct: int, samples: list[str] } } }

        Distinct counts and sample values are only collected for columns with
        <= _PROFILE_SAMPLE_CARDINALITY distinct values (categoricals, date cols, etc.)
        so the schema comment stays readable. High-cardinality columns (IDs, free text)
        only show the distinct count.

        Returns None on any error so schema inspection never fails.
        """
        try:
            con = duckdb.connect(self._path, read_only=True)
            try:
                cols = self._get_columns_duckdb(table)
                n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # type: ignore[index]
                col_profiles: dict[str, dict] = {}
                for col_name, _ in cols:
                    try:
                        n_distinct = con.execute(
                            f"SELECT COUNT(DISTINCT {col_name}) FROM {table}"
                        ).fetchone()[0]  # type: ignore[index]
                        samples: list[str] = []
                        if n_distinct <= self._PROFILE_SAMPLE_CARDINALITY:
                            rows = con.execute(
                                f"SELECT DISTINCT CAST({col_name} AS VARCHAR) "
                                f"FROM {table} WHERE {col_name} IS NOT NULL "
                                f"ORDER BY 1 LIMIT {self._PROFILE_MAX_SAMPLES}"
                            ).fetchall()
                            samples = [r[0] for r in rows if r[0] is not None]
                        col_profiles[col_name] = {"n_distinct": n_distinct, "samples": samples}
                    except Exception:
                        pass
                return {"n_rows": n_rows, "columns": col_profiles}
            finally:
                con.close()
        except Exception:
            return None

    def _get_tables(self) -> list[str]:
        if self.backend == "postgres":
            return self._get_tables_postgres()
        if self.backend == "mysql":
            return self._get_tables_mysql()
        if self.backend == "bigquery":
            return self._get_tables_bigquery()
        return self._get_tables_duckdb()

    def _get_columns(self, table: str) -> list[tuple[str, str]]:
        if self.backend == "postgres":
            return self._get_columns_postgres(table)
        if self.backend == "mysql":
            return self._get_columns_mysql(table)
        if self.backend == "bigquery":
            return self._get_columns_bigquery(table)
        return self._get_columns_duckdb(table)

    def _string_types(self) -> frozenset[str]:
        if self.backend == "mysql":
            return self._MYSQL_STRING_TYPES
        if self.backend == "bigquery":
            return self._BQ_STRING_TYPES
        return self._POSTGRES_STRING_TYPES

    def _get_tables_postgres(self) -> list[str]:
        df = self._query_postgres(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        return df["table_name"].tolist()

    def _get_columns_postgres(self, table: str) -> list[tuple[str, str]]:
        if not _SAFE_IDENT_RE.match(table):
            raise ValueError(f"Unsafe table name: {table!r}")
        df = self._query_postgres(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema = 'public' AND table_name = '{table}' "
            "ORDER BY ordinal_position"
        )
        return list(zip(df["column_name"], df["data_type"]))

    def _get_tables_mysql(self) -> list[str]:
        dbname = self._kwargs["dbname"]
        if not _SAFE_IDENT_RE.match(dbname):
            raise ValueError(f"Unsafe database name: {dbname!r}")
        df = self._query_mysql(
            "SELECT table_name AS table_name FROM information_schema.tables "
            f"WHERE table_schema = '{dbname}' ORDER BY table_name"
        )
        return df["table_name"].tolist()

    def _get_columns_mysql(self, table: str) -> list[tuple[str, str]]:
        if not _SAFE_IDENT_RE.match(table):
            raise ValueError(f"Unsafe table name: {table!r}")
        dbname = self._kwargs["dbname"]
        if not _SAFE_IDENT_RE.match(dbname):
            raise ValueError(f"Unsafe database name: {dbname!r}")
        df = self._query_mysql(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema = '{dbname}' AND table_name = '{table}' "
            "ORDER BY ordinal_position"
        )
        return list(zip(df["column_name"], df["data_type"]))

    def _get_tables_bigquery(self) -> list[str]:
        client = self._bq_client()
        dataset = self._kwargs["dataset"]
        return [t.table_id for t in client.list_tables(dataset)]

    def _get_columns_bigquery(self, table: str) -> list[tuple[str, str]]:
        if not _SAFE_IDENT_RE.match(table):
            raise ValueError(f"Unsafe table name: {table!r}")
        client = self._bq_client()
        project = self._kwargs["project_id"]
        dataset = self._kwargs["dataset"]
        full = f"{project}.{dataset}.{table}"
        meta = client.get_table(full)
        return [(f.name, f.field_type) for f in meta.schema]

    # String-ish types that may contain categorical values worth sampling.
    _POSTGRES_STRING_TYPES = frozenset({
        "text", "varchar", "character varying", "character", "char",
        "bpchar", "name", "citext",
    })
    _MYSQL_STRING_TYPES = frozenset({
        "char", "varchar", "tinytext", "text", "mediumtext", "longtext",
        "enum", "set",
    })
    _BQ_STRING_TYPES = frozenset({"STRING", "string", "BYTES", "bytes"})

    def _sample_distinct_values(
        self,
        table: str,
        col: str,
        max_cardinality: int = 50,
        max_show: int = 10,
    ) -> list[str] | None:
        """
        Return up to `max_show` distinct values if cardinality is low.
        Returns None on failure or high-cardinality columns.
        """
        try:
            safe_table = _quote_ident(table, self.backend)
            safe_col = _quote_ident(col, self.backend)
            if self.backend == "postgres":
                sql = (
                    f"SELECT DISTINCT {safe_col}::TEXT AS v "
                    f"FROM {safe_table} "
                    f"WHERE {safe_col} IS NOT NULL "
                    f"ORDER BY 1 LIMIT {max_cardinality + 1}"
                )
                df = self._query_postgres(sql)
            elif self.backend == "mysql":
                sql = (
                    f"SELECT DISTINCT CAST({safe_col} AS CHAR) AS v "
                    f"FROM {safe_table} "
                    f"WHERE {safe_col} IS NOT NULL "
                    f"ORDER BY 1 LIMIT {max_cardinality + 1}"
                )
                df = self._query_mysql(sql)
            elif self.backend == "bigquery":
                project = self._kwargs["project_id"]
                dataset = self._kwargs["dataset"]
                if not (_SAFE_IDENT_RE.match(project) and _SAFE_IDENT_RE.match(dataset)):
                    return None
                sql = (
                    f"SELECT DISTINCT CAST({safe_col} AS STRING) AS v "
                    f"FROM `{project}.{dataset}.{table}` "
                    f"WHERE {safe_col} IS NOT NULL "
                    f"ORDER BY 1 LIMIT {max_cardinality + 1}"
                )
                df = self._query_bigquery(sql)
            else:
                return None
            vals = df["v"].dropna().astype(str).tolist()
            if len(vals) <= max_cardinality:
                return vals[:max_show]
            return None
        except Exception:
            return None

    # Back-compat aliases used by API routes
    def _get_tables_postgres_public(self) -> list[str]:
        return self._get_tables_postgres()

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
                tables = self._get_tables()
            return {"success": True, "error": None, "table_count": len(tables)}
        except Exception as e:
            return {"success": False, "error": str(e), "table_count": 0}
