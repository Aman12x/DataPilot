"""
tools/db_tools.py — Unified DB layer: DuckDB + Postgres + MySQL + BigQuery.

All database interaction in DataPilot goes through DBConnection.
No LangGraph or Streamlit imports. Pure Python.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import duckdb
import pandas as pd

# The precount runs analyst/LLM SQL, and a failure quotes it back — DuckDB
# embeds the statement and pandas embeds column names, both customer data.
# tests/test_log_safety.py only scans agents/, but the hazard is identical here.
from agents.log_safety import redact_exception

logger = logging.getLogger(__name__)

# Connections are made to user-supplied hosts. Without an explicit timeout the
# drivers inherit the OS TCP timeout, so a host that silently drops packets
# holds the caller for minutes — and the connection-test endpoints let any
# authenticated user pick that host.
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))
DB_READ_TIMEOUT = int(os.getenv("DB_READ_TIMEOUT", "60"))

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

# Ceiling on rows we will materialise into pandas. This is NOT a truncation
# limit — nothing is ever silently sliced. A result above the ceiling raises
# ResultTooLargeError so the caller aggregates in SQL instead of analysing a
# partial extract. 0 disables the check entirely.
#
# The old behaviour appended `LIMIT 50000` to any query that lacked one, which
# handed CUPED and the t-test a non-deterministic 50k subset (no ORDER BY) and
# reported the effect size as if it covered the population.
_MAX_MATERIALIZE_ROWS = int(os.getenv("SQL_MAX_MATERIALIZE_ROWS", "5000000"))


class ResultTooLargeError(ValueError):
    """Raised when a result set exceeds the materialisation ceiling.

    Carries the real row count so the caller can tell the analyst how far over
    the query is, rather than just that it failed.
    """

    def __init__(self, rows: int, ceiling: int) -> None:
        self.rows = rows
        self.ceiling = ceiling
        super().__init__(
            f"Query returns {rows:,} rows, above the {ceiling:,}-row limit for "
            f"in-memory analysis. Aggregate in SQL (GROUP BY the unit of "
            f"analysis) so the warehouse returns the summary instead of the "
            f"raw rows. Raise SQL_MAX_MATERIALIZE_ROWS only if the full extract "
            f"genuinely has to be materialised."
        )


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


def _checkable_sql(sql: str) -> str:
    """Comments removed and string/identifier literal bodies blanked.

    For structural checks only (statement separators, mutation keywords) —
    never for execution. The LLM annotates its SQL with `-- Assumption: …`
    comments, and a semicolon inside one used to fail the multi-statement
    check; likewise a literal like 'DROP-off rate' must not read as a
    mutation. A scanner (not a regex) because `--` inside a string literal
    does not start a comment.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":                      # line comment
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":                      # block comment
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            continue
        if ch in ("'", '"', "`"):                          # literal / quoted ident
            quote = ch
            out.append(quote)
            i += 1
            while i < n:
                if sql[i] == quote and i + 1 < n and sql[i + 1] == quote:
                    i += 2                                 # doubled-quote escape
                    continue
                if sql[i] == quote:
                    break
                i += 1
            if i < n:
                out.append(quote)
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def validate_sql(sql: str) -> None:
    """
    Defence-in-depth checks before executing analyst/LLM SQL.
    Raises ValueError when the statement looks unsafe or out of scope.
    """
    stripped = sql.strip()
    if not stripped:
        raise ValueError("Empty SQL statement")

    checkable = _checkable_sql(stripped)
    if ";" in checkable.rstrip("; \n\t"):
        raise ValueError("Multi-statement SQL is not permitted")

    first_sql = _strip_leading_sql_comments(stripped)
    upper = first_sql.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT/WITH queries are permitted")

    if _MUTATION_RE.search(checkable):
        raise ValueError("Mutation or privileged SQL is not permitted")

    if _FILE_READ_RE.search(checkable):
        raise ValueError("File-read SQL functions are not permitted")


def _count_wrapper(sql: str) -> str:
    """Wrap a SELECT so the server counts the rows without shipping them.

    The derived table needs an alias on Postgres and MySQL; DuckDB and BigQuery
    accept one. `validate_sql` has already rejected multi-statement input, so
    stripping a trailing semicolon is enough to make this safe to nest.
    """
    return f"SELECT COUNT(*) FROM (\n{sql.rstrip().rstrip(';')}\n) AS _dp_rowcount"


# Strict allowlist, kept for the few places that interpolate a name where
# quoting is not possible — BigQuery's `project.dataset.table` path and the
# MySQL information_schema guards. Prefer quote_ident() everywhere else: this
# rejects legitimate Unicode and digit-leading column names.
_SAFE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# MySQL and BigQuery delimit identifiers with backticks; the rest use the
# SQL-standard double quote.
_BACKTICK_BACKENDS = frozenset({"mysql", "bigquery"})


def quote_ident(name: str, backend: str = "postgres") -> str:
    """Quote a table or column name for interpolation into SQL.

    Doubling the delimiter is the standard escape, and it is what makes an
    arbitrary CSV header safe: once `"` (or a backtick) is doubled there is no
    sequence that terminates the identifier early.

    Escaping rather than allowlisting is deliberate. Uploaded columns are
    normalised with `re.sub(r"[^\\w]", "_", ...)`, and `\\w` is Unicode-aware, so
    legitimate headers survive as `café` or `日本` — and a header like
    `2024 revenue` becomes `2024_revenue`. An `^[a-zA-Z_][a-zA-Z0-9_]*$`
    allowlist would reject all three.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("SQL identifier must be a non-empty string")
    if "\x00" in name:
        raise ValueError("SQL identifier contains a NUL byte")
    if backend in _BACKTICK_BACKENDS:
        return "`" + name.replace("`", "``") + "`"
    return '"' + name.replace('"', '""') + '"'


# Back-compat alias for callers that imported the private name.
_quote_ident = quote_ident


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
        """Execute SQL and return a DataFrame. Only SELECT is permitted.

        The full result set is returned — never a truncated prefix. When it
        exceeds the materialisation ceiling this raises ResultTooLargeError
        rather than analysing a partial extract.
        """
        validate_sql(sql)
        if self.backend == "bigquery":
            # BigQuery reports total_rows on the finished job, so the guard
            # costs no extra query and no extra bytes scanned.
            return self._query_bigquery(sql)

        self._enforce_row_budget(sql)
        if self.backend == "duckdb":
            return self._query_duckdb(sql)
        if self.backend == "postgres":
            return self._query_postgres(sql)
        return self._query_mysql(sql)

    def _enforce_row_budget(self, sql: str) -> None:
        """Count rows server-side and refuse oversized results before fetching."""
        if _MAX_MATERIALIZE_ROWS <= 0:
            return
        rows = self._count_rows(sql)
        if rows is not None and rows > _MAX_MATERIALIZE_ROWS:
            raise ResultTooLargeError(rows, _MAX_MATERIALIZE_ROWS)

    def _count_rows(self, sql: str) -> int | None:
        """Server-side row count for the given SELECT, or None if unavailable.

        A count that fails for any reason must not block the real query — the
        budget check is a guard rail, not a correctness requirement.
        """
        counter = _count_wrapper(sql)
        try:
            if self.backend == "duckdb":
                con = duckdb.connect(self._path, read_only=True)
                try:
                    return int(con.execute(counter).fetchone()[0])
                finally:
                    con.close()
            df = (
                self._query_postgres(counter)
                if self.backend == "postgres"
                else self._query_mysql(counter)
            )
            return int(df.iloc[0, 0])
        except Exception as exc:  # noqa: BLE001 — guard rail, never fatal
            logger.warning(
                "row-budget precount unavailable (%s) — proceeding without it",
                redact_exception(exc),
            )
            return None

    def _query_duckdb(self, sql: str) -> pd.DataFrame:
        con = duckdb.connect(self._path, read_only=True)
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    def _query_postgres(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
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
            connect_timeout=DB_CONNECT_TIMEOUT,
        )
        try:
            return pd.read_sql(sql, conn, params=params)
        finally:
            conn.close()

    def _query_mysql(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
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
            connect_timeout=DB_CONNECT_TIMEOUT,
            read_timeout=DB_READ_TIMEOUT,
            write_timeout=DB_READ_TIMEOUT,
        )
        try:
            return pd.read_sql(sql, conn, params=params)
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

        # result() blocks until the job finishes but downloads nothing, so
        # total_rows enforces the budget without a second query or extra bytes.
        rows = job.result()
        total = getattr(rows, "total_rows", None)
        if _MAX_MATERIALIZE_ROWS > 0 and total is not None and total > _MAX_MATERIALIZE_ROWS:
            raise ResultTooLargeError(int(total), _MAX_MATERIALIZE_ROWS)

        # The Storage Read API streams Arrow instead of paging REST JSON; it is
        # the difference between seconds and minutes on a large extract. Falls
        # back automatically when google-cloud-bigquery-storage is absent.
        return rows.to_dataframe(create_bqstorage_client=True)

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
            # The catalog view compares the table name as a plain value, so it
            # binds as a parameter. PRAGMA table_info cannot be used here: even
            # via pragma_table_info(?) DuckDB re-parses the bound string as a
            # qualified name, which fails on any name containing a quote.
            result = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            return [(row[0], row[1]) for row in result]
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
                # Table and column names are identifiers, so they cannot be
                # bound as parameters — they must be quoted instead. Column
                # names here originate from an uploaded CSV header.
                safe_table = quote_ident(table)
                n_rows = con.execute(
                    f"SELECT COUNT(*) FROM {safe_table}"
                ).fetchone()[0]  # type: ignore[index]
                col_profiles: dict[str, dict] = {}
                for col_name, _ in cols:
                    try:
                        safe_col = quote_ident(col_name)
                        n_distinct = con.execute(
                            f"SELECT COUNT(DISTINCT {safe_col}) FROM {safe_table}"
                        ).fetchone()[0]  # type: ignore[index]
                        samples: list[str] = []
                        if n_distinct <= self._PROFILE_SAMPLE_CARDINALITY:
                            rows = con.execute(
                                f"SELECT DISTINCT CAST({safe_col} AS VARCHAR) "
                                f"FROM {safe_table} WHERE {safe_col} IS NOT NULL "
                                f"ORDER BY 1 LIMIT {self._PROFILE_MAX_SAMPLES}"
                            ).fetchall()
                            samples = [r[0] for r in rows if r[0] is not None]
                        col_profiles[col_name] = {"n_distinct": n_distinct, "samples": samples}
                    except Exception as exc:
                        logger.debug("profile: skipping column %s — %s", col_name, exc)
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

    @staticmethod
    def _split_pg_table(table: str) -> tuple[str, str]:
        """Split a possibly schema-qualified table name into (schema, name).

        Names are emitted by _get_tables_postgres: bare for dot-free public
        tables, "schema.name" otherwise (a public table whose own name contains
        a dot is emitted as "public.the.name", so splitting on the FIRST dot is
        always correct)."""
        if "." in table:
            schema, name = table.split(".", 1)
            return schema, name
        return "public", table

    def _split_home_table(self, table: str, home: str) -> tuple[str, str]:
        """Split "schema.name" with `home` as the bare-name default.

        Same first-dot rule as _split_pg_table: a home table whose own name
        contains a dot is emitted qualified, so bare names never contain dots.
        """
        if "." in table:
            schema, name = table.split(".", 1)
            return schema, name
        return home, table

    def _scope(self) -> list[str] | None:
        """The connection's selected schema/dataset scope, or None (default).

        None keeps each backend's historical behaviour: Postgres sees every
        non-system schema; MySQL and BigQuery see only the home
        schema/dataset. A stored scope makes discovery *chosen*, not
        unbounded — "everything" on a large warehouse would blow up the
        schema context and degrade SQL quality.
        """
        scope = self._kwargs.get("schemas")
        if not scope:
            return None
        cleaned = [str(s).strip() for s in scope if str(s).strip()]
        return cleaned or None

    def _get_tables_postgres(self) -> list[str]:
        # Every schema the role can see, not just public — warehouses organise
        # tables into named schemas (analytics, marts, ...) and a public-only
        # view made them silently invisible. Public keeps bare names so
        # existing annotations and metric packs stay valid.
        scope = self._scope()
        if scope:
            placeholders = ", ".join(["%s"] * len(scope))
            df = self._query_postgres(
                "SELECT table_schema, table_name FROM information_schema.tables "
                f"WHERE table_schema IN ({placeholders}) "
                "ORDER BY (table_schema <> 'public'), table_schema, table_name",
                params=tuple(scope),
            )
        else:
            df = self._query_postgres(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                "ORDER BY (table_schema <> 'public'), table_schema, table_name"
            )
        names: list[str] = []
        for schema, name in zip(df["table_schema"], df["table_name"]):
            if schema == "public" and "." not in name:
                names.append(name)
            else:
                names.append(f"{schema}.{name}")
        return names

    def _get_columns_postgres(self, table: str) -> list[tuple[str, str]]:
        schema, name = self._split_pg_table(table)
        # Comparison values, not identifiers — bind them rather than quoting.
        df = self._query_postgres(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position",
            params=(schema, name),
        )
        return list(zip(df["column_name"], df["data_type"]))

    def _get_tables_mysql(self) -> list[str]:
        # Comparison values, not identifiers — bind them, exactly as the
        # Postgres path does. The allowlist that used to guard the string
        # interpolation here was both the weaker mechanism and the wrong one:
        # it rejects `café`, `日本`, and `2024_revenue`, all legal MySQL names.
        # Mirrors the Postgres naming pattern: bare names for the home schema
        # (annotations and metric packs stay valid), qualified for siblings.
        home = self._kwargs["dbname"]
        schemas = self._scope() or [home]
        placeholders = ", ".join(["%s"] * len(schemas))
        df = self._query_mysql(
            "SELECT table_schema AS table_schema, table_name AS table_name "
            "FROM information_schema.tables "
            f"WHERE table_schema IN ({placeholders}) "
            "ORDER BY (table_schema <> %s), table_schema, table_name",
            params=tuple(schemas) + (home,),
        )
        names: list[str] = []
        for schema, name in zip(df["table_schema"], df["table_name"]):
            if schema == home and "." not in name:
                names.append(name)
            else:
                names.append(f"{schema}.{name}")
        return names

    def _get_columns_mysql(self, table: str) -> list[tuple[str, str]]:
        schema, name = self._split_home_table(table, self._kwargs["dbname"])
        df = self._query_mysql(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position",
            params=(schema, name),
        )
        return list(zip(df["column_name"], df["data_type"]))

    def _get_tables_bigquery(self) -> list[str]:
        client = self._bq_client()
        home = self._kwargs["dataset"]
        datasets = self._scope() or [home]
        names: list[str] = []
        # Home first so its bare names lead the schema context.
        for dataset in sorted(datasets, key=lambda d: d != home):
            try:
                tables = client.list_tables(dataset)
            except Exception:
                # A scoped dataset may have been dropped or the role may have
                # lost access — skip it rather than failing all discovery.
                continue
            for t in tables:
                if dataset == home and "." not in t.table_id:
                    names.append(t.table_id)
                else:
                    names.append(f"{dataset}.{t.table_id}")
        return names

    def _get_columns_bigquery(self, table: str) -> list[tuple[str, str]]:
        dataset, name = self._split_home_table(table, self._kwargs["dataset"])
        # BigQuery is the documented allowlist exception: the project.dataset
        # .table path cannot be quoted per-part here, so each part must pass
        # the strict identifier check.
        for part in (dataset, name):
            if not _SAFE_IDENT_RE.match(part):
                raise ValueError(f"Unsafe table name: {table!r}")
        client = self._bq_client()
        project = self._kwargs["project_id"]
        full = f"{project}.{dataset}.{name}"
        meta = client.get_table(full)
        return [(f.name, f.field_type) for f in meta.schema]

    def list_available_schemas(self) -> list[str]:
        """Discoverable schemas/datasets, for the connection scope picker.

        Home first, then alphabetical. DuckDB has no picker (single file).
        """
        if self.backend == "postgres":
            df = self._query_postgres(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('pg_catalog', 'information_schema') "
                "ORDER BY (schema_name <> 'public'), schema_name"
            )
            return df["schema_name"].tolist()
        if self.backend == "mysql":
            home = self._kwargs["dbname"]
            df = self._query_mysql(
                "SELECT schema_name AS schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN "
                "('information_schema', 'mysql', 'performance_schema', 'sys') "
                "ORDER BY (schema_name <> %s), schema_name",
                params=(home,),
            )
            return df["schema_name"].tolist()
        if self.backend == "bigquery":
            client = self._bq_client()
            home = self._kwargs["dataset"]
            datasets = [d.dataset_id for d in client.list_datasets()]
            return sorted(datasets, key=lambda d: (d != home, d))
        return []

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
            if self.backend == "postgres":
                pg_schema, pg_name = self._split_pg_table(table)
                safe_table = f"{quote_ident(pg_schema)}.{quote_ident(pg_name)}"
            elif self.backend == "mysql":
                my_schema, my_name = self._split_home_table(table, self._kwargs["dbname"])
                safe_table = f"{quote_ident(my_schema, 'mysql')}.{quote_ident(my_name, 'mysql')}"
            else:
                # BigQuery backticks quote the whole dotted path as one
                # identifier, so a qualified "dataset.table" is fine as-is.
                safe_table = quote_ident(table, self.backend)
            safe_col = quote_ident(col, self.backend)
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
                # The table name was interpolated raw here while project and
                # dataset were guarded — escape it like every other identifier.
                # BigQuery quotes the whole path, so the backticks go outside.
                safe_path = "`" + f"{project}.{dataset}.{table}".replace("`", "``") + "`"
                sql = (
                    f"SELECT DISTINCT CAST({safe_col} AS STRING) AS v "
                    f"FROM {safe_path} "
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
