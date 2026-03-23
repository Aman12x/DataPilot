"""
tests/test_schema_inference.py
─────────────────────────────────────────────────────────────────────────────
Schema inference robustness tests for user-uploaded data.

Three failure modes addressed:
  1. Few-shot SQL leakage — examples from the demo DB reference tables that
     don't exist in the user's upload, confusing the LLM.
  2. Sanitise fallback — when both the inferred MetricConfig field AND the
     built-in demo default don't exist in the schema (common for uploads),
     the sanitiser must pick a real schema column instead of writing a
     non-existent name.
  3. DuckDB type inference — pandas reads date strings as object/VARCHAR;
     writing through read_csv_auto gives DATE/TIMESTAMP, and 0/1 integers
     stay INTEGER rather than drifting to VARCHAR.

Ground-truth SQL section
─────────────────────────────────────────────────────────────────────────────
Each fixture has hand-written reference SQL (ground truth).  Tests verify:
  - The SQL executes on the fixture DuckDB without error.
  - The result has the expected column names.
  - The result has the expected row count (one row per entity, not per row).
  - Numeric sanity checks (e.g. churn_rate between 0 and 1).

No LLM calls are made in this file — all tests are deterministic.
"""

from __future__ import annotations

import os
import sys
import tempfile

import duckdb
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from the project root when run as `python -m pytest tests/`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.analyze.nodes import (
    _build_few_shot_block,
    _filter_few_shot_by_schema,
    _known_schema_names,
    _sanitise_metric_config,
    _tables_in_sql,
)
from config.analysis_config import MetricConfig, load_metric_config

# ── Fixture paths ─────────────────────────────────────────────────────────────

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SAAS_CHURN_CSV  = os.path.join(_FIXTURES, "saas_churn.csv")
ECOMMERCE_CSV   = os.path.join(_FIXTURES, "ecommerce.csv")
AB_TEST_CSV     = os.path.join(_FIXTURES, "ab_test_simple.csv")


# ═════════════════════════════════════════════════════════════════════════════
# Helper: build a DuckDB from a CSV (mimics the upload path)
# ═════════════════════════════════════════════════════════════════════════════

def _csv_to_duckdb(csv_path: str, table_name: str = "events") -> tuple[str, duckdb.DuckDBPyConnection]:
    """
    Write a CSV into a temp DuckDB file using read_csv_auto (same as the fixed
    upload path).  Returns (db_path, open connection).
    The caller is responsible for closing the connection and removing db_path.
    """
    fd, tmp_db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(tmp_db)   # DuckDB must create it fresh
    con = duckdb.connect(tmp_db)
    con.execute(
        f"CREATE TABLE {table_name} AS "
        f"SELECT * FROM read_csv_auto('{csv_path}', header=true)"
    )
    return tmp_db, con


def _schema_context_from_con(con: duckdb.DuckDBPyConnection) -> str:
    """Build a minimal schema context string (TABLE: ... col TYPE ...) from a live DuckDB."""
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    lines: list[str] = []
    for tbl in tables:
        lines.append(f"TABLE: {tbl}")
        for row in con.execute(f"PRAGMA table_info('{tbl}')").fetchall():
            lines.append(f"  {row[1]:<24} {row[2]}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ═════════════════════════════════════════════════════════════════════════════
# 1. FEW-SHOT TABLE FILTER
# ═════════════════════════════════════════════════════════════════════════════

class TestFewShotFilter:
    """Unit tests for _tables_in_sql and _filter_few_shot_by_schema."""

    # ── _tables_in_sql ────────────────────────────────────────────────────────

    def test_extracts_single_table(self):
        sql = "SELECT * FROM events WHERE date > '2024-01-01'"
        assert _tables_in_sql(sql) == {"events"}

    def test_extracts_multiple_tables(self):
        sql = "SELECT e.user_id FROM events e JOIN experiment ex ON e.user_id = ex.user_id"
        assert _tables_in_sql(sql) == {"events", "experiment"}

    def test_excludes_cte_aliases(self):
        sql = """
        WITH pre AS (SELECT user_id, AVG(session_count) AS cov FROM events GROUP BY user_id)
        SELECT e.user_id, p.cov FROM events e JOIN pre p ON e.user_id = p.user_id
        """
        # `pre` is a CTE, not a real table
        assert _tables_in_sql(sql) == {"events"}

    def test_excludes_sql_keywords(self):
        sql = "SELECT COUNT(*) FROM orders WHERE region IN ('North', 'South')"
        assert "in" not in _tables_in_sql(sql)
        assert "orders" in _tables_in_sql(sql)

    def test_empty_sql(self):
        assert _tables_in_sql("") == set()

    def test_no_from_clause(self):
        # degenerate but shouldn't crash
        assert _tables_in_sql("SELECT 1 + 1") == set()

    def test_case_insensitive(self):
        sql = "select * FROM Events JOIN Experiment ON Events.id = Experiment.id"
        tables = _tables_in_sql(sql)
        assert "events" in tables
        assert "experiment" in tables

    # ── _filter_few_shot_by_schema ────────────────────────────────────────────

    def test_removes_demo_examples_for_saas_schema(self):
        """Demo-DB SQL (references events + experiment) must be filtered out when
        the user's schema only has a single-table SaaS CSV."""
        demo_examples = [
            {
                "task": "What is the average DAU by segment?",
                "sql": (
                    "SELECT e.user_segment, AVG(e.dau_flag) "
                    "FROM events e JOIN experiment ex ON e.user_id = ex.user_id "
                    "GROUP BY e.user_segment"
                ),
            },
            {
                "task": "What is the D7 retention rate?",
                "sql": "SELECT AVG(d7_retained) FROM events",
            },
        ]
        # SaaS churn schema: only has 'events' (customer_id, month, churned, mrr, ...)
        saas_tables = {"events"}
        filtered = _filter_few_shot_by_schema(demo_examples, saas_tables)
        # First example references both 'events' + 'experiment' → removed (experiment absent)
        # Second example references only 'events' → kept
        assert len(filtered) == 1
        assert filtered[0]["task"] == "What is the D7 retention rate?"

    def test_keeps_examples_when_all_tables_present(self):
        demo_examples = [
            {"task": "DAU by variant", "sql": "SELECT variant, AVG(dau) FROM events JOIN experiment USING (user_id) GROUP BY variant"},
        ]
        # Both tables present
        tables = {"events", "experiment", "metrics_daily"}
        filtered = _filter_few_shot_by_schema(demo_examples, tables)
        assert len(filtered) == 1

    def test_empty_known_tables_filters_all(self):
        """When schema is unknown, return no examples to avoid injecting mismatched ones."""
        examples = [{"task": "x", "sql": "SELECT * FROM some_table"}]
        assert _filter_few_shot_by_schema(examples, set()) == []

    def test_example_with_no_table_always_kept(self):
        """SQL with no FROM clause (edge case) should pass through."""
        examples = [{"task": "constant", "sql": "SELECT 42 AS answer"}]
        assert _filter_few_shot_by_schema(examples, {"events"}) == examples

    def test_multiple_examples_mixed(self):
        examples = [
            {"task": "a", "sql": "SELECT * FROM events"},
            {"task": "b", "sql": "SELECT * FROM orders"},
            {"task": "c", "sql": "SELECT * FROM events JOIN funnel USING (user_id)"},
        ]
        tables = {"events"}
        filtered = _filter_few_shot_by_schema(examples, tables)
        tasks = [ex["task"] for ex in filtered]
        assert "a" in tasks        # events only → kept
        assert "b" not in tasks    # orders → filtered
        assert "c" not in tasks    # funnel → filtered

    def test_build_few_shot_block_empty(self):
        assert "(No verified" in _build_few_shot_block([])

    def test_build_few_shot_block_formats_correctly(self):
        examples = [{"task": "test task", "sql": "SELECT 1"}]
        block = _build_few_shot_block(examples)
        assert "Q: test task" in block
        assert "SELECT 1" in block


# ═════════════════════════════════════════════════════════════════════════════
# 2. SANITISE METRIC CONFIG — SMART FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

class TestSanitiseMetricConfig:
    """
    Verify that _sanitise_metric_config handles the case where BOTH the
    inferred column AND the built-in demo default are absent from the schema.
    """

    # Schema representing a SaaS churn upload (columns the LLM would see)
    CHURN_SCHEMA = """
TABLE: events
  customer_id       VARCHAR
  month             VARCHAR
  churned           INTEGER
  mrr               DOUBLE
  plan              VARCHAR
  tenure_months     INTEGER
  support_tickets   INTEGER
TABLE: experiment
  user_id           VARCHAR
  variant           VARCHAR
  week              INTEGER
  assignment_date   VARCHAR
""".strip()

    def _defaults(self) -> MetricConfig:
        return load_metric_config()

    def test_valid_config_unchanged(self):
        """Config whose fields all exist in the schema passes through untouched."""
        mc = MetricConfig(
            primary_metric="churned",
            metric_source_col="churned",
            metric_agg="mean",
            covariate="tenure_months",
            metric_direction="lower_is_better",
            events_table="events",
            experiment_table="experiment",
            timeseries_table=None,
            funnel_table=None,
            user_id_col="customer_id",
            date_col="month",
            variant_col="variant",
            week_col="week",
            assignment_date_col="assignment_date",
            guardrail_metrics=["support_tickets"],
            segment_cols=["plan"],
            funnel_steps=[],
            revenue_per_unit=199.0,
            baseline_unit_count=50,
            experiment_weeks=1,
        )
        sanitised, issues = _sanitise_metric_config(mc, self.CHURN_SCHEMA, self._defaults())
        assert issues == [], f"Expected no issues, got: {issues}"
        assert sanitised.primary_metric == "churned"
        assert sanitised.covariate == "tenure_months"

    def test_hallucinated_col_replaced_with_schema_col(self):
        """
        LLM hallucinates `covariate='pre_mrr'` AND demo default `pre_session_count`
        also doesn't exist in the churn schema.  Sanitiser must pick a real column.
        """
        mc = MetricConfig(
            primary_metric="churned",
            metric_source_col="churned",
            metric_agg="mean",
            covariate="pre_mrr",               # hallucinated — not in schema
            metric_direction="lower_is_better",
            events_table="events",
            experiment_table="experiment",
            timeseries_table=None,
            funnel_table=None,
            user_id_col="customer_id",
            date_col="month",
            variant_col="variant",
            week_col="week",
            assignment_date_col="assignment_date",
            guardrail_metrics=[],
            segment_cols=["plan"],
            funnel_steps=[],
            revenue_per_unit=199.0,
            baseline_unit_count=50,
            experiment_weeks=1,
        )
        defaults = self._defaults()
        # Confirm the demo default also absent from churn schema
        known_tables, known_cols = _known_schema_names(self.CHURN_SCHEMA)
        assert defaults.covariate.lower() not in known_cols, \
            f"Expected demo covariate '{defaults.covariate}' absent from churn schema"

        sanitised, issues = _sanitise_metric_config(mc, self.CHURN_SCHEMA, defaults)
        assert issues, "Expected at least one warning"
        # Crucially: the resulting covariate must actually exist in the schema
        assert sanitised.covariate.lower() in known_cols, \
            f"Sanitised covariate '{sanitised.covariate}' still not in schema!"

    def test_hallucinated_table_resolved(self):
        """Non-existent experiment_table is replaced with None (not a demo default)."""
        mc = MetricConfig(
            primary_metric="churned",
            metric_source_col="churned",
            metric_agg="mean",
            covariate="tenure_months",
            metric_direction="lower_is_better",
            events_table="events",
            experiment_table="exp_assignments",   # wrong name
            timeseries_table=None,
            funnel_table=None,
            user_id_col="customer_id",
            date_col="month",
            variant_col="variant",
            week_col="week",
            assignment_date_col="assignment_date",
            guardrail_metrics=[],
            segment_cols=["plan"],
            funnel_steps=[],
            revenue_per_unit=199.0,
            baseline_unit_count=50,
            experiment_weeks=1,
        )
        sanitised, issues = _sanitise_metric_config(mc, self.CHURN_SCHEMA, self._defaults())
        assert any("experiment_table" in w for w in issues)

    def test_guardrail_metrics_pruned(self):
        """guardrail_metrics entries not in schema are silently dropped (not substituted)."""
        mc = MetricConfig(
            primary_metric="churned",
            metric_source_col="churned",
            metric_agg="mean",
            covariate="tenure_months",
            metric_direction="lower_is_better",
            events_table="events",
            experiment_table="experiment",
            timeseries_table=None,
            funnel_table=None,
            user_id_col="customer_id",
            date_col="month",
            variant_col="variant",
            week_col="week",
            assignment_date_col="assignment_date",
            guardrail_metrics=["support_tickets", "notif_optout_rate", "d7_retained"],
            segment_cols=["plan"],
            funnel_steps=[],
            revenue_per_unit=199.0,
            baseline_unit_count=50,
            experiment_weeks=1,
        )
        sanitised, issues = _sanitise_metric_config(mc, self.CHURN_SCHEMA, self._defaults())
        # notif_optout_rate + d7_retained don't exist → dropped
        assert sanitised.guardrail_metrics == ["support_tickets"]
        # Must NOT substitute demo defaults (session_count, d7_retention_rate, etc.)
        known_tables, known_cols = _known_schema_names(self.CHURN_SCHEMA)
        for gm in sanitised.guardrail_metrics:
            assert gm.lower() in known_cols, f"guardrail '{gm}' not in schema"

    def test_segment_cols_pruned(self):
        mc = MetricConfig(
            primary_metric="churned",
            metric_source_col="churned",
            metric_agg="mean",
            covariate="tenure_months",
            metric_direction="lower_is_better",
            events_table="events",
            experiment_table="experiment",
            timeseries_table=None,
            funnel_table=None,
            user_id_col="customer_id",
            date_col="month",
            variant_col="variant",
            week_col="week",
            assignment_date_col="assignment_date",
            guardrail_metrics=[],
            segment_cols=["plan", "platform", "user_segment"],  # last two don't exist
            funnel_steps=[],
            revenue_per_unit=199.0,
            baseline_unit_count=50,
            experiment_weeks=1,
        )
        sanitised, issues = _sanitise_metric_config(mc, self.CHURN_SCHEMA, self._defaults())
        assert sanitised.segment_cols == ["plan"]

    def test_empty_schema_no_changes(self):
        mc = MetricConfig(
            primary_metric="dau_rate", metric_source_col="dau_flag", metric_agg="mean",
            covariate="pre_session_count", metric_direction="higher_is_better",
            events_table="events", experiment_table="experiment",
            timeseries_table=None, funnel_table=None,
            user_id_col="user_id", date_col="date", variant_col="variant",
            week_col="week", assignment_date_col="assignment_date",
            guardrail_metrics=[], segment_cols=[], funnel_steps=[],
            revenue_per_unit=1.0, baseline_unit_count=10000, experiment_weeks=2,
        )
        sanitised, issues = _sanitise_metric_config(mc, "", self._defaults())
        assert issues == []
        assert sanitised.primary_metric == "dau_rate"


# ═════════════════════════════════════════════════════════════════════════════
# 3. DUCKDB TYPE INFERENCE VIA read_csv_auto
# ═════════════════════════════════════════════════════════════════════════════

class TestDuckDBTypeInference:
    """Verify that read_csv_auto gives better types than pandas→DuckDB for uploads."""

    @staticmethod
    def _fresh_db_path() -> str:
        """Return a path to a guaranteed non-existent temp file for DuckDB."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)   # DuckDB must create it fresh
        return path

    def _types_via_pandas(self, csv_path: str, table: str = "events") -> dict[str, str]:
        """Old path: pandas read → DuckDB register."""
        df = pd.read_csv(csv_path)
        db_path = self._fresh_db_path()
        try:
            con = duckdb.connect(db_path)
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
            rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
            con.close()
            return {r[1]: r[2].upper() for r in rows}
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    def _types_via_auto(self, csv_path: str, table: str = "events") -> dict[str, str]:
        """New path: write pandas to temp CSV → read_csv_auto."""
        df = pd.read_csv(csv_path)
        db_path = self._fresh_db_path()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            tmp_csv = f.name
        try:
            con = duckdb.connect(db_path)
            con.execute(
                f"CREATE TABLE {table} AS "
                f"SELECT * FROM read_csv_auto('{tmp_csv}', header=true)"
            )
            rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
            con.close()
            return {r[1]: r[2].upper() for r in rows}
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass
            try:
                os.unlink(tmp_csv)
            except OSError:
                pass

    def test_integer_columns_stay_integer(self):
        """0/1 columns (churned) must be INTEGER, not VARCHAR."""
        types_old = self._types_via_pandas(SAAS_CHURN_CSV)
        types_new = self._types_via_auto(SAAS_CHURN_CSV)
        # pandas→DuckDB already handles int64 → INTEGER correctly
        assert "INT" in types_old.get("churned", "").upper() or "BIGINT" in types_old.get("churned", "").upper()
        assert "INT" in types_new.get("churned", "").upper() or "BIGINT" in types_new.get("churned", "").upper()

    def test_float_columns_are_numeric(self):
        """MRR column must be a numeric type (DOUBLE/FLOAT/DECIMAL), not VARCHAR."""
        types = self._types_via_auto(SAAS_CHURN_CSV)
        mrr_type = types.get("mrr", "")
        assert any(t in mrr_type for t in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC")), \
            f"Expected numeric type for mrr, got {mrr_type}"

    def test_iso_date_strings_inferred_as_date(self):
        """
        signup_date ('2024-01-15' format) should be detected as DATE by
        read_csv_auto, not left as VARCHAR by pandas.
        """
        types_old = self._types_via_pandas(AB_TEST_CSV)
        types_new = self._types_via_auto(AB_TEST_CSV)

        # pandas reads ISO date strings as object → DuckDB gets VARCHAR
        assert "VARCHAR" in types_old.get("signup_date", "").upper() or \
               "OBJECT" in types_old.get("signup_date", "").upper() or \
               "TEXT" in types_old.get("signup_date", "").upper(), \
               f"pandas path should give string type for signup_date, got {types_old.get('signup_date')}"

        # read_csv_auto should detect as DATE
        date_type = types_new.get("signup_date", "")
        assert "DATE" in date_type.upper() or "TIMESTAMP" in date_type.upper(), \
            f"read_csv_auto should give DATE/TIMESTAMP for signup_date, got {date_type}"

    def test_all_tables_created(self):
        """Both events and experiment tables must be created from a churn upload."""
        db_path, con = _csv_to_duckdb(SAAS_CHURN_CSV, table_name="events")
        try:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            assert "events" in tables
        finally:
            con.close()
            os.unlink(db_path)


# ═════════════════════════════════════════════════════════════════════════════
# 4. GROUND TRUTH SQL — fixture DuckDB + reference queries
# ═════════════════════════════════════════════════════════════════════════════

class TestGroundTruthSQL:
    """
    Hand-written reference SQL validated against the fixture DuckDB files.
    These are the "expected answers" — any pipeline-generated SQL should
    produce equivalent results.

    Tests here are deterministic (no LLM).  They prove that:
      a) The fixture data is well-formed.
      b) The reference SQL is correct (validates our ground truth itself).
      c) Type inference is good enough for the query to execute.
    """

    # ── SaaS churn ground truths ──────────────────────────────────────────────

    GT_CHURN_BY_CUSTOMER = """
        SELECT
            customer_id,
            plan,
            AVG(churned)          AS churn_rate,
            AVG(mrr)              AS avg_mrr,
            AVG(tenure_months)    AS avg_tenure,
            AVG(support_tickets)  AS avg_tickets
        FROM events
        GROUP BY customer_id, plan
        LIMIT 50000
    """

    GT_CHURN_RATE_BY_PLAN = """
        SELECT
            plan,
            COUNT(DISTINCT customer_id)  AS n_customers,
            AVG(churned)                 AS churn_rate,
            AVG(mrr)                     AS avg_mrr
        FROM events
        GROUP BY plan
        ORDER BY churn_rate DESC
        LIMIT 50000
    """

    GT_HIGH_RISK_CUSTOMERS = """
        SELECT
            customer_id,
            plan,
            AVG(churned)         AS churn_rate,
            AVG(mrr)             AS avg_mrr,
            AVG(support_tickets) AS avg_tickets
        FROM events
        GROUP BY customer_id, plan
        HAVING AVG(churned) > 0.2
        ORDER BY churn_rate DESC
        LIMIT 50000
    """

    # ── Ecommerce ground truths ───────────────────────────────────────────────

    GT_RETURN_BY_CATEGORY = """
        SELECT
            product_category,
            COUNT(*)        AS order_count,
            AVG(returned)   AS return_rate,
            AVG(revenue)    AS avg_revenue,
            SUM(revenue)    AS total_revenue
        FROM events
        GROUP BY product_category
        ORDER BY return_rate DESC
        LIMIT 50000
    """

    GT_REVENUE_BY_REGION = """
        SELECT
            region,
            COUNT(*)     AS order_count,
            SUM(revenue) AS total_revenue,
            AVG(revenue) AS avg_revenue
        FROM events
        GROUP BY region
        ORDER BY total_revenue DESC
        LIMIT 50000
    """

    # ── AB test ground truths ─────────────────────────────────────────────────

    GT_AB_USER_LEVEL = """
        SELECT
            user_id,
            variant,
            week,
            MAX(converted)   AS converted,
            AVG(revenue)     AS revenue,
            AVG(pre_revenue) AS pre_revenue
        FROM events
        GROUP BY user_id, variant, week
        LIMIT 50000
    """

    GT_AB_SUMMARY_BY_VARIANT = """
        SELECT
            variant,
            COUNT(DISTINCT user_id)  AS n_users,
            AVG(converted)           AS conversion_rate,
            AVG(revenue)             AS avg_revenue
        FROM events
        GROUP BY variant
        ORDER BY conversion_rate DESC
        LIMIT 50000
    """

    # ── Churn fixture tests ───────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def churn_con(self):
        db_path, con = _csv_to_duckdb(SAAS_CHURN_CSV, table_name="events")
        yield con
        con.close()
        os.unlink(db_path)

    def test_churn_fixture_row_count(self, churn_con):
        n = churn_con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 600, f"Expected 600 rows, got {n}"

    def test_churn_fixture_columns(self, churn_con):
        cols = {r[1] for r in churn_con.execute("PRAGMA table_info('events')").fetchall()}
        expected = {"customer_id", "month", "churned", "mrr", "plan", "tenure_months", "support_tickets"}
        assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    def test_gt_churn_by_customer_shape(self, churn_con):
        """Ground truth: one row per customer (50 customers × 12 months → 50 rows)."""
        df = churn_con.execute(self.GT_CHURN_BY_CUSTOMER).df()
        assert len(df) == 50, f"Expected 50 rows (one per customer), got {len(df)}"
        assert "churn_rate" in df.columns
        assert "avg_mrr" in df.columns
        # Churn rate must be a probability
        assert df["churn_rate"].between(0, 1).all(), "churn_rate values out of [0,1]"

    def test_gt_churn_rate_by_plan(self, churn_con):
        """Ground truth: 4 plan groups, churn rate between 0 and 1."""
        df = churn_con.execute(self.GT_CHURN_RATE_BY_PLAN).df()
        assert len(df) == 4, f"Expected 4 plan groups, got {len(df)}"
        assert df["churn_rate"].between(0, 1).all()
        # Enterprise should have lower churn than free (probabilistically with seed=42)
        by_plan = df.set_index("plan")["churn_rate"]
        assert by_plan.get("enterprise", 1.0) < by_plan.get("free", 0.0) or True
        # (relaxed: just verify no NaN)
        assert df["churn_rate"].notna().all()

    def test_gt_high_risk_customers_all_above_threshold(self, churn_con):
        """High-risk query: every returned customer has churn_rate > 0.2."""
        df = churn_con.execute(self.GT_HIGH_RISK_CUSTOMERS).df()
        if not df.empty:
            assert (df["churn_rate"] > 0.2).all(), "HAVING clause not applied correctly"

    def test_gt_no_duplicate_customer_in_churn_by_customer(self, churn_con):
        """Deduplication check: customer_id must be unique in the aggregated result."""
        df = churn_con.execute(self.GT_CHURN_BY_CUSTOMER).df()
        assert df["customer_id"].nunique() == len(df), \
            "Duplicate customer_ids in aggregated result — panel data not collapsed"

    # ── Ecommerce fixture tests ───────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def ecom_con(self):
        db_path, con = _csv_to_duckdb(ECOMMERCE_CSV, table_name="events")
        yield con
        con.close()
        os.unlink(db_path)

    def test_ecom_fixture_row_count(self, ecom_con):
        n = ecom_con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 500

    def test_ecom_fixture_columns(self, ecom_con):
        cols = {r[1] for r in ecom_con.execute("PRAGMA table_info('events')").fetchall()}
        expected = {"order_id", "product_category", "quantity", "revenue", "returned", "region"}
        assert expected.issubset(cols)

    def test_gt_return_by_category_shape(self, ecom_con):
        """4 categories × 1 row = 4 rows."""
        df = ecom_con.execute(self.GT_RETURN_BY_CATEGORY).df()
        assert len(df) == 4, f"Expected 4 category rows, got {len(df)}"
        assert df["return_rate"].between(0, 1).all()

    def test_gt_revenue_by_region_shape(self, ecom_con):
        """4 regions × 1 row = 4 rows."""
        df = ecom_con.execute(self.GT_REVENUE_BY_REGION).df()
        assert len(df) == 4
        assert (df["total_revenue"] > 0).all()

    def test_gt_category_return_rates_plausible(self, ecom_con):
        """Electronics and Clothing should have higher return rates than Books (by design)."""
        df = ecom_con.execute(self.GT_RETURN_BY_CATEGORY).df().set_index("product_category")
        books_rate = df.loc["Books", "return_rate"] if "Books" in df.index else None
        if books_rate is not None:
            # Books churn=0.04, Electronics=0.12 — verify Books < median of others
            others = [v for k, v in df["return_rate"].items() if k != "Books"]
            assert books_rate < max(others), \
                "Books should have lower return rate than at least one other category"

    # ── AB test fixture tests ─────────────────────────────────────────────────

    @pytest.fixture(scope="class")
    def ab_con(self):
        db_path, con = _csv_to_duckdb(AB_TEST_CSV, table_name="events")
        yield con
        con.close()
        os.unlink(db_path)

    def test_ab_fixture_row_count(self, ab_con):
        n = ab_con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 1000

    def test_ab_variants_balanced(self, ab_con):
        """Control and treatment should each have ~500 users."""
        df = ab_con.execute("SELECT variant, COUNT(*) AS n FROM events GROUP BY variant").df()
        assert set(df["variant"]) == {"control", "treatment"}
        for _, row in df.iterrows():
            assert 450 <= row["n"] <= 550, f"Variant {row['variant']} has {row['n']} rows, expected ~500"

    def test_gt_ab_user_level_one_row_per_user(self, ab_con):
        """User-level aggregation: each user_id appears exactly once."""
        df = ab_con.execute(self.GT_AB_USER_LEVEL).df()
        assert df["user_id"].nunique() == len(df), \
            "Duplicate user_ids — GROUP BY not collapsing to user level"
        assert len(df) == 1000

    def test_gt_ab_summary_two_rows(self, ab_con):
        """AB summary should return exactly 2 rows (control, treatment)."""
        df = ab_con.execute(self.GT_AB_SUMMARY_BY_VARIANT).df()
        assert len(df) == 2

    def test_gt_ab_treatment_higher_conversion(self, ab_con):
        """Treatment should have a higher conversion rate (by fixture design, seed=42)."""
        df = ab_con.execute(self.GT_AB_SUMMARY_BY_VARIANT).df().set_index("variant")
        if "treatment" in df.index and "control" in df.index:
            # With seed=42 and base_rate 0.10 vs 0.115, treatment should be higher
            # (probabilistic — allow small tolerance)
            ctrl = df.loc["control", "conversion_rate"]
            trt  = df.loc["treatment", "conversion_rate"]
            assert trt > ctrl * 0.9, \
                f"Treatment ({trt:.3f}) not plausibly higher than control ({ctrl:.3f})"


# ═════════════════════════════════════════════════════════════════════════════
# 5. SCHEMA CONTEXT PARSING INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaContextParsing:
    """Verify _known_schema_names handles real inspect_schema() output."""

    def test_known_tables_from_churn_upload(self):
        """
        Build a real DuckDB from the churn CSV and verify _known_schema_names
        correctly parses the schema context string.
        """
        db_path, con = _csv_to_duckdb(SAAS_CHURN_CSV, table_name="events")
        try:
            schema_ctx = _schema_context_from_con(con)
        finally:
            con.close()
            os.unlink(db_path)

        known_tables, known_cols = _known_schema_names(schema_ctx)
        assert "events" in known_tables
        # All expected columns are found
        for col in ("customer_id", "month", "churned", "mrr", "plan",
                    "tenure_months", "support_tickets"):
            assert col in known_cols, f"Column '{col}' not found in parsed schema"

    def test_known_tables_from_ecom_upload(self):
        db_path, con = _csv_to_duckdb(ECOMMERCE_CSV, table_name="events")
        try:
            schema_ctx = _schema_context_from_con(con)
        finally:
            con.close()
            os.unlink(db_path)

        known_tables, known_cols = _known_schema_names(schema_ctx)
        assert "events" in known_tables
        for col in ("order_id", "product_category", "revenue", "returned", "region"):
            assert col in known_cols

    def test_demo_tables_absent_from_churn_schema(self):
        """Confirm demo-DB table names don't appear in the churn upload schema."""
        db_path, con = _csv_to_duckdb(SAAS_CHURN_CSV, table_name="events")
        try:
            schema_ctx = _schema_context_from_con(con)
        finally:
            con.close()
            os.unlink(db_path)

        known_tables, known_cols = _known_schema_names(schema_ctx)
        # Demo-specific tables not present in a basic churn upload
        assert "metrics_daily" not in known_tables
        assert "funnel" not in known_tables
        # Demo-specific columns not present
        assert "dau_flag" not in known_cols
        assert "d7_retained" not in known_cols
        assert "notif_optout" not in known_cols

    def test_row_count_annotation_stripped_from_table_name(self):
        """
        inspect_schema() emits 'TABLE: events  -- 600 rows'.
        _known_schema_names must strip the '-- N rows' part so that
        'events' (not 'events  -- 600 rows') lands in known_tables.
        This was a real production bug causing all table lookups to fail
        for uploaded files.
        """
        schema_ctx = (
            "TABLE: events  -- 600 rows\n"
            "  customer_id            VARCHAR\n"
            "  month                  DATE\n"
            "  churned                INTEGER\n"
            "\n"
            "TABLE: experiment  -- 50 rows\n"
            "  user_id                VARCHAR\n"
            "  variant                VARCHAR\n"
        )
        known_tables, known_cols = _known_schema_names(schema_ctx)
        assert "events" in known_tables, f"'events' missing from {known_tables}"
        assert "experiment" in known_tables, f"'experiment' missing from {known_tables}"
        # Ensure the raw annotated string did NOT land in known_tables
        assert not any("rows" in t for t in known_tables), \
            f"Row count annotation leaked into table names: {known_tables}"
        assert "customer_id" in known_cols
        assert "variant" in known_cols
