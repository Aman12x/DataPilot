"""
tests/test_canonical_generalisability.py

Generalisability tests: does the system work correctly with different MetricConfigs,
tables, aggregation types, and edge-case scenarios?

Tests cover:
  1. Canonical SQL execution — various MetricConfig permutations
  2. _sanitise_metric_config — hallucination handling
  3. _apply_intent_to_config — events-table guard, intent overrides
  4. Duplicate-column deduplication (covariate == guardrail)
  5. run_cuped + execute_query with alternate configs
"""

from __future__ import annotations

import sys
import os
import pytest
import duckdb
import pandas as pd
import numpy as np

# Ensure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.analysis_config import MetricConfig
from agents.analyze.nodes import (
    _canonical_experiment_sql,
    _sanitise_metric_config,
    _apply_intent_to_config,
    _columns_for_table,
    _known_schema_names,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_db(tmp_path_factory):
    """
    Full 4-table DuckDB with 200 users.

    Tables: events, experiment, funnel, metrics_daily
    Assignment date: 2024-02-15  (pre-exp: Jan, exp: Feb 15 – Mar)
    Variants: 100 control, 100 treatment
    """
    rng = np.random.default_rng(0)
    path = str(tmp_path_factory.mktemp("db") / "full.db")
    con = duckdb.connect(path)

    n_users = 200
    user_ids = [f"u_{i:04d}" for i in range(n_users)]
    variants = ["control"] * 100 + ["treatment"] * 100
    platforms = rng.choice(["android", "ios", "web"], n_users).tolist()
    segments  = rng.choice(["new", "returning", "power"], n_users).tolist()

    # ── events ────────────────────────────────────────────────────────────────
    # 14 pre-experiment days + 14 experiment days
    pre_dates = pd.date_range("2024-01-01", periods=14).tolist()
    exp_dates = pd.date_range("2024-02-15", periods=14).tolist()
    all_dates = pre_dates + exp_dates

    rows = []
    for uid, variant, platform, segment in zip(user_ids, variants, platforms, segments):
        baseline = rng.normal(0.6, 0.1)
        for date in all_dates:
            te = -0.15 if (variant == "treatment" and platform == "android") else 0.0
            dau = int(rng.random() < max(0, min(1, baseline + te + rng.normal(0, 0.05))))
            rows.append({
                "user_id":      uid,
                "date":         date.date(),
                "platform":     platform,
                "user_segment": segment,
                "dau_flag":     dau,
                "session_count": max(0, int(rng.poisson(3))),
                "notif_optout": int(rng.random() < (0.08 if variant == "treatment" else 0.03)),
                "d7_retained":  int(rng.random() < (0.42 if variant == "treatment" else 0.50)),
                "revenue":      round(float(rng.exponential(0.5)) * dau, 4),
            })

    events_df = pd.DataFrame(rows)
    con.execute("CREATE TABLE events AS SELECT * FROM events_df")

    # ── experiment ────────────────────────────────────────────────────────────
    exp_rows = []
    for uid, variant in zip(user_ids, variants):
        for week in [1, 2]:
            exp_rows.append({
                "user_id":         uid,
                "variant":         variant,
                "assignment_date": pd.Timestamp("2024-02-15").date(),
                "week":            week,
            })
    exp_df = pd.DataFrame(exp_rows)
    con.execute("CREATE TABLE experiment AS SELECT * FROM exp_df")

    # ── funnel ────────────────────────────────────────────────────────────────
    steps = ["impression", "click", "install", "d1_retain"]
    base_rates = {"impression": 1.0, "click": 0.35, "install": 0.55, "d1_retain": 0.45}
    funnel_rows = []
    for uid, variant, platform, segment in zip(user_ids, variants, platforms, segments):
        affected = (variant == "treatment" and platform == "android")
        prev = True
        for step in steps:
            rate = base_rates[step] * (0.75 if affected and step == "d1_retain" else 1.0)
            completed = int(prev and rng.random() < rate)
            funnel_rows.append({
                "user_id":      uid,
                "variant":      variant,
                "platform":     platform,
                "user_segment": segment,
                "step":         step,
                "completed":    completed,
            })
            prev = bool(completed)
    funnel_df = pd.DataFrame(funnel_rows)
    con.execute("CREATE TABLE funnel AS SELECT * FROM funnel_df")

    # ── metrics_daily ─────────────────────────────────────────────────────────
    daily_rows = []
    for day_idx in range(28):
        date = pd.Timestamp("2024-02-01") + pd.Timedelta(days=day_idx)
        for platform in ["android", "ios", "web"]:
            daily_rows.append({
                "date":              date.date(),
                "platform":          platform,
                "user_segment":      "all",
                "dau":               int(rng.normal(400, 20)),
                "avg_session_count": float(rng.normal(3.0, 0.2)),
                "d7_retention_rate": float(np.clip(rng.normal(0.45, 0.02), 0, 1)),
                "notif_optout_rate": float(np.clip(rng.normal(0.03, 0.005), 0, 1)),
            })
    daily_df = pd.DataFrame(daily_rows)
    con.execute("CREATE TABLE metrics_daily AS SELECT * FROM daily_df")

    con.close()
    yield path


@pytest.fixture(scope="module")
def db_conn(full_db):
    """Open DuckDB connection for SQL tests (read-only reuse)."""
    return duckdb.connect(full_db, read_only=True)


@pytest.fixture(scope="module")
def schema_context(full_db):
    """Schema context string matching what db_tools.inspect_schema() would produce."""
    con = duckdb.connect(full_db, read_only=True)
    lines = ["DIALECT: duckdb"]
    for tbl in ["events", "experiment", "funnel", "metrics_daily"]:
        lines.append(f"TABLE: {tbl}")
        cols = con.execute(f"PRAGMA table_info('{tbl}')").fetchdf()
        for _, row in cols.iterrows():
            lines.append(f"  {row['name']} {row['type']}")
    con.close()
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_sql(db_conn, sql: str) -> pd.DataFrame:
    """Execute SQL and return DataFrame."""
    return db_conn.execute(sql).df()


def _make_mc(**overrides) -> MetricConfig:
    """Build a MetricConfig from the default plus overrides."""
    defaults = dict(
        primary_metric="dau_rate",
        metric_source_col="dau_flag",
        metric_agg="mean",
        covariate="session_count",
        metric_direction="higher_is_better",
        events_table="events",
        experiment_table="experiment",
        timeseries_table="metrics_daily",
        funnel_table="funnel",
        user_id_col="user_id",
        date_col="date",
        variant_col="variant",
        week_col="week",
        guardrail_metrics=["notif_optout", "d7_retained"],
        segment_cols=["platform", "user_segment"],
        funnel_steps=["impression", "click", "install", "d1_retain"],
        revenue_per_unit=0.50,
        baseline_unit_count=200,
        experiment_weeks=2,
    )
    defaults.update(overrides)
    return MetricConfig(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Canonical SQL execution — various MetricConfig permutations
# ══════════════════════════════════════════════════════════════════════════════

class TestCanonicalSQL:
    """
    Each test builds a MetricConfig variant, generates canonical SQL,
    executes it against the full_db fixture, and asserts correctness.
    """

    def _check(self, db_conn, mc: MetricConfig):
        """Run canonical SQL, verify columns and non-empty result."""
        sql = _canonical_experiment_sql(mc)
        df  = _run_sql(db_conn, sql)

        # Required columns must be present
        assert mc.primary_metric in df.columns, f"Missing primary metric: {mc.primary_metric}"
        assert mc.covariate      in df.columns, f"Missing covariate: {mc.covariate}"
        assert "variant"         in df.columns
        assert "week"            in df.columns

        # Guardrail columns (not shadowed by primary/covariate)
        for g in mc.guardrail_metrics:
            if g not in (mc.primary_metric, mc.metric_source_col, mc.covariate):
                assert g in df.columns, f"Missing guardrail: {g}"

        # Segment columns
        for s in mc.segment_cols:
            assert s in df.columns, f"Missing segment: {s}"

        # No duplicate column names (our dedup guarantee)
        assert not df.columns.duplicated().any(), "Duplicate columns in result"

        # Must have rows
        assert len(df) > 0, "Empty result"

        # Variant values are correct
        assert set(df["variant"].unique()) <= {"control", "treatment"}

        return df

    def test_default_config(self, db_conn):
        """Default: dau_rate (mean dau_flag), session_count covariate."""
        mc = _make_mc()
        df = self._check(db_conn, mc)
        assert df["dau_rate"].between(0, 1).all()

    def test_sum_aggregation(self, db_conn):
        """Revenue total per user (sum aggregation)."""
        mc = _make_mc(
            primary_metric="total_revenue",
            metric_source_col="revenue",
            metric_agg="sum",
            covariate="session_count",
            guardrail_metrics=["notif_optout", "d7_retained"],
        )
        df = self._check(db_conn, mc)
        assert (df["total_revenue"] >= 0).all()

    def test_count_aggregation(self, db_conn):
        """Count aggregation (total events per user)."""
        mc = _make_mc(
            primary_metric="event_count",
            metric_source_col="dau_flag",
            metric_agg="count",
            covariate="session_count",
            guardrail_metrics=["notif_optout"],
        )
        df = self._check(db_conn, mc)
        assert (df["event_count"] >= 0).all()

    def test_lower_is_better_metric(self, db_conn):
        """notif_optout as primary (lower_is_better)."""
        mc = _make_mc(
            primary_metric="notif_optout",
            metric_source_col="notif_optout",
            metric_agg="mean",
            metric_direction="lower_is_better",
            covariate="session_count",
            guardrail_metrics=["d7_retained"],
        )
        df = self._check(db_conn, mc)
        assert df["notif_optout"].between(0, 1).all()

    def test_d7_retained_as_primary(self, db_conn):
        """7-day retention as primary metric."""
        mc = _make_mc(
            primary_metric="d7_retained",
            metric_source_col="d7_retained",
            metric_agg="mean",
            metric_direction="higher_is_better",
            covariate="session_count",
            guardrail_metrics=["notif_optout"],
        )
        df = self._check(db_conn, mc)
        assert df["d7_retained"].between(0, 1).all()

    def test_session_count_as_primary(self, db_conn):
        """Session count as primary metric (continuous, mean)."""
        mc = _make_mc(
            primary_metric="sessions_per_user",
            metric_source_col="session_count",
            metric_agg="mean",
            covariate="dau_flag",
            guardrail_metrics=["notif_optout", "d7_retained"],
        )
        df = self._check(db_conn, mc)
        assert (df["sessions_per_user"] >= 0).all()

    def test_single_segment(self, db_conn):
        """Only platform as segment (no user_segment)."""
        mc = _make_mc(segment_cols=["platform"])
        df = self._check(db_conn, mc)
        assert "platform"     in df.columns
        assert "user_segment" not in df.columns

    def test_no_segments(self, db_conn):
        """No segment columns at all."""
        mc = _make_mc(segment_cols=[])
        df = self._check(db_conn, mc)
        assert "platform"     not in df.columns
        assert "user_segment" not in df.columns

    def test_single_guardrail(self, db_conn):
        """Only one guardrail metric."""
        mc = _make_mc(guardrail_metrics=["notif_optout"])
        df = self._check(db_conn, mc)
        assert "notif_optout" in df.columns
        assert "d7_retained"  not in df.columns

    def test_covariate_equals_guardrail_no_duplicate(self, db_conn):
        """
        When covariate == a guardrail metric, canonical SQL must NOT emit
        a duplicate column — the guardrail filter should skip it.
        """
        mc = _make_mc(
            covariate="session_count",
            guardrail_metrics=["session_count", "notif_optout", "d7_retained"],
        )
        sql = _canonical_experiment_sql(mc)
        df  = _run_sql(db_conn, sql)
        # session_count should appear exactly once
        assert list(df.columns).count("session_count") == 1
        assert "notif_optout" in df.columns
        assert "d7_retained"  in df.columns

    def test_primary_equals_guardrail_no_duplicate(self, db_conn):
        """
        When primary_metric (alias) == a guardrail metric name, the guardrail
        filter should skip it to avoid duplicate column names.
        """
        mc = _make_mc(
            primary_metric="notif_optout",
            metric_source_col="notif_optout",
            guardrail_metrics=["notif_optout", "d7_retained"],
        )
        sql = _canonical_experiment_sql(mc)
        df  = _run_sql(db_conn, sql)
        assert list(df.columns).count("notif_optout") == 1

    def test_result_has_both_variants(self, db_conn):
        """Both control and treatment must be present in the output."""
        mc = _make_mc()
        df = _run_sql(db_conn, _canonical_experiment_sql(mc))
        assert "control"   in df["variant"].values
        assert "treatment" in df["variant"].values

    def test_result_has_both_weeks(self, db_conn):
        """Both week 1 and week 2 must appear (experiment table has 2 rows/user)."""
        mc = _make_mc()
        df = _run_sql(db_conn, _canonical_experiment_sql(mc))
        assert 1 in df["week"].values
        assert 2 in df["week"].values


# ══════════════════════════════════════════════════════════════════════════════
# 2. _sanitise_metric_config — hallucination handling
# ══════════════════════════════════════════════════════════════════════════════

class TestSanitiseMetricConfig:

    @pytest.fixture(autouse=True)
    def _defaults(self):
        self.defaults = _make_mc()

    def test_valid_config_unchanged(self, schema_context):
        """A fully valid config passes through unchanged."""
        mc = _make_mc()
        sanitised, warnings = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert sanitised.primary_metric  == mc.primary_metric
        assert sanitised.metric_source_col == mc.metric_source_col
        assert sanitised.covariate       == mc.covariate
        assert warnings == []

    def test_hallucinated_metric_source_col(self, schema_context):
        """Hallucinated metric_source_col falls back to default."""
        mc = _make_mc(metric_source_col="does_not_exist_col")
        sanitised, warnings = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert sanitised.metric_source_col != "does_not_exist_col"
        assert len(warnings) > 0

    def test_hallucinated_covariate(self, schema_context):
        """Hallucinated covariate falls back to default."""
        mc = _make_mc(covariate="made_up_covariate")
        sanitised, warnings = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert sanitised.covariate != "made_up_covariate"
        assert len(warnings) > 0

    def test_hallucinated_guardrail_dropped(self, schema_context):
        """Hallucinated guardrail metric is removed from the list."""
        mc = _make_mc(guardrail_metrics=["notif_optout", "ghost_column"])
        sanitised, warnings = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert "ghost_column" not in sanitised.guardrail_metrics
        assert "notif_optout" in sanitised.guardrail_metrics

    def test_all_guardrails_hallucinated_falls_back(self, schema_context):
        """If all guardrails are hallucinated, falls back to defaults."""
        mc = _make_mc(guardrail_metrics=["fake_a", "fake_b"])
        sanitised, warnings = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert len(sanitised.guardrail_metrics) > 0

    def test_hallucinated_segment_dropped(self, schema_context):
        """Hallucinated segment col is removed."""
        mc = _make_mc(segment_cols=["platform", "nonexistent_segment"])
        sanitised, warnings = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert "nonexistent_segment" not in sanitised.segment_cols
        assert "platform" in sanitised.segment_cols

    def test_hallucinated_events_table(self, schema_context):
        """Hallucinated events_table falls back to default table name."""
        mc = _make_mc(events_table="phantom_table")
        sanitised, warnings = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert sanitised.events_table != "phantom_table"
        assert len(warnings) > 0

    def test_empty_schema_context_no_crash(self):
        """With no schema context, sanitise should return the config unchanged."""
        mc = _make_mc()
        sanitised, warnings = _sanitise_metric_config(mc, "", self.defaults)
        assert sanitised.primary_metric == mc.primary_metric
        assert warnings == []

    def test_mixed_valid_and_hallucinated_guardrails(self, schema_context):
        """Valid guardrails are kept; hallucinated ones are dropped."""
        mc = _make_mc(guardrail_metrics=["notif_optout", "fake1", "d7_retained", "fake2"])
        sanitised, _ = _sanitise_metric_config(mc, schema_context, self.defaults)
        assert "notif_optout" in sanitised.guardrail_metrics
        assert "d7_retained"  in sanitised.guardrail_metrics
        assert "fake1"        not in sanitised.guardrail_metrics
        assert "fake2"        not in sanitised.guardrail_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 3. _apply_intent_to_config — events-table guard & intent overrides
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyIntentToConfig:

    @pytest.fixture(autouse=True)
    def _base(self):
        self.mc = _make_mc()

    def test_valid_intent_updates_primary(self, schema_context):
        """LLM resolves to a valid events column → primary_metric updated."""
        result = {"primary_metric": "d7_retained", "metric_direction": "higher_is_better"}
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        assert updated.primary_metric == "d7_retained"

    def test_timeseries_only_column_blocked(self, schema_context):
        """
        'dau' exists in metrics_daily but NOT in events.
        _apply_intent_to_config must NOT set metric_source_col = 'dau'.
        The original metric_source_col (dau_flag) must be preserved.
        """
        result = {"primary_metric": "dau", "metric_direction": "higher_is_better"}
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        # metric_source_col should remain dau_flag, not be changed to "dau"
        assert updated.metric_source_col == "dau_flag", (
            f"metric_source_col should be 'dau_flag', got '{updated.metric_source_col}'"
        )

    def test_covariate_not_in_events_blocked(self, schema_context):
        """
        LLM proposes covariate='dau' (only in metrics_daily) →
        original covariate kept.
        """
        result = {"primary_metric": "d7_retained", "covariate": "dau"}
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        assert updated.covariate == self.mc.covariate, (
            f"covariate should stay '{self.mc.covariate}', got '{updated.covariate}'"
        )

    def test_covariate_in_events_accepted(self, schema_context):
        """LLM proposes covariate='notif_optout' which IS in events → accepted."""
        result = {"primary_metric": "d7_retained", "covariate": "notif_optout"}
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        assert updated.covariate == "notif_optout"

    def test_unknown_primary_metric_rejected(self, schema_context):
        """LLM hallucinates a metric not in any table → original mc returned."""
        result = {"primary_metric": "totally_fake_metric"}
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        assert updated.primary_metric == self.mc.primary_metric

    def test_empty_result_returns_original(self, schema_context):
        """Empty LLM result returns original mc unchanged."""
        updated = _apply_intent_to_config({}, self.mc, schema_context)
        assert updated.primary_metric == self.mc.primary_metric

    def test_direction_update_preserved(self, schema_context):
        """Valid direction change is applied."""
        result = {"primary_metric": "notif_optout", "metric_direction": "lower_is_better"}
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        assert updated.metric_direction == "lower_is_better"

    def test_invalid_direction_ignored(self, schema_context):
        """Invalid direction string is ignored; original kept."""
        original_dir = self.mc.metric_direction
        result = {"primary_metric": "d7_retained", "metric_direction": "sideways"}
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        assert updated.metric_direction == original_dir

    def test_guardrail_update_filters_schema(self, schema_context):
        """Valid guardrails accepted; hallucinated ones dropped."""
        result = {
            "primary_metric":    "d7_retained",
            "guardrail_metrics": ["notif_optout", "phantom_col"],
        }
        updated = _apply_intent_to_config(result, self.mc, schema_context)
        assert "notif_optout" in updated.guardrail_metrics
        assert "phantom_col"  not in updated.guardrail_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. _columns_for_table and _known_schema_names parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaParsingHelpers:

    def test_known_schema_names_finds_tables(self, schema_context):
        tables, _ = _known_schema_names(schema_context)
        assert "events"       in tables
        assert "experiment"   in tables
        assert "funnel"       in tables
        assert "metrics_daily" in tables

    def test_known_schema_names_finds_columns(self, schema_context):
        _, cols = _known_schema_names(schema_context)
        assert "dau_flag"     in cols
        assert "session_count" in cols
        assert "variant"      in cols

    def test_columns_for_table_events(self, schema_context):
        cols = _columns_for_table(schema_context, "events")
        assert "dau_flag"      in cols
        assert "session_count" in cols
        assert "notif_optout"  in cols
        # metrics_daily-only column must NOT appear
        assert "dau" not in cols

    def test_columns_for_table_metrics_daily(self, schema_context):
        cols = _columns_for_table(schema_context, "metrics_daily")
        assert "dau" in cols
        # events-only column must NOT appear
        assert "dau_flag" not in cols

    def test_columns_for_nonexistent_table_empty(self, schema_context):
        cols = _columns_for_table(schema_context, "ghost_table")
        assert cols == set()

    def test_case_insensitive(self, schema_context):
        cols = _columns_for_table(schema_context, "EVENTS")
        assert "dau_flag" in cols


# ══════════════════════════════════════════════════════════════════════════════
# 5. run_cuped + execute_query with alternate configs
# ══════════════════════════════════════════════════════════════════════════════

class TestRunCupedWithAlternateConfigs:
    """
    Run CUPED on DataFrames built from canonical SQL output for each config.
    Validates no crashes and scalar outputs.
    """

    def _get_df(self, db_conn, mc: MetricConfig) -> pd.DataFrame:
        sql = _canonical_experiment_sql(mc)
        df  = db_conn.execute(sql).df()
        # Apply the same dedup fix that execute_query applies
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        return df

    @pytest.mark.parametrize("mc_kwargs,expected_metric,expected_cov", [
        # Default config
        (
            dict(primary_metric="dau_rate", metric_source_col="dau_flag",
                 covariate="session_count", guardrail_metrics=["notif_optout", "d7_retained"]),
            "dau_rate", "session_count",
        ),
        # d7_retained as primary, notif_optout as covariate
        (
            dict(primary_metric="d7_retained", metric_source_col="d7_retained",
                 covariate="notif_optout", guardrail_metrics=["session_count"]),
            "d7_retained", "notif_optout",
        ),
        # session_count as primary (continuous metric)
        (
            dict(primary_metric="sessions", metric_source_col="session_count",
                 metric_agg="mean", covariate="dau_flag",
                 guardrail_metrics=["notif_optout"]),
            "sessions", "dau_flag",
        ),
    ])
    def test_cuped_runs_to_scalar(self, db_conn, mc_kwargs, expected_metric, expected_cov):
        """CUPED must return a scalar float (no Series ambiguity)."""
        from tools import stats_tools
        mc = _make_mc(**mc_kwargs)
        df = self._get_df(db_conn, mc)

        assert expected_metric in df.columns, f"Missing {expected_metric}"
        assert expected_cov    in df.columns, f"Missing {expected_cov}"

        result = stats_tools.run_cuped(
            df,
            metric_col=expected_metric,
            covariate_col=expected_cov,
            variant_col="variant",
        )
        # Key outputs must be scalar (not Series / DataFrame)
        assert isinstance(result.theta,          float)
        assert isinstance(result.cuped_ate,      float)
        assert isinstance(result.variance_reduction_pct, float)

    def test_cuped_zero_variance_covariate_raises(self, db_conn):
        """Zero-variance covariate raises ValueError, not pandas ambiguity error."""
        from tools import stats_tools
        mc = _make_mc()
        df = self._get_df(db_conn, mc)
        # Force zero variance in covariate
        df = df.copy()
        df["session_count"] = 1.0

        with pytest.raises(ValueError, match="zero variance"):
            stats_tools.run_cuped(df, "dau_rate", "session_count", "variant")

    def test_no_duplicate_columns_after_dedup(self, db_conn):
        """When covariate is in guardrail_metrics, dedup produces unique columns."""
        mc = _make_mc(
            covariate="session_count",
            guardrail_metrics=["session_count", "notif_optout", "d7_retained"],
        )
        df = self._get_df(db_conn, mc)
        assert not df.columns.duplicated().any()
        assert "session_count" in df.columns


# ══════════════════════════════════════════════════════════════════════════════
# 6. Alternate MetricConfig — canonical SQL produces correct aggregation type
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregationTypes:

    def test_mean_agg_produces_fractions_for_binary(self, db_conn):
        """Mean agg of a 0/1 column produces values in [0, 1]."""
        mc  = _make_mc(primary_metric="dau_rate", metric_source_col="dau_flag", metric_agg="mean")
        df  = db_conn.execute(_canonical_experiment_sql(mc)).df()
        assert df["dau_rate"].between(0, 1).all()

    def test_sum_agg_produces_positive_values(self, db_conn):
        """Sum agg of revenue produces non-negative values."""
        mc  = _make_mc(primary_metric="total_rev", metric_source_col="revenue",
                       metric_agg="sum", guardrail_metrics=["notif_optout"])
        df  = db_conn.execute(_canonical_experiment_sql(mc)).df()
        assert (df["total_rev"] >= 0).all()

    def test_count_agg_produces_positive_integers(self, db_conn):
        """Count agg produces positive integer-like values."""
        mc  = _make_mc(primary_metric="n_events", metric_source_col="dau_flag",
                       metric_agg="count", guardrail_metrics=["notif_optout"])
        df  = db_conn.execute(_canonical_experiment_sql(mc)).df()
        assert (df["n_events"] >= 1).all()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Fully custom schema — sales/revenue table with non-standard column names
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sales_db(tmp_path_factory):
    """
    Completely different schema: a sales experiment with:
      - transactions table (not 'events')
      - orders table (not 'experiment')
      - start_date column (not 'assignment_date')
      - customer_id (not 'user_id')
      - order_date (not 'date')
      - arm (not 'variant')
      - exp_week (not 'week')
      - revenue, units_sold, refunded columns
    """
    rng  = np.random.default_rng(7)
    path = str(tmp_path_factory.mktemp("sales") / "sales.db")
    con  = duckdb.connect(path)

    n = 150
    customer_ids = [f"cust_{i:04d}" for i in range(n)]
    arms = ["control"] * 75 + ["treatment"] * 75

    pre_dates = pd.date_range("2024-01-01", periods=14).tolist()
    exp_dates = pd.date_range("2024-03-01", periods=14).tolist()

    # ── transactions ──────────────────────────────────────────────────────────
    tx_rows = []
    for cid, arm in zip(customer_ids, arms):
        baseline_rev = rng.exponential(20.0)
        for date in pre_dates + exp_dates:
            lift = 5.0 if (arm == "treatment" and date >= pd.Timestamp("2024-03-01")) else 0.0
            revenue = max(0.0, baseline_rev + lift + rng.normal(0, 3))
            tx_rows.append({
                "customer_id":  cid,
                "order_date":   date.date(),
                "region":       rng.choice(["us", "eu", "apac"]),
                "customer_tier":rng.choice(["gold", "silver", "bronze"]),
                "revenue":      round(revenue, 2),
                "units_sold":   max(0, int(rng.poisson(2))),
                "refunded":     int(rng.random() < 0.05),
            })
    tx_df = pd.DataFrame(tx_rows)
    con.execute("CREATE TABLE transactions AS SELECT * FROM tx_df")

    # ── orders (experiment assignment) ────────────────────────────────────────
    ord_rows = []
    for cid, arm in zip(customer_ids, arms):
        for exp_week in [1, 2]:
            ord_rows.append({
                "customer_id": cid,
                "arm":         arm,
                "start_date":  pd.Timestamp("2024-03-01").date(),
                "exp_week":    exp_week,
            })
    ord_df = pd.DataFrame(ord_rows)
    con.execute("CREATE TABLE orders AS SELECT * FROM ord_df")

    con.close()
    yield path


class TestSalesSchema:
    """
    End-to-end canonical SQL + CUPED on a fully custom schema
    with non-standard table and column names.
    """

    def _mc(self) -> MetricConfig:
        return MetricConfig(
            primary_metric="revenue_per_customer",
            metric_source_col="revenue",
            metric_agg="mean",
            metric_direction="higher_is_better",
            covariate="units_sold",
            events_table="transactions",
            experiment_table="orders",
            timeseries_table=None,
            funnel_table=None,
            user_id_col="customer_id",
            date_col="order_date",
            variant_col="arm",
            week_col="exp_week",
            assignment_date_col="start_date",   # ← non-standard column name
            guardrail_metrics=["refunded"],
            segment_cols=["region", "customer_tier"],
            funnel_steps=[],
            revenue_per_unit=20.0,
            baseline_unit_count=10000,
            experiment_weeks=2,
        )

    def test_canonical_sql_executes(self, sales_db):
        """Canonical SQL runs without error on a fully custom schema."""
        con = duckdb.connect(sales_db, read_only=True)
        mc  = self._mc()
        sql = _canonical_experiment_sql(mc)
        df  = con.execute(sql).df()
        con.close()
        assert len(df) > 0
        assert "revenue_per_customer" in df.columns
        assert "units_sold"           in df.columns
        assert "arm"                  not in df.columns   # aliased to 'variant'
        assert "variant"              in df.columns
        assert "refunded"             in df.columns

    def test_both_arms_present(self, sales_db):
        """Both control and treatment appear in result."""
        con = duckdb.connect(sales_db, read_only=True)
        df  = con.execute(_canonical_experiment_sql(self._mc())).df()
        con.close()
        assert set(df["variant"].unique()) == {"control", "treatment"}

    def test_revenue_values_positive(self, sales_db):
        """Revenue per customer is positive."""
        con = duckdb.connect(sales_db, read_only=True)
        df  = con.execute(_canonical_experiment_sql(self._mc())).df()
        con.close()
        assert (df["revenue_per_customer"] >= 0).all()

    def test_cuped_on_revenue(self, sales_db):
        """CUPED runs end-to-end on revenue with units_sold as covariate."""
        from tools import stats_tools
        con = duckdb.connect(sales_db, read_only=True)
        mc  = self._mc()
        df  = con.execute(_canonical_experiment_sql(mc)).df()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        con.close()

        result = stats_tools.run_cuped(
            df,
            metric_col="revenue_per_customer",
            covariate_col="units_sold",
            variant_col="variant",
        )
        assert isinstance(result.cuped_ate, float)
        assert isinstance(result.variance_reduction_pct, float)

    def test_non_standard_assignment_date_col(self, sales_db):
        """
        'start_date' (not 'assignment_date') correctly splits pre/post data.
        Pre-exp CTE should only contain rows before 2024-03-01.
        """
        con = duckdb.connect(sales_db, read_only=True)
        mc  = self._mc()
        sql = _canonical_experiment_sql(mc)
        # SQL must reference 'start_date', not 'assignment_date'
        assert "start_date"      in sql
        assert "assignment_date" not in sql
        df  = con.execute(sql).df()
        con.close()
        assert len(df) > 0
