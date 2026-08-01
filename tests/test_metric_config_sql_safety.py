"""
SQL identifier safety for LLM-inferred MetricConfig names.

The analysis nodes interpolate table and column names from MetricConfig into
DuckDB SQL. Those names are produced by an LLM reading the user's schema, and
_sanitise_metric_config is not a whitelist: it returns the config untouched when
schema_context is empty or unparseable (node_shared.py), which happens whenever
schema inference fails.
"""
import duckdb
import pandas as pd
import pytest

from agents.analyze.node_shared import _canonical_experiment_sql
from agents.analyze.nodes_analysis import (
    _aggregate_daily_from_events,
    run_power_analysis_node,
)
from config.analysis_config import MetricConfig


def _mc(**over) -> MetricConfig:
    base = dict(
        primary_metric="conversion_rate",
        metric_source_col="converted",
        covariate="sessions",
        metric_direction="higher_is_better",
        guardrail_metrics=["refund_rate"],
        segment_cols=["device"],
    )
    base.update(over)
    return MetricConfig(**base)


class _CapturingDB:
    """Captures SQL and runs it against a real DuckDB so syntax is validated."""

    def __init__(self, con=None):
        self.sql: list[str] = []
        self._con = con

    def query(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        if self._con is None:
            return pd.DataFrame()
        return self._con.execute(sql).df()


# ── Canonical experiment SQL ──────────────────────────────────────────────────


def test_canonical_sql_quotes_identifiers():
    sql = _canonical_experiment_sql(_mc())
    assert '"converted"' in sql
    assert '"conversion_rate"' in sql
    assert '"device"' in sql


def test_canonical_sql_neutralises_a_hostile_column_name():
    """A name carrying a payload must land inside quotes, not as SQL."""
    payload = 'converted) AS x FROM secrets --'
    sql = _canonical_experiment_sql(_mc(metric_source_col=payload))

    # Doubling makes the payload inert; it never terminates the identifier.
    assert f'"{payload}"' in sql
    assert "FROM secrets --" not in sql.replace(f'"{payload}"', "")


def test_canonical_sql_escapes_embedded_quotes():
    sql = _canonical_experiment_sql(_mc(primary_metric='we"ird'))
    assert '"we""ird"' in sql


def test_canonical_sql_quotes_every_segment_and_guardrail():
    sql = _canonical_experiment_sql(
        _mc(segment_cols=["device", "region"], guardrail_metrics=["refund_rate", "latency_ms"])
    )
    for name in ("device", "region", "refund_rate", "latency_ms"):
        assert f'"{name}"' in sql


def test_count_aggregation_does_not_require_a_source_column():
    """metric_agg='count' never references metric_source_col, so an empty one
    must not start raising where it previously worked."""
    sql = _canonical_experiment_sql(_mc(metric_agg="count", metric_source_col=""))
    assert "COUNT(*)" in sql


# ── Executed against real DuckDB ──────────────────────────────────────────────


@pytest.fixture
def hostile_con():
    con = duckdb.connect()
    con.execute('CREATE TABLE events ("odd""col" INTEGER, user_id INTEGER, "date" DATE)')
    con.execute("INSERT INTO events VALUES (1, 1, DATE '2024-01-01'), (2, 2, DATE '2024-01-02')")
    con.execute("CREATE TABLE secrets (token VARCHAR)")
    con.execute("INSERT INTO secrets VALUES ('S1'), ('S2'), ('S3')")
    return con


def test_daily_aggregation_runs_with_a_quoted_column(hostile_con):
    db = _CapturingDB(hostile_con)
    out = _aggregate_daily_from_events(
        db, _mc(metric_source_col='odd"col', primary_metric="m", date_col="date",
                events_table="events", segment_cols=[])
    )
    assert '"odd""col"' in db.sql[0]
    assert len(out) == 2


def test_injected_from_clause_does_not_execute(hostile_con):
    """Old behaviour: a crafted column rewrote FROM and read another table."""
    db = _CapturingDB(hostile_con)
    payload = "user_id) FROM secrets --"

    with pytest.raises(Exception):
        # The payload is now a column name that does not exist, so DuckDB
        # rejects it instead of silently reading `secrets`.
        _aggregate_daily_from_events(
            db, _mc(metric_source_col=payload, primary_metric="m", date_col="date",
                    events_table="events", segment_cols=[])
        )

    assert "FROM secrets" not in db.sql[0].replace(f'"{payload}"', "")


# ── Power analysis ────────────────────────────────────────────────────────────


def test_power_analysis_sql_quotes_identifiers(monkeypatch):
    db = _CapturingDB()

    def _fake_query(sql):
        db.sql.append(sql)
        return pd.DataFrame([{
            "baseline_mean": 0.2, "baseline_std": 0.4,
            "total_users": 10_000, "total_days": 20,
        }])

    db.query = _fake_query
    monkeypatch.setattr("agents.analyze.nodes_analysis._db_conn", lambda _s: db)

    run_power_analysis_node({
        "metric_config": _mc(),
        "metric": "conversion_rate",
        "power_mde_target_pct": 5.0,
    })

    sql = db.sql[0]
    assert '"converted"' in sql
    assert '"user_id"' in sql
    assert '"events"' in sql
