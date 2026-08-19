"""
tests/test_pushdown_postgres_integration.py — the warehouse pushdown on a real
Postgres produces the same statistics as the pandas path.

tests/test_pushdown.py proves parity on DuckDB, where the demo data lives.
The SQL builders are shared across dialects, but DuckDB is permissive
(implicit casts, CASE over mixed types) and Postgres is not — this loads the
demo experiment into the CI Postgres service and repeats the parity checks
there, including BOOLEAN metric/guardrail/completed columns, which is the
case the CAST/CASE forms exist for.
"""
from __future__ import annotations

import csv
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytestmark = pytest.mark.integration

from tools import funnel_tools, guardrail_tools, pushdown, stats_tools  # noqa: E402
from tools.db_tools import DBConnection  # noqa: E402

DEMO_DB = os.path.join(ROOT, "data", "dau_experiment.db")

_PG_TYPES = {"VARCHAR": "text", "DATE": "date", "BIGINT": "bigint",
             "INTEGER": "integer", "DOUBLE": "double precision", "BOOLEAN": "boolean"}


def _pg_config() -> dict:
    if os.getenv("DATAPILOT_POSTGRES_INTEGRATION") != "1":
        pytest.skip("DATAPILOT_POSTGRES_INTEGRATION=1 is not configured")
    return {
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", "datapilot_test"),
        "user": os.getenv("PGUSER", "datapilot"),
        "password": os.getenv("PGPASSWORD", "datapilot"),
    }


@pytest.fixture(scope="module")
def pg(tmp_path_factory):
    """Copy experiment/events/funnel from the demo DuckDB into Postgres as
    dp_push_* tables, with a few BOOLEAN columns added alongside."""
    psycopg2 = pytest.importorskip("psycopg2")
    duckdb = pytest.importorskip("duckdb")
    if not os.path.exists(DEMO_DB):
        pytest.skip("demo DuckDB not present — run data/generate_data.py")
    cfg = _pg_config()
    tmp = tmp_path_factory.mktemp("pgload")

    src = duckdb.connect(DEMO_DB, read_only=True)
    admin = psycopg2.connect(**cfg)
    admin.autocommit = True
    try:
        for table in ("experiment", "events", "funnel"):
            cols = src.execute(f"DESCRIBE {table}").fetchall()
            ddl = ", ".join(f'"{c[0]}" {_PG_TYPES[c[1]]}' for c in cols)
            path = str(tmp / f"{table}.csv")
            src.execute(f"COPY (SELECT * FROM {table}) TO '{path}' (HEADER, DELIMITER ',')")
            with admin.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS dp_push_{table}")
                cur.execute(f"CREATE TABLE dp_push_{table} ({ddl})")
                with open(path, newline="") as fh:
                    cur.copy_expert(f"COPY dp_push_{table} FROM STDIN WITH CSV HEADER", fh)
        with admin.cursor() as cur:
            # BOOLEAN twins of the integer flags — what read_csv_auto makes of a
            # true/false CSV column, and what broke `1.0 * col` on Postgres.
            cur.execute("ALTER TABLE dp_push_events ADD COLUMN dau_bool boolean, ADD COLUMN optout_bool boolean")
            cur.execute("UPDATE dp_push_events SET dau_bool = (dau_flag = 1), optout_bool = (notif_optout = 1)")
            cur.execute("ALTER TABLE dp_push_funnel ADD COLUMN completed_bool boolean")
            cur.execute("UPDATE dp_push_funnel SET completed_bool = (completed = 1)")
    finally:
        admin.close()
        src.close()
    return DBConnection("postgres", **cfg)


BASE_SQL = """
SELECT e.user_id,
       ex.variant                       AS variant,
       ex.week                          AS week,
       AVG(1.0 * e.dau_flag)            AS dau_flag,
       AVG(1.0 * e.session_count)       AS pre_session_count,
       AVG(1.0 * e.notif_optout)        AS notif_optout,
       e.platform                       AS platform,
       e.user_segment                   AS user_segment
FROM dp_push_experiment ex
JOIN dp_push_events e ON e.user_id = ex.user_id
GROUP BY e.user_id, ex.variant, ex.week, e.platform, e.user_segment
"""

BOOL_SQL = """
SELECT e.user_id, ex.variant AS variant, ex.week AS week,
       BOOL_OR(e.optout_bool) AS converted,
       AVG(1.0 * e.session_count) AS pre,
       BOOL_AND(e.dau_bool)   AS churned
FROM dp_push_experiment ex
JOIN dp_push_events e ON e.user_id = ex.user_id
GROUP BY e.user_id, ex.variant, ex.week
"""

FUNNEL_SQL = """
SELECT f.user_id, ex.variant AS variant, f.step, f.completed_bool AS completed
FROM   dp_push_funnel f
JOIN   dp_push_experiment ex ON f.user_id = ex.user_id AND ex.week = 1
"""
FUNNEL_STEPS = ["impression", "click", "install", "d1_retain"]


def test_nested_count_and_probe_on_postgres(pg):
    n = pg.count_rows(BASE_SQL + "; -- analyst note")
    assert n == 20000
    assert "dau_flag" in pushdown.probe_columns(pg, BASE_SQL)


def test_ttest_cuped_guardrails_match_pandas_on_postgres(pg):
    frame = pg.query(BASE_SQL)
    ss = pushdown.compute_sufficient_stats(
        pg, BASE_SQL, metric="dau_flag", covariate="pre_session_count",
        segment_cols=["platform", "user_segment"], guardrail_metrics=["notif_optout"],
        total_rows=len(frame), entity_col="user_id",
    )
    ctrl = frame[frame["variant"] == "control"]["dau_flag"].astype(float).dropna()
    trt  = frame[frame["variant"] == "treatment"]["dau_flag"].astype(float).dropna()
    ref = stats_tools.run_ttest(ctrl, trt, alternative="greater")
    got = pushdown.ttest_from_stats(ss, alternative="greater")
    assert got.n_control == ref.n_control and got.n_treatment == ref.n_treatment
    assert got.t_stat == pytest.approx(ref.t_stat, abs=1e-3)
    assert got.p_value == pytest.approx(ref.p_value, abs=1e-5)
    assert got.control_mean == pytest.approx(ref.control_mean, abs=1e-5)

    f2 = frame.copy()
    for c in ("dau_flag", "pre_session_count", "notif_optout"):
        f2[c] = f2[c].astype(float)
    cref = stats_tools.run_cuped(f2, metric_col="dau_flag", covariate_col="pre_session_count",
                                 variant_col="variant")
    cgot = pushdown.cuped_from_stats(ss)
    assert cgot.theta == pytest.approx(cref.theta, abs=1e-4)
    assert cgot.cuped_ate == pytest.approx(cref.cuped_ate, abs=1e-5)

    gref = guardrail_tools.check_guardrails(f2, variant_col="variant",
                                            guardrail_metrics=["notif_optout"],
                                            default_direction="increase")
    ggot = pushdown.guardrails_from_stats(ss, default_direction="increase")
    assert ggot.guardrails[0].p_value == pytest.approx(gref.guardrails[0].p_value, abs=1e-4)

    # Shape facts for content validation came back from the warehouse.
    assert ss.n_entities == frame["user_id"].nunique()
    assert ss.n_weeks == frame["week"].nunique()


def test_boolean_columns_push_down_on_postgres(pg):
    frame = pg.query(BOOL_SQL)
    ss = pushdown.compute_sufficient_stats(
        pg, BOOL_SQL, metric="converted", covariate="pre",
        segment_cols=[], guardrail_metrics=["churned"], total_rows=len(frame),
    )
    ctrl = frame[frame["variant"] == "control"]["converted"].astype(float)
    trt  = frame[frame["variant"] == "treatment"]["converted"].astype(float)
    ref = stats_tools.run_ttest(ctrl, trt)
    got = pushdown.ttest_from_stats(ss)
    assert got.control_mean == pytest.approx(ref.control_mean, abs=1e-9)
    assert got.treatment_mean == pytest.approx(ref.treatment_mean, abs=1e-9)
    assert got.p_value == pytest.approx(ref.p_value, abs=1e-9)
    g = pushdown.guardrails_from_stats(ss)
    assert g.guardrails[0].metric == "churned"


def test_boolean_funnel_matches_pandas_on_postgres(pg):
    frame = pg.query(FUNNEL_SQL)
    ref = funnel_tools.compute_funnel(frame.assign(completed=frame["completed"].astype(int)),
                                      variant_col="variant", steps=FUNNEL_STEPS)
    got = pushdown.funnel_from_warehouse(pg, FUNNEL_SQL, FUNNEL_STEPS)
    assert got is not None
    for g, r in zip(got.steps, ref.steps):
        assert g.step == r.step
        assert g.control_rate == pytest.approx(r.control_rate, abs=1e-4), g.step
        assert g.treatment_rate == pytest.approx(r.treatment_rate, abs=1e-4), g.step
        assert g.p_value == pytest.approx(r.p_value, abs=1e-4), g.step
