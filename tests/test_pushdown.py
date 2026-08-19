"""
tests/test_pushdown.py — warehouse pushdown produces the same statistics as
the pandas path, on the real demo experiment data.

This is the load-bearing test for the pushdown: every result model
(t-test, CUPED, SRM inputs, HTE, novelty, guardrails) is computed twice —
once by the existing pandas tools over the materialized frame, once from
in-warehouse sufficient statistics — and the two must agree to float
tolerance. If a formula in tools/pushdown.py drifts from its counterpart,
this fails naming the exact field.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import guardrail_tools, novelty_tools, pushdown, stats_tools
from tools.db_tools import DBConnection

DEMO_DB = os.path.join(ROOT, "data", "dau_experiment.db")

# The canonical user-level extract, mirroring what the approved SQL produces.
BASE_SQL = """
SELECT e.user_id,
       ex.variant                       AS variant,
       ex.week                          AS week,
       AVG(1.0 * e.dau_flag)            AS dau_flag,
       AVG(1.0 * e.session_count)       AS pre_session_count,
       AVG(1.0 * e.notif_optout)        AS notif_optout,
       e.platform                       AS platform,
       e.user_segment                   AS user_segment
FROM experiment ex
JOIN events e ON e.user_id = ex.user_id
GROUP BY e.user_id, ex.variant, ex.week, e.platform, e.user_segment
"""

METRIC, COVARIATE = "dau_flag", "pre_session_count"
SEGMENTS = ["platform", "user_segment"]
GUARDRAILS = ["notif_optout"]


@pytest.fixture(scope="module")
def db():
    if not os.path.exists(DEMO_DB):
        pytest.skip("demo DuckDB not present")
    return DBConnection("duckdb", path=DEMO_DB)


@pytest.fixture(scope="module")
def frame(db):
    return db.query(BASE_SQL)


@pytest.fixture(scope="module")
def stats(db, frame):
    return pushdown.compute_sufficient_stats(
        db, BASE_SQL,
        metric=METRIC, covariate=COVARIATE,
        segment_cols=SEGMENTS, guardrail_metrics=GUARDRAILS,
        total_rows=len(frame),
    )


def test_ttest_matches(frame, stats):
    ctrl = frame[frame["variant"] == "control"][METRIC].dropna()
    trt  = frame[frame["variant"] == "treatment"][METRIC].dropna()
    ref  = stats_tools.run_ttest(ctrl, trt, alternative="greater")
    got  = pushdown.ttest_from_stats(stats, alternative="greater")

    assert got.n_control == ref.n_control
    assert got.n_treatment == ref.n_treatment
    assert got.t_stat == pytest.approx(ref.t_stat, abs=1e-3)
    assert got.p_value == pytest.approx(ref.p_value, abs=1e-5)
    assert got.ci_lower == pytest.approx(ref.ci_lower, abs=1e-5)
    assert got.ci_upper == pytest.approx(ref.ci_upper, abs=1e-5)
    assert got.cohens_d == pytest.approx(ref.cohens_d, abs=1e-3)
    assert got.control_mean == pytest.approx(ref.control_mean, abs=1e-5)
    assert got.treatment_mean == pytest.approx(ref.treatment_mean, abs=1e-5)
    assert got.significant == ref.significant


def test_skewness_warning_matches(frame, stats):
    """The skewness gate depends on Σy³ surviving the SQL round trip."""
    ctrl = frame[frame["variant"] == "control"][METRIC].dropna()
    trt  = frame[frame["variant"] == "treatment"][METRIC].dropna()
    ref  = stats_tools.run_ttest(ctrl, trt)
    got  = pushdown.ttest_from_stats(stats)
    assert (got.skewness_warning is None) == (ref.skewness_warning is None)


def test_cuped_matches(frame, stats):
    ref = stats_tools.run_cuped(frame, metric_col=METRIC,
                                covariate_col=COVARIATE, variant_col="variant")
    got = pushdown.cuped_from_stats(stats)
    assert got.raw_ate == pytest.approx(ref.raw_ate, abs=1e-5)
    assert got.cuped_ate == pytest.approx(ref.cuped_ate, abs=1e-5)
    assert got.theta == pytest.approx(ref.theta, abs=1e-4)
    assert got.variance_reduction_pct == pytest.approx(ref.variance_reduction_pct, abs=0.05)


def test_hte_matches(frame, stats):
    ref = stats_tools.run_hte(frame, metric_col=METRIC, variant_col="variant",
                              segment_cols=SEGMENTS)
    got = pushdown.hte_from_stats(stats)

    assert got.top_segment == ref.top_segment
    assert got.effect_size == pytest.approx(ref.effect_size, abs=1e-5)
    assert got.segment_share == pytest.approx(ref.segment_share, abs=1e-3)
    assert len(got.all_segments) == len(ref.all_segments)
    ref_by_seg = {s.segment: s for s in ref.all_segments}
    for seg in got.all_segments:
        r = ref_by_seg[seg.segment]
        assert seg.n_control == r.n_control, seg.segment
        assert seg.n_treatment == r.n_treatment, seg.segment
        assert seg.effect_size == pytest.approx(r.effect_size, abs=1e-5), seg.segment
        assert seg.p_value == pytest.approx(r.p_value, abs=1e-4), seg.segment
        assert seg.significant == r.significant, seg.segment


def test_novelty_matches(frame, stats):
    ref = novelty_tools.detect_novelty_effect(frame, metric_col=METRIC,
                                              variant_col="variant", week_col="week")
    got = pushdown.novelty_from_stats(stats)
    assert got.week1_ate == pytest.approx(ref.week1_ate, abs=1e-5)
    assert got.week2_ate == pytest.approx(ref.week2_ate, abs=1e-5)
    assert got.effect_direction == ref.effect_direction
    assert got.novelty_likely == ref.novelty_likely


def test_guardrails_match(frame, stats):
    ref = guardrail_tools.check_guardrails(
        frame, variant_col="variant", guardrail_metrics=GUARDRAILS,
        default_direction="increase",
    )
    got = pushdown.guardrails_from_stats(stats, default_direction="increase")
    assert len(got.guardrails) == len(ref.guardrails)
    for g, r in zip(got.guardrails, ref.guardrails):
        assert g.metric == r.metric
        assert g.control_mean == pytest.approx(r.control_mean, abs=1e-5)
        assert g.treatment_mean == pytest.approx(r.treatment_mean, abs=1e-5)
        assert g.delta_pct == pytest.approx(r.delta_pct, abs=0.02)
        assert g.p_value == pytest.approx(r.p_value, abs=1e-4)
        assert g.breached == r.breached
    assert got.any_breached == ref.any_breached


def test_preview_frame_carries_plumbing_columns(stats):
    df = pushdown.preview_frame(stats)
    assert {"variant", METRIC, COVARIATE, "n_users"} <= set(df.columns)
    assert set(df["variant"]) == {"control", "treatment"}


def test_stats_survive_the_checkpoint_round_trip(stats):
    """Pushdown state must checkpoint: serialize + deserialize intact."""
    from agents.analyze.checkpoint_serde import SafeCheckpointSerde
    s = SafeCheckpointSerde()
    blob = s.dumps_typed({"sufficient_stats": stats})
    restored = s.loads_typed(blob)["sufficient_stats"]
    assert restored == stats
    # And it is small — the entire point.
    n = len(blob[1]) if isinstance(blob, tuple) else len(blob)
    assert n < 200_000, f"sufficient stats blob unexpectedly large: {n} bytes"


FUNNEL_SQL = """
SELECT f.user_id, ex.variant AS variant, f.step, f.completed
FROM   funnel f
JOIN   experiment ex ON f.user_id = ex.user_id AND ex.week = 1
"""
FUNNEL_STEPS = ["impression", "click", "install", "d1_retain"]


def test_funnel_matches(db):
    from tools import funnel_tools
    frame = db.query(FUNNEL_SQL)
    ref = funnel_tools.compute_funnel(frame, variant_col="variant", steps=FUNNEL_STEPS)
    got = pushdown.funnel_from_warehouse(db, FUNNEL_SQL, FUNNEL_STEPS)

    assert got is not None
    assert got.biggest_dropoff_step == ref.biggest_dropoff_step
    assert len(got.steps) == len(ref.steps)
    for g, r in zip(got.steps, ref.steps):
        assert g.step == r.step
        assert g.control_rate == pytest.approx(r.control_rate, abs=1e-4), g.step
        assert g.treatment_rate == pytest.approx(r.treatment_rate, abs=1e-4), g.step
        assert g.delta == pytest.approx(r.delta, abs=1e-4), g.step
        assert g.p_value == pytest.approx(r.p_value, abs=1e-4), g.step
        assert g.significant == r.significant, g.step


def test_funnel_returns_none_below_two_steps(db):
    got = pushdown.funnel_from_warehouse(db, FUNNEL_SQL, ["impression", "not_a_step"])
    assert got is None


def test_funnel_step_literal_escaping(db):
    """A step value with a quote must not break (or escape) the pivot SQL."""
    got = pushdown.funnel_from_warehouse(
        db, FUNNEL_SQL, ["impression", "click", "o'brien's step"]
    )
    # The odd step just isn't present; the two real ones still compute.
    assert got is not None
    assert [s.step for s in got.steps] == ["impression", "click"]


# ── Type coercion: BOOLEAN and wide-integer columns ──────────────────────────
# `1.0 * col` was DECIMAL arithmetic (Σy³ of a BIGINT overflowed the
# DECIMAL(38) accumulator) and a binder error on BOOLEAN — which is what
# read_csv_auto makes of a true/false CSV flag. The pandas path coerced both.

@pytest.fixture
def typed_db(tmp_path):
    import duckdb
    path = str(tmp_path / "typed.db")
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE users (
            user_id   VARCHAR,
            variant   VARCHAR,
            converted BOOLEAN,
            spend     BIGINT,
            sessions  INTEGER,
            churned   BOOLEAN,
            week      INTEGER
        )
    """)
    rows = []
    for i in range(40):
        v = "control" if i % 2 == 0 else "treatment"
        rows.append((f"u{i}", v, (i % 3 == 0) or (v == "treatment" and i % 5 == 0),
                     500_000_000_000 + i * 1_000_000_000, i % 7, i % 4 == 0, i % 2))
    con.executemany("INSERT INTO users VALUES (?,?,?,?,?,?,?)", rows)
    con.close()
    return DBConnection("duckdb", path=path)


TYPED_SQL = "SELECT user_id, variant, converted, spend, sessions, churned, week FROM users"


def test_boolean_metric_and_guardrail_push_down(typed_db):
    frame = typed_db.query(TYPED_SQL)
    ss = pushdown.compute_sufficient_stats(
        typed_db, TYPED_SQL,
        metric="converted", covariate="sessions",
        segment_cols=[], guardrail_metrics=["churned"], total_rows=len(frame),
    )
    got = pushdown.ttest_from_stats(ss)
    ctrl = frame[frame["variant"] == "control"]["converted"].astype(float)
    trt  = frame[frame["variant"] == "treatment"]["converted"].astype(float)
    ref = stats_tools.run_ttest(ctrl, trt)
    assert got.control_mean == pytest.approx(ref.control_mean, abs=1e-9)
    assert got.treatment_mean == pytest.approx(ref.treatment_mean, abs=1e-9)
    assert got.p_value == pytest.approx(ref.p_value, abs=1e-9)

    g = pushdown.guardrails_from_stats(ss)
    r = guardrail_tools.check_guardrails(
        frame.assign(churned=frame["churned"].astype(float)),
        variant_col="variant", guardrail_metrics=["churned"],
    )
    assert g.guardrails[0].control_mean == pytest.approx(r.guardrails[0].control_mean, abs=1e-9)
    assert g.guardrails[0].treatment_mean == pytest.approx(r.guardrails[0].treatment_mean, abs=1e-9)


def test_wide_bigint_metric_does_not_overflow(typed_db):
    """Σy³ of ~5e11 values: DECIMAL(38) overflowed (HUGEINT), DOUBLE must not."""
    frame = typed_db.query(TYPED_SQL)
    ss = pushdown.compute_sufficient_stats(
        typed_db, TYPED_SQL,
        metric="spend", covariate="sessions",
        segment_cols=[], guardrail_metrics=[], total_rows=len(frame),
    )
    got = pushdown.ttest_from_stats(ss)
    ctrl = frame[frame["variant"] == "control"]["spend"].astype(float)
    trt  = frame[frame["variant"] == "treatment"]["spend"].astype(float)
    ref = stats_tools.run_ttest(ctrl, trt)
    assert got.control_mean == pytest.approx(ref.control_mean, rel=1e-12)
    assert got.p_value == pytest.approx(ref.p_value, rel=1e-9)


def test_boolean_completed_funnel(tmp_path):
    import duckdb
    from tools import funnel_tools
    path = str(tmp_path / "funnel.db")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE f (user_id VARCHAR, variant VARCHAR, step VARCHAR, completed BOOLEAN)")
    rows = []
    for i in range(60):
        v = "control" if i % 2 == 0 else "treatment"
        rows.append((f"u{i}", v, "view", True))
        rows.append((f"u{i}", v, "click", i % 3 == 0 or (v == "treatment" and i % 4 == 0)))
    con.executemany("INSERT INTO f VALUES (?,?,?,?)", rows)
    con.close()
    db = DBConnection("duckdb", path=path)
    sql = "SELECT user_id, variant, step, completed FROM f"
    ref = funnel_tools.compute_funnel(db.query(sql), variant_col="variant", steps=["view", "click"])
    got = pushdown.funnel_from_warehouse(db, sql, ["view", "click"])
    assert got is not None
    for g, r in zip(got.steps, ref.steps):
        assert g.control_rate == pytest.approx(r.control_rate, abs=1e-4), g.step
        assert g.treatment_rate == pytest.approx(r.treatment_rate, abs=1e-4), g.step


# ── Nesting the approved SQL ──────────────────────────────────────────────────

def test_count_rows_tolerates_trailing_comment_after_semicolon(typed_db):
    """`SELECT …; -- note` runs standalone; it must count (and nest) too."""
    assert typed_db.count_rows(TYPED_SQL + "; -- analyst note\n/* end */") == 40


def test_probe_columns_tolerates_trailing_comment(typed_db):
    cols = pushdown.probe_columns(typed_db, TYPED_SQL + "; -- note")
    assert "converted" in cols


# ── Content validation in pushdown mode ──────────────────────────────────────
# The frame-based checks (JOIN fan-out, arm imbalance, rate-vs-percentage)
# used to be skipped entirely once the extract went in-warehouse — and an
# event-level extract from a missing GROUP BY is the canonical reason a
# result balloons past the pushdown threshold in the first place.

@pytest.fixture
def fanout_db(tmp_path):
    import duckdb
    path = str(tmp_path / "fanout.db")
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE ev (user_id VARCHAR, variant VARCHAR, week INTEGER,
                         conversion_rate DOUBLE, pre DOUBLE)
    """)
    rows = []
    for i in range(30):
        v = "control" if i % 2 == 0 else "treatment"
        for k in range(10):   # 10 event rows per user, one week → 10× fan-out
            rows.append((f"u{i}", v, 1, 0.2 + 0.01 * (k % 3) + (0.05 if v == "treatment" else 0), float(i % 5)))
    con.executemany("INSERT INTO ev VALUES (?,?,?,?,?)", rows)
    con.close()
    return DBConnection("duckdb", path=path)


def test_sufficient_stats_carry_shape_facts(fanout_db):
    sql = "SELECT user_id, variant, week, conversion_rate, pre FROM ev"
    ss = pushdown.compute_sufficient_stats(
        fanout_db, sql, metric="conversion_rate", covariate="pre",
        segment_cols=[], guardrail_metrics=[], total_rows=300, entity_col="user_id",
    )
    assert ss.entity_col == "user_id"
    assert ss.n_entities == 30
    assert ss.n_weeks == 1
    assert ss.metric_over_one == 0


def test_pushdown_fan_out_is_flagged_at_execute_query(fanout_db, monkeypatch):
    import agents.analyze.nodes_sql as ns
    from config.analysis_config import load_metric_config

    monkeypatch.setattr(ns, "_PUSHDOWN_ROWS", 50)          # 300 rows → pushdown
    monkeypatch.setattr(ns, "_db_conn", lambda state: fanout_db)
    mc = load_metric_config().model_copy(update={
        "primary_metric": "conversion_rate", "covariate": "pre",
        "segment_cols": [], "guardrail_metrics": [],
    })
    state = {
        "generated_sql":  "SELECT user_id, variant, week, conversion_rate, pre FROM ev",
        "analysis_mode":  "ab_test",
        "metric_config":  mc,
        "metric":         "conversion_rate",
        "covariate":      "pre",
        "schema_context": "TABLE: ev\nuser_id VARCHAR\n",
        "task":           "did conversion move?",
    }
    out = ns.execute_query(state)
    assert out["sufficient_stats"] is not None
    warnings = out["sql_validation_warnings"]
    assert any("JOIN fan-out" in w and "30 unique user_ids" in w for w in warnings), warnings


def test_pushdown_clean_extract_has_no_content_warnings(typed_db, monkeypatch):
    import agents.analyze.nodes_sql as ns
    from config.analysis_config import load_metric_config

    monkeypatch.setattr(ns, "_PUSHDOWN_ROWS", 10)           # 40 rows → pushdown
    monkeypatch.setattr(ns, "_db_conn", lambda state: typed_db)
    mc = load_metric_config().model_copy(update={
        "primary_metric": "converted", "covariate": "sessions",
        "segment_cols": [], "guardrail_metrics": ["churned"],
    })
    state = {
        "generated_sql":  TYPED_SQL,
        "analysis_mode":  "ab_test",
        "metric_config":  mc,
        "schema_context": "TABLE: users\nuser_id VARCHAR\n",
        "task":           "did conversion move?",
    }
    out = ns.execute_query(state)
    assert out["sufficient_stats"] is not None
    assert out["sql_validation_warnings"] == []


# ── CUPED parity with a third arm; zero-variance guard; SRM from moments ────

@pytest.fixture
def three_arm_db(tmp_path):
    import duckdb
    path = str(tmp_path / "three.db")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE u (user_id VARCHAR, variant VARCHAR, y DOUBLE, x DOUBLE, c DOUBLE)")
    rows = []
    for i in range(90):
        v = ("control", "treatment", "holdout")[i % 3]
        x = float(i % 11)
        y = 2.0 + 0.5 * x + (0.3 if v == "treatment" else 0.0) + (0.7 if v == "holdout" else 0.0) + 0.1 * ((i * 7) % 5)
        rows.append((f"u{i}", v, y, x, 0.1))
    con.executemany("INSERT INTO u VALUES (?,?,?,?,?)", rows)
    con.close()
    return DBConnection("duckdb", path=path)


def test_cuped_matches_pandas_with_a_third_arm(three_arm_db):
    sql = "SELECT user_id, variant, y, x FROM u"
    frame = three_arm_db.query(sql)
    ref = stats_tools.run_cuped(frame, metric_col="y", covariate_col="x", variant_col="variant")
    ss = pushdown.compute_sufficient_stats(
        three_arm_db, sql, metric="y", covariate="x",
        segment_cols=[], guardrail_metrics=[], total_rows=len(frame),
    )
    got = pushdown.cuped_from_stats(ss)
    assert got.theta == pytest.approx(ref.theta, abs=1e-5)
    assert got.cuped_ate == pytest.approx(ref.cuped_ate, abs=1e-5)
    assert got.variance_reduction_pct == pytest.approx(ref.variance_reduction_pct, abs=1e-2)


def test_cuped_constant_covariate_raises_like_pandas(three_arm_db):
    sql = "SELECT user_id, variant, y, c FROM u"
    frame = three_arm_db.query(sql)
    with pytest.raises(ValueError, match="zero variance"):
        stats_tools.run_cuped(frame, metric_col="y", covariate_col="c", variant_col="variant")
    ss = pushdown.compute_sufficient_stats(
        three_arm_db, sql, metric="y", covariate="c",
        segment_cols=[], guardrail_metrics=[], total_rows=len(frame),
    )
    with pytest.raises(ValueError, match="zero variance"):
        pushdown.cuped_from_stats(ss)


def test_srm_node_reads_arm_sizes_from_moments_when_ttest_is_absent(three_arm_db):
    from agents.analyze.nodes_analysis import check_srm_node
    sql = "SELECT user_id, variant, y, x FROM u"
    ss = pushdown.compute_sufficient_stats(
        three_arm_db, sql, metric="y", covariate="x",
        segment_cols=[], guardrail_metrics=[], total_rows=90,
    )
    out = check_srm_node({
        "ttest_result": None,
        "sufficient_stats": ss,
        "query_result": pushdown.preview_frame(ss),   # 3 rows — must not be counted
        "metric": "y",
    })
    srm = out["srm_result"]
    assert srm.n_control == 30 and srm.n_treatment == 30
