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
