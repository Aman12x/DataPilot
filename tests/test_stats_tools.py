"""
tests/test_stats_tools.py — Unit tests for tools/stats_tools.py

Uses base_experiment_df fixture (2000 users, known ground truth).
"""

import numpy as np
import pytest

from tools.stats_tools import check_srm, run_cuped, run_hte, run_ttest


# ── CUPED ──────────────────────────────────────────────────────────────────────

def test_cuped_reduces_variance(base_experiment_df):
    """CUPED adjustment reduces outcome variance when covariate correlates with metric."""
    result = run_cuped(base_experiment_df, "dau_rate", "pre_session_count", "variant")
    assert result.variance_reduction_pct > 0


def test_cuped_ate_closer_to_truth(base_experiment_df):
    """
    For the android/new subgroup, CUPED-adjusted values have lower variance
    than raw values, and the ATE points in the known correct direction (negative).
    """
    seg = base_experiment_df[
        (base_experiment_df["platform"] == "android") &
        (base_experiment_df["user_segment"] == "new")
    ].copy()
    result = run_cuped(seg, "dau_rate", "pre_session_count", "variant")

    # Direction must be correct: treatment lowers DAU
    assert result.raw_ate < 0
    assert result.cuped_ate < 0

    # CUPED-adjusted variance must be lower than raw variance
    adjusted = seg["dau_rate"] - result.theta * (
        seg["pre_session_count"] - seg["pre_session_count"].mean()
    )
    assert adjusted.var() < seg["dau_rate"].var()


# ── T-test ─────────────────────────────────────────────────────────────────────

def test_ttest_significant_on_affected_segment(base_experiment_df):
    """T-test is significant (p < 0.05) for the android/new segment with large effect."""
    seg  = base_experiment_df[
        (base_experiment_df["platform"] == "android") &
        (base_experiment_df["user_segment"] == "new")
    ]
    ctrl = seg[seg["variant"] == "control"]["dau_rate"]
    trt  = seg[seg["variant"] == "treatment"]["dau_rate"]
    result = run_ttest(ctrl, trt)
    assert result.p_value < 0.05


def test_ttest_not_significant_on_clean_segment(base_experiment_df):
    """T-test is not significant (p > 0.05) for a segment with no treatment effect."""
    seg  = base_experiment_df[
        (base_experiment_df["platform"] == "ios") &
        (base_experiment_df["user_segment"] == "returning")
    ]
    ctrl = seg[seg["variant"] == "control"]["dau_rate"]
    trt  = seg[seg["variant"] == "treatment"]["dau_rate"]
    result = run_ttest(ctrl, trt)
    assert result.p_value > 0.05


# ── HTE ────────────────────────────────────────────────────────────────────────

def test_hte_surfaces_correct_segment(base_experiment_df):
    """HTE surfaces platform=android,user_segment=new as the top affected segment."""
    result = run_hte(
        base_experiment_df, "dau_rate", "variant",
        segment_cols=["platform", "user_segment"]
    )
    assert "android" in result.top_segment
    assert "new" in result.top_segment


def test_hte_returns_all_segments(base_experiment_df):
    """HTE returns a non-empty list of all evaluated subgroups."""
    result = run_hte(
        base_experiment_df, "dau_rate", "variant",
        segment_cols=["platform", "user_segment"]
    )
    assert len(result.all_segments) > 0


# ── One-sided t-test ───────────────────────────────────────────────────────────

def test_ttest_one_sided_greater_halves_pvalue():
    """One-sided (greater) p-value ≈ two-sided p-value / 2 for a positive effect."""
    # Use moderate effect so p-values don't underflow to 0.0
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0.50, 0.10, 200)
    trt  = rng.normal(0.52, 0.10, 200)   # small positive effect → p ~ 0.04 two-sided

    two  = run_ttest(ctrl, trt, alternative="two-sided")
    one  = run_ttest(ctrl, trt, alternative="greater")

    assert one.p_value > 0.0
    assert two.p_value > 0.0
    assert one.p_value < two.p_value
    assert abs(one.p_value - two.p_value / 2) < 0.01


def test_ttest_one_sided_less_high_pvalue_for_positive_effect():
    """One-sided (less) returns p > 0.5 when treatment is actually higher than control."""
    rng = np.random.default_rng(1)
    ctrl = rng.normal(0.50, 0.10, 500)
    trt  = rng.normal(0.56, 0.10, 500)   # treatment higher → wrong direction for 'less'

    result = run_ttest(ctrl, trt, alternative="less")
    assert result.p_value > 0.5


def test_ttest_alternative_stored_in_result():
    """TtestResult.alternative reflects the parameter passed."""
    rng = np.random.default_rng(2)
    ctrl = rng.normal(0.5, 0.1, 200)
    trt  = rng.normal(0.5, 0.1, 200)

    r2 = run_ttest(ctrl, trt, alternative="two-sided")
    r1 = run_ttest(ctrl, trt, alternative="greater")
    assert r2.alternative == "two-sided"
    assert r1.alternative == "greater"


def test_ttest_invalid_alternative_raises():
    """Passing an invalid alternative raises ValueError."""
    rng = np.random.default_rng(3)
    ctrl = rng.normal(0.5, 0.1, 100)
    trt  = rng.normal(0.5, 0.1, 100)
    with pytest.raises(ValueError, match="alternative"):
        run_ttest(ctrl, trt, alternative="both")


# ── SRM ────────────────────────────────────────────────────────────────────────

def test_srm_not_detected_balanced():
    """check_srm does not flag a perfectly balanced 50/50 split."""
    result = check_srm(n_control=5000, n_treatment=5000)
    assert result.srm_detected is False
    assert result.p_value > 0.001


def test_srm_detected_broken_randomisation():
    """check_srm flags a badly imbalanced split (70/30 instead of 50/50)."""
    result = check_srm(n_control=7000, n_treatment=3000)
    assert result.srm_detected is True
    assert result.p_value < 0.001


def test_srm_result_fields():
    """SrmResult exposes n_control, n_treatment, chi2, and p_value."""
    result = check_srm(n_control=500, n_treatment=510)
    assert result.n_control  == 500
    assert result.n_treatment == 510
    assert isinstance(result.chi2, float)
    assert 0.0 <= result.p_value <= 1.0
