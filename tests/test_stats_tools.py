"""
tests/test_stats_tools.py — Unit tests for tools/stats_tools.py

Uses base_experiment_df fixture (2000 users, known ground truth).
"""

import numpy as np
import pytest

from tools.stats_tools import run_cuped, run_hte, run_ttest


# ── CUPED ──────────────────────────────────────────────────────────────────────

def test_cuped_reduces_variance(base_experiment_df):
    """CUPED adjustment reduces outcome variance when covariate correlates with metric."""
    result = run_cuped(base_experiment_df, "dau_rate", "pre_session_count", "variant")
    assert result["variance_reduction_pct"] > 0


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
    assert result["raw_ate"] < 0
    assert result["cuped_ate"] < 0

    # CUPED-adjusted variance must be lower than raw variance
    adjusted = seg["dau_rate"] - result["theta"] * (
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
    assert result["p_value"] < 0.05


def test_ttest_not_significant_on_clean_segment(base_experiment_df):
    """T-test is not significant (p > 0.05) for a segment with no treatment effect."""
    seg  = base_experiment_df[
        (base_experiment_df["platform"] == "ios") &
        (base_experiment_df["user_segment"] == "returning")
    ]
    ctrl = seg[seg["variant"] == "control"]["dau_rate"]
    trt  = seg[seg["variant"] == "treatment"]["dau_rate"]
    result = run_ttest(ctrl, trt)
    assert result["p_value"] > 0.05


# ── HTE ────────────────────────────────────────────────────────────────────────

def test_hte_surfaces_correct_segment(base_experiment_df):
    """HTE surfaces platform=android,user_segment=new as the top affected segment."""
    result = run_hte(
        base_experiment_df, "dau_rate", "variant",
        segment_cols=["platform", "user_segment"]
    )
    assert "android" in result["top_segment"]
    assert "new" in result["top_segment"]


def test_hte_returns_all_segments(base_experiment_df):
    """HTE returns a non-empty list of all evaluated subgroups."""
    result = run_hte(
        base_experiment_df, "dau_rate", "variant",
        segment_cols=["platform", "user_segment"]
    )
    assert len(result["all_segments"]) > 0
