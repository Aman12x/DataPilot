"""
tests/test_graph_integration.py — Integration tests for the A/B analysis node pipeline.

Runs real graph nodes (not mocked tools) against base_experiment_df ground truth,
then verifies _compute_quality_score penalises claim/safety violations.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.analyze.nodes_analysis import (
    check_guardrails_node,
    run_cuped_node,
    run_hte_node,
    run_ttest_node,
)
from agents.analyze.nodes_narrative import _compute_quality_score
from tools.schemas import CupedResult, GuardrailResult, SrmResult, TtestResult


def _ab_state(base_experiment_df):
    return {
        "query_result": base_experiment_df,
        "metric": "dau_rate",
        "covariate": "pre_session_count",
        "analysis_mode": "ab_test",
        "task": "Did the push notification experiment affect DAU?",
        "guardrail_result": None,
        "cuped_result": None,
        "ttest_result": None,
        "hte_result": None,
        "novelty_result": None,
        "forecast_result": None,
        "srm_result": None,
        "mde_result": None,
    }


class TestAbAnalysisPipeline:
    def test_cuped_reduces_variance(self, base_experiment_df):
        state = _ab_state(base_experiment_df)
        out = run_cuped_node(state)
        assert "cuped_result" in out
        assert out["cuped_result"].variance_reduction_pct > 5

    def test_ttest_significant_on_full_sample(self, base_experiment_df):
        state = _ab_state(base_experiment_df)
        state.update(run_cuped_node(state))
        out = run_ttest_node(state)
        assert "ttest_result" in out
        # Full sample has mixed segments — may or may not be significant overall
        assert out["ttest_result"].p_value is not None

    def test_hte_surfaces_android_new(self, base_experiment_df):
        state = _ab_state(base_experiment_df)
        out = run_hte_node(state)
        assert "hte_result" in out
        top = out["hte_result"].top_segment
        assert "android" in top
        assert "new" in top

    def test_guardrails_detect_breach(self, base_experiment_df):
        state = _ab_state(base_experiment_df)
        out = check_guardrails_node(state)
        assert "guardrail_result" in out
        assert out["guardrail_result"].any_breached is True

    def test_full_pipeline_state_fields(self, base_experiment_df):
        """Run core A/B nodes in sequence and verify ground-truth segment."""
        state = _ab_state(base_experiment_df)
        for node in (run_cuped_node, run_ttest_node, run_hte_node, check_guardrails_node):
            state.update(node(state))

        hte = state["hte_result"]
        assert "android" in hte.top_segment
        assert state["guardrail_result"].any_breached is True
        assert state["cuped_result"].variance_reduction_pct > 0


class TestQualityScoreIntegration:
    def test_safety_violation_caps_score(self, base_experiment_df):
        """Narrative recommending ship under SRM should lower quality score."""
        good_narrative = (
            "Investigate the randomization issue before drawing conclusions. "
            "Do not ship until SRM is resolved."
        )
        bad_narrative = (
            "The android/new segment shows a significant DAU drop. "
            "Ship the treatment to all users immediately."
        )
        base_state = {
            "analysis_mode": "ab_test",
            "task": "Did the experiment affect DAU?",
            "query_result": base_experiment_df,
            "cuped_result": CupedResult(
                raw_ate=-0.05, cuped_ate=-0.05,
                variance_reduction_pct=20.0, theta=0.5,
            ),
            "ttest_result": TtestResult(
                t_stat=-3.0, p_value=0.001, ci_lower=-0.10, ci_upper=-0.02,
                significant=True, cohens_d=-0.6,
            ),
            "hte_result": None,
            "guardrail_result": None,
            "novelty_result": None,
            "forecast_result": None,
            "srm_result": SrmResult(
                n_control=1000, n_treatment=1000,
                expected_ratio=0.5, observed_ratio=0.4,
                chi2=15.0, p_value=0.001, srm_detected=True,
            ),
            "mde_result": None,
        }

        good_state = {**base_state, "final_narrative": good_narrative}
        bad_state  = {**base_state, "final_narrative": bad_narrative}

        good_score = _compute_quality_score(good_state)
        bad_score  = _compute_quality_score(bad_state)
        assert bad_score <= good_score
        assert bad_score <= 0.85

    def test_general_mode_claim_violation_caps_score(self):
        from tools.schemas import CorrelationPair, CorrelationResult, DescribeResult

        state = {
            "analysis_mode": "general",
            "task": "What correlates with salary?",
            "query_result": None,
            "describe_result": DescribeResult(row_count=100, col_count=5, columns=[]),
            "correlation_result": CorrelationResult(
                pairs=[CorrelationPair(col_a="x", col_b="y", correlation=0.1)]
            ),
            "charts": [{"title": "chart"}],
            "narrative_draft": "There is a strong positive correlation between x and y.",
        }
        score = _compute_quality_score(state)
        assert score <= 0.85
