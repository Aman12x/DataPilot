"""
tests/test_claim_accuracy.py — Unit tests for deterministic claim/safety scorers.

Covers score_claim_accuracy, score_safety_constraints, and
score_general_claim_accuracy beyond the magnitude-claim tests in test_eval_tools.py.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.eval_tools import (
    score_claim_accuracy,
    score_general_claim_accuracy,
    score_safety_constraints,
)
from tools.schemas import (
    CorrelationPair,
    CorrelationResult,
    CupedResult,
    GuardrailMetric,
    GuardrailResult,
    MdeResult,
    SrmResult,
    TtestResult,
)


def _ttest(significant: bool = True, cohens_d: float = 0.6) -> TtestResult:
    return TtestResult(
        t_stat=2.5,
        p_value=0.01 if significant else 0.20,
        ci_lower=0.02 if significant else -0.05,
        ci_upper=0.08 if significant else 0.03,
        significant=significant,
        cohens_d=cohens_d,
        n_control=500,
        n_treatment=500,
    )


def _cuped(ate: float = 0.05) -> CupedResult:
    return CupedResult(
        raw_ate=ate,
        cuped_ate=ate,
        variance_reduction_pct=20.0,
        theta=0.5,
    )


class TestScoreClaimAccuracy:
    def test_significant_claim_when_not_significant(self):
        narrative = "The result is statistically significant with p < 0.01."
        result = score_claim_accuracy(narrative, _ttest(significant=False), None)
        assert any("significance" in v.lower() or "p_value" in v.lower()
                   for v in result["violations"])

    def test_not_significant_claim_when_significant(self):
        narrative = "The result is not significant and we cannot draw conclusions."
        result = score_claim_accuracy(narrative, _ttest(significant=True), None)
        assert any("non-significance" in v.lower() or "p_value" in v.lower()
                   for v in result["violations"])

    def test_large_effect_with_small_cohens_d(self):
        narrative = "We observe a large effect on the primary metric."
        result = score_claim_accuracy(narrative, _ttest(cohens_d=0.2), None)
        assert any("large effect" in v.lower() for v in result["violations"])

    def test_direction_mismatch_cuped_positive_decrease_claim(self):
        narrative = "The treatment decreased DAU significantly."
        result = score_claim_accuracy(narrative, _ttest(), _cuped(ate=0.10))
        assert any("decreasing" in v.lower() or "decreased" in v.lower()
                   for v in result["violations"])

    def test_clean_narrative_no_violations(self):
        narrative = (
            "The treatment increased DAU with a statistically significant result. "
            "The effect is moderate in size."
        )
        result = score_claim_accuracy(narrative, _ttest(), _cuped(ate=0.05))
        assert result["violations"] == []


class TestScoreSafetyConstraints:
    def _guardrail(self, breached: bool = True) -> GuardrailResult:
        return GuardrailResult(
            guardrails=[
                GuardrailMetric(
                    metric="notif_optout",
                    control_mean=0.02,
                    treatment_mean=0.08,
                    delta_pct=300.0,
                    p_value=0.001,
                    breached=breached,
                )
            ],
            any_breached=breached,
            breached_count=1 if breached else 0,
        )

    def test_ship_when_srm_detected(self):
        narrative = "We recommend we ship the treatment to all users immediately."
        srm = SrmResult(
            n_control=500, n_treatment=500,
            expected_ratio=0.5, observed_ratio=0.4,
            chi2=12.0, p_value=0.001, srm_detected=True,
        )
        result = score_safety_constraints(narrative, srm_result=srm)
        assert len(result["violations"]) >= 1

    def test_ship_when_guardrail_breached_and_significant(self):
        narrative = "Ship the winning variant — treatment revenue is up."
        result = score_safety_constraints(
            narrative,
            guardrail_result=self._guardrail(breached=True),
            ttest_result=_ttest(significant=True),
        )
        assert len(result["violations"]) >= 1

    def test_stop_language_passes_with_srm(self):
        narrative = "Stop the experiment and investigate the randomization issue."
        srm = SrmResult(
            n_control=500, n_treatment=500,
            expected_ratio=0.5, observed_ratio=0.4,
            chi2=12.0, p_value=0.001, srm_detected=True,
        )
        result = score_safety_constraints(narrative, srm_result=srm)
        assert result["violations"] == []

    def test_winners_curse_requires_confirmatory(self):
        narrative = "Ship the treatment — the result is statistically significant."
        mde = MdeResult(
            mde_absolute=0.01,
            mde_relative_pct=5.0,
            is_powered_for_observed_effect=False,
            post_hoc_power=0.30,
        )
        result = score_safety_constraints(
            narrative,
            mde_result=mde,
            ttest_result=_ttest(significant=True),
        )
        assert any("confirmatory" in v.lower() or "power" in v.lower()
                   for v in result["violations"])


class TestScoreGeneralClaimAccuracy:
    def _corr(self, r: float) -> CorrelationResult:
        return CorrelationResult(
            pairs=[CorrelationPair(col_a="age", col_b="salary", correlation=r)]
        )

    def test_strong_correlation_claim_with_weak_r(self):
        narrative = "There is a strong positive correlation between age and salary."
        result = score_general_claim_accuracy(narrative, correlation_result=self._corr(0.15))
        assert any("strong correlation" in v.lower() for v in result["violations"])

    def test_no_correlation_claim_with_moderate_r(self):
        narrative = "There is no significant correlation between the variables."
        result = score_general_claim_accuracy(narrative, correlation_result=self._corr(0.55))
        assert any("no" in v.lower() or "weak" in v.lower() for v in result["violations"])

    def test_positive_direction_claim_with_negative_r(self):
        narrative = "A positive correlation exists between age and salary."
        result = score_general_claim_accuracy(narrative, correlation_result=self._corr(-0.60))
        assert any("positive" in v.lower() for v in result["violations"])

    def test_accurate_correlation_claim_passes(self):
        narrative = "Age and salary show a strong positive correlation."
        result = score_general_claim_accuracy(narrative, correlation_result=self._corr(0.72))
        assert result["violations"] == []
