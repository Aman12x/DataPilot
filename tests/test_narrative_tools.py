"""
tests/test_narrative_tools.py — Unit tests for tools/narrative_tools.py

Validates that format_narrative produces a correctly structured markdown draft
from representative tool-result dicts matching the DataPilot ground truth scenario.
"""

import pytest

from tools.narrative_tools import format_narrative


# ── Minimal representative inputs mirroring ground truth ──────────────────────

DECOMP = {"dominant_change_component": "new_users"}

ANOMALY = {"anomaly_dates": ["2024-01-30"], "direction": "drop", "severity": 3.2}

CUPED = {"raw_ate": -0.08, "cuped_ate": -0.082, "variance_reduction_pct": 24.9, "theta": 0.31}

TTEST = {"t_stat": -4.5, "p_value": 0.000027, "ci_lower": -0.11, "ci_upper": -0.05, "significant": True}

HTE = {
    "top_segment": "platform=android,user_segment=new",
    "effect_size": -0.22,
    "segment_share": 20.0,
    "all_segments": [],
}

NOVELTY = {
    "week1_ate": -0.046,
    "week2_ate": -0.063,
    "effect_direction": "growing",
    "novelty_likely": False,
}

MDE = {"mde_absolute": 0.031, "mde_relative_pct": 5.7, "is_powered_for_observed_effect": False}

GUARDRAILS_BREACHED = {
    "any_breached": True,
    "breached_count": 2,
    "guardrails": [
        {"metric": "notif_optout", "control_mean": 0.02, "treatment_mean": 0.07,
         "delta_pct": 250.0, "p_value": 0.0001, "breached": True},
        {"metric": "d7_retained", "control_mean": 0.45, "treatment_mean": 0.30,
         "delta_pct": -33.0, "p_value": 0.0003, "breached": True},
    ],
}

GUARDRAILS_CLEAN = {
    "any_breached": False,
    "breached_count": 0,
    "guardrails": [
        {"metric": "session_count", "control_mean": 3.0, "treatment_mean": 3.01,
         "delta_pct": 0.3, "p_value": 0.8, "breached": False},
    ],
}

FUNNEL = {
    "biggest_dropoff_step": "d1_retain",
    "steps": [
        {"step": "impression",  "control_rate": 1.0,    "treatment_rate": 1.0,    "delta": 0.0,   "pct_change": 0.0,   "p_value": 1.0,    "significant": False},
        {"step": "click",       "control_rate": 0.30,   "treatment_rate": 0.30,   "delta": 0.0,   "pct_change": 0.0,   "p_value": 0.9,    "significant": False},
        {"step": "install",     "control_rate": 0.18,   "treatment_rate": 0.18,   "delta": 0.0,   "pct_change": 0.0,   "p_value": 0.85,   "significant": False},
        {"step": "d1_retain",   "control_rate": 0.45,   "treatment_rate": 0.25,   "delta": -0.20, "pct_change": -44.4, "p_value": 0.04,   "significant": True},
    ],
}

FORECAST = {"forecast_df": None, "actual_vs_forecast_delta": -80.0, "outside_ci": True, "method": "prophet", "warning": None}

BUSINESS_IMPACT = "At MDE of 5.7%, detects a dau_rate lift worth ~$1k/day at current scale."


def _build(**overrides):
    """Helper: build format_narrative kwargs, applying optional overrides."""
    base = dict(
        metric="dau_rate",
        decomposition_result=DECOMP,
        anomaly_result=ANOMALY,
        cuped_result=CUPED,
        ttest_result=TTEST,
        hte_result=HTE,
        novelty_result=NOVELTY,
        mde_result=MDE,
        guardrail_result=GUARDRAILS_BREACHED,
        funnel_result=FUNNEL,
        forecast_result=FORECAST,
        business_impact=BUSINESS_IMPACT,
    )
    base.update(overrides)
    return base


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_all_seven_sections_present():
    """Narrative draft contains all 7 required section headers."""
    result = format_narrative(**_build())
    draft = result.narrative_draft
    for section in ["TL;DR", "What we found", "Where it's concentrated",
                    "What else is affected", "Confidence level",
                    "Recommendation", "Caveats"]:
        assert section in draft, f"Missing section: {section}"


def test_affected_segment_mentioned():
    """Top HTE segment (android + new) appears in the narrative."""
    result = format_narrative(**_build())
    draft = result.narrative_draft.lower()
    assert "android" in draft
    assert "new" in draft


def test_guardrail_breach_flagged():
    """Breached guardrail metrics appear in the narrative."""
    result = format_narrative(**_build())
    draft = result.narrative_draft
    assert "notif_optout" in draft
    assert "d7_retained" in draft


def test_novelty_ruled_out_mentioned():
    """Narrative states novelty is ruled out when novelty_likely=False."""
    result = format_narrative(**_build())
    draft = result.narrative_draft
    assert "novelty ruled out" in draft.lower()


def test_recommendation_returned():
    """recommendation is a non-empty string."""
    result = format_narrative(**_build())
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation) > 0


def test_caveats_present():
    """Caveats section contains at least one caveat bullet."""
    result = format_narrative(**_build())
    caveats_section = result.narrative_draft.split("## Caveats")[-1]
    assert "- " in caveats_section


def test_clean_guardrails_message():
    """When no guardrails are breached, narrative says so."""
    result = format_narrative(**_build(guardrail_result=GUARDRAILS_CLEAN))
    draft = result.narrative_draft
    assert "within acceptable bounds" in draft


def test_analyst_notes_included():
    """Analyst notes appear in the draft when provided."""
    result = format_narrative(**_build(analyst_notes="Focus on week-2 effect only."))
    assert "Focus on week-2 effect only." in result.narrative_draft
