"""
tests/test_guardrail_tools.py — Unit tests for tools/guardrail_tools.py

Uses base_experiment_df fixture:
  - notif_optout:  android/new treatment 3-4x higher → breached
  - d7_retained:   android/new treatment lower       → breached
  - session_count: no treatment effect               → not breached
"""

import pytest

from tools.guardrail_tools import check_guardrails


GUARDRAIL_METRICS = ["notif_optout", "d7_retained", "session_count"]


def test_optout_rate_breached(base_experiment_df):
    """notif_optout is significantly higher in treatment → breached."""
    result = check_guardrails(base_experiment_df, "variant", GUARDRAIL_METRICS)
    optout = next(g for g in result.guardrails if g.metric == "notif_optout")
    assert optout.breached is True


def test_d7_retention_breached(base_experiment_df):
    """d7_retained is significantly lower in treatment → breached."""
    result = check_guardrails(base_experiment_df, "variant", GUARDRAIL_METRICS)
    d7 = next(g for g in result.guardrails if g.metric == "d7_retained")
    assert d7.breached is True


def test_clean_metric_not_breached(base_experiment_df):
    """session_count has no treatment effect → not breached."""
    result = check_guardrails(base_experiment_df, "variant", GUARDRAIL_METRICS)
    sessions = next(g for g in result.guardrails if g.metric == "session_count")
    assert sessions.breached is False


def test_returns_any_breached_true(base_experiment_df):
    """any_breached is True when at least one guardrail is violated."""
    result = check_guardrails(base_experiment_df, "variant", GUARDRAIL_METRICS)
    assert result.any_breached is True
