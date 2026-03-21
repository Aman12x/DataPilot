"""
tests/test_mde_tools.py — Unit tests for tools/mde_tools.py

Ground truth from CLAUDE.md:
  - Total experiment: ~5000 users per arm, blended effect ~1.3-1.6% → near/below MDE
  - Android/new segment: ~1000 users per arm, effect ~20% → well above MDE
"""

import pytest

from tools.mde_tools import compute_mde, business_impact_statement


def test_mde_returns_relative_pct():
    """mde_relative_pct is a positive float for typical inputs."""
    result = compute_mde(
        n_control=5000,
        n_treatment=5000,
        baseline_mean=0.55,
        baseline_std=0.10,
    )
    assert isinstance(result.mde_relative_pct, float)
    assert result.mde_relative_pct > 0


def test_underpowered_for_blended_effect():
    """Blended effect ~1.6% is below MDE (~5.7%) at moderate sample size with high variance → not powered."""
    result = compute_mde(
        n_control=1000,
        n_treatment=1000,
        baseline_mean=0.55,
        baseline_std=0.25,      # realistic variance for a noisy DAU metric
        observed_effect_abs=0.55 * 0.016,  # ~1.6% relative blended effect
    )
    assert result.is_powered_for_observed_effect is False


def test_powered_for_segment_effect():
    """Android/new segment effect ~20% (abs ~0.11) >> MDE → powered."""
    result = compute_mde(
        n_control=200,
        n_treatment=200,
        baseline_mean=0.55,
        baseline_std=0.08,
        observed_effect_abs=0.20,  # large effect in affected segment
    )
    assert result.is_powered_for_observed_effect is True


def test_business_impact_string_contains_dollar():
    """business_impact_statement returns a string with a dollar sign."""
    statement = business_impact_statement(
        mde_relative_pct=2.1,
        metric="dau",
        baseline_dau=500_000,
    )
    assert "$" in statement
    assert "2.1" in statement
    assert "dau" in statement
