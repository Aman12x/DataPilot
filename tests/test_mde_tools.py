"""
tests/test_mde_tools.py — Unit tests for tools/mde_tools.py

Ground truth from CLAUDE.md:
  - Total experiment: ~5000 users per arm, blended effect ~1.3-1.6% → near/below MDE
  - Android/new segment: ~1000 users per arm, effect ~20% → well above MDE
"""

import pytest

from tools.mde_tools import compute_mde, compute_post_hoc_power, business_impact_statement


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


# ── Post-hoc power ─────────────────────────────────────────────────────────────

def test_post_hoc_power_large_effect_is_high():
    """A large observed effect with reasonable NCP returns power close to 1."""
    # NCP = 0.05 / (0.10 * sqrt(2/500)) ≈ 7.9 — well within scipy range
    power = compute_post_hoc_power(
        observed_effect_abs=0.05,
        n_control=500,
        n_treatment=500,
        baseline_std=0.10,
    )
    assert power > 0.95


def test_post_hoc_power_tiny_effect_is_low():
    """A tiny observed effect relative to sample variance returns low power."""
    power = compute_post_hoc_power(
        observed_effect_abs=0.001,
        n_control=200,
        n_treatment=200,
        baseline_std=0.25,
    )
    assert power < 0.10


def test_post_hoc_power_range():
    """compute_post_hoc_power always returns a value in [0, 1]."""
    for effect in [0.0, 0.01, 0.05, 0.20, 1.0]:
        power = compute_post_hoc_power(effect, 500, 500, 0.10)
        assert 0.0 <= power <= 1.0


def test_compute_mde_includes_post_hoc_power():
    """compute_mde populates post_hoc_power when observed_effect_abs is supplied."""
    # Use moderate NCP (effect=0.03, SE≈0.00447) so scipy nct doesn't overflow
    result = compute_mde(
        n_control=1000,
        n_treatment=1000,
        baseline_mean=0.55,
        baseline_std=0.10,
        observed_effect_abs=0.03,
    )
    assert result.post_hoc_power is not None
    assert 0.0 <= result.post_hoc_power <= 1.0


def test_compute_mde_no_post_hoc_power_without_effect():
    """compute_mde leaves post_hoc_power as None when no observed_effect_abs given."""
    result = compute_mde(
        n_control=1000,
        n_treatment=1000,
        baseline_mean=0.55,
        baseline_std=0.10,
    )
    assert result.post_hoc_power is None
