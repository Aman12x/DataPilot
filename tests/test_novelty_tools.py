"""
tests/test_novelty_tools.py — Unit tests for tools/novelty_tools.py

Uses base_experiment_df fixture:
  - android/new treatment: ATE week1 ≈ -0.046, week2 ≈ -0.063  (growing, 1.4x)
  - Effect is growing → NOT novelty decay → novelty_likely == False
"""

import pytest

from tools.novelty_tools import detect_novelty_effect


def test_effect_is_growing(base_experiment_df):
    """Week 2 ATE is larger in magnitude than week 1 → effect_direction == 'growing'."""
    result = detect_novelty_effect(
        base_experiment_df, "dau_rate", "variant", "week"
    )
    assert result["effect_direction"] == "growing"


def test_novelty_not_likely(base_experiment_df):
    """Growing effect rules out novelty → novelty_likely == False."""
    result = detect_novelty_effect(
        base_experiment_df, "dau_rate", "variant", "week"
    )
    assert result["novelty_likely"] is False


def test_week_effects_computed(base_experiment_df):
    """Both week1_ate and week2_ate are non-zero floats."""
    result = detect_novelty_effect(
        base_experiment_df, "dau_rate", "variant", "week"
    )
    assert isinstance(result["week1_ate"], float)
    assert isinstance(result["week2_ate"], float)
    assert result["week1_ate"] != 0.0
    assert result["week2_ate"] != 0.0
