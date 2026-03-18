"""
tools/mde_tools.py — Minimum Detectable Effect calculation + business impact.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

import math
from typing import Any

from scipy import stats


def compute_mde(
    n_control: int,
    n_treatment: int,
    baseline_mean: float,
    baseline_std: float,
    alpha: float = 0.05,
    power: float = 0.80,
    observed_effect_abs: float | None = None,
) -> dict[str, Any]:
    """
    Compute the Minimum Detectable Effect for a two-sample t-test.

    MDE = (z_α/2 + z_β) × σ × sqrt(1/n_ctrl + 1/n_trt)

    Args:
        n_control:            Number of control-group observations.
        n_treatment:          Number of treatment-group observations.
        baseline_mean:        Control group mean (used to express MDE as a %).
        baseline_std:         Pooled / control-group standard deviation.
        alpha:                Two-tailed significance level (default 0.05).
        power:                Desired statistical power (default 0.80).
        observed_effect_abs:  Optional: the actual observed ATE (absolute).
                              If provided, is_powered_for_observed_effect is
                              True when |observed_effect_abs| >= mde_absolute.

    Returns:
        {
            mde_absolute:                  float,
            mde_relative_pct:              float,
            is_powered_for_observed_effect: bool | None,
        }
    """
    if n_control < 1 or n_treatment < 1:
        raise ValueError("Sample sizes must be positive integers.")
    if baseline_std <= 0:
        raise ValueError("baseline_std must be > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if not (0 < power < 1):
        raise ValueError("power must be in (0, 1).")

    z_alpha = stats.norm.ppf(1 - alpha / 2)   # e.g. 1.96 for alpha=0.05
    z_beta  = stats.norm.ppf(power)            # e.g. 0.84 for power=0.80

    se = baseline_std * math.sqrt(1 / n_control + 1 / n_treatment)
    mde_absolute = (z_alpha + z_beta) * se

    mde_relative_pct = (
        mde_absolute / abs(baseline_mean) * 100
        if baseline_mean != 0 else float("inf")
    )

    if observed_effect_abs is not None:
        is_powered = bool(abs(observed_effect_abs) >= mde_absolute)
    else:
        is_powered = None

    return {
        "mde_absolute":                   round(mde_absolute, 6),
        "mde_relative_pct":               round(mde_relative_pct, 2),
        "is_powered_for_observed_effect": is_powered,
    }


def business_impact_statement(
    mde_relative_pct: float,
    metric: str,
    baseline_dau: int,
    revenue_per_dau: float = 0.50,
) -> str:
    """
    Translate an MDE percentage into a dollar-value statement.

    Returns:
        e.g. "At MDE of 2.1%, detects a lift worth ~$1k/day at current scale."
    """
    if baseline_dau <= 0:
        raise ValueError("baseline_dau must be positive.")

    daily_impact_usd = baseline_dau * (mde_relative_pct / 100) * revenue_per_dau
    impact_k         = daily_impact_usd / 1_000

    if impact_k >= 1:
        impact_str = f"~${impact_k:.0f}k"
    else:
        impact_str = f"~${daily_impact_usd:.0f}"

    return (
        f"At MDE of {mde_relative_pct:.1f}%, detects a {metric} lift worth "
        f"{impact_str}/day at current scale."
    )
