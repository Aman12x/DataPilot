"""
tools/novelty_tools.py — Week-over-week treatment effect decay detection.

novelty_likely=True only if the effect is DECAYING and week2_ate < 0.5 * week1_ate.
A growing or stable effect rules out novelty.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


def detect_novelty_effect(
    df: pd.DataFrame,
    metric_col: str,
    variant_col: str,
    week_col: str,
) -> dict[str, Any]:
    """
    Compare the Average Treatment Effect in week 1 vs week 2 to determine
    whether the treatment effect is decaying (novelty), growing, or stable.

    Args:
        df:          DataFrame with one row per user, containing metric_col,
                     variant_col ('control'|'treatment'), and week_col (1|2).
        metric_col:  Outcome metric (e.g. 'dau_rate').
        variant_col: Column with 'control' / 'treatment' values.
        week_col:    Column with experiment week number (1 or 2).

    Returns:
        {
            week1_ate:        float,   # ATE in week 1 (treatment_mean - control_mean)
            week2_ate:        float,   # ATE in week 2
            effect_direction: str,    # 'decaying' | 'growing' | 'stable'
            novelty_likely:   bool,   # True only if decaying AND |week2| < 0.5*|week1|
        }
    """
    for col in [metric_col, variant_col, week_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    variants = set(df[variant_col].dropna().unique())
    if not {"control", "treatment"}.issubset(variants):
        raise ValueError(
            f"variant_col must contain 'control' and 'treatment'. Found: {variants}"
        )

    weeks = set(df[week_col].dropna().unique())
    if not {1, 2}.issubset(weeks):
        raise ValueError(
            f"week_col must contain weeks 1 and 2. Found: {weeks}"
        )

    def ate_for_week(week: int) -> float:
        wdf  = df[df[week_col] == week]
        ctrl = wdf[wdf[variant_col] == "control"][metric_col].dropna()
        trt  = wdf[wdf[variant_col] == "treatment"][metric_col].dropna()
        if len(ctrl) < 2 or len(trt) < 2:
            raise ValueError(f"Not enough data in week {week} to compute ATE.")
        return float(trt.mean() - ctrl.mean())

    week1_ate = ate_for_week(1)
    week2_ate = ate_for_week(2)

    # Direction: compare absolute effect sizes
    abs1, abs2 = abs(week1_ate), abs(week2_ate)

    if abs1 == 0:
        effect_direction = "stable"
    elif abs2 > abs1 * 1.10:          # >10% larger → growing
        effect_direction = "growing"
    elif abs2 < abs1 * 0.90:          # >10% smaller → decaying
        effect_direction = "decaying"
    else:
        effect_direction = "stable"

    # Novelty: effect must be decaying AND more than halved
    novelty_likely = (
        effect_direction == "decaying"
        and abs1 > 0
        and abs2 < 0.5 * abs1
    )

    return {
        "week1_ate":        round(week1_ate, 6),
        "week2_ate":        round(week2_ate, 6),
        "effect_direction": effect_direction,
        "novelty_likely":   novelty_likely,
    }
