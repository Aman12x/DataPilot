"""
tools/novelty_tools.py — Week-over-week treatment effect decay detection.

novelty_likely=True only if the effect is DECAYING and week2_ate < 0.5 * week1_ate.
A growing or stable effect rules out novelty.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from tools.schemas import NoveltyResult


def detect_novelty_effect(
    df: pd.DataFrame,
    metric_col: str,
    variant_col: str,
    week_col: str,
) -> NoveltyResult:
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

    # Normalise week values to int: accept 1/2, "1"/"2", "W1"/"W2", "week_1"/"week_2".
    def _to_week_int(val) -> int | None:
        if isinstance(val, (int, float)) and not pd.isna(val):
            return int(val)
        s = str(val).strip().lower()
        # Strip common prefixes
        for prefix in ("week_", "week", "w"):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    df = df.copy()
    normalised = df[week_col].map(_to_week_int)
    if normalised.isna().any() and not df[week_col].isna().any():
        raise ValueError(
            f"week_col '{week_col}' contains values that cannot be parsed as week numbers: "
            f"{sorted(df[week_col].dropna().unique().tolist())}"
        )
    df[week_col] = normalised

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

    if abs1 == 0 and abs2 == 0:
        effect_direction = "stable"
    elif abs1 == 0:
        # No effect in week 1, but an effect appeared in week 2.
        effect_direction = "growing" if week2_ate > 0 else "decaying"
    elif abs2 > abs1 * 1.10:          # >10% larger → growing
        effect_direction = "growing"
    elif abs2 < abs1 * 0.90:          # >10% smaller → decaying
        effect_direction = "decaying"
    # Sign reversal: week1 positive but week2 negative (or vice versa) — always decaying
    elif (week1_ate > 0) != (week2_ate > 0) and abs2 > 0:
        effect_direction = "decaying"
    else:
        effect_direction = "stable"

    # Novelty: effect must be decaying AND more than halved (or reversed sign)
    sign_reversed = abs1 > 0 and abs2 > 0 and (week1_ate > 0) != (week2_ate > 0)
    novelty_likely = (
        effect_direction == "decaying"
        and abs1 > 0
        and (abs2 < 0.5 * abs1 or sign_reversed)
    )

    return NoveltyResult(
        week1_ate=round(week1_ate, 6),
        week2_ate=round(week2_ate, 6),
        effect_direction=effect_direction,
        novelty_likely=novelty_likely,
    )
