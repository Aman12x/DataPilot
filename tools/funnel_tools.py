"""
tools/funnel_tools.py — Conversion funnel drop-off analysis.

Rates are computed conditionally: step k's rate is among users who completed step k-1.
This gives proper per-step conversion rates and powered significance tests.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


def compute_funnel(
    df: pd.DataFrame,
    variant_col: str,
    steps: list[str] | None = None,
    segment_filter: dict[str, str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute per-step conditional conversion rates for control vs treatment.

    Step k's rate = fraction of users who completed step k-1 AND completed step k.
    Significance is tested only among users eligible at each step (completed k-1).

    Args:
        df:             DataFrame with one row per (user, step). Must contain
                        at least `variant_col`, a 'step' column, a 'user_id' column,
                        and a 'completed' column (1 = completed, 0 = not).
        variant_col:    Column with 'control' / 'treatment' values.
        steps:          Ordered list of funnel step names. Defaults to
                        ['impression', 'click', 'install', 'd1_retain'].
        segment_filter: Optional dict of {col: value} to pre-filter rows
                        (e.g. {'platform': 'android', 'user_segment': 'new'}).
        alpha:          Significance level for per-step z-test (default 0.05).

    Returns:
        {
            steps: list[{
                step:           str,
                control_rate:   float,   # conditional completion rate in control
                treatment_rate: float,   # conditional completion rate in treatment
                delta:          float,   # treatment_rate - control_rate
                pct_change:     float,   # delta / control_rate * 100
                p_value:        float,   # two-proportion z-test (among eligible users)
                significant:    bool,
            }],
            biggest_dropoff_step: str,   # step with largest |delta|
        }
    """
    if steps is None:
        steps = ["impression", "click", "install", "d1_retain"]

    required_cols = {variant_col, "step", "user_id", "completed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    variants = set(df[variant_col].dropna().unique())
    if not {"control", "treatment"}.issubset(variants):
        raise ValueError(
            f"variant_col must contain 'control' and 'treatment'. Found: {variants}"
        )

    present_steps = set(df["step"].dropna().unique())
    missing_steps = [s for s in steps if s not in present_steps]
    if missing_steps:
        raise ValueError(f"Steps not found in data: {missing_steps}")

    if segment_filter:
        for col, val in segment_filter.items():
            if col not in df.columns:
                raise ValueError(f"segment_filter column '{col}' not in DataFrame.")
            df = df[df[col] == val].copy()

    # Pivot: one row per user, one column per step (0/1)
    pivot = (
        df.pivot_table(index=["user_id", variant_col], columns="step", values="completed", aggfunc="max")
        .reset_index()
        .fillna(0)
    )
    # Ensure all steps are present as columns
    for s in steps:
        if s not in pivot.columns:
            pivot[s] = 0

    step_results = []
    largest_delta = 0.0
    biggest_dropoff_step = steps[0]

    for i, step in enumerate(steps):
        if i == 0:
            # First step: all users are eligible
            eligible = pivot
        else:
            # Only users who completed the previous step
            prev_step = steps[i - 1]
            eligible = pivot[pivot[prev_step] == 1]

        ctrl = eligible[eligible[variant_col] == "control"][step]
        trt  = eligible[eligible[variant_col] == "treatment"][step]

        n_ctrl = len(ctrl)
        n_trt  = len(trt)

        if n_ctrl < 2 or n_trt < 2:
            raise ValueError(
                f"Not enough eligible users at step '{step}' "
                f"(ctrl={n_ctrl}, trt={n_trt})."
            )

        ctrl_rate = float(ctrl.mean())
        trt_rate  = float(trt.mean())
        delta     = trt_rate - ctrl_rate
        pct_change = (delta / ctrl_rate * 100) if ctrl_rate != 0 else float("inf")

        # Two-proportion z-test among eligible users
        p_pool = (ctrl.sum() + trt.sum()) / (n_ctrl + n_trt)
        se = (p_pool * (1 - p_pool) * (1 / n_ctrl + 1 / n_trt)) ** 0.5

        if se == 0:
            z_stat, p_value = 0.0, 1.0
        else:
            z_stat  = (trt_rate - ctrl_rate) / se
            p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        significant = p_value < alpha

        step_results.append({
            "step":           step,
            "control_rate":   round(ctrl_rate, 4),
            "treatment_rate": round(trt_rate, 4),
            "delta":          round(delta, 4),
            "pct_change":     round(pct_change, 2),
            "p_value":        round(p_value, 6),
            "significant":    significant,
        })

        if abs(delta) > abs(largest_delta):
            largest_delta = delta
            biggest_dropoff_step = step

    return {
        "steps":                step_results,
        "biggest_dropoff_step": biggest_dropoff_step,
    }
