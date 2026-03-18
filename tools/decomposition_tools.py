"""
tools/decomposition_tools.py — DAU component breakdown: new/retained/resurrected/churned.

Pure Python, no LangGraph or Streamlit imports.
Input: pre-aggregated metrics_daily DataFrame (one row per date × platform × segment).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def decompose_dau(
    df: pd.DataFrame,
    date_col: str = "date",
    window_days: int = 28,
) -> dict[str, Any]:
    """
    Decompose DAU into new / retained / resurrected / churned components and
    identify which component drove the largest change between the baseline
    period and the recent period.

    Args:
        df:          DataFrame with at minimum: date_col, dau, new_users,
                     retained_users, resurrected_users, churned_users.
                     May contain multiple rows per date (e.g. one per platform) —
                     they are summed before analysis.
        date_col:    Name of the date column.
        window_days: Number of days to use as the baseline period (oldest dates).
                     The remaining dates form the "recent" (experiment) period.

    Returns:
        {
            "new":        {time_series, baseline_avg, recent_avg, delta, pct_of_dau},
            "retained":   {time_series, baseline_avg, recent_avg, delta, pct_of_dau},
            "resurrected":{time_series, baseline_avg, recent_avg, delta, pct_of_dau},
            "churned":    {time_series, baseline_avg, recent_avg, delta, pct_of_dau},
            "dominant_change_component": str,   # name of component with largest |delta|
        }
    """
    required = {date_col, "dau", "new_users", "retained_users",
                "resurrected_users", "churned_users"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Aggregate across any dimension columns (platform, user_segment, etc.)
    daily = (
        df.groupby(date_col, as_index=False)
        .agg(
            dau=("dau", "sum"),
            new_users=("new_users", "sum"),
            retained_users=("retained_users", "sum"),
            resurrected_users=("resurrected_users", "sum"),
            churned_users=("churned_users", "sum"),
        )
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    n = len(daily)
    if n < 2:
        raise ValueError("Need at least 2 dates to compute decomposition.")

    # Baseline = first window_days; recent = remainder
    baseline = daily.iloc[:window_days]
    recent   = daily.iloc[window_days:]

    if len(recent) == 0:
        # Fallback: split 50/50 if data is shorter than window
        mid      = n // 2
        baseline = daily.iloc[:mid]
        recent   = daily.iloc[mid:]

    components = {
        "new":         "new_users",
        "retained":    "retained_users",
        "resurrected": "resurrected_users",
        "churned":     "churned_users",
    }

    result: dict[str, Any] = {}
    deltas: dict[str, float] = {}

    mean_dau = float(daily["dau"].mean()) if daily["dau"].mean() > 0 else 1.0

    for key, col in components.items():
        baseline_avg = float(baseline[col].mean())
        recent_avg   = float(recent[col].mean())
        delta        = recent_avg - baseline_avg
        deltas[key]  = delta

        # Time series: {date_str: value}
        ts = {
            str(row[date_col].date()): int(row[col])
            for _, row in daily.iterrows()
        }

        result[key] = {
            "time_series":   ts,
            "baseline_avg":  round(baseline_avg, 2),
            "recent_avg":    round(recent_avg, 2),
            "delta":         round(delta, 2),
            "pct_of_dau":    round(float(daily[col].mean()) / mean_dau * 100, 2),
        }

    # Dominant = component with largest absolute delta
    dominant = max(deltas, key=lambda k: abs(deltas[k]))
    result["dominant_change_component"] = dominant

    return result
