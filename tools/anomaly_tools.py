"""
tools/anomaly_tools.py — Time series anomaly detection + slice-and-dice.

Pure Python, no LangGraph or Streamlit imports.
Inputs: pre-aggregated metrics_daily DataFrame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_anomaly(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str = "date",
    method: str = "zscore",
    window: int = 14,
    threshold: float = 2.5,
) -> dict[str, Any]:
    """
    Detect anomalous dates in a metric time series using a rolling Z-score.

    The DataFrame may contain multiple rows per date (e.g. one per platform) —
    they are summed before analysis.

    Args:
        df:         DataFrame with date_col and metric_col.
        metric_col: Metric to analyse (e.g. 'dau').
        date_col:   Date column name.
        method:     Only 'zscore' is supported for now.
        window:     Rolling lookback window for mean/std (days).
        threshold:  |Z-score| threshold to flag an anomaly (default 2.5).

    Returns:
        {
            anomaly_dates: list[str],   # ISO date strings of flagged dates
            severity:      float,       # max |Z-score| across all dates
            direction:     str,         # 'drop' | 'spike'
        }
    """
    if metric_col not in df.columns:
        raise ValueError(f"metric_col '{metric_col}' not found in DataFrame.")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame.")
    if method != "zscore":
        raise ValueError(f"Unsupported method '{method}'. Use 'zscore'.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily = (
        df.groupby(date_col, as_index=False)[metric_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if len(daily) < 3:
        raise ValueError("Need at least 3 dates for anomaly detection.")

    series = daily[metric_col].astype(float)
    rolling_mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    rolling_std  = series.rolling(window, min_periods=max(3, window // 2)).std()

    # Avoid division by zero on perfectly flat segments
    rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(series.std() or 1.0)

    z_scores = (series - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0.0)

    # Try threshold; if nothing found, relax by 0.5 (once)
    anomaly_mask = z_scores.abs() > threshold
    if not anomaly_mask.any():
        anomaly_mask = z_scores.abs() > (threshold - 0.5)

    anomaly_dates = [
        str(daily[date_col].iloc[i].date())
        for i in range(len(daily))
        if anomaly_mask.iloc[i]
    ]

    severity  = float(z_scores.abs().max())
    flagged_z = z_scores[anomaly_mask]
    direction = "drop" if (len(flagged_z) > 0 and flagged_z.mean() < 0) else "spike"

    return {
        "anomaly_dates": anomaly_dates,
        "severity":      round(severity, 3),
        "direction":     direction,
    }


def slice_and_dice(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str = "date",
    dimension_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Rank which dimension values contributed most to a metric change between
    the first half and second half of the date range.

    For each dimension in dimension_cols, computes:
        delta_v = after_avg(v) - before_avg(v)
        contribution_pct = delta_v / total_delta * 100

    Contributions for all values of a single dimension sum to 100%.

    Args:
        df:             DataFrame with date_col, metric_col, and dimension_cols.
        metric_col:     Metric to slice (e.g. 'dau').
        date_col:       Date column name.
        dimension_cols: List of categorical columns to slice by (e.g. ['platform']).

    Returns:
        {
            ranked_dimensions: list[{
                dimension:        str,
                value:            str,
                delta:            float,
                contribution_pct: float,
            }]
        }
        Sorted by abs(contribution_pct) descending.
    """
    if dimension_cols is None:
        dimension_cols = []

    required = {metric_col, date_col} | set(dimension_cols)
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not dimension_cols:
        raise ValueError("dimension_cols must contain at least one column.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Split at the midpoint of the date range
    dates      = sorted(df[date_col].unique())
    mid_idx    = len(dates) // 2
    split_date = dates[mid_idx]

    before = df[df[date_col] <  split_date]
    after  = df[df[date_col] >= split_date]

    n_before = len(before[date_col].unique()) or 1
    n_after  = len(after[date_col].unique())  or 1

    total_before_avg = before[metric_col].sum() / n_before
    total_after_avg  = after[metric_col].sum()  / n_after
    total_delta      = total_after_avg - total_before_avg

    results: list[dict] = []

    for dim_col in dimension_cols:
        for val in sorted(df[dim_col].dropna().unique()):
            before_val = before[before[dim_col] == val][metric_col].sum() / n_before
            after_val  = after[after[dim_col]   == val][metric_col].sum() / n_after
            delta      = after_val - before_val

            contrib = (delta / total_delta * 100) if total_delta != 0 else 0.0

            results.append({
                "dimension":        dim_col,
                "value":            str(val),
                "delta":            round(float(delta), 2),
                "contribution_pct": round(float(contrib), 2),
            })

    results.sort(key=lambda r: abs(r["contribution_pct"]), reverse=True)

    return {"ranked_dimensions": results}
