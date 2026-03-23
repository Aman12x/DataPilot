"""
tools/decomposition_tools.py — DAU component breakdown: new/retained/resurrected/churned.

Pure Python, no LangGraph or Streamlit imports.
Input: pre-aggregated metrics_daily DataFrame (one row per date × platform × segment).
"""

from __future__ import annotations

import logging
import pandas as pd

from tools.schemas import ComponentStats, DecompositionResult, SegmentBreakdown

logger = logging.getLogger(__name__)


def decompose_dau(
    df: pd.DataFrame,
    date_col: str = "date",
    window_days: int = 28,
) -> DecompositionResult:
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

    if len(recent) < 3:
        logger.warning(
            "decompose_dau: 'recent' window has only %d row(s) — "
            "baseline split may be skewed; results may be unreliable.",
            len(recent),
        )

    components = {
        "new":         "new_users",
        "retained":    "retained_users",
        "resurrected": "resurrected_users",
        "churned":     "churned_users",
    }

    component_stats: dict[str, ComponentStats] = {}
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

        component_stats[key] = ComponentStats(
            time_series=ts,
            baseline_avg=round(baseline_avg, 2),
            recent_avg=round(recent_avg, 2),
            delta=round(delta, 2),
            pct_of_dau=round(float(daily[col].mean()) / mean_dau * 100, 2),
        )

    # Dominant = component most responsible for dragging DAU down.
    # If any component declined, pick the one with the most negative delta
    # (the "what broke" answer, regardless of what other components did).
    # Only fall back to largest absolute delta when all components are growing.
    declining = {k: v for k, v in deltas.items() if v < 0}
    if declining:
        dominant = min(declining, key=lambda k: declining[k])
    else:
        dominant = max(deltas, key=lambda k: abs(deltas[k]))

    return DecompositionResult(
        new=component_stats["new"],
        retained=component_stats["retained"],
        resurrected=component_stats["resurrected"],
        churned=component_stats["churned"],
        dominant_change_component=dominant,
        segments=[],   # DAU path uses named components; generic segments not used here
    )


def decompose_metric(
    df: pd.DataFrame,
    metric_col: str,
    segment_cols: list[str],
    date_col: str = "date",
    experiment_start=None,
) -> DecompositionResult:
    """
    Generic segment-based breakdown for any metric.

    For each (segment_col, segment_value) pair, computes before/after delta and
    contribution_pct relative to the total metric change.

    dominant_change_component is set to the "col=value" pair with the highest
    absolute contribution_pct.

    Args:
        df:               DataFrame with date_col, metric_col, and all segment_cols.
        metric_col:       Name of the metric column to decompose.
        segment_cols:     List of dimension columns to break down by.
        date_col:         Name of the date column.
        experiment_start: If provided, split before/after at this date string/Timestamp.
                          Otherwise, split at the calendar midpoint.

    Returns:
        DecompositionResult with segments list and dominant_change_component.
    """
    required = {date_col, metric_col} | set(segment_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Determine before/after split point
    if experiment_start is not None:
        split = pd.to_datetime(experiment_start)
    else:
        dates = df[date_col].sort_values().unique()
        split = dates[len(dates) // 2]

    before = df[df[date_col] < split]
    after  = df[df[date_col] >= split]

    if before.empty or after.empty:
        raise ValueError(
            f"decompose_metric: before/after split at {split} produced an empty partition."
        )

    segments: list[SegmentBreakdown] = []

    for col in segment_cols:
        if col not in df.columns:
            continue
        values = df[col].dropna().unique()
        for val in values:
            before_val = before[before[col] == val][metric_col]
            after_val  = after[after[col] == val][metric_col]
            if before_val.empty or after_val.empty:
                continue
            metric_before = float(before_val.mean())
            metric_after  = float(after_val.mean())
            delta         = metric_after - metric_before
            segments.append(SegmentBreakdown(
                segment_col=col,
                segment_value=str(val),
                metric_before=round(metric_before, 6),
                metric_after=round(metric_after, 6),
                delta=round(delta, 6),
                contribution_pct=0.0,  # filled in below
            ))

    # Compute contribution_pct: each segment's |delta| / sum of all |delta|
    total_abs_delta = sum(abs(s.delta) for s in segments) or 1.0
    for s in segments:
        s.contribution_pct = round(abs(s.delta) / total_abs_delta * 100, 2)

    # Warn when segments are moving in opposite directions and largely cancel out,
    # since contribution_pct shows movement magnitude, not net impact.
    if segments:
        positive_sum = sum(s.delta for s in segments if s.delta > 0)
        negative_sum = sum(s.delta for s in segments if s.delta < 0)
        if positive_sum > 0 and negative_sum < 0:
            net = positive_sum + negative_sum
            if abs(net) < 0.5 * total_abs_delta:
                logger.warning(
                    "decompose_metric: segments are moving in opposite directions "
                    "(net_delta=%.4f vs total_abs_movement=%.4f). "
                    "contribution_pct reflects movement magnitude, not net impact.",
                    net, total_abs_delta,
                )

    # Sort by descending |delta|
    segments.sort(key=lambda s: abs(s.delta), reverse=True)

    # dominant: pick the segment with the largest absolute movement,
    # but report "no_change" when all deltas are zero (uninformative otherwise).
    if segments and any(abs(s.delta) > 0 for s in segments):
        dominant = f"{segments[0].segment_col}={segments[0].segment_value}"
    elif segments:
        dominant = "no_change"
    else:
        dominant = "unknown"

    return DecompositionResult(
        dominant_change_component=dominant,
        segments=segments,
    )
