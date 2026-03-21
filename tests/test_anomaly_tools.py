"""
tests/test_anomaly_tools.py — Unit tests for tools/anomaly_tools.py

Uses base_metrics_daily_df fixture (44 days, android step-down at day 30).
"""

import pytest

from tools.anomaly_tools import detect_anomaly, slice_and_dice


# ── detect_anomaly ─────────────────────────────────────────────────────────────

def test_detects_step_down(base_metrics_daily_df):
    """Anomaly dates include the experiment start date (2024-01-31 = day index 30)."""
    result = detect_anomaly(base_metrics_daily_df, metric_col="dau", date_col="date")
    assert len(result.anomaly_dates) > 0
    assert "2024-01-31" in result.anomaly_dates


def test_direction_is_drop(base_metrics_daily_df):
    """Step-down fixture produces direction == 'drop'."""
    result = detect_anomaly(base_metrics_daily_df, metric_col="dau", date_col="date")
    assert result.direction == "drop"


# ── slice_and_dice ─────────────────────────────────────────────────────────────

def test_slice_ranks_android_first(base_metrics_daily_df):
    """Android contributes most to the DAU drop, so it ranks first."""
    result = slice_and_dice(
        base_metrics_daily_df,
        metric_col="dau",
        date_col="date",
        dimension_cols=["platform"],
    )
    assert result.ranked_dimensions[0].value == "android"


def test_slice_contribution_sums_to_100(base_metrics_daily_df):
    """Contributions for all values of a single dimension sum to ~100%."""
    result = slice_and_dice(
        base_metrics_daily_df,
        metric_col="dau",
        date_col="date",
        dimension_cols=["platform"],
    )
    platform_entries = [
        r for r in result.ranked_dimensions if r.dimension == "platform"
    ]
    total = sum(r.contribution_pct for r in platform_entries)
    assert abs(total - 100.0) < 1.0
