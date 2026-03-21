"""
tests/test_decomposition_tools.py — Unit tests for tools/decomposition_tools.py

Uses base_metrics_daily_df fixture (44 days, android drop at day 30).
"""

import pytest

from tools.decomposition_tools import decompose_dau


def test_components_sum_to_dau(base_metrics_daily_df):
    """new + retained + resurrected ≈ total DAU (within 10% — int rounding only)."""
    result = decompose_dau(base_metrics_daily_df, date_col="date", window_days=30)

    new_avg  = result.new.recent_avg
    ret_avg  = result.retained.recent_avg
    res_avg  = result.resurrected.recent_avg

    # Reconstruct recent-period mean DAU from the daily aggregates
    import pandas as pd
    df = base_metrics_daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily_dau = df.groupby("date")["dau"].sum().sort_values()
    recent_dau_avg = float(daily_dau.iloc[30:].mean())

    component_sum = new_avg + ret_avg + res_avg
    assert abs(component_sum - recent_dau_avg) / recent_dau_avg < 0.10


def test_dominant_component_is_new(base_metrics_daily_df):
    """dominant_change_component is 'new' when new_users fraction drops at experiment start."""
    result = decompose_dau(base_metrics_daily_df, date_col="date", window_days=30)
    assert "new" in result.dominant_change_component
