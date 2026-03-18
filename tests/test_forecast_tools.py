"""
tests/test_forecast_tools.py — Unit tests for tools/forecast_tools.py

Uses base_metrics_daily_df fixture (30-day baseline, 14-day experiment drop).
"""

import sys
from unittest.mock import patch

import pandas as pd
import pytest

from tools.forecast_tools import forecast_baseline


def test_outside_ci_true(base_metrics_daily_df):
    """Experiment-period actuals (android drop) fall below forecast lower CI."""
    result = forecast_baseline(
        base_metrics_daily_df, metric_col="dau", date_col="date", forecast_days=14
    )
    assert result["outside_ci"] is True


def test_forecast_returns_dataframe(base_metrics_daily_df):
    """forecast_df is a non-empty DataFrame with expected columns."""
    result = forecast_baseline(
        base_metrics_daily_df, metric_col="dau", date_col="date", forecast_days=14
    )
    df = result["forecast_df"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 14
    assert {"yhat", "yhat_lower", "yhat_upper", "actual"}.issubset(df.columns)


def test_fallback_works_without_prophet(base_metrics_daily_df):
    """Rolling mean fallback runs cleanly when prophet import raises ImportError."""
    with patch.dict(sys.modules, {"prophet": None}):
        result = forecast_baseline(
            base_metrics_daily_df, metric_col="dau", date_col="date", forecast_days=14
        )
    assert result["method"] == "rolling_mean"
    assert isinstance(result["forecast_df"], pd.DataFrame)
    assert result["outside_ci"] is True
