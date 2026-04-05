"""
tools/forecast_tools.py — Forecast baseline vs actuals.

Primary:  Prophet (if available)
Fallback: 7-day rolling mean + 2-sigma CI

Never hard-fails — degrades gracefully and sets 'warning' in result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.schemas import ForecastResult


def forecast_baseline(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str = "date",
    forecast_days: int = 14,
) -> ForecastResult:
    """
    Fit a baseline on all but the last `forecast_days` rows, then compare
    the actual values in that held-out window against the forecast CI.

    Args:
        df:            DataFrame with date_col and metric_col. May have
                       multiple rows per date (they are summed).
        metric_col:    Metric to forecast (e.g. 'dau').
        date_col:      Date column name.
        forecast_days: Number of most-recent days to treat as the
                       "experiment window" to compare against the forecast.

    Returns:
        {
            forecast_df:              pd.DataFrame,  # date, yhat, yhat_lower, yhat_upper, actual
            actual_vs_forecast_delta: float,         # mean(actual) - mean(yhat)
            outside_ci:               bool,          # True if ANY actual < yhat_lower or > yhat_upper
            method:                   str,           # 'prophet' | 'rolling_mean'
            warning:                  str | None,    # set if fallback was triggered
        }
    """
    if metric_col not in df.columns:
        raise ValueError(f"metric_col '{metric_col}' not found.")
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily = (
        df.groupby(date_col, as_index=False)[metric_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    # Gracefully reduce forecast window when the series is too short
    if len(daily) <= forecast_days:
        forecast_days = max(1, len(daily) // 3)

    train = daily.iloc[:-forecast_days].copy()
    test  = daily.iloc[-forecast_days:].copy()

    if len(train) < 3:
        # Not enough history to fit any baseline — return a neutral result
        return ForecastResult(
            forecast_df=pd.DataFrame(),
            actual_vs_forecast_delta=0.0,
            outside_ci=False,
            method="rolling_mean",
            warning=f"Series too short ({len(daily)} dates) for a reliable forecast.",
        )

    try:
        return _forecast_prophet(train, test, date_col, metric_col)
    except ImportError:
        r = _forecast_rolling(train, test, date_col, metric_col)
        return ForecastResult(
            forecast_df=r.forecast_df,
            actual_vs_forecast_delta=r.actual_vs_forecast_delta,
            outside_ci=r.outside_ci,
            method=r.method,
            warning="Prophet not installed — used rolling mean fallback.",
        )
    except Exception as e:
        r = _forecast_rolling(train, test, date_col, metric_col)
        return ForecastResult(
            forecast_df=r.forecast_df,
            actual_vs_forecast_delta=r.actual_vs_forecast_delta,
            outside_ci=r.outside_ci,
            method=r.method,
            warning=f"Prophet failed ({e}) — used rolling mean fallback.",
        )


# ── Prophet ────────────────────────────────────────────────────────────────────

def _forecast_prophet(
    train: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str,
    metric_col: str,
) -> ForecastResult:
    import io, sys
    _stderr = sys.stderr
    sys.stderr = io.StringIO()          # silence Prophet's "Importing plotly failed" noise
    try:
        from prophet import Prophet     # raises ImportError if not installed
    finally:
        sys.stderr = _stderr

    prophet_train = train.rename(columns={date_col: "ds", metric_col: "y"})

    model = Prophet(
        interval_width=0.95,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    model.fit(prophet_train)

    future   = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    # Align forecast to test window
    test_forecast = forecast.tail(len(test)).reset_index(drop=True)
    test_reset    = test.reset_index(drop=True)

    actual    = test_reset[metric_col].astype(float).values
    yhat      = test_forecast["yhat"].values
    yhat_lo   = test_forecast["yhat_lower"].values
    yhat_hi   = test_forecast["yhat_upper"].values

    delta      = float(actual.mean() - yhat.mean())
    outside_ci = bool(np.any(actual < yhat_lo) or np.any(actual > yhat_hi))

    forecast_df = pd.DataFrame({
        date_col:     test_reset[date_col].values,
        "yhat":       yhat,
        "yhat_lower": yhat_lo,
        "yhat_upper": yhat_hi,
        "actual":     actual,
    })

    return ForecastResult(
        forecast_df=forecast_df,
        actual_vs_forecast_delta=round(delta, 2),
        outside_ci=outside_ci,
        method="prophet",
        warning=None,
    )


# ── Rolling mean fallback ──────────────────────────────────────────────────────

def _forecast_rolling(
    train: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str,
    metric_col: str,
) -> ForecastResult:
    series = train[metric_col].astype(float)

    window       = min(7, len(series))
    rolling_mean = float(series.iloc[-window:].mean())
    rolling_std  = float(series.std(ddof=1)) if len(series) > 1 else 0.0

    ci_lower = rolling_mean - 2.0 * rolling_std
    ci_upper = rolling_mean + 2.0 * rolling_std

    test_reset = test.reset_index(drop=True)
    actual     = test_reset[metric_col].astype(float).values

    delta      = float(actual.mean() - rolling_mean)
    outside_ci = bool(np.any(actual < ci_lower) or np.any(actual > ci_upper))

    forecast_df = pd.DataFrame({
        date_col:     test_reset[date_col].values,
        "yhat":       np.full(len(test), rolling_mean),
        "yhat_lower": np.full(len(test), ci_lower),
        "yhat_upper": np.full(len(test), ci_upper),
        "actual":     actual,
    })

    return ForecastResult(
        forecast_df=forecast_df,
        actual_vs_forecast_delta=round(delta, 2),
        outside_ci=outside_ci,
        method="rolling_mean",
        warning=None,
    )
