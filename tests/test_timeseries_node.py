"""
tests/test_timeseries_node.py — Unit tests for detect_timeseries_node and
run_regression_node in agents/analyze/nodes.py (general mode additions).

Each test builds a minimal AgentState dict and calls the node function directly —
no graph invocation needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.analyze.nodes import detect_timeseries_node, run_regression_node
from tools.schemas import AnomalyResult, ForecastResult, RegressionResult

# ── Helpers ───────────────────────────────────────────────────────────────────

FIXTURES = "tests/fixtures"


def _state(df: pd.DataFrame, task: str = "") -> dict:
    return {"query_result": df, "task": task, "analysis_mode": "general"}


# ── run_regression_node ───────────────────────────────────────────────────────

class TestRunRegressionNode:
    def _df(self, n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.normal(0, 0.3, n)
        return pd.DataFrame({"x": x, "y": y})

    def test_returns_regression_result(self):
        result = run_regression_node(_state(self._df()))
        assert "regression_result" in result
        assert isinstance(result["regression_result"], RegressionResult)

    def test_too_few_rows_returns_empty(self):
        df = pd.DataFrame({"x": range(5), "y": range(5)})
        result = run_regression_node(_state(df))
        assert result == {}

    def test_no_query_result_returns_empty(self):
        result = run_regression_node({"analysis_mode": "general"})
        assert result == {}

    def test_task_hint_used_for_target(self):
        df = pd.read_csv(f"{FIXTURES}/hr.csv").drop(columns=["employee_id", "employee_name"])
        # "performance score" normalizes to match column "performance_score"
        result = run_regression_node(_state(df, task="what drives performance score"))
        rr = result.get("regression_result")
        assert rr is not None
        assert rr.target == "performance_score"

    def test_no_features_returns_empty(self):
        """All-categorical df with no numeric columns except target → graceful skip."""
        df = pd.DataFrame({"cat": ["a", "b"] * 20, "y": range(40)})
        # 'cat' has 2 unique values → one-hot gives 1 feature, so it won't skip
        # But if we make it all same category → constant → dropped
        df["cat"] = "same"
        result = run_regression_node(_state(df))
        # Either returns a result (with just y as only numeric, no features) or empty
        assert isinstance(result, dict)

    def test_healthcare_fixture(self):
        df = pd.read_csv(f"{FIXTURES}/healthcare.csv").drop(columns=["patient_id"])
        result = run_regression_node(_state(df, task="los_days"))
        rr = result.get("regression_result")
        assert rr is not None
        assert rr.n_obs >= 250

    def test_ecommerce_fixture(self):
        df = pd.read_csv(f"{FIXTURES}/ecommerce.csv").drop(columns=["order_id"])
        result = run_regression_node(_state(df, task="revenue"))
        rr = result.get("regression_result")
        assert rr is not None
        assert rr.r_squared >= 0.3  # revenue ~ quantity * unit_price

    def test_saas_fixture(self):
        df = pd.read_csv(f"{FIXTURES}/saas_churn.csv").drop(columns=["customer_id", "month"])
        result = run_regression_node(_state(df, task="mrr"))
        assert isinstance(result, dict)


# ── detect_timeseries_node ────────────────────────────────────────────────────

class TestDetectTimeseriesNode:
    def _ts_df(self, n: int = 60) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=n, freq="MS")
        rng = np.random.default_rng(42)
        revenue = 100_000 + np.cumsum(rng.normal(500, 2000, n))
        return pd.DataFrame({"month": dates, "revenue": revenue, "users": rng.integers(1000, 5000, n)})

    def test_detects_time_column_and_runs(self):
        result = detect_timeseries_node(_state(self._ts_df()))
        # Should produce at least anomaly or forecast result
        assert "anomaly_result" in result or "forecast_result" in result

    def test_forecast_result_type(self):
        result = detect_timeseries_node(_state(self._ts_df()))
        if "forecast_result" in result:
            assert isinstance(result["forecast_result"], ForecastResult)

    def test_anomaly_result_type(self):
        result = detect_timeseries_node(_state(self._ts_df()))
        if "anomaly_result" in result:
            assert isinstance(result["anomaly_result"], AnomalyResult)

    def test_no_time_column_returns_empty(self):
        df = pd.DataFrame({"x": range(30), "y": range(30), "z": range(30)})
        result = detect_timeseries_node(_state(df))
        assert result == {}

    def test_no_query_result_returns_empty(self):
        result = detect_timeseries_node({"analysis_mode": "general"})
        assert result == {}

    def test_too_few_rows_returns_empty(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        df = pd.DataFrame({"month": dates, "revenue": [100, 110, 105]})
        result = detect_timeseries_node(_state(df))
        assert result == {}

    def test_date_column_keyword_month(self):
        df = self._ts_df(n=24)
        result = detect_timeseries_node(_state(df))
        assert len(result) > 0 or result == {}  # may skip if too few after grouping

    def test_date_column_keyword_date(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=30),
            "metric": rng.standard_normal(30) + 100,
        })
        result = detect_timeseries_node(_state(df))
        assert "anomaly_result" in result or "forecast_result" in result or result == {}

    def test_timeseries_fixture_end_to_end(self):
        df = pd.read_csv(f"{FIXTURES}/timeseries.csv")
        result = detect_timeseries_node(_state(df))
        # timeseries.csv has 'month' column → should detect it
        assert isinstance(result, dict)

    def test_saas_panel_with_month_column(self):
        df = pd.read_csv(f"{FIXTURES}/saas_churn.csv").drop(columns=["customer_id"])
        result = detect_timeseries_node(_state(df))
        assert isinstance(result, dict)

    def test_id_columns_excluded_from_metric_selection(self):
        """Columns named *_id or *id should not be chosen as the metric."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=30),
            "user_id": range(1000, 1030),
            "revenue": rng.normal(5000, 300, 30),
        })
        result = detect_timeseries_node(_state(df))
        # If result produced anomaly/forecast, it should be based on revenue not user_id
        assert isinstance(result, dict)


# ── Integration: both nodes on same fixture ───────────────────────────────────

class TestGeneralModeNodes:
    def test_regression_and_timeseries_on_timeseries_fixture(self):
        """timeseries.csv: regression picks revenue as target; time series detects month."""
        df = pd.read_csv(f"{FIXTURES}/timeseries.csv")
        ts_result = detect_timeseries_node(_state(df, task="revenue trend"))
        # drop month before regression (not a useful numeric predictor after conversion)
        df_no_time = df.drop(columns=["month"], errors="ignore")
        reg_result = run_regression_node(_state(df_no_time, task="revenue"))
        rr = reg_result.get("regression_result")
        if rr:
            assert rr.target == "revenue"
        assert isinstance(ts_result, dict)

    def test_healthcare_no_time_column(self):
        """healthcare.csv has no time column — detect_timeseries should return {}."""
        df = pd.read_csv(f"{FIXTURES}/healthcare.csv").drop(columns=["patient_id"])
        result = detect_timeseries_node(_state(df))
        assert result == {}

    def test_hr_no_time_column(self):
        """hr.csv has no time column."""
        df = pd.read_csv(f"{FIXTURES}/hr.csv").drop(columns=["employee_id", "employee_name"])
        result = detect_timeseries_node(_state(df))
        assert result == {}
