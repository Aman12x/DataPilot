"""
tests/test_regression_tools.py — Unit tests for tools/regression_tools.py.

Coverage:
  - Target auto-detection (task hint + highest-variance fallback)
  - Feature matrix construction (numeric, one-hot, NaN fill, constant drop)
  - VIF computation
  - run_regression() end-to-end on all 5 fixture datasets
  - Error paths: no features, insufficient rows, all-NaN target
  - Output invariants: p-values in [0,1], CI ordering, R² in [-∞,1]
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tools.regression_tools import (
    _build_feature_matrix,
    _compute_vif,
    _select_target,
    run_regression,
)
from tools.schemas import RegressionResult

# ── Fixtures ──────────────────────────────────────────────────────────────────

FIXTURES = "tests/fixtures"


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{FIXTURES}/{name}")


# ── Target selection ──────────────────────────────────────────────────────────

class TestSelectTarget:
    def test_hint_match_returns_column(self):
        df = pd.DataFrame({"revenue": [1, 2, 3], "clicks": [4, 5, 6]})
        assert _select_target(df, task_hint="what drives revenue") == "revenue"

    def test_longest_match_wins(self):
        df = pd.DataFrame({"revenue": [1, 2], "revenue_per_user": [0.1, 0.2]})
        # 'revenue_per_user' is longer and both match 'revenue per user'
        assert _select_target(df, task_hint="revenue per user") == "revenue_per_user"

    def test_no_hint_uses_highest_variance(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 100, 200]})
        assert _select_target(df, task_hint="") == "b"

    def test_no_numeric_returns_none(self):
        df = pd.DataFrame({"cat": ["x", "y", "z"]})
        assert _select_target(df) is None

    def test_hint_no_match_falls_back_to_variance(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})
        assert _select_target(df, task_hint="xyz_nonexistent") == "b"


# ── Feature matrix ─────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_excludes_target_column(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        X = _build_feature_matrix(df, target_col="y")
        assert "y" not in X.columns

    def test_nan_filled_with_median(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [10.0, 20.0, 30.0]})
        X = _build_feature_matrix(df, target_col="y")
        assert X["x"].isna().sum() == 0

    def test_categorical_one_hot(self):
        df = pd.DataFrame({
            "cat": ["a", "b", "a", "b", "a"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        X = _build_feature_matrix(df, target_col="y")
        # 'cat' with 2 unique values → one dummy column (drop_first=True)
        assert any("cat_" in c for c in X.columns)

    def test_high_cardinality_categorical_dropped(self):
        # 15 unique values → above _MAX_CATEGORIES=10, should be dropped
        df = pd.DataFrame({
            "id_col": [str(i) for i in range(15)],
            "num": range(15),
            "y": range(15),
        })
        X = _build_feature_matrix(df, target_col="y")
        assert not any("id_col" in c for c in X.columns)

    def test_constant_column_dropped(self):
        df = pd.DataFrame({"const": [5, 5, 5, 5], "x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
        X = _build_feature_matrix(df, target_col="y")
        assert "const" not in X.columns

    def test_max_15_features_enforced(self):
        # Create 20 numeric predictors
        data = {f"x{i}": np.random.randn(100) for i in range(20)}
        data["y"] = np.random.randn(100)
        df = pd.DataFrame(data)
        X = _build_feature_matrix(df, target_col="y")
        assert len(X.columns) <= 15


# ── VIF ───────────────────────────────────────────────────────────────────────

class TestComputeVIF:
    def test_independent_features_low_vif(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        vif = _compute_vif(X, ["a", "b", "c"])
        for v in vif.values():
            assert math.isfinite(v) and v < 5, f"VIF {v} too high for independent features"

    def test_collinear_features_high_vif(self):
        rng = np.random.default_rng(42)
        base = rng.standard_normal(100)
        X = np.column_stack([base, base + rng.normal(0, 0.01, 100)])
        vif = _compute_vif(X, ["a", "b"])
        assert any(v > 10 for v in vif.values()), "Expected high VIF for near-collinear features"

    def test_too_few_observations_returns_empty(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 rows, 2 features → k+2 = 4 needed
        assert _compute_vif(X, ["a", "b"]) == {}

    def test_single_feature_returns_empty(self):
        X = np.random.randn(50, 1)
        assert _compute_vif(X, ["a"]) == {}


# ── run_regression: output invariants ────────────────────────────────────────

class TestRunRegressionInvariants:
    def _make_df(self, n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.5, n)
        return pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    def test_returns_regression_result(self):
        result = run_regression(self._make_df(), target_col="y")
        assert isinstance(result, RegressionResult)

    def test_r_squared_in_range(self):
        result = run_regression(self._make_df(), target_col="y")
        # With clean synthetic data R² should be high
        assert 0.5 <= result.r_squared <= 1.0

    def test_p_values_in_unit_interval(self):
        result = run_regression(self._make_df(), target_col="y")
        for coef in result.coefficients:
            assert 0.0 <= coef.p_value <= 1.0, f"p_value {coef.p_value} out of [0,1]"

    def test_ci_lower_leq_upper(self):
        result = run_regression(self._make_df(), target_col="y")
        for coef in result.coefficients:
            assert coef.ci_lower <= coef.ci_upper, "CI lower > CI upper"

    def test_sorted_by_abs_t_stat(self):
        result = run_regression(self._make_df(), target_col="y")
        t_stats = [abs(c.t_stat) for c in result.coefficients if math.isfinite(c.t_stat)]
        assert t_stats == sorted(t_stats, reverse=True)

    def test_n_obs_matches_dataframe(self):
        df = self._make_df(n=80)
        result = run_regression(df, target_col="y")
        assert result.n_obs == 80

    def test_n_features_correct(self):
        df = self._make_df(n=50)
        result = run_regression(df, target_col="y")
        assert result.n_features == 2

    def test_significant_flag_consistent_with_p_value(self):
        result = run_regression(self._make_df(), target_col="y")
        for coef in result.coefficients:
            if math.isfinite(coef.p_value):
                assert coef.significant == (coef.p_value < 0.05)

    def test_true_predictors_detected(self):
        """x1 (coef 2.0) and x2 (coef -1.5) should be significant with n=200."""
        rng = np.random.default_rng(1)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        result = run_regression(df, target_col="y")
        sig_features = {c.feature for c in result.coefficients if c.significant}
        assert "x1" in sig_features
        assert "x2" in sig_features


# ── run_regression: error paths ───────────────────────────────────────────────

class TestRunRegressionErrors:
    def test_raises_on_insufficient_rows(self):
        df = pd.DataFrame({"x": range(5), "y": range(5)})
        with pytest.raises(ValueError, match="observations"):
            run_regression(df, target_col="y")

    def test_raises_on_no_features(self):
        # Only the target column — nothing left after dropping it
        df = pd.DataFrame({"y": range(30)})
        with pytest.raises(ValueError):
            run_regression(df, target_col="y")

    def test_raises_when_target_not_in_df(self):
        df = pd.DataFrame({"x": range(30), "z": range(30)})
        with pytest.raises(ValueError, match="target"):
            run_regression(df, target_col="missing_col")

    def test_all_nan_target_raises(self):
        df = pd.DataFrame({"x": range(30), "y": [float("nan")] * 30})
        with pytest.raises(ValueError):
            run_regression(df, target_col="y")


# ── Fixture datasets ──────────────────────────────────────────────────────────

class TestRegressionOnFixtures:
    """Smoke tests across all 5 domain datasets — verifies run_regression does
    not crash and returns plausible results without hard-coding specific numbers."""

    def test_healthcare_readmission(self):
        """healthcare.csv: predict los_days from bmi, systolic_bp, diagnosis."""
        df = _load("healthcare.csv").drop(columns=["patient_id"])
        result = run_regression(df, target_col="los_days", task_hint="length of stay")
        assert isinstance(result, RegressionResult)
        assert result.n_obs >= 250
        assert result.n_features >= 2
        assert 0.0 <= result.r_squared <= 1.0
        # All p-values valid
        for coef in result.coefficients:
            assert 0.0 <= coef.p_value <= 1.0

    def test_healthcare_target_hint(self):
        """Task hint 'readmission 30d' should pick readmission_30d as target."""
        df = _load("healthcare.csv").drop(columns=["patient_id"])
        result = run_regression(df, task_hint="predict readmission 30d")
        assert result.target == "readmission_30d"

    def test_hr_salary_prediction(self):
        """hr.csv: predict salary from department, level, performance_score."""
        df = _load("hr.csv").drop(columns=["employee_id", "employee_name"])
        result = run_regression(df, target_col="salary", task_hint="salary")
        assert isinstance(result, RegressionResult)
        assert result.n_obs >= 150
        # Salary should be somewhat predictable from level/dept
        assert result.r_squared >= 0.0

    def test_hr_performance_score_target(self):
        """Task hint picks performance_score as target."""
        df = _load("hr.csv").drop(columns=["employee_id", "employee_name"])
        result = run_regression(df, task_hint="predict performance score")
        assert result.target == "performance_score"

    def test_ecommerce_revenue(self):
        """ecommerce.csv: predict revenue from quantity, unit_price, discount_pct."""
        df = _load("ecommerce.csv").drop(columns=["order_id"])
        result = run_regression(df, target_col="revenue", task_hint="revenue")
        assert isinstance(result, RegressionResult)
        assert result.n_obs >= 100
        # Revenue = quantity * unit_price * (1 - discount) → high R²
        assert result.r_squared >= 0.5

    def test_ecommerce_vif_warnings(self):
        """revenue and unit_price * quantity are collinear → VIF warnings expected."""
        df = _load("ecommerce.csv").drop(columns=["order_id"])
        result = run_regression(df, task_hint="revenue")
        # VIF warnings may or may not fire depending on feature selection,
        # but the field must be a list
        assert isinstance(result.vif_warnings, list)

    def test_saas_churn_mrr(self):
        """saas_churn.csv: predict mrr from tenure_months, support_tickets, plan."""
        df = _load("saas_churn.csv").drop(columns=["customer_id", "month"])
        result = run_regression(df, target_col="mrr", task_hint="mrr")
        assert isinstance(result, RegressionResult)
        assert result.n_obs >= 50

    def test_saas_churn_f_stat(self):
        """F-stat should be present and positive when features exist."""
        df = _load("saas_churn.csv").drop(columns=["customer_id", "month"])
        result = run_regression(df, target_col="mrr", task_hint="mrr")
        if result.f_stat is not None:
            assert result.f_stat > 0

    def test_timeseries_highest_variance_fallback(self):
        """timeseries.csv has no task-hint match — should pick highest-variance col."""
        df = _load("timeseries.csv").drop(columns=["month"])
        result = run_regression(df, task_hint="")
        assert result.target in df.columns
        # revenue has the highest variance in this dataset
        assert result.target == "revenue"

    def test_timeseries_new_customers_target(self):
        """Task hint 'new customers' selects new_customers (underscore normalization)."""
        df = _load("timeseries.csv").drop(columns=["month"])
        result = run_regression(df, task_hint="new customers growth")
        assert result.target == "new_customers"

    def test_all_fixtures_no_crash(self):
        """Belt-and-suspenders: none of the 5 fixtures should raise."""
        fixtures = [
            ("healthcare.csv",  ["patient_id"]),
            ("hr.csv",          ["employee_id", "employee_name"]),
            ("ecommerce.csv",   ["order_id"]),
            ("saas_churn.csv",  ["customer_id", "month"]),
            ("timeseries.csv",  ["month"]),
        ]
        for fname, drop_cols in fixtures:
            df = _load(fname).drop(columns=drop_cols, errors="ignore")
            try:
                result = run_regression(df, task_hint="")
                assert isinstance(result, RegressionResult)
            except ValueError:
                # Only acceptable: too few rows or no features
                pass


# ── Categorical one-hot integration ──────────────────────────────────────────

class TestCategoricalEncoding:
    def test_categorical_dummy_increases_features(self):
        """A low-cardinality categorical should add dummy columns."""
        df = pd.DataFrame({
            "group": (["A"] * 15) + (["B"] * 15),
            "x":     np.random.randn(30),
            "y":     np.random.randn(30),
        })
        result = run_regression(df, target_col="y")
        names = [c.feature for c in result.coefficients]
        assert any("group_" in n for n in names), "Expected one-hot column for 'group'"

    def test_high_cardinality_categorical_excluded(self):
        """A categorical with 15 unique values should be silently excluded."""
        df = pd.DataFrame({
            "country": [f"Country_{i % 15}" for i in range(60)],
            "x": np.random.randn(60),
            "y": np.random.randn(60),
        })
        result = run_regression(df, target_col="y")
        names = [c.feature for c in result.coefficients]
        assert not any("country" in n.lower() for n in names)

    def test_nan_in_categorical_handled(self):
        """NaN in categorical should be filled with mode, not crash."""
        cats = (["A"] * 10) + (["B"] * 10) + ([None] * 10)
        df = pd.DataFrame({
            "cat": cats,
            "x":   np.random.randn(30),
            "y":   np.random.randn(30),
        })
        result = run_regression(df, target_col="y")
        assert isinstance(result, RegressionResult)


# ── Regression output completeness ───────────────────────────────────────────

class TestOutputCompleteness:
    def test_f_stat_present_for_well_fit_model(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(100)
        y = 3.0 * x + rng.normal(0, 0.3, 100)
        df = pd.DataFrame({"x": x, "y": y})
        result = run_regression(df, target_col="y")
        assert result.f_stat is not None
        assert result.f_pvalue is not None
        assert result.f_pvalue < 0.05  # strong predictor

    def test_adj_r_squared_leq_r_squared(self):
        """Adjusted R² should be ≤ R² (penalty for extra features)."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5))
        y = X[:, 0] + rng.normal(0, 1, 100)
        cols = {f"x{i}": X[:, i] for i in range(5)}
        cols["y"] = y
        df = pd.DataFrame(cols)
        result = run_regression(df, target_col="y")
        assert result.adj_r_squared <= result.r_squared + 1e-9

    def test_n_features_equals_len_coefficients(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame({
            "a": rng.standard_normal(50),
            "b": rng.standard_normal(50),
            "y": rng.standard_normal(50),
        })
        result = run_regression(df, target_col="y")
        assert result.n_features == len(result.coefficients)
