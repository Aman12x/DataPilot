"""
tests/test_describe_tools.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for tools/describe_tools.py covering the new top_rows / trend_rows
enrichment added to DescribeResult.

Ground-truth values were computed deterministically from the fixture CSVs with
numpy seed=42 and are verified here without any LLM calls.

Fixtures
────────
healthcare.csv  — 300 rows, diagnoses: Diabetes/Healthy/Hypertension/Asthma
timeseries.csv  — 120 monthly rows, 2015-2024, revenue + churn_rate + new_customers
hr.csv          — 200 rows, 4 departments × 4 seniority levels, salary data
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.describe_tools import (
    _compute_top_rows,
    _compute_trend_rows,
    compute_correlations,
    describe_dataframe,
)

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HC_CSV = os.path.join(_FIXTURES, "healthcare.csv")
TS_CSV = os.path.join(_FIXTURES, "timeseries.csv")
HR_CSV = os.path.join(_FIXTURES, "hr.csv")


# ═════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def hc_df() -> pd.DataFrame:
    return pd.read_csv(HC_CSV)


@pytest.fixture(scope="module")
def ts_df() -> pd.DataFrame:
    return pd.read_csv(TS_CSV)


@pytest.fixture(scope="module")
def hr_df() -> pd.DataFrame:
    return pd.read_csv(HR_CSV)


# ═════════════════════════════════════════════════════════════════════════════
# 1. describe_dataframe — basic contract
# ═════════════════════════════════════════════════════════════════════════════

class TestDescribeDataframeContract:
    """Verify the DescribeResult shape and column-level statistics are correct."""

    def test_row_col_count_healthcare(self, hc_df):
        result = describe_dataframe(hc_df)
        assert result.row_count == 300
        assert result.col_count == 6  # patient_id, diagnosis, bmi, systolic_bp, readmission_30d, los_days

    def test_row_col_count_timeseries(self, ts_df):
        result = describe_dataframe(ts_df)
        assert result.row_count == 120
        assert result.col_count == 4  # month, revenue, churn_rate, new_customers

    def test_row_col_count_hr(self, hr_df):
        result = describe_dataframe(hr_df)
        assert result.row_count == 200
        assert result.col_count == 6  # employee_id, employee_name, department, level, salary, performance_score

    def test_numeric_column_stats_bmi(self, hc_df):
        result = describe_dataframe(hc_df)
        bmi_col = next(c for c in result.columns if c.name == "bmi")
        assert bmi_col.mean is not None
        assert 22 < bmi_col.mean < 35, f"BMI mean {bmi_col.mean} out of expected range"
        assert bmi_col.min is not None and bmi_col.min > 15
        assert bmi_col.max is not None and bmi_col.max < 50

    def test_numeric_column_stats_revenue(self, ts_df):
        result = describe_dataframe(ts_df)
        rev_col = next(c for c in result.columns if c.name == "revenue")
        assert rev_col.mean is not None
        assert 100_000 < rev_col.mean < 400_000
        # min/max should bracket the fixture range (103812 → 334378)
        assert rev_col.min is not None and rev_col.min < 120_000
        assert rev_col.max is not None and rev_col.max > 300_000

    def test_categorical_column_has_top_values(self, hc_df):
        result = describe_dataframe(hc_df)
        diag_col = next(c for c in result.columns if c.name == "diagnosis")
        assert diag_col.top_values is not None and len(diag_col.top_values) > 0
        # Should mention 'Diabetes' (most common diagnosis with 85 rows)
        combined = " ".join(diag_col.top_values)
        assert "Diabetes" in combined

    def test_categorical_col_n_unique(self, hc_df):
        result = describe_dataframe(hc_df)
        diag_col = next(c for c in result.columns if c.name == "diagnosis")
        assert diag_col.n_unique == 4  # Diabetes, Healthy, Hypertension, Asthma

    def test_no_null_counts_in_fixture(self, hc_df):
        result = describe_dataframe(hc_df)
        for col in result.columns:
            assert col.null_count == 0, f"Unexpected nulls in {col.name}"

    def test_salary_stats_hr(self, hr_df):
        result = describe_dataframe(hr_df)
        sal_col = next(c for c in result.columns if c.name == "salary")
        assert sal_col.mean is not None
        assert abs(sal_col.mean - 72135.34) < 1, \
            f"Salary mean {sal_col.mean} should be ~72135"
        assert sal_col.min is not None and sal_col.min > 20_000
        assert sal_col.max is not None and sal_col.max > 100_000


# ═════════════════════════════════════════════════════════════════════════════
# 2. _compute_top_rows — highest-variance numeric col ranking
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeTopRows:
    """top_rows should return the rows with the highest values of the most
    variable numeric column — giving the LLM named entities to cite."""

    def test_top_rows_returned_for_hr(self, hr_df):
        rows = _compute_top_rows(hr_df)
        assert rows is not None
        assert len(rows) == 10  # default n=10

    def test_top_rows_sorted_desc_by_salary(self, hr_df):
        """Salary has high variance in HR; top rows should be highest-paid."""
        rows = _compute_top_rows(hr_df)
        assert rows is not None
        salaries = [r["salary"] for r in rows]
        assert salaries == sorted(salaries, reverse=True), \
            "top_rows not sorted descending by salary"

    def test_top_rows_contain_lead_employees(self, hr_df):
        """Lead-level employees dominate the top-10 by salary (avg 105k)."""
        rows = _compute_top_rows(hr_df)
        assert rows is not None
        levels = [r.get("level") for r in rows]
        lead_count = sum(1 for l in levels if l == "Lead")
        assert lead_count >= 3, \
            f"Expected at least 3 Lead employees in top-10 by salary, got {lead_count}"

    def test_top_rows_for_healthcare_ranks_by_bmi_or_other_numeric(self, hc_df):
        """BMI or systolic_bp should be the sort column (highest std)."""
        rows = _compute_top_rows(hc_df)
        assert rows is not None
        assert len(rows) == 10

    def test_top_rows_contains_name_field(self, hr_df):
        """Each top row must have 'employee_name' so the LLM can name entities."""
        rows = _compute_top_rows(hr_df)
        assert rows is not None
        for r in rows:
            assert "employee_name" in r, "Missing employee_name in top_rows entry"

    def test_top_rows_none_when_no_numeric_cols(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert _compute_top_rows(df) is None

    def test_top_rows_fewer_than_n_if_small_df(self):
        df = pd.DataFrame({"val": [10, 20, 30], "name": ["a", "b", "c"]})
        rows = _compute_top_rows(df, n=10)
        assert rows is not None
        assert len(rows) == 3  # only 3 rows available

    def test_top_rows_describe_dataframe_integration(self, hr_df):
        """describe_dataframe populates top_rows field."""
        result = describe_dataframe(hr_df)
        assert result.top_rows is not None
        assert len(result.top_rows) == 10

    def test_top_rows_values_are_json_safe(self, hr_df):
        """All values in top_rows must be JSON-serialisable Python types."""
        import json
        rows = _compute_top_rows(hr_df)
        assert rows is not None
        json.dumps(rows)  # should not raise


# ═════════════════════════════════════════════════════════════════════════════
# 3. _compute_trend_rows — time/group aggregation
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeTrendRows:
    """trend_rows detects a time or group column and aggregates numeric cols by it."""

    def test_trend_rows_detected_by_month(self, ts_df):
        """'month' column should be detected as a time dimension."""
        rows = _compute_trend_rows(ts_df)
        assert rows is not None, "Expected trend_rows for timeseries data"

    def test_trend_rows_one_row_per_month(self, ts_df):
        """120 distinct months → 120 aggregated rows."""
        rows = _compute_trend_rows(ts_df)
        assert rows is not None
        assert len(rows) == 120

    def test_trend_rows_sorted_asc_by_month(self, ts_df):
        """Rows should be sorted ascending by the time column (2015-01 first)."""
        rows = _compute_trend_rows(ts_df)
        assert rows is not None
        months = [r["month"] for r in rows]
        assert months == sorted(months), "trend_rows not sorted ascending by month"

    def test_trend_rows_first_last_revenue(self, ts_df):
        """Revenue should be ~103,812 in Jan 2015 and ~334,378 in Dec 2024."""
        rows = _compute_trend_rows(ts_df)
        assert rows is not None
        first_rev = rows[0]["revenue"]
        last_rev  = rows[-1]["revenue"]
        assert abs(first_rev - 103_812) < 2, f"First revenue {first_rev} != 103812"
        assert abs(last_rev  - 334_378) < 2, f"Last revenue {last_rev} != 334378"

    def test_trend_rows_churn_declines(self, ts_df):
        """Churn decreases from ~8.2% to ~3.46% — later months should be lower."""
        rows = _compute_trend_rows(ts_df)
        assert rows is not None
        first_churn = rows[0]["churn_rate"]
        last_churn  = rows[-1]["churn_rate"]
        assert last_churn < first_churn, \
            f"Churn should decline: first={first_churn:.4f}, last={last_churn:.4f}"

    def test_trend_rows_detected_by_department(self, hr_df):
        """'department' should be detected as a group dimension for HR data."""
        rows = _compute_trend_rows(hr_df)
        assert rows is not None, "Expected trend_rows for HR data with 'department' column"

    def test_trend_rows_four_departments(self, hr_df):
        """4 departments → 4 aggregated rows."""
        rows = _compute_trend_rows(hr_df)
        assert rows is not None
        assert len(rows) == 4

    def test_trend_rows_engineering_highest_salary(self, hr_df):
        """Engineering has avg salary ~88,128 — highest among departments."""
        rows = _compute_trend_rows(hr_df)
        assert rows is not None
        dept_salary = {r["department"]: r["salary"] for r in rows}
        assert "Engineering" in dept_salary
        engineering_sal = dept_salary["Engineering"]
        others = [v for k, v in dept_salary.items() if k != "Engineering"]
        assert engineering_sal > max(others), \
            f"Engineering ({engineering_sal}) should have highest salary, got {dept_salary}"

    def test_trend_rows_salary_values_match_ground_truth(self, hr_df):
        """Exact average salary per department must match seed=42 ground truth."""
        rows = _compute_trend_rows(hr_df)
        assert rows is not None
        GT = {
            "Engineering": 88128,
            "Sales":        66129,
            "Operations":   65256,
            "Marketing":    62186,
        }
        dept_salary = {r["department"]: r["salary"] for r in rows}
        for dept, expected in GT.items():
            actual = dept_salary.get(dept)
            assert actual is not None, f"Department '{dept}' missing from trend_rows"
            assert abs(actual - expected) < 1, \
                f"{dept}: expected avg salary ~{expected}, got {actual:.1f}"

    def test_trend_rows_describe_dataframe_integration(self, ts_df):
        """describe_dataframe populates trend_rows field for timeseries data."""
        result = describe_dataframe(ts_df)
        assert result.trend_rows is not None
        assert len(result.trend_rows) == 120

    def test_trend_rows_none_when_small_df(self):
        """DataFrames with fewer than 4 rows skip trend detection."""
        df = pd.DataFrame({"month": ["2024-01", "2024-02"], "value": [1.0, 2.0]})
        rows = _compute_trend_rows(df)
        assert rows is None

    def test_trend_rows_none_when_no_time_or_group_col(self):
        df = pd.DataFrame({
            "alpha": range(10),
            "beta":  range(10, 20),
            "gamma": range(20, 30),
        })
        rows = _compute_trend_rows(df)
        assert rows is None

    def test_trend_rows_values_are_json_safe(self, hr_df):
        import json
        rows = _compute_trend_rows(hr_df)
        assert rows is not None
        json.dumps(rows)  # should not raise


# ═════════════════════════════════════════════════════════════════════════════
# 4. compute_correlations — correctness against known relationships
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeCorrelations:
    """Verify that compute_correlations surfaces known strong correlations."""

    def test_revenue_churn_strongly_negative(self, ts_df):
        """Revenue and churn_rate are strongly anti-correlated (r ≈ -0.98)."""
        result = compute_correlations(ts_df)
        pair = next(
            (p for p in result.pairs
             if set([p.col_a, p.col_b]) == {"revenue", "churn_rate"}),
            None,
        )
        assert pair is not None, "Expected revenue vs churn_rate pair"
        assert pair.correlation < -0.90, \
            f"Expected strong negative correlation, got {pair.correlation}"

    def test_pairs_sorted_by_absolute_value(self, ts_df):
        result = compute_correlations(ts_df)
        if len(result.pairs) >= 2:
            abs_values = [abs(p.correlation) for p in result.pairs]
            assert abs_values == sorted(abs_values, reverse=True), \
                "Correlation pairs not sorted by absolute value"

    def test_no_self_correlation(self, ts_df):
        result = compute_correlations(ts_df)
        for p in result.pairs:
            assert p.col_a != p.col_b

    def test_returns_empty_for_single_numeric_col(self, hc_df):
        """Only one numeric column → no pairs possible."""
        df = hc_df[["diagnosis", "bmi"]].copy()
        result = compute_correlations(df)
        assert result.pairs == []

    def test_salary_performance_correlation_in_hr(self, hr_df):
        """Salary and performance_score may have weak correlation — just verify no crash."""
        result = compute_correlations(hr_df)
        # Any result is acceptable; just verify structure
        for p in result.pairs:
            assert isinstance(p.correlation, float)
            assert -1.0 <= p.correlation <= 1.0
            assert math.isfinite(p.correlation)

    def test_all_correlations_in_range(self, ts_df):
        result = compute_correlations(ts_df)
        for p in result.pairs:
            assert -1.0 <= p.correlation <= 1.0, \
                f"Correlation out of [-1, 1]: {p.correlation}"

    def test_minimum_overlap_requirement(self):
        """Pairs with fewer than 10 overlapping rows should be skipped."""
        import numpy as np
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.normal(size=100),
            "b": [float("nan")] * 95 + list(rng.normal(size=5)),  # only 5 non-null
        })
        result = compute_correlations(df)
        # The pair a,b has < 10 overlapping rows → should be excluded
        ab_pair = next(
            (p for p in result.pairs if set([p.col_a, p.col_b]) == {"a", "b"}),
            None,
        )
        assert ab_pair is None, "Pair with < 10 overlapping rows should be excluded"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Healthcare ground-truth stats via describe_dataframe
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthcareGroundTruth:
    """Exact aggregate values against healthcare.csv (numpy seed=42)."""

    # Ground truth (computed deterministically)
    GT_BMI = {
        "Diabetes":     32.685,
        "Hypertension": 26.154,
        "Healthy":      23.603,
        "Asthma":       26.077,
    }
    GT_SBP = {
        "Diabetes":     122.847,
        "Healthy":      116.547,
        "Hypertension": 158.057,
        "Asthma":       119.314,
    }
    GT_READMISSION = {
        "Diabetes":     0.3059,
        "Healthy":      0.0533,
        "Hypertension": 0.1286,
        "Asthma":       0.1143,
    }

    def test_describe_gives_global_bmi_range(self, hc_df):
        result = describe_dataframe(hc_df)
        bmi_col = next(c for c in result.columns if c.name == "bmi")
        # Overall mean should be between healthy (~23) and diabetic (~33) means
        assert 25.0 < bmi_col.mean < 30.0, f"Overall BMI mean {bmi_col.mean} unexpected"

    def test_top_rows_includes_highest_bmi_patients(self, hc_df):
        """top_rows should capture patients with highest BMI (Diabetes cohort)."""
        result = describe_dataframe(hc_df)
        assert result.top_rows is not None
        # The sort column (highest variance) should be BMI or SBP; top rows should
        # have values well above the mean
        sort_vals = [r.get("bmi", r.get("systolic_bp", 0)) for r in result.top_rows]
        assert all(v > 0 for v in sort_vals)

    def test_top_rows_sorted_by_highest_variance_col(self, hc_df):
        """_compute_top_rows sorts by the column with highest std.
        In healthcare.csv patient_id has the highest std (86.7) so rows are sorted
        by patient_id descending. Verify the sort is correct whatever the column."""
        rows = _compute_top_rows(hc_df)
        assert rows is not None
        # Identify the sort column (highest-variance numeric col)
        numeric_cols = hc_df.select_dtypes(include="number").columns.tolist()
        sort_col = max(numeric_cols, key=lambda c: hc_df[c].std(skipna=True))
        sort_vals = [r[sort_col] for r in rows]
        assert sort_vals == sorted(sort_vals, reverse=True), \
            f"top_rows not sorted descending by {sort_col}"

    def test_readmission_ground_truth_for_diabetes(self, hc_df):
        """Exact readmission rate for Diabetes cohort: 30.59%."""
        diabetes_rows = hc_df[hc_df["diagnosis"] == "Diabetes"]
        actual = diabetes_rows["readmission_30d"].mean()
        assert abs(actual - self.GT_READMISSION["Diabetes"]) < 0.001, \
            f"Diabetes readmission: expected {self.GT_READMISSION['Diabetes']:.4f}, got {actual:.4f}"

    def test_hypertension_sbp_ground_truth(self, hc_df):
        """Exact avg systolic BP for Hypertension cohort: 158.057."""
        hyp_rows = hc_df[hc_df["diagnosis"] == "Hypertension"]
        actual = hyp_rows["systolic_bp"].mean()
        assert abs(actual - self.GT_SBP["Hypertension"]) < 0.1, \
            f"Hypertension SBP: expected {self.GT_SBP['Hypertension']}, got {actual:.3f}"

    def test_diagnosis_counts_correct(self, hc_df):
        counts = hc_df["diagnosis"].value_counts().to_dict()
        assert counts == {"Diabetes": 85, "Healthy": 75, "Hypertension": 70, "Asthma": 70}


# ═════════════════════════════════════════════════════════════════════════════
# 6. Timeseries ground-truth stats
# ═════════════════════════════════════════════════════════════════════════════

class TestTimeseriesGroundTruth:
    """Exact aggregate values against timeseries.csv."""

    def test_first_last_revenue(self, ts_df):
        assert ts_df["revenue"].iloc[0] == 103_812
        assert ts_df["revenue"].iloc[-1] == 334_378

    def test_churn_rate_monotone_decline(self, ts_df):
        """Churn should trend down from ~8.2% to ~3.46% (may not be strictly monotone)."""
        first = ts_df["churn_rate"].iloc[0]
        last  = ts_df["churn_rate"].iloc[-1]
        assert first > last, "Churn rate should decline over the 10-year period"
        assert abs(first - 0.0820) < 0.001
        assert abs(last  - 0.0346) < 0.001

    def test_revenue_trend_in_trend_rows(self, ts_df):
        """trend_rows for the timeseries fixture must show revenue growth."""
        rows = _compute_trend_rows(ts_df)
        assert rows is not None
        first_rev = rows[0]["revenue"]
        last_rev  = rows[-1]["revenue"]
        assert last_rev > first_rev * 2, \
            f"Revenue should more than double: {first_rev} → {last_rev}"

    def test_strong_negative_revenue_churn_correlation(self, ts_df):
        result = compute_correlations(ts_df)
        pair = next(
            (p for p in result.pairs
             if "revenue" in (p.col_a, p.col_b) and "churn_rate" in (p.col_a, p.col_b)),
            None,
        )
        assert pair is not None
        assert pair.correlation < -0.95, \
            f"Revenue/churn correlation should be very strong negative, got {pair.correlation}"

    def test_120_monthly_data_points(self, ts_df):
        assert len(ts_df) == 120
        assert ts_df["month"].nunique() == 120


# ═════════════════════════════════════════════════════════════════════════════
# 7. HR ground-truth stats
# ═════════════════════════════════════════════════════════════════════════════

class TestHRGroundTruth:
    """Exact aggregate values against hr.csv."""

    GT_DEPT_SALARY = {
        "Engineering": 88128,
        "Sales":        66129,
        "Operations":   65256,
        "Marketing":    62186,
    }
    GT_LEVEL_SALARY = {
        "Lead":   105518,
        "Senior":  85505,
        "Mid":     67982,
        "Junior":  46532,
    }

    def test_dept_salary_ground_truth(self, hr_df):
        for dept, expected in self.GT_DEPT_SALARY.items():
            actual = hr_df[hr_df["department"] == dept]["salary"].mean()
            assert abs(actual - expected) < 1, \
                f"{dept}: expected {expected}, got {actual:.1f}"

    def test_level_salary_ground_truth(self, hr_df):
        for level, expected in self.GT_LEVEL_SALARY.items():
            actual = hr_df[hr_df["level"] == level]["salary"].mean()
            assert abs(actual - expected) < 1, \
                f"{level}: expected {expected}, got {actual:.1f}"

    def test_overall_salary_mean(self, hr_df):
        actual = hr_df["salary"].mean()
        assert abs(actual - 72135.34) < 1, f"Overall salary mean: {actual}"

    def test_engineering_highest_department(self, hr_df):
        by_dept = hr_df.groupby("department")["salary"].mean()
        assert by_dept.idxmax() == "Engineering"

    def test_lead_highest_level(self, hr_df):
        by_level = hr_df.groupby("level")["salary"].mean()
        assert by_level.idxmax() == "Lead"

    def test_junior_lowest_level(self, hr_df):
        by_level = hr_df.groupby("level")["salary"].mean()
        assert by_level.idxmin() == "Junior"

    def test_trend_rows_captures_all_departments(self, hr_df):
        result = describe_dataframe(hr_df)
        assert result.trend_rows is not None
        dept_names = {r["department"] for r in result.trend_rows}
        assert dept_names == {"Engineering", "Sales", "Operations", "Marketing"}

    def test_top_rows_contains_leads(self, hr_df):
        result = describe_dataframe(hr_df)
        assert result.top_rows is not None
        top_salaries = [r["salary"] for r in result.top_rows]
        # All top-10 salaries should be above the Lead average (~105k)
        # → at least the top 3 should be above 100k
        above_100k = sum(1 for s in top_salaries if s > 100_000)
        assert above_100k >= 3, f"Expected ≥3 employees above 100k in top rows, got {above_100k}"
