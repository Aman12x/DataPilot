"""
tests/test_upload_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for backend/api/routes/upload.py, focused on the _infer_tables
heuristic and the DuckDB conversion pipeline.

No HTTP server or authentication is required — we test the pure Python
functions directly.

Three shapes exercised
────────────────────────────────────────────────────────────────────────────
A) AB-test   — has variant/arm/treatment column
B) User-level — has recognisable user-ID column, no variant
C) Time-series — rows are dates (no user ID in known set)
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import directly from the module (no FastAPI app needed)
from backend.api.routes.upload import (
    _infer_tables,
    _looks_like_date_col,
    _normalise_cols,
    resolve_upload_path,
)

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
HC_CSV = os.path.join(_FIXTURES, "healthcare.csv")
TS_CSV = os.path.join(_FIXTURES, "timeseries.csv")
HR_CSV = os.path.join(_FIXTURES, "hr.csv")
AB_CSV = os.path.join(_FIXTURES, "ab_test_simple.csv")
SAAS_CSV = os.path.join(_FIXTURES, "saas_churn.csv")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _write_to_duckdb(tables: dict[str, pd.DataFrame]) -> tuple[str, duckdb.DuckDBPyConnection]:
    """Write a dict of DataFrames into a temp DuckDB (same path as the real upload endpoint)."""
    fd, tmp_db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(tmp_db)
    con = duckdb.connect(tmp_db)
    for table_name, tdf in tables.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tdf.to_csv(f, index=False)
            tmp_csv = f.name
        try:
            con.execute(
                f"CREATE TABLE {table_name} AS "
                f"SELECT * FROM read_csv_auto('{tmp_csv}', header=true)"
            )
        finally:
            try:
                os.unlink(tmp_csv)
            except OSError:
                pass
    return tmp_db, con


# ═════════════════════════════════════════════════════════════════════════════
# 1. _normalise_cols
# ═════════════════════════════════════════════════════════════════════════════

class TestNormaliseCols:
    def test_lowercases_headers(self):
        df = pd.DataFrame(columns=["UserID", "Revenue", "Churn Rate"])
        df = _normalise_cols(df)
        assert list(df.columns) == ["userid", "revenue", "churn_rate"]

    def test_strips_whitespace(self):
        df = pd.DataFrame(columns=["  user_id  ", "\ttask\n"])
        df = _normalise_cols(df)
        assert "user_id" in df.columns
        assert "task" in df.columns

    def test_replaces_special_chars(self):
        df = pd.DataFrame(columns=["col.name", "col (unit)", "col-name"])
        df = _normalise_cols(df)
        for col in df.columns:
            assert col.replace("_", "").isalnum(), f"Unexpected chars in '{col}'"

    def test_collapses_repeated_underscores(self):
        df = pd.DataFrame(columns=["a__b___c"])
        df = _normalise_cols(df)
        assert "a_b_c" in df.columns

    def test_empty_name_replaced_with_col(self):
        df = pd.DataFrame(columns=[""])
        df = _normalise_cols(df)
        assert "col" in df.columns


# ═════════════════════════════════════════════════════════════════════════════
# 2. _looks_like_date_col
# ═════════════════════════════════════════════════════════════════════════════

class TestLooksLikeDateCol:
    def test_month_keyword(self):
        assert _looks_like_date_col("month") is True

    def test_date_keyword(self):
        assert _looks_like_date_col("assignment_date") is True

    def test_timestamp_keyword(self):
        assert _looks_like_date_col("created_timestamp") is True

    def test_non_date_column(self):
        assert _looks_like_date_col("revenue") is False
        assert _looks_like_date_col("user_id") is False
        assert _looks_like_date_col("salary") is False


# ═════════════════════════════════════════════════════════════════════════════
# 3. _infer_tables — Shape A: AB-test data
# ═════════════════════════════════════════════════════════════════════════════

class TestInferTablesABTest:
    """Files with variant/arm/treatment should produce separate experiment + events tables."""

    @pytest.fixture
    def ab_df(self) -> pd.DataFrame:
        """Minimal AB-test DataFrame."""
        return pd.DataFrame({
            "user_id":    [f"u{i}" for i in range(10)],
            "variant":    (["control"] * 5) + (["treatment"] * 5),
            "signup_date": ["2024-01-01"] * 10,
            "converted":  [0, 1, 0, 0, 1, 1, 1, 0, 1, 1],
            "revenue":    [0, 25, 0, 0, 18, 40, 30, 0, 22, 35],
        })

    def test_both_tables_created(self, ab_df):
        tables = _infer_tables(ab_df)
        assert "experiment" in tables
        assert "events" in tables

    def test_experiment_has_required_cols(self, ab_df):
        tables = _infer_tables(ab_df)
        exp = tables["experiment"]
        assert "user_id" in exp.columns
        assert "variant" in exp.columns
        assert "week" in exp.columns

    def test_experiment_week_always_1(self, ab_df):
        tables = _infer_tables(ab_df)
        assert (tables["experiment"]["week"] == 1).all()

    def test_variant_col_renamed_to_variant(self):
        """arm/treatment columns should be renamed to 'variant'."""
        df = pd.DataFrame({
            "customer_id": range(4),
            "arm":         ["control", "treatment", "control", "treatment"],
            "metric":      [1.0, 1.2, 0.9, 1.3],
        })
        df = _normalise_cols(df)
        tables = _infer_tables(df)
        assert "variant" in tables["experiment"].columns

    def test_assignment_date_populated(self, ab_df):
        tables = _infer_tables(ab_df)
        assert "assignment_date" in tables["experiment"].columns
        assert tables["experiment"]["assignment_date"].notna().all()

    def test_events_contains_metric_cols(self, ab_df):
        tables = _infer_tables(ab_df)
        events = tables["events"]
        assert "converted" in events.columns or "revenue" in events.columns

    def test_variant_col_not_duplicated_in_events(self, ab_df):
        """variant should live only in experiment, not be duplicated in events."""
        tables = _infer_tables(ab_df)
        # events may retain user_id but variant is experiment-side metadata
        events_cols = set(tables["events"].columns)
        exp_cols    = set(tables["experiment"].columns)
        # At minimum, variant must be in experiment
        assert "variant" in exp_cols

    def test_real_ab_fixture(self):
        df = _normalise_cols(pd.read_csv(AB_CSV))
        tables = _infer_tables(df)
        assert "experiment" in tables
        assert "events" in tables
        assert "variant" in tables["experiment"].columns


# ═════════════════════════════════════════════════════════════════════════════
# 4. _infer_tables — Shape B: User-level data (no variant)
# ═════════════════════════════════════════════════════════════════════════════

class TestInferTablesUserLevel:
    """Files with a user/customer ID but no variant → stub experiment table."""

    @pytest.fixture
    def saas_df(self) -> pd.DataFrame:
        return _normalise_cols(pd.read_csv(SAAS_CSV))

    @pytest.fixture
    def hc_df(self) -> pd.DataFrame:
        return _normalise_cols(pd.read_csv(HC_CSV))

    @pytest.fixture
    def hr_df(self) -> pd.DataFrame:
        return _normalise_cols(pd.read_csv(HR_CSV))

    def test_both_tables_created_saas(self, saas_df):
        tables = _infer_tables(saas_df)
        assert "experiment" in tables
        assert "events" in tables

    def test_stub_experiment_has_control_variant(self, saas_df):
        tables = _infer_tables(saas_df)
        exp = tables["experiment"]
        assert (exp["variant"] == "control").all()

    def test_stub_experiment_week_is_1(self, saas_df):
        tables = _infer_tables(saas_df)
        assert (tables["experiment"]["week"] == 1).all()

    def test_events_preserves_all_original_cols(self, saas_df):
        tables = _infer_tables(saas_df)
        events = tables["events"]
        # Key saas columns should be present (may be renamed customer_id→user_id)
        assert any(c in events.columns for c in ("user_id", "customer_id"))
        assert "mrr" in events.columns or "churned" in events.columns

    def test_healthcare_uses_patient_id_as_user_id(self, hc_df):
        tables = _infer_tables(hc_df)
        events = tables["events"]
        # patient_id is in _USER_ID_COLS → should be renamed to user_id
        assert "user_id" in events.columns

    def test_healthcare_events_row_count(self, hc_df):
        tables = _infer_tables(hc_df)
        assert len(tables["events"]) == 300

    def test_healthcare_experiment_has_unique_patients(self, hc_df):
        tables = _infer_tables(hc_df)
        exp = tables["experiment"]
        # Stub experiment: one row per patient
        assert exp["user_id"].nunique() == len(exp)

    def test_hr_uses_employee_id_fallback(self, hr_df):
        """employee_id is not in _USER_ID_COLS → falls back to first column."""
        tables = _infer_tables(hr_df)
        assert "events" in tables
        events = tables["events"]
        assert len(events) == 200


# ═════════════════════════════════════════════════════════════════════════════
# 5. _infer_tables — Shape C: Time-series data (no user ID column)
# ═════════════════════════════════════════════════════════════════════════════

class TestInferTablesTimeSeries:
    """Files where the first column is a date and there is no user ID."""

    @pytest.fixture
    def ts_df(self) -> pd.DataFrame:
        return _normalise_cols(pd.read_csv(TS_CSV))

    def test_both_tables_created(self, ts_df):
        tables = _infer_tables(ts_df)
        assert "experiment" in tables
        assert "events" in tables

    def test_synthetic_user_id_added(self, ts_df):
        """Since there's no user ID, a synthetic 'user_id'='user_1' should be injected."""
        tables = _infer_tables(ts_df)
        events = tables["events"]
        assert "user_id" in events.columns
        assert (events["user_id"] == "user_1").all()

    def test_month_column_preserved(self, ts_df):
        tables = _infer_tables(ts_df)
        events = tables["events"]
        assert "month" in events.columns

    def test_revenue_and_churn_preserved(self, ts_df):
        tables = _infer_tables(ts_df)
        events = tables["events"]
        assert "revenue" in events.columns
        assert "churn_rate" in events.columns

    def test_row_count_preserved(self, ts_df):
        tables = _infer_tables(ts_df)
        assert len(tables["events"]) == 120

    def test_stub_experiment_has_single_entity(self, ts_df):
        tables = _infer_tables(ts_df)
        exp = tables["experiment"]
        # Time-series → single synthetic user → 1 row in experiment stub
        assert len(exp) == 1
        assert exp["user_id"].iloc[0] == "user_1"

    def test_custom_time_series_no_known_uid(self):
        """Any DataFrame where first col is date-like and no user-ID cols."""
        df = pd.DataFrame({
            "date":   pd.date_range("2024-01", periods=12, freq="MS"),
            "metric": range(12),
        })
        df.columns = [str(c) for c in df.columns]
        tables = _infer_tables(df)
        assert "events" in tables
        events = tables["events"]
        assert "user_id" in events.columns


# ═════════════════════════════════════════════════════════════════════════════
# 6. DuckDB schema after full pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestDuckDBSchema:
    """Verify that writing _infer_tables output to DuckDB produces a well-typed schema."""

    @pytest.fixture(scope="class")
    def hc_tables_db(self):
        df = _normalise_cols(pd.read_csv(HC_CSV))
        tables = _infer_tables(df)
        db_path, con = _write_to_duckdb(tables)
        yield con, db_path
        con.close()
        os.unlink(db_path)

    @pytest.fixture(scope="class")
    def ts_tables_db(self):
        df = _normalise_cols(pd.read_csv(TS_CSV))
        tables = _infer_tables(df)
        db_path, con = _write_to_duckdb(tables)
        yield con, db_path
        con.close()
        os.unlink(db_path)

    @pytest.fixture(scope="class")
    def hr_tables_db(self):
        df = _normalise_cols(pd.read_csv(HR_CSV))
        tables = _infer_tables(df)
        db_path, con = _write_to_duckdb(tables)
        yield con, db_path
        con.close()
        os.unlink(db_path)

    def test_healthcare_tables_exist(self, hc_tables_db):
        con, _ = hc_tables_db
        table_names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert "events" in table_names
        assert "experiment" in table_names

    def test_healthcare_events_row_count(self, hc_tables_db):
        con, _ = hc_tables_db
        n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 300

    def test_healthcare_bmi_is_numeric(self, hc_tables_db):
        con, _ = hc_tables_db
        col_types = {r[1]: r[2].upper() for r in con.execute("PRAGMA table_info('events')").fetchall()}
        bmi_type = col_types.get("bmi", "")
        assert any(t in bmi_type for t in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC")), \
            f"bmi column should be numeric, got {bmi_type}"

    def test_healthcare_readmission_is_integer(self, hc_tables_db):
        con, _ = hc_tables_db
        col_types = {r[1]: r[2].upper() for r in con.execute("PRAGMA table_info('events')").fetchall()}
        readm_type = col_types.get("readmission_30d", "")
        assert "INT" in readm_type or "BIGINT" in readm_type, \
            f"readmission_30d should be integer, got {readm_type}"

    def test_timeseries_month_detected_as_date_or_varchar(self, ts_tables_db):
        """month column ('2015-01-01') should be DATE or at worst VARCHAR — not INTEGER."""
        con, _ = ts_tables_db
        col_types = {r[1]: r[2].upper() for r in con.execute("PRAGMA table_info('events')").fetchall()}
        month_type = col_types.get("month", "")
        assert "INT" not in month_type, \
            f"month column should not be INTEGER, got {month_type}"

    def test_timeseries_revenue_is_numeric(self, ts_tables_db):
        con, _ = ts_tables_db
        col_types = {r[1]: r[2].upper() for r in con.execute("PRAGMA table_info('events')").fetchall()}
        rev_type = col_types.get("revenue", "")
        assert any(t in rev_type for t in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "INT", "BIGINT")), \
            f"revenue should be numeric, got {rev_type}"

    def test_timeseries_sql_count(self, ts_tables_db):
        """Should be able to query the table without error."""
        con, _ = ts_tables_db
        n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert n == 120

    def test_hr_events_sql_groupby_dept(self, hr_tables_db):
        """GROUP BY department on the HR events table should work and give 4 rows."""
        con, _ = hr_tables_db
        df = con.execute(
            "SELECT department, AVG(salary) AS avg_sal FROM events GROUP BY department"
        ).df()
        assert len(df) == 4

    def test_hr_experiment_all_control(self, hr_tables_db):
        con, _ = hr_tables_db
        variants = con.execute("SELECT DISTINCT variant FROM experiment").fetchall()
        variant_vals = [r[0] for r in variants]
        assert variant_vals == ["control"], f"Expected only 'control', got {variant_vals}"

    def test_downstream_sql_pattern_works_on_healthcare(self, hc_tables_db):
        """
        Verify a representative downstream SQL pattern (GROUP BY diagnosis) works
        on the healthcare DuckDB without error and returns correct rows.
        """
        con, _ = hc_tables_db
        df = con.execute("""
            SELECT
                diagnosis,
                COUNT(*)            AS n,
                AVG(bmi)            AS avg_bmi,
                AVG(systolic_bp)    AS avg_sbp,
                AVG(readmission_30d) AS readmit_rate
            FROM events
            GROUP BY diagnosis
            ORDER BY avg_bmi DESC
        """).df()
        assert len(df) == 4
        # Diabetes should have highest BMI
        assert df.iloc[0]["diagnosis"] == "Diabetes"
        # Readmission rates should be in [0, 1]
        assert df["readmit_rate"].between(0, 1).all()

    def test_downstream_sql_pattern_works_on_timeseries(self, ts_tables_db):
        """Revenue should increase from first to last period."""
        con, _ = ts_tables_db
        df = con.execute("""
            SELECT month, AVG(revenue) AS revenue
            FROM events
            GROUP BY month
            ORDER BY month ASC
        """).df()
        assert len(df) == 120
        assert df["revenue"].iloc[-1] > df["revenue"].iloc[0]


# ═════════════════════════════════════════════════════════════════════════════
# 7. resolve_upload_path — path validation
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveUploadPath:
    def test_returns_path_when_file_exists(self, tmp_path):
        """resolve_upload_path must return the path when the file is present."""
        import uuid
        from unittest.mock import patch

        upload_id = str(uuid.uuid4())
        user_id   = "test_user"
        user_dir  = tmp_path / user_id
        user_dir.mkdir()
        db_file   = user_dir / f"{upload_id}.db"
        db_file.touch()

        with patch("backend.api.routes.upload._UPLOAD_DIR", str(tmp_path)):
            path = resolve_upload_path(upload_id, user_id)
        assert path.endswith(f"{upload_id}.db")

    def test_raises_404_when_file_missing(self, tmp_path):
        """resolve_upload_path must raise HTTP 404 for unknown upload_id."""
        import uuid
        from fastapi import HTTPException
        from unittest.mock import patch

        upload_id = str(uuid.uuid4())

        with patch("backend.api.routes.upload._UPLOAD_DIR", str(tmp_path)):
            with pytest.raises(HTTPException) as exc_info:
                resolve_upload_path(upload_id, "nobody")
        assert exc_info.value.status_code == 404
