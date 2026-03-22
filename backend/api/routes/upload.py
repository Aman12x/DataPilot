"""
backend/api/routes/upload.py

POST   /upload             multipart: file (.csv/.xlsx/.xls)  → {upload_id, columns, row_count, preview}
DELETE /upload/{upload_id}                                     → {status: "ok"}
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

# UUID v4 pattern — upload_id must match exactly to prevent path traversal
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ..deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])

_UPLOAD_DIR   = os.getenv("UPLOAD_DIR", "tmp_uploads")
_MAX_BYTES    = 50 * 1024 * 1024  # 50 MB
_ALLOWED_EXT  = {".csv", ".xlsx", ".xls"}
_VARIANT_COLS = {"variant", "arm", "treatment", "group", "exp_group"}
_USER_ID_COLS = {"user_id", "userid", "uid", "id", "patient_id", "customer_id", "shipment_id"}
_DATE_KEYWORDS = ("date", "time", "day", "month", "timestamp", "ts", "dt")


def resolve_upload_path(upload_id: str, user_id: str) -> str:
    user_dir = Path(_UPLOAD_DIR).resolve() / user_id
    target   = (user_dir / f"{upload_id}.db").resolve()
    # Ensure the resolved path stays inside the user's directory (defence-in-depth)
    if not str(target).startswith(str(user_dir)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload ID")
    if not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")
    return str(target)


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip whitespace, replace any non-alphanumeric char with underscore."""
    def _clean(c: str) -> str:
        c = c.strip().lower()
        c = re.sub(r"[^\w]", "_", c)   # parens, spaces, dots → _
        c = re.sub(r"_+", "_", c).strip("_")  # collapse runs, trim edges
        return c or "col"
    df.columns = [_clean(c) for c in df.columns]
    return df


def _looks_like_date_col(col_name: str) -> bool:
    return any(k in col_name for k in _DATE_KEYWORDS)


def _infer_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Heuristic split into `experiment` + `events` tables so downstream SQL
    templates work unchanged regardless of input file structure.

    Handles three shapes:
      A) AB-test data   — has a variant/arm/treatment column
      B) User-level data — has a recognisable user-ID column (user_id, customer_id, …)
      C) Time-series    — rows are dates with no user ID (Apple Watch, sensors, stocks…)
                          → adds synthetic user_id='user_1', preserves the date column
    """
    variant_col = next((c for c in df.columns if c in _VARIANT_COLS), None)

    # Find user ID column by name (explicit match only — don't fall back to first col yet)
    uid_col = next((c for c in df.columns if c in _USER_ID_COLS), None)

    date_col = next(
        (c for c in df.columns if _looks_like_date_col(c)),
        None,
    )

    # ── Case C: no user ID found, first column is date-like → time-series ──────
    if uid_col is None and date_col is not None and df.columns[0] == date_col:
        # Add a synthetic single-entity user_id so downstream SQL works.
        # The date column is preserved as-is.
        df = df.copy()
        df.insert(0, "user_id", "user_1")
        uid_col = "user_id"

    # ── Fallback: still no user ID → use first column ─────────────────────────
    if uid_col is None:
        uid_col = df.columns[0]

    # ── Case A: AB-test data ──────────────────────────────────────────────────
    if variant_col:
        exp_cols = [uid_col, variant_col]
        if date_col and date_col not in exp_cols:
            exp_cols.append(date_col)

        exp_df = df[exp_cols].copy().rename(columns={
            uid_col:     "user_id",
            variant_col: "variant",
            **({date_col: "assignment_date"} if date_col else {}),
        })
        if "assignment_date" not in exp_df.columns:
            exp_df["assignment_date"] = pd.Timestamp.today().date()
        exp_df["week"] = 1

        remaining = [c for c in df.columns if c not in exp_cols or c == uid_col]
        events_df = df[remaining].copy().rename(columns={uid_col: "user_id"})
        return {"experiment": exp_df, "events": events_df}

    # ── Case B/C: no variant column → general data or time-series ────────────
    events_df = df.copy()
    if uid_col != "user_id":
        events_df = events_df.rename(columns={uid_col: "user_id"})

    min_date = None
    if date_col:
        try:
            min_date = pd.to_datetime(events_df[date_col]).min().date()
        except Exception:
            pass

    # Minimal experiment table so downstream SQL templates don't break.
    # For non-experiment data this table is essentially a stub.
    exp_df = pd.DataFrame({
        "user_id":         events_df["user_id"].drop_duplicates().values,
        "variant":         "control",
        "week":            1,
        "assignment_date": min_date or pd.Timestamp.today().date(),
    })
    return {"events": events_df, "experiment": exp_df}


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXT:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(_ALLOWED_EXT))}",
        )

    raw = await file.read()
    if len(raw) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 50 MB limit",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw)) if suffix == ".csv" \
            else pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Could not parse file. Ensure it is a valid CSV or Excel file.")

    df = _normalise_cols(df)
    if df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")

    upload_id = str(uuid.uuid4())
    user_dir  = os.path.join(_UPLOAD_DIR, current_user["user_id"])
    os.makedirs(user_dir, exist_ok=True)
    db_path   = os.path.join(user_dir, f"{upload_id}.db")

    tables = _infer_tables(df)
    try:
        con = duckdb.connect(db_path)
        for table_name, tdf in tables.items():
            # Write through a temp CSV so DuckDB's read_csv_auto can infer types
            # properly (e.g. ISO dates → DATE, 0/1 strings → INTEGER) rather than
            # inheriting the coarser pandas object dtype for those columns.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as tmp:
                tdf.to_csv(tmp, index=False)
                tmp_path = tmp.name
            try:
                con.execute(
                    f"CREATE TABLE {table_name} AS "
                    f"SELECT * FROM read_csv_auto('{tmp_path}', header=true)"
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        con.close()
    except Exception as exc:
        logger.exception("DuckDB write failed for upload %s", upload_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    logger.info("upload user=%s id=%s rows=%d cols=%d",
                current_user["user_id"], upload_id, len(df), len(df.columns))

    # Return column list for display — exclude the synthetic user_id column that
    # _infer_tables inserts for time-series data (it wasn't in the user's file).
    _had_uid_originally = any(c in _USER_ID_COLS for c in df.columns)
    display_cols = [c for c in df.columns if c != "user_id" or _had_uid_originally]
    preview = df.head(5).where(pd.notnull(df.head(5)), None).to_dict(orient="records")
    return {
        "upload_id": upload_id,
        "columns":   display_cols,
        "row_count": len(df),
        "preview":   preview,
    }


@router.delete("/upload/{upload_id}")
def delete_upload_endpoint(
    upload_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict[str, str]:
    if not _UUID_RE.match(upload_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload ID")
    target = Path(resolve_upload_path(upload_id, current_user["user_id"]))
    try:
        target.unlink(missing_ok=True)
    except OSError:
        pass
    return {"status": "ok"}
