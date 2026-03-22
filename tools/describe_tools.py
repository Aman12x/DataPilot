"""
tools/describe_tools.py — General-purpose data description and correlation tools.

Used by the general analysis path (analysis_mode == "general").
No LangGraph or Streamlit imports. Pure Python + pandas.
"""
from __future__ import annotations

import math

import pandas as pd

from tools.schemas import ColumnSummary, CorrelationPair, CorrelationResult, DescribeResult


def describe_dataframe(df: pd.DataFrame, max_top_values: int = 5) -> DescribeResult:
    """
    Compute per-column summary statistics for any DataFrame.

    Numeric columns get mean/std/min/quartiles/max.
    Categorical / string columns get n_unique and top-N value counts.
    Also computes top_rows (ranked by highest-variance numeric col) and
    trend_rows (aggregated by detected time/group column) for the narrative LLM.
    """
    columns: list[ColumnSummary] = []

    for col in df.columns:
        series    = df[col]
        non_null  = int(series.notna().sum())
        null_count = int(series.isna().sum())
        dtype_str  = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            desc = series.dropna().describe()
            columns.append(ColumnSummary(
                name       = col,
                dtype      = dtype_str,
                non_null   = non_null,
                null_count = null_count,
                mean       = _safe_float(desc.get("mean")),
                std        = _safe_float(desc.get("std")),
                min        = _safe_float(desc.get("min")),
                p25        = _safe_float(desc.get("25%")),
                median     = _safe_float(desc.get("50%")),
                p75        = _safe_float(desc.get("75%")),
                max        = _safe_float(desc.get("max")),
                n_unique   = int(series.nunique()),
            ))
        else:
            vc = series.dropna().astype(str).value_counts().head(max_top_values)
            top_values = [f"{v}: {c}" for v, c in vc.items()]
            columns.append(ColumnSummary(
                name       = col,
                dtype      = dtype_str,
                non_null   = non_null,
                null_count = null_count,
                n_unique   = int(series.nunique()),
                top_values = top_values,
            ))

    top_rows   = _compute_top_rows(df)
    trend_rows = _compute_trend_rows(df)

    return DescribeResult(
        row_count  = len(df),
        col_count  = len(df.columns),
        columns    = columns,
        top_rows   = top_rows,
        trend_rows = trend_rows,
    )


def compute_correlations(df: pd.DataFrame, top_n: int = 10) -> CorrelationResult:
    """
    Return the top-N column pairs by absolute Pearson correlation.

    Only numeric columns are considered. Pairs with fewer than 10 non-null
    overlapping observations are skipped.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return CorrelationResult(pairs=[])

    corr_matrix = df[numeric_cols].corr(method="pearson")
    pairs: list[CorrelationPair] = []

    seen: set[frozenset] = set()
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            key = frozenset({col_a, col_b})
            if key in seen:
                continue
            seen.add(key)

            r = corr_matrix.loc[col_a, col_b]
            if math.isnan(r):
                continue
            # Require enough overlapping non-null rows
            overlap = df[[col_a, col_b]].dropna()
            if len(overlap) < 10:
                continue

            pairs.append(CorrelationPair(
                col_a       = col_a,
                col_b       = col_b,
                correlation = round(float(r), 4),
            ))

    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return CorrelationResult(pairs=pairs[:top_n])


# ── Helpers ───────────────────────────────────────────────────────────────────

_TIME_KEYWORDS  = {"season", "year", "month", "week", "date", "period", "quarter", "day"}
_GROUP_KEYWORDS = {"position", "pos", "team", "category", "segment", "group", "type", "tier", "region"}


def _compute_top_rows(df: pd.DataFrame, n: int = 10) -> list[dict] | None:
    """Return top-N rows sorted descending by the highest-variance numeric column.

    This lets the narrative LLM identify named entities (e.g. players) ranked by a metric
    rather than just seeing aggregate statistics like max=24.42 with no name attached.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    # Pick the numeric column with the highest std (most informative for ranking)
    sort_col = max(numeric_cols, key=lambda c: df[c].std(skipna=True))
    try:
        top = df.nlargest(n, sort_col)
        return _df_to_records(top)
    except Exception:
        return None


def _compute_trend_rows(df: pd.DataFrame) -> list[dict] | None:
    """If the dataframe has a time or group column, aggregate numeric cols by it.

    Returns avg of all numeric cols grouped by the detected dimension so the
    narrative LLM can cite trends (e.g. 3PA by season) or breakdowns (PPG by position).
    """
    if len(df) < 4:
        return None

    col_lower = {c: c.lower() for c in df.columns}
    # Prefer time column, fall back to group column
    time_col  = next((c for c, l in col_lower.items() if any(k in l for k in _TIME_KEYWORDS)), None)
    group_col = next((c for c, l in col_lower.items() if any(k in l for k in _GROUP_KEYWORDS)), None)
    dim_col   = time_col or group_col
    if dim_col is None:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude the dim_col itself if numeric (e.g. season=2015 is a number)
    numeric_cols = [c for c in numeric_cols if c != dim_col]
    if not numeric_cols:
        return None

    try:
        agg = df.groupby(dim_col)[numeric_cols].mean().round(3).reset_index()
        agg = agg.sort_values(dim_col)
        return _df_to_records(agg)
    except Exception:
        return None


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert dataframe to list of dicts with JSON-safe values."""
    result = []
    for _, row in df.iterrows():
        record = {}
        for k, v in row.items():
            if hasattr(v, "item"):  # numpy scalar
                v = v.item()
            record[str(k)] = v
        result.append(record)
    return result


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return round(f, 6) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
