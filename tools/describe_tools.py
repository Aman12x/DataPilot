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

    return DescribeResult(
        row_count = len(df),
        col_count = len(df.columns),
        columns   = columns,
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

def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return round(f, 6) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None
