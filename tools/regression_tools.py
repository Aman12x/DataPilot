"""
tools/regression_tools.py — OLS regression for general analysis mode.

Identifies which features predict a target column using ordinary least squares.
Uses numpy/scipy/sklearn only — no statsmodels dependency.

Design:
  - Target auto-detection: column mentioned in task hint, or highest-variance numeric
  - Features: all numeric + one-hot encoded categoricals (≤ 10 unique values)
  - Max 15 features (highest-variance kept when over limit)
  - VIF computed via sklearn to flag multicollinearity (threshold: 10)
  - P-values from scipy t-distribution
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

_MAX_FEATURES   = 15
_MAX_CATEGORIES = 10    # max unique values for one-hot encoding
_VIF_THRESHOLD  = 10.0  # variance inflation factor cutoff


# ── Target selection ──────────────────────────────────────────────────────────

def _select_target(df: pd.DataFrame, task_hint: str = "") -> str | None:
    """
    Pick regression target column.

    Priority:
      1. Numeric column whose name appears in the task hint (longest match wins)
      2. Highest-variance numeric column as fallback
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    if task_hint:
        hint_lower = task_hint.lower()
        # Normalise underscores → spaces so "performance_score" matches "performance score"
        hint_normalised = hint_lower.replace("_", " ")

        def _col_matches(col: str) -> bool:
            cl = col.lower()
            cl_norm = cl.replace("_", " ")
            return cl in hint_lower or cl_norm in hint_normalised

        # Sort by length descending so "revenue_per_user" matches before "revenue"
        matches = sorted(
            [c for c in numeric_cols if _col_matches(c)],
            key=len,
            reverse=True,
        )
        if matches:
            return matches[0]

    return str(df[numeric_cols].var().idxmax())


# ── Feature matrix ────────────────────────────────────────────────────────────

def _build_feature_matrix(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Prepare feature matrix:
      - Numeric columns (excluding target): used as-is, NaN → median
      - Categorical columns with 2–_MAX_CATEGORIES unique values: one-hot (drop_first)
      - High-cardinality categoricals: dropped
      - Constant columns: dropped
      - Truncated to _MAX_FEATURES highest-variance columns
    """
    feature_df = df.drop(columns=[target_col], errors="ignore").copy()

    # Numeric
    num_df = feature_df.select_dtypes(include="number").copy()
    for col in num_df.columns:
        if num_df[col].isna().any():
            num_df[col] = num_df[col].fillna(num_df[col].median())

    # Categorical → one-hot
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns
    parts: list[pd.DataFrame] = [num_df]
    for col in cat_cols:
        n_unique = feature_df[col].nunique()
        if 2 <= n_unique <= _MAX_CATEGORIES:
            mode_val = feature_df[col].mode()
            filled = feature_df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "")
            dummies = pd.get_dummies(filled, prefix=col, drop_first=True, dtype=float)
            parts.append(dummies)

    result = pd.concat(parts, axis=1)

    # Drop constant and all-NaN columns
    result = result.loc[:, result.std() > 0]
    result = result.dropna(axis=1, how="all")

    # Truncate to highest-variance features
    if len(result.columns) > _MAX_FEATURES:
        top_cols = result.var().nlargest(_MAX_FEATURES).index
        result = result[top_cols]

    return result


# ── VIF ───────────────────────────────────────────────────────────────────────

def _compute_vif(X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """
    Variance Inflation Factor for each feature via sklearn.
    VIF(j) = 1 / (1 - R²_j), where R²_j is R² from regressing feature j on all others.
    """
    n, k = X.shape
    if k < 2 or n < k + 2:
        return {}
    vif: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        others = np.delete(X, j, axis=1)
        try:
            r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
            vif[name] = round(1.0 / (1.0 - r2) if r2 < 1.0 - 1e-10 else float("inf"), 2)
        except Exception:
            vif[name] = float("nan")
    return vif


# ── Main entry point ──────────────────────────────────────────────────────────

def run_regression(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    task_hint: str = "",
) -> "RegressionResult":
    """
    Fit OLS regression on df to identify predictors of target_col.

    Args:
        df:         Input DataFrame (query_result from general mode).
        target_col: Column to predict. Auto-detected from task_hint or variance if None.
        task_hint:  Analyst task string used for target detection.

    Returns:
        RegressionResult with per-feature statistics, R², F-stat, and VIF warnings.

    Raises:
        ValueError: No usable target/features, or insufficient observations.
    """
    from tools.schemas import RegressionCoef, RegressionResult

    # Resolve target
    if target_col is None:
        target_col = _select_target(df, task_hint=task_hint)
    if target_col is None or target_col not in df.columns:
        raise ValueError("run_regression: no usable target column found.")

    # Build features
    X_df = _build_feature_matrix(df, target_col)
    y    = df[target_col].copy()

    # Align on valid (non-NaN) target rows
    valid_idx = y.dropna().index
    X_df = X_df.loc[valid_idx]
    y    = y.loc[valid_idx]

    n = len(y)
    k = len(X_df.columns)

    if k == 0:
        raise ValueError("run_regression: no feature columns available after encoding.")
    if n < max(10, k + 2):
        raise ValueError(
            f"run_regression: only {n} observations for {k} features — need at least {max(10, k+2)}."
        )

    feature_names = list(X_df.columns)
    X      = X_df.values.astype(float)
    y_vals = y.values.astype(float)

    # OLS via least squares (numerically stable)
    X_const = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_const, y_vals, rcond=None)

    # Residuals and fit quality
    y_hat     = X_const @ beta
    residuals = y_vals - y_hat
    rss = float(np.dot(residuals, residuals))
    tss = float(np.dot(y_vals - y_vals.mean(), y_vals - y_vals.mean()))

    r2     = 1.0 - rss / tss if tss > 1e-12 else 0.0
    dof    = n - k - 1
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else r2

    # Standard errors
    mse = rss / dof if dof > 0 else float("nan")
    try:
        cov  = mse * np.linalg.inv(X_const.T @ X_const)
        se   = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), float("nan"))

    t_stats = beta / se
    p_vals  = [
        float(2.0 * scipy_stats.t.sf(abs(t), df=dof)) if (dof > 0 and np.isfinite(t)) else float("nan")
        for t in t_stats
    ]
    t_crit = float(scipy_stats.t.ppf(0.975, df=dof)) if dof > 0 else float("nan")

    # F-statistic
    if k > 0 and dof > 0 and tss > 1e-12 and (1.0 - r2) > 1e-12:
        f_stat = (r2 / k) / ((1.0 - r2) / dof)
        f_pval = float(scipy_stats.f.sf(f_stat, k, dof))
    else:
        f_stat = f_pval = float("nan")

    # Build coefficient records (skip intercept at index 0)
    coefs: list[RegressionCoef] = []
    for i, name in enumerate(feature_names, start=1):
        coef = float(beta[i])
        s    = float(se[i])
        t    = float(t_stats[i])
        p    = float(p_vals[i])
        ci_l = coef - t_crit * s if np.isfinite(t_crit) else float("nan")
        ci_u = coef + t_crit * s if np.isfinite(t_crit) else float("nan")
        coefs.append(RegressionCoef(
            feature     = name,
            coefficient = round(coef, 6),
            std_err     = round(s,    6),
            t_stat      = round(t,    4),
            p_value     = round(p,    4),
            ci_lower    = round(ci_l, 6),
            ci_upper    = round(ci_u, 6),
            significant = np.isfinite(p) and p < 0.05,
        ))

    coefs.sort(key=lambda c: abs(c.t_stat) if np.isfinite(c.t_stat) else 0.0, reverse=True)

    # VIF
    vif_map      = _compute_vif(X, feature_names)
    vif_warnings = [
        f"{col} (VIF={v:.1f})"
        for col, v in vif_map.items()
        if np.isfinite(v) and v > _VIF_THRESHOLD
    ]

    logger.info(
        "run_regression: target=%s, n=%d, k=%d, R²=%.4f, adj-R²=%.4f, F-p=%.4f",
        target_col, n, k, r2, adj_r2,
        f_pval if np.isfinite(f_pval) else -1,
    )

    return RegressionResult(
        target        = target_col,
        n_obs         = n,
        n_features    = k,
        r_squared     = round(r2,     4),
        adj_r_squared = round(adj_r2, 4),
        f_stat        = round(f_stat, 4) if np.isfinite(f_stat) else None,
        f_pvalue      = round(f_pval, 4) if np.isfinite(f_pval) else None,
        coefficients  = coefs,
        vif_warnings  = vif_warnings,
    )
