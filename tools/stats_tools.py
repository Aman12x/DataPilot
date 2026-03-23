"""
tools/stats_tools.py — CUPED variance reduction, t-test, HTE subgroup analysis.

All functions are pure Python (scipy/numpy only). No LangGraph or Streamlit imports.
Each returns a typed dict. Raises ValueError with a human-readable message on bad input.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from tools.schemas import CupedResult, HteResult, SegmentResult, TtestResult


def run_cuped(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    variant_col: str,
) -> CupedResult:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

    Adjusts the metric by regressing out the pre-experiment covariate, reducing
    variance and improving sensitivity.

    Args:
        df:            DataFrame with one row per user. Must contain metric_col,
                       covariate_col, and variant_col.
        metric_col:    Post-experiment outcome (e.g. 'dau_flag', 'session_count').
        covariate_col: Pre-experiment covariate (e.g. 'pre_session_count').
        variant_col:   Column with 'control' / 'treatment' values.

    Returns:
        {
            raw_ate:               float,  # unadjusted average treatment effect
            cuped_ate:             float,  # CUPED-adjusted ATE
            variance_reduction_pct: float, # % reduction in outcome variance
            theta:                 float,  # regression coefficient (cov/var of covariate)
        }
    """
    _validate_columns(df, [metric_col, covariate_col, variant_col])
    _validate_variants(df, variant_col)

    df = df[[metric_col, covariate_col, variant_col]].dropna()

    control   = df[df[variant_col] == "control"]
    treatment = df[df[variant_col] == "treatment"]

    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Need at least 2 observations per variant for CUPED.")

    # Theta: OLS coefficient of metric ~ covariate (pooled)
    cov_matrix = np.cov(df[metric_col], df[covariate_col])
    var_covariate = float(df[covariate_col].var(ddof=1))
    if var_covariate == 0:
        raise ValueError(f"Covariate '{covariate_col}' has zero variance — cannot apply CUPED.")

    theta = cov_matrix[0, 1] / var_covariate

    # CUPED-adjusted outcome: Y_adj = Y - theta * (X - mean(X))
    mean_covariate = df[covariate_col].mean()
    df = df.copy()
    df["_cuped"] = df[metric_col] - theta * (df[covariate_col] - mean_covariate)

    control_adj   = df[df[variant_col] == "control"]["_cuped"]
    treatment_adj = df[df[variant_col] == "treatment"]["_cuped"]

    raw_ate   = treatment[metric_col].mean() - control[metric_col].mean()
    cuped_ate = treatment_adj.mean() - control_adj.mean()

    # Variance reduction: compare outcome variance before vs after adjustment
    var_before = df[metric_col].var(ddof=1)
    var_after  = df["_cuped"].var(ddof=1)
    variance_reduction_pct = (1 - var_after / var_before) * 100 if var_before > 0 else 0.0

    return CupedResult(
        raw_ate=round(float(raw_ate), 6),
        cuped_ate=round(float(cuped_ate), 6),
        variance_reduction_pct=round(float(variance_reduction_pct), 2),
        theta=round(float(theta), 6),
    )


def run_ttest(
    control: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    alpha: float = 0.05,
    winsorize_pct: float = 0.0,
) -> TtestResult:
    """
    Two-sample Welch's t-test (unequal variance).

    Args:
        control:       Outcome values for control group.
        treatment:     Outcome values for treatment group.
        alpha:         Significance threshold (default 0.05).
        winsorize_pct: If > 0, clip both arrays at this quantile on each tail before
                       testing (e.g. 0.01 clips bottom/top 1%). Useful for revenue-
                       style metrics dominated by a handful of outliers. Default 0 = off.

    Returns:
        TtestResult with t_stat, p_value, ci_lower, ci_upper, significant,
        cohens_d (pooled effect size), n_control, n_treatment.
    """
    control   = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    control   = control[~np.isnan(control)]
    treatment = treatment[~np.isnan(treatment)]

    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("Need at least 2 observations per group for t-test.")

    if winsorize_pct > 0.0:
        control   = _winsorize(control,   winsorize_pct)
        treatment = _winsorize(treatment, winsorize_pct)

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    # 95% CI on the mean difference using Welch-Satterthwaite degrees of freedom
    mean_diff = treatment.mean() - control.mean()
    se_diff   = np.sqrt(treatment.var(ddof=1) / len(treatment) +
                        control.var(ddof=1)  / len(control))

    n1, n2   = len(treatment), len(control)
    s1, s2   = treatment.var(ddof=1), control.var(ddof=1)
    df_welch = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
    t_crit   = stats.t.ppf(1 - alpha / 2, df=df_welch)

    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Cohen's d: mean difference divided by the pooled standard deviation.
    # Interpretation: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large.
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    cohens_d   = float(mean_diff / pooled_std) if pooled_std > 1e-12 else 0.0

    return TtestResult(
        t_stat=round(float(t_stat), 4),
        p_value=round(float(p_value), 6),
        ci_lower=round(float(ci_lower), 6),
        ci_upper=round(float(ci_upper), 6),
        significant=bool(p_value < alpha),
        cohens_d=round(cohens_d, 4),
        n_control=int(n2),
        n_treatment=int(n1),
    )


def _winsorize(arr: np.ndarray, pct: float) -> np.ndarray:
    """Clip array values below `pct`-th and above `(1-pct)`-th quantile."""
    lo = np.percentile(arr, pct * 100)
    hi = np.percentile(arr, (1.0 - pct) * 100)
    return np.clip(arr, lo, hi)


def run_hte(
    df: pd.DataFrame,
    metric_col: str,
    variant_col: str,
    segment_cols: list[str],
    alpha: float = 0.05,
    min_segment_size: int = 30,
) -> HteResult:
    """
    Heterogeneous Treatment Effect analysis via manual subgroup t-tests.

    For each subgroup defined by all combinations of segment_cols values,
    computes the ATE and tests significance. Returns subgroups ranked by
    absolute effect size.

    Args:
        df:               DataFrame with one row per user.
        metric_col:       Outcome column.
        variant_col:      Column with 'control' / 'treatment'.
        segment_cols:     List of columns to slice by (e.g. ['platform', 'user_segment']).
        alpha:            Significance threshold.
        min_segment_size: Minimum observations per variant per segment to include.

    Returns:
        {
            top_segment:   str,    # e.g. "platform=android,user_segment=new"
            effect_size:   float,  # ATE in the top segment
            segment_share: float,  # fraction of total users in the top segment
            all_segments:  list[dict],  # all subgroups, sorted by abs(effect_size) desc
        }
    """
    _validate_columns(df, [metric_col, variant_col] + segment_cols)
    _validate_variants(df, variant_col)

    df = df[[metric_col, variant_col] + segment_cols].dropna()
    total_users = len(df)

    # Build all unique value combinations for the segment columns
    unique_values = [df[col].unique().tolist() for col in segment_cols]
    combos = list(itertools.product(*unique_values))

    results = []
    for combo in combos:
        # Build boolean mask for this subgroup
        mask = pd.Series([True] * len(df), index=df.index)
        label_parts = []
        for col, val in zip(segment_cols, combo):
            mask &= df[col] == val
            label_parts.append(f"{col}={val}")

        subdf = df[mask]
        ctrl  = subdf[subdf[variant_col] == "control"][metric_col]
        trt   = subdf[subdf[variant_col] == "treatment"][metric_col]

        if len(ctrl) < min_segment_size or len(trt) < min_segment_size:
            continue

        try:
            ttest = run_ttest(ctrl, trt, alpha=alpha)
        except ValueError:
            continue

        ate           = float(trt.mean() - ctrl.mean())
        segment_share = len(subdf) / total_users

        results.append(SegmentResult(
            segment=",".join(label_parts),
            effect_size=round(ate, 6),
            segment_share=round(segment_share, 4),
            p_value=ttest.p_value,
            significant=ttest.significant,
            n_control=len(ctrl),
            n_treatment=len(trt),
        ))

    if not results:
        raise ValueError(
            f"No subgroups had >= {min_segment_size} observations per variant. "
            "Try reducing min_segment_size or adding more data."
        )

    # Bonferroni correction: adjust significance threshold for multiple comparisons.
    # Running K independent tests at α=0.05 gives a familywise error rate of
    # 1-(0.95)^K, so we divide alpha by the number of tests performed.
    n_tests = len(results)
    if n_tests > 1:
        bonferroni_alpha = alpha / n_tests
        for r in results:
            r.significant = r.p_value < bonferroni_alpha

    # Sort: significant segments first, then by absolute effect size within each group.
    results.sort(key=lambda r: (not r.significant, -abs(r.effect_size)))

    top = results[0]
    return HteResult(
        top_segment=top.segment,
        effect_size=top.effect_size,
        segment_share=top.segment_share,
        all_segments=results,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")


def _validate_variants(df: pd.DataFrame, variant_col: str) -> None:
    variants = set(df[variant_col].dropna().unique())
    if not {"control", "treatment"}.issubset(variants):
        raise ValueError(
            f"variant_col '{variant_col}' must contain both 'control' and 'treatment'. "
            f"Found: {variants}"
        )
