"""
tools/guardrail_tools.py — Automated guardrail metric sweep.

For each guardrail metric, runs a t-test between control and treatment.
A metric is "breached" when: p < alpha AND the delta moves in the harmful direction.
Harmful direction is inferred from the metric name or supplied explicitly.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


# ── Harm-direction inference by keyword ───────────────────────────────────────

# Metrics where an *increase* in treatment vs control is harmful
_INCREASE_BAD = {
    "optout", "churn", "error", "crash", "bounce",
    "spam", "unsubscribe", "block", "report", "latency",
}

# Metrics where a *decrease* in treatment vs control is harmful
_DECREASE_BAD = {
    "retain", "retention", "dau", "mau", "wau",
    "revenue", "session", "engagement", "conversion",
    "open", "click", "active", "install",
}


def _infer_harm_direction(metric: str) -> str:
    """
    Returns 'increase', 'decrease', or 'both' based on metric name keywords.
    'increase' → higher treatment value is harmful.
    'decrease' → lower treatment value is harmful.
    'both'     → any significant change is flagged.
    """
    lower = metric.lower()
    if any(kw in lower for kw in _INCREASE_BAD):
        return "increase"
    if any(kw in lower for kw in _DECREASE_BAD):
        return "decrease"
    return "both"


# ── Main function ──────────────────────────────────────────────────────────────

def check_guardrails(
    df: pd.DataFrame,
    variant_col: str,
    guardrail_metrics: list[str],
    alpha: float = 0.05,
    harm_directions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Check whether any guardrail metric was harmed by the treatment.

    Args:
        df:                DataFrame with one row per user, containing
                           variant_col and all guardrail_metrics columns.
        variant_col:       Column with 'control' / 'treatment' values.
        guardrail_metrics: List of metric column names to evaluate.
        alpha:             Significance threshold (default 0.05).
        harm_directions:   Optional per-metric override.
                           Values: 'increase' | 'decrease' | 'both'.
                           If omitted, direction is inferred from the metric name.

    Returns:
        {
            guardrails: list[{
                metric:         str,
                control_mean:   float,
                treatment_mean: float,
                delta_pct:      float,   # (treatment - control) / control * 100
                p_value:        float,
                breached:       bool,
            }],
            any_breached:  bool,
            breached_count: int,
        }
    """
    if variant_col not in df.columns:
        raise ValueError(f"variant_col '{variant_col}' not found in DataFrame.")

    variants = set(df[variant_col].dropna().unique())
    if not {"control", "treatment"}.issubset(variants):
        raise ValueError(
            f"variant_col must contain 'control' and 'treatment'. Found: {variants}"
        )

    missing = [m for m in guardrail_metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Guardrail metrics not found in DataFrame: {missing}")

    harm_directions = harm_directions or {}

    guardrails = []
    for metric in guardrail_metrics:
        ctrl = df[df[variant_col] == "control"][metric].dropna().astype(float)
        trt  = df[df[variant_col] == "treatment"][metric].dropna().astype(float)

        if len(ctrl) < 2 or len(trt) < 2:
            continue

        ctrl_mean = float(ctrl.mean())
        trt_mean  = float(trt.mean())
        delta_pct = ((trt_mean - ctrl_mean) / ctrl_mean * 100) if ctrl_mean != 0 else 0.0

        _, p_value = stats.ttest_ind(trt, ctrl, equal_var=False)
        p_value = float(p_value)

        # Determine harm direction
        direction = harm_directions.get(metric, _infer_harm_direction(metric))

        significant = p_value < alpha
        if direction == "increase":
            breached = significant and (trt_mean > ctrl_mean)
        elif direction == "decrease":
            breached = significant and (trt_mean < ctrl_mean)
        else:  # 'both'
            breached = significant

        guardrails.append({
            "metric":         metric,
            "control_mean":   round(ctrl_mean, 6),
            "treatment_mean": round(trt_mean, 6),
            "delta_pct":      round(delta_pct, 2),
            "p_value":        round(p_value, 6),
            "breached":       breached,
        })

    breached_count = sum(1 for g in guardrails if g["breached"])

    return {
        "guardrails":    guardrails,
        "any_breached":  breached_count > 0,
        "breached_count": breached_count,
    }
