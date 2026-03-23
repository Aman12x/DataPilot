"""
tools/guardrail_tools.py — Automated guardrail metric sweep.

For each guardrail metric, runs a t-test between control and treatment.
A metric is "breached" when: p < alpha AND the delta moves in the harmful direction.
Harmful direction is inferred from the metric name or supplied explicitly.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

import re

import pandas as pd
from scipy import stats

from tools.schemas import GuardrailMetric, GuardrailResult


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

# Separator pattern: split metric names into words on underscore, dash, dot, slash
_SEP_RE = re.compile(r"[_\-./]+")


def _infer_harm_direction(metric: str) -> str:
    """
    Returns 'increase', 'decrease', or 'both' based on metric name keywords.
    'increase' → higher treatment value is harmful.
    'decrease' → lower treatment value is harmful.
    'both'     → any significant change is flagged.

    Splits on word separators (_-./) then uses prefix matching so that
    'retained' matches 'retain', 'sessions' matches 'session', etc.
    Compound names like 'retention_vs_churn_ratio' match both sets and
    return 'both' (the safest default — avoids false-positive breach suppression).
    """
    words = _SEP_RE.split(metric.lower())

    def _matches(kw_set: set[str]) -> bool:
        return any(w.startswith(kw) for w in words for kw in kw_set)

    has_increase = _matches(_INCREASE_BAD)
    has_decrease = _matches(_DECREASE_BAD)
    if has_increase and not has_decrease:
        return "increase"
    if has_decrease and not has_increase:
        return "decrease"
    return "both"


# ── Main function ──────────────────────────────────────────────────────────────

def check_guardrails(
    df: pd.DataFrame,
    variant_col: str,
    guardrail_metrics: list[str],
    alpha: float = 0.05,
    harm_directions: dict[str, str] | None = None,
    default_direction: str = "both",
) -> GuardrailResult:
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
                           When provided, takes full precedence over keyword
                           inference for the metrics it covers.
        default_direction: Fallback direction used when a metric is not
                           covered by harm_directions AND keyword inference
                           returns no match. Replaces the previous hardcoded
                           'both' fallback. Set to 'decrease' when the
                           primary metric is higher_is_better so unknown
                           guardrail drops are treated as harmful.

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

    guardrails: list[GuardrailMetric] = []
    for metric in guardrail_metrics:
        ctrl = df[df[variant_col] == "control"][metric].dropna().astype(float)
        trt  = df[df[variant_col] == "treatment"][metric].dropna().astype(float)

        if len(ctrl) < 2 or len(trt) < 2:
            continue

        ctrl_mean = float(ctrl.mean())
        trt_mean  = float(trt.mean())
        if ctrl_mean != 0:
            delta_pct = (trt_mean - ctrl_mean) / abs(ctrl_mean) * 100
        else:
            # Control baseline is 0 — relative % is undefined.
            # Report as percentage-point absolute change (e.g. 0→0.05 = +5pp).
            delta_pct = (trt_mean - ctrl_mean) * 100

        _, p_value = stats.ttest_ind(trt, ctrl, equal_var=False)
        p_value = float(p_value)

        # Determine harm direction.
        # Priority: explicit harm_directions > keyword inference > default_direction.
        if harm_directions and metric in harm_directions:
            direction = harm_directions[metric]
        else:
            inferred = _infer_harm_direction(metric)
            direction = inferred if inferred != "both" else default_direction

        significant = p_value < alpha
        if direction == "increase":
            breached = significant and (trt_mean > ctrl_mean)
        elif direction == "decrease":
            breached = significant and (trt_mean < ctrl_mean)
        else:  # 'both'
            breached = significant

        guardrails.append(GuardrailMetric(
            metric=metric,
            control_mean=round(ctrl_mean, 6),
            treatment_mean=round(trt_mean, 6),
            delta_pct=round(delta_pct, 2),
            p_value=round(p_value, 6),
            breached=breached,
        ))

    breached_count = sum(1 for g in guardrails if g.breached)

    return GuardrailResult(
        guardrails=guardrails,
        any_breached=breached_count > 0,
        breached_count=breached_count,
    )
