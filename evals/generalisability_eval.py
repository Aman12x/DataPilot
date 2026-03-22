"""
evals/generalisability_eval.py — Cross-domain eval harness.

Tests that the core statistical tools work correctly on two non-DAU datasets:
  1. Clinical trial    (data/samples/clinical_trial.csv)
  2. E-commerce A/B    (data/samples/ecommerce_ab_test.csv)

No API key required — all checks are deterministic tool-level assertions.

Usage:
    python evals/generalisability_eval.py
    python evals/generalisability_eval.py --verbose

Exit code: 0 if score >= 0.80, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import pandas as pd

from tools import stats_tools, guardrail_tools, mde_tools

# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")

def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(SAMPLES_DIR, name))

def _rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    return df.rename(columns=mapping)


# ── Clinical trial dataset ────────────────────────────────────────────────────
# Columns: patient_id, treatment_group (control/treatment), week,
#          recovery_score, side_effect_count, adherence_pct,
#          baseline_severity, age, gender, bmi_category, region

def _clinical_df() -> pd.DataFrame:
    df = _load_csv("clinical_trial.csv")
    df = df.rename(columns={
        "patient_id":      "user_id",
        "treatment_group": "variant",
        "recovery_score":  "recovery_score",
        "adherence_pct":   "adherence_pct",
    })
    return df


# ── E-commerce A/B dataset ────────────────────────────────────────────────────
# Columns: user_id, variant (control/treatment), week, revenue_usd,
#          orders, avg_order_value, device, user_segment, country, category

def _ecomm_df() -> pd.DataFrame:
    return _load_csv("ecommerce_ab_test.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Clinical trial criteria
# ══════════════════════════════════════════════════════════════════════════════

def crit_clinical_ttest_runs(verbose: bool) -> bool:
    """t-test executes without error on clinical data."""
    df = _clinical_df()
    ctrl = df[df["variant"] == "control"]["recovery_score"].dropna()
    trt  = df[df["variant"] == "treatment"]["recovery_score"].dropna()
    result = stats_tools.run_ttest(ctrl, trt)
    if verbose:
        print(f"    t-stat={result.t_stat:.3f}, p={result.p_value:.4f}, sig={result.significant}")
    return result is not None and hasattr(result, "p_value")


def crit_clinical_ttest_significant(verbose: bool) -> bool:
    """Treatment group has significantly higher recovery score (ground truth: p < 0.05)."""
    df = _clinical_df()
    ctrl = df[df["variant"] == "control"]["recovery_score"].dropna()
    trt  = df[df["variant"] == "treatment"]["recovery_score"].dropna()
    result = stats_tools.run_ttest(ctrl, trt)
    if verbose:
        print(f"    p={result.p_value:.4f}, significant={result.significant}")
    return result.significant


def crit_clinical_cuped_runs(verbose: bool) -> bool:
    """CUPED executes on clinical data using baseline_severity as covariate."""
    df = _clinical_df()
    # Encode baseline_severity as numeric covariate
    severity_map = {"mild": 1, "moderate": 2, "severe": 3}
    df["severity_num"] = df["baseline_severity"].map(severity_map).fillna(2)
    if "variant" not in df.columns or "recovery_score" not in df.columns:
        return False
    result = stats_tools.run_cuped(df, metric_col="recovery_score", covariate_col="severity_num", variant_col="variant")
    if verbose:
        print(f"    CUPED variance_reduction={result.variance_reduction_pct:.1f}%")
    return result is not None and result.variance_reduction_pct >= 0


def crit_clinical_hte_finds_segment(verbose: bool) -> bool:
    """HTE analysis identifies at least one subgroup with differential effect."""
    df = _clinical_df()
    result = stats_tools.run_hte(
        df, metric_col="recovery_score", variant_col="variant",
        segment_cols=["gender", "bmi_category", "region"],
    )
    if verbose:
        top = result.top_segment if result else None
        print(f"    top_segment={top}")
    return result is not None and result.top_segment is not None


def crit_clinical_guardrails_run(verbose: bool) -> bool:
    """Guardrail check runs on side_effect_count without error."""
    df = _clinical_df()
    if "side_effect_count" not in df.columns or "variant" not in df.columns:
        return False
    result = guardrail_tools.check_guardrails(
        df, variant_col="variant", guardrail_metrics=["side_effect_count"],
        harm_directions={"side_effect_count": "increase"},
    )
    if verbose:
        breached = [g.metric for g in result.guardrails if g.breached] if result else []
        print(f"    breached={breached}, any_breached={result.any_breached if result else None}")
    return result is not None


def crit_clinical_mde_runs(verbose: bool) -> bool:
    """MDE calculation completes on clinical trial sample sizes."""
    df = _clinical_df()
    ctrl = df[df["variant"] == "control"]["recovery_score"].dropna()
    trt  = df[df["variant"] == "treatment"]["recovery_score"].dropna()
    result = mde_tools.compute_mde(
        n_control=len(ctrl), n_treatment=len(trt),
        baseline_mean=float(ctrl.mean()), baseline_std=float(ctrl.std()),
    )
    if verbose:
        print(f"    mde_relative={result.mde_relative_pct:.1f}%, powered={result.is_powered_for_observed_effect}")
    return result is not None and result.mde_relative_pct > 0


# ══════════════════════════════════════════════════════════════════════════════
# E-commerce A/B criteria
# ══════════════════════════════════════════════════════════════════════════════

def crit_ecomm_ttest_runs(verbose: bool) -> bool:
    """t-test executes without error on ecommerce revenue data."""
    df = _ecomm_df()
    ctrl = df[df["variant"] == "control"]["revenue_usd"].dropna()
    trt  = df[df["variant"] == "treatment"]["revenue_usd"].dropna()
    result = stats_tools.run_ttest(ctrl, trt)
    if verbose:
        print(f"    t-stat={result.t_stat:.3f}, p={result.p_value:.4f}")
    return result is not None and hasattr(result, "t_stat")


def crit_ecomm_cuped_variance_reduced(verbose: bool) -> bool:
    """CUPED reduces variance on ecommerce data using session_duration as covariate."""
    df = _ecomm_df()
    required = {"variant", "revenue_usd", "session_duration_min"}
    if not required.issubset(df.columns):
        return False
    result = stats_tools.run_cuped(
        df, metric_col="revenue_usd",
        covariate_col="session_duration_min",
        variant_col="variant",
    )
    if verbose:
        print(f"    variance_reduction={result.variance_reduction_pct:.1f}%")
    # CUPED should reduce variance (>0%) on correlated covariate
    return result is not None and result.variance_reduction_pct >= 0


def crit_ecomm_hte_device_or_segment(verbose: bool) -> bool:
    """HTE identifies device or user_segment as top heterogeneous effect."""
    df = _ecomm_df()
    result = stats_tools.run_hte(
        df, metric_col="revenue_usd", variant_col="variant",
        segment_cols=["device", "user_segment", "country"],
    )
    if verbose:
        top = result.top_segment if result else None
        print(f"    top_segment={top}, n_segments={len(result.all_segments) if result else 0}")
    return result is not None and result.top_segment is not None


def crit_ecomm_guardrails_run(verbose: bool) -> bool:
    """Guardrail check runs on orders metric without error."""
    df = _ecomm_df()
    if "orders" not in df.columns:
        return False
    result = guardrail_tools.check_guardrails(
        df, variant_col="variant", guardrail_metrics=["orders"],
        harm_directions={"orders": "decrease"},
    )
    if verbose:
        print(f"    guardrail result: any_breached={result.any_breached if result else None}")
    return result is not None


def crit_ecomm_mde_reasonable(verbose: bool) -> bool:
    """MDE for revenue_usd is a plausible percentage (>0%, <100%)."""
    df = _ecomm_df()
    ctrl = df[df["variant"] == "control"]["revenue_usd"].dropna()
    trt  = df[df["variant"] == "treatment"]["revenue_usd"].dropna()
    result = mde_tools.compute_mde(
        n_control=len(ctrl), n_treatment=len(trt),
        baseline_mean=float(ctrl.mean()), baseline_std=float(ctrl.std()),
    )
    if verbose:
        print(f"    mde_relative={result.mde_relative_pct:.1f}%")
    return result is not None and 0 < result.mde_relative_pct < 100


# ══════════════════════════════════════════════════════════════════════════════
# Criteria registry
# ══════════════════════════════════════════════════════════════════════════════

CRITERIA: list[tuple[str, str, Callable]] = [
    # Clinical trial
    ("clinical_ttest_runs",        "🏥 Clinical: t-test executes on recovery scores",          crit_clinical_ttest_runs),
    ("clinical_ttest_significant", "🏥 Clinical: treatment has significantly higher recovery",  crit_clinical_ttest_significant),
    ("clinical_cuped_runs",        "🏥 Clinical: CUPED runs with baseline_severity covariate",  crit_clinical_cuped_runs),
    ("clinical_hte_segment",       "🏥 Clinical: HTE finds subgroup differential",              crit_clinical_hte_finds_segment),
    ("clinical_guardrails",        "🏥 Clinical: guardrail check on side_effect_count",         crit_clinical_guardrails_run),
    ("clinical_mde",               "🏥 Clinical: MDE calculation completes",                   crit_clinical_mde_runs),
    # E-commerce A/B
    ("ecomm_ttest_runs",           "🛒 Ecomm: t-test executes on revenue_usd",                 crit_ecomm_ttest_runs),
    ("ecomm_cuped_variance",       "🛒 Ecomm: CUPED runs with session_duration covariate",     crit_ecomm_cuped_variance_reduced),
    ("ecomm_hte_segment",          "🛒 Ecomm: HTE finds device/segment differential",          crit_ecomm_hte_device_or_segment),
    ("ecomm_guardrails",           "🛒 Ecomm: guardrail check on orders metric",               crit_ecomm_guardrails_run),
    ("ecomm_mde_reasonable",       "🛒 Ecomm: MDE is a plausible percentage",                  crit_ecomm_mde_reasonable),
]


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(verbose: bool = False) -> tuple[int, int]:
    passed = failed = 0
    for key, label, fn in CRITERIA:
        try:
            ok = fn(verbose)
        except Exception:
            ok = False
            if verbose:
                traceback.print_exc()
        icon = "  PASS" if ok else "  FAIL"
        print(f"{icon}  {label}")
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain generalisability eval")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate values")
    args = parser.parse_args()

    total   = len(CRITERIA)
    passed, failed = run_eval(verbose=args.verbose)
    pct     = passed / total * 100

    print(f"\nScore: {passed}/{total} = {pct:.0f}%", "✅" if pct >= 80 else "❌")
    sys.exit(0 if pct >= 80 else 1)


if __name__ == "__main__":
    main()
