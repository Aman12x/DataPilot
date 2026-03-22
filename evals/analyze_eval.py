"""
evals/analyze_eval.py — Offline evaluation harness for the Analyze module.

Runs all analysis tools directly against the real DuckDB dataset and checks
each result against known ground truth. Narrative criteria require a valid
ANTHROPIC_API_KEY; all other 9 criteria run without any API calls.

Usage:
    python evals/analyze_eval.py
    python evals/analyze_eval.py --db data/dau_experiment.db
    python evals/analyze_eval.py --skip-narrative   # skip LLM criteria

Exit code: 0 if score >= 0.80, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import traceback
from typing import Any, Callable

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools import (
    anomaly_tools,
    decomposition_tools,
    forecast_tools,
    funnel_tools,
    guardrail_tools,
    mde_tools,
    narrative_tools,
    novelty_tools,
    stats_tools,
)
from tools.db_tools import DBConnection
from tools.eval_tools import score_faithfulness, score_relevancy
from memory.store import log_run, update_eval_score

# ── Ground-truth SQL queries ──────────────────────────────────────────────────

# Experiment-level data: one row per (user, week) with aggregated metrics.
# Pre-experiment session count is computed from days before experiment start.
_EXPERIMENT_SQL = """
WITH pre_exp AS (
    SELECT user_id,
           AVG(session_count)::FLOAT AS pre_session_count
    FROM   events
    WHERE  date < DATE '2024-01-29'
    GROUP  BY user_id
)
SELECT
    e.user_id,
    ex.variant,
    ex.week,
    AVG(e.dau_flag)::FLOAT            AS dau_rate,
    COALESCE(p.pre_session_count, 0)  AS pre_session_count,
    AVG(e.notif_optout)::FLOAT        AS notif_optout,
    AVG(e.d7_retained)::FLOAT         AS d7_retained,
    AVG(e.session_count)::FLOAT       AS session_count,
    e.platform,
    e.user_segment
FROM       experiment ex
JOIN       events e  ON  ex.user_id = e.user_id
                     AND e.date >= DATE '2024-01-29'
LEFT JOIN  pre_exp p ON  ex.user_id = p.user_id
GROUP BY   e.user_id, ex.variant, ex.week, e.platform, e.user_segment,
           p.pre_session_count
LIMIT 50000
"""

# Time-series metrics: full metrics_daily for all platforms/segments.
_METRICS_DAILY_SQL = "SELECT * FROM metrics_daily ORDER BY date"

# Funnel data: join with experiment to get variant column.
_FUNNEL_SQL = """
SELECT f.user_id, ex.variant, f.step, f.completed
FROM   funnel f
JOIN   experiment ex ON f.user_id = ex.user_id AND ex.week = 1
"""

# ── Eval criteria ─────────────────────────────────────────────────────────────
# Each value is a (description, criterion_fn) pair.
# criterion_fn receives the full state dict; returns True on pass.

def _safe(fn: Callable[[dict], bool]) -> Callable[[dict], bool]:
    """Wrap criterion to return False (not crash) if a key is missing."""
    def wrapped(state: dict) -> bool:
        try:
            return bool(fn(state))
        except (KeyError, TypeError, IndexError, AttributeError):
            return False
    return wrapped


EVAL_CRITERIA: dict[str, tuple[str, Callable[[dict], bool]]] = {
    "hte_correct_segment": (
        "HTE surfaces android/new as the top affected segment",
        _safe(lambda s: s["hte_result"].top_segment == "platform=android,user_segment=new"),
    ),
    "cuped_variance_reduced": (
        "CUPED reduces variance by >15%",
        _safe(lambda s: s["cuped_result"].variance_reduction_pct > 15),
    ),
    "ttest_significant": (
        "T-test is significant (p < 0.05)",
        _safe(lambda s: s["ttest_result"].significant is True),
    ),
    "decomp_identifies_new": (
        "Decomposition identifies 'new' as dominant change component",
        _safe(lambda s: "new" in s["decomposition_result"].dominant_change_component.lower()),
    ),
    "slice_ranks_android_first": (
        "Slice-and-dice ranks android as top dimension",
        _safe(lambda s: s["slice_result"].ranked_dimensions[0].value == "android"),
    ),
    "forecast_flags_drop": (
        "Forecast baseline shows actuals outside confidence interval",
        _safe(lambda s: s["forecast_result"].outside_ci is True),
    ),
    "guardrails_breached_found": (
        "At least one guardrail metric is breached",
        _safe(lambda s: s["guardrail_result"].any_breached is True),
    ),
    "optout_breached": (
        "notif_optout is flagged as a breached guardrail",
        _safe(lambda s: any(
            g.metric == "notif_optout" and g.breached
            for g in s["guardrail_result"].guardrails
        )),
    ),
    "novelty_ruled_out": (
        "Novelty effect correctly ruled out (effect is growing, not decaying)",
        _safe(lambda s: s["novelty_result"].novelty_likely is False),
    ),
    "narrative_mentions_segment": (
        "Narrative mentions 'android' and 'new'",
        _safe(lambda s: (
            "android" in s["narrative_draft"].lower()
            and "new" in s["narrative_draft"].lower()
        )),
    ),
    "narrative_has_caveats": (
        "Narrative contains a caveats or limitations section",
        _safe(lambda s: (
            "caveat" in s["narrative_draft"].lower()
            or "limitation" in s["narrative_draft"].lower()
        )),
    ),
    # ── RAGAS-inspired: faithfulness ─────────────────────────────────────────
    "narrative_faithful": (
        "Numbers cited in narrative are supported by query results (faithfulness ≥ 0.70)",
        _safe(lambda s: score_faithfulness(
            s.get("narrative_draft", ""),
            s.get("query_result"),
        )["score"] >= 0.70),
    ),
    # ── RAGAS-inspired: relevancy ────────────────────────────────────────────
    "narrative_relevant": (
        "Narrative is semantically relevant to the task (relevancy ≥ 0.60)",
        _safe(lambda s: score_relevancy(
            s.get("task", "DAU drop investigation"),
            s.get("narrative_draft", ""),
        ) >= 0.60),
    ),
}


# ── Tool runners ──────────────────────────────────────────────────────────────

def _run_tools(db_path: str, skip_narrative: bool) -> dict[str, Any]:
    """
    Run all analysis tools against the real DuckDB and return a state-like dict.
    No LangGraph, no LLM (except optionally for narrative).
    """
    conn = DBConnection(backend="duckdb", path=db_path)
    state: dict[str, Any] = {}

    # ── Load data ─────────────────────────────────────────────────────────────
    print("  Loading experiment data...")
    exp_df = conn.query(_EXPERIMENT_SQL)

    print("  Loading metrics_daily...")
    daily_df = conn.query(_METRICS_DAILY_SQL)

    print("  Loading funnel data...")
    funnel_df = conn.query(_FUNNEL_SQL)

    # ── Experiment analysis ───────────────────────────────────────────────────
    print("  Running CUPED...")
    state["cuped_result"] = stats_tools.run_cuped(
        exp_df,
        metric_col="dau_rate",
        covariate_col="pre_session_count",
        variant_col="variant",
    )

    print("  Running t-test...")
    ctrl = exp_df[exp_df["variant"] == "control"]["dau_rate"].dropna()
    trt  = exp_df[exp_df["variant"] == "treatment"]["dau_rate"].dropna()
    state["ttest_result"] = stats_tools.run_ttest(ctrl, trt)

    print("  Running HTE...")
    state["hte_result"] = stats_tools.run_hte(
        exp_df,
        metric_col="dau_rate",
        variant_col="variant",
        segment_cols=["platform", "user_segment"],
    )

    print("  Running novelty detection...")
    state["novelty_result"] = novelty_tools.detect_novelty_effect(
        exp_df,
        metric_col="dau_rate",
        variant_col="variant",
        week_col="week",
    )

    print("  Computing MDE...")
    cuped_ate = state["cuped_result"].cuped_ate
    state["mde_result"] = mde_tools.compute_mde(
        n_control=len(ctrl),
        n_treatment=len(trt),
        baseline_mean=float(ctrl.mean()),
        baseline_std=float(ctrl.std()),
        observed_effect_abs=cuped_ate,
    )
    state["business_impact"] = mde_tools.business_impact_statement(
        mde_relative_pct=state["mde_result"].mde_relative_pct,
        metric="dau_rate",
        baseline_dau=int(os.getenv("BASELINE_DAU", "500000")),
        revenue_per_dau=float(os.getenv("REVENUE_PER_DAU", "0.50")),
    )

    # ── Guardrails ────────────────────────────────────────────────────────────
    print("  Checking guardrails...")
    guardrail_metrics = [m for m in ["notif_optout", "d7_retained", "session_count"]
                         if m in exp_df.columns]
    state["guardrail_result"] = guardrail_tools.check_guardrails(
        exp_df,
        variant_col="variant",
        guardrail_metrics=guardrail_metrics,
        default_direction="decrease",
    )

    # ── Funnel ────────────────────────────────────────────────────────────────
    print("  Computing funnel...")
    present_steps = set(funnel_df["step"].dropna().unique())
    steps = [s for s in ["impression", "click", "install", "d1_retain"]
             if s in present_steps]
    if len(steps) >= 2:
        state["funnel_result"] = funnel_tools.compute_funnel(
            funnel_df, variant_col="variant", steps=steps
        )

    # ── Time-series (uses metrics_daily, aggregated to platform level) ────────
    # Sum across user_segments for a cleaner time series
    agg_daily = (
        daily_df
        .groupby(["date", "platform"])
        [["dau", "new_users", "retained_users", "resurrected_users", "churned_users"]]
        .sum()
        .reset_index()
    )

    print("  Decomposing DAU...")
    try:
        state["decomposition_result"] = decomposition_tools.decompose_dau(agg_daily)
    except Exception as exc:
        print(f"    decompose_dau skipped: {exc}")

    print("  Detecting anomaly...")
    try:
        state["anomaly_result"] = anomaly_tools.detect_anomaly(
            agg_daily, metric_col="dau", date_col="date"
        )
        state["slice_result"] = anomaly_tools.slice_and_dice(
            agg_daily, metric_col="dau", date_col="date",
            dimension_cols=["platform"]
        )
    except Exception as exc:
        print(f"    anomaly detection skipped: {exc}")

    print("  Forecasting baseline...")
    try:
        state["forecast_result"] = forecast_tools.forecast_baseline(
            agg_daily, metric_col="dau", date_col="date"
        )
    except Exception as exc:
        print(f"    forecast skipped: {exc}")

    # ── Narrative (requires API key) ──────────────────────────────────────────
    if skip_narrative or not os.getenv("ANTHROPIC_API_KEY"):
        print("  Skipping narrative (no ANTHROPIC_API_KEY or --skip-narrative).")
        # Use template-only narrative for partial credit on narrative criteria
        def _to_dict(v: Any) -> dict:
            if v is None: return {}
            if hasattr(v, "model_dump"): return v.model_dump()
            return v if isinstance(v, dict) else {}

        try:
            template_out = narrative_tools.format_narrative(
                metric="dau_rate",
                decomposition_result=_to_dict(state.get("decomposition_result")),
                anomaly_result=_to_dict(state.get("anomaly_result")),
                cuped_result=_to_dict(state.get("cuped_result")),
                ttest_result=_to_dict(state.get("ttest_result")),
                hte_result=_to_dict(state.get("hte_result")),
                novelty_result=_to_dict(state.get("novelty_result")),
                mde_result=_to_dict(state.get("mde_result")),
                guardrail_result=_to_dict(state.get("guardrail_result")),
                funnel_result=_to_dict(state.get("funnel_result")),
                forecast_result=_to_dict(state.get("forecast_result")),
                business_impact=state.get("business_impact") or "",
                analyst_notes="",
            )
            state["narrative_draft"] = template_out.narrative_draft
        except Exception as exc:
            print(f"    template narrative skipped: {exc}")
            state["narrative_draft"] = ""
    else:
        print("  Generating narrative via LLM...")
        try:
            from agents.analyze.nodes import generate_narrative as _gen_narrative
            from agents.state import AgentState
            # Build a minimal AgentState-compatible dict and call the node
            fake_state: dict = {
                "metric": "dau_rate",
                "schema_context": "",
                "relevant_history": [],
                "analyst_notes": "",
                "conversation_history": [],
                **{k: v for k, v in state.items()},
            }
            result = _gen_narrative(fake_state)  # type: ignore[arg-type]
            state["narrative_draft"] = result.get("narrative_draft", "")
        except Exception as exc:
            print(f"    LLM narrative failed: {exc}")
            state["narrative_draft"] = ""

    return state


# ── Scorer ────────────────────────────────────────────────────────────────────

def score(state: dict[str, Any], skip_narrative: bool = False) -> dict[str, Any]:
    """Evaluate state against all criteria. Returns per-criterion results + summary."""
    results = {}
    for name, (description, fn) in EVAL_CRITERIA.items():
        is_narrative = "narrative" in name
        if skip_narrative and is_narrative and not state.get("narrative_draft"):
            results[name] = {"description": description, "passed": None, "skipped": True}
        else:
            passed = fn(state)
            results[name] = {"description": description, "passed": passed, "skipped": False}

    evaluated = [r for r in results.values() if not r["skipped"]]
    n_pass     = sum(1 for r in evaluated if r["passed"])
    n_total    = len(evaluated)
    score_val  = n_pass / n_total if n_total else 0.0

    return {"criteria": results, "score": score_val, "n_pass": n_pass, "n_total": n_total}


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(result: dict[str, Any]) -> None:
    width = 70
    print()
    print("=" * width)
    print("DATAPILOT EVAL REPORT — Analyze module")
    print("=" * width)

    criteria = result["criteria"]
    for name, r in criteria.items():
        if r["skipped"]:
            status = "  SKIP"
            colour = ""
        elif r["passed"]:
            status = "  PASS"
        else:
            status = "  FAIL"

        desc = textwrap.shorten(r["description"], width=width - 12)
        print(f"{status}  {name}")
        print(f"       {desc}")

    print()
    score_val = result["score"]
    n_pass    = result["n_pass"]
    n_total   = result["n_total"]
    target    = "✅ ABOVE TARGET (≥80%)" if score_val >= 0.80 else "❌ BELOW TARGET (<80%)"
    print(f"Score: {n_pass}/{n_total} = {score_val:.0%}  {target}")
    print("=" * width)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="DataPilot offline eval harness")
    parser.add_argument("--db", default=os.getenv("DUCKDB_PATH", "data/dau_experiment.db"),
                        help="Path to DuckDB file")
    parser.add_argument("--skip-narrative", action="store_true",
                        help="Skip LLM-dependent narrative criteria")
    parser.add_argument("--json", dest="json_out", action="store_true",
                        help="Print JSON result to stdout instead of human report")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DuckDB not found at '{args.db}'.")
        print("Run:  python data/generate_data.py")
        return 2

    print(f"Running eval against: {args.db}")
    print()

    try:
        state = _run_tools(args.db, skip_narrative=args.skip_narrative)
    except Exception:
        print("ERROR during tool execution:")
        traceback.print_exc()
        return 2

    result = score(state, skip_narrative=args.skip_narrative)

    if args.json_out:
        # Serialise Pydantic models before printing
        serialisable = {
            "criteria": result["criteria"],
            "score":    result["score"],
            "n_pass":   result["n_pass"],
            "n_total":  result["n_total"],
        }
        print(json.dumps(serialisable, indent=2))
    else:
        print_report(result)

    # Feed eval score back into memory store so history injection can learn from it
    run_id = log_run(
        task="offline eval — DAU drop investigation",
        metric="dau_rate",
        covariate="pre_session_count",
        db_backend="duckdb",
        top_segment=(
            state.get("hte_result").top_segment
            if state.get("hte_result") and hasattr(state["hte_result"], "top_segment")
            else ""
        ),
        eval_score=result["score"],
    )
    update_eval_score(run_id, result["score"])
    print(f"\nEval run logged to memory store (run_id: {run_id}, score: {result['score']:.2f})")

    return 0 if result["score"] >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
