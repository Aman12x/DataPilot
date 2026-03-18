"""
agents/analyze/nodes.py — Node functions for the Analyze module graph.

Each node:
  - Takes AgentState, returns a partial AgentState dict (LangGraph merges it).
  - Calls tools from tools/ only — no inline stats, SQL, or string formatting.
  - Never calls other nodes directly.

HITL gates use langgraph.types.interrupt() — never input() or st.text_input().
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any

import anthropic
import pandas as pd
from dotenv import load_dotenv
from langgraph.types import interrupt

from agents.analyze.prompts import (
    ANALYST_NOTES_BLOCK,
    HISTORY_INJECTION_PREFIX,
    NARRATIVE_PROMPT,
    SQL_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from agents.state import AgentState
from agents.tracer import flush, observe, trace_generation
from memory import retriever, semantic_cache
from memory.store import log_run
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

load_dotenv()

logger = logging.getLogger(__name__)

# ── Defaults (all env-backed so they can be overridden without code changes) ──

def _csv(env_key: str, fallback: str) -> list[str]:
    """Read a comma-separated env var, fall back to a default list."""
    return [v.strip() for v in os.getenv(env_key, fallback).split(",") if v.strip()]

_DEFAULT_METRIC       = os.getenv("DEFAULT_METRIC",     "dau_rate")
_DEFAULT_COVARIATE    = os.getenv("DEFAULT_COVARIATE",  "pre_session_count")
_DEFAULT_GUARDRAILS   = _csv("DEFAULT_GUARDRAILS",  "notif_optout,d7_retained,session_count")
_DEFAULT_SEGMENT_COLS = _csv("DEFAULT_SEGMENT_COLS", "platform,user_segment")
_DEFAULT_FUNNEL_STEPS = _csv("DEFAULT_FUNNEL_STEPS", "impression,click,install,d1_retain")
_SCHEMA_CACHE_PATH    = os.getenv("SCHEMA_CACHE_PATH", "memory/schema_cache.json")

# LLM token limits — named constants so they're visible and changeable
_MAX_TOKENS_SQL       = int(os.getenv("MAX_TOKENS_SQL",       "512"))
_MAX_TOKENS_NARRATIVE = int(os.getenv("MAX_TOKENS_NARRATIVE", "2048"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _anthropic_client() -> anthropic.Anthropic:
    """Return an Anthropic client. Reads ANTHROPIC_API_KEY from env."""
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _model() -> str:
    return os.getenv("MODEL", "claude-sonnet-4-20250514")


def _build_cached_messages(
    schema_context: str,
    history_text: str,
    task_prompt: str,
) -> list[dict]:
    """
    Construct the message array with prompt-cached static prefix.

    Static blocks (system, schema, history) get cache_control so they're
    cached across runs. The task prompt at the end is never cached.
    """
    static_blocks: list[dict] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if schema_context:
        static_blocks.append({
            "type": "text",
            "text": schema_context,
            "cache_control": {"type": "ephemeral"},
        })
    if history_text:
        static_blocks.append({
            "type": "text",
            "text": history_text,
            "cache_control": {"type": "ephemeral"},
        })
    # Dynamic task block — no cache_control
    static_blocks.append({"type": "text", "text": task_prompt})

    return [{"role": "user", "content": static_blocks}]


def _format_history(relevant_history: list[dict]) -> str:
    """Format past runs into the history injection prefix string."""
    if not relevant_history:
        return ""
    lines = []
    for r in relevant_history:
        override = r.get("analyst_override") or {}
        override_str = (
            f" (analyst overrode: {json.dumps(override)})" if override else ""
        )
        lines.append(
            f"- Task: \"{r['task']}\" | Metric: {r['metric']} | "
            f"Top segment: {r['top_segment']} | Eval: {r.get('eval_score', 'n/a')}"
            f"{override_str}"
        )
    history_text = "\n".join(lines)
    return HISTORY_INJECTION_PREFIX.format(history_text=history_text)


def _extract_sql(text: str) -> str:
    """Extract SQL from a ```sql ... ``` code block in LLM output."""
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: if no fences, return raw text stripped
    return text.strip()


def _db_conn(state: AgentState) -> DBConnection:
    backend = state.get("db_backend", "duckdb")
    path    = os.getenv("DUCKDB_PATH", "data/dau_experiment.db")
    return DBConnection(backend=backend, path=path)


def _safe_df(state: AgentState) -> pd.DataFrame | None:
    """Return query_result DataFrame from state, or None if missing/empty."""
    df = state.get("query_result")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


# ── Node 1: check_semantic_cache ──────────────────────────────────────────────

@observe(name="check_semantic_cache")
def check_semantic_cache(state: AgentState) -> dict:
    task = state.get("task", "")
    hit  = semantic_cache.check_cache(task, "generate_sql")
    if hit is None:
        return {}
    return {
        "semantic_cache_hit":        True,
        "semantic_cache_similarity": hit.get("similarity", 0.0),
        # Cached SQL is surfaced at query_gate for analyst approval
        "generated_sql":             hit["result"].get("sql", ""),
    }


# ── Node 2: inject_history ─────────────────────────────────────────────────

@observe(name="inject_history")
def inject_history(state: AgentState) -> dict:
    task    = state.get("task", "")
    history = retriever.retrieve_relevant_history(task)
    return {"relevant_history": history}


# ── Node 3: load_schema ───────────────────────────────────────────────────────

@observe(name="load_schema")
def load_schema(state: AgentState) -> dict:
    task = state.get("task", "")

    # Use cache unless task explicitly requests refresh
    refresh = "schema changed" in task.lower() or "refresh schema" in task.lower()

    if not refresh and os.path.exists(_SCHEMA_CACHE_PATH):
        try:
            with open(_SCHEMA_CACHE_PATH) as f:
                cached = json.load(f)
            return {"schema_context": cached["schema_context"]}
        except (KeyError, json.JSONDecodeError):
            pass  # fall through to re-fetch

    schema_context = _db_conn(state).inspect_schema()

    os.makedirs(os.path.dirname(_SCHEMA_CACHE_PATH), exist_ok=True)
    with open(_SCHEMA_CACHE_PATH, "w") as f:
        json.dump({"schema_context": schema_context}, f, indent=2)

    return {"schema_context": schema_context}


# ── Node 4: generate_sql ──────────────────────────────────────────────────────

@observe(name="generate_sql", as_type="generation")
def generate_sql(state: AgentState) -> dict:
    # If semantic cache already provided SQL, skip the API call
    if state.get("semantic_cache_hit") and state.get("generated_sql"):
        return {}

    task           = state.get("task", "")
    schema_context = state.get("schema_context", "")
    history_text   = _format_history(state.get("relevant_history", []))
    db_backend     = state.get("db_backend", "duckdb")

    task_prompt = SQL_GENERATION_PROMPT.format(
        task=task,
        schema_context=schema_context,
        db_backend=db_backend,
    )
    messages = _build_cached_messages(schema_context, history_text, task_prompt)

    with trace_generation("generate_sql", _model(), task_prompt) as gen:
        response = _anthropic_client().messages.create(
            model=_model(),
            max_tokens=_MAX_TOKENS_SQL,
            messages=messages,
        )
        cost_info = gen.update(response)

    sql = _extract_sql(response.content[0].text)

    return {
        "generated_sql":      sql,
        "cache_read_tokens":  cost_info.get("cache_read_tokens", 0),
        "cache_write_tokens": cost_info.get("cache_write_tokens", 0),
    }


# ── Node 5: query_gate (HITL interrupt 1) ────────────────────────────────────

@observe(name="query_gate")
def query_gate(state: AgentState) -> dict:
    payload = {
        "gate":            "query",
        "generated_sql":   state.get("generated_sql", ""),
        "cache_hit":       state.get("semantic_cache_hit", False),
        "message":         "Review the generated SQL. Approve, or provide a corrected query.",
    }
    analyst_response = interrupt(payload)

    # analyst_response expected: {"approved": bool, "sql": str | None}
    approved    = analyst_response.get("approved", True)
    edited_sql  = analyst_response.get("sql") or state.get("generated_sql", "")

    return {
        "query_approved": approved,
        "generated_sql":  edited_sql,
    }


# ── Node 6: execute_query ─────────────────────────────────────────────────────

@observe(name="execute_query")
def execute_query(state: AgentState) -> dict:
    sql = state.get("generated_sql", "")
    if not sql:
        raise ValueError("No SQL to execute — generate_sql must run first.")
    df = _db_conn(state).query(sql)
    return {"query_result": df}


# ── Node 7: decompose_metric ──────────────────────────────────────────────────

@observe(name="decompose_metric")
def decompose_metric(state: AgentState) -> dict:
    df = _safe_df(state)
    if df is None:
        logger.warning("decompose_metric: no query_result in state, skipping.")
        return {}

    # Only run decomposition if the required columns exist
    required = {"date", "new_users", "retained_users", "resurrected_users", "churned_users"}
    if not required.issubset(df.columns):
        logger.warning("decompose_metric: missing DAU component columns, skipping.")
        return {}

    result = decomposition_tools.decompose_dau(df)
    return {"decomposition_result": result}


# ── Node 8: detect_anomaly_node ───────────────────────────────────────────────

@observe(name="detect_anomaly")
def detect_anomaly_node(state: AgentState) -> dict:
    df     = _safe_df(state)
    metric = state.get("metric", _DEFAULT_METRIC)

    if df is None or "date" not in df.columns or metric not in df.columns:
        logger.warning("detect_anomaly: required columns missing, skipping.")
        return {}

    anomaly = anomaly_tools.detect_anomaly(df, metric_col=metric, date_col="date")

    dimension_cols = [c for c in _DEFAULT_SEGMENT_COLS if c in df.columns]
    if dimension_cols:
        slices = anomaly_tools.slice_and_dice(
            df, metric_col=metric, date_col="date", dimension_cols=dimension_cols
        )
    else:
        slices = {"ranked_dimensions": []}

    return {
        "anomaly_result": anomaly,
        "slice_result":   slices,
    }


# ── Node 9: forecast_baseline_node ────────────────────────────────────────────

@observe(name="forecast_baseline")
def forecast_baseline_node(state: AgentState) -> dict:
    df     = _safe_df(state)
    metric = state.get("metric", _DEFAULT_METRIC)

    if df is None or "date" not in df.columns or metric not in df.columns:
        logger.warning("forecast_baseline: required columns missing, skipping.")
        return {}

    result = forecast_tools.forecast_baseline(df, metric_col=metric, date_col="date")
    return {"forecast_result": result}


# ── Node 10: run_cuped_node ───────────────────────────────────────────────────

@observe(name="run_cuped")
def run_cuped_node(state: AgentState) -> dict:
    df        = _safe_df(state)
    metric    = state.get("metric", _DEFAULT_METRIC)
    covariate = state.get("covariate", _DEFAULT_COVARIATE)
    variant   = "variant"

    if df is None:
        return {}
    for col in [metric, covariate, variant]:
        if col not in df.columns:
            logger.warning("run_cuped: column '%s' missing, skipping.", col)
            return {}

    result = stats_tools.run_cuped(
        df, metric_col=metric, covariate_col=covariate, variant_col=variant
    )
    return {"cuped_result": result}


# ── Node 11: run_ttest_node ───────────────────────────────────────────────────

@observe(name="run_ttest")
def run_ttest_node(state: AgentState) -> dict:
    df        = _safe_df(state)
    metric    = state.get("metric", _DEFAULT_METRIC)
    variant   = "variant"
    cuped     = state.get("cuped_result", {})

    if df is None:
        return {}

    # Use CUPED-adjusted values if available
    adjusted_col = cuped.get("adjusted_col")
    use_col      = adjusted_col if (adjusted_col and adjusted_col in df.columns) else metric

    if variant not in df.columns or use_col not in df.columns:
        return {}

    ctrl = df[df[variant] == "control"][use_col].dropna()
    trt  = df[df[variant] == "treatment"][use_col].dropna()

    result = stats_tools.run_ttest(ctrl, trt)
    return {"ttest_result": result}


# ── Node 12: run_hte_node ─────────────────────────────────────────────────────

@observe(name="run_hte")
def run_hte_node(state: AgentState) -> dict:
    df      = _safe_df(state)
    metric  = state.get("metric", _DEFAULT_METRIC)
    variant = "variant"

    if df is None:
        return {}

    segment_cols = [c for c in _DEFAULT_SEGMENT_COLS if c in df.columns]
    if not segment_cols or metric not in df.columns or variant not in df.columns:
        return {}

    result = stats_tools.run_hte(
        df, metric_col=metric, variant_col=variant, segment_cols=segment_cols
    )
    return {"hte_result": result}


# ── Node 13: detect_novelty_node ──────────────────────────────────────────────

@observe(name="detect_novelty")
def detect_novelty_node(state: AgentState) -> dict:
    df     = _safe_df(state)
    metric = state.get("metric", _DEFAULT_METRIC)

    if df is None:
        return {}
    for col in [metric, "variant", "week"]:
        if col not in df.columns:
            logger.warning("detect_novelty: column '%s' missing, skipping.", col)
            return {}

    result = novelty_tools.detect_novelty_effect(
        df, metric_col=metric, variant_col="variant", week_col="week"
    )
    return {"novelty_result": result}


# ── Node 14: compute_mde_node ─────────────────────────────────────────────────

@observe(name="compute_mde")
def compute_mde_node(state: AgentState) -> dict:
    df     = _safe_df(state)
    metric = state.get("metric", _DEFAULT_METRIC)

    if df is None or "variant" not in df.columns or metric not in df.columns:
        return {}

    ctrl = df[df["variant"] == "control"][metric].dropna()
    trt  = df[df["variant"] == "treatment"][metric].dropna()

    if len(ctrl) < 2 or len(trt) < 2:
        return {}

    cuped_ate = state.get("cuped_result", {}).get("cuped_ate")

    result = mde_tools.compute_mde(
        n_control=len(ctrl),
        n_treatment=len(trt),
        baseline_mean=float(ctrl.mean()),
        baseline_std=float(ctrl.std()),
        observed_effect_abs=cuped_ate,
    )

    baseline_dau = int(os.getenv("BASELINE_DAU", "500000"))
    revenue_per_dau = float(os.getenv("REVENUE_PER_DAU", "0.50"))
    impact = mde_tools.business_impact_statement(
        mde_relative_pct=result["mde_relative_pct"],
        metric=metric,
        baseline_dau=baseline_dau,
        revenue_per_dau=revenue_per_dau,
    )

    return {
        "mde_result":      result,
        "business_impact": impact,
    }


# ── Node 15: check_guardrails_node ────────────────────────────────────────────

@observe(name="check_guardrails")
def check_guardrails_node(state: AgentState) -> dict:
    df = _safe_df(state)
    if df is None or "variant" not in df.columns:
        return {}

    present = [m for m in _DEFAULT_GUARDRAILS if m in df.columns]
    if not present:
        return {}

    result = guardrail_tools.check_guardrails(df, variant_col="variant", guardrail_metrics=present)
    return {"guardrail_result": result}


# ── Node 16: compute_funnel_node ──────────────────────────────────────────────

@observe(name="compute_funnel")
def compute_funnel_node(state: AgentState) -> dict:
    df = _safe_df(state)
    if df is None or "variant" not in df.columns:
        return {}

    required = {"user_id", "step", "completed"}
    if not required.issubset(df.columns):
        logger.warning("compute_funnel: funnel columns missing, skipping.")
        return {}

    present_steps = set(df["step"].dropna().unique())
    steps = [s for s in _DEFAULT_FUNNEL_STEPS if s in present_steps]
    if len(steps) < 2:
        return {}

    result = funnel_tools.compute_funnel(df, variant_col="variant", steps=steps)
    return {"funnel_result": result}


# ── Node 17: analysis_gate (HITL interrupt 2) ─────────────────────────────────

@observe(name="analysis_gate")
def analysis_gate(state: AgentState) -> dict:
    slice_dims    = state.get("slice_result", {}).get("ranked_dimensions", [])
    top_slice     = slice_dims[0] if slice_dims else {}
    breached      = [
        g for g in state.get("guardrail_result", {}).get("guardrails", [])
        if g.get("breached")
    ]

    payload = {
        "gate":                    "analysis",
        "decomposition":           state.get("decomposition_result"),
        "top_anomaly_slice":       top_slice,
        "forecast_outside_ci":     state.get("forecast_result", {}).get("outside_ci"),
        "cuped_variance_reduction": state.get("cuped_result", {}).get("variance_reduction_pct"),
        "significant":             state.get("ttest_result", {}).get("significant"),
        "top_segment":             state.get("hte_result", {}).get("top_segment"),
        "novelty_likely":          state.get("novelty_result", {}).get("novelty_likely"),
        "guardrails_breached":     state.get("guardrail_result", {}).get("any_breached"),
        "breached_metrics":        breached,
        "biggest_funnel_dropoff":  state.get("funnel_result", {}).get("biggest_dropoff_step"),
        "mde_powered":             state.get("mde_result", {}).get("is_powered_for_observed_effect"),
        "business_impact":         state.get("business_impact"),
        "message":                 "Review the analysis results. Approve or add notes/overrides.",
    }
    analyst_response = interrupt(payload)

    return {
        "analysis_approved": analyst_response.get("approved", True),
        "analyst_notes":     analyst_response.get("notes", ""),
    }


# ── Node 18: generate_narrative ───────────────────────────────────────────────

@observe(name="generate_narrative", as_type="generation")
def generate_narrative(state: AgentState) -> dict:
    metric = state.get("metric", _DEFAULT_METRIC)

    # Build template draft via narrative_tools (pure Python, no LLM)
    try:
        template_out = narrative_tools.format_narrative(
            metric=metric,
            decomposition_result=state.get("decomposition_result") or {},
            anomaly_result=state.get("anomaly_result") or {},
            cuped_result=state.get("cuped_result") or {},
            ttest_result=state.get("ttest_result") or {},
            hte_result=state.get("hte_result") or {},
            novelty_result=state.get("novelty_result") or {},
            mde_result=state.get("mde_result") or {},
            guardrail_result=state.get("guardrail_result") or {},
            funnel_result=state.get("funnel_result") or {},
            forecast_result=state.get("forecast_result") or {},
            business_impact=state.get("business_impact") or "",
            analyst_notes=state.get("analyst_notes") or "",
        )
    except Exception as exc:
        logger.warning("narrative_tools.format_narrative failed: %s", exc)
        template_out = {"narrative_draft": "", "recommendation": ""}

    draft_narrative = template_out["narrative_draft"]

    # Collect tool results for the LLM prompt (exclude DataFrames)
    tool_results = {
        k: v for k, v in state.items()
        if k.endswith("_result") and isinstance(v, dict)
    }
    tool_results_json = json.dumps(tool_results, default=str, indent=2)

    analyst_notes     = state.get("analyst_notes") or ""
    analyst_notes_section = (
        ANALYST_NOTES_BLOCK.format(analyst_notes=analyst_notes)
        if analyst_notes.strip() else ""
    )

    task_prompt = NARRATIVE_PROMPT.format(
        metric=metric,
        tool_results_json=tool_results_json,
        draft_narrative=draft_narrative,
        analyst_notes_section=analyst_notes_section,
    )

    schema_context = state.get("schema_context", "")
    history_text   = _format_history(state.get("relevant_history", []))

    # Multi-turn: prepend static blocks then conversation history
    messages = _build_cached_messages(schema_context, history_text, task_prompt)
    for turn in state.get("conversation_history", []):
        messages.append(turn)

    with trace_generation("generate_narrative", _model(), task_prompt, max_tokens=_MAX_TOKENS_NARRATIVE) as gen:
        response = _anthropic_client().messages.create(
            model=_model(),
            max_tokens=_MAX_TOKENS_NARRATIVE,
            messages=messages,
        )
        cost_info = gen.update(response)

    polished_narrative = response.content[0].text

    # Append this turn to conversation history for potential refinement
    new_history = list(state.get("conversation_history") or [])
    new_history.append({"role": "assistant", "content": polished_narrative})

    return {
        "narrative_draft":      polished_narrative,
        "recommendation":       template_out["recommendation"],
        "conversation_history": new_history,
        "cache_read_tokens":    (state.get("cache_read_tokens") or 0) + cost_info.get("cache_read_tokens", 0),
        "cache_write_tokens":   (state.get("cache_write_tokens") or 0) + cost_info.get("cache_write_tokens", 0),
    }


# ── Node 19: narrative_gate (HITL interrupt 3) ────────────────────────────────

@observe(name="narrative_gate")
def narrative_gate(state: AgentState) -> dict:
    payload = {
        "gate":             "narrative",
        "narrative_draft":  state.get("narrative_draft", ""),
        "recommendation":   state.get("recommendation", ""),
        "message":          "Review the narrative. Approve, or add notes to trigger a revision.",
    }
    analyst_response = interrupt(payload)

    approved      = analyst_response.get("approved", True)
    analyst_notes = analyst_response.get("notes", "")

    if approved:
        return {
            "narrative_approved": True,
            "final_narrative":    state.get("narrative_draft", ""),
            "analyst_notes":      analyst_notes,
        }

    # Analyst wants a revision — updated notes will cause graph to re-run generate_narrative
    return {
        "narrative_approved": False,
        "analyst_notes":      analyst_notes,
    }


# ── Node 20: log_run_node ─────────────────────────────────────────────────────

@observe(name="log_run")
def log_run_node(state: AgentState) -> dict:
    run_id = state.get("run_id") or str(uuid.uuid4())
    task   = state.get("task", "")

    # Persist to memory store
    log_run(
        task=task,
        run_id=run_id,
        metric=state.get("metric") or "",
        covariate=state.get("covariate") or "",
        db_backend=state.get("db_backend") or "duckdb",
        top_segment=state.get("hte_result", {}).get("top_segment") or "",
        eval_score=state.get("eval_score"),
        cache_read_tokens=state.get("cache_read_tokens") or 0,
        cache_write_tokens=state.get("cache_write_tokens") or 0,
        estimated_cost_usd=0.0,  # updated from token counts if needed
        semantic_cache_hits=1 if state.get("semantic_cache_hit") else 0,
        notes=state.get("analyst_notes") or "",
    )

    # Store SQL result in semantic cache for future runs
    if state.get("generated_sql"):
        semantic_cache.store_cache(
            task=task,
            node_name="generate_sql",
            result={"sql": state["generated_sql"]},
            run_id=run_id,
        )

    flush()

    return {"run_id": run_id}
