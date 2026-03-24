"""
agents/analyze/graph.py — LangGraph graph for the Analyze module.

Graph flow:
  START
    └─► check_semantic_cache
         ├─ hard hit (>0.92) ─► semantic_cache_gate ─► accepted? ─► log_run ─► END
         │                                              └─ declined ─► inject_history
         └─ miss / soft hit ─────────────────────────► inject_history
              └─► load_schema
                   └─► resolve_task_intent  (HITL 0 — conditional, only if ambiguous)
                        └─► infer_metric_config
                             └─► generate_sql
                        └─► query_gate  (HITL 1)
                             └─► execute_query
                                  └─► decompose_metric
                                       └─► detect_anomaly
                                            └─► forecast_baseline
                                                 └─► run_cuped
                                                      └─► run_ttest
                                                           └─► run_hte
                                                                └─► detect_novelty
                                                                     └─► compute_mde
                                                                          └─► check_guardrails
                                                                               └─► compute_funnel
                                                                                    └─► generate_charts
                                                                                         └─► analysis_gate  (HITL 2)
                                                                                         └─► generate_narrative
                                                                                              └─► narrative_gate  (HITL 3)
                                                                                                   ├─ approved ─► log_run ─► END
                                                                                                   └─ declined ─► generate_narrative (revision loop)

Rules:
  - All HITL interrupts use langgraph.types.interrupt() — never input() or polling.
  - The graph is compiled with MemorySaver so state survives across resume() calls.
  - graph and build_graph() are the public exports consumed by ui/app.py.
"""

from __future__ import annotations

import os
import pickle

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.analyze.nodes import (
    analysis_gate,
    check_guardrails_node,
    check_semantic_cache,
    compute_funnel_node,
    compute_mde_node,
    decompose_metric,
    describe_data_node,
    detect_anomaly_node,
    detect_novelty_node,
    execute_query,
    find_correlations_node,
    forecast_baseline_node,
    generate_charts_node,
    generate_narrative,
    generate_sql,
    infer_metric_config_node,
    inject_history,
    load_auxiliary_data,
    load_schema,
    log_run_node,
    narrative_gate,
    query_gate,
    resolve_task_intent,
    check_srm_node,
    run_cuped_node,
    run_hte_node,
    run_power_analysis_node,
    run_ttest_node,
    semantic_cache_gate,
)
from agents.state import AgentState

_HARD_HIT_THRESHOLD = float(
    os.getenv("SEMANTIC_CACHE_HARD_THRESHOLD", "0.92")
)


# ── Pickle serde for MemorySaver ──────────────────────────────────────────────
# LangGraph's default msgpack serde cannot serialize pd.DataFrame (stored in
# AgentState.query_result). We use pickle so all Python objects round-trip
# correctly through the in-process checkpointer.

class _PickleSerde:
    """Minimal serde that uses pickle for all values."""

    def dumps_typed(self, obj: object) -> tuple[str, bytes]:
        return ("pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def loads_typed(self, data: tuple[str, bytes]) -> object:
        type_tag, payload = data
        if type_tag == "pickle":
            return pickle.loads(payload)
        raise ValueError(f"Unknown serde type tag: {type_tag!r}")


def _default_checkpointer() -> MemorySaver:
    return MemorySaver(serde=_PickleSerde())


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_cache_check(state: AgentState) -> str:
    """
    After check_semantic_cache:
      - Hard hit (similarity >= hard threshold, default 0.92) → interrupt analyst
      - Soft hit / miss                                       → inject history and run fresh
    Only truly near-duplicate questions (≥0.92 similarity) interrupt the analyst.
    Soft hits no longer cause a gate — they were annoying friction for different
    questions that happened to share keywords.
    """
    similarity = state.get("semantic_cache_similarity", 0.0)
    if state.get("semantic_cache_hit") and similarity >= _HARD_HIT_THRESHOLD:
        return "semantic_cache_gate"
    return "inject_history"


def _route_after_query_gate(state: AgentState) -> str:
    """
    After query_gate:
      - Approved → execute the query
      - Declined → regenerate SQL (re-run generate_sql)
    """
    if state.get("query_approved", True):
        return "execute_query"
    return "generate_sql"


def _route_after_cache_gate(state: AgentState) -> str:
    """
    After semantic_cache_gate:
      - Analyst accepted cached result → skip to log_run
      - Analyst declined               → run full pipeline from inject_history
    """
    if state.get("semantic_cache_accepted"):
        return "log_run"
    return "inject_history"


def _route_after_infer_metric_config(state: AgentState) -> str:
    """
    After infer_metric_config:
      - power_analysis mode → skip SQL/query, go straight to run_power_analysis
      - everything else     → generate_sql (normal path)
    """
    if state.get("analysis_mode") == "power_analysis":
        return "run_power_analysis"
    return "generate_sql"


def _route_after_execute_query(state: AgentState) -> str:
    """Route to the general analysis path or the A/B test path."""
    if state.get("analysis_mode", "ab_test") == "general":
        return "describe_data"
    return "load_auxiliary_data"


def _route_after_narrative_gate(state: AgentState) -> str:
    """
    After narrative_gate:
      - Approved → log_run and finish
      - Declined → loop back to generate_narrative for a revision
    """
    if state.get("narrative_approved"):
        return "log_run"
    return "generate_narrative"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    """
    Compile and return the Analyze module graph.

    Args:
        checkpointer: LangGraph checkpointer for HITL persistence.
                      Defaults to MemorySaver (in-process, suitable for demo).
                      Pass a SqliteSaver or RedisSaver for persistent sessions.

    Returns:
        A compiled LangGraph CompiledGraph ready for .invoke() / .stream().

    Usage:
        graph = build_graph()
        config = {"configurable": {"thread_id": "run-001"}}

        # First call — runs until first interrupt
        result = graph.invoke({"task": "Why did DAU drop?"}, config)

        # Resume after analyst approves SQL at query_gate
        from langgraph.types import Command
        result = graph.invoke(Command(resume={"approved": True}), config)
    """
    if checkpointer is None:
        checkpointer = _default_checkpointer()

    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("check_semantic_cache", check_semantic_cache)
    builder.add_node("semantic_cache_gate",  semantic_cache_gate)
    builder.add_node("inject_history",        inject_history)
    builder.add_node("load_schema",           load_schema)
    builder.add_node("resolve_task_intent",   resolve_task_intent)
    builder.add_node("infer_metric_config",   infer_metric_config_node)
    builder.add_node("generate_sql",          generate_sql)
    builder.add_node("query_gate",           query_gate)
    builder.add_node("execute_query",        execute_query)
    builder.add_node("load_auxiliary_data", load_auxiliary_data)

    # Pre-experiment context
    builder.add_node("decompose_metric",   decompose_metric)
    builder.add_node("detect_anomaly",     detect_anomaly_node)
    builder.add_node("forecast_baseline",  forecast_baseline_node)

    # Experiment analysis
    builder.add_node("run_cuped",       run_cuped_node)
    builder.add_node("run_ttest",       run_ttest_node)
    builder.add_node("check_srm",       check_srm_node)
    builder.add_node("run_hte",         run_hte_node)
    builder.add_node("detect_novelty",  detect_novelty_node)
    builder.add_node("compute_mde",     compute_mde_node)

    # Secondary metric health
    builder.add_node("check_guardrails", check_guardrails_node)
    builder.add_node("compute_funnel",   compute_funnel_node)

    builder.add_node("generate_charts",    generate_charts_node)

    # HITL gates + narrative
    builder.add_node("analysis_gate",      analysis_gate)
    builder.add_node("generate_narrative", generate_narrative)
    builder.add_node("narrative_gate",     narrative_gate)
    builder.add_node("log_run",            log_run_node)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "check_semantic_cache")

    # Semantic cache routing
    builder.add_conditional_edges(
        "check_semantic_cache",
        _route_after_cache_check,
        {"semantic_cache_gate": "semantic_cache_gate", "inject_history": "inject_history"},
    )
    builder.add_conditional_edges(
        "semantic_cache_gate",
        _route_after_cache_gate,
        {"log_run": "log_run", "inject_history": "inject_history"},
    )

    # Main pipeline — linear through query execution
    builder.add_edge("inject_history",       "load_schema")
    builder.add_edge("load_schema",          "resolve_task_intent")
    builder.add_edge("resolve_task_intent",  "infer_metric_config")
    builder.add_conditional_edges(
        "infer_metric_config",
        _route_after_infer_metric_config,
        {"generate_sql": "generate_sql", "run_power_analysis": "run_power_analysis"},
    )
    builder.add_node("run_power_analysis", run_power_analysis_node)
    builder.add_edge("run_power_analysis", "generate_charts")
    builder.add_edge("generate_sql",    "query_gate")
    builder.add_conditional_edges(
        "query_gate",
        _route_after_query_gate,
        {"execute_query": "execute_query", "generate_sql": "generate_sql"},
    )

    # General-analysis nodes
    builder.add_node("describe_data",      describe_data_node)
    builder.add_node("find_correlations",  find_correlations_node)

    # Route after execute_query: general → describe_data, ab_test → load_auxiliary_data
    builder.add_conditional_edges(
        "execute_query",
        _route_after_execute_query,
        {"describe_data": "describe_data", "load_auxiliary_data": "load_auxiliary_data"},
    )

    # General analysis path: describe → correlations → (shared) generate_charts → analysis_gate
    builder.add_edge("describe_data",     "find_correlations")
    builder.add_edge("find_correlations", "generate_charts")

    # Pre-experiment context (sequential for initial build; can be parallelised later)
    builder.add_edge("load_auxiliary_data",  "decompose_metric")
    builder.add_edge("decompose_metric", "detect_anomaly")
    builder.add_edge("detect_anomaly",   "forecast_baseline")

    # Experiment analysis
    builder.add_edge("forecast_baseline", "run_cuped")
    builder.add_edge("run_cuped",         "run_ttest")
    builder.add_edge("run_ttest",         "check_srm")
    builder.add_edge("check_srm",         "run_hte")
    builder.add_edge("run_hte",           "detect_novelty")
    builder.add_edge("detect_novelty",    "compute_mde")

    # Secondary metrics
    builder.add_edge("compute_mde",      "check_guardrails")
    builder.add_edge("check_guardrails", "compute_funnel")

    # Charts → HITL gate 2 → narrative
    builder.add_edge("compute_funnel",    "generate_charts")
    builder.add_edge("generate_charts",   "analysis_gate")
    builder.add_edge("analysis_gate",  "generate_narrative")
    builder.add_edge("generate_narrative", "narrative_gate")

    # Narrative revision loop / finish
    builder.add_conditional_edges(
        "narrative_gate",
        _route_after_narrative_gate,
        {"log_run": "log_run", "generate_narrative": "generate_narrative"},
    )

    # Terminal
    builder.add_edge("log_run", END)

    return builder.compile(checkpointer=checkpointer)


# ── Module-level default instance ─────────────────────────────────────────────
# Used by backend/api/main.py: `from agents.analyze.graph import build_graph`
# Each API request creates its own thread_id config, so one compiled
# graph object is safe to share across requests.

graph = build_graph(_default_checkpointer())
