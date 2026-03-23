"""
agents/state.py — AgentState TypedDict: single source of truth for all graph nodes.

All data passed between nodes lives here. Nodes never call each other directly.
If a node needs something, it must be in state.
"""

from __future__ import annotations

import pandas as pd
from typing_extensions import TypedDict, NotRequired

from config.analysis_config import MetricConfig
from tools.schemas import (
    AnomalyResult,
    CorrelationResult,
    CupedResult,
    DecompositionResult,
    DescribeResult,
    ForecastResult,
    FunnelResult,
    GuardrailResult,
    HteResult,
    MdeResult,
    NoveltyResult,
    PowerAnalysisResult,
    SliceResult,
    TtestResult,
    TrustIndicators,
)


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    task: str                           # raw analyst/PM question
    analysis_mode: str                  # 'ab_test' | 'general' | 'power_analysis'
    task_clarification: str             # analyst answer to the intent clarifying question (if any)
    relevant_history: list[dict]        # injected from memory store at run start
    db_backend: str                     # 'duckdb' | 'postgres'
    duckdb_path: str                    # path to a user-uploaded DuckDB file (CSV/Excel upload)
    pg_host:     str                    # postgres credentials (only used when db_backend='postgres')
    pg_port:     int
    pg_dbname:   str
    pg_user:     str
    pg_password: str
    metric_config: MetricConfig         # single source of truth for all metric references
    user_id: str                        # authenticated user — scopes memory store queries

    # ── Caching metadata ──────────────────────────────────────────────────────
    semantic_cache_hit: bool            # True if this run was served from semantic cache
    semantic_cache_similarity: float    # similarity score of the cache hit
    semantic_cache_hit_type: str        # 'hard' (>0.92) or 'soft' (0.80-0.92)
    semantic_cache_accepted: bool       # True if analyst accepted a hard cache hit at the gate
    cache_read_tokens: int              # from Anthropic API response
    cache_write_tokens: int             # from Anthropic API response
    estimated_cost_usd: float           # accumulated cost across all API calls this run

    # ── Query phase ───────────────────────────────────────────────────────────
    schema_context: str                 # table names + columns from DB
    generated_sql: str                  # SQL produced by agent
    sql_validation_warnings: list[str]  # suspected hallucinated tables, surfaced at query gate
    query_result: pd.DataFrame          # raw result — user-level experiment data
    daily_df: pd.DataFrame              # metrics_daily time series — for decomp/anomaly/forecast
    funnel_df: pd.DataFrame             # funnel table — for compute_funnel

    # ── HITL gate 1: query confirmation ───────────────────────────────────────
    query_approved: bool

    # ── Pre-experiment context ─────────────────────────────────────────────────
    decomposition_result: DecompositionResult
    anomaly_result: AnomalyResult
    slice_result: SliceResult
    forecast_result: ForecastResult

    # ── Experiment analysis ────────────────────────────────────────────────────
    metric: str                         # e.g. 'dau', 'd7_retention'
    covariate: str                      # for CUPED pre-experiment covariate
    cuped_result: CupedResult
    ttest_result: TtestResult
    hte_result: HteResult
    novelty_result: NoveltyResult
    mde_result: MdeResult
    business_impact: str                # human-readable MDE → revenue statement

    # ── Guardrail phase ────────────────────────────────────────────────────────
    guardrail_result: GuardrailResult

    # ── Funnel phase ───────────────────────────────────────────────────────────
    funnel_result: FunnelResult

    # ── HITL gate 2: analysis validation ──────────────────────────────────────
    analysis_approved: bool
    analyst_notes: str                  # free-text override/annotation from analyst
    conversation_history: list[dict]    # for multi-turn narrative refinement

    # ── Narrative phase ────────────────────────────────────────────────────────
    narrative_draft: str                # PM-ready markdown writeup
    recommendation: str                 # one-sentence action recommendation

    # ── HITL gate 3: narrative sign-off ───────────────────────────────────────
    narrative_approved: bool
    final_narrative: str

    # ── Analyst overrides (accumulated across all 3 HITL gates) ──────────────
    analyst_override: dict              # keys: sql_edited, analysis_notes, narrative_notes, recommendation_override

    # ── General analysis (analysis_mode == 'general') ─────────────────────────
    describe_result:     DescribeResult
    correlation_result:  CorrelationResult

    # ── Visualisations ────────────────────────────────────────────────────────
    charts: list[dict]                  # list of ChartSpec dicts (serialised for SSE)
    trust_indicators: dict              # TrustIndicators dict (serialised for SSE)

    # ── Power analysis (analysis_mode == 'power_analysis') ───────────────────
    power_mde_target_pct:  float               # target MDE % from task (default 5.0)
    power_analysis_result: PowerAnalysisResult

    # ── Memory ────────────────────────────────────────────────────────────────
    run_id: str
    eval_score: float                   # 0-1, did the system surface the right answer?
