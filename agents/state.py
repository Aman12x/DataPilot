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
    NarrativeAuditResult,
    NoveltyResult,
    PowerAnalysisResult,
    RegressionResult,
    SliceResult,
    SrmResult,
    SufficientStats,
    TtestResult,
    TrustIndicators,
)


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    task: str                           # raw analyst/PM question
    analysis_mode: str                  # 'ab_test' | 'general' | 'power_analysis'
    query_type: str                     # 'lookup' | 'exploratory' — set by resolve_task_intent
    task_clarification: str             # analyst answer to the intent clarifying question (if any)
    relevant_history: list[dict]        # injected from memory store at run start
    db_backend: str                     # 'duckdb' | 'postgres' | 'mysql' | 'bigquery'
    duckdb_path: str                    # path to a user-uploaded DuckDB file (CSV/Excel upload)
    connection_id: str                  # saved DB connection — secrets resolved from vault at query time
    pg_host:     str                    # postgres/mysql host (inline ephemeral)
    pg_port:     int
    pg_dbname:   str
    pg_user:     str
    pg_password: str
    pg_sslmode:  str                    # prefer | require | disable | …
    bq_project_id: str                  # BigQuery project (inline ephemeral)
    bq_dataset: str
    bq_credentials_json: str            # wiped from checkpoint after schema load
    metric_config: MetricConfig         # single source of truth for all metric references
    metric_pack_id: str                 # saved metric pack id (if any)
    metric_pack_version: int            # pack version used for fingerprinting
    metric_pack_certified: bool         # True → skip Metric Config Gate
    metric_config_approved: bool        # HITL: metric mapping approved
    force_metric_gate: bool             # True when drift forces re-confirm of certified pack
    schema_drift_warnings: list[str]    # pack / snapshot vs live schema warnings
    dataset_fingerprint: str            # scopes semantic cache + few-shot isolation
    user_id: str                        # authenticated user — scopes memory store queries
    workspace_id: str                   # active workspace — shared history + resource scope

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
                                        # (per-variant preview frame in pushdown mode)
    sufficient_stats: SufficientStats   # in-warehouse moments; set when the extract
                                        # exceeded PUSHDOWN_ROWS and stats nodes
                                        # compute from moments instead of the frame
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
    srm_result: SrmResult               # sample ratio mismatch check
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
    audit_result:  NarrativeAuditResult # LLM audit result
    audit_blocked: bool                 # True if UNPATCHABLE critical findings exist
    audit_unpatched: list               # critical findings the in-place patcher could not fix

    # ── HITL gate 3: narrative sign-off ───────────────────────────────────────
    narrative_approved: bool
    final_narrative: str
    narrative_revision_count: int       # auto-correction attempts; capped at 3
    deck_data: dict                     # structured stakeholder deck (generated after approval)
    srm_acknowledged: bool              # analyst explicitly confirmed SRM at analysis_gate

    # ── Analyst overrides (accumulated across all 3 HITL gates) ──────────────
    analyst_override: dict              # keys: sql_edited, analysis_notes, narrative_notes, recommendation_override

    # ── General analysis (analysis_mode == 'general') ─────────────────────────
    describe_result:     DescribeResult
    correlation_result:  CorrelationResult
    regression_result:   RegressionResult

    # ── Visualisations ────────────────────────────────────────────────────────
    charts: list[dict]                  # list of ChartSpec dicts (serialised for SSE)
    trust_indicators: dict              # TrustIndicators dict (serialised for SSE)

    # ── Power analysis (analysis_mode == 'power_analysis') ───────────────────
    power_mde_target_pct:  float               # target MDE % from task (default 5.0)
    power_analysis_result: PowerAnalysisResult

    # ── Follow-up / conversation context ─────────────────────────────────────
    context_narrative: str              # narrative from parent run, injected for follow-up queries

    # ── Memory ────────────────────────────────────────────────────────────────
    run_id: str
    eval_score: float                   # 0-1, did the system surface the right answer?
