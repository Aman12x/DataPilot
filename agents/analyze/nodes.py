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
    SCHEMA_CONFIG_INFERENCE_PROMPT,
    SQL_CORRECTION_PROMPT,
    SQL_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from agents.state import AgentState
from config.analysis_config import MetricConfig, load_metric_config
from agents.tracer import flush, observe, trace_generation
from memory import retriever, semantic_cache
from memory.retriever import retrieve_sql_examples
from memory.store import log_run, update_eval_score
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
from tools.schemas import SliceResult

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

# Max LLM-based SQL correction retries in execute_query (0 = disabled)
_MAX_SQL_RETRIES = int(os.getenv("MAX_SQL_RETRIES", "2"))

# SQL keywords to exclude from table-name validation
_SQL_KEYWORDS = frozenset({
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL",
    "ON", "AND", "OR", "NOT", "IN", "LIKE", "AS", "IS", "NULL", "GROUP", "BY",
    "ORDER", "HAVING", "LIMIT", "OFFSET", "DISTINCT", "WITH", "UNION", "INTERSECT",
    "EXCEPT", "EXISTS", "CASE", "WHEN", "THEN", "ELSE", "END", "INTO", "VALUES",
    "TRUE", "FALSE", "AVG", "SUM", "COUNT", "MAX", "MIN", "COALESCE", "CAST",
    "FLOAT", "INT", "INTEGER", "VARCHAR", "TEXT", "DATE", "OVER", "PARTITION",
    "ROW", "ROWS", "UNBOUNDED", "PRECEDING", "FOLLOWING", "CURRENT",
})


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
    """Format past runs into instructional history for the LLM."""
    if not relevant_history:
        return ""
    lines = []
    for r in relevant_history:
        override = r.get("analyst_override") or {}
        score_str = f"{r['eval_score']:.2f}" if r.get("eval_score") is not None else "n/a"
        lines.append(
            f'- Task: "{r["task"]}" | Metric: {r["metric"]} | '
            f"Top segment: {r['top_segment']} | Score: {score_str}"
        )
        if override.get("sql_edited"):
            lines.append(
                "  → ANALYST CORRECTED SQL. Double-check JOINs and table references."
            )
        if override.get("analysis_notes"):
            lines.append(
                f'  → ANALYST NOTED: "{override["analysis_notes"]}" — '
                "apply unless current task clearly differs."
            )
        if override.get("narrative_notes"):
            lines.append(f'  → NARRATIVE FEEDBACK: "{override["narrative_notes"]}"')
        if override.get("recommendation_override"):
            lines.append(
                f'  → ANALYST OVERRODE RECOMMENDATION: '
                f'"{override["recommendation_override"]}" — use this framing for similar conclusions.'
            )
    history_text = "\n".join(lines)
    return HISTORY_INJECTION_PREFIX.format(history_text=history_text)


def _to_dict(v: Any) -> dict:
    """Convert a Pydantic model to dict, or return {} for None, as-is for plain dicts."""
    if v is None:
        return {}
    if hasattr(v, "model_dump"):
        return v.model_dump()
    return v if isinstance(v, dict) else {}


def _metric_context(mc: MetricConfig) -> str:
    """Format a MetricConfig into a human-readable block for prompt injection."""
    lines = [
        f"Primary metric:    {mc.primary_metric}",
        f"Direction:         {mc.metric_direction}",
        f"Covariate:         {mc.covariate}",
        f"Guardrail metrics: {', '.join(mc.guardrail_metrics)}",
        f"Segment columns:   {', '.join(mc.segment_cols)}",
        f"Funnel steps:      {', '.join(mc.funnel_steps)}",
    ]
    return "\n".join(lines)


def _canonical_experiment_sql(mc: "MetricConfig") -> str:
    """
    Return the known-good user-level experiment SQL for the current MetricConfig.

    Fully dynamic — uses mc fields for all table names, column names, and
    aggregation expressions. Used as automatic fallback when LLM SQL is invalid.
    """
    agg_map = {
        "mean":  f"AVG(e.{mc.metric_source_col})::FLOAT",
        "sum":   f"SUM(e.{mc.metric_source_col})::FLOAT",
        "count": f"COUNT(*)::FLOAT",
    }
    pre_agg_map = {
        "mean":  f"AVG(pre_events.{mc.covariate})::FLOAT",
        "sum":   f"SUM(pre_events.{mc.covariate})::FLOAT",
        "count": f"COUNT(*)::FLOAT",
    }
    metric_agg_expr   = agg_map.get(mc.metric_agg,   agg_map["mean"])
    covariate_agg_expr = pre_agg_map.get(mc.metric_agg, pre_agg_map["mean"])

    guardrail_selects = "\n".join(
        f"    AVG(e.{m})::FLOAT       AS {m},"
        for m in mc.guardrail_metrics
        if m not in (mc.primary_metric, mc.metric_source_col, mc.covariate)
    )
    segment_selects = "\n".join(f"    e.{c}," for c in mc.segment_cols)
    segment_group   = ", ".join(f"e.{c}" for c in mc.segment_cols)

    return f"""\
WITH pre_exp AS (
    SELECT {mc.user_id_col},
           {covariate_agg_expr} AS {mc.covariate}
    FROM   {mc.events_table} pre_events
    WHERE  {mc.date_col} < (SELECT MIN(assignment_date) FROM {mc.experiment_table})
    GROUP  BY {mc.user_id_col}
)
SELECT
    e.{mc.user_id_col},
    ex.{mc.variant_col}              AS variant,
    ex.{mc.week_col}                 AS week,
    {metric_agg_expr}                AS {mc.primary_metric},
    COALESCE(p.{mc.covariate}, 0)    AS {mc.covariate},
{guardrail_selects}
{segment_selects}
FROM       {mc.experiment_table} ex
JOIN       {mc.events_table} e
           ON  e.{mc.user_id_col} = ex.{mc.user_id_col}
           AND e.{mc.date_col} >= (SELECT MIN(assignment_date) FROM {mc.experiment_table})
LEFT JOIN  pre_exp p ON ex.{mc.user_id_col} = p.{mc.user_id_col}
GROUP BY   e.{mc.user_id_col}, ex.{mc.variant_col}, ex.{mc.week_col},
           {segment_group}, p.{mc.covariate}
LIMIT 50000"""


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


def _validate_sql_references(sql: str, schema_context: str) -> dict[str, list[str]]:
    """
    Validate table names AND dotted column references (alias.col) in the SQL.

    Returns:
        {
            "bad_tables":  list[str],   # FROM/JOIN targets not in schema
            "bad_columns": list[str],   # "alias.col (table: tbl)" where col not in tbl
        }

    CTE-aware: aliases defined in WITH ... AS (...) clauses are excluded from
    table lookups so CTE output columns don't produce false positives.

    Design choice: dotted-notation only for column checks (alias.col) rather
    than bare identifiers — reduces false positives from SQL keywords, function
    names, and aliases without a full SQL parser.
    """
    # ── Build schema: {table_lower: {col_lower, ...}} ─────────────────────────
    tables: dict[str, set[str]] = {}
    current: str | None = None
    for line in schema_context.splitlines():
        s = line.strip()
        if s.startswith("TABLE:"):
            current = s.split(":", 1)[1].strip().lower()
            tables[current] = set()
        elif current and s and not s.startswith("--") and not s.startswith("DIALECT"):
            col = s.split()[0].lower()
            if col:
                tables[current].add(col)

    if not tables:
        return {"bad_tables": [], "bad_columns": []}

    # ── Identify CTE names (skip these in alias resolution) ──────────────────
    cte_names: set[str] = set()
    for m in re.finditer(r'\bWITH\s+(\w+)\s+AS\b', sql, re.IGNORECASE):
        cte_names.add(m.group(1).lower())
    for m in re.finditer(r',\s*(\w+)\s+AS\s*\(', sql, re.IGNORECASE):
        cte_names.add(m.group(1).lower())

    # ── Build alias → real_table mapping ────────────────────────────────────
    alias_map: dict[str, str] = {}

    def _register(tbl: str, alias: str | None) -> None:
        t, a = tbl.lower(), (alias or tbl).lower()
        if t not in cte_names:
            alias_map[a] = t   # short alias (e, ex, p) → real table
            alias_map[t] = t   # table name itself

    for m in re.finditer(r'\bFROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql, re.IGNORECASE):
        _register(m.group(1), m.group(2))
    for m in re.finditer(r'\bJOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql, re.IGNORECASE):
        _register(m.group(1), m.group(2))

    # ── Table validation ─────────────────────────────────────────────────────
    bad_tables: list[str] = []
    seen: set[str] = set()
    for alias, tbl in alias_map.items():
        if alias != tbl:
            continue                          # skip short aliases — report real name only
        if tbl in seen or tbl in cte_names:
            continue
        seen.add(tbl)
        if tbl not in tables and tbl.upper() not in _SQL_KEYWORDS and len(tbl) > 2:
            bad_tables.append(tbl)

    # ── Column validation (dotted notation only) ──────────────────────────────
    bad_columns: list[str] = []
    seen_cols: set[str] = set()
    for m in re.finditer(r'\b(\w+)\.(\w+)\b', sql):
        alias, col = m.group(1).lower(), m.group(2).lower()
        key = f"{alias}.{col}"
        if key in seen_cols:
            continue
        seen_cols.add(key)
        if alias in cte_names:
            continue                          # CTE output column — can't validate
        if alias not in alias_map:
            continue                          # unknown alias (subquery etc.) — skip
        tbl = alias_map[alias]
        if tbl not in tables:
            continue                          # table itself is bad_tables — already reported
        if col.upper() in _SQL_KEYWORDS or len(col) <= 1:
            continue                          # SQL keywords / single-char names
        if col not in tables[tbl]:
            bad_columns.append(f"{alias}.{col} (table: {tbl})")

    return {"bad_tables": bad_tables, "bad_columns": bad_columns}


def _build_few_shot_block(examples: list[dict]) -> str:
    """
    Format a list of {task, sql} dicts as a few-shot in-context block for
    injection into SQL_GENERATION_PROMPT.  Returns empty string when no examples.
    """
    if not examples:
        return "(No verified past queries available yet.)"
    lines = []
    for ex in examples:
        lines.append(f"Q: {ex['task'].strip()}")
        lines.append(f"```sql\n{ex['sql'].strip()}\n```")
        lines.append("")
    return "\n".join(lines).rstrip()


def _known_schema_names(schema_context: str) -> tuple[set[str], set[str]]:
    """
    Parse schema_context and return (known_tables, known_columns) — both lowercased.
    Used for MetricConfig column validation after LLM inference.
    """
    known_tables:  set[str] = set()
    known_columns: set[str] = set()
    current: str | None = None
    for line in schema_context.splitlines():
        s = line.strip()
        if s.startswith("TABLE:"):
            current = s.split(":", 1)[1].strip().lower()
            known_tables.add(current)
        elif current and s and not s.startswith("--") and not s.startswith("DIALECT"):
            col = s.split()[0].lower()
            if col:
                known_columns.add(col)
    return known_tables, known_columns


def _sanitise_metric_config(
    mc: MetricConfig,
    schema_context: str,
    defaults: MetricConfig,
) -> tuple[MetricConfig, list[str]]:
    """
    Cross-check MetricConfig column/table fields against the live schema.

    For any field whose value doesn't appear in the schema, log a warning and
    substitute the corresponding value from `defaults`.  Table name fields are
    checked against known_tables; column name fields against known_columns.

    Returns:
        (sanitised MetricConfig, list of warning strings)

    This prevents hallucinated column names from infer_metric_config_node
    from propagating to _canonical_experiment_sql() and generating broken SQL.
    """
    if not schema_context.strip():
        return mc, []  # no schema to validate against

    known_tables, known_columns = _known_schema_names(schema_context)
    if not known_tables and not known_columns:
        return mc, []

    warnings: list[str] = []
    overrides: dict = {}

    def _check_col(field: str, value: str, fallback: str) -> None:
        if value.lower() not in known_columns:
            warnings.append(
                f"{field}={value!r} not found in schema — using default {fallback!r}"
            )
            overrides[field] = fallback

    def _check_table(field: str, value: str | None, fallback: str | None) -> None:
        if value and value.lower() not in known_tables:
            warnings.append(
                f"{field}={value!r} not found in schema — using default {fallback!r}"
            )
            overrides[field] = fallback

    # Column name fields
    _check_col("metric_source_col", mc.metric_source_col, defaults.metric_source_col)
    _check_col("covariate",         mc.covariate,         defaults.covariate)
    _check_col("user_id_col",       mc.user_id_col,       defaults.user_id_col)
    _check_col("date_col",          mc.date_col,          defaults.date_col)
    _check_col("variant_col",       mc.variant_col,       defaults.variant_col)
    _check_col("week_col",          mc.week_col,          defaults.week_col)

    # Table name fields (optional tables may be None)
    _check_table("events_table",      mc.events_table,      defaults.events_table)
    _check_table("experiment_table",  mc.experiment_table,  defaults.experiment_table)
    if mc.timeseries_table:
        _check_table("timeseries_table", mc.timeseries_table, defaults.timeseries_table)
    if mc.funnel_table:
        _check_table("funnel_table", mc.funnel_table, defaults.funnel_table)

    # List fields — drop any element not in schema columns
    clean_guardrails = [m for m in mc.guardrail_metrics if m.lower() in known_columns]
    if len(clean_guardrails) < len(mc.guardrail_metrics):
        dropped = set(mc.guardrail_metrics) - set(clean_guardrails)
        warnings.append(
            f"guardrail_metrics: removed {sorted(dropped)} (not in schema)"
        )
        overrides["guardrail_metrics"] = clean_guardrails or defaults.guardrail_metrics

    clean_segments = [c for c in mc.segment_cols if c.lower() in known_columns]
    if len(clean_segments) < len(mc.segment_cols):
        dropped = set(mc.segment_cols) - set(clean_segments)
        warnings.append(
            f"segment_cols: removed {sorted(dropped)} (not in schema)"
        )
        overrides["segment_cols"] = clean_segments or defaults.segment_cols

    if not overrides:
        return mc, []

    sanitised = mc.model_copy(update=overrides)
    return sanitised, warnings


def _llm_correct_sql(
    sql: str,
    error: str,
    schema_context: str,
    task: str,
) -> str:
    """
    Ask the LLM to fix a SQL execution error.  Returns the corrected SQL string,
    or the original `sql` if the correction call itself fails.
    """
    prompt = SQL_CORRECTION_PROMPT.format(
        sql=sql,
        error=error,
        schema_context=schema_context,
        task=task,
    )
    try:
        response = _anthropic_client().messages.create(
            model=_model(),
            max_tokens=_MAX_TOKENS_SQL,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_sql(response.content[0].text)
    except Exception as exc:
        logger.warning("_llm_correct_sql: correction call failed: %s", exc)
        return sql


# ── Node 1: check_semantic_cache ──────────────────────────────────────────────

@observe(name="check_semantic_cache")
def check_semantic_cache(state: AgentState) -> dict:
    task = state.get("task", "")
    hit  = semantic_cache.check_cache(task, "generate_sql")
    if hit is None:
        return {}
    cached    = hit["result"]
    narrative = cached.get("narrative", "")
    hit_type  = hit.get("hit_type", "hard")   # "hard" (>0.92) or "soft" (0.80-0.92)
    return {
        "semantic_cache_hit":        True,
        "semantic_cache_similarity": hit.get("similarity", 0.0),
        "semantic_cache_accepted":   False,     # analyst hasn't decided yet
        "generated_sql":             cached.get("sql", ""),
        # Restore narrative so the cache path can show the full result on acceptance.
        "narrative_draft":           narrative,
        "recommendation":            cached.get("recommendation", ""),
        "final_narrative":           narrative,
        # hit_type drives the gate label so analyst knows whether this is mandatory review
        "semantic_cache_hit_type":   hit_type,
    }


# ── Node 1b: semantic_cache_gate (HITL interrupt — hard cache hit only) ──────

@observe(name="semantic_cache_gate")
def semantic_cache_gate(state: AgentState) -> dict:
    """
    Interrupt when the semantic cache returns a hard hit (similarity > 0.92).
    Asks the analyst: "Use cached result, or re-run analysis?"
    If accepted: the graph routes directly to log_run, skipping all computation.
    If declined: the graph continues normally from inject_history.
    """
    hit_type   = state.get("semantic_cache_hit_type", "hard")
    similarity = state.get("semantic_cache_similarity", 0.0)
    hit_label  = "identical" if hit_type == "hard" else "very similar"
    payload = {
        "gate":             "semantic_cache",
        "hit_type":         hit_type,
        "similarity":       similarity,
        "generated_sql":    state.get("generated_sql", ""),
        "narrative_draft":  state.get("narrative_draft", ""),
        "recommendation":   state.get("recommendation", ""),
        "message": (
            f"This task looks {hit_label} to a prior analysis "
            f"(similarity={similarity:.2f}). "
            "Use the cached result, or re-run the full analysis?"
        ),
    }
    analyst_response = interrupt(payload)
    accepted = analyst_response.get("approved", False)
    return {"semantic_cache_accepted": accepted}


# ── Node 2: inject_history ─────────────────────────────────────────────────

@observe(name="inject_history")
def inject_history(state: AgentState) -> dict:
    task    = state.get("task", "")
    user_id = state.get("user_id")
    history = retriever.retrieve_relevant_history(task, user_id=user_id)
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
            schema_context = cached["schema_context"]
        except (KeyError, json.JSONDecodeError):
            schema_context = None  # fall through to re-fetch
    else:
        schema_context = None

    if schema_context is None:
        schema_context = _db_conn(state).inspect_schema()
        os.makedirs(os.path.dirname(_SCHEMA_CACHE_PATH), exist_ok=True)
        with open(_SCHEMA_CACHE_PATH, "w") as f:
            json.dump({"schema_context": schema_context}, f, indent=2)

    # Prepend SQL dialect so the LLM never has to guess the engine.
    # This is the single most effective guard against dialect-specific syntax errors.
    backend = state.get("db_backend", "duckdb")
    dialect = "DuckDB SQL" if backend == "duckdb" else "PostgreSQL"
    schema_context = f"-- Dialect: {dialect}\n\n{schema_context}"

    # Load MetricConfig once here — all subsequent nodes read from state
    mc = state.get("metric_config") or load_metric_config()

    return {
        "schema_context": schema_context,
        "metric_config":  mc,
        "metric":         mc.primary_metric,
        "covariate":      mc.covariate,
    }


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
    mc             = state.get("metric_config") or load_metric_config()

    # ── Few-shot retrieval ────────────────────────────────────────────────────
    # Retrieve up to 2 verified question-SQL pairs from the memory store and
    # inject them as in-context examples.  Empty on first run; grows over time.
    sql_examples    = retrieve_sql_examples(task)
    few_shot_block  = _build_few_shot_block(sql_examples)

    task_prompt = SQL_GENERATION_PROMPT.format(
        task=task,
        schema_context=schema_context,
        db_backend=db_backend,
        metric_context=_metric_context(mc),
        primary_metric=mc.primary_metric,
        metric_source_col=mc.metric_source_col,
        metric_agg=mc.metric_agg,
        covariate=mc.covariate,
        variant_col=mc.variant_col,
        week_col=mc.week_col,
        guardrail_metrics_csv=", ".join(mc.guardrail_metrics),
        segment_cols_csv=", ".join(mc.segment_cols),
        sql_template=_canonical_experiment_sql(mc),
        few_shot_block=few_shot_block,
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

    # ── Schema validation + one auto-correction pass ──────────────────────────
    # Check both table names and dotted column references against schema_context.
    # If issues are found, send them back to the LLM for a targeted correction
    # before showing anything to the analyst at the HITL gate.
    # This catches hallucinations early — before execution — and avoids wasting
    # the analyst's attention on obviously broken SQL.
    validation  = _validate_sql_references(sql, schema_context)
    all_issues  = validation["bad_tables"] + validation["bad_columns"]

    if all_issues:
        logger.warning("generate_sql: schema issues detected %s — auto-correcting.", all_issues)
        issue_lines = "\n".join(f"  - {v}" for v in all_issues)
        correction_hint = (
            f"The following names in the SQL don't exist in the schema:\n"
            f"{issue_lines}\n\n"
            "For each invalid name, replace it with the correct name from the schema. "
            "Do not invent names — use only what's listed in the schema."
        )
        corrected = _llm_correct_sql(sql, correction_hint, schema_context, task)
        if corrected.strip() and corrected.strip() != sql.strip():
            sql = corrected
            # Re-validate once so remaining issues are surfaced at the HITL gate
            validation = _validate_sql_references(sql, schema_context)
            all_issues = validation["bad_tables"] + validation["bad_columns"]
            if all_issues:
                logger.warning(
                    "generate_sql: issues remain after correction: %s", all_issues
                )

    result = {
        "generated_sql":      sql,
        "cache_read_tokens":  cost_info.get("cache_read_tokens", 0),
        "cache_write_tokens": cost_info.get("cache_write_tokens", 0),
        "estimated_cost_usd": cost_info.get("estimated_cost_usd", 0.0),
    }
    if all_issues:
        result["sql_validation_warnings"] = all_issues
    return result


# ── Node 5: query_gate (HITL interrupt 1) ────────────────────────────────────

@observe(name="query_gate")
def query_gate(state: AgentState) -> dict:
    payload = {
        "gate":                     "query",
        "generated_sql":            state.get("generated_sql", ""),
        "cache_hit":                state.get("semantic_cache_hit", False),
        "sql_validation_warnings":  state.get("sql_validation_warnings", []),
        "message":                  "Review the generated SQL. Approve, or provide a corrected query.",
    }
    analyst_response = interrupt(payload)

    # analyst_response expected: {"approved": bool, "sql": str | None}
    approved    = analyst_response.get("approved", True)
    edited_sql  = analyst_response.get("sql") or state.get("generated_sql", "")

    override: dict = {}
    if edited_sql.strip() != state.get("generated_sql", "").strip():
        override["sql_edited"] = True

    return {
        "query_approved":   approved,
        "generated_sql":    edited_sql,
        "analyst_override": override,
    }


# ── Node 6: execute_query ─────────────────────────────────────────────────────

@observe(name="execute_query")
def execute_query(state: AgentState) -> dict:
    sql = state.get("generated_sql", "")
    if not sql:
        raise ValueError("No SQL to execute — generate_sql must run first.")

    mc             = state.get("metric_config") or load_metric_config()
    schema_context = state.get("schema_context", "")
    task           = state.get("task", "")

    # ── Phase 1: LLM SQL with error-correction retries ────────────────────────
    # On each execution failure, send (sql, error, schema) to the LLM for a
    # targeted correction and retry.  Up to _MAX_SQL_RETRIES corrections.
    # This implements the "execution feedback loop" pattern that is now standard
    # in production text-to-SQL systems (AWS, CHESS, ReFoRCE, etc.).
    current_sql = sql
    df: pd.DataFrame | None = None

    for attempt in range(_MAX_SQL_RETRIES + 1):
        try:
            df = _db_conn(state).query(current_sql)
            if attempt > 0:
                logger.info("execute_query: SQL succeeded after %d LLM correction(s).", attempt)
            break
        except Exception as exc:
            if attempt < _MAX_SQL_RETRIES:
                logger.warning(
                    "execute_query: attempt %d failed (%s) — requesting LLM correction.",
                    attempt + 1, exc,
                )
                corrected = _llm_correct_sql(current_sql, str(exc), schema_context, task)
                if corrected.strip() and corrected.strip() != current_sql.strip():
                    current_sql = corrected
                else:
                    logger.warning(
                        "execute_query: LLM returned unchanged SQL on attempt %d — "
                        "stopping retries early.",
                        attempt + 1,
                    )
                    break
            else:
                logger.warning(
                    "execute_query: all %d LLM retries exhausted — falling back to canonical SQL.",
                    _MAX_SQL_RETRIES,
                )

    # ── Phase 2: Column validation + canonical SQL fallback ───────────────────
    if df is not None:
        required = {mc.primary_metric, mc.covariate, "variant"}
        missing  = required - set(df.columns)
        if missing:
            logger.warning(
                "execute_query: result missing columns %s — trying canonical SQL.",
                sorted(missing),
            )
            df = None   # trigger canonical fallback below

    if df is None:
        canonical_sql = _canonical_experiment_sql(mc)
        try:
            df = _db_conn(state).query(canonical_sql)
            current_sql = canonical_sql
            logger.info("execute_query: canonical SQL succeeded (%d rows).", len(df))
        except Exception as exc:
            raise ValueError(
                f"execute_query: LLM SQL and canonical SQL both failed. Last error: {exc}"
            ) from exc

    result: dict = {"query_result": df}
    if current_sql != sql:
        result["generated_sql"] = current_sql
    return result


# ── Node 6b: load_auxiliary_data ─────────────────────────────────────────────
# Loads timeseries and funnel tables using MetricConfig — no LLM involved.
# decompose_metric / detect_anomaly / forecast_baseline need daily_df.
# compute_funnel needs funnel_df.
# These are separate from query_result (the user-level experiment DataFrame).

def _aggregate_daily_from_events(conn, mc: MetricConfig) -> pd.DataFrame:
    """Fallback: aggregate daily metric from raw events when no timeseries_table."""
    agg_map = {
        "mean":  f"AVG({mc.metric_source_col})",
        "sum":   f"SUM({mc.metric_source_col})",
        "count": f"COUNT(*)",
    }
    agg_expr = agg_map.get(mc.metric_agg, agg_map["mean"])
    segment_cols_sql = ", ".join(mc.segment_cols) if mc.segment_cols else ""
    group_by_cols    = f"{mc.date_col}" + (f", {segment_cols_sql}" if segment_cols_sql else "")
    select_extra     = (f", {segment_cols_sql}" if segment_cols_sql else "")
    return conn.query(f"""
        SELECT {mc.date_col} AS date{select_extra},
               {agg_expr} AS {mc.primary_metric}
        FROM {mc.events_table}
        GROUP BY {group_by_cols}
        ORDER BY {mc.date_col}
    """)


@observe(name="load_auxiliary_data")
def load_auxiliary_data(state: AgentState) -> dict:
    mc   = state.get("metric_config") or load_metric_config()
    conn = _db_conn(state)
    result: dict = {}

    # ── Timeseries: try pre-aggregated table first, fall back to event aggregation ──
    if mc.timeseries_table:
        try:
            daily = conn.query(f"SELECT * FROM {mc.timeseries_table} ORDER BY {mc.date_col}")
            # Aggregate DAU component columns to platform level for cleaner time series
            agg_cols = [c for c in ["dau", "new_users", "retained_users", "resurrected_users", "churned_users"]
                        if c in daily.columns]
            if agg_cols and "platform" in daily.columns and mc.date_col in daily.columns:
                daily = (
                    daily
                    .groupby([mc.date_col, "platform"])[agg_cols]
                    .sum()
                    .reset_index()
                )
            result["daily_df"] = daily
        except Exception as exc:
            logger.warning(
                "load_auxiliary_data: %s query failed (%s) — falling back to event aggregation.",
                mc.timeseries_table, exc,
            )
            try:
                result["daily_df"] = _aggregate_daily_from_events(conn, mc)
            except Exception as exc2:
                logger.warning("load_auxiliary_data: event aggregation also failed: %s", exc2)
    else:
        try:
            result["daily_df"] = _aggregate_daily_from_events(conn, mc)
        except Exception as exc:
            logger.warning("load_auxiliary_data: event aggregation failed: %s", exc)

    # ── Funnel: optional ──────────────────────────────────────────────────────
    if mc.funnel_table:
        funnel_sql = f"""\
SELECT f.{mc.user_id_col}, ex.{mc.variant_col} AS variant, f.step, f.completed
FROM   {mc.funnel_table} f
JOIN   {mc.experiment_table} ex
       ON f.{mc.user_id_col} = ex.{mc.user_id_col}
      AND ex.{mc.week_col} = 1
"""
        try:
            result["funnel_df"] = conn.query(funnel_sql)
        except Exception as exc:
            logger.warning("load_auxiliary_data: funnel query failed: %s", exc)

    return result


# ── Node 7: decompose_metric ──────────────────────────────────────────────────

@observe(name="decompose_metric")
def decompose_metric(state: AgentState) -> dict:
    df = state.get("daily_df")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.warning("decompose_metric: no daily_df in state, skipping.")
        return {}

    mc = state.get("metric_config") or load_metric_config()

    # Use DAU-specific decomposition only when all DAU component columns are present
    dau_cols = {"new_users", "retained_users", "resurrected_users", "churned_users"}
    if dau_cols.issubset(df.columns):
        try:
            result = decomposition_tools.decompose_dau(df, date_col=mc.date_col)
            return {"decomposition_result": result}
        except Exception as exc:
            logger.warning("decompose_metric: decompose_dau failed (%s), trying generic.", exc)

    # Generic path: segment-based breakdown for any metric
    metric_col = mc.primary_metric
    segment_cols = [c for c in mc.segment_cols if c in df.columns]
    if metric_col not in df.columns or not segment_cols:
        logger.warning(
            "decompose_metric: metric_col '%s' or segment_cols %s not in daily_df, skipping.",
            metric_col, mc.segment_cols,
        )
        return {}

    try:
        result = decomposition_tools.decompose_metric(
            df, metric_col=metric_col, segment_cols=segment_cols, date_col=mc.date_col
        )
        return {"decomposition_result": result}
    except Exception as exc:
        logger.warning("decompose_metric: decompose_metric failed: %s", exc)
        return {}


# ── Node 8: detect_anomaly_node ───────────────────────────────────────────────

@observe(name="detect_anomaly")
def detect_anomaly_node(state: AgentState) -> dict:
    df     = state.get("daily_df")
    mc     = state.get("metric_config") or load_metric_config()
    # For time-series anomaly, use "dau" from metrics_daily (not the per-user primary metric)
    metric = "dau" if df is not None and "dau" in (df.columns if hasattr(df, "columns") else []) else (state.get("metric") or mc.primary_metric)

    if df is None or (isinstance(df, pd.DataFrame) and df.empty) or "date" not in df.columns or metric not in df.columns:
        logger.warning("detect_anomaly: required columns missing in daily_df, skipping.")
        return {}

    anomaly = anomaly_tools.detect_anomaly(df, metric_col=metric, date_col="date")

    dimension_cols = [c for c in mc.segment_cols if c in df.columns]
    if dimension_cols:
        # Use the first anomaly date as the before/after split so slice_and_dice
        # compares pre-experiment baseline against the experiment window rather
        # than splitting at the calendar midpoint.
        anomaly_dates = anomaly.anomaly_dates if anomaly else []
        experiment_start = anomaly_dates[0] if anomaly_dates else None
        slices = anomaly_tools.slice_and_dice(
            df, metric_col=metric, date_col="date",
            dimension_cols=dimension_cols, experiment_start=experiment_start,
        )
    else:
        slices = SliceResult(ranked_dimensions=[])

    return {
        "anomaly_result": anomaly,
        "slice_result":   slices,
    }


# ── Node 9: forecast_baseline_node ────────────────────────────────────────────

@observe(name="forecast_baseline")
def forecast_baseline_node(state: AgentState) -> dict:
    df     = state.get("daily_df")
    mc     = state.get("metric_config") or load_metric_config()
    # Use "dau" from metrics_daily for the forecast baseline
    metric = "dau" if df is not None and "dau" in (df.columns if hasattr(df, "columns") else []) else (state.get("metric") or mc.primary_metric)

    if df is None or (isinstance(df, pd.DataFrame) and df.empty) or "date" not in df.columns or metric not in df.columns:
        logger.warning("forecast_baseline: required columns missing in daily_df, skipping.")
        return {}

    try:
        result = forecast_tools.forecast_baseline(df, metric_col=metric, date_col="date")
    except Exception as exc:
        logger.warning("forecast_baseline: failed (%s), skipping.", exc)
        return {}
    return {"forecast_result": result}


# ── Node 10: run_cuped_node ───────────────────────────────────────────────────

@observe(name="run_cuped")
def run_cuped_node(state: AgentState) -> dict:
    df        = _safe_df(state)
    mc        = state.get("metric_config") or load_metric_config()
    metric    = state.get("metric") or mc.primary_metric
    covariate = state.get("covariate") or mc.covariate
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
    mc        = state.get("metric_config") or load_metric_config()
    metric    = state.get("metric") or mc.primary_metric
    variant   = "variant"

    if df is None:
        return {}

    # CUPED adjusts the ATE internally; run_ttest always operates on the original metric col
    use_col = metric

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
    mc      = state.get("metric_config") or load_metric_config()
    metric  = state.get("metric") or mc.primary_metric
    variant = "variant"

    if df is None:
        return {}

    segment_cols = [c for c in mc.segment_cols if c in df.columns]
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
    mc     = state.get("metric_config") or load_metric_config()
    metric = state.get("metric") or mc.primary_metric

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
    mc     = state.get("metric_config") or load_metric_config()
    metric = state.get("metric") or mc.primary_metric

    if df is None or "variant" not in df.columns or metric not in df.columns:
        return {}

    ctrl = df[df["variant"] == "control"][metric].dropna()
    trt  = df[df["variant"] == "treatment"][metric].dropna()

    if len(ctrl) < 2 or len(trt) < 2:
        return {}

    cuped = state.get("cuped_result")
    cuped_ate = cuped.cuped_ate if cuped else None

    result = mde_tools.compute_mde(
        n_control=len(ctrl),
        n_treatment=len(trt),
        baseline_mean=float(ctrl.mean()),
        baseline_std=float(ctrl.std()),
        observed_effect_abs=cuped_ate,
    )

    # Use baseline_unit_count from MetricConfig (env BASELINE_DAU as a final fallback)
    baseline_units = mc.baseline_unit_count or int(os.getenv("BASELINE_DAU", "500000"))
    impact = mde_tools.business_impact_statement(
        mde_relative_pct=result.mde_relative_pct,
        metric=metric,
        baseline_dau=baseline_units,
        revenue_per_dau=mc.revenue_per_unit,
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

    mc = state.get("metric_config") or load_metric_config()
    present = [m for m in mc.guardrail_metrics if m in df.columns]
    if not present:
        return {}

    # When primary metric is higher_is_better, unknown guardrail drops are harmful
    default_direction = (
        "decrease" if mc.metric_direction == "higher_is_better" else "increase"
    )

    result = guardrail_tools.check_guardrails(
        df,
        variant_col="variant",
        guardrail_metrics=present,
        harm_directions=mc.guardrail_harm_directions,
        default_direction=default_direction,
    )
    return {"guardrail_result": result}


# ── Node 16: compute_funnel_node ──────────────────────────────────────────────

@observe(name="compute_funnel")
def compute_funnel_node(state: AgentState) -> dict:
    df = state.get("funnel_df")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.warning("compute_funnel: no funnel_df in state, skipping.")
        return {}

    required = {"user_id", "step", "completed", "variant"}
    if not required.issubset(df.columns):
        logger.warning("compute_funnel: funnel columns missing in funnel_df, skipping.")
        return {}

    mc            = state.get("metric_config") or load_metric_config()
    present_steps = set(df["step"].dropna().unique())
    steps = [s for s in mc.funnel_steps if s in present_steps]
    if len(steps) < 2:
        return {}

    result = funnel_tools.compute_funnel(df, variant_col="variant", steps=steps)
    return {"funnel_result": result}


# ── Node 17: analysis_gate (HITL interrupt 2) ─────────────────────────────────

@observe(name="analysis_gate")
def analysis_gate(state: AgentState) -> dict:
    slice_res     = state.get("slice_result")
    slice_dims    = slice_res.ranked_dimensions if slice_res else []
    top_slice     = slice_dims[0] if slice_dims else {}

    guardrail_res = state.get("guardrail_result")
    breached      = [g for g in (guardrail_res.guardrails if guardrail_res else []) if g.breached]

    forecast_res  = state.get("forecast_result")
    cuped_res     = state.get("cuped_result")
    ttest_res     = state.get("ttest_result")
    hte_res       = state.get("hte_result")
    novelty_res   = state.get("novelty_result")
    funnel_res    = state.get("funnel_result")
    mde_res       = state.get("mde_result")

    payload = {
        "gate":                    "analysis",
        "decomposition":           _to_dict(state.get("decomposition_result")),
        "top_anomaly_slice":       top_slice,
        "forecast_outside_ci":     forecast_res.outside_ci if forecast_res else None,
        "cuped_variance_reduction": cuped_res.variance_reduction_pct if cuped_res else None,
        "significant":             ttest_res.significant if ttest_res else None,
        "top_segment":             hte_res.top_segment if hte_res else None,
        "novelty_likely":          novelty_res.novelty_likely if novelty_res else None,
        "guardrails_breached":     guardrail_res.any_breached if guardrail_res else None,
        "breached_metrics":        [g.model_dump() for g in breached],
        "biggest_funnel_dropoff":  funnel_res.biggest_dropoff_step if funnel_res else None,
        "mde_powered":             mde_res.is_powered_for_observed_effect if mde_res else None,
        "business_impact":         state.get("business_impact"),
        "message":                 "Review the analysis results. Approve or add notes/overrides.",
    }
    analyst_response = interrupt(payload)

    notes = analyst_response.get("notes", "")
    override = dict(state.get("analyst_override") or {})
    if notes.strip():
        override["analysis_notes"] = notes.strip()

    return {
        "analysis_approved": analyst_response.get("approved", True),
        "analyst_notes":     notes,
        "analyst_override":  override,
    }


# ── Node 18: generate_narrative ───────────────────────────────────────────────

@observe(name="generate_narrative", as_type="generation")
def generate_narrative(state: AgentState) -> dict:
    mc     = state.get("metric_config") or load_metric_config()
    metric = state.get("metric") or mc.primary_metric

    # Build template draft via narrative_tools (pure Python, no LLM)
    # narrative_tools accepts plain dicts — convert Pydantic models via _to_dict
    try:
        template_out = narrative_tools.format_narrative(
            metric=metric,
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
            analyst_notes=state.get("analyst_notes") or "",
        )
    except Exception as exc:
        logger.warning("narrative_tools.format_narrative failed: %s", exc)
        from tools.schemas import NarrativeResult
        template_out = NarrativeResult(narrative_draft="", recommendation="")

    draft_narrative = template_out.narrative_draft

    # Collect tool results for the LLM prompt — convert Pydantic models to dicts
    tool_results: dict = {}
    for k, v in state.items():
        if k.endswith("_result") and v is not None:
            tool_results[k] = _to_dict(v)
    # Drop forecast_df to avoid DataFrame serialization issues
    if "forecast_result" in tool_results:
        tool_results["forecast_result"].pop("forecast_df", None)
    tool_results_json = json.dumps(tool_results, default=str, indent=2)

    analyst_notes     = state.get("analyst_notes") or ""
    analyst_notes_section = (
        ANALYST_NOTES_BLOCK.format(analyst_notes=analyst_notes)
        if analyst_notes.strip() else ""
    )

    task_prompt = NARRATIVE_PROMPT.format(
        metric=metric,
        metric_direction=mc.metric_direction,
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
        "recommendation":       template_out.recommendation,
        "conversation_history": new_history,
        "cache_read_tokens":    (state.get("cache_read_tokens") or 0) + cost_info.get("cache_read_tokens", 0),
        "cache_write_tokens":   (state.get("cache_write_tokens") or 0) + cost_info.get("cache_write_tokens", 0),
        "estimated_cost_usd":   (state.get("estimated_cost_usd") or 0.0) + cost_info.get("estimated_cost_usd", 0.0),
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

    override = dict(state.get("analyst_override") or {})
    if analyst_notes.strip():
        override["narrative_notes"] = analyst_notes.strip()
    if rec_override := analyst_response.get("recommendation_override", ""):
        override["recommendation_override"] = rec_override.strip()

    if approved:
        return {
            "narrative_approved": True,
            "final_narrative":    state.get("narrative_draft", ""),
            "analyst_notes":      analyst_notes,
            "analyst_override":   override,
        }

    # Analyst wants a revision — updated notes will cause graph to re-run generate_narrative
    return {
        "narrative_approved": False,
        "analyst_notes":      analyst_notes,
        "analyst_override":   override,
    }


# ── Quality score (completeness signal, no ground truth needed) ───────────────

def _compute_quality_score(state: AgentState) -> float:
    """
    Estimate run quality from tool-result completeness.

    Returns a 0–1 float. Used when no ground-truth eval_score is available
    so every run still contributes a learning signal to the memory store.
    """
    cuped = state.get("cuped_result")
    checks = [
        bool(cuped and cuped.variance_reduction_pct > 5),
        bool(state.get("ttest_result")),
        bool(state.get("hte_result") and state["hte_result"].top_segment),
        bool(state.get("guardrail_result")),
        bool(state.get("novelty_result")),
        bool(state.get("forecast_result")),
    ]
    return sum(checks) / len(checks)


# ── Node 20: log_run_node ─────────────────────────────────────────────────────

@observe(name="log_run")
def log_run_node(state: AgentState) -> dict:
    run_id = state.get("run_id") or str(uuid.uuid4())
    task   = state.get("task", "")

    # Persist to memory store
    log_run(
        task=task,
        run_id=run_id,
        user_id=state.get("user_id"),
        metric=state.get("metric") or "",
        covariate=state.get("covariate") or "",
        db_backend=state.get("db_backend") or "duckdb",
        analyst_override=state.get("analyst_override") or None,
        top_segment=(hte := state.get("hte_result")) and hte.top_segment or "",
        eval_score=state.get("eval_score"),
        cache_read_tokens=state.get("cache_read_tokens") or 0,
        cache_write_tokens=state.get("cache_write_tokens") or 0,
        estimated_cost_usd=state.get("estimated_cost_usd") or 0.0,
        semantic_cache_hits=1 if state.get("semantic_cache_hit") else 0,
        notes=state.get("analyst_notes") or "",
    )

    # In-band completeness scoring — fills eval_score when offline eval hasn't run yet
    if state.get("eval_score") is None:
        quality_score = _compute_quality_score(state)
        update_eval_score(run_id, quality_score)
        logger.info("log_run: quality score %.2f stored for run %s", quality_score, run_id)

    # Store SQL result in semantic cache only when the result has all required columns
    # (prevents poisoning the cache with bad date-level SQL)
    if state.get("generated_sql"):
        mc_log  = state.get("metric_config") or load_metric_config()
        df_log  = state.get("query_result")
        required_cols = {mc_log.primary_metric, mc_log.covariate, "variant"}
        result_cols   = set(df_log.columns) if df_log is not None and hasattr(df_log, "columns") else set()
        if required_cols.issubset(result_cols):
            narrative = state.get("final_narrative") or state.get("narrative_draft") or ""
            semantic_cache.store_cache(
                task=task,
                node_name="generate_sql",
                result={
                    "sql":            state["generated_sql"],
                    "narrative":      narrative,
                    "recommendation": state.get("recommendation") or "",
                },
                run_id=run_id,
            )
        else:
            logger.info(
                "log_run: skipping SQL cache — result still missing %s",
                sorted(required_cols - result_cols),
            )

    flush()

    return {"run_id": run_id}


# ── Node 21: infer_metric_config_node ─────────────────────────────────────────
# Called once when connecting a new external DB that has no config file.
# No-op if metric_config is already present in state (DuckDB demo / pre-loaded config).

@observe(name="infer_metric_config")
def infer_metric_config_node(state: AgentState) -> dict:
    """
    LLM infers MetricConfig from schema. Result stored in state for UI form pre-fill.
    Only runs when metric_config is not already set in state.
    """
    if state.get("metric_config"):
        return {}

    schema_context = state.get("schema_context", "")
    if not schema_context:
        mc = load_metric_config()
        return {"metric_config": mc, "metric": mc.primary_metric, "covariate": mc.covariate}

    prompt = SCHEMA_CONFIG_INFERENCE_PROMPT.format(schema_context=schema_context)

    with trace_generation("infer_metric_config", _model(), prompt) as gen:
        response = _anthropic_client().messages.create(
            model=_model(),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        gen.update(response)

    defaults = load_metric_config()

    try:
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        inferred = MetricConfig(**json.loads(raw))
    except Exception as exc:
        logger.warning("infer_metric_config: LLM response parsing failed (%s), using defaults.", exc)
        inferred = defaults

    # ── Cross-check every inferred column/table name against the live schema ──
    # If the LLM hallucinated a column name, _sanitise_metric_config replaces it
    # with the default and logs a warning.  This prevents bad names from
    # propagating to _canonical_experiment_sql() and generating broken SQL.
    inferred, issues = _sanitise_metric_config(inferred, schema_context, defaults)
    for w in issues:
        logger.warning("infer_metric_config: schema mismatch — %s", w)

    return {
        "metric_config": inferred,
        "metric":         inferred.primary_metric,
        "covariate":      inferred.covariate,
    }
