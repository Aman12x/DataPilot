"""
agents/analyze/node_shared.py — Shared helpers for Analyze graph nodes.

Each node:
  - Takes AgentState, returns a partial AgentState dict (LangGraph merges it).
  - Calls tools from tools/ only — no inline stats, SQL, or string formatting.
  - Never calls other nodes directly.

HITL gates use langgraph.types.interrupt() — never input() or st.text_input().
"""

from __future__ import annotations

import json
import logging
import math
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
    DECK_PROMPT,
    HISTORY_INJECTION_PREFIX,
    INSIGHTS_NARRATIVE_PROMPT,
    LOOKUP_NARRATIVE_PROMPT,
    NARRATIVE_AUDIT_PROMPT,
    NARRATIVE_PROMPT,
    POWER_ANALYSIS_NARRATIVE_PROMPT,
    SCHEMA_CONFIG_INFERENCE_PROMPT,
    SQL_CORRECTION_PROMPT,
    SQL_GENERATION_GENERAL_PROMPT,
    SQL_GENERATION_PROMPT,
    SYSTEM_PROMPT,
    TASK_INTENT_PROMPT,
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
    describe_tools,
    forecast_tools,
    funnel_tools,
    guardrail_tools,
    mde_tools,
    narrative_tools,
    novelty_tools,
    regression_tools,
    stats_tools,
)
from tools.db_tools import DBConnection
from tools.schemas import PowerAnalysisResult, SensitivityRow, SliceResult, SrmResult
from tools.chart_tools import (
    compute_trust_indicators,
    generate_ab_charts,
    generate_general_charts,
)

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
_MAX_TOKENS_SQL       = int(os.getenv("MAX_TOKENS_SQL",       "4096"))
_MAX_TOKENS_NARRATIVE = int(os.getenv("MAX_TOKENS_NARRATIVE", "4096"))

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
    return os.getenv("MODEL") or _fast_model()


def _fast_model() -> str:
    """Haiku for latency-sensitive tasks (SQL gen, correction) where speed > depth."""
    return os.getenv("FAST_MODEL", "claude-haiku-4-5-20251001")


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
        f"Guardrail metrics: {', '.join(mc.guardrail_metrics) or '(none)'}",
        f"Segment columns:   {', '.join(mc.segment_cols) or '(none)'}",
        f"Funnel steps:      {', '.join(mc.funnel_steps) or '(none)'}",
    ]
    return "\n".join(lines)


def _canonical_experiment_sql(mc: "MetricConfig") -> str:
    """
    Return the known-good user-level experiment SQL for the current MetricConfig.

    Fully dynamic — uses mc fields for all table names, column names, and
    aggregation expressions. Used as automatic fallback when LLM SQL is invalid.
    """
    agg_map = {
        "mean":  f"CAST(AVG(e.{mc.metric_source_col}) AS FLOAT)",
        "sum":   f"CAST(SUM(e.{mc.metric_source_col}) AS FLOAT)",
        "count": f"CAST(COUNT(*) AS FLOAT)",
    }
    pre_agg_map = {
        "mean":  f"CAST(AVG(pre_events.{mc.covariate}) AS FLOAT)",
        "sum":   f"CAST(SUM(pre_events.{mc.covariate}) AS FLOAT)",
        "count": f"CAST(COUNT(*) AS FLOAT)",
    }
    metric_agg_expr   = agg_map.get(mc.metric_agg,   agg_map["mean"])
    covariate_agg_expr = pre_agg_map.get(mc.metric_agg, pre_agg_map["mean"])

    guardrail_selects = "\n".join(
        f"    CAST(AVG(e.{m}) AS FLOAT) AS {m},"
        for m in mc.guardrail_metrics
        if m not in (mc.primary_metric, mc.metric_source_col, mc.covariate)
    )
    segment_selects = "\n".join(f"    e.{c}," for c in mc.segment_cols)
    segment_group   = ", ".join(f"e.{c}" for c in mc.segment_cols)

    # Cast both sides to VARCHAR for date comparisons so that VARCHAR date
    # columns (e.g. "month" as "2023-01") never cause type-mismatch errors
    # against the DATE assignment_date column in the stub experiment table.
    min_assign = f"CAST((SELECT MIN({mc.assignment_date_col}) FROM {mc.experiment_table}) AS VARCHAR)"
    date_cast  = f"CAST(e.{mc.date_col} AS VARCHAR)"

    return f"""\
WITH pre_exp AS (
    SELECT {mc.user_id_col},
           {covariate_agg_expr} AS {mc.covariate}
    FROM   {mc.events_table} pre_events
    WHERE  CAST(pre_events.{mc.date_col} AS VARCHAR) < {min_assign}
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
           AND {date_cast} >= {min_assign}
LEFT JOIN  pre_exp p ON ex.{mc.user_id_col} = p.{mc.user_id_col}
GROUP BY   e.{mc.user_id_col}, ex.{mc.variant_col}, ex.{mc.week_col}{", " + segment_group if segment_group else ""}, p.{mc.covariate}
LIMIT 50000"""


def _extract_sql(text: str) -> str:
    """Extract SQL from a ```sql ... ``` code block in LLM output."""
    # Prefer fully closed fence
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Handle unclosed fence (LLM output truncated before closing ```)
    match = re.search(r"```sql\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip("`").strip()
    # Fallback: return raw text stripped of any stray backticks
    return text.strip().strip("`").strip()


def _db_conn(state: AgentState) -> DBConnection:
    backend = state.get("db_backend", "duckdb")
    if backend == "postgres":
        return DBConnection(
            backend="postgres",
            host=state.get("pg_host")     or os.getenv("PG_HOST", "localhost"),
            port=int(state.get("pg_port") or os.getenv("PG_PORT", "5432")),
            dbname=state.get("pg_dbname") or os.getenv("PG_DBNAME", ""),
            user=state.get("pg_user")     or os.getenv("PG_USER", ""),
            password=state.get("pg_password") or os.getenv("PG_PASSWORD", ""),
        )
    # prefer state-injected path (CSV/Excel upload) over env-var default
    path = state.get("duckdb_path") or os.getenv("DUCKDB_PATH", "data/dau_experiment.db")
    return DBConnection(backend=backend, path=path)


def _safe_df(state: AgentState) -> pd.DataFrame | None:
    """Return query_result DataFrame from state, or None if missing/empty."""
    df = state.get("query_result")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    return df


def _validate_query_content(
    df: pd.DataFrame,
    mc: "MetricConfig",
    mode: str,
) -> list[str]:
    """
    Content-level validation of query results.

    Checks for semantically broken data that would silently produce wrong
    reports: zero rows, too-few rows, missing arms, severe arm imbalance,
    and metric values that look like percentages instead of rates.

    Returns a list of warning strings (empty = all clear).
    """
    warnings: list[str] = []

    if len(df) == 0:
        warnings.append(
            "Query returned 0 rows — cannot perform analysis on empty data."
        )
        return warnings  # further checks would error on empty df

    if len(df) < 10:
        warnings.append(
            f"Query returned only {len(df)} rows — too few for reliable statistics "
            "(minimum recommended: 10)."
        )

    if mode == "ab_test" and "variant" in df.columns:
        variants = df["variant"].dropna().unique()
        variant_lower = {str(v).lower() for v in variants}
        has_ctrl = any("control" in v for v in variant_lower)
        has_trt  = any("treatment" in v for v in variant_lower)

        if not has_ctrl or not has_trt:
            warnings.append(
                f"Variant column contains only: {list(variants)}. "
                "Expected both 'control' and 'treatment' arms — "
                "assignment join may be broken."
            )
        else:
            counts    = df["variant"].value_counts()
            ctrl_name = next((v for v in counts.index if "control"   in str(v).lower()), None)
            trt_name  = next((v for v in counts.index if "treatment" in str(v).lower()), None)
            if ctrl_name and trt_name:
                n_ctrl = int(counts[ctrl_name])
                n_trt  = int(counts[trt_name])
                total  = n_ctrl + n_trt
                ratio  = min(n_ctrl, n_trt) / total if total > 0 else 0.5
                if ratio < 0.30:
                    warnings.append(
                        f"Arm imbalance: control={n_ctrl:,}, treatment={n_trt:,} "
                        f"(minority arm is {ratio:.1%} of total). "
                        "Possible SRM — check assignment logs."
                    )

    if mode == "ab_test" and mc.primary_metric in df.columns:
        col = df[mc.primary_metric].dropna()
        # Only flag >1 as suspicious for metrics that should be rates (0–1).
        # Revenue, duration, count, and score columns are legitimately > 1.
        _RATE_KEYWORDS = {"rate", "ctr", "cvr", "conversion", "retention", "churn",
                          "open_rate", "click_rate", "bounce", "engagement_rate"}
        metric_lower = mc.primary_metric.lower()
        is_rate_metric = any(kw in metric_lower for kw in _RATE_KEYWORDS)
        if is_rate_metric and len(col) > 0 and (col > 1.0).mean() > 0.8:
            warnings.append(
                f"Metric '{mc.primary_metric}' has {(col > 1.0).mean():.0%} of values > 1.0 "
                "— looks like a percentage (0–100) rather than a rate (0–1). "
                "Verify the aggregation expression in the SQL."
            )

    # ── JOIN fan-out detection ─────────────────────────────────────────────────
    # A fan-out occurs when a JOIN produces more rows than the entity table has
    # entities, usually from a missing GROUP BY or a 1-to-many JOIN key.
    # For A/B: rows should be ≤ n_users × n_variants × n_weeks (3× buffer).
    # For general: any entity column with duplicates is suspicious.
    _ENTITY_COLS_LOWER = {"user_id", "customer_id", "patient_id", "userid",
                          "uid", "entity_id", "shipment_id"}
    entity_col = next(
        (c for c in df.columns if c.lower() in _ENTITY_COLS_LOWER), None
    )
    if entity_col:
        n_rows    = len(df)
        n_unique  = df[entity_col].nunique()
        if mode == "ab_test" and "variant" in df.columns:
            n_variants = max(df["variant"].nunique(), 1)
            n_weeks    = max(df["week"].nunique(), 1) if "week" in df.columns else 1
            expected   = n_unique * n_variants * n_weeks
            if expected > 0 and n_rows > expected * 3:
                warnings.append(
                    f"JOIN fan-out: {n_rows:,} rows but {n_unique:,} unique {entity_col}s "
                    f"× {n_variants} variant(s) × {n_weeks} week(s) = {expected:,} expected "
                    f"({n_rows / expected:.1f}× over). Missing GROUP BY or duplicate JOIN key."
                )
        elif mode == "general" and df[entity_col].duplicated().any():
            ratio = round(n_rows / max(n_unique, 1), 1)
            if ratio > 5:
                warnings.append(
                    f"JOIN fan-out: {n_rows:,} rows but only {n_unique:,} unique {entity_col}s "
                    f"({ratio}× ratio). Missing GROUP BY — results are likely double-counted."
                )

    return warnings


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
            raw = s.split(":", 1)[1].strip()
            current = raw.split("--")[0].strip().lower()  # strip "-- N rows" annotation
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


def _tables_in_sql(sql: str) -> set[str]:
    """
    Extract real (non-CTE) table names from FROM/JOIN clauses.
    Used to filter few-shot examples whose SQL references tables absent from
    the current schema — prevents demo-DB examples from misleading the LLM
    when the user has uploaded a completely different dataset.
    """
    # Identify CTE aliases so we don't count them as real tables
    cte_names: set[str] = set()
    for m in re.finditer(r'\bWITH\s+(\w+)\s+AS\b', sql, re.IGNORECASE):
        cte_names.add(m.group(1).lower())
    for m in re.finditer(r',\s*(\w+)\s+AS\s*\(', sql, re.IGNORECASE):
        cte_names.add(m.group(1).lower())

    tables: set[str] = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE):
        name = m.group(1).lower()
        if name.upper() not in _SQL_KEYWORDS and len(name) > 2 and name not in cte_names:
            tables.add(name)
    return tables


def _filter_few_shot_by_schema(
    examples: list[dict],
    known_tables: set[str],
) -> list[dict]:
    """
    Drop any few-shot example whose SQL references a table that does not exist
    in the current schema.

    Rationale: the memory store accumulates examples from all past runs.
    If a user uploads a SaaS-churn CSV, we must not inject examples from the
    built-in demo DB (which reference `events`, `experiment`, `metrics_daily`)
    because these table names do not exist in their upload — the LLM may mimic
    the JOIN pattern and generate broken SQL.

    An example is kept when:
      - Its SQL references no tables (degenerate but safe), OR
      - Every table it references is present in the current schema.
    """
    if not known_tables:
        # Schema parse failed — inject no examples rather than potentially
        # mismatched ones from a different dataset.
        return []
    return [
        ex for ex in examples
        if not _tables_in_sql(ex.get("sql", ""))
        or _tables_in_sql(ex.get("sql", "")).issubset(known_tables)
    ]


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
            raw = s.split(":", 1)[1].strip()
            current = raw.split("--")[0].strip().lower()  # strip "-- N rows" annotation
            known_tables.add(current)
        elif current and s and not s.startswith("--") and not s.startswith("DIALECT"):
            col = s.split()[0].lower()
            if col:
                known_columns.add(col)
    return known_tables, known_columns


def _columns_for_table(schema_context: str, table_name: str) -> set[str]:
    """Return the lowercased column names for a specific table in the schema."""
    cols: set[str] = set()
    inside = False
    for line in schema_context.splitlines():
        s = line.strip()
        if s.startswith("TABLE:"):
            raw = s.split(":", 1)[1].strip()
            inside = raw.split("--")[0].strip().lower() == table_name.lower()
        elif inside and s and not s.startswith("--") and not s.startswith("DIALECT"):
            col = s.split()[0].lower()
            if col:
                cols.add(col)
    return cols


def _sanitise_metric_config(
    mc: MetricConfig,
    schema_context: str,
    defaults: MetricConfig,
) -> tuple[MetricConfig, list[str]]:
    """
    Cross-check MetricConfig column/table fields against the live schema.

    For any field whose value doesn't appear in the schema:
      1. Try the corresponding `defaults` value.
      2. If that also isn't in the schema (common for uploaded files where
         built-in demo defaults like 'pre_session_count' don't apply), fall
         back to the first available column from the live schema rather than
         silently writing a non-existent name into the config.

    Table name fields are checked against known_tables; column name fields
    against known_columns.

    Returns:
        (sanitised MetricConfig, list of warning strings)
    """
    if not schema_context.strip():
        return mc, []  # no schema to validate against

    known_tables, known_columns = _known_schema_names(schema_context)
    if not known_tables and not known_columns:
        return mc, []

    # Stable fallback pool: columns sorted so the pick is deterministic
    _sorted_cols = sorted(known_columns)

    def _best_col(*candidates: str) -> str:
        """Return first candidate present in schema, or first schema column."""
        for c in candidates:
            if c and c.lower() in known_columns:
                return c
        return _sorted_cols[0] if _sorted_cols else candidates[-1]

    def _best_table(*candidates: str | None) -> str | None:
        for c in candidates:
            if c and c.lower() in known_tables:
                return c
        return None

    warnings: list[str] = []
    overrides: dict = {}

    def _check_col(field: str, value: str, fallback: str) -> None:
        if value.lower() in known_columns:
            return
        effective = _best_col(fallback, value)
        warnings.append(
            f"{field}={value!r} not in schema"
            + (f" (default {fallback!r} also absent)" if fallback.lower() not in known_columns else "")
            + f" — using {effective!r}"
        )
        overrides[field] = effective

    def _check_table(field: str, value: str | None, fallback: str | None) -> None:
        if not value or value.lower() in known_tables:
            return
        effective = _best_table(fallback, value)
        warnings.append(
            f"{field}={value!r} not in schema — using {effective!r}"
        )
        overrides[field] = effective

    # Column name fields
    _check_col("metric_source_col", mc.metric_source_col, defaults.metric_source_col)
    _check_col("covariate",         mc.covariate,         defaults.covariate)
    _check_col("user_id_col",       mc.user_id_col,       defaults.user_id_col)
    _check_col("date_col",          mc.date_col,          defaults.date_col)
    _check_col("variant_col",       mc.variant_col,       defaults.variant_col)
    _check_col("week_col",          mc.week_col,          defaults.week_col)

    # Table name fields (optional tables may be None)
    _check_table("events_table",     mc.events_table,     defaults.events_table)
    _check_table("experiment_table", mc.experiment_table, defaults.experiment_table)
    if mc.timeseries_table:
        _check_table("timeseries_table", mc.timeseries_table, defaults.timeseries_table)
    if mc.funnel_table:
        _check_table("funnel_table", mc.funnel_table, defaults.funnel_table)

    # List fields — drop elements not in schema; try defaults as fallback only if
    # the default values themselves exist in the schema (demo DB case).
    # For uploads where defaults also aren't in the schema, the result is an
    # empty list rather than a list of non-existent column names.
    clean_guardrails = [m for m in mc.guardrail_metrics if m.lower() in known_columns]
    if len(clean_guardrails) < len(mc.guardrail_metrics):
        dropped = set(mc.guardrail_metrics) - set(clean_guardrails)
        warnings.append(f"guardrail_metrics: removed {sorted(dropped)} (not in schema)")
        schema_defaults = [m for m in defaults.guardrail_metrics if m.lower() in known_columns]
        overrides["guardrail_metrics"] = clean_guardrails or schema_defaults

    clean_segments = [c for c in mc.segment_cols if c.lower() in known_columns]
    if len(clean_segments) < len(mc.segment_cols):
        dropped = set(mc.segment_cols) - set(clean_segments)
        warnings.append(f"segment_cols: removed {sorted(dropped)} (not in schema)")
        schema_defaults = [c for c in defaults.segment_cols if c.lower() in known_columns]
        overrides["segment_cols"] = clean_segments or schema_defaults

    if not overrides:
        return mc, []

    sanitised = mc.model_copy(update=overrides)
    return sanitised, warnings


from agents.analyze.prompt_safety import wrap_untrusted_content


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
        schema_context=wrap_untrusted_content(schema_context, label="database_schema"),
        task=wrap_untrusted_content(task, label="analyst_task"),
    )
    try:
        response = _anthropic_client().messages.create(
            model=_fast_model(),
            max_tokens=_MAX_TOKENS_SQL,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_sql(response.content[0].text)
    except Exception as exc:
        logger.warning("_llm_correct_sql: correction call failed: %s", exc)
        return sql
