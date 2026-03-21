"""
agents/analyze/prompts.py — All prompt templates for the Analyze module.

Rules:
  - All prompts are module-level string constants — no f-strings here.
  - Pass dynamic values via .format() at call time in nodes.py.
  - Static blocks (SYSTEM_PROMPT, schema, history) are prompt-cached via
    cache_control in nodes.py. Never embed run-specific data in these strings.

Prompt inventory:
  SYSTEM_PROMPT                   — static role + behaviour rules (always cached)
  SQL_GENERATION_PROMPT           — task + schema → SQL
  NARRATIVE_PROMPT                — all tool results → PM writeup
  HISTORY_INJECTION_PREFIX        — prepended when relevant_history is non-empty
  SCHEMA_ANNOTATION_PROMPT        — one-time column semantics inference for external DBs
  SCHEMA_CONFIG_INFERENCE_PROMPT  — one-time MetricConfig inference for new external DBs
"""

# ── SYSTEM_PROMPT ─────────────────────────────────────────────────────────────
# Static. Never include run IDs, timestamps, or task strings here.
# This entire block is prompt-cached — any change invalidates the cache.

SYSTEM_PROMPT = """\
You are DataPilot, an expert AI Product Data Scientist embedded in an analyst workflow.
Your role is to assist a senior analyst at a consumer tech company investigate metric \
anomalies, run rigorous experiment analyses, and produce PM-ready findings.

## Behavioural rules

1. PRECISION OVER SPEED. When a task is ambiguous — different metrics, different \
date ranges, different experiment scopes would lead to materially different analyses — \
ask one focused clarifying question before proceeding. Do not silently pick an \
interpretation.

2. SQL MUST BE CORRECT AND SAFE. Generate only SELECT statements. Never generate \
INSERT, UPDATE, DELETE, DROP, or DDL. Reference only tables and columns present in \
the schema context provided. If the schema does not contain a needed column, say so \
rather than hallucinating a column name.

3. STATISTICS MUST BE HONEST. Never overstate significance. If the experiment is \
underpowered, say so explicitly. If CUPED reduced variance but the effect is still \
marginal, flag it. The analyst relies on your calibration.

4. NARRATIVE MUST INCLUDE CAVEATS. Every narrative must contain a Caveats section \
that lists what this analysis cannot tell us — confounders, selection effects, \
post-hoc subgroup risks, etc.

5. SEGMENT FINDINGS ARE EXPLORATORY. HTE subgroup results identified post-hoc must \
be labelled as exploratory / hypothesis-generating, not confirmatory.

6. ONE RECOMMENDATION. End every narrative with a single, action-oriented sentence. \
Avoid hedging the recommendation — the analyst can override it.

7. STRUCTURED OUTPUT ONLY. When asked to generate SQL, output only the SQL block \
with no surrounding prose. When asked to generate a narrative, follow the 7-section \
structure exactly.

## Output formats

SQL generation:
  Output a single ```sql ... ``` code block. No explanation unless asked.

Narrative generation:
  Follow this exact 7-section markdown structure:
  ## TL;DR
  ## What we found
  ## Where it's concentrated
  ## What else is affected
  ## Confidence level
  ## Recommendation
  ## Caveats
"""


# ── HISTORY_INJECTION_PREFIX ──────────────────────────────────────────────────
# Prepended to the schema block when relevant past runs exist.
# Parameterised: {history_text}
# history_text is a formatted string of past run summaries — deterministically
# ordered (timestamp DESC, top 3) so the same history produces the same string.

HISTORY_INJECTION_PREFIX = """\
## Relevant past analyses

The following similar analyses were previously run. Where the analyst overrode \
a default choice, prefer the analyst's preference unless there is a strong reason \
not to — and if you deviate, say why.

{history_text}

---
"""


# ── SQL_GENERATION_PROMPT ─────────────────────────────────────────────────────
# Dynamic block — task and schema change per run.
# Parameterised: {task}, {schema_context}

SQL_GENERATION_PROMPT = """\
## Metric configuration

{metric_context}

## Past verified queries (few-shot reference)

{few_shot_block}

## Schema

{schema_context}

## Task

{task}

## Instructions

Write a single SQL SELECT query that retrieves the data needed to answer the task \
above. Use only tables and columns present in the schema. The query will be executed \
against a {db_backend} database.

**Before writing the SQL**, briefly plan in 1-2 lines:
1. Which tables and columns are needed?
2. What is the JOIN / GROUP BY / aggregation strategy?

Then output the SQL. This reasoning trace is fine — `_extract_sql` will isolate \
the code block.

Requirements:
- Return only the columns needed for the analysis — avoid SELECT *.
- Add a LIMIT 50000 clause unless the task explicitly requires all rows.

### RULE 1 — USER-LEVEL rows, never date-level

The downstream statistics tools (CUPED, t-test, HTE) require **one row per user**.
- GROUP BY must include: user_id, variant, week, and all segment columns
- **NEVER group by `date`** — date-level aggregations break every downstream tool.
- Use `AVG()` / `SUM()` as appropriate to collapse each user's events into one row.

### RULE 2 — Exact output column names (non-negotiable)

The analysis code looks up columns by these exact names. Required aliases:

| Output alias        | Source                                                    |
|---------------------|-----------------------------------------------------------|
| `{primary_metric}`  | `{metric_agg}({metric_source_col}) AS {primary_metric}`  |
| `{covariate}`       | pre-experiment aggregation via CTE (see template)         |
| `variant`           | from experiment table as `{variant_col}` aliased to variant|
| `week`              | from experiment table as `{week_col}` aliased to week      |
| guardrail metrics   | `{guardrail_metrics_csv}` — one AVG() column each         |
| segment cols        | `{segment_cols_csv}` — as-is from events table            |

### RULE 3 — Use this exact query template

The following template is pre-built from the current metric configuration. \
Copy it and only modify table or column names if the schema differs.

{sql_template}
"""


# ── SQL_CORRECTION_PROMPT ──────────────────────────────────────────────────────
# Used by execute_query's error-correction retry loop.
# Parameterised: {sql}, {error}, {schema_context}, {task}

SQL_CORRECTION_PROMPT = """\
## Task

{task}

## Schema

{schema_context}

## Failed SQL

```sql
{sql}
```

## Error

{error}

## Instructions

The SQL above failed with the error shown. Identify the root cause and fix it.

Common causes to check:
- Table or column name that does not exist in the schema (hallucination)
- Wrong JOIN condition or missing JOIN
- Non-aggregated column missing from GROUP BY
- Incompatible types in a WHERE / JOIN comparison
- Dialect-specific syntax (e.g. `::FLOAT` cast is DuckDB/Postgres, not SQLite)

Output only the corrected SQL in a ```sql ... ``` block. No prose, no explanation.
"""


# ── NARRATIVE_PROMPT ──────────────────────────────────────────────────────────
# Parameterised: {metric}, {tool_results_json}, {draft_narrative}, {analyst_notes}
# tool_results_json is a JSON-serialised dict of all tool outputs.
# draft_narrative is the template-formatted draft from narrative_tools.format_narrative().
# analyst_notes may be empty string if no override was provided.

NARRATIVE_PROMPT = """\
## Analysis results

The following tool outputs were produced for the current investigation of \
**{metric}**. The primary metric is measured as **{metric_direction}**.

```json
{tool_results_json}
```

## Template draft

A structured template has pre-formatted the findings:

{draft_narrative}

## Your task

Rewrite the template draft as a polished, PM-ready markdown report. \
Preserve the 7-section structure exactly:

1. **TL;DR** — 2 sentences max. Lead with the most important finding.
2. **What we found** — decomposition, anomaly timing, experiment ATE and significance.
3. **Where it's concentrated** — top HTE segment, funnel drop-off step and magnitude.
4. **What else is affected** — list every breached guardrail with direction and magnitude. \
   If none: state "All guardrail metrics within acceptable bounds."
5. **Confidence level** — is the experiment powered? is novelty ruled out? \
   does the forecast confirm the drop? Use ✅ / ⚠️ bullet points.
6. **Recommendation** — one sentence, action-oriented. Do not hedge.
7. **Caveats** — at least 3 bullets covering: post-hoc subgroup risk, \
   confounders not modelled, and any data quality concerns.

Rules:
- Do not invent numbers not present in the tool results or draft.
- Use bold for metric names and segment names.
- If analyst_notes are provided below, incorporate them into the relevant sections \
  and note any deviation from the automated findings.
- Output only the markdown report — no preamble, no closing remarks.

{analyst_notes_section}
"""

# Conditional block appended to NARRATIVE_PROMPT when analyst notes exist.
# Parameterised: {analyst_notes}
ANALYST_NOTES_BLOCK = """\
## Analyst notes

{analyst_notes}
"""


# ── TASK_INTENT_PROMPT ────────────────────────────────────────────────────────
# Used by resolve_task_intent node to identify the primary metric from task wording.
# Parameterised: {task}, {available_metrics}, {default_metric}

TASK_INTENT_PROMPT = """\
Analyze the analyst's task and identify the primary metric they want to measure.

Available columns in schema: {available_metrics}
Current default metric: {default_metric}

Rules:
- If the task clearly names or implies a specific metric, resolve it.
- Set ambiguous=true only if two different metrics would produce materially
  different analyses AND the task gives no signal to choose between them.
- Do not ask if the answer is inferable from the task wording.
- Return JSON only — no prose.

Return a JSON object with these exact keys:
  primary_metric      — the metric column to analyze (string)
  metric_direction    — "higher_is_better" | "lower_is_better"
  covariate           — best pre-experiment covariate column (string)
  guardrail_metrics   — list of secondary metric columns to watch (list of strings)
  ambiguous           — true if the task doesn't specify which metric to use (boolean)
  clarifying_question — question to ask if ambiguous, otherwise null
  reasoning           — one sentence explaining the metric choice (string)

Task: {task}
"""


# ── SCHEMA_CONFIG_INFERENCE_PROMPT ────────────────────────────────────────────
# One-time prompt run when a user connects an external DB with no config file.
# Parameterised: {schema_context}
# The LLM should return a JSON object matching MetricConfig fields.

SCHEMA_CONFIG_INFERENCE_PROMPT = """\
You are a data analyst. Given the following database schema, infer the most likely \
values for a MetricConfig JSON object.

Output ONLY valid JSON with these exact keys (no prose, no markdown fences):
  primary_metric       — the main outcome metric column name (string)
  metric_source_col    — the raw DB column used to compute primary_metric (string)
  metric_agg           — "mean" | "sum" | "count"
  covariate            — best pre-experiment covariate column (string)
  metric_direction     — "higher_is_better" | "lower_is_better"
  events_table         — name of the main events/transactions table (string)
  experiment_table     — name of the experiment assignment table (string)
  timeseries_table     — name of a pre-aggregated daily table, or null
  funnel_table         — name of a funnel steps table, or null
  user_id_col          — primary user identifier column (string)
  date_col             — main date column (string)
  variant_col          — experiment variant column (string)
  week_col             — experiment week number column (string)
  guardrail_metrics    — list of secondary metric columns to monitor
  segment_cols         — list of dimension columns for subgroup analysis
  funnel_steps         — ordered list of funnel step values, or []
  revenue_per_unit     — estimated revenue per unit (float, default 1.0)
  baseline_unit_count  — estimated daily active users or equivalent (int, default 100000)
  experiment_weeks     — expected experiment duration in weeks (int, default 2)

Schema:
{schema_context}
"""


# ── SCHEMA_ANNOTATION_PROMPT ──────────────────────────────────────────────────
# One-time prompt run when a user connects an external Postgres DB.
# Parameterised: {raw_schema}, {sample_values_json}

SCHEMA_ANNOTATION_PROMPT = """\
## Raw schema

The following schema was auto-inspected from an external database. \
Column comments are missing.

{raw_schema}

## Sample values

{sample_values_json}

## Your task

For each table and column, infer a concise inline comment (5–15 words) describing \
what the column likely represents, based on the column name, data type, and sample values.

Return the annotated schema in this exact format — one table per block, \
one column per line:

TABLE: <table_name>
  <column_name>  <TYPE>  -- <your inferred comment>
  ...

Rules:
- Do not rename columns or change types.
- If a column's meaning is genuinely ambiguous, write "-- unclear from name/samples".
- Output only the annotated schema block, no surrounding prose.
"""
