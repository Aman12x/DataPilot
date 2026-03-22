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
You are DataPilot, an expert AI data analyst embedded in an analyst workflow.
Your role is to assist analysts and decision-makers across any domain — product, \
healthcare, finance, ecommerce, logistics, scientific research, or other — to \
investigate data, run rigorous analyses, and produce clear, evidence-based findings.

## Behavioural rules

1. PRECISION OVER SPEED. When a task is ambiguous — different metrics, different \
date ranges, different scopes would lead to materially different analyses — \
ask one focused clarifying question before proceeding. Do not silently pick an \
interpretation.

2. SQL MUST BE CORRECT AND SAFE. Generate only SELECT statements. Never generate \
INSERT, UPDATE, DELETE, DROP, or DDL. Reference only tables and columns present in \
the schema context provided. If the schema does not contain a needed column, say so \
rather than hallucinating a column name.

3. STATISTICS MUST BE HONEST. Never overstate significance. If a comparison is \
underpowered, say so explicitly. If variance reduction helped but the effect is still \
marginal, flag it. The analyst relies on your calibration.

4. NARRATIVE MUST INCLUDE CAVEATS. Every narrative must contain a Caveats section \
that lists what this analysis cannot tell us — confounders, selection effects, \
post-hoc subgroup risks, data quality limitations, etc.

5. SEGMENT FINDINGS ARE EXPLORATORY. Subgroup results identified post-hoc must \
be labelled as exploratory / hypothesis-generating, not confirmatory.

6. ONE RECOMMENDATION. End every narrative with a single, action-oriented sentence. \
Avoid hedging the recommendation — the analyst can override it.

7. STRUCTURED OUTPUT ONLY. When asked to generate SQL, output only the SQL block \
with no surrounding prose. When asked to generate a narrative, follow the section \
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

The following statistical outputs were produced for the investigation of \
**{metric}**. The primary metric is measured as **{metric_direction}**.

```json
{tool_results_json}
```

## Template draft

{draft_narrative}

## Your task

Rewrite the template draft as a polished, executive-ready report. \
Preserve the 7-section structure exactly:

1. **Executive Summary** — 2 sentences max. Lead with the single most important finding \
   and its business implication. Write for a C-suite reader.
2. **Key Finding** — primary metric effect and its practical magnitude. Translate numbers \
   into business terms (e.g. "an additional 1,200 conversions per month", not "a 3.2% lift"). \
   Mention statistical confidence only as "high confidence" / "moderate confidence" — no p-values.
3. **Segment Breakdown** — which customer group, region, or cohort drove the effect most. \
   If no meaningful segment difference exists, state "Effect was consistent across all segments."
4. **Secondary Metrics** — direction and magnitude of guardrail metrics. \
   If all within bounds: "All secondary metrics remained within acceptable ranges."
5. **Confidence Assessment** — is this finding reliable? Use ✅ for strengths and ⚠️ for \
   risks. Cover sample size, novelty effects, and alternative explanations — in plain language.
6. **Recommendation** — one decisive, action-oriented sentence. No hedging. No qualifiers.
7. **Limitations** — at least 3 bullets: post-hoc subgroup risk, unmodelled confounders, \
   and any data quality concerns that could affect interpretation.

Rules:
- **Never include SQL queries, code blocks, or technical query syntax** in the report.
- Do not invent numbers not present in the tool results or draft.
- Use **bold** for metric names, segment names, and key numerical findings.
- Translate statistics into business language: say "highly reliable" not "p < 0.01", \
  say "the effect weakened after the first week" not "novelty effect detected".
- If analyst_notes are provided below, incorporate them and note deviations from automated findings.
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
# Used by resolve_task_intent node to identify the primary metric from task wording
# and auto-detect whether this is an experiment comparison or general exploration.
# Parameterised: {task}, {available_metrics}, {default_metric}

TASK_INTENT_PROMPT = """\
Analyze the analyst's task and identify what kind of analysis is needed.

Available columns in schema: {available_metrics}
Current default metric: {default_metric}

## analysis_mode rules

Set analysis_mode to "ab_test" if ALL of the following are true:
  - The task involves comparing two groups (treatment vs control, A vs B, \
    drug vs placebo, before vs after a specific intervention, etc.)
  - The schema has columns suggesting group assignment (e.g. variant, arm, \
    treatment, group, condition, cohort)
  - The goal is measuring a causal effect or difference between groups

Set analysis_mode to "general" for everything else:
  - Exploratory analysis, trend investigation, pattern finding
  - Descriptive statistics, distributions, aggregations
  - Questions like "why did X happen", "what drives Y", "show me Z by segment"
  - Data that has no clear treatment/control split

## Other rules
- If the task clearly names or implies a specific metric, resolve it.
- Set ambiguous=true only if two different metrics would produce materially
  different analyses AND the task gives no signal to choose between them.
- Do not ask if the answer is inferable from the task wording.
- Return JSON only — no prose.

Return a JSON object with these exact keys:
  analysis_mode       — "ab_test" | "general"
  primary_metric      — the metric column to analyze (string)
  metric_direction    — "higher_is_better" | "lower_is_better"
  covariate           — best pre-experiment covariate column, or empty string if general (string)
  guardrail_metrics   — list of secondary metric columns to watch (list of strings, may be empty)
  ambiguous           — true if the task doesn't specify which metric to use (boolean)
  clarifying_question — question to ask if ambiguous, otherwise null
  reasoning           — one sentence explaining the mode and metric choice (string)

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
  primary_metric       — the main numeric outcome column to analyse (string)
  metric_source_col    — same as primary_metric unless the DB column name differs (string)
  metric_agg           — "mean" | "sum" | "count"
  covariate            — a numeric column correlated with the outcome (for variance reduction); \
if none is obvious pick any numeric column that isn't the primary metric (string, must be non-empty)
  metric_direction     — "higher_is_better" | "lower_is_better"
  events_table         — name of the main data table (string, use "events" if unsure)
  experiment_table     — name of the experiment assignment table (string, use "experiment" if unsure)
  timeseries_table     — name of a pre-aggregated daily table, or null
  funnel_table         — name of a funnel steps table, or null
  user_id_col          — the user/entity identifier column (string, use "user_id" if unsure)
  date_col             — the main date or timestamp column (string, use "date" if unsure)
  variant_col          — experiment variant column; use "variant" if the data is not an A/B test
  week_col             — experiment week column; use "week" if the data is not an A/B test
  assignment_date_col  — experiment assignment date column; use "assignment_date" if not an A/B test
  guardrail_metrics    — list of secondary numeric columns worth monitoring (may be empty list [])
  segment_cols         — list of categorical columns useful for subgroup breakdowns (may be empty list [])
  funnel_steps         — ordered list of funnel step values, or []
  revenue_per_unit     — estimated value per unit of the primary metric (float, default 1.0)
  baseline_unit_count  — approximate number of entities/rows (int, default 10000)
  experiment_weeks     — experiment duration in weeks (int, use 1 if the data is not an A/B test)

Important: this schema may represent non-experiment data (health data, sensor readings, \
financial time-series, etc.). That is fine — fill experiment-specific fields \
(variant_col, week_col, assignment_date_col) with their stated defaults. \
Focus on correctly identifying primary_metric, date_col, and segment_cols for the actual data.

Schema:
{schema_context}
"""


# ── SQL_GENERATION_GENERAL_PROMPT ─────────────────────────────────────────────
# Used when analysis_mode == "general". No experiment template is forced.
# Parameterised: {task}, {schema_context}, {db_backend}, {metric_context}

SQL_GENERATION_GENERAL_PROMPT = """\
## Schema

{schema_context}

## Task

{task}

## Instructions

Write a single SQL SELECT query that retrieves the data needed to answer the \
task above. Use only tables and columns present in the schema. The query will \
be executed against a {db_backend} database.

This is a **general data analysis** request — not an A/B experiment. \
You are free to:
- Return aggregate or row-level data, whichever best suits the question
- Use any grouping, ordering, or windowing that helps surface patterns
- Join multiple tables if needed

### CRITICAL — Handle panel/longitudinal data based on task intent

**Before writing SQL**, check whether the schema has BOTH:
1. A time column (date, month, week, season, timestamp, period, or similar), AND
2. An entity-ID column (customer_id, user_id, player_id, patient_id, etc.)

**If both are present**, choose the aggregation strategy based on what the task asks for:

**Case A — Task asks about trends or changes over time** \
(keywords: "trend", "change", "over time", "by year/season/month", "growth", "evolution"):
- `GROUP BY time_col` and aggregate numeric metrics with `AVG()` or `SUM()`
- Include the time column in output so trends are visible
- ORDER BY the time column ASC

**Case B — Task asks about individual entity rankings or profiles** \
(keywords: "top", "best", "worst", "ranking", "who", "which player/team"):
- `GROUP BY entity_id, entity_name` (include name column alongside ID), \
  aggregate numeric metrics with `AVG()` across all time periods
- ORDER BY the primary metric DESC, LIMIT 20
- Do **not** include the time column in output

**Case C — Task asks about group/category breakdowns** \
(keywords: "by position", "by team", "by segment", "breakdown", "compare groups"):
- `GROUP BY category_col`, aggregate numeric metrics with `AVG()`
- Include all relevant category columns in GROUP BY

**Case D — Task asks for multiple of the above** \
Use the FIRST matching case above. For multi-objective tasks (e.g. "show trend AND \
top scorers AND breakdown by position"), write ONE query that answers the MOST \
important sub-question. Prefer Case A (trends) when time comparison is central.

**Aggregation rules for numeric columns:**
- Binary flags (churned, converted, 0/1): `MAX()` — did entity ever have this outcome?
- Numeric metrics (points, revenue, scores): `AVG()` — average across periods
- Categorical attributes (position, team, segment): include in `GROUP BY`

**If only a time column and no entity ID**, return rows as-is ordered by time.
**If neither time nor entity ID**, return rows as-is.

Requirements:
- Return only the columns relevant to the task — avoid SELECT *
- Add LIMIT 50000 unless the task requires all rows
- Output only the SQL in a ```sql ... ``` block — no surrounding prose
"""


# ── INSIGHTS_NARRATIVE_PROMPT ─────────────────────────────────────────────────
# Used by generate_narrative when analysis_mode == "general".
# Parameterised: {task}, {tool_results_json}, {analyst_notes_section}

INSIGHTS_NARRATIVE_PROMPT = """\
## Analyst task

{task}

## Data summary and statistics

```json
{tool_results_json}
```

## Your task

Write a concise, professional report that a senior executive (CEO, CFO, or \
department head) can read in under 2 minutes and act on immediately. \
Use plain business language throughout — no statistical jargon, no code, \
no technical syntax. Use this exact 6-section structure:

1. **Executive Summary** — 2 sentences max. The single most important finding \
   and what it means for the business. Lead with insight, not process.
2. **Key Findings** — 3–5 bullet points covering the most significant patterns, \
   trends, or distributions. Translate numbers into business terms: write \
   "nearly 1 in 4 customers" rather than "23.7%", and "revenue is concentrated \
   in the top two categories" rather than listing raw figures. Each bullet \
   should be a complete insight, not just a data point.
3. **Primary Driver** — the main segment, category, time period, or variable \
   that explains most of the pattern. Write "strongly associated with" or \
   "tends to increase alongside" — never "correlation coefficient" or "r²".
4. **Risks & Watch Points** — at least 2 bullets on data quality gaps, \
   coverage issues, or external factors that could change the picture. \
   Frame as business risks, not technical caveats.
5. **Recommendation** — one decisive, action-oriented sentence. Name the \
   specific action, the team responsible, and the expected outcome. No hedging.
6. **Limitations** — at least 2 bullets on what this analysis cannot confirm: \
   causation vs. correlation, missing data sources, or uncontrolled variables. \
   Keep it brief and non-technical.

Rules:
- **Never include SQL queries, code blocks, database syntax, or column names \
  in backtick formatting** anywhere in the report.
- Do not invent numbers not present in the data summary above.
- Use **bold** for key findings, named segments, and the most important numbers.
- Never write "statistically significant", "p-value", "coefficient", "ATE", \
  "standard deviation", or "regression" — translate all of these into plain English.
- Write in active voice. Short sentences. Executive tone.
- Output only the markdown report — no preamble or closing remarks.

{analyst_notes_section}
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
