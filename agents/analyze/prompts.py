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

Statistical outputs for **{metric}** (measured as **{metric_direction}**):

```json
{tool_results_json}
```

## Template draft

{draft_narrative}

## Your task

Rewrite the template draft as a numbers-forward report for business stakeholders. \
Short sentences. Active voice. No preamble.

The report has two parts separated by the exact marker `<!-- details -->` on its own line.

**Part 1 — Brief Report** (shown immediately to stakeholders):

1. **Result** - State the headline number in the first sentence. \
   Format: "[Treatment] [lifted/reduced] [metric] by [X] ([treatment value] vs [control value])." \
   Sentence 2: business translation (how many users, orders, or dollars this represents).
2. **What This Means** - 2-3 sentences on practical business impact. \
   Use absolute counts alongside percentages. State confidence as \
   "high confidence" or "moderate confidence" only. No p-values or test names.
3. **Recommendation** - one sentence. Name the action and the expected outcome. No hedging.

Then output this exact line:

<!-- details -->

**Part 2 — Additional Details** (shown only on request):

4. **Segment Breakdown** - bullet each key segment with its numbers. \
   Format: "[Segment]: [treatment value] vs [control value] ([delta])." \
   If no meaningful segment difference exists, write one sentence stating so.
5. **Secondary Metrics** - one bullet per guardrail metric with its direction and value. \
   Flag breached guardrails with "FLAGGED:" prefix. \
   If all within bounds: "All secondary metrics within acceptable ranges."
6. **Confidence** - use checkmarks for strengths, warning signs for risks. \
   Cover sample size, whether the effect persisted over time, and any data quality concerns. \
   Plain language only.
7. **Limitations** - at least 3 bullets: subgroup analysis was exploratory (not pre-registered), \
   uncontrolled external factors, and any data quality gaps.

Formatting rules:
- No em dashes (-- or -). Use a period or restructure the sentence instead.
- No SQL, no code blocks, no column names in backticks anywhere in the report.
- Do not invent numbers. Every number must appear in the tool results or draft.
- Bold the actual numbers and segment names, not adjectives.
- Never write "statistically significant", "p-value", "ATE", "CUPED", \
  "standard deviation", "t-test", or "regression". Translate all of these into plain English.
- If analyst_notes are provided, incorporate them and note any deviation from automated findings.
- Output only the markdown report. No preamble, no closing remarks.

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

Set analysis_mode to "power_analysis" if the task is asking about experiment DESIGN \
(before running the experiment):
  - How many users/participants/samples do I need?
  - How long should I run this experiment?
  - What effect size can I detect with N users?
  - Sample size calculation, power calculation, runtime estimation
  - "Is my experiment powered?", "what's the MDE for N users?"
  Also extract mde_target_pct: the MDE % mentioned in the task. If not stated, default to 5.0.

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
  analysis_mode       — "ab_test" | "general" | "power_analysis"
  primary_metric      — the metric column to analyze (string)
  metric_direction    — "higher_is_better" | "lower_is_better"
  covariate           — best pre-experiment covariate column, or empty string if general (string)
  guardrail_metrics   — list of secondary metric columns to watch (list of strings, may be empty)
  mde_target_pct      — target MDE % for power_analysis mode (float, default 5.0, ignored for other modes)
  ambiguous           — true if the task doesn't specify which metric to use (boolean)
  clarifying_question — question to ask if ambiguous, otherwise null
  reasoning           — one sentence explaining the mode and metric choice (string)

Task: {task}
"""


# ── POWER_ANALYSIS_NARRATIVE_PROMPT ───────────────────────────────────────────
# Used by generate_narrative when analysis_mode == "power_analysis".
# Parameterised: {task}, {power_result_json}, {analyst_notes_section}

POWER_ANALYSIS_NARRATIVE_PROMPT = """\
## Analyst task

{task}

## Power analysis results

```json
{power_result_json}
```

## Your task

Write a concise experiment design brief that a product manager or clinical lead \
can act on immediately. Lead with the headline number. No preamble.

The report has two parts separated by the exact marker `<!-- details -->` on its own line.

**Part 1 — Brief Report** (shown immediately to stakeholders):

1. **Sample Size** - Open with the required sample size per arm and total. \
   Format: "You need **N per arm** (**2N total**) to detect a **X%** lift in [metric]." \
   Sentence 2: estimated runtime at current traffic levels.
2. **What This Means** - 2-3 sentences. Translate numbers into business terms. \
   E.g. how many days or weeks, whether this is achievable, what daily traffic this assumes.
3. **Recommendation** - one sentence. Either "proceed with this design" or flag a blocker \
   (e.g. traffic too low, MDE too ambitious). Name the specific action.

Then output this exact line:

<!-- details -->

**Part 2 — Sensitivity Analysis** (shown only on request):

4. **Sensitivity Table** - Render the full sensitivity table as markdown. \
   Format: | MDE (%) | N per arm | Runtime (days) | \
   Include all rows from the sensitivity data. Bold the target MDE row.
5. **Statistical Assumptions** - bullet list: significance level (alpha), \
   statistical power, metric baseline mean and std, daily traffic estimate. \
   Plain language only — no test names or Greek letters.
6. **Guardrails to Watch** - list each guardrail metric and what to monitor. \
   If none are configured, write one sentence noting that guardrails should be defined.
7. **Limitations** - at least 2 bullets: \
   baseline estimates assume stable traffic and no seasonality; \
   binary metrics require different variance assumptions than rates; \
   CUPED variance reduction could reduce required N by up to 30%.

Formatting rules:
- No em dashes (-- or -). Use a period or restructure the sentence instead.
- No SQL, no code blocks, no column names in backtick formatting.
- Do not invent numbers. Every number must appear in the power analysis results above.
- Bold the actual numbers. Do not bold adjectives.
- Write in active voice. Short sentences.
- Output only the markdown report. No preamble, no closing remarks.

{analyst_notes_section}
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

---

### STEP 1 — Identify the aggregation strategy

Check whether the schema has BOTH a time column (date, month, week, season, \
timestamp, period) AND an entity-ID column (customer_id, user_id, carrier, etc.).

**Case A — Task asks about trends or changes over time** \
(keywords: "trend", "over time", "by month/season/year", "growth", "evolution"):
- GROUP BY the time column, aggregate numeric metrics with AVG() or SUM()
- Include the time column in SELECT, ORDER BY it ASC

**Case B — Task asks for group/category breakdowns or rankings** \
(keywords: "which carrier", "by segment", "breakdown", "compare groups", \
"worst", "best", "ranking", "root cause", "what drives"):
- GROUP BY every relevant categorical column the task mentions
- If the task says "does it vary by X and Y" → include BOTH X and Y in GROUP BY
- Always SELECT every column you GROUP BY

**Case C — Task asks for individual entity profiles** \
(keywords: "top N", "show me each", "list all"):
- GROUP BY entity_id (and name column if present)
- Aggregate numeric metrics with AVG() across all periods
- ORDER BY primary metric DESC, LIMIT 20

Priority when task matches multiple cases: **A > B > C**

**If only a time column and no entity ID**, return rows as-is ordered by time.
**If neither**, return rows as-is.

---

### STEP 2 — Apply these rules for every metric column

**Binary outcome columns (0/1 flags: churned, on_time_delivery, converted, etc.):**
- "rate", "percentage", "how often", "what share" → `AVG(col)` — gives the rate directly (0.0–1.0)
- "how many", "count of failures" → `SUM(col)` or `SUM(CASE WHEN col = 0 THEN 1 END)`
- Always add `COUNT(*) AS total_records` alongside any rate or count so volume is visible
- **Per-entity collapse** (one row per user/entity, not one row per group) → `MAX(col)` — did this entity ever have the outcome?
  - Only use MAX() when you are collapsing to entity level first, then grouping. Never use MAX() when computing a group-level rate.

**Continuous numeric columns (revenue, days, scores):**
- Use `AVG()` for typical value, `SUM()` for totals, `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)` for median

**Categorical columns:** include in GROUP BY, never aggregate with AVG/SUM

---

### STEP 3 — Check for these anti-patterns before writing SQL

**1. Never filter on the outcome metric you are measuring.**
If the task asks "which carriers have the worst on-time rate", do NOT add \
`WHERE on_time_delivery = 0`. That removes on-time records and makes every \
rate 0%. Include ALL records; let AVG() compute the rate across the full population.
Only add a WHERE filter if the task explicitly restricts to a subpopulation \
("among churned customers", "for delayed shipments only").

**2. Never report a count without a denominator.**
If you write `COUNT(*) AS delayed_shipments`, also write \
`SUM(total_shipments)` or include `AVG(outcome_flag)` so the rate is clear. \
A count with no total is uninterpretable.

**3. Always SELECT every column you GROUP BY.**
Never group by a column that is absent from your SELECT list.

**4. Rankings: match ORDER BY direction to the task.**
- "worst", "lowest", "least reliable" → ORDER BY rate ASC
- "best", "highest", "most" → ORDER BY rate DESC
- Always include both the rate and COUNT(*) so the reader can judge sample size.

**5. Never use UNPIVOT, PIVOT, or UNNEST to reshape columns into rows.**
Keep original column names as columns. Do not melt the table into \
(variable, value) pairs — this breaks downstream analysis.

**6. Never write metadata commands.**
Do not use DESCRIBE, SUMMARIZE, SHOW TABLES, SHOW COLUMNS, or PRAGMA \
statements. Write a real SELECT query.

**7. Never conflate a subpopulation with the full population.**
Filtering to `churned = 1` to study churn drivers is fine, but state it as \
a comment. Do not then imply the result represents all customers.

---

Requirements:
- Return only the columns relevant to the task — avoid SELECT *
- Add LIMIT 50000 unless the task requires all rows
- Output only the SQL in a ```sql ... ``` block — no surrounding prose
- **Never ask clarifying questions.** If the task is ambiguous, make a reasonable \
assumption and write the SQL. State any assumption as a single SQL comment at the top.
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

Write a numbers-forward report that a department head or executive can act on in \
under 2 minutes. Lead every section with the actual number. No preamble.

The report has two parts separated by the exact marker `<!-- details -->` on its own line.

**Part 1 — Brief Report** (shown immediately to stakeholders):

1. **Bottom Line** - 2 sentences max. Open with the most important number or trend \
   from the data. Sentence 2: what it means for the business.
2. **Key Findings** - 3-5 bullets. Each bullet must include a specific number from \
   the data. Format: "[What happened]: [number] ([context or comparison])." \
   Do not write a bullet without a number. Use plain counts and rates that \
   non-technical readers understand ("1 in 4 customers", "$2.3M revenue gap", \
   "down 18% quarter-over-quarter").
3. **Recommendation** - one sentence. Name the specific action, the team \
   responsible, and the expected outcome. No hedging. No qualifiers.

Then output this exact line:

<!-- details -->

**Part 2 — Additional Details** (shown only on request):

4. **What Drives It** - the main segment, category, or time period that explains \
   most of the pattern. State the size of the gap between top and bottom groups. \
   Avoid academic language.
5. **Watch Points** - at least 2 bullets. Flag data coverage gaps, recent trend \
   changes, or external factors that could change the picture. Frame each as a \
   business risk with a suggested check.
6. **Limitations** - at least 2 bullets on what this analysis cannot confirm: \
   association vs. causation, missing data sources, or uncontrolled variables. \
   One sentence each.

Formatting rules:
- No em dashes (-- or -). Use a period or restructure the sentence instead.
- No SQL, no code blocks, no column names in backtick formatting anywhere.
- Do not invent numbers. Every number must appear in the data summary above.
- Bold the actual numbers and named segments. Do not bold adjectives.
- Never write "statistically significant", "p-value", "coefficient", \
  "correlation", "r-squared", "standard deviation", or "regression". \
  Translate all of these into plain business language.
- Write in active voice. Short sentences.
- Output only the markdown report. No preamble, no closing remarks.

NUMERICAL ACCURACY — violations here are critical errors:
- When stating a difference or gap ("underperforms by X points", "X points higher"), \
  you MUST verify the arithmetic: the stated gap must equal the larger value minus the \
  smaller value. If 77.4% vs 77.0%, the gap is 0.4 points, not 12.
- When using directional words ("underperforms", "worse", "lower"), the number you cite \
  for that entity MUST actually be lower than the comparison value. If it is higher, \
  the correct word is "outperforms" or "higher".
- Every comparison sentence must state BOTH values explicitly: \
  "[Entity]: [value] vs [comparison]: [comparison value] ([computed gap])."
- Never blend a gap figure from one comparison with entity values from a different \
  comparison. Each sentence must use numbers that all come from the same row or group.

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
