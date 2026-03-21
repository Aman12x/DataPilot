# DataPilot — AI Product Data Analyst
## CLAUDE.md · Project context for Claude Code

---

## What this project is

DataPilot is an agentic AI system that replicates the core workflow of a senior Product Data Scientist:
query data → decompose metrics → detect anomalies → run experiments → analyze results → check guardrails → generate PM-ready recommendations.

The system is human-in-the-loop by design: at each major decision gate, the analyst (Aman) reviews,
overrides, or approves before the pipeline continues. Every run is logged to a memory store so the
system improves its defaults over time.

This is a portfolio project targeting Meta Product DS roles. The demo scenario is a **DAU drop
investigation** on simulated Meta-flavored data. The ground truth cause is known (built into the
data generator), so correctness is verifiable.

The analyst skillset this system covers:
- Metric decomposition (DAU components: new, retained, resurrected, churned)
- Anomaly detection on metric time series with automated slice-and-dice
- Experiment analysis: CUPED variance reduction, t-test, HTE subgroup analysis
- Novelty effect detection (week-over-week treatment effect decay)
- Guardrail metric monitoring (did we break something else?)
- Funnel analysis (drop-off across conversion steps)
- Forecasting baseline (is the drop vs expected or just vs last week?)
- MDE / power reasoning tied to business impact
- PM-ready narrative with explicit caveats and one-sentence recommendation

Not yet included (future modules): observational causal inference (DiD, RDD, PSM).

---

## Repo structure

```
datapilot/
├── CLAUDE.md                  # this file
├── data/
│   ├── generate_data.py       # synthetic DAU drop dataset with known ground truth
│   └── dau_experiment.db      # DuckDB database (gitignored, generated at runtime)
├── agents/
│   ├── state.py               # AgentState TypedDict — single source of truth
│   ├── orchestrator.py        # top-level LangGraph graph, routes to modules
│   └── analyze/
│       ├── graph.py           # Analyze module LangGraph graph
│       ├── nodes.py           # individual node functions (pure, no side effects)
│       └── prompts.py         # all system/user prompt templates
├── tools/
│   ├── db_tools.py            # unified DB layer: DuckDB + Postgres (see Database connectivity)
│   ├── stats_tools.py         # t-test, CUPED, HTE (pure Python, no LangGraph deps)
│   ├── decomposition_tools.py # DAU component breakdown: new/retained/resurrected/churned
│   ├── anomaly_tools.py       # time series anomaly detection + slice-and-dice
│   ├── funnel_tools.py        # conversion funnel drop-off analysis
│   ├── forecast_tools.py      # Prophet/rolling baseline for contextualizing drops
│   ├── guardrail_tools.py     # automated check of secondary/guardrail metrics
│   ├── novelty_tools.py       # week-over-week treatment effect decay detection
│   ├── mde_tools.py           # MDE calculation tied to business impact estimate
│   └── narrative_tools.py     # structured finding → PM narrative formatter
├── memory/
│   ├── store.py               # SQLite logger: every run, override, eval score
│   ├── retriever.py           # query past runs for relevant history injection
│   ├── semantic_cache.py      # embedding-based cache: skip LLM on near-identical tasks
│   └── schema_cache.json      # persisted schema snapshot — delete to force refresh
├── ui/
│   ├── app.py                 # Streamlit frontend — renders only, zero agent logic
│   └── db_connect.py          # Streamlit page: database connection setup UI
├── evals/
│   └── analyze_eval.py        # offline eval: does CUPED surface the right segment?
├── tests/
│   ├── test_stats_tools.py
│   ├── test_decomposition_tools.py
│   ├── test_anomaly_tools.py
│   ├── test_guardrail_tools.py
│   ├── test_novelty_tools.py
│   ├── test_forecast_tools.py
│   ├── test_db_tools.py
│   └── test_memory_store.py
├── .env.example
└── requirements.txt
```

---

## Core architecture rules

**Rule 1: Agents read/write state. Tools compute.**
Node functions in `agents/` may only call tools from `tools/`. They must not contain
statistical logic, SQL, or string formatting inline. If you find yourself writing
`scipy.stats` inside a node, stop and put it in `tools/stats_tools.py`.

**Rule 2: HITL via LangGraph `interrupt()` only.**
Never use `input()`, `st.text_input` polled in a loop, or threading hacks for human approval.
Use `interrupt(payload)` to pause the graph. Resume with `graph.invoke(Command(resume=value))`.
This keeps the graph serializable and the demo clean.

**Rule 3: State is the contract between nodes.**
All data passed between nodes lives in `AgentState`. Nodes never call each other directly.
If a node needs something, it must be in state. Add fields to state, not function args.

**Rule 4: Streamlit renders, agents decide.**
`ui/app.py` only reads from state or calls `graph.invoke()`. It never runs stats, formats
narratives, or calls the LLM directly. The hard boundary is: if it's logic, it's in `agents/`
or `tools/`. If it's display, it's in `ui/`.

**Rule 5: Every run gets logged.**
`memory/store.py` must be called at the end of every successful graph run. Log: task, params
chosen by agent, analyst overrides (if any), eval score (did it surface the right answer),
timestamp. This is the self-improvement substrate.

**Rule 6: Ask before assuming on ambiguous tasks.**
If the task string is ambiguous enough that two reasonable interpretations would produce
materially different analyses — different metric, different table, different experiment scope —
stop and ask one focused clarifying question before proceeding. Do not silently pick an
interpretation and bury it in a log. Surface it. Examples that require a question:
"investigate the drop" (which metric? which date range?), "analyze the experiment" (which
variant is treatment?), "check if this worked" (define success metric). Examples that do NOT
require a question: minor SQL ambiguity resolvable from schema, choice of covariate when
session_count is the obvious default, alpha=0.05 as a standard default.

**Rule 7: Schema is loaded once and cached — never re-fetched mid-run.**
On first run, `load_schema` node calls `db_tools.inspect_schema()` and writes the result
to `AgentState.schema_context`. It also persists the schema snapshot to
`memory/schema_cache.json`. On all subsequent runs, `load_schema` reads from the cache file
first — only re-fetches from the database if the cache is missing or if the task explicitly
says "schema changed" or "refresh schema". This means Claude Code does not need to be
re-briefed on the schema between sessions. If you add a new table or column, delete
`memory/schema_cache.json` to force a refresh.

`inspect_schema()` must return a structured string in this exact format:
```
TABLE: events
  user_id         STRING    -- unique user identifier
  date            DATE      -- event date
  platform        STRING    -- 'android' | 'ios' | 'web'
  ...
```
Include inline comments for every column — they are the primary way the LLM understands
what each column means without asking.

**Rule 8: Check semantic cache before every LLM call.**
Before invoking the Anthropic API for SQL generation or narrative generation, call
`memory/semantic_cache.py:check_cache(task, node_name)`. If similarity > 0.92, return the
cached result and skip the API call entirely. If similarity is between 0.80 and 0.92,
surface the cached result to the analyst at the HITL gate with a note: "Similar analysis
found — using cached result. Override?" On a cache miss, run normally and store the result.

**Rule 9: Apply prompt caching to all static content.**
Every Anthropic API call must mark static content with `cache_control: {"type": "ephemeral"}`.
Static content = anything that doesn't change between runs: system prompt, schema context,
tool definitions, history injection prefix. Dynamic content = the task string and any
run-specific data. Cached tokens cost 10% of normal input price. Cache TTL is 5 minutes and
resets on each hit, so long-running sessions benefit automatically. See Caching section for
exact message construction order.

**Rule 10: Never connect to a user's database without explicit confirmation.**
When a user provides Postgres credentials in the UI, display a confirmation dialog showing
the host, port, and database name before any connection attempt. Never store raw credentials
in state or logs — store only the connection string in `st.session_state` for the duration
of the session. On session end, credentials are gone.

---

## Caching strategy

DataPilot uses three caching layers in order of cost savings:

### Layer 1: Semantic cache (zero API cost)

Implemented in `memory/semantic_cache.py`. Uses `sentence-transformers/all-MiniLM-L6-v2`
(local, no API cost) to embed task strings and compare against past runs in the memory store.

```python
class SemanticCache:
    def check_cache(self, task: str, node_name: str) -> dict | None:
        # Embed task with MiniLM, query SQLite for stored embeddings
        # If cosine_similarity > 0.92: return cached result (full cache hit)
        # If 0.80-0.92: return cached result with hit_type='soft' for analyst review
        # If < 0.80: return None (cache miss, proceed to API)

    def store(self, task: str, node_name: str, result: dict) -> None:
        # Embed and store after a successful API call
        # stored in memory store runs table: task_embedding BLOB column
```

Cache is keyed by (task_embedding, node_name) — so SQL generation and narrative generation
have separate caches. A near-identical DAU drop question won't return a cached narrative
for a retention question.

Threshold rationale:
- 0.92+ = same question rephrased ("why did DAU drop" vs "investigate DAU decline") → safe to
  return directly
- 0.80-0.92 = related question ("why did DAU drop on Android" vs "why did DAU drop") → show
  analyst the cached result but require explicit approval before skipping the API call
- <0.80 = different enough to warrant fresh analysis

### Layer 2: Prompt caching (90% token cost reduction on cache hits)

Anthropic's native caching via `cache_control`. Applied in `agents/analyze/nodes.py` on
every API call. The message array must be constructed in this exact order so the cache
prefix is as long as possible:

```python
messages = [
    # Block 1 — ALWAYS STATIC, always cached
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,           # never changes
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": schema_context,          # changes only on schema refresh
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": history_injection,       # changes slowly (new runs accumulate)
                "cache_control": {"type": "ephemeral"}
            },
        ]
    },
    # Block 2 — DYNAMIC, never cached
    {
        "role": "user",
        "content": task_string                   # changes every run
    }
]
```

Rules for maintaining cache hit rate:
- Never concatenate dynamic content into the static blocks. Even one changed character
  breaks the cache prefix for everything after it.
- The schema block must be identical across calls — no timestamps, no run IDs embedded in it.
- history_injection should be deterministically ordered (sort by timestamp DESC, take top 3)
  so the same history produces the same string.
- Log cache hit/miss per call using the `cache_read_input_tokens` field in the API response.
  Surface cumulative savings in the Streamlit sidebar.

### Layer 3: Context caching (stable prefix reuse within a session)

Within a single Streamlit session, the system prompt + schema block does not change. By
constructing messages with static content first (as above), Anthropic's KV cache naturally
reuses the prefix across all API calls in the session without any extra configuration.

For multi-turn nodes (e.g. narrative refinement after analyst edits), maintain a
`conversation_history` list in state and always prepend the static blocks at position 0.
Never re-order or shuffle the message array mid-session — it breaks the cache prefix.

```python
# In nodes.py — narrative refinement example
def refine_narrative(state: AgentState) -> AgentState:
    messages = build_cached_prefix(state)   # always the same static blocks first
    messages += state.get("conversation_history", [])  # prior turns
    messages.append({"role": "user", "content": state["analyst_notes"]})  # new turn
    response = call_anthropic(messages)
    # append assistant response to conversation_history for next turn
    ...
```

### Cache cost tracking

Add to `memory/store.py` runs table:

```sql
cache_read_tokens    INTEGER,   -- from response.usage.cache_read_input_tokens
cache_write_tokens   INTEGER,   -- from response.usage.cache_creation_input_tokens
uncached_tokens      INTEGER,   -- from response.usage.input_tokens
semantic_cache_hits  INTEGER,   -- number of API calls skipped via semantic cache
estimated_cost_usd   REAL       -- compute from token counts + Sonnet pricing
```

Show cumulative cost saved in Streamlit sidebar: "Saved ~$X.XX this session via caching."

---

## Database connectivity

Users can connect either the built-in DuckDB demo dataset or their own Postgres database.
All database interaction goes through `tools/db_tools.py` — a unified interface that abstracts
the backend.

### Unified interface

```python
class DBConnection:
    def __init__(self, backend: str, **kwargs):
        # backend: 'duckdb' | 'postgres'
        # kwargs for duckdb: path (str)
        # kwargs for postgres: host, port, dbname, user, password, sslmode

    def query(self, sql: str) -> pd.DataFrame:
        # Execute SQL, return DataFrame
        # For DuckDB: use duckdb.connect(path).execute(sql).df()
        # For Postgres: use psycopg2 or sqlalchemy, return pd.read_sql()

    def inspect_schema(self) -> str:
        # Return formatted schema string (see Rule 7 format)
        # For DuckDB: PRAGMA table_info + SHOW TABLES
        # For Postgres: query information_schema.columns
        # Always append inline comments if known (from schema_cache.json annotations)

    def test_connection(self) -> dict:
        # Returns: {success: bool, error: str | None, table_count: int}
        # Used by UI before saving connection to session state
```

`AgentState` gets one new field: `db_backend: str` — set at session start, read by all nodes
that need to query data. Never hardcode 'duckdb' inside a node.

### Postgres connection UI

`ui/db_connect.py` is a Streamlit sidebar expander (not a separate page):

```
[ ] Use built-in demo dataset (DuckDB)         ← default
[ ] Connect my Postgres database

  Host:     [________________]
  Port:     [5432           ]
  Database: [________________]
  User:     [________________]
  Password: [****************]
  SSL mode: [require ▼      ]

  [ Test connection ]   → shows "Connected: 14 tables found" or error
  [ Use this database ]
```

On "Use this database": call `db_tools.DBConnection.test_connection()`. If success, store
the `DBConnection` object in `st.session_state.db_conn` and trigger a schema refresh
(delete `schema_cache.json`, set `schema_needs_refresh=True` in session state).

On session end or page refresh: `st.session_state` is cleared — credentials are not persisted
to disk under any circumstances. No `.env` writes, no SQLite writes of credentials.

### Schema annotation for external databases

When a user connects their own Postgres DB, the auto-inspected schema will have no inline
comments. After schema inspection, run one LLM call (prompt-cached) to infer likely column
semantics from names and sample values, then write annotations back to `schema_cache.json`.
Ask the analyst to confirm or edit annotations before the first analysis run. This is a
one-time step per database connection.

---

## AgentState schema

Defined in `agents/state.py`. All fields optional with defaults where possible.

```python
class AgentState(TypedDict):
    # Input
    task: str                          # raw analyst/PM question
    relevant_history: list[dict]       # injected from memory store at run start
    db_backend: str                    # 'duckdb' | 'postgres'

    # Caching metadata
    semantic_cache_hit: bool           # True if this run was served from semantic cache
    semantic_cache_similarity: float   # similarity score of the cache hit
    cache_read_tokens: int             # from Anthropic API response
    cache_write_tokens: int            # from Anthropic API response

    # Query phase
    schema_context: str                # table names + columns from DB
    generated_sql: str                 # SQL produced by agent
    query_result: pd.DataFrame         # raw result

    # HITL gate 1: query confirmation
    query_approved: bool

    # Decomposition + anomaly phase (pre-experiment context)
    decomposition_result: dict         # {new, retained, resurrected, churned, dominant_change_component}
    anomaly_result: dict               # {anomaly_dates, severity, direction}
    slice_result: dict                 # {ranked_dimensions} — which slice drives the drop
    forecast_result: dict              # {forecast_df, actual_vs_forecast_delta, outside_ci}

    # Experiment analysis phase
    metric: str                        # e.g. "dau", "d7_retention"
    covariate: str                     # for CUPED pre-experiment covariate
    cuped_result: dict                 # {raw_ate, cuped_ate, variance_reduction_pct, theta}
    ttest_result: dict                 # {t_stat, p_value, ci_lower, ci_upper, significant}
    hte_result: dict                   # {top_segment, effect_size, segment_share, all_segments}
    novelty_result: dict               # {week1_ate, week2_ate, effect_direction, novelty_likely}
    mde_result: dict                   # {mde_absolute, mde_relative_pct, is_powered_for_observed_effect}
    business_impact: str               # human-readable MDE → revenue statement

    # Guardrail phase
    guardrail_result: dict             # {guardrails: list, any_breached, breached_count}

    # Funnel phase
    funnel_result: dict                # {steps, biggest_dropoff_step}

    # HITL gate 2: analysis validation
    analysis_approved: bool
    analyst_notes: str                 # free-text override/annotation from analyst
    conversation_history: list[dict]   # for multi-turn narrative refinement

    # Narrative phase
    narrative_draft: str               # PM-ready markdown writeup
    recommendation: str                # one-sentence action recommendation

    # HITL gate 3: narrative sign-off
    narrative_approved: bool
    final_narrative: str

    # Memory
    run_id: str
    eval_score: float                  # 0-1, did the system surface the right answer?
```

---

## Data contract

Simulated dataset lives in `data/generate_data.py`. Must be regeneratable deterministically
with a fixed seed. Schema:

```
events table:
  user_id           STRING
  date              DATE
  platform          STRING    -- 'android' | 'ios' | 'web'
  user_segment      STRING    -- 'new' | 'returning' | 'power'
  is_new_user       INT       -- 1 if first 7 days since install
  dau_flag          INT       -- 1 if active that day
  session_count     INT
  notif_received    INT       -- notifications received that day
  notif_opened      INT
  notif_optout      INT       -- 1 if user opted out of notifications that day
  d7_retained       INT       -- 1 if user was active 7 days after first seen
  install_date      DATE      -- for cohort / funnel analysis

funnel table:                 -- one row per user per funnel step attempt
  user_id           STRING
  date              DATE
  step              STRING    -- 'impression' | 'click' | 'install' | 'd1_retain'
  completed         INT       -- 1 if step was completed

experiment table:
  user_id           STRING
  variant           STRING    -- 'control' | 'treatment'
  assignment_date   DATE
  week              INT       -- experiment week number (1 or 2), for novelty detection

metrics_daily table:          -- pre-aggregated daily metric snapshots for time series
  date              DATE
  platform          STRING
  user_segment      STRING
  dau               INT
  new_users         INT
  retained_users    INT       -- active today AND active in prior 28d window
  resurrected_users INT       -- active today, NOT active prior 28d, but active before that
  churned_users     INT       -- active 28d ago, not active today
  d7_retention_rate FLOAT
  notif_optout_rate FLOAT
  avg_session_count FLOAT
```

**Ground truth:** DAU drop is caused by a push notification bug that reduces `notif_opened`
for Android new_users in treatment. Effect size: ~8% DAU reduction in that segment.
All other segments: no effect. CUPED covariate: pre-experiment `session_count` (D-7 to D-1).
HTE should surface `platform=android AND user_segment=new` as the affected segment.

**Secondary ground truths for new tools:**
- Anomaly detection: `metrics_daily` has a visible step-down in DAU starting at experiment
  start date, concentrated in `platform=android`. Slice-and-dice should rank android highest.
- Guardrails: `notif_optout_rate` increases in treatment (users annoyed by broken notifs),
  `d7_retention_rate` also slightly decreases. Both should be flagged.
- Novelty effect: Treatment effect in week 2 is 1.5x week 1 (bug compounds, not novelty).
  Novelty tool should report "effect is growing, not decaying — not a novelty effect."
- Funnel: Drop-off worsens at `d1_retain` step for treatment android new_users.
- Forecast: A Prophet baseline fit on pre-experiment `metrics_daily` should show the
  experiment period DAU is below the lower confidence bound — confirming real signal.
- MDE: At observed sample size, MDE is ~3% DAU lift. Ground truth effect is ~8% in the
  affected segment (~20% of total users), so blended effect ~1.6% — near the MDE boundary,
  which is why CUPED variance reduction matters.

---

## Tools contracts

All functions in `tools/` are pure Python. No LangGraph, no Streamlit imports.
Each returns a typed dict. Raise `ValueError` with a human-readable message on bad input.

### db_tools.py

```python
class DBConnection:
    def __init__(self, backend: str, **kwargs): ...
    def query(self, sql: str) -> pd.DataFrame: ...
    def inspect_schema(self) -> str: ...
    def test_connection(self) -> dict: ...
```

### stats_tools.py

```python
def run_cuped(df, metric_col, covariate_col, variant_col) -> dict:
    # Returns: {raw_ate, cuped_ate, variance_reduction_pct, theta}

def run_ttest(control, treatment) -> dict:
    # Returns: {t_stat, p_value, ci_lower, ci_upper, significant}

def run_hte(df, metric_col, variant_col, segment_cols) -> dict:
    # Returns: {top_segment, effect_size, segment_share, all_segments: list[dict]}
    # Use manual subgroup t-tests; avoid heavy causal ML libraries for now
```

### decomposition_tools.py

```python
def decompose_dau(df, date_col, window_days=28) -> dict:
    # Returns: {new, retained, resurrected, churned} time series
    # + pct_of_dau for each + dominant_change_component
```

### anomaly_tools.py

```python
def detect_anomaly(df, metric_col, date_col, method="zscore") -> dict:
    # Returns: {anomaly_dates, severity, direction: 'drop'|'spike'}

def slice_and_dice(df, metric_col, date_col, dimension_cols) -> dict:
    # Returns: {ranked_dimensions: list[{dimension, value, contribution_pct, delta}]}
    # Ranks by absolute contribution to total metric change
```

### funnel_tools.py

```python
def compute_funnel(df, variant_col, steps=['impression','click','install','d1_retain']) -> dict:
    # Returns: {steps: list[{step, control_rate, treatment_rate, delta, pct_change}]}
    # + biggest_dropoff_step
```

### forecast_tools.py

```python
def forecast_baseline(df, metric_col, date_col, forecast_days) -> dict:
    # Returns: {forecast_df, actual_vs_forecast_delta, outside_ci: bool}
    # Use Prophet if available; fall back to 7-day rolling mean + 2-sigma CI
    # Never hard-fail — degrade gracefully and set warning in result
```

### guardrail_tools.py

```python
def check_guardrails(df, variant_col, guardrail_metrics, alpha=0.05) -> dict:
    # Returns: {guardrails: list[{metric, control_mean, treatment_mean, delta_pct,
    #           p_value, breached}], any_breached, breached_count}
    # Breached = p < alpha AND delta is in the harmful direction
```

### novelty_tools.py

```python
def detect_novelty_effect(df, metric_col, variant_col, week_col) -> dict:
    # Returns: {week1_ate, week2_ate,
    #           effect_direction: 'decaying'|'growing'|'stable',
    #           novelty_likely: bool}
    # novelty_likely=True only if decaying AND week2_ate < 0.5 * week1_ate
```

### mde_tools.py

```python
def compute_mde(n_control, n_treatment, baseline_mean, baseline_std,
                alpha=0.05, power=0.80) -> dict:
    # Returns: {mde_absolute, mde_relative_pct, is_powered_for_observed_effect}

def business_impact_statement(mde_relative_pct, metric, baseline_dau,
                               revenue_per_dau=0.50) -> str:
    # Returns: "At MDE of 2.1%, detects a lift worth ~$Xk/day at current scale."
```

---

## LangGraph graph structure (Analyze module)

```
START
  └─► check_semantic_cache       # check memory/semantic_cache.py — skip graph if hit > 0.92
  └─► inject_history             # query memory store, populate relevant_history
  └─► load_schema                # load from cache or re-inspect DB
  └─► generate_sql               # LLM call (prompt-cached prefix)
  └─► [INTERRUPT: query_gate]    # show SQL + preview → analyst approves/edits
  └─► execute_query              # run SQL via db_tools.DBConnection

  -- Pre-experiment context (runs in parallel where possible) --
  └─► decompose_metric           # DAU component breakdown
  └─► detect_anomaly             # time series anomaly + slice-and-dice
  └─► forecast_baseline          # Prophet/rolling baseline vs actuals

  -- Experiment analysis --
  └─► run_cuped                  # CUPED-adjusted ATE
  └─► run_ttest                  # significance test on adjusted metric
  └─► run_hte                    # subgroup analysis
  └─► detect_novelty             # week-over-week effect decay check
  └─► compute_mde                # MDE + business impact statement

  -- Secondary metric health --
  └─► check_guardrails           # automated guardrail metric sweep
  └─► compute_funnel             # funnel drop-off analysis

  └─► [INTERRUPT: analysis_gate] # show full stats summary → analyst validates
  └─► generate_narrative         # LLM call (prompt-cached prefix + conversation_history)
  └─► [INTERRUPT: narrative_gate] # show draft → analyst edits/approves
  └─► log_run                    # write to memory store + semantic cache + cost tracking
END
```

`check_semantic_cache` is the first node. On a hard hit (>0.92), interrupt immediately with
the cached result and ask the analyst: "This looks identical to a prior analysis. Use cached
result?" If they approve, skip to `log_run`. If they decline, continue normally.

Nodes from `decompose_metric` through `compute_funnel` can be run as a LangGraph `Send`
parallel fan-out if performance matters. For initial build, sequential is fine.

**Analysis gate payload** (what the analyst sees before approving):
```python
{
  "decomposition": decomposition_result,
  "top_anomaly_slice": slice_result["ranked_dimensions"][0],
  "forecast_outside_ci": forecast_result["outside_ci"],
  "cuped_variance_reduction": cuped_result["variance_reduction_pct"],
  "significant": ttest_result["significant"],
  "top_segment": hte_result["top_segment"],
  "novelty_likely": novelty_result["novelty_likely"],
  "guardrails_breached": guardrail_result["any_breached"],
  "breached_metrics": [g for g in guardrail_result["guardrails"] if g["breached"]],
  "biggest_funnel_dropoff": funnel_result["biggest_dropoff_step"],
  "mde_powered": mde_result["is_powered_for_observed_effect"],
  "business_impact": business_impact
}
```

---

## Prompts

All prompts live in `agents/analyze/prompts.py` as module-level constants. No f-strings
inside node functions. Pass variables via `.format()` at call time.

Key prompts needed:
- `SYSTEM_PROMPT` — role, behavior rules, output format constraints (static, always cached)
- `SQL_GENERATION_PROMPT` — task + schema → SQL
- `NARRATIVE_PROMPT` — all tool results + analyst_notes → PM writeup (must include caveats)
- `HISTORY_INJECTION_PREFIX` — prepended when relevant_history is non-empty
- `SCHEMA_ANNOTATION_PROMPT` — used once per new external DB to infer column semantics

The narrative prompt must instruct the LLM to structure output as:
1. TL;DR (2 sentences max)
2. What we found (decomposition + anomaly + experiment results)
3. Where it's concentrated (HTE + funnel)
4. What else is affected (guardrails)
5. Confidence level (MDE powered? novelty ruled out? forecast confirms?)
6. Recommendation (one sentence, action-oriented)
7. Caveats (what this analysis can't tell us)

---

## Streamlit UI structure

`ui/app.py` manages a session state machine mirroring the LangGraph state:

```
st.session_state.graph_thread      # persists graph execution across reruns
st.session_state.current_gate      # None | 'semantic_cache' | 'query' | 'analysis' | 'narrative'
st.session_state.run_history       # list of completed run summaries for sidebar
st.session_state.db_conn           # DBConnection object (cleared on session end)
st.session_state.session_cost_usd  # running API cost for this session
```

Layout:
- Sidebar top: database connection expander (DuckDB default / Postgres form)
- Sidebar mid: past runs from memory store, eval scores, self-improvement trend
- Sidebar bottom: "Session cost: $X.XX | Saved: $X.XX via caching"
- Main: active gate display (SQL preview / stats cards / narrative draft)
- Bottom: approve / edit / override controls

---

## Memory store schema

`memory/store.py` wraps a SQLite database.

```sql
CREATE TABLE runs (
    run_id               TEXT PRIMARY KEY,
    timestamp            TEXT,
    task                 TEXT,
    task_embedding       BLOB,        -- MiniLM embedding for semantic cache lookup
    metric               TEXT,
    covariate            TEXT,
    db_backend           TEXT,        -- 'duckdb' | 'postgres'
    analyst_override     TEXT,        -- JSON: what the analyst changed vs agent default
    top_segment          TEXT,
    eval_score           REAL,
    cache_read_tokens    INTEGER,
    cache_write_tokens   INTEGER,
    uncached_tokens      INTEGER,
    semantic_cache_hits  INTEGER,
    estimated_cost_usd   REAL,
    notes                TEXT
);
```

`memory/retriever.py` takes the current task string and returns the top-3 most relevant past
runs by keyword overlap. Injected into state as `relevant_history` before the graph runs.

---

## Self-improvement behavior

At graph start, `inject_history` node does:
1. Query SQLite for past runs where task keywords overlap current task
2. If found, prepend to system prompt: "In similar past analyses, you chose covariate X.
   The analyst overrode this to Y. Prefer Y unless there's a reason not to."
3. Log whether the agent adopted the historical preference or deviated (and why)

This means after 3-4 runs, the agent should stop making the same correctable mistakes.

---

## Eval harness

`evals/analyze_eval.py` runs the full graph on a fixed task with no HITL (auto-approve all
gates) and checks against known ground truth. Score: 0/1 per criterion, average = `eval_score`.

```python
EVAL_CRITERIA = {
    "hte_correct_segment":         lambda s: s["hte_result"]["top_segment"] == "platform=android,user_segment=new",
    "cuped_variance_reduced":      lambda s: s["cuped_result"]["variance_reduction_pct"] > 15,
    "ttest_significant":           lambda s: s["ttest_result"]["significant"] == True,
    "decomp_identifies_component": lambda s: "new" in s["decomposition_result"]["dominant_change_component"],
    "slice_ranks_android_first":   lambda s: s["slice_result"]["ranked_dimensions"][0]["value"] == "android",
    "forecast_flags_drop":         lambda s: s["forecast_result"]["outside_ci"] == True,
    "guardrails_breached_found":   lambda s: s["guardrail_result"]["any_breached"] == True,
    "optout_rate_flagged":         lambda s: any(g["metric"] == "notif_optout_rate" and g["breached"]
                                                for g in s["guardrail_result"]["guardrails"]),
    "novelty_correctly_ruled_out": lambda s: s["novelty_result"]["novelty_likely"] == False,
    "narrative_mentions_segment":  lambda s: "android" in s["narrative_draft"].lower()
                                          and "new" in s["narrative_draft"].lower(),
    "narrative_has_caveats":       lambda s: "caveat" in s["narrative_draft"].lower()
                                          or "limitation" in s["narrative_draft"].lower(),
}
```

Run after any change to prompts, tool logic, or graph structure. Target: >80% criteria passing.

---

## Environment

```
ANTHROPIC_API_KEY=
MODEL=claude-sonnet-4-20250514
DUCKDB_PATH=data/dau_experiment.db
MEMORY_DB_PATH=memory/datapilot_memory.db
FORECAST_BACKEND=prophet            # 'prophet' | 'rolling_mean' fallback
REVENUE_PER_DAU=0.50                # for MDE business impact statement, configurable
SEMANTIC_CACHE_HARD_THRESHOLD=0.92  # above this: skip API call entirely
SEMANTIC_CACHE_SOFT_THRESHOLD=0.80  # above this: show cached result for analyst approval
LOG_LEVEL=INFO
```

---

## What NOT to do

- Do not use `gpt-4` or any non-Anthropic model. All LLM calls use `MODEL` from env.
- Do not add vector DB (ChromaDB, Pinecone) — semantic cache uses SQLite + MiniLM embeddings.
- Do not build the orchestrator (multi-module router) until Analyze module has passing evals.
- Do not add auth, multi-user support, or cloud deployment to the UI. Local demo only.
- Do not mock the stats tools. They must run real scipy/numpy on the simulated data.
- Do not put the eval score calculation inside the agent. It's offline-only in `evals/`.
- Do not add observational causal inference (DiD, RDD, PSM) yet — that's a future module.
- Do not use EconML or CausalML for HTE — manual subgroup t-tests only for now.
- Do not let `forecast_tools.py` fail hard if Prophet is unavailable — fall back gracefully.
- Do not persist Postgres credentials to disk, SQLite, or logs under any circumstances.
- Do not put dynamic content (run IDs, timestamps, task strings) inside prompt-cached blocks.

---

## Current status

**Data layer**
- [ ] `data/generate_data.py` — all tables with ground truth baked in

**State**
- [ ] `agents/state.py` — full AgentState TypedDict

**Database layer**
- [ ] `tools/db_tools.py` — DBConnection: DuckDB + Postgres unified interface
- [ ] `ui/db_connect.py` — Streamlit connection form

**Tools (build and unit test each before wiring into graph)**
- [ ] `tools/stats_tools.py` — CUPED, t-test, HTE
- [ ] `tools/decomposition_tools.py` — DAU components
- [ ] `tools/anomaly_tools.py` — anomaly detection + slice-and-dice
- [ ] `tools/forecast_tools.py` — Prophet baseline
- [ ] `tools/guardrail_tools.py` — guardrail metric sweep
- [ ] `tools/novelty_tools.py` — week-over-week effect decay
- [ ] `tools/mde_tools.py` — MDE + business impact
- [ ] `tools/funnel_tools.py` — funnel drop-off
- [ ] `tools/narrative_tools.py` — narrative formatter

**Caching**
- [ ] `memory/semantic_cache.py` — MiniLM embeddings + SQLite lookup
- [ ] Prompt caching applied in all nodes (verify via cache_read_input_tokens in logs)
- [ ] Cost tracking in memory store + Streamlit sidebar display

**Agent**
- [ ] `agents/analyze/prompts.py`
- [ ] `agents/analyze/nodes.py`
- [ ] `agents/analyze/graph.py` — LangGraph graph with semantic cache check + 3 HITL interrupts

**Memory**
- [ ] `memory/store.py` + `memory/retriever.py` — updated schema with cost tracking cols
- [ ] `memory/schema_cache.json` — auto-generated on first run

**UI**
- [ ] `ui/app.py` — Streamlit with all gate rendering + sidebar history + cost display
- [ ] `ui/db_connect.py` — database connection form

**Evals**
- [ ] `evals/analyze_eval.py` — all 11 criteria, target >80%

**Demo**
- [ ] End-to-end run on DAU drop scenario, all ground truth criteria passing
- [ ] Semantic cache demo: run same task twice, second run shows "Saved $X via cache"

---

## Decision log

When you make a non-obvious implementation choice — where a reasonable alternative existed
and you picked one over the other — append a one-liner to `decisions.md` in this format:

```
[file] [function] — chose X over Y because Z
```

Examples:
```
tools/anomaly_tools.py detect_anomaly — chose zscore over IQR because IQR was too insensitive to the step-change pattern in fixture data
tools/forecast_tools.py forecast_baseline — chose 7-day rolling window over 14-day because 14-day smoothed over the pre-experiment signal
```

Do not log obvious defaults (alpha=0.05, seed=42). Only log decisions where someone reading
the code later would reasonably ask "why did they do it this way?"