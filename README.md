# DataPilot

An agentic AI system that replicates the core workflow of a senior Product Data Scientist.

**Demo scenario:** DAU drop investigation on simulated Meta-flavored data. Ground truth is
baked into the dataset so correctness is fully verifiable. The eval harness scores 11 criteria
automatically — current score: **11/11 (100%)**.

---

## What it does

Given a natural-language task ("Why did DAU drop in the most recent experiment?"), DataPilot:

1. Generates SQL against your database and shows it for review
2. Runs the full analysis pipeline: decomposition → anomaly detection → CUPED experiment analysis → HTE → novelty detection → guardrails → funnel → forecast
3. Writes a PM-ready narrative with explicit caveats and a one-sentence recommendation
4. Logs every run to a memory store so future runs benefit from past corrections

The analyst reviews and can override at three checkpoints — nothing is sent forward without approval.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATAPILOT                                      │
│                                                                             │
│  ┌──────────────┐     ┌──────────────────────────────────────────────────┐ │
│  │   Streamlit  │     │              LangGraph Agent                     │ │
│  │    ui/app.py │────►│                                                  │ │
│  │              │     │  START                                           │ │
│  │  Gate 1: SQL │◄────│  └─► check_semantic_cache                       │ │
│  │  Gate 2: Stats│    │       ├─ hit  ─► semantic_cache_gate 🛑         │ │
│  │  Gate 3: Narr│    │       └─ miss ─► inject_history                 │ │
│  │              │     │                  └─► load_schema                │ │
│  │  sidebar:    │     │                       └─► infer_metric_config   │ │
│  │  · past runs │     │                            └─► generate_sql     │ │
│  │  · cost saved│     │                                 └─► query_gate 🛑│ │
│  │  · DB picker │     │                                      └─► execute_query│ │
│  └──────────────┘     │                                           │      │ │
│                        │              ┌────────────────────────────┘      │ │
│                        │              ▼                                   │ │
│  ┌──────────────┐     │  load_auxiliary_data                             │ │
│  │   tools/     │     │  ├─► decompose_metric   (new/retained/resurrected)│ │
│  │              │◄────│  ├─► detect_anomaly     (zscore + slice_and_dice)│ │
│  │ stats_tools  │     │  ├─► forecast_baseline  (Prophet / rolling mean) │ │
│  │ decomp_tools │     │  ├─► run_cuped          (variance reduction)     │ │
│  │ anomaly_tools│     │  ├─► run_ttest          (Welch t-test)           │ │
│  │ forecast_tool│     │  ├─► run_hte            (subgroup t-tests)       │ │
│  │ guardrail_t  │     │  ├─► detect_novelty     (week1 vs week2 ATE)     │ │
│  │ novelty_tools│     │  ├─► compute_mde        (power + biz impact)     │ │
│  │ mde_tools    │     │  ├─► check_guardrails   (secondary metrics)      │ │
│  │ funnel_tools │     │  └─► compute_funnel     (conditional step rates) │ │
│  │ narrative_t  │     │                 │                                 │ │
│  └──────────────┘     │                 └─► analysis_gate 🛑             │ │
│                        │                      └─► generate_narrative      │ │
│  ┌──────────────┐     │                           └─► narrative_gate 🛑  │ │
│  │   memory/    │     │                                └─► log_run ──► END│ │
│  │              │◄────│                                                  │ │
│  │ store.py     │     └──────────────────────────────────────────────────┘ │
│  │ retriever.py │                                                           │
│  │ semantic_    │     🛑 = HITL interrupt — analyst approves or overrides  │
│  │   cache.py   │                                                           │
│  └──────────────┘                                                           │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│  │  tools/      │     │  agents/     │     │  config/     │               │
│  │  db_tools.py │     │  state.py    │     │  analysis_   │               │
│  │  DuckDB ─┐  │     │  AgentState  │     │  config.py   │               │
│  │  Postgres─┘  │     │  TypedDict   │     │  MetricConfig│               │
│  └──────────────┘     └──────────────┘     └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Caching — three layers

```
Request
  │
  ▼
Layer 1: Semantic cache (SQLite + MiniLM embeddings)
  similarity > 0.92 → return cached result, skip entire graph   ← zero API cost
  similarity 0.80–0.92 → show cached result, ask analyst        ← analyst decides
  similarity < 0.80 → cache miss, run normally
  │
  ▼
Layer 2: Prompt caching (Anthropic native cache_control)
  [STATIC — always cached]                [DYNAMIC — never cached]
  system prompt                           task string
  schema context               +          run-specific data
  history injection prefix
                                                                  ← 90% token cost reduction on hits
  │
  ▼
Layer 3: KV prefix reuse (within session)
  Same compiled prefix reused across all API calls in a session  ← automatic, no config needed
```

### Self-improvement loop

```
Run N                              Memory store (SQLite)
  │                                       │
  ├─ analyst edits SQL        ──────────► analyst_override{"sql_edited": true}
  ├─ analyst adds notes       ──────────► analyst_override{"analysis_notes": "..."}
  ├─ analyst revises narrative──────────► analyst_override{"narrative_notes": "..."}
  └─ completeness score       ──────────► eval_score (0–1, auto-computed)
                                          │
Run N+1                                   │
  └─ inject_history ◄───────────────────-┘
       │
       └─ "ANALYST CORRECTED SQL — double-check JOINs"
          "ANALYST NOTED: '...' — apply unless task clearly differs"
          "ANALYST OVERRODE RECOMMENDATION: '...'"
```

---

## Analyst skillset covered

| Capability | Tool | Ground truth verifiable? |
|---|---|---|
| Metric decomposition (new / retained / resurrected / churned) | `decomposition_tools` | ✅ new_users drives drop |
| Anomaly detection + slice-and-dice | `anomaly_tools` | ✅ android ranks first |
| CUPED variance reduction | `stats_tools.run_cuped` | ✅ >15% reduction |
| T-test significance | `stats_tools.run_ttest` | ✅ p < 0.05 |
| HTE subgroup analysis | `stats_tools.run_hte` | ✅ platform=android, user_segment=new |
| Novelty effect detection | `novelty_tools` | ✅ effect growing, not decaying |
| Guardrail metric monitoring | `guardrail_tools` | ✅ notif_optout breached |
| Funnel drop-off analysis | `funnel_tools` | ✅ d1_retain worsens for android/new |
| Forecast baseline (Prophet) | `forecast_tools` | ✅ actuals outside CI |
| MDE + business impact | `mde_tools` | ✅ ~3% MDE, near observed effect |
| PM-ready narrative | `narrative_tools` + LLM | ✅ mentions android, new, caveats |

---

## Repo structure

```
datapilot/
├── agents/
│   ├── state.py                # AgentState TypedDict — single contract between nodes
│   └── analyze/
│       ├── graph.py            # LangGraph graph — 21 nodes, 3 HITL gates
│       ├── nodes.py            # node functions (pure: call tools, no inline logic)
│       └── prompts.py          # all prompt templates as module-level constants
├── tools/                      # pure Python, no LangGraph/Streamlit deps
│   ├── db_tools.py             # DuckDB + Postgres unified interface
│   ├── stats_tools.py          # CUPED, t-test, HTE
│   ├── decomposition_tools.py  # DAU component breakdown
│   ├── anomaly_tools.py        # zscore anomaly + slice-and-dice
│   ├── forecast_tools.py       # Prophet (rolling mean fallback)
│   ├── guardrail_tools.py      # secondary metric sweep
│   ├── novelty_tools.py        # week-over-week ATE decay
│   ├── mde_tools.py            # MDE + business impact statement
│   ├── funnel_tools.py         # conditional step conversion rates
│   └── narrative_tools.py      # structured PM narrative formatter
├── memory/
│   ├── store.py                # SQLite run logger with cost tracking
│   ├── retriever.py            # keyword-overlap history retrieval
│   └── semantic_cache.py       # MiniLM embeddings + SQLite cache
├── ui/
│   ├── app.py                  # Streamlit frontend (renders only, zero agent logic)
│   ├── auth_page.py            # sign-in / sign-up
│   ├── db_connect.py           # database connection + MetricConfig sidebar
│   └── report_export.py        # PDF export via fpdf2
├── config/
│   ├── analysis_config.py      # MetricConfig Pydantic model
│   ├── metric_config.json      # default DAU drop config
│   └── examples/               # preset configs for other scenarios
├── auth/
│   └── store.py                # user auth (SQLite, bcrypt)
├── data/
│   └── generate_data.py        # deterministic synthetic dataset (seed=42)
├── evals/
│   └── analyze_eval.py         # 11-criterion offline eval harness
├── tests/                      # 52 unit tests, all passing
├── .env.example
├── Makefile
└── requirements.txt
```

---

## Quick start

```bash
# 1. Clone and install
git clone <repo>
cd datapilot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# 3. Generate the demo dataset
make data

# 4. Run the eval (no API key needed for 9/11 criteria)
make eval

# 5. Start the app
make app
# → http://localhost:8501
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for SQL generation and narrative |
| `MODEL` | `claude-sonnet-4-20250514` | Anthropic model ID |
| `DUCKDB_PATH` | `data/dau_experiment.db` | Path to DuckDB file |
| `MEMORY_DB_PATH` | `memory/datapilot_memory.db` | SQLite memory store |
| `FORECAST_BACKEND` | `prophet` | `prophet` or `rolling_mean` |
| `REVENUE_PER_DAU` | `0.50` | USD per DAU for MDE business impact |
| `BASELINE_DAU` | `500000` | Scale denominator for business impact |
| `SEMANTIC_CACHE_HARD_THRESHOLD` | `0.92` | Above this: skip API, return cached |
| `SEMANTIC_CACHE_SOFT_THRESHOLD` | `0.80` | Above this: show cached, ask analyst |
| `LANGFUSE_HOST` | — | Optional: Langfuse tracing endpoint |

---

## Eval harness

```bash
make eval          # skip narrative (no API key needed) — scores 9 of 11 criteria
make eval-full     # all 11 criteria including LLM narrative
```

Current scores:

```
  PASS  hte_correct_segment          android/new surfaces as top HTE segment
  PASS  cuped_variance_reduced       >15% variance reduction
  PASS  ttest_significant            p < 0.05
  PASS  decomp_identifies_new        new_users is dominant declining component
  PASS  slice_ranks_android_first    slice-and-dice ranks android #1
  PASS  forecast_flags_drop          actuals outside Prophet CI
  PASS  guardrails_breached_found    at least one guardrail breached
  PASS  optout_breached              notif_optout specifically flagged
  PASS  novelty_ruled_out            effect growing, not decaying
  PASS  narrative_mentions_segment   narrative mentions android + new
  PASS  narrative_has_caveats        caveats section present

Score: 11/11 = 100%  ✅
```

Eval scores are written back to the memory store after each run, so the self-improvement loop
has ground-truth signal for the demo scenario.

---

## Key design rules

**Rule 1 — Agents read/write state. Tools compute.**
No stats, SQL, or string formatting inside node functions. If it's logic, it lives in `tools/`.

**Rule 2 — HITL via `interrupt()` only.**
Never `input()`, never Streamlit polling. LangGraph `interrupt()` + `Command(resume=...)` keeps
the graph serializable.

**Rule 3 — State is the contract.**
All data between nodes lives in `AgentState`. Nodes never call each other.

**Rule 4 — Streamlit renders, agents decide.**
`ui/app.py` only calls `graph.invoke()` and reads from state. Zero stats, zero SQL.

**Rule 5 — Every run gets logged.**
`memory/store.py` captures task, overrides, eval score, token costs, and quality signal
on every completed run.

---

## Stack

| Layer | Library | Version |
|---|---|---|
| LLM | Anthropic Claude | `anthropic` 0.86 |
| Agent graph | LangGraph | 1.1.2 |
| UI | Streamlit | 1.55 |
| Database | DuckDB | 1.5 |
| Forecasting | Prophet | 1.3 |
| Semantic cache | sentence-transformers (MiniLM) | 5.3 |
| Stats | scipy, numpy | — |
| PDF export | fpdf2 | 2.8.7 |
| Memory | SQLite (stdlib) | — |
