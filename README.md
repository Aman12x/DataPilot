# DataPilot

An agentic AI data analyst that helps anyone — product, healthcare, finance, ecommerce, logistics — investigate their data and make better decisions.

Ask a question in plain English. DataPilot generates SQL, runs rigorous statistical analysis, and produces a clear evidence-based report. You review and approve at every step.

**Eval scores:** 11/11 on DAU experiment · 11/11 cross-domain generalisability (clinical trial + ecommerce A/B)

---

## What it does

Given a natural-language task, DataPilot:

1. **Understands your question** — auto-detects whether it's an experiment comparison or general exploration; asks one clarifying question if genuinely ambiguous
2. **Generates SQL** — against your database or uploaded file, shown for review before execution
3. **Runs the right analysis** — for experiments: CUPED, t-test, HTE, novelty detection, guardrails, funnel, forecast; for general: describe, correlations, trend analysis
4. **Writes a structured report** — TL;DR, findings, subgroup breakdown, confidence level, recommendation, caveats
5. **Remembers past runs** — future runs on similar tasks inherit analyst corrections and preferences

Nothing is sent forward without your approval. Every checkpoint is interruptible.

---

## Works across domains

Upload a CSV or connect a database. DataPilot infers the schema and adapts:

| Domain | Example task |
|--------|-------------|
| Product | "Did the new checkout flow increase revenue? Which devices benefited most?" |
| Healthcare | "Did Drug A improve recovery scores vs placebo? Check for side effects by age group." |
| SaaS | "What factors most strongly predict customer churn? Which plans churn fastest?" |
| Logistics | "Why are deliveries being delayed? Which carriers have the worst on-time rates?" |
| Finance | "Which customer segments have the highest default rates? Is the trend worsening?" |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATAPILOT                                     │
│                                                                         │
│  ┌──────────────────┐     ┌─────────────────────────────────────────┐  │
│  │  React Frontend  │     │          LangGraph Agent                │  │
│  │  (Vite + TS)     │     │                                         │  │
│  │                  │     │  START                                  │  │
│  │  ✦ Login / Auth  │     │  └─► check_semantic_cache               │  │
│  │  ✦ Task input    │────►│       └─► inject_history                │  │
│  │  ✦ File upload   │     │            └─► load_schema              │  │
│  │  ✦ Gate UIs      │◄────│                 └─► resolve_task_intent │  │
│  │  ✦ Live progress │ SSE │  (auto-detects ab_test vs general)      │  │
│  │  ✦ Analysis view │     │                      └─► generate_sql   │  │
│  │  ✦ History       │     │                           └─► query_gate 🛑│ │
│  └──────────────────┘     │                                └─► execute │  │
│                            │                                    │       │  │
│  ┌──────────────────┐     │         A/B experiment path:        │       │  │
│  │  FastAPI Backend │     │  decompose → anomaly → forecast      │       │  │
│  │                  │     │  → CUPED → t-test → HTE → novelty   │       │  │
│  │  POST /runs      │     │  → MDE → guardrails → funnel         │       │  │
│  │  GET  /runs/     │     │           └─► analysis_gate 🛑        │       │  │
│  │       {id}/stream│     │                └─► generate_narrative │       │  │
│  │  POST /runs/     │     │                     └─► narrative_gate 🛑    │  │
│  │       {id}/resume│     │                          └─► log_run → END   │  │
│  │  POST /upload    │     │                                         │  │
│  │  GET  /samples   │     │         General analysis path:          │  │
│  │  POST /auth/*    │     │  describe → correlations → analysis_gate│  │
│  └──────────────────┘     └─────────────────────────────────────────┘  │
│                                                                         │
│  🛑 = HITL interrupt — analyst approves or overrides before continuing  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three caching layers

```
Request
  │
  ▼
Layer 1: Semantic cache (SQLite + MiniLM embeddings)
  similarity > 0.92 → return cached result instantly        ← zero API cost
  similarity 0.80–0.92 → show cached, ask analyst
  similarity < 0.80 → cache miss, run pipeline
  │
  ▼
Layer 2: Anthropic prompt caching
  Static blocks (system prompt, schema) cached across runs  ← 90% token cost reduction
  │
  ▼
Layer 3: KV prefix reuse (within session, automatic)
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite |
| Backend | FastAPI + uvicorn + SSE |
| Agent graph | LangGraph 1.1 (SqliteSaver / PostgresSaver) |
| LLM | Anthropic Claude (claude-sonnet-4-20250514) |
| Database | DuckDB (local/upload) · PostgreSQL (external) |
| Auth | JWT (HS256) + bcrypt · refresh token revocation |
| Caching | Semantic cache (MiniLM) · Anthropic prompt cache |
| Run state | Redis Streams (multi-pod) · asyncio.Queue (local) |
| Email | Resend (password reset) |
| Observability | Sentry · structured logging · `/health` |
| Stats | scipy · numpy · Prophet |
| Tests | pytest · 135 tests |

---

## Quick start (local dev)

```bash
# 1. Clone and install
git clone <repo>
cd datapilot
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Configure
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# 3. Generate demo dataset
python data/generate_data.py

# 4. Start backend
cd backend
uvicorn api.main:app --reload --port 8000

# 5. Start frontend (separate terminal)
cd frontend
npm install && npm run dev
# → http://localhost:5173
```

---

## Deploy to Railway

Two services from the same repo:

| Service | Root dir | Key env vars |
|---------|----------|-------------|
| `backend` | `backend/` | `ANTHROPIC_API_KEY`, `SECRET_KEY`, `CORS_ORIGINS`, `APP_URL` |
| `frontend` | `frontend/` | `VITE_API_URL` (backend's public URL) |

**Required before deploying:**
1. `openssl rand -hex 32` → set as `SECRET_KEY`
2. Set `VITE_API_URL` on the frontend service (backend's Railway URL)
3. Set `CORS_ORIGINS` + `APP_URL` on the backend service (frontend's Railway URL)
4. Add a Railway volume at `/app/memory` on the backend service (persists graph state + auth DB)

**Optional:**
- `RESEND_API_KEY` + `EMAIL_FROM` — enables password reset emails
- `REDIS_URL` — Redis plugin in Railway, enables multi-pod safe run state
- `SENTRY_DSN` — error tracking

---

## Repo structure

```
datapilot/
├── frontend/                   React + TypeScript SPA
│   ├── src/
│   │   ├── api/client.ts       axios + JWT refresh interceptor
│   │   ├── hooks/useSSE.ts     SSE subscription + reconnect
│   │   ├── hooks/useTokenRefresh.ts  proactive JWT refresh
│   │   ├── pages/
│   │   │   ├── Login.tsx       sign in / register / forgot password
│   │   │   ├── Analysis.tsx    task input + sample quick-start + gate router
│   │   │   └── History.tsx     past runs with inline narrative view
│   │   └── components/gates/   one component per HITL interrupt type
│   └── Dockerfile
│
├── backend/                    FastAPI server
│   ├── api/
│   │   ├── main.py             app, CORS, lifespan, Sentry
│   │   ├── deps.py             JWT auth dependency
│   │   ├── run_manager.py      Redis Streams / asyncio.Queue run state
│   │   ├── pdf.py              fpdf2 report generation
│   │   └── routes/
│   │       ├── auth.py         register, login, refresh, logout, password reset
│   │       ├── runs.py         create, stream (SSE), resume, list, detail, PDF
│   │       ├── upload.py       CSV/Excel → DuckDB
│   │       └── samples.py      serve built-in sample datasets
│   └── Dockerfile
│
├── agents/
│   ├── state.py                AgentState TypedDict
│   └── analyze/
│       ├── graph.py            LangGraph graph — nodes, routing, HITL gates
│       ├── nodes.py            node functions (auto-detects analysis_mode)
│       └── prompts.py          all prompt templates (domain-neutral)
│
├── tools/                      pure Python, no framework deps
│   ├── db_tools.py             DuckDB + Postgres unified interface
│   ├── stats_tools.py          CUPED, t-test, HTE
│   ├── decomposition_tools.py  metric component breakdown
│   ├── anomaly_tools.py        zscore anomaly + slice-and-dice
│   ├── forecast_tools.py       Prophet (rolling mean fallback)
│   ├── guardrail_tools.py      secondary metric sweep
│   ├── novelty_tools.py        week-over-week ATE decay
│   ├── mde_tools.py            MDE + business impact
│   ├── funnel_tools.py         conditional step conversion
│   └── narrative_tools.py      structured report formatter
│
├── memory/
│   ├── store.py                SQLite run logger
│   ├── retriever.py            history retrieval for few-shot injection
│   └── semantic_cache.py       MiniLM embeddings + similarity cache
│
├── auth/store.py               user auth (bcrypt, JWT, password reset tokens)
├── config/
│   ├── analysis_config.py      MetricConfig Pydantic model
│   ├── metric_config.json      default DAU config (overridden by inference)
│   └── examples/               preset configs for other scenarios
│
├── data/
│   ├── generate_data.py        deterministic synthetic DAU dataset (seed=42)
│   └── samples/                domain sample CSVs (clinical, ecommerce, SaaS, logistics)
│
├── evals/
│   ├── analyze_eval.py         11-criterion DAU eval — score: 11/11
│   └── generalisability_eval.py  11-criterion cross-domain eval — score: 11/11
│
└── tests/                      135 unit + API integration tests
```

---

## Eval scores

```bash
python evals/analyze_eval.py          # DAU experiment scenario
python evals/generalisability_eval.py # cross-domain: clinical + ecommerce
```

**DAU experiment (11/11):**
```
  PASS  hte_correct_segment       android/new surfaces as top HTE segment
  PASS  cuped_variance_reduced    >15% variance reduction
  PASS  ttest_significant         p < 0.05
  PASS  decomp_identifies_new     new_users is dominant declining component
  PASS  slice_ranks_android_first slice-and-dice ranks android #1
  PASS  forecast_flags_drop       actuals outside Prophet CI
  PASS  guardrails_breached_found at least one guardrail breached
  PASS  optout_breached           notif_optout specifically flagged
  PASS  novelty_ruled_out         effect growing, not decaying
  PASS  narrative_mentions_segment narrative mentions android + new
  PASS  narrative_has_caveats     caveats section present
```

**Cross-domain generalisability (11/11):**
```
  PASS  🏥 Clinical: t-test executes on recovery scores
  PASS  🏥 Clinical: treatment has significantly higher recovery
  PASS  🏥 Clinical: CUPED runs with baseline_severity covariate
  PASS  🏥 Clinical: HTE finds subgroup differential
  PASS  🏥 Clinical: guardrail check on side_effect_count
  PASS  🏥 Clinical: MDE calculation completes
  PASS  🛒 Ecomm: t-test executes on revenue_usd
  PASS  🛒 Ecomm: CUPED runs with session_duration covariate
  PASS  🛒 Ecomm: HTE finds device/segment differential
  PASS  🛒 Ecomm: guardrail check on orders metric
  PASS  🛒 Ecomm: MDE is a plausible percentage
```
