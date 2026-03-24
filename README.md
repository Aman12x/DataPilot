# DataPilot

**Agentic AI data analyst with human-in-the-loop approval gates.**

Ask a question in plain English. DataPilot generates SQL, runs rigorous statistical analysis, and produces an evidence-based report — pausing for your review at every decision point before moving forward.

**Eval:** 11/11 DAU experiment · 11/11 cross-domain generalisability · 444 tests passing

---

## Pipeline

```
  User submits task (natural language)
         │
         ▼
  ┌─────────────────────────────────────┐
  │  1. Semantic cache check            │
  │     similarity > 0.92 → return      │──── cache hit ────► Cached result
  │     similarity 0.80–0.92 → confirm  │                     (zero API cost)
  │     similarity < 0.80 → run         │
  └──────────────┬──────────────────────┘
                 │ cache miss
                 ▼
  ┌─────────────────────────────────────┐
  │  2. Schema load + intent resolution │
  │     auto-detects: experiment vs     │
  │     general analysis                │
  │     asks ONE question if ambiguous  │
  └──────────────┬──────────────────────┘
                 │
                 ▼
         🛑 INTENT GATE
         analyst confirms or corrects
         task interpretation
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  3. SQL generation                  │
  │     LLM writes query against        │
  │     inferred schema                 │
  └──────────────┬──────────────────────┘
                 │
                 ▼
         🛑 QUERY GATE
         analyst reviews SQL before
         any data is touched
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  4. Execute query → DataFrame       │
  └──────┬──────────────────────────────┘
         │
         ├──── A/B Experiment ───────────────────────────────────────────────────┐
         │                                                                        │
         ▼                                                                        ▼
  ┌──────────────────────┐                                        ┌──────────────────────────┐
  │  General analysis    │                                        │  Experiment analysis      │
  │                      │                                        │                           │
  │  • describe          │                                        │  • metric decomposition   │
  │  • correlations      │                                        │  • anomaly detection      │
  │  • OLS regression    │                                        │  • forecast (Prophet)     │
  │  • time series       │                                        │  • CUPED variance reduction│
  │  • anomaly/forecast  │                                        │                           │
  │  • top/bottom rows   │                                        │                           │
  └──────────┬───────────┘                                        │  • t-test (p-value, CI)   │
             │                                                     │  • HTE subgroup analysis  │
             │                                                     │  • novelty effect check   │
             │                                                     │  • MDE + business impact  │
             │                                                     │  • guardrail sweep        │
             │                                                     │  • funnel analysis        │
             │                                                     └──────────┬───────────────┘
             │                                                                │
             └──────────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
                                🛑 ANALYSIS GATE
                                analyst reviews all findings,
                                can override any result
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │  5. Generate narrative report  │
                        │                                │
                        │  • TL;DR                       │
                        │  • Key findings                │
                        │  • Subgroup breakdown          │
                        │  • Confidence level            │
                        │  • Recommendation              │
                        │  • Caveats                     │
                        └──────────────┬─────────────────┘
                                       │
                                       ▼
                               🛑 NARRATIVE GATE
                               analyst approves or
                               requests revision
                                       │
                                       ▼
                        ┌───────────────────────────────┐
                        │  6. Log run + quality score    │
                        │     RAGAS eval → memory store  │
                        │     future runs inherit this   │
                        │     analyst's corrections      │
                        └──────────────┬─────────────────┘
                                       │
                                       ▼
                              📄 Final report
                                 (PDF export)

  🛑 = human-in-the-loop interrupt. Nothing moves forward without analyst approval.
```

---

## Use cases

DataPilot infers the schema of any uploaded CSV or connected database and adapts its analysis accordingly. No configuration required.

**Product / Growth**
> *"Did the new checkout redesign increase revenue? Which device types drove the effect?"*

Runs a full A/B test pipeline: CUPED-adjusted t-test, HTE by device/platform/segment, novelty effect decay check, guardrail sweep on engagement and opt-out rates.

**Clinical research**
> *"Did Drug A improve 30-day recovery scores vs placebo? Check for differential effects by age group and flag any side effect signals."*

Same pipeline — treatment/control comparison with covariate adjustment, subgroup HTE, guardrail on adverse event rate.

**SaaS / Retention**
> *"What factors most strongly predict churn? Which plans and cohorts churn fastest?"*

General analysis mode: OLS regression identifies strongest churn predictors (tenure, support tickets, plan tier), correlation matrix, time series anomaly detection on MRR, top contributing rows.

**Ecommerce**
> *"Is the new recommendation widget increasing basket size? Is it hurting conversion for mobile users?"*

A/B path: revenue uplift with session-duration CUPED, HTE by device, MDE to answer "was the experiment underpowered?", funnel impact.

**Finance / Risk**
> *"Which customer segments have the highest default rates? Is the trend worsening over the last 6 months?"*

General mode: segment breakdown, trend decomposition, anomaly detection, correlation with leading indicators.

**HR / People analytics**
> *"Is there a pay gap by department or level? How does it change after controlling for tenure?"*

General mode: OLS regression with department/level one-hot encoding, VIF check for multicollinearity, correlation analysis, top/bottom rows by salary.

**Clinical research (general mode)**
> *"Which patient factors best predict length of stay? Flag any anomalies in readmission trends."*

OLS regression on los_days with diagnosis, BMI, and systolic BP as predictors; time series anomaly detection on readmission_30d if a date column is present.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite |
| Backend | FastAPI + uvicorn |
| Agent graph | LangGraph 1.1 (SqliteSaver / PostgresSaver) |
| LLM | Claude claude-sonnet-4-20250514 |
| Query engine | DuckDB (upload/local) · PostgreSQL (external) |
| Auth | JWT HS256 + bcrypt + refresh token revocation |
| Semantic cache | MiniLM (all-MiniLM-L6-v2) + SQLite |
| Run state | Redis Streams (multi-pod) · asyncio.Queue (local) |
| Stats | scipy · numpy · scikit-learn · Prophet |
| Eval | RAGAS-inspired (faithfulness + relevancy + key findings) |
| Observability | Sentry · structured logging |
| Tests | pytest · 444 tests |

---

## Eval scores

```bash
python evals/analyze_eval.py           # DAU experiment scenario
python evals/generalisability_eval.py  # cross-domain: clinical + ecommerce
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
  PASS  🏥 Clinical: t-test on recovery scores
  PASS  🏥 Clinical: treatment has significantly higher recovery
  PASS  🏥 Clinical: CUPED with baseline_severity covariate
  PASS  🏥 Clinical: HTE finds age-group differential
  PASS  🏥 Clinical: guardrail on side_effect_count
  PASS  🏥 Clinical: MDE calculation completes
  PASS  🛒 Ecomm: t-test on revenue_usd
  PASS  🛒 Ecomm: CUPED with session_duration covariate
  PASS  🛒 Ecomm: HTE finds device/segment differential
  PASS  🛒 Ecomm: guardrail on orders metric
  PASS  🛒 Ecomm: MDE is a plausible percentage
```

---

## Quick start

```bash
# 1. Clone and install
git clone <repo> && cd datapilot
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Set API key
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# 3. Generate demo data
python data/generate_data.py

# 4. Backend
cd backend && uvicorn api.main:app --reload --port 8000

# 5. Frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

---

## Deploy (Railway)

Two services from the same repo:

| Service | Root dir | Required env vars |
|---------|----------|-------------------|
| `backend` | `backend/` | `ANTHROPIC_API_KEY`, `SECRET_KEY`, `CORS_ORIGINS` |
| `frontend` | `frontend/` | `VITE_API_URL` (backend's public Railway URL) |

Add a Railway volume at `/app/memory` on the backend service — persists graph checkpoints and auth DB across deploys.

Optional: `REDIS_URL` (multi-pod run state), `SENTRY_DSN` (error tracking), `RESEND_API_KEY` (password reset emails).

---

## Trust and robustness

Five layers prevent wrong reports from reaching stakeholders.

| Layer | What it catches |
|-------|----------------|
| **SQL content validation** | 0-row results, missing experiment arms, severe arm imbalance (< 30:70), JOIN fan-out (row explosion from bad joins), metric values that look like percentages not rates |
| **Claim accuracy blocking** | Narrative says "significant" but CI crosses zero; "large effect" but Cohen's d < 0.5; stated direction contradicts the sign of the ATE — auto-corrected before the analyst sees it |
| **Safety constraint checks** | Blocks "ship" language when SRM is detected, a guardrail is breached, or winner's curse risk is present (significant result but post-hoc power < 50%) |
| **Winner's curse banner** | Prepended to TL;DR (not buried in caveats) when the experiment is underpowered or the observed effect is likely inflated |
| **Audit log** | Every approved report appends a structured record: run ID, analyst decisions at each gate, SRM acknowledgment, auto-correction count, SQL warnings |

---

## Architecture choices

The decisions that aren't obvious from the stack table.

### LangGraph over a plain chain or custom FSM

The pipeline has conditional branching (experiment vs general), four independent interrupt points, and needs to resume mid-graph after a human approves a gate. LangGraph's `interrupt()` + `Command(resume=...)` handles this cleanly with built-in SQLite/Postgres checkpointing. A plain chain can't interrupt and resume. A hand-rolled FSM would require reimplementing graph traversal, state persistence, and retry logic from scratch.

### HITL gates over a fully autonomous agent

Fully autonomous agents hallucinate silently. The query gate ensures LLM-generated SQL is human-verified before touching data. The analysis gate lets an analyst catch a subtly wrong interpretation before it becomes a PDF report. The narrative gate closes the loop. The agent is fast; the gates take seconds to approve. The cost of skipping them — wrong SQL runs, hallucinated statistics ship — is far higher.

### DuckDB over running everything in pandas

Analytical queries — aggregations, group-bys, window functions, joins — are what columnar query engines are built for. DuckDB on a 500k-row dataset is 10–50x faster than pandas equivalents and supports out-of-core execution. More importantly, the LLM generates SQL naturally; prompting for pandas method chains is harder and harder to inspect. DuckDB also provides a single unified interface across uploaded CSVs, the demo DB, and external Postgres — no separate code paths.

### SSE over WebSockets for streaming

Each analysis run is a one-way stream of events: gate payloads arrive, the client sends a resume POST, the next gate payload arrives. There's no need for a persistent bidirectional channel. SSE is HTTP/1.1, works through every proxy and load balancer without configuration, reconnects automatically, and requires no handshake overhead. WebSockets add complexity for no benefit with this access pattern.

### Semantic cache with MiniLM over exact-match or no cache

Analyst questions are semantically similar but rarely textually identical ("which segment drove the DAU drop" ≈ "what caused the DAU decline by cohort"). Exact-match misses these. No cache means every run costs $0.15–0.40 in API tokens and 45–90 seconds of latency. MiniLM (22M params, 384-dim embeddings) runs locally in ~5ms. The three-tier threshold — hard hit → return instantly, soft hit → confirm with analyst, miss → run — gives control over when to trust a cache hit.

### CUPED for variance reduction instead of raw t-test

A raw t-test on DAU or revenue has high variance from pre-existing user differences. CUPED (Controlled-experiment Using Pre-Experiment Data) uses a pre-experiment covariate to partial out this variance, typically reducing it 15–40%. The same experiment reaches statistical significance with fewer users — or equivalently, the same sample size produces a tighter confidence interval. It's the technique used in production at Netflix, Airbnb, and Booking.com for this reason.

### Deterministic RAGAS eval over LLM-as-judge

LLM-as-judge costs $0.10–0.50 per eval run, introduces stochastic variance, and creates a circular dependency (the same model evaluating itself). The three metrics used here — faithfulness (cited numbers traceable to the DataFrame), relevancy (MiniLM cosine similarity to the task), key findings (keyword presence) — run in under 100ms with zero API cost. Faithfulness specifically catches the most common AI failure mode: hallucinated statistics that sound plausible but aren't in the data.

### Self-improvement loop via memory store

Every completed run logs its task, SQL, findings, narrative, and eval score to SQLite. Future runs retrieve the top-k most similar past runs by keyword and semantic similarity, injecting them as few-shot examples into SQL generation. Analyst corrections — overriding a wrong HTE segment, editing a narrative — are stored and retrieved on similar future tasks. The agent improves on your specific domain and data conventions over time, without retraining.
