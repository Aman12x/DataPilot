# DataPilot

DataPilot is an AI data analyst with human review built into the pipeline. You ask a question in plain English; it writes the SQL, runs the statistics, and drafts a report. Before any of that reaches a stakeholder, a person has approved the query, the analysis, and the wording. I built it because most LLM analytics tools fail silently, and the interesting problem is not generating SQL but making sure a wrong number never leaves the building.

There is a live instance at [datapilotapp.singhaman.dev](https://datapilotapp.singhaman.dev) with a guest login, no signup needed. API health: [datapilot.singhaman.dev/health](https://datapilot.singhaman.dev/health).

![DataPilot home](docs/screenshots/home.png)

## How it works

A run moves through a LangGraph state machine with four interrupt points. At each one the run pauses, persists to a checkpoint, and waits for a person. Approvals survive page reloads and server restarts because the graph resumes from the checkpoint, not from memory.

```
  question
     |
  semantic cache ------ close match to a past run? -> cached result
     |
  schema load + intent resolution -> INTENT GATE   (confirm the interpretation)
     |
  SQL generation ------------------> QUERY GATE    (review the SQL before it runs)
     |
  execute query
     |
     +-- general analysis: describe, correlations, regression,
     |   time series, anomaly detection, forecast
     +-- experiment analysis: CUPED, t-test, subgroup effects,
     |   novelty check, power/MDE, guardrails, funnel
     |
  ANALYSIS GATE   (review findings, override anything)
     |
  narrative generation + claim audit
     |
  NARRATIVE GATE  (approve or request a revision)
     |
  report + stakeholder deck + PDF
```

The LLM's only job is generating SQL and prose. Query validation, execution, and every statistic are deterministic Python (scipy, scikit-learn, Prophet). After the narrative is drafted, an audit step checks its claims against the computed numbers: if the text says "significant" while the confidence interval crosses zero, or "ship it" while a guardrail is breached, the claim is corrected or the narrative is sent back for revision before an analyst ever sees it.

| Reviewing generated SQL | The finished report |
|---|---|
| ![SQL gate](docs/screenshots/sql-gate.png) | ![Report](docs/screenshots/report.png) |

## Why approval gates?

Autonomous agents fail quietly. A hallucinated filter produces a plausible-looking number, and nobody notices until it is in a slide deck. The gates change the failure mode from "wrong number in production" to "analyst clicks reject." Each gate takes a few seconds to approve, and each one has caught real mistakes during development: wrong table joins at the query gate, a misread experiment direction at the analysis gate.

The tradeoff is real: a fully autonomous run would be faster. For exploratory questions on trusted data the gates can feel heavy, which is why cached results skip the pipeline entirely and certified metric packs skip the metric-confirmation step.

## Data sources

Upload a CSV or Excel file, or connect Postgres, MySQL, or BigQuery. Connections are saved objects with a tested/failed/untested health state, and can be re-tested, edited, and deleted from the UI. Credentials are encrypted at rest, never returned by the API, and wiped from workflow checkpoints once the schema is loaded. Private-network hosts are refused by default to prevent SSRF.

![Data sources](docs/screenshots/data-sources.png)

Postgres discovery covers all schemas the role can see, not just `public`. Column annotations and business synonyms can be attached per connection and are injected into the schema context, so generated SQL uses the vocabulary your team actually uses. Workspaces add owner/analyst roles with shared connections, metric packs, and run history.

## Testing

The backend has 937 tests that run in about 30 seconds. A few are unusual enough to mention:

- An AST scan walks every async route and fails on blocking calls in the event loop. The hand-written grep it replaced had found 6 call sites; the scan found 56.
- A log-safety test fails if any code under `agents/` logs a raw exception, because INFO records become Sentry breadcrumbs and pandas exceptions contain customer column names.
- The CSP suite (30 Playwright tests) runs against the production build with its generated policy and includes tests that deliberately violate the policy, so a header that silently stopped enforcing would fail the suite.

Four offline eval harnesses check the statistics layer against golden answers on every push, and CI fails if any score drops more than 2% below a committed baseline. They are deterministic and run in milliseconds; there is no LLM-as-judge in the gate. Honest caveat: these harnesses cover the deterministic tools, not the LLM stages. SQL generation quality is currently exercised by end-to-end runs rather than a gated benchmark, and building that benchmark is the next planned piece of work.

| Harness | Score | Checks |
|---------|-------|--------|
| DAU experiment | 12/13 | subgroup effects, CUPED, t-test, guardrails, forecast |
| Cross-domain | 13/13 | clinical trial and ecommerce A/B |
| Transactions Q&A | 7/7 | golden answers on a 10k-row dataset |
| CSV fixtures | 4/4 | four domains, keyword and faithfulness checks |

## Running it locally

```bash
git clone https://github.com/Aman12x/DataPilot && cd DataPilot
python -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env          # add ANTHROPIC_API_KEY
python data/generate_data.py  # demo dataset

cd backend && uvicorn api.main:app --reload --port 8000
# in another terminal:
cd frontend && npm install && npm run dev   # http://localhost:5173
```

```bash
./venv/bin/python -m pytest tests/ -m "not integration and not slow" -q   # backend tests
make eval                                                                 # offline evals, no API key
cd frontend && npx playwright test                                        # end to end
```

Deployment (Railway, volumes, environment variables) is covered in [docs/production-operations.md](docs/production-operations.md).

## Design notes

**Why LangGraph?** The pipeline needs conditional branching and the ability to pause mid-graph, persist, and resume after a human acts. A plain chain cannot do that, and a hand-rolled state machine would mean reimplementing checkpointing.

**Why DuckDB for execution?** One SQL interface across uploaded files, demo data, and external warehouses, and columnar execution is simply the right tool for aggregation. It also spares the LLM from generating pandas.

**Why no LLM-as-judge in CI?** Judges cost money per run, add variance, and grade the model with a model. Checking whether the narrative's numbers exist in the data is a string-and-arithmetic problem, and it catches the failure that matters most.

**Why SSE instead of WebSockets?** A run is a one-way event stream with occasional POSTs back. SSE reconnects on its own and passes through proxies without ceremony.

**Why a local semantic cache?** Analysts ask the same question in different words. MiniLM embeddings run locally in a few milliseconds, and a cache hit costs zero tokens and returns instantly. Borderline similarity asks the user instead of guessing.

The longer versions, with the options that lost, are in [decisions.md](decisions.md).

## Limitations

Worth knowing before you take the demo apart:

- The default deployment runs a single worker with SQLite checkpoints. That is fine for a demo and small teams, and is the first thing to change for real scale.
- The eval gate covers the statistics layer, not SQL generation (see Testing above).
- Guest sessions are rate-limited and spend-capped per IP, so heavy demo use can hit a budget wall by design.

## More documentation

| Doc | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Codebase layout, invariants, and traps for anyone working in it |
| [decisions.md](decisions.md) | Architecture decisions with tradeoffs |
| [docs/production-operations.md](docs/production-operations.md) | Config, spend caps, retention, CSP, runbook |
| [evals/README.md](evals/README.md) | How the eval harnesses work and how to add one |

---

Built by [Aman Singh](https://github.com/Aman12x). If you read this far, the live demo is the fastest way to see whether any of it holds up.
