# DataPilot — Dev Log

Chronological record of key issues, root causes, fixes, and architectural decisions.

---

## Railway Deployment

### Issue: Volume mount at `/app/memory` crashed backend
**Error:** `ImportError: cannot import name 'retriever' from 'memory' (unknown location)`
**Root cause:** Railway volume mounted at `/app/memory` overwrote the `memory/` Python package directory, making all imports from it fail.
**Fix:** Change volume mount path to `/app/db`. Set env vars:
- `GRAPH_DB_PATH=/app/db/graph.db`
- `AUTH_DB_PATH=/app/db/auth.db`
- `MEMORY_DB_PATH=/app/db/datapilot_memory.db`
- `UPLOAD_DIR=/app/db/uploads`

**Follow-up (2026-08-01):** the original fix listed only the first two vars.
The other two kept their in-image defaults (`memory/datapilot_memory.db` and
`tmp_uploads`), which live *outside* the volume — so run history and every user
upload were silently wiped on each redeploy, and follow-up runs 404'd on
checkpoints referencing uploads that no longer existed. The mount alone
persists nothing; each database needs its own variable pointing into it.
Leave `DUCKDB_PATH` at its default — `data/generate_data.py` hardcodes its
write to `/app/data`, so repointing only the read path breaks it.

### Issue: `PowerAnalysisResult` ImportError on Railway
**Error:** `ImportError: cannot import name 'PowerAnalysisResult' from 'tools.schemas'`
**Root cause:** Power analysis classes (`PowerAnalysisResult`, `SensitivityRow`) were added to `tools/schemas.py` locally and imported in `nodes.py`, but `schemas.py` was never committed. Railway deployed the old committed version without the classes.
**Fix:** Commit `tools/schemas.py` with the missing classes alongside all other pending local changes.

---

## Production Hardening (2026-08-01)

Found by auditing the deployed app rather than reading the code. Each of these
was live in production and silent — no error surfaced to a user, and in most
cases nothing surfaced in CI either. Full context in
[docs/production-operations.md](docs/production-operations.md).

### Issue: run history and uploads wiped on every redeploy
**Symptom:** `/health` reported `memory_db: "not_created_yet"`; follow-up runs
404'd on uploads from an earlier session.
**Root cause:** see the follow-up on the volume-mount entry above — two of the
four path variables were never set, so those files lived outside the volume.
**Fix:** set `MEMORY_DB_PATH` and `UPLOAD_DIR` into `/app/db`.

### Issue: shutdown crashed on every deploy
**Error:** `ImportError: cannot import name 'cancel_active_runs' from 'backend.api.run_manager'`
**Root cause:** `main.py` imported a symbol that was never implemented, so the
lifespan teardown raised before cancelling background tasks or closing Redis.
Invisible because `tests/test_api.py` replaces the lifespan with a stub.
**Fix:** implement it; add real lifespan coverage in `tests/test_run_lifecycle.py`.

### Issue: password reset and signup verification silently broken
**Error (logs only):** ``Invalid `from` field`` from Resend.
**Root cause:** `EMAIL_FROM` was `DataPilot <singhaman.dev>` — a bare domain
where an address belongs. `email.py` swallows the exception and
`/auth/forgot-password` returns 202 regardless, so users saw "a reset link has
been sent" and nothing arrived.
**Fix:** `DataPilot <noreply@singhaman.dev>`. Verified by triggering a real
reset and reading the logs.

### Issue: per-IP rate limiting was inert
**Symptom:** 20 concurrent bad logins, zero 429s against a 10/60s cap. Brute
force protection off; the guest spend cap keys on the same value.
**Root cause:** self-inflicted. `client_ip` was changed from the leftmost
`X-Forwarded-For` entry to a fixed one-hop-from-the-right, on the general
principle that the leftmost entry is client-controlled. Railway's actual shape
is `<client>, <railway-edge>` where the edge address is public and *rotates per
request*, so every request landed in its own bucket. A second attempt — skip
infrastructure ranges — picked the same rotating edge.
**Fix:** back to leftmost. Railway *replaces* an inbound `X-Forwarded-For`
(verified: a request sent carrying `9.9.9.9` arrives without it), so leftmost is
both the real client and unforgeable here. `TRUSTED_PROXY_HOPS` pins an exact
position for other topologies.
**Lesson:** two "more correct" fixes both made it worse. A one-deploy
`DEBUG_CLIENT_IP` log settled it in minutes; reasoning from the generic proxy
model did not.

### Issue: retention pruned rows but reclaimed no disk
**Symptom:** `graph_bytes_reclaimed: 0` after deleting 267 checkpoints.
**Root cause:** under WAL, VACUUM writes the rebuilt database into the WAL and
the main file keeps its size until a checkpoint. Local tests missed it because
their fixtures used rollback-journal mode; a corrected WAL test still passed,
because SQLite auto-checkpoints when the *last* connection closes and production
always holds one open.
**Fix:** `PRAGMA wal_checkpoint(TRUNCATE)` after VACUUM; test now holds a
connection across the vacuum.

---

## Analysis Accuracy

### Issue: Hallucinated numbers in narrative ("12 percentage points")
**Example:** Pipeline reported "underperforms by 12 percentage points" when actual gap was 77.4% vs 77.0% = 0.4 points.
**Root cause:** LLM blended a gap figure from one comparison into a sentence about a different entity — a common LLM reasoning error when multiple numeric comparisons are in context.
**Fix:** Added `NUMERICAL ACCURACY` block to `INSIGHTS_NARRATIVE_PROMPT`:
- Explicitly require both values to be stated in every comparison sentence
- Require arithmetic verification: stated gap must equal larger − smaller
- Forbid blending gap figures from one comparison with entities from another

---

## Trust Indicators

### Issue: "254 data points" on an 18,000-row logistics dataset
**Root cause:** `describe.row_count` reflected the number of aggregated groups (254 route/carrier combos) returned by the SQL query, not the underlying shipment rows.
**Fix:** In `generate_charts` node, check for `total_records` column in query result and sum it to recover the underlying row count:
```python
if "total_records" in qr.columns:
    n_underlying = int(qr["total_records"].sum())
```

---

## RAGAS / Eval Scoring

### Issue: General mode always scored ~0.30–0.37 (vs A/B ~0.65–0.69)

**Bug 1 — Completeness always 0 for general runs**
`_compute_quality_score` had 6 hardcoded completeness checks (cuped, ttest, hte, guardrail, novelty, forecast) — all A/B-specific. General runs never populate any of these, so `completeness = 0/6 = 0.0`, capping the final score at `0.4 × ragas_score`.
**Fix:** Made completeness mode-aware:
- `ab_test`: original 6 A/B checks
- `general`: checks describe_result, correlation_result, charts, narrative_draft, query_result
- `power_analysis`: checks power_analysis_result, narrative_draft

**Bug 2 — Faithfulness penalises accurate percentage citations**
SQL stores rates as fractions (e.g., `AVG(on_time_delivery) = 0.774`). Narrative writes "77.4%". Number extractor pulls `77.4`. Relative error vs `0.774` is 99% → marked unsupported → faithfulness penalised.
**Fix:** In `score_faithfulness`, also check `num / 100` against DataFrame values:
```python
num_as_frac = num / 100.0
if match_direct or match_as_fraction:
    supported += 1
```

---

## Frontend Refactoring

### Shared components extracted
- `Spinner.tsx` — three variants: `button` (gradient btn), `page` (full page load), `inline`
- `FormField.tsx` — shared label+input for all auth forms
- `AuthCard.tsx` — outer wrapper (orbs, card, logo) + `authShared` styles (errorBox, btn, linkBtn)
- `gates/shared.ts` — shared gate styles (gateCard, gateTitle, gateTextarea, gateActions, gateBtnApprove, gateBtnSecondary)

### Pages slimmed
- `Login.tsx`: 175 → 92 lines
- `ForgotPassword.tsx`: 117 → 86 lines
- `ResetPassword.tsx`: 149 → 89 lines
- `Analysis.tsx`: 897 → 796 lines (inline utilities moved to `types/analysis`, `utils/markdown`, `utils/error`)

---

## Auth & Session Persistence

### How it works
- Access token (1h) + refresh token (30d) in **HttpOnly cookies** — JavaScript
  cannot read them, so XSS cannot exfiltrate a session
- Axios interceptor auto-refreshes on 401 responses (`withCredentials: true`,
  no token handling in JS)
- Refresh tokens rotate on use: the old `jti` is revoked before a new one is
  minted, and a password reset bumps `session_version` to invalidate all sessions
- `localStorage` holds only `datapilot.workspace_id`, a non-secret UI preference
- User records in `memory/auth.db` (SQLite), graph state in `memory/graph.db`
- Both files persisted via Railway volume mount at `/app/db`

> **Superseded (2026-08-01):** this section previously described access and
> refresh tokens living in `localStorage`. That was the original design and it
> was XSS-vulnerable; it has since moved to HttpOnly cookies. The old text is
> corrected rather than kept, because a reader could otherwise reintroduce it.

---

## Architecture Notes

### SSE event shape
```json
{ "type": "gate",  "gate": "<gate_name>", "payload": { ... } }
{ "type": "done",  "state": { "narrative_draft": "...", "recommendation": "..." } }
{ "type": "error", "message": "..." }
```
EventSource cannot send headers, so the stream authenticates with a **short-lived
scoped token**, not the session JWT: the client calls `GET /runs/{id}/stream-token`
(cookie-authenticated) and passes the result as `?stream_token=...`. The token is
scoped to one run and expires in 15 minutes, so a URL leaking into access logs,
`Referer`, or browser history does not expose a session. `/runs/{id}/pdf` uses the
same pattern with a 5-minute token.

### Graph checkpointer
`SqliteSaver` (langgraph-checkpoint) persists graph state across container restarts.
Path resolved via `GRAPH_DB_PATH` env var, defaults to `memory/graph.db`.

### Analysis modes
| Mode | Key state keys | Completeness checks |
|------|---------------|---------------------|
| `ab_test` | cuped, ttest, hte, guardrail, novelty, forecast | 6 A/B tool results |
| `general` | describe, correlation, charts, narrative, query_result | 5 general outputs |
| `power_analysis` | power_analysis_result, narrative_draft | 2 outputs |

### File upload flow
1. `POST /upload` → pandas reads CSV/Excel → writes to temp DuckDB at `tmp_uploads/{user_id}/{upload_id}.db`
2. `POST /runs` with `duckdb_path=<upload_id>` → backend resolves upload_id → actual path → injected into AgentState
3. Graph's `_db_conn` prefers `state["duckdb_path"]` over env-var default

---

## Commit History (notable)

| Commit | Description |
|--------|-------------|
| `4978029` | Fix missing PowerAnalysisResult/SensitivityRow + all pending changes |
| `d82c8ac` | Fix trust indicator: underlying row count not aggregated group count |
| `f9a22cc` | Fix narrative hallucination: add numerical accuracy rules to prompt |
| `6ffdb8b` | Fix A/B pipeline: 20 correctness issues resolved |
| `827c5ad` | feat: resolve_task_intent node — Rule 6 (ask before assuming) |
| `11c9a42` | feat: complete DataPilot — 11/11 eval, self-improvement loop, full UI |
2026-04-03T20:55:20-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T20:55:37-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T20:55:48-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T21:00:46-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-03T21:00:55-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T21:01:21-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T21:12:00-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T21:30:12-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T21:33:00-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-03T21:34:26-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-03T21:34:35-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-03T21:35:53-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T12:33:29-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T12:36:19-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T12:36:32-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T12:50:11-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T12:50:20-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T16:40:07-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-04-04T17:06:00-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T17:06:07-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T17:07:24-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T17:07:37-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T17:14:05-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T18:43:46-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T18:45:37-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T18:58:12-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-04T19:03:07-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T19:08:33-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T19:14:17-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T19:43:48-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T19:46:11-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:02:07-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:02:47-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:03:24-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-04-04T20:06:44-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:06:51-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:06:57-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-04-04T20:37:10-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-04-04T20:40:03-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:40:12-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:40:19-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T20:40:27-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-04-04T21:32:22-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T21:36:14-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-04-04T21:36:28-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T21:42:49-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes.py | prompt file modified
2026-04-04T21:46:31-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompts.py | prompt file modified
2026-08-01T08:32:43-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:32:47-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:32:53-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:33:00-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:46:56-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_narrative.py | prompt file modified
2026-08-01T08:47:07-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:47:16-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/prompt_safety.py | prompt file modified
2026-08-01T08:47:19-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:47:23-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:47:31-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T08:48:28-04:00 | /Users/amansingh/Desktop/datapilot/tests/test_history_prompt_safety.py | prompt file modified
2026-08-01T08:48:45-04:00 | /Users/amansingh/Desktop/datapilot/tests/test_history_prompt_safety.py | prompt file modified
2026-08-01T08:49:19-04:00 | /Users/amansingh/Desktop/datapilot/tests/test_history_prompt_safety.py | prompt file modified
2026-08-01T09:00:48-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_intent.py | prompt file modified
2026-08-01T09:01:00-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_intent.py | prompt file modified
2026-08-01T09:06:33-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_analysis.py | prompt file modified
2026-08-01T09:06:37-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_analysis.py | prompt file modified
2026-08-01T09:06:47-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_analysis.py | prompt file modified
2026-08-01T09:06:53-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_analysis.py | prompt file modified
2026-08-01T17:49:26-04:00 | /Users/amansingh/Desktop/datapilot/backend/api/routes/runs.py | prompt file modified
2026-08-01T22:26:45-04:00 | /Users/amansingh/Desktop/datapilot/agents/analyze/nodes_narrative.py | prompt file modified
