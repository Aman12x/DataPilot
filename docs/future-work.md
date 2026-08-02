# Future work

The known-open items, with enough context that whoever picks one up — in a month
or in a fresh session — doesn't have to re-derive why it's open, what the fix
looks like, or what will break if it's done naively. Ordered by what I'd do
first.

One-line versions live in [CLAUDE.md](../CLAUDE.md#known-open-issues); this file
is the detail. When an item ships, remove it from both and record anything
surprising in [DEVLOG.md](../DEVLOG.md).

---

## 1. Move `FAST_MODEL` off Haiku 4.5

**What:** `FAST_MODEL` (default `claude-haiku-4-5-20251001` in
`agents/analyze/node_shared.py::_fast_model`) drives everything except intent
resolution: SQL generation, the narrative, the audit, corrections.

**Why it matters more than it looks:** this is not purely an upgrade. The
trailing-assistant-prefill bug meant that on Haiku, every analyst-requested
revision was silently *continuing* the previous narrative instead of rewriting
it (measured: asked to continue "Hello there", Haiku returned "! 👋"). The
blockers are fixed (`1cfdad9`) — `_conversation_turns` normalises the history
and `response_text()` reads past thinking blocks — and `generate_narrative` was
run against `claude-sonnet-5` three times end-to-end, audit included. The move
itself has simply not been made.

**How:** set `FAST_MODEL=claude-sonnet-5` (Railway variable; `.env.example`
documents it). The audit call is now sized for a thinking model
(`MAX_TOKENS_AUDIT`, default 8192), so no blocker remains.

**Verify:** run one analysis to completion on the deployed app, decline the
narrative gate with notes, and confirm the revision is a *rewrite*, not an
extension. `railway logs | grep run.spend` — expect roughly 3× Haiku's cost per
run at Sonnet list rates ($3/$15 vs $1/$5 per MTok); `agents/pricing.py` already
has the entry, so metering is correct on day one.

---

## 2. Split-brain storage when `DATABASE_URL` is set

**What:** `langgraph-checkpoint-postgres` is commented out in
`backend/requirements.txt`. With `DATABASE_URL` set, accounts and run history
move to Postgres but LangGraph checkpoints silently stay on local SQLite
(`main.py` falls back without logging anything of note).

**Why:** it breaks the mental model an operator has after setting
`DATABASE_URL` ("my state is in Postgres now"). Checkpoints — in-flight runs,
gate state, resumability — are still on the ephemeral-ish volume. It also
blocks scaling past one replica: see the "Scaling limits" section of
[production-operations.md](production-operations.md).

**How, in order of effort:**
1. *Minimum:* log a prominent startup warning when `DATABASE_URL` is set but
   the Postgres checkpointer is unavailable, so the split is at least visible.
2. *Real fix:* uncomment the dependency, wire
   `PostgresSaver` in `main.py`'s lifespan when `DATABASE_URL` is present, and
   port `SafeCheckpointSerde` (pickle stays disabled). Retention's
   checkpoint-age logic reads UUIDv6 timestamps from `checkpoint_id` and its
   VACUUM/`wal_checkpoint` logic is SQLite-specific — the retention pass needs a
   Postgres branch or an explicit "SQLite only" guard.

**Verify:** with `DATABASE_URL` set, create a run, restart the process, resume
the gate — the run must survive. `-m integration` CI already runs a Postgres
container.

---

## 3. Off-box backups

**What:** the retention pass (`backend/api/retention.py`) snapshots `auth.db`
and `datapilot_memory.db` via `VACUUM INTO`, keeping `BACKUP_KEEP=7` — onto the
same Railway volume as the live databases.

**Why:** covers corruption, bad migrations, and accidental deletes. Does not
cover losing the volume, which is the scenario people usually mean by
"backups". `graph.db` is deliberately excluded (transient run state, and it's
the thing filling the disk) — keep it excluded.

**How:** after each snapshot, upload to object storage (S3/R2/B2 via boto3-
compatible API). Config: `BACKUP_S3_BUCKET`, endpoint + credentials via env.
Keep it inside the existing `run_maintenance` pass (already off the event loop
via `asyncio.to_thread`); failures log at warning and never fail the pass.
Prune remote copies with the same `BACKUP_KEEP` logic. Encrypting with the
existing Fernet key is tempting but wrong — a backup you can't decrypt after
losing the box is not a backup; use bucket-side encryption.

**Verify:** extend `tests/test_retention.py` with a fake S3 client asserting
upload + remote prune; then one manual restore drill: download a snapshot,
point `AUTH_DB_PATH` at it locally, log in.

---

## 4. CSP sweep — three uncovered surfaces

**What:** `frontend/e2e/csp-sweep.spec.ts` renders every screen, gate, and
modal *except*: PackStudio's inner flows (template → form → save), Annotation-
Studio with a live connection (it currently renders its empty state), and
MembersPanel.

**Why:** the tightened `style-src` (no `'unsafe-inline'`, `d59617d`) is
justified by coverage. These three are variations on covered screens and very
likely fine, but "very likely" is the phrase that preceded both CSP gaps found
this session. The pattern to follow is already in the file: stub the API with
populated payloads, assert on the component's own copy, then
`expectNoViolations`.

**How:** PackStudio — click a template, fill the form, save (stub
`POST /metric-packs`). AnnotationStudio — stub `/connections` with one saved
connection and `/connections/{id}/annotations` with rows. MembersPanel — stub
`/workspaces` with a workspace and `/workspaces/{id}/members` with 2–3 members,
open the panel.

**Verify:** the tests themselves; keep the detector tests (inline script +
inline style attribute) untouched — they're what makes green results mean
anything.

---

## Unverified audit claims (triage before trusting)

Items from the April/August audits that have **not** been re-verified recently.
Each needs an hour of confirmation before it's worth scheduling — some may
already be partially fixed by this session's work.

- **No metrics or alerts**; `/health` returns 200 even on a wiped volume.
- **In-flight runs orphaned on restart** — no reaper marks them failed, so the
  UI waits on a stream that will never emit. (Shutdown now cancels cleanly at
  node boundaries since `b5c06a7`, but a *crash* still orphans.)
- **Frontend never reconnects** after a dropped SSE stream mid-run
  (`useSSE.ts` — `onerror` gives up unless a gate was already received).
- **~10 bare `except Exception: pass` sites** swallowing failures.
- **Silent degradation**: missing `RESEND_API_KEY` locks every new account out
  with `email_sent: true`; MiniLM load failure falls back to a hash-bucket
  embedder that serves wrong cached analyses at the same similarity thresholds.
- **Single-instance constraints** (documented, deliberate — but re-check before
  any scale-out): `--workers 1`, in-process queues/rate-limits without Redis,
  per-call SQLite connections re-running `CREATE TABLE IF NOT EXISTS`.
