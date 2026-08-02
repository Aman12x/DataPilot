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

## 5. Scope connections beyond a single dataset

**What:** a saved connection covers exactly one BigQuery dataset or one
MySQL schema (`tools/db_tools.py::_get_tables_bigquery` /
`_get_tables_mysql`). Postgres now spans all non-system schemas (fixed
alongside this note), but BigQuery and MySQL cannot see sibling
datasets/schemas, and there is no way to widen or narrow what any
connection covers.

**Why:** real warehouses split data across datasets (`raw`, `analytics`,
`marts`); a one-dataset connection forces users to pick one slice and lose
joins across them. The opposite failure also matters: "everything" on a
large warehouse would blow up the schema context (already truncated at
20K chars in `nodes_narrative`) and degrade SQL quality, so the goal is a
*chosen* scope, not an unbounded one.

**How, in order:**
1. *Schema plumbing:* emit qualified names for BigQuery
   (`dataset.table` via `client.list_datasets()`) and MySQL sibling
   schemas, mirroring the Postgres pattern: bare names for the
   connection's home dataset/schema, qualified otherwise. The SQL
   validator already accepts dotted names (`node_shared._validate_sql_references`),
   and `_split_pg_table`-style helpers cover quoting in sampling.
2. *Scope picker:* store a `schemas` list (JSON) on `db_connections`;
   at save/edit time in ConnectionsPanel, run the connection test, then
   show the discovered datasets/schemas as checkboxes (default: the home
   one). `inspect_schema` filters discovery to the stored scope.
3. *Ripples:* annotations and metric packs key by table name — qualified
   names work as plain strings, but document that annotations for
   non-home schemas must use the qualified name. The drift `schema_hash`
   changes meaning when scope changes; reset the snapshot on scope edit
   (same pattern as the credential-change health reset).

**Verify:** unit tests mirroring `test_postgres_tables_span_all_schemas`
for BigQuery/MySQL; a scope-picker flow test in the CSP sweep's Sources
modal; live check that a two-dataset BigQuery connection can join across
datasets through the SQL gate.

---

## 6. Verified-query repository

**What:** persist SQL that a human approved at the query gate, keyed by the
question (embedding), and retrieve it as few-shot context for future
generations. The semantic cache already stores question embeddings and
cached results; this extends the same machinery to certified SQL.

**Why:** the strongest accuracy lever in the field. Snowflake Cortex
Analyst reaches ~90% (vs ~51% raw-schema prompting) largely on a semantic
model plus a Verified Query Repository; LinkedIn's certified example
notebooks are the same idea. DataPilot already produces the raw material:
every gate-approved query is a human-verified question-to-SQL pair that
currently evaporates after the run.

**How:** two intake paths into one repository, both keyed by
(task/name, embedding, SQL, connection, schema fingerprint):
1. *Automatic:* on query-gate approval, store the approved SQL — every
   gated run is a free human-verified pair.
2. *User-contributed:* let users paste their org's canonical queries with
   a name/description (ConnectionsPanel or PackStudio is the natural
   home) — the "teach it how we write queries" path. Uber's custom
   workspace samples and LinkedIn's self-serve indexed example queries
   are this lever; Meta goes further and mines all query history into
   per-user context.
Retrieval: in `generate_sql`, top-k matching examples for the connection
whose schema fingerprint still matches, added to the few-shot block
(`_filter_few_shot_by_schema` already guards stale tables). Invalidate on
schema drift via the snapshot hash.

**Caution from Anthropic's internal build:** raw retrieval over thousands
of historical queries moved their accuracy by less than one point; the
gains came from *curated* distillations (skills, semantic layer,
21% to 95%). So keep contributed queries few, named, and deliberate —
exemplars, not a query-log dump — and treat volume mining as raw material
for future curation, not direct context.

**Verify:** unit tests for store/retrieve/invalidation; measure SQL-gate
edit rate before/after on the demo datasets. **Do item 8 first** — the
current eval gate cannot see SQL-generation quality at all.

---

## 7. Table retrieval and column pruning before generation

**What:** a pre-generation stage that selects relevant tables (and prunes
irrelevant columns) instead of shipping the entire schema context,
which is truncated at 20K chars today.

**Why:** every serious system converged here after trying give-the-model-
everything: Uber's Intent/Table/Column-Prune agents, LinkedIn's
retrieve-20 → rank-to-7 pipeline. Currently harmless at demo scale, it
becomes the binding constraint once multi-schema Postgres (shipped) and
the BigQuery/MySQL scope picker (item 5) widen discovery — a curated
scope dilutes into noise without retrieval.

**How:** cheap first version, no embeddings: an LLM ranking call that
receives the task plus the table list with one-line summaries (name,
row-count note, annotation headline) and returns the relevant subset;
only those tables' full column blocks go into the SQL prompt. Column
pruning can wait — table-level selection captures most of the win.
Sequence after item 5, or land it first so item 5's wider scopes arrive
pre-filtered.

**Verify:** golden-question eval comparing SQL quality with and without
retrieval on a multi-schema fixture; assert token count of the
generation prompt drops on wide schemas. **Do item 8 first** — without
it this change is unmeasurable.

---

## 8. Component-level golden-question eval set — PREREQUISITE for 6 and 7

**What:** a few dozen golden questions against the demo datasets with
expected intent, tables, and result shape, scored per pipeline stage
(intent routing, table choice, SQL validity/execution, audit catch rate)
rather than end-to-end only. **Build this before items 6 and 7**, plus
two hardening fixes to the existing scorers.

**Why (audit, 2026-08-02):** the four offline harnesses in `evals/` are
solid for what they cover — deterministic, CI-gated at 2% tolerance
against a committed baseline — but all of them call the stats tools
directly on DataFrames. Intent routing, table selection, SQL generation,
and gate behavior are entirely outside the regression gate, which is
precisely the layer items 6 and 7 change. Either could ship, regress SQL
quality, and every gated number would stay green. Uber's
vanilla/decoupled split and LinkedIn's 133-question internal benchmark
exist for exactly this reason; Anthropic's ablation discipline (vary one
component against a pinned eval set) is the acceptance test for 6 and 7.

**Hardening fixes to land with it:**
1. *Fail-open faithfulness:* `score_faithfulness` returns 1.0 both when
   the narrative has no numbers and when there are no reference values —
   "could not verify" scores identically to "verified". Return
   `score: None` in the no-reference case and exclude it from
   composites.
2. *Completeness masquerading as quality:* the in-band production
   `eval_score` is 60% "did the expected nodes produce any result"; the
   History UI renders it as "High quality". Label it as completeness
   wherever it surfaces, or split the two numbers.

**How:** a fixtures file of (question, mode, expected tables, expected
result predicate); a slow-marked pytest harness that runs the graph
against the demo DuckDB with the LLM live, recording per-stage outcomes;
a small report script comparing runs. Start with ~20 questions spanning
lookup/exploratory/A-B modes, and include **one multi-schema fixture DB**
so item 7's pruning benefit is demonstrable. Keep the existing
deterministic gate untouched — this is a new layer above it (manual or
nightly, not per-PR), not a replacement.

**Verify:** the harness itself runs green on `-m slow` locally; baseline
numbers recorded in DEVLOG so items 6 and 7 land with before/after
evidence rather than assertions.

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
