# CLAUDE.md — orientation for coding sessions

Read this before changing anything. It covers the layout, the invariants that
are easy to break, and the decisions that look wrong until you know why.

For the product pitch and eval scores, see [README.md](README.md). For the
production runbook and the reasoning behind the operational subsystems, see
[docs/production-operations.md](docs/production-operations.md).

## Running things

```bash
make test-fast   # ~1.5 min — inner loop
make test        # ~5 min  — exactly what CI runs. Use this before pushing.
```

**`make test-fast` passing does not mean CI passes.** It deselects `slow`, and
CI does not — `ci.yml` runs `-m "not integration"`, so 11 tests
(`test_eval_tools.py`, `test_semantic_cache_isolation.py`,
`test_sql_generation_eval.py`) run *only* in CI under the fast command. That gap
is where a local-green/CI-red divergence hides, and it is the whole reason
`make test` exists as a separate target. `make test` is the same command string
as `ci.yml`, so the two cannot drift silently.

**Use `./venv/bin/python`, not the system Python** — the system interpreter is
missing `duckdb` and `jose`, and 12 test files fail to collect without them. The
Makefile picks the venv automatically when it is present; the raw commands are
`./venv/bin/python -m pytest tests/ -q -m "not integration"` and the same with
`and not slow`.

**Adding or upgrading a Python dependency: edit `backend/requirements.in`, then**

```bash
uv pip compile backend/requirements.in -o backend/requirements.txt \
    --python-version 3.13 --universal
```

`requirements.txt` is generated — a hand edit is lost on the next compile, and
`tests/test_requirements_integrity.py` fails if the two drift. Only *direct*
dependencies belong in the `.in`; everything else is derived. Pinning a
transitive package by hand is what caused `pydantic_core` vs `pydantic`,
`wrapt` vs `langfuse`, and `tokenizers` vs `transformers` — the last of which
merged and broke the production image build. The resolver picks compatible sets;
Dependabot bumping one pin in isolation cannot.

**Don't merge Dependabot's *pip* PRs; redo the bump on `main`.** Its compiler
ignores `--universal`, so the PR's `requirements.txt` loses every environment
marker (the Linux-only CUDA/triton wheels torch pulls in then break
`pip install -r` on a Mac) and drops `setuptools`. #68 and #71 both did it.
Bump the pin in the `.in`, recompile with the command above plus
`--constraint <previous requirements.txt minus the bumped line>` so nothing
else moves, diff the package set ignoring markers to prove that, push, close the
PR as superseded. Dependabot's GitHub-actions and npm PRs are fine to merge.

- `-m integration` needs live Redis and Postgres containers; CI runs them, and
  `make test-all` runs everything if you have them up.
- `-m slow` downloads the MiniLM model or calls the LLM. Only
  `test_sql_generation_eval.py` needs `ANTHROPIC_API_KEY` and it skips itself
  without one — the other 10 just want the model, which is why they are runnable
  locally and worth running before you push.
- Frontend E2E: `cd frontend && npx playwright test`.
- Against the deployed app: `npx playwright test --config=playwright.prod.config.mjs`.

## Layout

| Path | What lives there |
|---|---|
| `backend/api/` | FastAPI app. `main.py` owns the lifespan; routes in `routes/`. |
| `agents/analyze/` | LangGraph nodes, split by stage (`nodes_sql`, `nodes_analysis`, `nodes_narrative`, `nodes_intent`, `nodes_cache`). `graph.py` wires them. |
| `agents/` | Cross-cutting: `pricing.py`, `spend.py`, `log_safety.py`, `tracer.py`. |
| `tools/` | Pure stats/analysis functions. `db_tools.py` owns all SQL construction. |
| `auth/`, `memory/` | SQLite stores for accounts, workspaces, run history, semantic cache. |
| `config/` | MetricConfig and analysis defaults. |

`agents/analyze/nodes_analysis.py` starts with
`globals().update(vars(_shared))` — anything defined in `node_shared.py` is
available there unqualified. Surprising, but load-bearing.

## Invariants

**Never interpolate an identifier into SQL.** Use
`tools.db_tools.quote_ident(name, backend)`. It *escapes* (doubles the
delimiter) rather than allowlisting, because uploaded CSV headers are
normalised with a Unicode-aware `[^\w]` substitution — `café`, `日本`, and
`2024_revenue` are all legitimate column names that an
`^[a-zA-Z_][a-zA-Z0-9_]*$` allowlist would reject. `_SAFE_IDENT_RE` still
exists for the few places quoting is impossible (BigQuery's
`project.dataset.table` path); prefer `quote_ident` everywhere else.

And when the thing is a *value* rather than an identifier — a schema or table
name in an `information_schema` predicate — bind it. `_get_*_mysql` used to
interpolate it into a string literal behind `_SAFE_IDENT_RE`, which is both the
weaker mechanism and the one that rejects `2024_revenue`.

**Nothing blocking on the event loop.** The backend runs `--workers 1`, so one
slow call freezes every other request. Pandas, DuckDB, PBKDF2, `requests`, any DB
connect, **every `auth`/`memory`/`workspace_store` call, `graph.get_state`, and
the reportlab PDF render** go through `asyncio.to_thread`. Graph execution has
its own pool (`run_manager._get_graph_executor`) so analyses can't be starved by
other work.

A sync `def` route handler is *also* correct — FastAPI runs those in its own
threadpool — which is why `list_runs` and the `deps.py` dependencies never
needed changing. `tests/test_event_loop_blocking.py` walks the AST of
`backend/api/` and fails on any blocking call inside an `async def`; a hand-grep
for `graph.get_state` found six sites, that scan found fifty-six.
`tests/test_event_loop_liveness.py` is the behavioural half: it slows one store
call and asserts an unrelated request completes *before* it finishes.

**User content is delimiter-wrapped before it reaches a prompt**, via
`agents/analyze/prompt_safety.wrap_untrusted_content`. Directives go *before*
the wrapped block, never after — a trailing imperative reads as a continuation
of whatever was injected.

**User content stays out of logs.** INFO records become Sentry breadcrumbs, so a
verbatim log ships customer data to a third party. Use
`agents.log_safety.redact()` for identifiers and free text, and
`redact_exception()` for any exception raised in the agent layer — pandas
raises `KeyError: 'revenue_usd'` with a column from someone's upload, and DuckDB
quotes the failing SQL. `redact_exception` keeps the exception *class* (the
operational signal) and drops the message.

**No raw exception ever reaches a logger under `agents/`** — no carve-out for
`debug`, so there is no rule to remember.
`tests/test_log_safety.py::test_agent_layer_does_not_log_raw_exceptions` scans
the tree and fails on any new one. It caught six sites the original hand-written
grep missed. Infrastructure logs under `backend/api/` are exempt: they carry no
customer data and read better in full.

**The request must never end on an assistant turn.** `generate_narrative` is
the only multi-turn call. It appends its own output to `conversation_history`,
which is appended after the task prompt, so an analyst-requested revision used
to end the request on the previous narrative. Sonnet 4.6+ and Opus 4.6+ reject
that outright; Haiku 4.5 **accepts it and continues the old narrative** instead
of rewriting — measured, and the quieter of the two failures.
`nodes_narrative._conversation_turns` normalises the history and writes the
result back to state, so no producer has to remember the rule.

**Never index `response.content[0]`.** Use
`agents.llm_response.response_text()`. Any model with adaptive thinking returns
`[ThinkingBlock, TextBlock]`, and `content[0].text` raises `AttributeError`
inside the node, which reads like a node bug rather than a model mismatch.

**Environment posture is `environment.is_deployed()`, never an env *name*.**
The old `ENV in ("production", "prod")` was an allowlist of strict
environments, so it failed *open*: `staging` got insecure cookies, no HSTS, and
`allow_origins=["*"]`. `environment.py` inverts it — only known-local names are
local, everything else is deployed.

**A worker thread can only be stopped between nodes.** `asyncio.wait_for`
cancels the coroutine, never the thread. `run_manager` passes a
`threading.Event` into the worker and checks it at each `graph.stream()`
boundary, so a timed-out or shutting-down run stops after one more node instead
of running the whole graph. The admission slot is held until the thread really
exits — `_MAX_CONCURRENT` sizes both the cap and the executor, so releasing
early admits a run onto a busy worker and it silently queues.

**Access checks re-read membership; they never trust a stored identity.**
`_user_can_mutate_connection` granted write on creator identity alone, so
demotion to analyst or removal from the workspace took nothing away — the person
could still rewrite the host and rotate the stored password. A workspace
connection is now owner-only, symmetric with creation.

**Every LLM call is metered.** `_anthropic_client()` returns a wrapper that
prices each response. Don't reach around it — four of seven call sites used to
record nothing because metering lived at the call sites. Two ways it still
silently stops working: the meter is a **contextvar**, so any
`ThreadPoolExecutor` you create yourself must submit via
`contextvars.copy_context().run` (LangGraph's executor does this; the intent/
config-inference pool did not and priced to nowhere); and an LLM call that runs
**outside the graph** — `POST /runs/{id}/deck` — is outside `run_manager`'s
`spend.meter` too, so it needs its own meter, `record_spend` to the caller's
scope, and a rate bucket (a failed generation is not persisted, so without one
every retry is a fresh paid call).

**Pushdown SQL is built for four dialects and for types pandas coerced.** In
`tools/pushdown.py` go through `_f()` (explicit `CAST … AS DOUBLE/DOUBLE
PRECISION/FLOAT64`; BOOLEAN columns — what `read_csv_auto` makes of a
true/false flag — take the `CASE WHEN col THEN 1.0 ELSE 0.0 END` form, found by
`probe_bool_columns`). `1.0 * col` looked portable and was DECIMAL arithmetic:
Σy³ of a BIGINT overflows DECIMAL(38), and it is a binder error on BOOLEAN. Wrap
the approved statement with `db_tools.nestable_sql()` — `rstrip(';')` misses
`SELECT …; -- note` — and never let a failed `count_rows` veto the analyst's
query (a derived table has stricter rules than standalone execution); fall back
to materializing. When you add a check to `_validate_query_content`, mirror it
in `_validate_sufficient_stats`: pushdown mode never sees the frame, and an
event-level extract from a missing GROUP BY is the canonical reason a result
crosses the pushdown threshold in the first place.

**The lookup fast path skips both human gates and the audit.** Only the
classifier's "lookup" *or* `_is_lookup_task` can put a run on it. Keep the regex
conservative: anything that reads as a comparison or a cut (`vs`, `by variant`,
`per region`, `treatment`, `over time`) is analysis, however the sentence opens
— "what was the average revenue per user by variant" shipped as an approved
two-row table before `_ANALYSIS_RE` learned those tokens.

**`audit_result is None` means "passed" only when the audit was not
attempted.** The fast-lookup path legitimately has nothing to audit; an audit
that ran and did not complete (truncated, malformed JSON, API error) sets
`audit_skipped`, appends a visible "Audit unavailable" note to the draft, and
`log_run` records `audit_passed=0`. The audit's `max_tokens` scales with the
draft — it must echo the whole narrative back on a moderate finding — and the
in-place patcher only replaces a quote that occurs exactly once; an ambiguous
quote rides to the gate rather than editing the wrong occurrence.

## Decisions that look wrong but aren't

**`client_ip` reads the *leftmost* `X-Forwarded-For` entry.** The generic
advice is to count from the right. That is wrong here, and it was measured:
Railway sends `<client>, <railway-edge>` where the edge address is *public* and
rotates per request, and it **replaces** any inbound header (a request carrying
`X-Forwarded-For: 9.9.9.9` arrives without it). Counting from the right keys
every request to a different bucket and silently disables all rate limiting.
`TRUSTED_PROXY_HOPS` pins an exact position for other topologies.

**Every guest limit keys on IP, not `user_id`.** `POST /auth/guest` mints a
fresh `guest-{uuid4}` on demand, so any user-keyed limit resets for free.
Budgets, the run and resume rate limits, the per-guest concurrency cap, and
the upload quota all use `budget.scope_for(user_id, ip)`. Uploads need a quota
of their own because they are the one guest action with no LLM spend — the
$-caps never bind — and the retention pass deletes `guest-*` upload dirs after
48 h (guest tokens die in 60 min, so old guest files are unreachable garbage).

**Checkpoint age comes from the `checkpoint_id`.** The table has no timestamp
column, but LangGraph ids are UUIDv6 with an embedded clock (verified against
the `ts` inside the msgpack blob, 40/40). Reading the id avoids decoding every
blob just to learn its age.

**`_get_columns_duckdb` uses `information_schema`, not `pragma_table_info(?)`.**
DuckDB re-parses that bound value as a qualified name, so it fails on any name
containing a quote. Parameterising it is not enough.

**The deck is billed to whoever asked for it, not the run's creator.** A
workspace teammate can open a finished run from history and generate the deck;
charging the creator's scope would let one person drain another's budget.
`_check_run_access(mutate=False)` stays — viewing (and deck generation) is a
teammate action — while the route's `update_state` is only a cache write.

**BigQuery tries the Storage Read API and falls back to REST on its own.** The
client falls back only when the `bigquery-storage` package is absent; a service
account without `bigquery.readsessions.create` raises instead. `_query_bigquery`
catches the Read-API error family (`PermissionDenied`/`Forbidden`/…) and re-reads
the same finished job over REST — a customer SA with plain jobUser/dataViewer
must still get rows. Query failures are not in that list and still propagate.

**Unknown models price at the most expensive known tier.** Under-charging lets
spend slip past the cap; over-charging only trips it early. An unpriced model
also logs a warning — the silent fallback is how a mispriced model hides.

## Traps

**`main.py` is not covered by `tests/test_api.py`.** That file replaces
`app.router.lifespan_context` with a stub, so the real startup and shutdown path
has no coverage there — which is how an `ImportError` on *every* shutdown
shipped. Real lifespan coverage lives in `tests/test_run_lifecycle.py`.

**Test fixtures must match production's SQLite mode.** The app opens databases
in WAL. VACUUM behaves differently under WAL — it writes into the WAL and the
main file keeps its size until a checkpoint. A non-WAL fixture will pass while
production reclaims nothing.

**Playwright has three configs, and the prefix picks one.** The default config
starts the local stack and *excludes* `e2e/prod-*` and `e2e/csp-*`. `prod-*`
drives the deployed app (`playwright.prod.config.mjs`); without the exclusion CI
collects it, points it at `127.0.0.1`, and fails on a cookie that was never set.
`csp-*` needs the production build and its generated CSP header
(`playwright.csp.config.mjs`) — the dev server sends no CSP and injects inline
scripts that `script-src 'self'` would block, so it can never test the policy.
Because CI never runs `prod-*`, a UI copy change can break those selectors
unnoticed; the `Prod Smoke` workflow (daily, plus `workflow_dispatch`) is the
only thing that exercises them. It registers three accounts on production per
run; `retention.prune_test_accounts` deletes them after 48 h.

**The test-account prune is the only one that deletes from `users`.** A
candidate must match a username prefix, an `@example.com` address *and* the age
bound — all three. Any one alone is unsafe: a real user can pick a colliding
username, and a developer can register a throwaway `example.com` address by
hand. The prefixes are matched with `startswith`, never SQL `LIKE`, because they
come from an env var and `_`/`%` are wildcards that would widen a `DELETE`.

**A CSP test without a deliberate violation proves nothing.** A detector that
never fires and a policy that never blocks look the same from the outside.
`csp-render.spec.ts` injects an inline script *and* an inline style attribute and
asserts both are blocked — separate directives, so one does not cover the other.

**CSP does not police the CSSOM.** `node.style.foo = …` — what React and Recharts
do — is unrestricted; only literal `style=` attributes in parsed HTML and
`<style>` blocks are governed. This is why the SPA policy carries no
`'unsafe-inline'` despite being full of `style={{}}` props.

**Deprecations in our own code fail the test run.** `pytest.ini` turns
`DeprecationWarning`/`FutureWarning`/`PendingDeprecationWarning` attributed to
our modules (and Starlette's `StarletteDeprecationWarning`, which subclasses
`UserWarning` and is attributed to `<sys>`) into errors; third-party internals
stay warnings. `--disable-warnings` in CI only hides the summary, so this is
the only thing that stops them accumulating (the `HTTP_413` rename sat for
months). A library deprecation surfacing at *our* call site counts — change the
call. Before acting on one, check the version **CI** resolves; the local venv
has run ahead of the pins before.

**Tests import two different module trees.** `tests/test_api.py` uses `api.*`
(because `backend/` is on `sys.path`); newer tests use `backend.api.*`. They are
*separate module objects* with separate state — monkeypatching one does not
affect the other.

**`MODEL` only affects intent resolution** (`nodes_intent.py`), a small JSON
call. Everything else uses `FAST_MODEL`. If the `MODEL` pin 404s
(`anthropic.NotFoundError`), intent retries once on `FAST_MODEL` and logs at
ERROR; any other error still lands in the safe default.

**SSE resume is by stream id, and only in Redis mode.** Every event the
server yields carries its Redis stream id as the SSE `id`; `useSSE` remembers
the last one and passes `?last_id=` when it opens a fresh `EventSource` (on
resume and on token refresh — a browser auto-reconnect would send
`Last-Event-ID`, which is honoured too). Before that, a reconnect read from `$`
and anything published in the reconnect window — the `narrative_start` reset,
the first deltas of a revision — was gone, which is why the draft is *also*
cleared client-side on the `gate` event. In-memory mode has no ids and cannot
replay, so that class of bug only reproduces in production; keep client state
independent of server events that land in a reconnect window.

**`npx playwright install --with-deps` can hang for the whole job.** The runner
image picks its apt mirror through `mirror+file:/etc/apt/apt-mirrors.txt`, and
`azure.archive.ubuntu.com` has stalled mid-`apt-get update` with no timeout
(four runs on 2026-08-19, all killed by the job limit — conclusion "cancelled",
e2e never ran). Both workflows now rewrite `apt-mirrors.txt` to
`archive.ubuntu.com`, run `apt-get update` under a hard timeout with a
kill-and-retry (a timed-out first attempt leaves a child `apt-get` holding the
lock), then `install-deps` and `install` separately. Editing `ubuntu.sources`
alone does nothing. Keep that step when touching the workflows.

**`/health` reports the deployed commit** (`"commit"`, from
`RAILWAY_GIT_COMMIT_SHA`). `curl -s …/health | jq -r .commit` against
`git rev-parse origin/main` is the deploy check; `railway deployment list
--json` is the fallback. Both Railway services (`DataPilot` backend,
`pretty-emotion` frontend) deploy every push to `main`.

## Known-open issues

Detail — why each is open, the intended fix, and how to verify — lives in
`docs/future-work.md` (local, untracked). Keep the two in sync.

- **Per-stage eval is partial.** `evals/sql_generation_eval.py` (LLM-live,
  manual/nightly, not per-PR) now scores SQL generation and table choice per
  stage — baseline 20/20 strict on claude-sonnet-5 — and `score_faithfulness`
  no longer fails open. Still ungated: intent routing, audit catch rate, and
  the production `eval_score` mislabelling completeness as quality.
- **Pushdown parity is proven on DuckDB and Postgres** (`tests/test_pushdown.py`,
  `tests/test_pushdown_postgres_integration.py` in the integration job); MySQL
  and BigQuery share the builders but have no integration test.
