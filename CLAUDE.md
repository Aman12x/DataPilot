# CLAUDE.md — orientation for coding sessions

Read this before changing anything. It covers the layout, the invariants that
are easy to break, and the decisions that look wrong until you know why.

For the product pitch and eval scores, see [README.md](README.md). For the
production runbook and the reasoning behind the operational subsystems, see
[docs/production-operations.md](docs/production-operations.md).

## Running things

```bash
./venv/bin/python -m pytest tests/ -m "not integration and not slow" -q
```

**Use `./venv/bin/python`, not the system Python** — the system interpreter is
missing `duckdb` and `jose`, and 12 test files fail to collect without them.

- `-m integration` needs live Redis and Postgres containers; CI runs them.
- `-m slow` downloads the MiniLM model or calls the LLM.
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
slow call freezes every other request. Pandas, DuckDB, PBKDF2, `requests`, and
any DB connect go through `asyncio.to_thread`. Graph execution has its own pool
(`run_manager._get_graph_executor`) so analyses can't be starved by other work.

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
record nothing because metering lived at the call sites.

## Decisions that look wrong but aren't

**`client_ip` reads the *leftmost* `X-Forwarded-For` entry.** The generic
advice is to count from the right. That is wrong here, and it was measured:
Railway sends `<client>, <railway-edge>` where the edge address is *public* and
rotates per request, and it **replaces** any inbound header (a request carrying
`X-Forwarded-For: 9.9.9.9` arrives without it). Counting from the right keys
every request to a different bucket and silently disables all rate limiting.
`TRUSTED_PROXY_HOPS` pins an exact position for other topologies.

**Guest budgets key on IP, not `user_id`.** `POST /auth/guest` mints a fresh
`guest-{uuid4}` on demand, so a user-keyed cap resets for free.

**Checkpoint age comes from the `checkpoint_id`.** The table has no timestamp
column, but LangGraph ids are UUIDv6 with an embedded clock (verified against
the `ts` inside the msgpack blob, 40/40). Reading the id avoids decoding every
blob just to learn its age.

**`_get_columns_duckdb` uses `information_schema`, not `pragma_table_info(?)`.**
DuckDB re-parses that bound value as a qualified name, so it fails on any name
containing a quote. Parameterising it is not enough.

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

**A CSP test without a deliberate violation proves nothing.** A detector that
never fires and a policy that never blocks look the same from the outside.
`csp-render.spec.ts` injects an inline script and asserts it *is* blocked.

**Tests import two different module trees.** `tests/test_api.py` uses `api.*`
(because `backend/` is on `sys.path`); newer tests use `backend.api.*`. They are
*separate module objects* with separate state — monkeypatching one does not
affect the other.

**`MODEL` only affects intent resolution** (`nodes_intent.py`), a small JSON
call. Everything else uses `FAST_MODEL`.

## Known-open issues

- **Backups live on the same volume as the data.** They cover corruption and bad
  deletes, not disk loss.
