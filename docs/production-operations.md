# Production operations

How the operational subsystems work, why they are built the way they are, and
what to do when something goes wrong. Written after a hardening pass that fixed
a set of production defects; the reasoning is recorded here so it doesn't have
to be re-derived.

Companion docs: [CLAUDE.md](../CLAUDE.md) for code invariants,
[.env.example](../.env.example) for the full variable list.

---

## Deployment shape

Two Railway services from one repo.

| Service | Serves | Key config |
|---|---|---|
| backend | FastAPI API | volume, DB paths, budgets, `SECRET_KEY` |
| frontend | Static SPA via `serve` | `VITE_API_URL`, `CSP_REPORT_ONLY` |

### The volume

**Mount at `/app/db`. Never `/app/memory`.**

`memory/` is a Python package in the image. Mounting a volume over it replaces
the package and the backend dies at startup with:

```
ImportError: cannot import name 'retriever' from 'memory' (unknown location)
```

The mount alone persists nothing — four variables must point into it:

```
GRAPH_DB_PATH=/app/db/graph.db
AUTH_DB_PATH=/app/db/auth.db
MEMORY_DB_PATH=/app/db/datapilot_memory.db
UPLOAD_DIR=/app/db/uploads
```

Missing `MEMORY_DB_PATH` loses run history on every deploy; missing
`UPLOAD_DIR` loses every user upload and leaves checkpoints pointing at files
that no longer exist.

**Leave `DUCKDB_PATH` alone.** `data/generate_data.py` hardcodes its write to
`/app/data`; only the read path is configurable, so repointing it sends the app
looking for a file nothing creates.

### Boot-time hard failures

The backend refuses to start when deployed if `SECRET_KEY` is missing, shorter
than 32 characters, low-entropy, or a known placeholder (RFC 7518 requires 256
bits for HS256). It also refuses to start in production without `CORS_ORIGINS`
or `APP_URL`.

---

## LLM spend caps

Three daily ceilings, tracked per UTC day in Redis when available and in-process
otherwise:

| Variable | Default | Response when hit |
|---|---|---|
| `LLM_DAILY_BUDGET_USD` | 50 | 503 for everyone |
| `LLM_USER_DAILY_BUDGET_USD` | 5 | 429 for that user |
| `LLM_GUEST_DAILY_BUDGET_USD` | 0.50 | 429 for that IP |

Checked at run creation *and* gate resume — a resume restarts the graph and
spends more tokens.

**Guests are keyed on IP.** `POST /auth/guest` mints a fresh `guest-{uuid4}` on
request, so a user-keyed cap could be reset indefinitely. That makes correct
client-IP resolution part of the spend control, not just rate limiting.

**Billing happens on every exit path**, including failure and timeout — those
tokens were spent regardless. The meter object is created on the async side and
passed into the worker thread so a timed-out run's partial spend still records.

**Accounting caveat.** Costs are computed from a static table in
`agents/pricing.py`. An unpriced model bills at the most expensive known tier
and logs a warning. Prices there are list rates and do not account for
introductory pricing or partner platforms.

---

## Retention and backups

A maintenance pass runs `RETENTION_INTERVAL_SEC` (default daily), 120 seconds
after boot.

**Why it exists:** `graph.db` serialises full query-result DataFrames into every
checkpoint. Locally that reached 331 MB across 507 checkpoints, one run
accounting for 45 MB. It shares a fixed-size volume with `auth.db`, so an
unpruned disk takes user accounts down with it.

Each pass:

1. Drops checkpoint threads idle past `CHECKPOINT_RETENTION_DAYS` (30). Keyed on
   each thread's **newest** checkpoint, so a long-running or resumed analysis is
   never collected mid-flight. Deletes from `writes` too.
2. Trims run history past `RUN_RETENTION_DAYS` (180) and clears spent/expired
   auth tokens.
3. VACUUMs `graph.db` when this pass deleted something **or** the freelist
   exceeds `VACUUM_FREE_BYTES` — the second condition cleans up after a pass
   that deleted without reclaiming.
4. Snapshots `auth.db` and `datapilot_memory.db`, keeping `BACKUP_KEEP` (7).
5. Logs a size breakdown (`sizes_mb`) so growth is visible before the disk fills.

### Two things that bit us

**VACUUM under WAL doesn't shrink the file.** The rebuilt database goes into the
WAL and the main file keeps its size until a checkpoint folds it back. `vacuum()`
runs `PRAGMA wal_checkpoint(TRUNCATE)` afterwards. Without it the space is never
returned to the filesystem and the reported figure is always 0.

**`revoked_tokens` has no `expires_at` column.** It is `(jti, revoked_at)`.
Querying the wrong column raised, the error was swallowed at debug level, and
the one table that grows on every `/auth/refresh` was never pruned. Schema-drift
skips now log at **warning**.

### What backups do and don't cover

Only `auth.db` and `datapilot_memory.db` — small and irreplaceable. `graph.db`
is deliberately excluded: it is transient run state and it is the thing filling
the disk.

Snapshots use `VACUUM INTO`, which produces a consistent copy of a live WAL
database without blocking writers and with no `-wal` sidecar to go missing.
Copying the file instead can capture a torn page.

**They live on the same volume as the data.** That covers corruption, a bad
migration, and accidental deletion — **not** losing the disk. Off-box copies
need external storage and are not implemented.

---

## Client IP resolution

Every per-IP limit — auth rate limiting and the guest spend cap — keys on
`auth_rate.client_ip`.

Measured from the deployed app rather than assumed:

```
xff='74.105.77.244, 152.233.47.65'  peer=100.64.0.3
xff='74.105.77.244, 152.233.47.67'  peer=100.64.0.4
```

Railway sends `<client>, <railway-edge>`; the edge address is **public and
rotates**, and the peer is CGNAT. Railway also **replaces** an inbound
`X-Forwarded-For` — a request sent carrying `9.9.9.9` arrives without it.

So the leftmost entry is both the real client and unforgeable *here*. Two
earlier attempts failed by reasoning from the generic model: counting one hop
from the right picked the CGNAT peer, and skipping infrastructure ranges picked
the rotating public edge. Both gave every request its own bucket and disabled
rate limiting entirely — 20 concurrent bad logins produced zero 429s.

Leftmost is only safe where the edge strips the inbound header. Behind a bare
reverse proxy, set `TRUSTED_PROXY_HOPS` to count from the right instead. Set
`DEBUG_CLIENT_IP=true` for one deploy to log what your proxy actually sends —
remove it afterwards, since it logs client IPs at INFO.

---

## Content-Security-Policy

Two policies.

**API** — `default-src 'none'` with `frame-ancestors`, `base-uri`, and
`form-action` all `'none'`. It returns JSON and SSE and never needs to load
anything. `/docs` and `/redoc` get a scoped exemption because Swagger bootstraps
from an inline script and a jsDelivr bundle.

**SPA** — generated at container start by `frontend/runtime-config.js`, because
`connect-src` must name the API origin and that is only known from
`VITE_API_URL` at runtime. `serve` reads the emitted `dist/serve.json`.

**No `'unsafe-inline'` anywhere in the SPA policy**, for scripts or styles. The
style exemption was there on the belief that React `style={{}}` props need it.
They do not: React and Recharts write through the CSSOM (`node.style.foo = …`),
which CSP does not police at all. Only literal `style=` attributes in parsed HTML
and `<style>` blocks are, and the build emits neither — no
`dangerouslySetInnerHTML`, no `setAttribute("style", …)`, no `<style>` injection
in the shipped bundle.

That was measured, not reasoned: on the finished view, both policies produce
**47 inline-styled elements, 143 DOM nodes, the same computed chart fill, and
zero violations**. Byte-identical rendering.

**Rolling it out:** set `CSP_REPORT_ONLY=true` for one deploy, load the app,
confirm the console is clean, then remove the variable.

**Verification.** `e2e/csp-render.spec.ts` runs against the production build
served with the generated `dist/serve.json`, so the policy under test is the one
the generator emits. It renders all four chart types, exports the CSV blob, and
downloads the PDF, asserting zero violations on each. The Vite dev server cannot
be used for this — it sends no CSP and injects inline scripts that `script-src
'self'` would rightly block.

One test in that file deliberately triggers a violation. Without it, "no
violations" is unfalsifiable: a detector that never fires and a policy that never
blocks are indistinguishable.

**Coverage.** `e2e/csp-sweep.spec.ts` walks every screen the app can render:
the four unauthenticated routes, the register form, the mode picker, both task
forms, the chain-of-thought list, all seven HITL gates, the finished view with
its disclosure expanded, history with a run expanded, and both modals. That
breadth is what justified dropping `'unsafe-inline'`; three screens would not
have.

One test asserts the *opposite* direction — that inline style props are still
computed. A policy that silently stopped applying styles would render the app
unstyled rather than broken, and a violations-only assertion cannot see that.

---

## Email

`EMAIL_FROM` must be a bare address or `Name <addr@domain>`. A bare domain in
the brackets — `DataPilot <example.com>` — is rejected by Resend, and the
failure is invisible: `email.py` swallows the exception and
`POST /auth/forgot-password` returns 202 either way (correct for anti-enumeration,
unhelpful for debugging). **Password reset and signup verification were both
silently broken in production for this reason.** The only symptom was a log line.

The sending domain must also be verified in Resend. `EMAIL_TIMEOUT_SECONDS`
(default 10) tightens the SDK's 30-second default.

---

## Runbook

| Symptom | Likely cause |
|---|---|
| Backend won't boot, `ImportError ... 'memory'` | Volume mounted at `/app/memory` |
| Backend won't boot, `SECRET_KEY` in the error | Missing, short, low-entropy, or placeholder key |
| Run history / uploads vanish on deploy | `MEMORY_DB_PATH` or `UPLOAD_DIR` not set |
| Password reset silently does nothing | `EMAIL_FROM` malformed or domain unverified — check logs |
| Brute force isn't rate limited | Client IP resolving to a rotating address; `DEBUG_CLIENT_IP=true` |
| No chain-of-thought events with Redis on | `_publish_sync` needs the passed-in loop |
| Disk filling | Check `sizes_mb` in the retention log; `graph_free` shows reclaimable bytes |
| Analyses stall while API stays up | Graph executor saturated — `MAX_CONCURRENT_GRAPH_INVOKES` |
| Whole API stalls | Something blocking on the event loop |

Useful commands:

```bash
railway logs -s DataPilot | grep retention.pass   # sizes + what was pruned
railway logs -s DataPilot | grep run.spend        # per-run cost
railway volume list                               # disk usage
```

---

## Scaling limits

Single-instance by design today. `backend/Dockerfile` pins `--workers 1`, and
without `REDIS_URL` the run queues, rate limits, budget counters, and run
ownership are all in-process dicts. Railway volumes cannot attach to multiple
replicas either, so a second replica gets its own `auth.db` and users appear to
vanish depending on which instance answers.

Before scaling out: set `REDIS_URL`, move the databases to Postgres (note
`langgraph-checkpoint-postgres` is currently commented out in
`backend/requirements.txt`, so `DATABASE_URL` moves accounts and history but
leaves checkpoints on local SQLite), and add connection pooling —
`auth/store.py` and `memory/store.py` open a fresh connection and re-run
`CREATE TABLE IF NOT EXISTS` on every call.
