"""Analyze graph nodes — cache."""
from __future__ import annotations

import agents.analyze.node_shared as _shared
globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

# ── Node 1: check_semantic_cache ──────────────────────────────────────────────

@observe(name="check_semantic_cache")
def check_semantic_cache(state: AgentState) -> dict:
    task        = state.get("task", "")
    fingerprint = state.get("duckdb_path", "")  # empty string for demo DB
    hit  = semantic_cache.check_cache(
        task, "generate_sql", dataset_fingerprint=fingerprint, user_id=state.get("user_id")
    )
    if hit is None:
        return {}
    cached    = hit["result"]
    narrative = cached.get("narrative", "")
    hit_type  = hit.get("hit_type", "hard")   # "hard" (>0.92) or "soft" (0.80-0.92)
    return {
        "semantic_cache_hit":        True,
        "semantic_cache_similarity": hit.get("similarity", 0.0),
        "semantic_cache_accepted":   False,     # analyst hasn't decided yet
        "generated_sql":             cached.get("sql", ""),
        # Restore narrative so the cache path can show the full result on acceptance.
        "narrative_draft":           narrative,
        "recommendation":            cached.get("recommendation", ""),
        "final_narrative":           narrative,
        # hit_type drives the gate label so analyst knows whether this is mandatory review
        "semantic_cache_hit_type":   hit_type,
    }


# ── Node 1b: semantic_cache_gate (HITL interrupt — hard cache hit only) ──────

@observe(name="semantic_cache_gate")
def semantic_cache_gate(state: AgentState) -> dict:
    """
    Interrupt when the semantic cache returns a hard hit (similarity > 0.92).
    Asks the analyst: "Use cached result, or re-run analysis?"
    If accepted: the graph routes directly to log_run, skipping all computation.
    If declined: the graph continues normally from inject_history.
    """
    hit_type   = state.get("semantic_cache_hit_type", "hard")
    similarity = state.get("semantic_cache_similarity", 0.0)
    hit_label  = "identical" if hit_type == "hard" else "very similar"
    payload = {
        "gate":             "semantic_cache",
        "hit_type":         hit_type,
        "similarity":       similarity,
        "generated_sql":    state.get("generated_sql", ""),
        "narrative_draft":  state.get("narrative_draft", ""),
        "recommendation":   state.get("recommendation", ""),
        "message": (
            f"This task looks {hit_label} to a prior analysis "
            f"(similarity={similarity:.2f}). "
            "Use the cached result, or re-run the full analysis?"
        ),
    }
    analyst_response = interrupt(payload)
    accepted = analyst_response.get("approved", False)
    return {"semantic_cache_accepted": accepted}


# ── Node 2: inject_history ─────────────────────────────────────────────────

@observe(name="inject_history")
def inject_history(state: AgentState) -> dict:
    task    = state.get("task", "")
    user_id = state.get("user_id")
    history = retriever.retrieve_relevant_history(task, user_id=user_id)
    return {"relevant_history": history}


# ── Node 3: load_schema ───────────────────────────────────────────────────────

@observe(name="load_schema")
def load_schema(state: AgentState) -> dict:
    task      = state.get("task", "")
    is_upload = bool(state.get("duckdb_path"))

    # Uploads always get a fresh schema inspection — each file is unique and
    # must never read from or write to the shared demo-DB cache.
    refresh = is_upload or "schema changed" in task.lower() or "refresh schema" in task.lower()

    if not refresh and os.path.exists(_SCHEMA_CACHE_PATH):
        try:
            with open(_SCHEMA_CACHE_PATH) as f:
                cached = json.load(f)
            schema_context = cached["schema_context"]
        except (KeyError, json.JSONDecodeError):
            schema_context = None  # fall through to re-fetch
    else:
        schema_context = None

    if schema_context is None:
        schema_context = _db_conn(state).inspect_schema()
        if not is_upload:
            # Only cache the shared demo-DB schema, never per-upload schemas.
            os.makedirs(os.path.dirname(_SCHEMA_CACHE_PATH), exist_ok=True)
            with open(_SCHEMA_CACHE_PATH, "w") as f:
                json.dump({"schema_context": schema_context}, f, indent=2)

    # Prepend SQL dialect so the LLM never has to guess the engine.
    # This is the single most effective guard against dialect-specific syntax errors.
    backend = state.get("db_backend", "duckdb")
    dialect = "DuckDB SQL" if backend == "duckdb" else "PostgreSQL"
    schema_context = f"-- Dialect: {dialect}\n\n{schema_context}"

    # Load MetricConfig once here — all subsequent nodes read from state
    mc = state.get("metric_config") or load_metric_config()

    return {
        "schema_context": schema_context,
        "metric_config":  mc,
        "metric":         mc.primary_metric,
        "covariate":      mc.covariate,
        # Wipe Postgres credentials from the checkpoint immediately after use.
        # They are only needed for _db_conn(); keeping them in state leaks
        # them into the SQLite/Postgres checkpoint file on disk.
        "pg_password": "",
        "pg_user":     "",
        "pg_host":     "",
        "pg_dbname":   "",
    }

