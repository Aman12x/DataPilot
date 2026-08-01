"""Analyze graph nodes — cache + schema load (Phase 2 soft cache / annotations)."""
from __future__ import annotations

import agents.analyze.node_shared as _shared
globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

# ── Node 1: check_semantic_cache ──────────────────────────────────────────────

@observe(name="check_semantic_cache")
def check_semantic_cache(state: AgentState) -> dict:
    task = state.get("task", "")
    # Prefer the Phase-2 fingerprint when already computed; else fall back to
    # a provisional fingerprint from durable IDs (schema hash filled in later).
    fingerprint = state.get("dataset_fingerprint") or ""
    if not fingerprint:
        from agents.analyze.semantic_layer import compute_dataset_fingerprint
        fingerprint = compute_dataset_fingerprint(
            connection_id=state.get("connection_id") or "",
            duckdb_path=state.get("duckdb_path") or "",
            metric_pack_id=state.get("metric_pack_id") or "",
        )
    hit = semantic_cache.check_cache(
        task, "generate_sql", dataset_fingerprint=fingerprint, user_id=state.get("user_id")
    )
    if hit is None:
        return {"dataset_fingerprint": fingerprint}
    cached = hit["result"]
    narrative = cached.get("narrative", "")
    hit_type = hit.get("hit_type", "hard")
    return {
        "dataset_fingerprint":       fingerprint,
        "semantic_cache_hit":        True,
        "semantic_cache_similarity": hit.get("similarity", 0.0),
        "semantic_cache_accepted":   False,
        "generated_sql":             cached.get("sql", ""),
        "narrative_draft":           narrative,
        "recommendation":            cached.get("recommendation", ""),
        "final_narrative":           narrative,
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
    task = state.get("task", "")
    user_id = state.get("user_id")
    history = retriever.retrieve_relevant_history(
        task,
        user_id=user_id,
        metric_pack_id=state.get("metric_pack_id") or None,
        connection_id=state.get("connection_id") or None,
    )
    return {"relevant_history": history}


# ── Node 3: load_schema ───────────────────────────────────────────────────────

def _connection_schema_cache_path(connection_id: str) -> str:
    base = os.getenv("SCHEMA_CACHE_DIR") or os.path.join(
        os.path.dirname(_SCHEMA_CACHE_PATH) or "memory", "schema_cache"
    )
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in connection_id)[:64]
    return os.path.join(base, f"{safe}.json")


@observe(name="load_schema")
def load_schema(state: AgentState) -> dict:
    from agents.analyze.semantic_layer import (
        compute_dataset_fingerprint,
        detect_pack_drift,
        schema_hash,
        strip_sql_dialect_header,
    )

    task = state.get("task", "")
    is_upload = bool(state.get("duckdb_path"))
    connection_id = state.get("connection_id") or ""
    user_id = state.get("user_id") or ""
    is_byo_db = bool(connection_id) or (
        state.get("db_backend") == "postgres" and bool(state.get("pg_host"))
    )

    force_refresh = (
        "schema changed" in task.lower()
        or "refresh schema" in task.lower()
    )

    # Resolve saved column annotations for BYO connections
    annotations: dict = {}
    synonyms: dict = {}
    if connection_id and user_id:
        try:
            from auth.workspace_store import get_annotations
            ann = get_annotations(user_id, connection_id)
            if ann:
                annotations = ann.annotations or {}
                synonyms = ann.synonyms or {}
        except Exception as exc:
            logger.debug("load_schema: annotations lookup failed: %s", exc)

    schema_context = None
    cache_path = None
    if connection_id and not force_refresh:
        cache_path = _connection_schema_cache_path(connection_id)
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                schema_context = cached.get("schema_context")
            except (KeyError, json.JSONDecodeError, OSError):
                schema_context = None
    elif not is_upload and not is_byo_db and not force_refresh:
        if os.path.exists(_SCHEMA_CACHE_PATH):
            try:
                with open(_SCHEMA_CACHE_PATH) as f:
                    cached = json.load(f)
                schema_context = cached.get("schema_context")
            except (KeyError, json.JSONDecodeError):
                schema_context = None

    if schema_context is None:
        schema_context = _db_conn(state).inspect_schema(annotations=annotations or None)
        # Soft-cache per connection; demo keeps the shared file cache.
        if connection_id:
            cache_path = cache_path or _connection_schema_cache_path(connection_id)
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump({
                        "schema_context": schema_context,
                        "schema_hash": schema_hash(schema_context),
                    }, f, indent=2)
            except OSError as exc:
                logger.debug("load_schema: could not write connection cache: %s", exc)
            if user_id:
                try:
                    from auth.workspace_store import record_schema_snapshot
                    record_schema_snapshot(
                        user_id, connection_id,
                        schema_context=strip_sql_dialect_header(schema_context),
                        schema_hash=schema_hash(schema_context),
                    )
                except Exception as exc:
                    logger.debug("load_schema: snapshot persist failed: %s", exc)
        elif not is_upload and not is_byo_db:
            os.makedirs(os.path.dirname(_SCHEMA_CACHE_PATH) or ".", exist_ok=True)
            with open(_SCHEMA_CACHE_PATH, "w") as f:
                json.dump({"schema_context": schema_context}, f, indent=2)

    # Prepend SQL dialect so the LLM never has to guess the engine.
    backend = state.get("db_backend", "duckdb")
    dialect = "DuckDB SQL" if backend == "duckdb" else "PostgreSQL"
    schema_context = f"-- Dialect: {dialect}\n\n{schema_context}"

    # Append synonym hints when present
    if synonyms:
        from agents.analyze.semantic_layer import format_synonyms
        syn_block = format_synonyms(synonyms)
        if syn_block:
            schema_context = (
                f"{schema_context}\n\n-- Business synonyms\n{syn_block}"
            )

    mc = state.get("metric_config") or load_metric_config()

    # Pack-vs-live drift
    drift_warnings: list[str] = []
    if state.get("metric_pack_id") and mc:
        drift_warnings = detect_pack_drift(mc, schema_context)

    pack_version = state.get("metric_pack_version")
    fingerprint = compute_dataset_fingerprint(
        connection_id=connection_id,
        duckdb_path=state.get("duckdb_path") or "",
        metric_pack_id=state.get("metric_pack_id") or "",
        pack_version=pack_version if isinstance(pack_version, int) else None,
        schema_context=schema_context,
    )

    # Certified packs with material drift must re-open the metric gate.
    force_metric_gate = bool(drift_warnings) and bool(state.get("metric_pack_certified"))

    return {
        "schema_context": schema_context,
        "metric_config":  mc,
        "metric":         mc.primary_metric,
        "covariate":      mc.covariate,
        "schema_drift_warnings": drift_warnings,
        "dataset_fingerprint": fingerprint,
        "force_metric_gate": force_metric_gate,
        # Wipe inline Postgres credentials from the checkpoint immediately.
        "pg_password": "",
        "pg_user":     "",
        "pg_host":     "",
        "pg_dbname":   "",
    }
