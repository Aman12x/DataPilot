"""Analyze graph nodes — intent."""
from __future__ import annotations

import agents.analyze.node_shared as _shared
from agents.analyze.prompt_safety import wrap_untrusted_content
from agents.log_safety import redact, redact_exception

globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

# ── Lookup-vs-exploratory heuristics ─────────────────────────────────────────
# Used as a fallback when the LLM returns "exploratory" for a task that is
# clearly a simple retrieval/count question.  Prevents "how many TVs were sold?"
# from triggering the full correlation/regression pipeline.

_LOOKUP_RE = re.compile(
    r"^(how\s+many|what\s+is\s+the\s+(total|average|count|number)|"
    r"what\s+was\s+the|what\s+are\s+the\s+(top|bottom|\d+)|"
    r"show\s+(me\s+)?(the\s+)?total|list\s+(the\s+|all\s+)?|"
    r"get\s+(the\s+|me\s+)?|count\s+(of\s+|the\s+)?|"
    r"total\s+(number|count|revenue|sales)|number\s+of\s+)",
    re.IGNORECASE,
)
_ANALYSIS_RE = re.compile(
    r"(why|trend|pattern|correlat|impact|cause|relationship|"
    r"significant|compare|breakdown|investigat|driver|anomal|differ|"
    r"segment|cohort|funnel|retention|churn|uplift|effect)",
    re.IGNORECASE,
)


def _is_lookup_task(task: str) -> bool:
    """Return True when a task looks like a simple retrieval, not an analysis."""
    return bool(_LOOKUP_RE.search(task)) and not bool(_ANALYSIS_RE.search(task))


# ── Helpers for resolve_task_intent ──────────────────────────────────────────

def _llm_resolve_intent(
    task: str,
    schema_context: str,
    mc: MetricConfig,
) -> dict:
    """
    Call the LLM to identify which metric the analyst wants to measure.

    Uses the cached-prefix message pattern (same as generate_sql) and limits
    tokens to 256 — we only need a small structured JSON response.

    Returns (result, cost_info) where result has keys: primary_metric,
    metric_direction, covariate, guardrail_metrics, ambiguous,
    clarifying_question, reasoning.

    Falls back to a safe default (ambiguous=False, mc defaults preserved) on
    any parse failure — never hard-fails.
    """
    # Extract metric-like column names from schema for the prompt
    _, known_columns = _known_schema_names(schema_context)
    available_metrics = ", ".join(sorted(known_columns)) if known_columns else "(schema not available)"

    task_prompt = TASK_INTENT_PROMPT.format(
        task=wrap_untrusted_content(task, label="analyst_task"),
        available_metrics=available_metrics,
        default_metric=mc.primary_metric,
    )
    history_text = ""   # intent resolution doesn't need history injection
    # Canonical schema block: this is usually the first LLM call of a run, so
    # it WRITES the cache entry that SQL gen, corrections, and the narrative
    # then read. It previously cached the unsliced schema — bytes no other
    # call sent.
    messages = _build_cached_messages(_cached_schema_block(schema_context), history_text, task_prompt)

    safe_default = {
        "analysis_mode":       "ab_test",
        "primary_metric":      mc.primary_metric,
        "metric_direction":    mc.metric_direction,
        "covariate":           mc.covariate,
        "guardrail_metrics":   mc.guardrail_metrics,
        "ambiguous":           False,
        "clarifying_question": None,
        "reasoning":           "Defaulting to current metric config.",
    }

    try:
        try:
            with trace_generation("resolve_task_intent", _model(), task_prompt,
                                  max_tokens=256) as gen:
                response = _anthropic_client().messages.create(
                    model=_model(),
                    max_tokens=256,
                    messages=messages,
                )
                cost_info = gen.update(response)
        except anthropic.NotFoundError:
            # A dead MODEL pin is a permanent config error, not a transient
            # failure — falling into safe_default would silently disable
            # intent resolution on EVERY run (this happened: a retired model
            # id sat in the env and intent 404'd for weeks unnoticed).
            # Retry once on the workhorse model and shout.
            logger.error(
                "resolve_task_intent: model %r not found (stale MODEL env pin?) "
                "— retrying on FAST_MODEL %r. Fix the pin.",
                _model(), _fast_model(),
            )
            with trace_generation("resolve_task_intent", _fast_model(), task_prompt,
                                  max_tokens=256) as gen:
                response = _anthropic_client().messages.create(
                    model=_fast_model(),
                    max_tokens=256,
                    messages=messages,
                )
                cost_info = gen.update(response)
        raw = response_text(response).strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        # Ensure all required keys exist, filling from defaults where missing
        for key, default_val in safe_default.items():
            result.setdefault(key, default_val)
        return result, cost_info
    except Exception as exc:
        logger.warning("_llm_resolve_intent: parse failed (%s) — using defaults.", redact_exception(exc))
        return safe_default, {}


def _apply_intent_to_config(
    result: dict,
    mc: MetricConfig,
    schema_context: str,
) -> MetricConfig:
    """
    Apply LLM intent resolution to produce an updated MetricConfig.

    Validates that result["primary_metric"] exists in the schema before
    overriding mc.  If validation fails, returns original mc unchanged.
    Only touches primary_metric, metric_direction, covariate, and
    guardrail_metrics — segment_cols, funnel_steps, and table names are
    owned by the schema-only inference (_llm_infer_config).
    """
    defaults = load_metric_config()
    primary = result.get("primary_metric", "")
    if not primary:
        return mc

    _, known_columns = _known_schema_names(schema_context)
    # If schema is available, validate the resolved metric exists in it
    if known_columns and primary.lower() not in known_columns:
        logger.warning(
            "_apply_intent_to_config: resolved metric %r not in schema — keeping original.",
            primary,
        )
        return mc

    # Build override dict from intent result, preserving mc values for missing fields
    overrides: dict = {"primary_metric": primary}
    if direction := result.get("metric_direction"):
        if direction in ("higher_is_better", "lower_is_better"):
            overrides["metric_direction"] = direction
    if covariate := result.get("covariate"):
        events_cols_cov = _columns_for_table(schema_context, mc.events_table)
        in_events = not events_cols_cov or covariate.lower() in events_cols_cov
        if in_events and (not known_columns or covariate.lower() in known_columns):
            overrides["covariate"] = covariate
        else:
            logger.warning(
                "_apply_intent_to_config: covariate %r not in events table — keeping original.",
                covariate,
            )
    if guardrails := result.get("guardrail_metrics"):
        if isinstance(guardrails, list) and guardrails:
            valid = [g for g in guardrails if not known_columns or g.lower() in known_columns]
            if valid:
                overrides["guardrail_metrics"] = valid

    # metric_source_col should match primary_metric only if primary exists in
    # the events table specifically.  A metric like "dau" may exist in a
    # timeseries table but NOT in events; in that case preserve the original
    # metric_source_col (e.g. "dau_flag") so canonical SQL stays valid.
    if "primary_metric" in overrides:
        events_cols = _columns_for_table(schema_context, mc.events_table)
        if not events_cols or primary.lower() in events_cols:
            overrides["metric_source_col"] = primary
        # else: primary is not an events column — keep original metric_source_col

    try:
        updated = mc.model_copy(update=overrides)
        # Run through sanitise to catch any edge-case mismatches
        updated, warnings = _sanitise_metric_config(updated, schema_context, defaults)
        for w in warnings:
            logger.warning("_apply_intent_to_config: %s", redact(w))
        return updated
    except Exception as exc:
        logger.warning(
            "_apply_intent_to_config: MetricConfig update failed (%s) — keeping original.", exc
        )
        return mc


# ── Schema-only config inference (runs concurrently with intent) ─────────────

def _llm_infer_config(schema_context: str) -> tuple["MetricConfig", dict]:
    """
    LLM infers a MetricConfig from the schema alone — no task input, so it is
    independent of intent resolution and safe to run concurrently with it.

    Returns (config, cost_info). Falls back to defaults on any failure; every
    inferred name is cross-checked against the live schema so a hallucinated
    column cannot propagate to _canonical_experiment_sql().
    """
    defaults = load_metric_config()
    if not schema_context:
        return defaults, {}

    # The schema rides in the canonical cached block — the same bytes the
    # concurrent intent call sends, so whichever call lands first writes the
    # cache entry and the other (plus SQL gen and the narrative) reads it.
    messages = _build_cached_messages(
        _cached_schema_block(schema_context), "", SCHEMA_CONFIG_INFERENCE_PROMPT
    )

    try:
        with trace_generation("infer_metric_config", _fast_model(), SCHEMA_CONFIG_INFERENCE_PROMPT) as gen:
            response = _anthropic_client().messages.create(
                model=_fast_model(),
                max_tokens=512,
                messages=messages,
            )
            cost_info = gen.update(response)
    except Exception as exc:
        logger.warning("_llm_infer_config: LLM call failed (%s), using defaults.", redact_exception(exc))
        return defaults, {}

    try:
        raw = response_text(response).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        inferred = MetricConfig(**json.loads(raw))
    except Exception as exc:
        logger.warning("_llm_infer_config: response parsing failed (%s), using defaults.", redact_exception(exc))
        return defaults, cost_info

    inferred, issues = _sanitise_metric_config(inferred, schema_context, load_metric_config())
    for w in issues:
        logger.warning("_llm_infer_config: schema mismatch — %s", redact(w))
    return inferred, cost_info


# ── Node 3b: resolve_task_intent ──────────────────────────────────────────────

@observe(name="resolve_task_intent")
def resolve_task_intent(state: AgentState) -> dict:
    """
    Reads the analyst's task, identifies the intended metric, and asks one
    clarifying question if the task is genuinely ambiguous.

    Implements Rule 6: ask before assuming on ambiguous tasks.

    For uploads this node also infers a MetricConfig from the schema — the two
    LLM calls have independent inputs (task+schema vs schema alone), so they
    run concurrently instead of as two sequential graph nodes. Intent's
    task-informed resolution is then applied ON TOP of the schema-inferred
    base. The old two-node ordering did the opposite: infer_metric_config ran
    second and overwrote intent's metric resolution wholesale on every upload.
    """
    task           = state.get("task", "")
    schema_context = state.get("schema_context", "")
    mc             = state.get("metric_config") or load_metric_config()

    # Schema-only inference is needed when the data's real shape is unknown —
    # an uploaded file. A saved metric pack means the config is already
    # certified; anything else with a config in state keeps it. This mirrors
    # the retired infer_metric_config node's skip conditions, evaluated
    # against pre-intent state (that node ran after intent had always set
    # metric_config, so its live path reduced to exactly this).
    run_infer = bool(state.get("duckdb_path")) and not state.get("metric_pack_id")

    costs: list[dict] = []
    if run_infer:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="intent") as pool:
            intent_future = pool.submit(_llm_resolve_intent, task, schema_context, mc)
            infer_future  = pool.submit(_llm_infer_config, schema_context)
            result, intent_cost      = intent_future.result()
            inferred_mc, infer_cost  = infer_future.result()
        costs += [intent_cost, infer_cost]
        # Intent resolves against the schema-inferred base, not demo defaults —
        # the base has the upload's real table names and guardrail metrics.
        mc = inferred_mc

    else:
        result, intent_cost = _llm_resolve_intent(task, schema_context, mc)
        costs.append(intent_cost)

    clarification = ""
    # Guard: if the LLM claims ambiguity because a column "doesn't exist" but it
    # actually IS in the schema, suppress the interrupt — the LLM is hallucinating.
    if result.get("ambiguous"):
        _, known_cols = _known_schema_names(schema_context)
        question = result.get("clarifying_question", "")
        # If the question contains the name of an actual schema column, the LLM
        # is confused — clear ambiguous flag and continue with what it resolved.
        question_lower = question.lower()
        hallucinating = any(col in question_lower for col in known_cols if len(col) > 3)
        if hallucinating:
            logger.info(
                "resolve_task_intent: suppressing spurious ambiguity gate — "
                "LLM asked about columns that exist in schema: %s",
                [c for c in known_cols if len(c) > 3 and c in question_lower],
            )
            result["ambiguous"] = False

    if result.get("ambiguous"):
        analyst_response = interrupt({
            "gate":     "intent",
            "question": result.get("clarifying_question", "Which metric should this analysis focus on?"),
            "task":     task,
            "message":  "One question before proceeding.",
        })
        clarification = analyst_response.get("answer", "")
        if clarification.strip():
            full_task = f"{task}\n\nAnalyst clarification: {clarification}"
            result, reresolve_cost = _llm_resolve_intent(full_task, schema_context, mc)
            costs.append(reresolve_cost)

    updated_mc = _apply_intent_to_config(result, mc, schema_context)

    # Auto-detect analysis_mode from LLM — only if not explicitly set by the caller.
    # "general" tasks shouldn't be forced through the full A/B experiment pipeline.
    detected_mode = result.get("analysis_mode", "ab_test")
    if detected_mode not in ("ab_test", "general", "power_analysis"):
        detected_mode = "ab_test"
    # Prefer an explicitly passed mode (e.g. from API caller who knows their data),
    # but fall back to LLM detection when the state has no mode or has the default.
    current_mode = state.get("analysis_mode", "")
    final_mode = current_mode if current_mode in ("ab_test", "general", "power_analysis") else detected_mode

    # Extract MDE target for power analysis (default 5.0 if not stated in task)
    mde_target_pct = float(result.get("mde_target_pct") or 5.0)

    # query_type: "lookup" for simple retrieval/count, "exploratory" for analysis.
    # Only meaningful for general mode; ab_test always runs the full pipeline.
    raw_query_type = result.get("query_type", "exploratory")
    query_type = raw_query_type if raw_query_type in ("lookup", "exploratory") else "exploratory"

    # Heuristic fallback: if the LLM returned "exploratory" but the task reads
    # like a plain retrieval question, override to "lookup" so we skip the
    # heavy correlation/regression pipeline.
    if final_mode == "general" and query_type == "exploratory" and _is_lookup_task(task):
        # run_id is the correlation key an operator actually needs here; the
        # task itself is customer content and stays redacted unless explicitly
        # opted in via LOG_USER_CONTENT.
        logger.info(
            "resolve_task_intent: overriding query_type to 'lookup' via heuristic run=%s task=%s",
            state.get("run_id", "unknown"),
            redact(task),
        )
        query_type = "lookup"

    return {
        "metric_config":        updated_mc,
        "metric":               updated_mc.primary_metric,
        "covariate":            updated_mc.covariate,
        "task_clarification":   clarification,
        "analysis_mode":        final_mode,
        "power_mde_target_pct": mde_target_pct,
        "query_type":           query_type,
        # These calls' usage previously vanished — the runs table showed zero
        # cache tokens run after run and the gap was unmeasurable.
        "cache_read_tokens":  (state.get("cache_read_tokens") or 0) + sum(c.get("cache_read_tokens", 0) for c in costs),
        "cache_write_tokens": (state.get("cache_write_tokens") or 0) + sum(c.get("cache_write_tokens", 0) for c in costs),
        "estimated_cost_usd": (state.get("estimated_cost_usd") or 0.0) + sum(c.get("estimated_cost_usd", 0.0) for c in costs),
    }

