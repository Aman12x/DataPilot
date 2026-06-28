"""Analyze graph nodes — intent."""
from __future__ import annotations

import agents.analyze.node_shared as _shared
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

    Returns a dict with keys: primary_metric, metric_direction, covariate,
    guardrail_metrics, ambiguous, clarifying_question, reasoning.

    Falls back to a safe default (ambiguous=False, mc defaults preserved) on
    any parse failure — never hard-fails.
    """
    # Extract metric-like column names from schema for the prompt
    _, known_columns = _known_schema_names(schema_context)
    available_metrics = ", ".join(sorted(known_columns)) if known_columns else "(schema not available)"

    task_prompt = TASK_INTENT_PROMPT.format(
        task=task,
        available_metrics=available_metrics,
        default_metric=mc.primary_metric,
    )
    history_text = ""   # intent resolution doesn't need history injection
    messages = _build_cached_messages(schema_context, history_text, task_prompt)

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
        response = _anthropic_client().messages.create(
            model=_model(),
            max_tokens=256,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        # Ensure all required keys exist, filling from defaults where missing
        for key, default_val in safe_default.items():
            result.setdefault(key, default_val)
        return result
    except Exception as exc:
        logger.warning("_llm_resolve_intent: parse failed (%s) — using defaults.", exc)
        return safe_default


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
    owned by infer_metric_config.
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
            logger.warning("_apply_intent_to_config: %s", w)
        return updated
    except Exception as exc:
        logger.warning(
            "_apply_intent_to_config: MetricConfig update failed (%s) — keeping original.", exc
        )
        return mc


# ── Node 3b: resolve_task_intent ──────────────────────────────────────────────

@observe(name="resolve_task_intent")
def resolve_task_intent(state: AgentState) -> dict:
    """
    Reads the analyst's task, identifies the intended metric, and asks one
    clarifying question if the task is genuinely ambiguous.

    Implements Rule 6: ask before assuming on ambiguous tasks.

    Positioned after load_schema (schema_context is available) and before
    infer_metric_config (metric_config can still be overridden).
    """
    task           = state.get("task", "")
    schema_context = state.get("schema_context", "")
    mc             = state.get("metric_config") or load_metric_config()

    result = _llm_resolve_intent(task, schema_context, mc)

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
            result = _llm_resolve_intent(full_task, schema_context, mc)

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
        logger.info("resolve_task_intent: overriding query_type to 'lookup' via heuristic for task: %s", task[:80])
        query_type = "lookup"

    return {
        "metric_config":        updated_mc,
        "metric":               updated_mc.primary_metric,
        "covariate":            updated_mc.covariate,
        "task_clarification":   clarification,
        "analysis_mode":        final_mode,
        "power_mde_target_pct": mde_target_pct,
        "query_type":           query_type,
    }

# ── Node 21: infer_metric_config_node ─────────────────────────────────────────
# Called once when connecting a new external DB that has no config file.
# No-op if metric_config is already present in state (DuckDB demo / pre-loaded config).

@observe(name="infer_metric_config")
def infer_metric_config_node(state: AgentState) -> dict:
    """
    LLM infers MetricConfig from schema. Result stored in state for UI form pre-fill.
    Only runs when metric_config is not already set in state.
    """
    # Skip if config is already set — UNLESS this is an uploaded file.
    # Uploads have a unique schema that differs from the DAU demo defaults that
    # load_schema/resolve_task_intent fall back to; inference must always run
    # so guardrail_metrics, segment_cols, table names etc. match the real data.
    if state.get("metric_config") and not state.get("duckdb_path"):
        return {}

    schema_context = state.get("schema_context", "")
    if not schema_context:
        mc = load_metric_config()
        return {"metric_config": mc, "metric": mc.primary_metric, "covariate": mc.covariate}

    prompt = SCHEMA_CONFIG_INFERENCE_PROMPT.format(schema_context=schema_context)

    defaults = load_metric_config()

    try:
        with trace_generation("infer_metric_config", _fast_model(), prompt) as gen:
            response = _anthropic_client().messages.create(
                model=_fast_model(),
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            gen.update(response)
    except Exception as exc:
        logger.warning("infer_metric_config: LLM call failed (%s), using defaults.", exc)
        return {
            "metric_config": defaults,
            "metric": defaults.primary_metric,
            "covariate": defaults.covariate,
        }

    try:
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        inferred = MetricConfig(**json.loads(raw))
    except Exception as exc:
        logger.warning("infer_metric_config: LLM response parsing failed (%s), using defaults.", exc)
        inferred = defaults

    # ── Cross-check every inferred column/table name against the live schema ──
    # If the LLM hallucinated a column name, _sanitise_metric_config replaces it
    # with the default and logs a warning.  This prevents bad names from
    # propagating to _canonical_experiment_sql() and generating broken SQL.
    inferred, issues = _sanitise_metric_config(inferred, schema_context, defaults)
    for w in issues:
        logger.warning("infer_metric_config: schema mismatch — %s", w)

    return {
        "metric_config": inferred,
        "metric":         inferred.primary_metric,
        "covariate":      inferred.covariate,
    }
