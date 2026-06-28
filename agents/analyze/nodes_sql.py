"""Analyze graph nodes — sql."""
from __future__ import annotations

import agents.analyze.node_shared as _shared
globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

# ── Node 4: generate_sql ──────────────────────────────────────────────────────

@observe(name="generate_sql", as_type="generation")
def generate_sql(state: AgentState) -> dict:
    # If semantic cache already provided SQL, skip the API call
    if state.get("semantic_cache_hit") and state.get("generated_sql"):
        return {}

    task           = state.get("task", "")
    schema_context = state.get("schema_context", "")
    history_text   = _format_history(state.get("relevant_history", []))
    db_backend     = state.get("db_backend", "duckdb")
    mc             = state.get("metric_config") or load_metric_config()
    mode           = state.get("analysis_mode", "ab_test")

    # ── Few-shot retrieval — schema-filtered ─────────────────────────────────
    # Only inject examples whose SQL references tables present in the current
    # schema.  Prevents demo-DB examples (events, experiment, metrics_daily)
    # from misleading the LLM when the user uploads a different dataset.
    current_tables = _known_schema_names(schema_context)[0]
    sql_examples   = retrieve_sql_examples(task, user_id=state.get("user_id"))
    sql_examples   = _filter_few_shot_by_schema(sql_examples, current_tables)
    few_shot_block = _build_few_shot_block(sql_examples)

    if mode == "general":
        task_prompt = SQL_GENERATION_GENERAL_PROMPT.format(
            task=task,
            schema_context=schema_context,
            db_backend=db_backend,
            metric_context=_metric_context(mc),
        )
    else:
        task_prompt = SQL_GENERATION_PROMPT.format(
            task=task,
            schema_context=schema_context,
            db_backend=db_backend,
            metric_context=_metric_context(mc),
            primary_metric=mc.primary_metric,
            metric_source_col=mc.metric_source_col,
            metric_agg=mc.metric_agg,
            covariate=mc.covariate,
            variant_col=mc.variant_col,
            week_col=mc.week_col,
            guardrail_metrics_csv=", ".join(mc.guardrail_metrics) or "(none)",
            segment_cols_csv=", ".join(mc.segment_cols),
            sql_template=_canonical_experiment_sql(mc),
            few_shot_block=few_shot_block,
        )

    context_narrative = state.get("context_narrative", "")
    if context_narrative:
        task_prompt = (
            f"Previous analysis context (same database, different question):\n"
            f"{context_narrative[:2000]}\n\n"
        ) + task_prompt

    messages = _build_cached_messages(schema_context, history_text, task_prompt)

    with trace_generation("generate_sql", _fast_model(), task_prompt) as gen:
        response = _anthropic_client().messages.create(
            model=_fast_model(),
            max_tokens=_MAX_TOKENS_SQL,
            messages=messages,
        )
        cost_info = gen.update(response)

    sql = _extract_sql(response.content[0].text)

    # ── Schema validation + one auto-correction pass ──────────────────────────
    # Check both table names and dotted column references against schema_context.
    # If issues are found, send them back to the LLM for a targeted correction
    # before showing anything to the analyst at the HITL gate.
    # This catches hallucinations early — before execution — and avoids wasting
    # the analyst's attention on obviously broken SQL.
    validation  = _validate_sql_references(sql, schema_context)
    all_issues  = validation["bad_tables"] + validation["bad_columns"]

    if all_issues:
        logger.warning("generate_sql: schema issues detected %s — auto-correcting.", all_issues)
        issue_lines = "\n".join(f"  - {v}" for v in all_issues)
        correction_hint = (
            f"The following names in the SQL don't exist in the schema:\n"
            f"{issue_lines}\n\n"
            "For each invalid name, replace it with the correct name from the schema. "
            "Do not invent names — use only what's listed in the schema."
        )
        corrected = _llm_correct_sql(sql, correction_hint, schema_context, task)
        if corrected.strip() and corrected.strip() != sql.strip():
            sql = corrected
            # Re-validate once so remaining issues are surfaced at the HITL gate
            validation = _validate_sql_references(sql, schema_context)
            all_issues = validation["bad_tables"] + validation["bad_columns"]
            if all_issues:
                logger.warning(
                    "generate_sql: issues remain after correction: %s", all_issues
                )

    result = {
        "generated_sql":      sql,
        "cache_read_tokens":  cost_info.get("cache_read_tokens", 0),
        "cache_write_tokens": cost_info.get("cache_write_tokens", 0),
        "estimated_cost_usd": cost_info.get("estimated_cost_usd", 0.0),
    }
    if all_issues:
        result["sql_validation_warnings"] = all_issues
    return result


# ── Node 5: query_gate (HITL interrupt 1) ────────────────────────────────────

@observe(name="query_gate")
def query_gate(state: AgentState) -> dict:
    warnings    = state.get("sql_validation_warnings", [])
    db_backend  = state.get("db_backend", "duckdb")

    payload = {
        "gate":                     "query",
        "generated_sql":            state.get("generated_sql", ""),
        "cache_hit":                state.get("semantic_cache_hit", False),
        "sql_validation_warnings":  warnings,
        "message":                  "Review the generated SQL. Approve, or provide a corrected query.",
    }
    analyst_response = interrupt(payload)

    # analyst_response expected: {"approved": bool, "sql": str | None}
    approved    = analyst_response.get("approved", True)
    edited_sql  = analyst_response.get("sql") or state.get("generated_sql", "")

    schema_context = state.get("schema_context", "")
    validation_issues: list[str] = []
    if edited_sql.strip():
        validation = _validate_sql_references(edited_sql, schema_context)
        validation_issues = validation.get("bad_tables", []) + validation.get("bad_columns", [])

    override: dict = {}
    if edited_sql.strip() != state.get("generated_sql", "").strip():
        override["sql_edited"] = True

    result = {
        "query_approved":   approved,
        "generated_sql":    edited_sql,
        "analyst_override": override,
    }
    if validation_issues:
        result["sql_validation_warnings"] = validation_issues
    return result


# ── Node 6: execute_query ─────────────────────────────────────────────────────

@observe(name="execute_query")
def execute_query(state: AgentState) -> dict:
    sql = state.get("generated_sql", "")
    if not sql:
        raise ValueError("No SQL to execute — generate_sql must run first.")

    mc             = state.get("metric_config") or load_metric_config()
    schema_context = state.get("schema_context", "")
    task           = state.get("task", "")

    # ── Phase 1: LLM SQL with error-correction retries ────────────────────────
    # On each execution failure, send (sql, error, schema) to the LLM for a
    # targeted correction and retry.  Up to _MAX_SQL_RETRIES corrections.
    # This implements the "execution feedback loop" pattern that is now standard
    # in production text-to-SQL systems (AWS, CHESS, ReFoRCE, etc.).
    current_sql = sql
    df: pd.DataFrame | None = None

    for attempt in range(_MAX_SQL_RETRIES + 1):
        try:
            df = _db_conn(state).query(current_sql)
            if attempt > 0:
                logger.info("execute_query: SQL succeeded after %d LLM correction(s).", attempt)
            break
        except Exception as exc:
            if attempt < _MAX_SQL_RETRIES:
                logger.warning(
                    "execute_query: attempt %d failed (%s) — requesting LLM correction.",
                    attempt + 1, exc,
                )
                corrected = _llm_correct_sql(current_sql, str(exc), schema_context, task)
                if corrected.strip() and corrected.strip() != current_sql.strip():
                    current_sql = corrected
                else:
                    logger.warning(
                        "execute_query: LLM returned unchanged SQL on attempt %d — "
                        "stopping retries early.",
                        attempt + 1,
                    )
                    break
            else:
                logger.warning(
                    "execute_query: all %d LLM retries exhausted — falling back to canonical SQL.",
                    _MAX_SQL_RETRIES,
                )

    # ── Phase 2: Column validation + canonical SQL fallback ───────────────────
    # Canonical SQL fallback only applies to ab_test mode — it uses experiment
    # tables and variant columns that don't exist in general / upload schemas.
    is_ab = state.get("analysis_mode", "ab_test") == "ab_test"

    if df is not None and is_ab:
        required = {mc.primary_metric, mc.covariate, "variant"}
        missing  = required - set(df.columns)
        if missing:
            logger.warning(
                "execute_query: result missing columns %s — trying canonical SQL.",
                sorted(missing),
            )
            df = None   # trigger canonical fallback below

    if df is None:
        if not is_ab:
            logger.warning(
                "execute_query: LLM SQL failed for general-mode query — "
                "returning empty DataFrame. Check schema context and SQL generation prompt."
            )
            return {"query_result": pd.DataFrame()}
        canonical_sql = _canonical_experiment_sql(mc)
        try:
            df = _db_conn(state).query(canonical_sql)
            current_sql = canonical_sql
            logger.info("execute_query: canonical SQL succeeded (%d rows).", len(df))
        except Exception as exc:
            raise ValueError(
                f"execute_query: LLM SQL and canonical SQL both failed. Last error: {exc}"
            ) from exc

    # Deduplicate columns — LLM SQL may emit the same column twice (e.g. when
    # covariate == a guardrail metric).  Keep the first occurrence only.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # ── Phase 3 (general mode only): panel-data deduplication guard ───────────
    # If the result has a recognisable entity-ID column with duplicates, the
    # LLM probably forgot to GROUP BY and returned raw panel/longitudinal rows.
    # Ask the LLM to add the missing aggregation and re-execute once.
    _ENTITY_COLS = {"user_id", "customer_id", "patient_id", "userid",
                    "uid", "entity_id", "id", "shipment_id"}
    if state.get("analysis_mode") == "general":
        entity_col = next((c for c in df.columns if c.lower() in _ENTITY_COLS), None)
        if entity_col and df[entity_col].duplicated().any():
            n_rows    = len(df)
            n_entity  = df[entity_col].nunique()
            ratio     = round(n_rows / max(n_entity, 1), 1)
            logger.warning(
                "execute_query: panel data detected — %d rows but only %d distinct %s "
                "(%.1f rows/entity). Requesting aggregation fix.",
                n_rows, n_entity, entity_col, ratio,
            )
            hint = (
                f"The query returned {n_rows} rows but only {n_entity} distinct "
                f"'{entity_col}' values ({ratio} rows per entity). "
                f"This is panel/longitudinal data that must be collapsed to one row "
                f"per {entity_col}. Add GROUP BY {entity_col} (and any categorical "
                f"columns) with MAX() for binary flags and AVG() for numeric metrics."
            )
            fixed = _llm_correct_sql(current_sql, hint, schema_context, task)
            if fixed.strip() and fixed.strip() != current_sql.strip():
                try:
                    df_fixed = _db_conn(state).query(fixed)
                    entity_col2 = next(
                        (c for c in df_fixed.columns if c.lower() in _ENTITY_COLS), None
                    )
                    if entity_col2 and not df_fixed[entity_col2].duplicated().any():
                        logger.info(
                            "execute_query: aggregation fix succeeded — %d rows → %d.",
                            n_rows, len(df_fixed),
                        )
                        df = df_fixed
                        current_sql = fixed
                    else:
                        logger.warning("execute_query: aggregation fix still has duplicates — keeping original.")
                except Exception as exc:
                    logger.warning("execute_query: aggregation fix failed (%s) — keeping original.", exc)

    # ── Phase 4: Content validation ───────────────────────────────────────────
    # Replace (not append) so stale warnings from a prior 0-row attempt don't
    # persist after the analyst fixes the SQL and re-executes.
    content_warnings = _validate_query_content(df, mc, state.get("analysis_mode", "ab_test"))
    for w in content_warnings:
        logger.warning("execute_query: content validation — %s", w)

    result: dict = {
        "query_result":            df,
        "sql_validation_warnings": content_warnings,
    }
    if current_sql != sql:
        result["generated_sql"] = current_sql
    return result

