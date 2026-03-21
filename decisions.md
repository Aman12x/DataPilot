# DataPilot — Decision Log

Non-obvious implementation choices made during build.
Format: [file] [function] — chose X over Y because Z

---

tests/conftest.py base_experiment_df — chose latent-baseline design (shared `baseline` variable drives both pre_sessions and post_dau) over independent draws because independent draws would give near-zero CUPED theta, making variance reduction untestable

tests/conftest.py base_experiment_df — chose mild guardrail effects on ALL treatment users (not just android/new) because android/new alone is 20% of users and the blended t-test was not significant enough to trigger breached=True

tools/anomaly_tools.py detect_anomaly — chose zscore over IQR because the fixture has a clean step-change (not outlier spikes), and IQR was insensitive to that pattern at the fixture's noise level

tools/anomaly_tools.py detect_anomaly — chose auto-relaxing threshold (−0.5 per retry) over a fixed fallback because a hard fallback would silently hide near-threshold signals; progressive relaxation surfaces the best available signal

tools/forecast_tools.py forecast_baseline — chose Prophet over statsmodels ARIMA as primary because Prophet handles weekly seasonality and missing dates without manual configuration; ARIMA requires stationarity checks not appropriate for a graceful-fallback path

tools/funnel_tools.py compute_funnel — chose conditional step rates (eligible = users who completed prior step) over raw completion rates across all users because raw rates use the wrong denominator at deep funnel steps, diluting the d1_retain signal from 20pp to ~0

tools/guardrail_tools.py check_guardrails — chose keyword-based harm direction inference over requiring caller to pass directions explicitly because it reduces boilerplate at call sites and the keyword heuristic covers all metrics in the project schema

tests/test_mde_tools.py test_underpowered_for_blended_effect — chose n=1000/std=0.25 (MDE=5.7%) over n=5000/std=0.10 (MDE=1.02%) because at n=5000 the blended 1.6% effect is actually powered, contradicting the CLAUDE.md ground truth that it sits near the MDE boundary

tools/narrative_tools.py format_narrative — chose template-based formatter over calling the LLM directly because the tool layer must be pure Python per CLAUDE.md Rule 1; the LLM in nodes.py refines this draft

memory/retriever.py retrieve_relevant_history — chose keyword overlap (token intersection) over embedding similarity for retrieval because semantic similarity is handled by semantic_cache.py; retriever is fast and runs before any model is loaded

memory/semantic_cache.py check_cache — chose pickle for cached_result serialisation over JSON because tool results contain numpy floats and DataFrames that aren't JSON-serialisable without a custom encoder

agents/tracer.py observe — chose graceful no-op fallback over raising on missing credentials because tests and local dev without Langfuse keys should work identically; tracing is additive, never blocking

agents/tracer.py GenerationContext — chose manual cost calculation over Langfuse's built-in cost tracking because we need the same cost numbers in memory/store.py for self-improvement; single source of truth avoids drift between the two systems

agents/analyze/nodes.py generate_sql — chose single user message with multiple content blocks over two separate user messages for prompt caching because the Anthropic API alternates user/assistant; multiple content blocks in one message achieve the same cache prefix behaviour

agents/analyze/nodes.py generate_narrative — chose narrative_tools.format_narrative as a pre-pass over sending raw tool results directly to the LLM because it gives the LLM a structured starting point, reducing hallucinated numbers and missing sections

config/analysis_config.py load_metric_config — chose JSON-first with env-var fallback over env-vars-only because JSON allows full MetricConfig (including guardrail_harm_directions) to be version-controlled and swapped at runtime without code changes; env vars remain as the backward-compatible fallback

tools/guardrail_tools.py check_guardrails — chose explicit harm_directions takes full precedence over keyword inference (instead of merging) because partial overrides from MetricConfig could silently conflict with keyword inference on the same metric, leading to unpredictable breach decisions

tools/guardrail_tools.py check_guardrails — chose default_direction='decrease' in nodes.py when metric_direction is higher_is_better over 'both' because the common failure mode for unknown guardrails is a harmful drop (not a harmful increase), so 'decrease' reduces false negatives

tests/conftest.py make_experiment_df — chose plain function over pytest fixture for the factory because fixtures can't take arguments without fixtures_params/indirect overhead; plain functions let any test call make_experiment_df('revenue') directly

agents/analyze/graph.py _PickleSerde — chose pickle over msgpack (LangGraph default) for the MemorySaver serde because pd.DataFrame stored in AgentState.query_result is not msgpack-serializable; pickle handles all Python objects including DataFrames and Pydantic models natively

config/analysis_config.py MetricConfig — chose model_validator to default metric_source_col to primary_metric over requiring explicit field in every JSON because most configs (retention, custom) share the same column name for DB source and alias; only DAU needs an explicit override (dau_flag → dau_rate)

agents/analyze/nodes.py _canonical_experiment_sql — chose pre_exp CTE aliased to `pre_events` over the table alias `e` to avoid join ambiguity when events_table appears in both the main query and the pre-experiment subquery

agents/analyze/nodes.py load_auxiliary_data — chose timeseries_table first with event-aggregation fallback over always aggregating from events because pre-aggregated tables (metrics_daily) contain DAU component columns (new_users, retained_users) needed for decompose_dau; raw aggregation loses those columns and triggers the generic path

agents/analyze/nodes.py decompose_metric — chose DAU-path detection by column presence (new_users, retained_users, churned_users all in df.columns) over config flag because it makes the node self-healing: if a user connects a DB with those columns it gets the richer decomposition automatically

tools/decomposition_tools.py decompose_metric — chose before/after split at experiment_start when provided over midpoint because midpoint is arbitrary and silently wrong when pre-experiment period is much longer than the experiment window

tools/db_tools.py _sample_distinct_values_postgres — chose LIMIT 51 (one over threshold) over COUNT(DISTINCT) + second query because a single query is faster and the off-by-one approach tells us exactly what we need: ≤50 = show values, 51 = skip

agents/analyze/nodes.py load_schema — chose prepending dialect comment to schema_context over adding it to the task prompt because schema_context is the prompt-cached static block; keeping the dialect there ensures it's always present in the cache prefix without adding cost per call

agents/analyze/nodes.py _validate_sql_tables — chose table-only validation over full column validation because table hallucinations always cause hard execution failures (testable without running the query), while column hallucinations are already caught by the execute_query column-presence check; full SQL parsing would require adding sqlglot as a dependency

agents/analyze/nodes.py execute_query — chose execute → LLM-correct → retry loop (max 2) over immediate canonical fallback because the canonical SQL only works for the DAU demo schema; on external Postgres DBs the correction loop is the only recovery path; canonical SQL remains as the final fallback

agents/analyze/nodes.py generate_sql — chose attaching sql_validation_warnings to state (surfaced at HITL gate) over blocking execution because the analyst may legitimately be querying a CTE or subquery alias that looks like a hallucinated table; non-blocking with visibility is safer than a hard block

memory/retriever.py retrieve_sql_examples — chose min_similarity=0.40 (lower than semantic cache 0.80/0.92 thresholds) because few-shot examples are useful even when loosely related — the model benefits from seeing schema patterns and query structure even for a different metric or time range

tools/decomposition_tools.py decompose_dau — chose "most negative delta" over "largest absolute delta" for dominant_change_component when any component declines, because the question being answered is always "what drove the DROP" — a massive positive retained-users growth should not mask a declining new-users cohort

data/generate_data.py BUG_OPTOUT_MULTIPLIER — chose 6.0 over 3.0 because the affected segment (android new_users) is only ~8% of the total experiment population; at 3x the blended treatment-wide t-test was not significant (p≈0.17), failing the optout_breached eval criterion

agents/analyze/nodes.py _format_history — chose instructional prose hints over raw JSON override dumps because the LLM cannot act on {"sql_edited": true} but can act on "ANALYST CORRECTED SQL — double-check JOINs"; instructional format closes the self-improvement loop

agents/analyze/nodes.py _compute_quality_score — chose completeness-based scoring (6 binary tool-result checks) over no in-band scoring because eval_score was always null for normal graph runs, making history injection useless; completeness gives a noisy-but-real signal on every run
