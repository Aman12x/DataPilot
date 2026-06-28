"""Analyze graph nodes — narrative."""
from __future__ import annotations

import agents.analyze.node_shared as _shared
globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

# ── Node 18: generate_narrative ───────────────────────────────────────────────

@observe(name="generate_narrative", as_type="generation")
def generate_narrative(state: AgentState) -> dict:
    mc     = state.get("metric_config") or load_metric_config()
    metric = state.get("metric") or mc.primary_metric
    mode   = state.get("analysis_mode", "ab_test")

    analyst_notes = state.get("analyst_notes") or ""
    analyst_notes_section = (
        ANALYST_NOTES_BLOCK.format(analyst_notes=analyst_notes)
        if analyst_notes.strip() else ""
    )

    # Collect all *_result fields for the LLM prompt
    # Exclude audit_result — it's an internal quality check, not source data.
    _EXCLUDE_FROM_TOOL_RESULTS = {"audit_result"}
    tool_results: dict = {}
    for k, v in state.items():
        if k.endswith("_result") and v is not None and k not in _EXCLUDE_FROM_TOOL_RESULTS:
            tool_results[k] = _to_dict(v)
    if "forecast_result" in tool_results:
        tool_results["forecast_result"].pop("forecast_df", None)
    tool_results_json = json.dumps(tool_results, default=str)
    # Hard cap: narrative prompt must stay well under the 200k token limit.
    # ~4 chars per token → cap at 80k chars (~20k tokens) for tool results.
    _MAX_TOOL_JSON = 80_000
    if len(tool_results_json) > _MAX_TOOL_JSON:
        logger.warning(
            "generate_narrative: tool_results_json too large (%d chars), truncating to %d",
            len(tool_results_json), _MAX_TOOL_JSON,
        )
        tool_results_json = tool_results_json[:_MAX_TOOL_JSON] + "\n... [truncated]"

    context_narrative = state.get("context_narrative", "")
    context_narrative_section = (
        f"\n\nPrevious analysis context (same database, follow-up question):\n"
        f"{context_narrative[:2000]}"
        if context_narrative else ""
    )

    if mode == "power_analysis":
        pa = state.get("power_analysis_result")
        power_result_json = json.dumps(_to_dict(pa), default=str) if pa else "{}"
        task_prompt = POWER_ANALYSIS_NARRATIVE_PROMPT.format(
            task=state.get("task", ""),
            power_result_json=power_result_json,
            analyst_notes_section=analyst_notes_section,
        ) + context_narrative_section
        from tools.schemas import NarrativeResult
        template_out = NarrativeResult(narrative_draft="", recommendation="")

    elif mode == "general" and state.get("query_type") == "lookup":
        task_prompt = LOOKUP_NARRATIVE_PROMPT.format(
            task=state.get("task", ""),
            tool_results_json=tool_results_json,
            analyst_notes_section=analyst_notes_section,
        ) + context_narrative_section
        from tools.schemas import NarrativeResult
        template_out = NarrativeResult(narrative_draft="", recommendation="")

    elif mode == "general":
        task_prompt = INSIGHTS_NARRATIVE_PROMPT.format(
            task=state.get("task", ""),
            tool_results_json=tool_results_json,
            analyst_notes_section=analyst_notes_section,
        ) + context_narrative_section
        from tools.schemas import NarrativeResult
        template_out = NarrativeResult(narrative_draft="", recommendation="")
    else:
        # A/B test path: build template draft via narrative_tools first
        try:
            template_out = narrative_tools.format_narrative(
                metric=metric,
                decomposition_result=_to_dict(state.get("decomposition_result")),
                anomaly_result=_to_dict(state.get("anomaly_result")),
                cuped_result=_to_dict(state.get("cuped_result")),
                ttest_result=_to_dict(state.get("ttest_result")),
                hte_result=_to_dict(state.get("hte_result")),
                novelty_result=_to_dict(state.get("novelty_result")),
                mde_result=_to_dict(state.get("mde_result")),
                guardrail_result=_to_dict(state.get("guardrail_result")),
                funnel_result=_to_dict(state.get("funnel_result")),
                forecast_result=_to_dict(state.get("forecast_result")),
                business_impact=state.get("business_impact") or "",
                analyst_notes=analyst_notes,
                srm_result=_to_dict(state.get("srm_result")),
            )
        except Exception as exc:
            logger.warning("narrative_tools.format_narrative failed: %s", exc)
            from tools.schemas import NarrativeResult
            template_out = NarrativeResult(narrative_draft="", recommendation="")

        task_prompt = NARRATIVE_PROMPT.format(
            metric=metric,
            metric_direction=mc.metric_direction,
            tool_results_json=tool_results_json,
            draft_narrative=template_out.narrative_draft,
            analyst_notes_section=analyst_notes_section,
        ) + context_narrative_section

    schema_context = (state.get("schema_context", "") or "")[:20_000]
    history_text   = _format_history(state.get("relevant_history", []))

    # Multi-turn: prepend static blocks then conversation history.
    # Cap each conversation turn to avoid runaway growth across revisions.
    messages = _build_cached_messages(schema_context, history_text, task_prompt)
    for turn in state.get("conversation_history", []):
        capped = {**turn, "content": turn["content"][:8_000]} if isinstance(turn.get("content"), str) else turn
        messages.append(capped)

    with trace_generation("generate_narrative", _fast_model(), task_prompt, max_tokens=_MAX_TOKENS_NARRATIVE) as gen:
        response = _anthropic_client().messages.create(
            model=_fast_model(),
            max_tokens=_MAX_TOKENS_NARRATIVE,
            messages=messages,
        )
        cost_info = gen.update(response)

    polished_narrative = response.content[0].text.strip()
    # Strip outer code fence if Claude wrapped the entire response in one.
    # (The frontend's sanitiseNarrative would remove fenced content, leaving nothing.)
    if polished_narrative.startswith("```"):
        polished_narrative = re.sub(r"^```[a-z]*\n?", "", polished_narrative).rstrip("`").strip()

    # ── Narrative audit ───────────────────────────────────────────────────────
    from tools.schemas import NarrativeAuditResult, NarrativeFinding  # noqa: F401

    audit_result:  Any = None
    audit_blocked: bool = False
    try:
        audit_prompt = NARRATIVE_AUDIT_PROMPT.format(
            narrative=polished_narrative,
            tool_results_json=tool_results_json,
        )
        audit_resp = _anthropic_client().messages.create(
            model=_fast_model(),
            max_tokens=2048,
            messages=[{"role": "user", "content": audit_prompt}],
        )
        raw = audit_resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        audit_result = NarrativeAuditResult(**json.loads(raw))

        critical = [f for f in audit_result.findings if f.severity == "critical"]
        moderate = [f for f in audit_result.findings if f.severity == "moderate"]

        if not critical:
            if audit_result.corrected_narrative:
                corrected = audit_result.corrected_narrative.strip()
                if corrected.startswith("```"):
                    corrected = re.sub(r"^```[a-z]*\n?", "", corrected).rstrip("`").strip()
                polished_narrative = corrected
                if moderate:
                    issues = "; ".join(f.issue for f in moderate)
                    polished_narrative += f"\n\n> **Auto-corrected:** {issues}"
        else:
            audit_blocked = True
    except Exception as exc:
        logger.warning("generate_narrative: audit failed — %s", exc)

    # Append this turn to conversation history for potential refinement
    new_history = list(state.get("conversation_history") or [])
    new_history.append({"role": "assistant", "content": polished_narrative})

    # When audit finds critical issues, append a precise correction request so
    # the next auto-retry sees exactly what to fix. The LLM picks this up from
    # conversation_history on the next generate_narrative call.
    if audit_blocked and audit_result is not None:
        critical = [f for f in audit_result.findings if f.severity == "critical"]
        fix_lines = []
        for f in critical:
            line = f"- [{f.severity}] {f.issue}"
            if f.corrected_sentence:
                line += f" → should be: {f.corrected_sentence}"
            fix_lines.append(line)
        correction_msg = (
            "The narrative above contains critical accuracy errors that must be fixed "
            "before it can be published. Please rewrite the narrative correcting ONLY "
            "these specific issues (keep all other content unchanged):\n\n"
            + "\n".join(fix_lines)
        )
        new_history.append({"role": "user", "content": correction_msg})
        logger.info(
            "generate_narrative: appended %d critical audit corrections to history "
            "(revision_count=%d)",
            len(critical),
            (state.get("narrative_revision_count") or 0) + 1,
        )

    return {
        "narrative_draft":           polished_narrative,
        "recommendation":            template_out.recommendation,
        "conversation_history":      new_history,
        "audit_result":              audit_result,
        "audit_blocked":             audit_blocked,
        "narrative_revision_count":  (state.get("narrative_revision_count") or 0) + 1,
        "cache_read_tokens":         (state.get("cache_read_tokens") or 0) + cost_info.get("cache_read_tokens", 0),
        "cache_write_tokens":        (state.get("cache_write_tokens") or 0) + cost_info.get("cache_write_tokens", 0),
        "estimated_cost_usd":        (state.get("estimated_cost_usd") or 0.0) + cost_info.get("estimated_cost_usd", 0.0),
    }


def _build_audit_log(state: AgentState) -> str:
    """
    Build a structured markdown audit block appended to every approved narrative.

    Records all gate decisions, warnings, and overrides so the report is
    self-documenting — reviewers can see exactly what was known and approved.
    """
    from datetime import datetime, timezone

    mode     = state.get("analysis_mode", "ab_test")
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    user     = state.get("user_id") or "unknown"
    override = state.get("analyst_override") or {}

    lines: list[str] = [
        "---",
        "",
        "**Analysis audit log**",
        f"- **Run ID:** `{state.get('run_id') or 'n/a'}`",
        f"- **Mode:** {mode}",
        f"- **Approved by:** {user} at {now}",
    ]

    # ── SQL gate ──────────────────────────────────────────────────────────────
    sql_warnings = state.get("sql_validation_warnings") or []
    is_postgres  = state.get("db_backend", "duckdb") == "postgres"
    if is_postgres or sql_warnings:
        edit_note = " (analyst edited SQL)" if override.get("sql_edited") else ""
        lines.append(f"- **SQL gate:** human-reviewed{edit_note}")
    else:
        lines.append("- **SQL gate:** auto-approved (no warnings)")
    for w in sql_warnings:
        lines.append(f"  - ⚠️ {w}")

    if mode == "ab_test":
        # ── SRM ───────────────────────────────────────────────────────────────
        srm          = state.get("srm_result")
        srm_detected = srm.srm_detected if srm else False
        srm_ack      = state.get("srm_acknowledged", False)
        if srm_detected:
            ack_str = "analyst acknowledged and proceeded" if srm_ack else "acknowledgment status unknown"
            lines.append(f"- **SRM:** ⛔ detected — {ack_str}")
        else:
            lines.append("- **SRM:** ✅ not detected")

        # ── Guardrails ────────────────────────────────────────────────────────
        gr = state.get("guardrail_result")
        if gr and gr.any_breached:
            breached_names = [g.metric for g in gr.guardrails if g.breached]
            lines.append(f"- **Guardrails:** ⚠️ breached — {', '.join(breached_names)}")
        elif gr:
            lines.append("- **Guardrails:** ✅ all clear")

        # ── Power / winner's curse ────────────────────────────────────────────
        mde   = state.get("mde_result")
        ttest = state.get("ttest_result")
        if mde and mde.post_hoc_power is not None:
            pwr = mde.post_hoc_power
            sig = ttest.significant if ttest else False
            if sig and pwr < 0.50:
                lines.append(
                    f"- **Winner's curse:** ⚠️ present (post-hoc power={pwr*100:.0f}%)"
                    " — analyst approved despite inflation risk"
                )
            else:
                pwr_icon = "✅" if pwr >= 0.80 else "⚠️"
                lines.append(f"- **Post-hoc power:** {pwr_icon} {pwr*100:.0f}%")

        # ── Auto-corrections ──────────────────────────────────────────────────
        rev = state.get("narrative_revision_count") or 0
        if rev > 0:
            lines.append(f"- **Auto-corrections:** {rev} attempt(s) before approval")
        else:
            lines.append("- **Auto-corrections:** none required")

    # ── Analyst notes (any gate) ─────────────────────────────────────────────
    if override.get("analysis_notes"):
        lines.append(f"- **Analyst notes (analysis gate):** {override['analysis_notes']}")
    if override.get("narrative_notes"):
        lines.append(f"- **Analyst notes (narrative gate):** {override['narrative_notes']}")
    if override.get("recommendation_override"):
        lines.append(f"- **Recommendation overridden by analyst**")

    return "\n".join(lines)


def _generate_deck(state: "AgentState", final_narrative: str) -> dict:
    """Generate a structured stakeholder deck JSON from the approved narrative."""
    mode = state.get("analysis_mode", "general")
    try:
        prompt = DECK_PROMPT.format(
            mode=mode,
            narrative=final_narrative[:4000],
        )
        resp = _anthropic_client().messages.create(
            model=_fast_model(),
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning("deck generation failed: %s", exc)
        return {}


# ── Node 19: narrative_gate (HITL interrupt 3) ────────────────────────────────

@observe(name="narrative_gate")
def narrative_gate(state: AgentState) -> dict:
    # narrative_gate is a pure HITL review step — no auto-blocking here.
    # Auto-corrections happen in generate_narrative (via audit_blocked loop).
    # Surfacing violations to the analyst is done via the payload message.
    audit_result = state.get("audit_result")
    audit_findings: list[str] = []
    if audit_result and hasattr(audit_result, "findings"):
        critical = [f for f in (audit_result.findings or []) if f.severity == "critical"]
        audit_findings = [f'{f.issue} (quote: "{f.quote}")' for f in critical]

    payload: dict = {
        "gate":             "narrative",
        "narrative_draft":  state.get("narrative_draft", ""),
        "recommendation":   state.get("recommendation", ""),
        "message":          "Review the narrative. Approve, or add notes to trigger a revision.",
    }
    if audit_findings:
        payload["audit_critical_findings"] = audit_findings
        payload["message"] = (
            "⚠️ AUDIT: Potential accuracy issues detected — review before approving: "
            + "; ".join(audit_findings)
        )
    analyst_response = interrupt(payload)

    approved      = analyst_response.get("approved", True)
    analyst_notes = analyst_response.get("notes", "")

    override = dict(state.get("analyst_override") or {})
    if analyst_notes.strip():
        override["narrative_notes"] = analyst_notes.strip()
    if rec_override := analyst_response.get("recommendation_override", ""):
        override["recommendation_override"] = rec_override.strip()

    if approved:
        audit_log  = _build_audit_log(state)
        final      = state.get("narrative_draft", "") + "\n\n" + audit_log
        deck_data  = _generate_deck(state, final)
        return {
            "narrative_approved": True,
            "final_narrative":    final,
            "deck_data":          deck_data,
            "analyst_notes":      analyst_notes,
            "analyst_override":   override,
        }

    # Analyst wants a revision — updated notes will cause graph to re-run generate_narrative
    return {
        "narrative_approved": False,
        "analyst_notes":      analyst_notes,
        "analyst_override":   override,
    }


# ── Quality score (completeness signal, no ground truth needed) ───────────────

def _compute_quality_score(state: AgentState) -> float:
    """
    Estimate run quality from tool-result completeness + RAGAS-inspired signals.

    Returns a 0–1 float. Used when no ground-truth eval_score is available
    so every run still contributes a learning signal to the memory store.

    Composite:
      60%  completeness  — did all expected tool nodes produce results?
      40%  eval_tools    — faithfulness + relevancy of the final narrative
    """
    # ── Completeness (mode-aware) ─────────────────────────────────────────────
    mode = state.get("analysis_mode", "general")
    if mode == "ab_test":
        cuped = state.get("cuped_result")
        checks = [
            bool(cuped and cuped.variance_reduction_pct > 5),
            bool(state.get("ttest_result")),
            bool(state.get("hte_result") and state["hte_result"].top_segment),
            bool(state.get("guardrail_result")),
            bool(state.get("novelty_result")),
            bool(state.get("forecast_result")),
        ]
    elif mode == "power_analysis":
        checks = [
            bool(state.get("power_analysis_result")),
            bool(state.get("narrative_draft")),
        ]
    else:  # general
        checks = [
            bool(state.get("describe_result")),
            bool(state.get("correlation_result")),
            bool(state.get("charts")),
            bool(state.get("narrative_draft")),
            bool(state.get("query_result") is not None),
        ]
    completeness = sum(checks) / len(checks)

    # ── RAGAS signals (best-effort; degrade gracefully) ───────────────────────
    narrative = state.get("final_narrative") or state.get("narrative_draft") or ""
    task      = state.get("task") or ""
    df        = state.get("query_result")

    ragas_score: float | None = None
    if narrative and task:
        try:
            from tools.eval_tools import evaluate_run
            result = evaluate_run(task, narrative, df=df)
            ragas_score = result.score if result.relevancy >= 0 else result.faithfulness
        except Exception as exc:
            logger.debug("_compute_quality_score: eval_tools failed — %s", exc)

    # ── Claim accuracy (always-on for A/B, deterministic, zero cost) ─────────────
    if mode == "ab_test" and narrative:
        ttest  = state.get("ttest_result")
        cuped  = state.get("cuped_result")
        if ttest is not None:
            try:
                from tools.eval_tools import score_claim_accuracy
                claim = score_claim_accuracy(narrative, ttest, cuped)
                if claim["violations"]:
                    logger.warning("Claim accuracy violations: %s", claim["violations"])
                    ragas_score = min(ragas_score if ragas_score is not None else 1.0, 0.6)
            except Exception as exc:
                logger.debug("_compute_quality_score: claim_accuracy failed — %s", exc)

    # ── LLM judge (opt-in via ENABLE_LLM_JUDGE=true) ─────────────────────────
    if os.getenv("ENABLE_LLM_JUDGE") == "true" and narrative and task:
        recommendation = state.get("recommendation") or ""
        if recommendation:
            try:
                from tools.eval_tools import score_recommendation
                rec = score_recommendation(recommendation, narrative, task)
                ragas_score = 0.7 * (ragas_score or 0.0) + 0.3 * rec["score"]
                logger.info("LLM judge score=%.3f actionability=%.2f specificity=%.2f grounding=%.2f",
                            rec["score"], rec["actionability"], rec["specificity"], rec["grounding"])
            except Exception as exc:
                logger.debug("LLM judge failed: %s", exc)

    if ragas_score is not None:
        return round(0.6 * completeness + 0.4 * ragas_score, 4)
    return round(completeness, 4)


# ── Node 20: log_run_node ─────────────────────────────────────────────────────

@observe(name="log_run")
def log_run_node(state: AgentState) -> dict:
    run_id = state.get("run_id") or str(uuid.uuid4())
    task   = state.get("task", "")

    # Persist to memory store
    log_run(
        task=task,
        run_id=run_id,
        user_id=state.get("user_id"),
        analysis_mode=state.get("analysis_mode") or "ab_test",
        metric=state.get("metric") or "",
        covariate=state.get("covariate") or "",
        db_backend=state.get("db_backend") or "duckdb",
        analyst_override=state.get("analyst_override") or None,
        top_segment=(hte := state.get("hte_result")) and hte.top_segment or "",
        eval_score=state.get("eval_score"),
        cache_read_tokens=state.get("cache_read_tokens") or 0,
        cache_write_tokens=state.get("cache_write_tokens") or 0,
        estimated_cost_usd=state.get("estimated_cost_usd") or 0.0,
        semantic_cache_hits=1 if state.get("semantic_cache_hit") else 0,
        notes=state.get("analyst_notes") or "",
        audit_passed=(
            (ar := state.get("audit_result")) is None
            or (hasattr(ar, "passed") and ar.passed)
        ),
    )

    # In-band completeness scoring — fills eval_score when offline eval hasn't run yet
    if state.get("eval_score") is None:
        quality_score = _compute_quality_score(state)
        update_eval_score(run_id, quality_score)
        logger.info("log_run: quality score %.2f stored for run %s", quality_score, run_id)

    # Store SQL result in semantic cache only when the result has all required columns
    # (prevents poisoning the cache with bad date-level SQL)
    if state.get("generated_sql"):
        mc_log  = state.get("metric_config") or load_metric_config()
        df_log  = state.get("query_result")
        required_cols = {mc_log.primary_metric, mc_log.covariate, "variant"}
        result_cols   = set(df_log.columns) if df_log is not None and hasattr(df_log, "columns") else set()
        if required_cols.issubset(result_cols):
            narrative = state.get("final_narrative") or state.get("narrative_draft") or ""
            if not narrative.strip():
                logger.info("log_run: skipping SQL cache — narrative is empty")
                return {"run_id": run_id}
            semantic_cache.store_cache(
                task=task,
                node_name="generate_sql",
                result={
                    "sql":            state["generated_sql"],
                    "narrative":      narrative,
                    "recommendation": state.get("recommendation") or "",
                },
                run_id=run_id,
                dataset_fingerprint=state.get("duckdb_path", ""),
                user_id=state.get("user_id"),
            )
        else:
            logger.info(
                "log_run: skipping SQL cache — result still missing %s",
                sorted(required_cols - result_cols),
            )

    flush()

    return {"run_id": run_id}
