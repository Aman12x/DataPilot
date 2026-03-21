"""
ui/app.py — Streamlit frontend for DataPilot.

Renders only. Zero agent logic, zero stats, zero SQL.
All decisions live in agents/ or tools/.

Session state machine:
  st.session_state.thread_id      — active graph run thread (None = idle)
  st.session_state.current_gate   — last known gate for display continuity
  st.session_state.run_history    — list of completed run summaries
  st.session_state.db_conn        — DBConnection (cleared on session end)
  st.session_state.metric_config  — confirmed MetricConfig
  st.session_state.session_cost_usd
  st.session_state.session_saved_usd

Interaction model:
  1. User submits a task → graph.invoke(initial_state, config) → hits query_gate
  2. Streamlit reruns; app reads g.get_state(config) → renders gate UI
  3. Analyst approves/edits → g.invoke(Command(resume=...), config) → st.rerun()
  4. Repeat until snap.next is empty (graph finished)
"""

from __future__ import annotations

import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from agents.analyze.graph import graph
from memory.store import get_all_runs
from ui.auth_page import render_auth_page
from ui.db_connect import render_connection_sidebar
from ui.report_export import build_pdf

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataPilot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────

for _k, _v in {
    "thread_id":         None,
    "current_gate":      None,
    "run_history":       [],
    "session_cost_usd":  0.0,
    "session_saved_usd": 0.0,
    "user":              None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _snap():
    """Return the current graph snapshot, or None if no active run.

    If the thread_id in session_state refers to a run that no longer exists in
    the current MemorySaver (e.g. after a Streamlit hot-reload that recreated
    the module-level graph), reset to idle rather than showing a broken gate.
    """
    if not st.session_state.thread_id:
        return None
    try:
        snap = graph.get_state(_config())
        # get_state returns a snapshot with empty values dict when thread is unknown
        if snap is not None and not snap.values and not snap.next:
            _reset()
            return None
        return snap
    except Exception:
        _reset()
        return None


def _interrupt_payload(snap) -> dict | None:
    """Extract the interrupt payload from a snapshot, or None if not interrupted."""
    if snap and snap.tasks and snap.tasks[0].interrupts:
        return snap.tasks[0].interrupts[0].value
    return None


def _finished(snap) -> bool:
    return snap is not None and not snap.next


def _resume(response: dict) -> None:
    """Resume the graph with analyst response, then rerun Streamlit."""
    graph.invoke(Command(resume=response), _config())
    st.rerun()


def _start_run(task: str) -> None:
    """Create a new thread and start the graph."""
    mc = st.session_state.get("metric_config")
    thread_id = str(uuid.uuid4())
    st.session_state.thread_id = thread_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: dict = {
        "task":       task,
        "db_backend": "duckdb",
        "run_id":     thread_id,   # link memory store entry to this Streamlit thread
    }
    if mc:
        initial_state["metric_config"] = mc
    if db := st.session_state.get("db_conn"):
        initial_state["db_backend"] = db.backend
    if user := st.session_state.get("user"):
        initial_state["user_id"] = user["user_id"]

    graph.invoke(initial_state, config)
    st.rerun()


def _reset() -> None:
    st.session_state.thread_id = None
    st.session_state.current_gate = None
    st.rerun()


def _download_button(narrative: str, snap=None) -> None:
    """Render a PDF download button for the given narrative."""
    if not narrative:
        return
    vals = (snap.values or {}) if snap else {}
    task           = vals.get("task", "DataPilot Analysis")
    recommendation = vals.get("recommendation", "")
    metric         = vals.get("metric", "")
    cost_usd       = vals.get("estimated_cost_usd") or 0.0
    try:
        pdf_bytes = build_pdf(
            task=task,
            narrative=narrative,
            recommendation=recommendation,
            metric=metric,
            cost_usd=cost_usd,
        )
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name="datapilot_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"PDF generation failed: {exc}")


def _update_session_costs(snap) -> None:
    """Read token counts from graph state and update sidebar cost accumulators."""
    if snap is None:
        return
    vals = snap.values or {}
    cost  = vals.get("estimated_cost_usd") or 0.0
    cr    = vals.get("cache_read_tokens")   or 0
    # Saving = tokens that were served from cache at $0.30/M instead of $3.00/M
    saved = cr * (3.00 - 0.30) / 1_000_000
    st.session_state.session_cost_usd  = cost
    st.session_state.session_saved_usd = saved


# ── Gate renderers ────────────────────────────────────────────────────────────

def _render_query_gate(payload: dict, snap=None) -> None:
    st.subheader("Gate 1 of 3 — Review SQL")
    cache_hit = payload.get("cache_hit", False)
    if cache_hit:
        st.info("This SQL was retrieved from the semantic cache.")

    # Show a preview of what the query returned so the analyst can catch
    # wrong column names before approving.
    if snap:
        df = (snap.values or {}).get("query_result")
        if df is not None and not df.empty:
            mc = (snap.values or {}).get("metric_config")
            required = set()
            if mc:
                required = {mc.primary_metric, mc.covariate, "variant"}
            missing = required - set(df.columns) if required else set()
            if missing:
                st.error(
                    f"⚠️ Query result is missing required columns: "
                    f"**{', '.join(sorted(missing))}**  \n"
                    "Edit the SQL below to alias them correctly, then re-approve."
                )
            else:
                st.success(f"Query returned {len(df):,} rows, {len(df.columns)} columns.")
            with st.expander("Preview (first 5 rows)"):
                st.dataframe(df.head())

    sql = st.text_area(
        "Generated SQL — edit if needed:",
        value=payload.get("generated_sql", ""),
        height=200,
        key="sql_editor",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("✅ Approve & run", type="primary", use_container_width=True):
            _resume({"approved": True, "sql": sql})
    with col2:
        if st.button("🔄 Re-generate SQL", use_container_width=True):
            _resume({"approved": False, "sql": None})


def _metric_badge(label: str, value, good: bool | None = None) -> None:
    """Render a single metric as a coloured badge."""
    if value is None:
        colour = "gray"
    elif good is True:
        colour = "green"
    elif good is False:
        colour = "red"
    else:
        colour = "blue"
    st.markdown(
        f"<span style='background:{colour};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.85em'>{label}: {value}</span>",
        unsafe_allow_html=True,
    )


def _render_analysis_gate(payload: dict, snap) -> None:
    st.subheader("Gate 2 of 3 — Review Analysis Results")

    vals = snap.values if snap else {}

    # ── Summary cards ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Experiment**")
        sig = payload.get("significant")
        vr  = payload.get("cuped_variance_reduction")
        st.metric("Significant?",    "Yes ✅" if sig else "No ❌")
        st.metric("CUPED variance ↓", f"{vr:.1f}%" if vr else "n/a")
        st.metric("Top segment",     payload.get("top_segment") or "—")

    with col2:
        st.markdown("**Time series**")
        st.metric("Forecast outside CI?", "Yes ✅" if payload.get("forecast_outside_ci") else "No")
        novelty = payload.get("novelty_likely")
        st.metric("Novelty effect?",      "Yes ⚠️" if novelty else "No ✅")

    with col3:
        st.markdown("**Secondary metrics**")
        breached = payload.get("guardrails_breached")
        st.metric("Guardrails breached?", "Yes ⚠️" if breached else "No ✅")
        st.metric("Biggest funnel drop",  payload.get("biggest_funnel_dropoff") or "—")
        powered = payload.get("mde_powered")
        st.metric("Powered?",             "Yes ✅" if powered else "No ⚠️")

    # ── Business impact ───────────────────────────────────────────────────────
    if impact := payload.get("business_impact"):
        st.info(impact)

    # ── Breached guardrails detail ────────────────────────────────────────────
    breached_list = payload.get("breached_metrics", [])
    if breached_list:
        st.warning(f"**{len(breached_list)} guardrail(s) breached:**")
        for g in breached_list:
            st.markdown(
                f"- **{g.get('metric')}**: control={g.get('control_mean', 0):.4f}, "
                f"treatment={g.get('treatment_mean', 0):.4f}, "
                f"Δ={g.get('delta_pct', 0):+.1f}%, p={g.get('p_value', 1):.4f}"
            )

    # ── Decomposition ─────────────────────────────────────────────────────────
    decomp = payload.get("decomposition", {})
    if decomp:
        with st.expander("DAU decomposition"):
            for k in ["new", "retained", "resurrected", "churned"]:
                v = decomp.get(k, {})
                if v:
                    pct = v.get("pct_of_dau", 0)
                    st.write(f"**{k.capitalize()}**: {v.get('count', 0):,} ({pct:.1f}%)")
            dominant = decomp.get("dominant_change_component")
            if dominant:
                st.write(f"**Dominant change:** {dominant}")

    # ── Analyst notes ─────────────────────────────────────────────────────────
    st.markdown("---")
    analyst_notes = st.text_area(
        "Analyst notes / overrides (optional):",
        placeholder="Add context, flag data quality issues, override a finding…",
        key="analyst_notes_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("✅ Approve & generate narrative", type="primary", use_container_width=True):
            _resume({"approved": True, "notes": analyst_notes})
    with col2:
        if st.button("↩ Back to task", use_container_width=True):
            _reset()


def _render_narrative_gate(payload: dict, snap=None) -> None:
    st.subheader("Gate 3 of 3 — Review Narrative")

    narrative = payload.get("narrative_draft", "")
    recommendation = payload.get("recommendation", "")

    if recommendation:
        st.success(f"**Recommendation:** {recommendation}")

    st.markdown(narrative)
    st.markdown("---")

    analyst_notes = st.text_area(
        "Request revisions (leave blank to approve as-is):",
        placeholder="e.g. 'Add a note about the sample size being smaller in week 2'",
        key="narrative_notes_input",
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            _resume({"approved": True, "notes": analyst_notes})
    with col2:
        if st.button("✏️ Revise", use_container_width=True):
            if not analyst_notes.strip():
                st.warning("Add revision notes before requesting a revision.")
            else:
                _resume({"approved": False, "notes": analyst_notes})
    with col3:
        _download_button(narrative, snap=snap)


def _render_semantic_cache_gate(payload: dict) -> None:
    hit_type = payload.get("hit_type", "hard")
    st.subheader("Cached analysis found" if hit_type == "hard" else "Similar analysis found")
    sim = payload.get("similarity", 0.0)
    if hit_type == "hard":
        st.info(
            f"This task is **{sim:.0%} identical** to a previous analysis. "
            "Use the cached result (instant) or re-run the full pipeline?"
        )
    else:
        st.warning(
            f"This task is **{sim:.0%} similar** to a previous analysis. "
            "Review the cached result below — re-run if your question differs."
        )
    if rec := payload.get("recommendation"):
        st.success(f"**Cached recommendation:** {rec}")
    if narrative := payload.get("narrative_draft"):
        with st.expander("Preview cached narrative", expanded=True):
            st.markdown(narrative)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Use cached result", type="primary", use_container_width=True):
            _resume({"approved": True})
    with col2:
        if st.button("🔄 Re-run full analysis", use_container_width=True):
            _resume({"approved": False})


def _render_finished(snap) -> None:
    st.success("Analysis complete.")
    vals = snap.values if snap else {}

    final = vals.get("final_narrative") or vals.get("narrative_draft", "")
    rec   = vals.get("recommendation", "")

    if rec:
        st.success(f"**Recommendation:** {rec}")

    if final:
        st.markdown(final)
    else:
        st.info("No narrative generated. Check that ANTHROPIC_API_KEY is set.")

    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        _download_button(final, snap=snap)
    with col2:
        if st.button("🔁 Start a new analysis", use_container_width=False):
            _reset()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    # ── User badge + logout ───────────────────────────────────────────────────
    if user := st.session_state.get("user"):
        st.sidebar.markdown(
            f"**{user['username']}** &nbsp;·&nbsp; "
            f"<span style='font-size:0.8em;color:#888'>{user['email']}</span>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Sign out", use_container_width=True):
            # Clear all session state on logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.sidebar.markdown("---")

    render_connection_sidebar()

    # ── Past runs ─────────────────────────────────────────────────────────────
    with st.sidebar.expander("Past runs", expanded=False):
        try:
            _uid = (st.session_state.get("user") or {}).get("user_id")
            runs = get_all_runs(limit=10, user_id=_uid)
        except Exception:
            runs = []
        if runs:
            for run in runs:
                score = run.get("eval_score")
                score_str = f" | score={score:.2f}" if score else ""
                st.markdown(
                    f"**{run.get('task', '')[:50]}**  \n"
                    f"`{run.get('timestamp', '')[:16]}` | "
                    f"{run.get('metric', '')} | "
                    f"top={run.get('top_segment', '')}"
                    f"{score_str}"
                )
                st.divider()
        else:
            st.caption("No past runs yet.")

    # ── Cost ──────────────────────────────────────────────────────────────────
    cost  = st.session_state.session_cost_usd
    saved = st.session_state.session_saved_usd
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Session cost: **${cost:.4f}**  \n"
        f"Saved via caching: **${saved:.4f}**"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Auth gate — must be authenticated to use the app ──────────────────────
    if not st.session_state.get("user"):
        render_auth_page()
        return

    st.title("🧭 DataPilot")
    st.caption("AI Product Data Scientist — DAU drop investigation demo")

    _render_sidebar()

    snap = _snap()
    _update_session_costs(snap)
    payload = _interrupt_payload(snap)

    # ── Idle: show task input ──────────────────────────────────────────────────
    if not st.session_state.thread_id:
        st.markdown("### What do you want to investigate?")
        task = st.text_area(
            "Task",
            value="Why did DAU drop in the most recent experiment?",
            height=80,
            label_visibility="collapsed",
        )
        if st.button("🚀 Run analysis", type="primary"):
            if not task.strip():
                st.warning("Enter a task first.")
            else:
                with st.spinner("Starting analysis…"):
                    _start_run(task.strip())
        return

    # ── Active run: show progress header ──────────────────────────────────────
    if snap:
        next_nodes = snap.next
        task_str   = (snap.values or {}).get("task", "")
        if task_str:
            st.markdown(f"**Task:** {task_str}")
        if next_nodes and not _finished(snap):
            st.progress(
                _gate_progress(payload),
                text=f"Running: {', '.join(next_nodes)}…" if not payload else "",
            )

    # ── Render correct gate ────────────────────────────────────────────────────
    if payload:
        gate = payload.get("gate")
        st.session_state.current_gate = gate

        if gate == "semantic_cache":
            _render_semantic_cache_gate(payload)
        elif gate == "query":
            _render_query_gate(payload, snap)
        elif gate == "analysis":
            _render_analysis_gate(payload, snap)
        elif gate == "narrative":
            _render_narrative_gate(payload, snap)
        else:
            st.warning(f"Unknown gate: {gate!r}")

    elif _finished(snap):
        _render_finished(snap)

    else:
        # Graph is running (no interrupt yet) — this branch shouldn't normally
        # be visible since invoke() blocks until an interrupt or completion.
        st.info("Analysis in progress…")
        if st.button("↩ Cancel"):
            _reset()


def _gate_progress(payload: dict | None) -> float:
    if payload is None:
        return 1.0
    gate = (payload or {}).get("gate", "")
    return {"semantic_cache": 0.05, "query": 0.15, "analysis": 0.75, "narrative": 0.95}.get(gate, 0.5)


if __name__ == "__main__":
    main()
