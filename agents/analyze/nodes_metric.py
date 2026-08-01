"""
agents/analyze/nodes_metric.py — Metric Config HITL gate.

After infer_metric_config, pause so the analyst can approve or edit the
metric mapping before SQL generation.  Certified metric packs skip this gate.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from agents.analyze.node_shared import _sanitise_metric_config
from agents.state import AgentState
from agents.tracer import observe
from config.analysis_config import MetricConfig, load_metric_config

logger = logging.getLogger(__name__)


def _config_summary(mc: MetricConfig) -> dict:
    return {
        "primary_metric": mc.primary_metric,
        "metric_source_col": mc.metric_source_col,
        "metric_agg": mc.metric_agg,
        "covariate": mc.covariate,
        "metric_direction": mc.metric_direction,
        "events_table": mc.events_table,
        "experiment_table": mc.experiment_table,
        "user_id_col": mc.user_id_col,
        "date_col": mc.date_col,
        "variant_col": mc.variant_col,
        "guardrail_metrics": list(mc.guardrail_metrics or []),
        "segment_cols": list(mc.segment_cols or []),
        "revenue_per_unit": mc.revenue_per_unit,
        "baseline_unit_count": mc.baseline_unit_count,
    }


@observe(name="metric_config_gate")
def metric_config_gate(state: AgentState) -> dict:
    """
    HITL gate: confirm metric / table / segment mapping.

    Skip when:
      - metric_pack_certified is True (SMB certified pack)
      - SKIP_METRIC_GATE=true (tests / automated evals)
    """
    import os
    if os.getenv("SKIP_METRIC_GATE", "").lower() in ("1", "true", "yes"):
        return {"metric_config_approved": True}

    if state.get("metric_pack_certified"):
        logger.info("metric_config_gate: skipping — certified metric pack")
        return {"metric_config_approved": True}

    # Built-in demo DuckDB (no upload, no saved connection) uses the deploy
    # MetricConfig — no need to ask the analyst to re-confirm every run.
    is_demo = (
        state.get("db_backend", "duckdb") == "duckdb"
        and not state.get("duckdb_path")
        and not state.get("connection_id")
        and not state.get("metric_pack_id")
    )
    if is_demo:
        logger.info("metric_config_gate: skipping — built-in demo dataset")
        return {"metric_config_approved": True}

    mc = state.get("metric_config") or load_metric_config()
    schema_context = state.get("schema_context", "")

    payload = {
        "gate": "metric",
        "metric_config": _config_summary(mc),
        "metric_pack_id": state.get("metric_pack_id") or "",
        "source": "pack" if state.get("metric_pack_id") else "inferred",
        "message": (
            "Confirm how metrics and tables map to your data before SQL is generated. "
            "Edit any field that looks wrong."
        ),
    }
    analyst_response = interrupt(payload)

    approved = analyst_response.get("approved", True)
    edits = analyst_response.get("metric_config") or {}

    if not approved:
        return {"metric_config_approved": False}

    if edits and isinstance(edits, dict):
        try:
            # Merge edits onto current config
            base = mc.model_dump()
            base.update({k: v for k, v in edits.items() if v is not None})
            updated = MetricConfig(**base)
            updated, issues = _sanitise_metric_config(updated, schema_context, mc)
            if issues:
                logger.info("metric_config_gate: sanitiser issues: %s", issues)
            return {
                "metric_config_approved": True,
                "metric_config": updated,
                "metric": updated.primary_metric,
                "covariate": updated.covariate,
            }
        except Exception as exc:
            logger.warning("metric_config_gate: invalid edits ignored: %s", exc)

    return {
        "metric_config_approved": True,
        "metric_config": mc,
        "metric": mc.primary_metric,
        "covariate": mc.covariate,
    }
