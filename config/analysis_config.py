"""
config/analysis_config.py — MetricConfig Pydantic model + loader.

MetricConfig is the single source of truth for all metric references that
were previously hardcoded across nodes.py, guardrail_tools.py, and prompts.

Loading order:
  1. Try to read config/metric_config.json (or path arg)
  2. Fall back to env-var defaults (same values as current _DEFAULT_* constants)
  3. Absolute fallback: DAU scenario hard-coded constants

This means existing deployments that rely on env vars continue to work
without a JSON file present.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


# ── Model ─────────────────────────────────────────────────────────────────────

class MetricConfig(BaseModel):
    # ── Core metric identity ──────────────────────────────────────────────────
    primary_metric: str               # output column alias, e.g. "dau_rate"
    metric_source_col: str = ""       # raw DB column, e.g. "dau_flag" (may differ from alias)
    metric_agg: str = "mean"          # "mean" | "sum" | "count"
    covariate: str
    metric_direction: Literal["higher_is_better", "lower_is_better"]

    # ── Table names (all overridable) ─────────────────────────────────────────
    events_table: str = "events"
    experiment_table: str = "experiment"
    timeseries_table: Optional[str] = None   # pre-aggregated daily table; None = aggregate from events
    funnel_table: Optional[str] = None       # funnel step table; None = skip funnel

    # ── Column name overrides ─────────────────────────────────────────────────
    user_id_col: str = "user_id"
    date_col: str = "date"
    variant_col: str = "variant"
    week_col: str = "week"

    # ── Analysis configuration ────────────────────────────────────────────────
    guardrail_metrics: list[str]
    segment_cols: list[str]
    funnel_steps: list[str] = []

    # ── Business impact ───────────────────────────────────────────────────────
    revenue_per_unit: float = 0.50
    baseline_unit_count: int = 500_000   # replaces BASELINE_DAU env var
    experiment_weeks: int = 2

    # ── Optional per-metric harm direction override ───────────────────────────
    # Values: 'increase' | 'decrease' | 'both'
    # When None, guardrail_tools falls back to keyword inference.
    guardrail_harm_directions: Optional[dict[str, str]] = None

    # ── Field validators ──────────────────────────────────────────────────────

    @field_validator(
        "primary_metric", "covariate",
        "events_table", "experiment_table",
        "user_id_col", "date_col", "variant_col", "week_col",
        mode="before",
    )
    @classmethod
    def _non_empty_string(cls, v: object) -> str:
        """
        Reject empty or whitespace-only strings for any column / table name field.
        Catches hallucinated empty strings from LLM inference before they
        propagate to _canonical_experiment_sql() and produce invalid SQL.
        """
        if not isinstance(v, str) or not v.strip():
            raise ValueError(
                f"Column/table name must be a non-empty string, got {v!r}"
            )
        return v.strip()

    @field_validator("metric_source_col", mode="before")
    @classmethod
    def _strip_source_col(cls, v: object) -> str:
        """Strip whitespace; allow empty string (model_validator fills it in)."""
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

    @field_validator("guardrail_metrics", "segment_cols", mode="before")
    @classmethod
    def _non_empty_list_elements(cls, v: object) -> list[str]:
        """
        Reject lists containing empty, whitespace-only, or non-string elements.
        Prevents SQL errors from empty column names in AVG(e.) patterns.
        """
        if not isinstance(v, list):
            raise ValueError(f"Expected a list, got {type(v).__name__}")
        result: list[str] = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"All list elements must be non-empty strings; got {item!r}"
                )
            result.append(item.strip())
        return result

    @model_validator(mode="after")
    def _default_source_col(self) -> "MetricConfig":
        """
        If metric_source_col is not explicitly set, default to primary_metric.
        This means for configs where the DB column and alias are the same
        (e.g. retention_experiment), no extra field is required.
        """
        if not self.metric_source_col:
            self.metric_source_col = self.primary_metric
        return self


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv(env_key: str, fallback: str) -> list[str]:
    """Read a comma-separated env var; fall back to a default list."""
    return [v.strip() for v in os.getenv(env_key, fallback).split(",") if v.strip()]


# ── Default DAU config (built from env vars, matching current _DEFAULT_* values) ──

DEFAULT_DAU_CONFIG = MetricConfig(
    primary_metric=os.getenv("DEFAULT_METRIC",          "dau_rate"),
    metric_source_col=os.getenv("DEFAULT_METRIC_SOURCE_COL", "dau_flag"),
    metric_agg=os.getenv("DEFAULT_METRIC_AGG",          "mean"),
    covariate=os.getenv("DEFAULT_COVARIATE",            "pre_session_count"),
    metric_direction=os.getenv("DEFAULT_METRIC_DIRECTION", "higher_is_better"),  # type: ignore[arg-type]
    events_table=os.getenv("DEFAULT_EVENTS_TABLE",      "events"),
    experiment_table=os.getenv("DEFAULT_EXPERIMENT_TABLE", "experiment"),
    timeseries_table=os.getenv("DEFAULT_TIMESERIES_TABLE", "metrics_daily") or None,
    funnel_table=os.getenv("DEFAULT_FUNNEL_TABLE",      "funnel") or None,
    user_id_col=os.getenv("DEFAULT_USER_ID_COL",        "user_id"),
    date_col=os.getenv("DEFAULT_DATE_COL",              "date"),
    variant_col=os.getenv("DEFAULT_VARIANT_COL",        "variant"),
    week_col=os.getenv("DEFAULT_WEEK_COL",              "week"),
    guardrail_metrics=_csv("DEFAULT_GUARDRAILS",        "notif_optout,d7_retained,session_count"),
    segment_cols=_csv("DEFAULT_SEGMENT_COLS",           "platform,user_segment"),
    funnel_steps=_csv("DEFAULT_FUNNEL_STEPS",           "impression,click,install,d1_retain"),
    revenue_per_unit=float(os.getenv("REVENUE_PER_DAU",     "0.50")),
    baseline_unit_count=int(os.getenv("BASELINE_DAU",       "500000")),
    experiment_weeks=2,
    guardrail_harm_directions=None,
)


# ── Loader ────────────────────────────────────────────────────────────────────

def load_metric_config(path: str = "config/metric_config.json") -> MetricConfig:
    """
    Load MetricConfig from a JSON file.

    Falls back to DEFAULT_DAU_CONFIG (env-backed) if the file is absent or
    malformed — so the system works out of the box without a config file.
    """
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return MetricConfig(**data)
        except Exception:
            pass  # fall through to env-var defaults
    return DEFAULT_DAU_CONFIG
