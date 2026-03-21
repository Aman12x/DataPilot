"""
tools/schemas.py — Pydantic v2 return-type models for all tool functions.

Every tool function returns one of these models. No plain dicts leave tools/.
Consumers access fields by attribute (result.cuped_ate) not by key (result["cuped_ate"]).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


# ── stats_tools ───────────────────────────────────────────────────────────────

class CupedResult(BaseModel):
    raw_ate:                float
    cuped_ate:              float
    variance_reduction_pct: float
    theta:                  float


class TtestResult(BaseModel):
    t_stat:     float
    p_value:    float
    ci_lower:   float
    ci_upper:   float
    significant: bool


class SegmentResult(BaseModel):
    segment:       str
    effect_size:   float
    segment_share: float
    n_control:     int
    n_treatment:   int
    p_value:       float
    significant:   bool


class HteResult(BaseModel):
    top_segment:   str
    effect_size:   float
    segment_share: float
    all_segments:  list[SegmentResult]


# ── decomposition_tools ───────────────────────────────────────────────────────

class ComponentStats(BaseModel):
    time_series:  dict[str, int]
    baseline_avg: float
    recent_avg:   float
    delta:        float
    pct_of_dau:   float


class SegmentBreakdown(BaseModel):
    """Generic per-segment metric breakdown used by decompose_metric."""
    segment_col:      str
    segment_value:    str
    metric_before:    float
    metric_after:     float
    delta:            float
    contribution_pct: float


class DecompositionResult(BaseModel):
    # DAU-specific (Optional — only set when decompose_dau is called)
    new:          Optional[ComponentStats] = None
    retained:     Optional[ComponentStats] = None
    resurrected:  Optional[ComponentStats] = None
    churned:      Optional[ComponentStats] = None
    # Generic (always set)
    dominant_change_component: str
    segments: list[SegmentBreakdown] = Field(default_factory=list)


# ── anomaly_tools ─────────────────────────────────────────────────────────────

class AnomalyResult(BaseModel):
    anomaly_dates: list[str]
    severity:      float
    direction:     str          # 'drop' | 'spike'


class SliceDimension(BaseModel):
    dimension:        str
    value:            str
    contribution_pct: float
    delta:            float


class SliceResult(BaseModel):
    ranked_dimensions: list[SliceDimension]


# ── funnel_tools ──────────────────────────────────────────────────────────────

class FunnelStep(BaseModel):
    step:           str
    control_rate:   float
    treatment_rate: float
    delta:          float
    pct_change:     float
    p_value:        float
    significant:    bool


class FunnelResult(BaseModel):
    steps:                list[FunnelStep]
    biggest_dropoff_step: str


# ── forecast_tools ────────────────────────────────────────────────────────────

class ForecastResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    forecast_df:              Optional[pd.DataFrame]
    actual_vs_forecast_delta: float
    outside_ci:               bool
    method:                   str
    warning:                  Optional[str]


# ── guardrail_tools ───────────────────────────────────────────────────────────

class GuardrailMetric(BaseModel):
    metric:         str
    control_mean:   float
    treatment_mean: float
    delta_pct:      float
    p_value:        float
    breached:       bool


class GuardrailResult(BaseModel):
    guardrails:    list[GuardrailMetric]
    any_breached:  bool
    breached_count: int


# ── novelty_tools ─────────────────────────────────────────────────────────────

class NoveltyResult(BaseModel):
    week1_ate:        float
    week2_ate:        float
    effect_direction: str   # 'decaying' | 'growing' | 'stable'
    novelty_likely:   bool


# ── mde_tools ─────────────────────────────────────────────────────────────────

class MdeResult(BaseModel):
    mde_absolute:                   float
    mde_relative_pct:               float
    is_powered_for_observed_effect: Optional[bool]


# ── narrative_tools ───────────────────────────────────────────────────────────

class NarrativeResult(BaseModel):
    narrative_draft: str
    recommendation:  str
