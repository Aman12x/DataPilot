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
    t_stat:           float
    p_value:          float
    ci_lower:         float
    ci_upper:         float
    significant:      bool
    cohens_d:         float         = 0.0   # pooled-std effect size; |d|<0.2 small, 0.2–0.5 medium, >0.5 large
    n_control:        int           = 0
    n_treatment:      int           = 0
    winsorized:       bool          = False  # True when outlier-clipping was applied
    skewness_warning: Optional[str] = None   # set when |skew| > 2 in either arm


class SegmentResult(BaseModel):
    segment:       str
    effect_size:   float
    segment_share: float
    n_control:     int
    n_treatment:   int
    p_value:       float
    significant:   bool


class HteResult(BaseModel):
    top_segment:        str
    effect_size:        float
    segment_share:      float
    all_segments:       list[SegmentResult]
    interaction_p_value: Optional[float] = None  # OLS interaction F-test p-value; None if not computable


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


class SensitivityRow(BaseModel):
    mde_pct:      float   # target MDE as % of baseline
    n_per_arm:    int     # required sample size per arm
    runtime_days: int     # estimated days to collect that many users


class PowerAnalysisResult(BaseModel):
    baseline_mean:       float
    baseline_std:        float
    daily_traffic:       float          # estimated unique users per day
    mde_target_pct:      float          # primary MDE target the user asked about
    mde_target_abs:      float          # absolute value of the primary MDE
    required_n_per_arm:  int            # sample size at mde_target_pct
    required_total_n:    int            # = 2 * required_n_per_arm
    runtime_days:        int            # days to reach required_total_n at daily_traffic
    alpha:               float
    power:               float
    guardrails_to_watch: list[str]      # guardrail metric names from metric config
    sensitivity:         list[SensitivityRow]   # table across MDE levels


# ── narrative_tools ───────────────────────────────────────────────────────────

class NarrativeResult(BaseModel):
    narrative_draft: str
    recommendation:  str


# ── describe_tools ────────────────────────────────────────────────────────────

class ColumnSummary(BaseModel):
    name:       str
    dtype:      str
    non_null:   int
    null_count: int
    # numeric
    mean:       Optional[float] = None
    std:        Optional[float] = None
    min:        Optional[float] = None
    p25:        Optional[float] = None
    median:     Optional[float] = None
    p75:        Optional[float] = None
    max:        Optional[float] = None
    # categorical
    n_unique:   Optional[int]   = None
    top_values: Optional[list[str]] = None   # top-5 value: count strings


class DescribeResult(BaseModel):
    row_count:   int
    col_count:   int
    columns:     list[ColumnSummary]
    top_rows:    Optional[list[dict]] = None   # top-10 rows by highest-variance numeric col
    trend_rows:  Optional[list[dict]] = None   # rows aggregated by detected time/group col


class CorrelationPair(BaseModel):
    col_a:       str
    col_b:       str
    correlation: float          # Pearson r, rounded to 4 dp


class CorrelationResult(BaseModel):
    pairs: list[CorrelationPair]  # top N by |r|, descending


# ── chart_tools ───────────────────────────────────────────────────────────────

class ChartSpec(BaseModel):
    """Serialisable chart definition consumed by recharts in the frontend."""
    chart_type:     str                         # 'bar' | 'bar_horizontal' | 'line' | 'scatter'
    title:          str
    insight:        str                         # 1-sentence plain-language takeaway
    data:           list[dict]                  # recharts-compatible array of dicts
    x_key:          str
    y_key:          str
    y_key2:         Optional[str]  = None       # second series (grouped bars)
    color:          str            = "#89b4fa"
    color2:         Optional[str]  = None
    error_bar_low:  Optional[str]  = None
    error_bar_high: Optional[str]  = None
    x_label:        Optional[str]  = None
    y_label:        Optional[str]  = None


class TrustIndicators(BaseModel):
    n_data_points:     int
    confidence_level:  str   # 'high' | 'medium' | 'low'
    confidence_reason: str
