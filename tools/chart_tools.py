"""
tools/chart_tools.py — Deterministic chart generation from analysis results.

No LLM calls. Consumes Pydantic result objects already in AgentState and
produces a list of ChartSpec dicts ready for recharts on the frontend.

Two public entry points:
  generate_general_charts(describe, correlation)  → list[ChartSpec]
  generate_ab_charts(metric, ttest, cuped, hte, novelty, funnel) → list[ChartSpec]

And one helper:
  compute_trust_indicators(describe, ttest, n_rows) → TrustIndicators
"""

from __future__ import annotations

from typing import Optional

from tools.schemas import (
    ChartSpec,
    CorrelationResult,
    CupedResult,
    DescribeResult,
    FunnelResult,
    HteResult,
    NoveltyResult,
    TrustIndicators,
    TtestResult,
)

# ── Colour palette (Catppuccin Mocha) ─────────────────────────────────────────
_BLUE   = "#89b4fa"
_PURPLE = "#cba6f7"
_GREEN  = "#a6e3a1"
_RED    = "#f38ba8"
_PEACH  = "#fab387"
_YELLOW = "#f9e2af"


# ── General-analysis charts ────────────────────────────────────────────────────

def generate_general_charts(
    describe: DescribeResult,
    correlation: CorrelationResult,
) -> list[ChartSpec]:
    """
    Produce up to 4 charts for general (non-A/B) analysis:
      1. Top correlations (horizontal bar)
      2. Missing-data overview (bar)        — only if any nulls exist
      3. Top categorical distribution (bar) — first low-cardinality column
      4. Numeric spread summary (bar)       — percentile chart for most variable column
    """
    charts: list[ChartSpec] = []

    # 1. Top correlations ──────────────────────────────────────────────────────
    if correlation.pairs:
        top = sorted(correlation.pairs, key=lambda p: abs(p.correlation), reverse=True)[:8]
        data = [
            {"pair": f"{p.col_a} × {p.col_b}", "r": round(p.correlation, 3)}
            for p in top
        ]
        strongest = top[0]
        direction = "positively" if strongest.correlation > 0 else "negatively"
        charts.append(ChartSpec(
            chart_type="bar_horizontal",
            title="Strongest relationships in the data",
            insight=(
                f"**{strongest.col_a}** and **{strongest.col_b}** move "
                f"{direction} together (r\u202f=\u202f{strongest.correlation:.2f}) "
                f"— the strongest link found."
            ),
            data=data,
            x_key="r",
            y_key="pair",
            color=_BLUE,
            x_label="Correlation (r)",
        ))

    # 2. Missing data ──────────────────────────────────────────────────────────
    cols_with_nulls = [
        c for c in describe.columns
        if c.null_count and c.null_count > 0
    ]
    if cols_with_nulls:
        total_per_col = {
            c.name: (c.non_null or 0) + (c.null_count or 0)
            for c in describe.columns
        }
        by_missing = sorted(cols_with_nulls, key=lambda c: c.null_count or 0, reverse=True)[:10]
        data = [
            {
                "column": c.name,
                "missing_%": round(100 * (c.null_count or 0) / max(total_per_col.get(c.name, 1), 1), 1),
            }
            for c in by_missing
        ]
        worst = data[0]
        charts.append(ChartSpec(
            chart_type="bar",
            title="Missing values by column",
            insight=(
                f"**{worst['column']}** is missing in {worst['missing_%']}% of rows "
                f"— verify data collection for this field."
            ),
            data=data,
            x_key="column",
            y_key="missing_%",
            color=_RED,
            y_label="% Missing",
        ))

    # 3. Top categorical distribution ──────────────────────────────────────────
    # Skip pivot/unpivot artifact column names and columns where every value
    # appears exactly once (count = 1 for all top_values — meaningless distribution).
    _SKIP_CAT_NAMES = {"variable", "value", "metric", "measure", "column_name",
                       "statistic", "feature", "name", "key", "field"}

    def _all_count_one(col_summary) -> bool:
        """Return True if every top_value entry has count 1 (fully unique — not a useful category)."""
        tvs = col_summary.top_values or []
        if not tvs:
            return False
        return all(tv.endswith(": 1") for tv in tvs)

    cat_cols = [
        c for c in describe.columns
        if c.top_values
        and c.n_unique is not None
        and 1 < c.n_unique <= 30
        and c.name.lower() not in _SKIP_CAT_NAMES
        and not _all_count_one(c)
    ]
    if cat_cols:
        col = cat_cols[0]
        data = []
        for tv in (col.top_values or [])[:10]:
            if ": " in tv:
                val, cnt_str = tv.rsplit(": ", 1)
                try:
                    data.append({"category": val, "count": int(cnt_str)})
                except ValueError:
                    pass
        if data:
            top_cat = data[0]["category"]
            charts.append(ChartSpec(
                chart_type="bar",
                title=f"Breakdown of {col.name}",
                insight=f"**{top_cat}** is the most common value for {col.name}.",
                data=data,
                x_key="category",
                y_key="count",
                color=_PURPLE,
                y_label="Count",
            ))

    # 4. Numeric spread (percentile chart) ────────────────────────────────────
    # Skip columns that are themselves statistical artefacts (correlation scores,
    # p-values, aggregation indices) — their spread is not meaningful as raw data.
    _SKIP_NUM_NAMES = {"correlation", "r", "r2", "p_value", "pvalue", "p", "coef",
                       "coefficient", "weight", "score", "rank", "index", "value"}
    num_cols = [
        c for c in describe.columns
        if c.median is not None
        and c.std is not None
        and c.min is not None
        and c.max is not None
        and c.name.lower() not in _SKIP_NUM_NAMES
    ]
    if num_cols:
        # Pick column with highest coefficient of variation
        col = max(
            num_cols,
            key=lambda c: (c.std or 0) / max(abs(c.mean or 0), 1e-9),
        )
        data = [
            {"percentile": "Min",    "value": round(col.min or 0, 3)},
            {"percentile": "25th",   "value": round(col.p25 or 0, 3)},
            {"percentile": "Median", "value": round(col.median or 0, 3)},
            {"percentile": "75th",   "value": round(col.p75 or 0, 3)},
            {"percentile": "Max",    "value": round(col.max or 0, 3)},
        ]
        spread = (col.max or 0) - (col.min or 0)
        iqr    = (col.p75 or 0) - (col.p25 or 0)
        if spread > 0 and iqr > 0 and spread / iqr > 5:
            insight = (
                f"**{col.name}** has a wide spread — a few extreme values are "
                f"pulling the average away from the typical value."
            )
        else:
            insight = (
                f"**{col.name}** median is {col.median:.3f}, "
                f"with most values between {col.p25:.3f} and {col.p75:.3f}."
            )
        charts.append(ChartSpec(
            chart_type="bar",
            title=f"Spread of {col.name}",
            insight=insight,
            data=data,
            x_key="percentile",
            y_key="value",
            color=_GREEN,
            y_label=col.name,
        ))

    return charts


# ── A/B-test charts ───────────────────────────────────────────────────────────

def generate_ab_charts(
    metric:  str,
    ttest:   TtestResult,
    cuped:   CupedResult,
    hte:     Optional[HteResult]    = None,
    novelty: Optional[NoveltyResult] = None,
    funnel:  Optional[FunnelResult]  = None,
) -> list[ChartSpec]:
    """
    Produce up to 4 charts for A/B experiment analysis:
      1. Primary effect (treatment ATE ± CI)
      2. Segment effects HTE (horizontal bar)  — if hte is available
      3. Novelty check (week 1 vs week 2)      — if novelty is available
      4. Funnel comparison (control vs treatment bar) — if funnel is available
    """
    charts: list[ChartSpec] = []

    # 1. Primary effect ────────────────────────────────────────────────────────
    ate   = cuped.cuped_ate
    color = _GREEN if ate >= 0 else _RED
    sig   = ttest.significant

    data = [
        {
            "label": f"Δ {metric}",
            "ate":   round(ate, 5),
            "ci_lo": round(ttest.ci_lower, 5),
            "ci_hi": round(ttest.ci_upper, 5),
        }
    ]
    if sig:
        insight = (
            f"Treatment {'increased' if ate > 0 else 'decreased'} **{metric}** "
            f"by {ate:+.4f} (p\u202f=\u202f{ttest.p_value:.3f}) — statistically significant."
        )
    else:
        insight = (
            f"No significant change in **{metric}** detected "
            f"(p\u202f=\u202f{ttest.p_value:.3f}). The difference is likely noise."
        )
    charts.append(ChartSpec(
        chart_type="bar",
        title=f"Treatment effect on {metric}",
        insight=insight,
        data=data,
        x_key="label",
        y_key="ate",
        error_bar_low="ci_lo",
        error_bar_high="ci_hi",
        color=color,
        y_label=f"Δ {metric}",
    ))

    # 2. HTE — segment effects ─────────────────────────────────────────────────
    if hte and hte.all_segments:
        segs = sorted(hte.all_segments, key=lambda s: abs(s.effect_size), reverse=True)[:8]
        data = [
            {
                "segment":   s.segment,
                "effect":    round(s.effect_size, 5),
                "sig":       s.significant,
            }
            for s in segs
        ]
        top = segs[0]
        charts.append(ChartSpec(
            chart_type="bar_horizontal",
            title="Who benefits most? (segment effects)",
            insight=(
                f"**{top.segment}** shows the largest effect "
                f"({top.effect_size:+.4f})"
                + (" — significant." if top.significant else " — not yet significant.")
            ),
            data=data,
            x_key="effect",
            y_key="segment",
            color=_PURPLE,
            x_label=f"Δ {metric}",
        ))

    # 3. Novelty check ────────────────────────────────────────────────────────
    if novelty:
        data = [
            {"week": "Week 1", "effect": round(novelty.week1_ate, 5)},
            {"week": "Week 2", "effect": round(novelty.week2_ate, 5)},
        ]
        if novelty.novelty_likely:
            insight = (
                f"Effect is {novelty.effect_direction} over time — "
                f"the early boost may be a novelty effect, not a lasting change."
            )
        else:
            insight = (
                f"Effect is {novelty.effect_direction} across both weeks — "
                f"this looks like a real, sustained improvement."
            )
        charts.append(ChartSpec(
            chart_type="bar",
            title="Is the effect holding up over time?",
            insight=insight,
            data=data,
            x_key="week",
            y_key="effect",
            color=_PEACH,
            y_label=f"Δ {metric}",
        ))

    # 4. Funnel comparison ────────────────────────────────────────────────────
    if funnel and funnel.steps:
        data = [
            {
                "step":      fs.step,
                "Control":   round(fs.control_rate * 100, 1),
                "Treatment": round(fs.treatment_rate * 100, 1),
            }
            for fs in funnel.steps
        ]
        charts.append(ChartSpec(
            chart_type="bar",
            title="Funnel: control vs treatment",
            insight=f"Biggest difference at the **{funnel.biggest_dropoff_step}** step.",
            data=data,
            x_key="step",
            y_key="Control",
            y_key2="Treatment",
            color=_BLUE,
            color2=_PURPLE,
            y_label="Completion rate (%)",
        ))

    return charts


# ── Trust indicators ──────────────────────────────────────────────────────────

def compute_trust_indicators(
    describe: Optional[DescribeResult],
    ttest:    Optional[TtestResult],
    n_rows:   int,
) -> TrustIndicators:
    """
    Return a simple confidence signal for the finished-view trust banner.
    """
    if ttest is not None:
        # A/B experiment — combine p-value, effect size, CI width, and sample size.
        cohens_d  = getattr(ttest, "cohens_d", 0.0)
        n_min     = min(getattr(ttest, "n_control", 0), getattr(ttest, "n_treatment", 0))
        ci_width  = ttest.ci_upper - ttest.ci_lower
        mean_diff = (ttest.ci_lower + ttest.ci_upper) / 2.0
        # Wide CI: uncertainty larger than 2× the effect size (not precise enough to act on)
        ci_wide   = abs(ci_width) > 2.0 * abs(mean_diff) if abs(mean_diff) > 1e-9 else False

        if ttest.significant and ttest.p_value < 0.01 and abs(cohens_d) >= 0.2 and not ci_wide:
            level  = "high"
            reason = (
                f"p\u202f=\u202f{ttest.p_value:.3f}, Cohen's\u202fd\u202f=\u202f{cohens_d:+.2f} "
                f"({'medium' if abs(cohens_d) < 0.5 else 'large'} effect) across "
                f"{n_rows:,} observations — result is robust."
            )
        elif ttest.significant and abs(cohens_d) < 0.1:
            level  = "medium"
            reason = (
                f"p\u202f=\u202f{ttest.p_value:.3f} — statistically significant but Cohen's\u202f"
                f"d\u202f=\u202f{cohens_d:+.2f} is negligible. Validate practical significance "
                f"before shipping."
            )
        elif ttest.significant and (ci_wide or n_min < 100):
            level  = "medium"
            reason = (
                f"p\u202f=\u202f{ttest.p_value:.3f}, Cohen's\u202fd\u202f=\u202f{cohens_d:+.2f} "
                f"— significant but {'CI is wide relative to the effect' if ci_wide else 'small sample'}. "
                f"Consider a follow-up holdout before fully shipping."
            )
        elif ttest.significant:
            level  = "medium"
            reason = (
                f"p\u202f=\u202f{ttest.p_value:.3f}, Cohen's\u202fd\u202f=\u202f{cohens_d:+.2f} "
                f"— significant but borderline p-value. "
                f"Consider a follow-up holdout before fully shipping."
            )
        else:
            level  = "low"
            reason = (
                f"p\u202f=\u202f{ttest.p_value:.3f}, Cohen's\u202fd\u202f=\u202f{cohens_d:+.2f} "
                f"— not statistically significant. "
                f"Run the experiment longer or increase sample size."
            )
    else:
        # General exploration — use row count as a proxy for reliability
        if n_rows >= 1000:
            level  = "high"
            reason = (
                f"Based on {n_rows:,} data points — patterns shown are unlikely to be noise."
            )
        elif n_rows >= 100:
            level  = "medium"
            reason = (
                f"Based on {n_rows:,} data points — patterns are plausible "
                f"but validate with additional data before making decisions."
            )
        else:
            level  = "low"
            reason = (
                f"Only {n_rows:,} data points — treat all findings as preliminary; "
                f"collect more data before drawing firm conclusions."
            )

    return TrustIndicators(
        n_data_points=n_rows,
        confidence_level=level,
        confidence_reason=reason,
    )
