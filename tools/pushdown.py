"""
tools/pushdown.py — A/B statistics from in-warehouse sufficient statistics.

The A/B path materialized one row per user into pandas and ran scipy over the
frame. At enterprise scale that is the wall: 10M rows is ~1.4 GB in memory and
~27s of checkpoint serialization per super-step. But every statistic the
pipeline computes — Welch t, CUPED, SRM, HTE, novelty, MDE, guardrails — is a
function of counts, sums, and sums of squares. This module:

  1. wraps the analyst-approved user-level SQL as a subquery and computes
     those moments in the warehouse (three aggregate queries, tens of rows
     returned, regardless of extract size);
  2. reproduces each stats tool's exact output model from the moments,
     mirroring tools/stats_tools.py et al. formula for formula.

The approved SQL is untouched — the query gate reviews the same statement;
the warehouse just aggregates over it instead of shipping the rows.

Known limitation vs the pandas path: winsorization needs quantiles, which
moments cannot provide. ttest_from_stats reports winsorized=False and appends
a note when clipping was requested, rather than silently pretending.
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Any

from scipy import stats as sps

from tools.db_tools import DBConnection, nestable_sql, quote_ident
from tools.schemas import (
    CupedResult,
    FunnelResult,
    FunnelStep,
    GroupMoments,
    GuardrailMetric,
    GuardrailResult,
    HteResult,
    NoveltyResult,
    SegmentResult,
    SufficientStats,
    TtestResult,
)

logger = logging.getLogger(__name__)


# ── Moment arithmetic ─────────────────────────────────────────────────────────

def _mean(g: GroupMoments) -> float:
    return g.s1 / g.n if g.n else 0.0


def _var(g: GroupMoments) -> float:
    """Sample variance (ddof=1) from raw moments."""
    if g.n < 2:
        return 0.0
    return max((g.s2 - g.s1 * g.s1 / g.n) / (g.n - 1), 0.0)


def _skew(g: GroupMoments) -> float:
    """Population skewness (scipy.stats.skew default, bias=True)."""
    if g.n < 2:
        return 0.0
    mean = _mean(g)
    m2 = g.s2 / g.n - mean * mean
    if m2 <= 1e-24:
        return 0.0
    m3 = g.s3 / g.n - 3.0 * mean * (g.s2 / g.n) + 2.0 * mean ** 3
    return m3 / m2 ** 1.5


def _welch(n_t: int, mean_t: float, var_t: float,
           n_c: int, mean_c: float, var_c: float,
           alpha: float, alternative: str) -> tuple[float, float, float, float, float]:
    """Welch t-test from summary stats: (t, p, ci_lower, ci_upper, cohens_d).

    Mirrors stats_tools.run_ttest: scipy ttest_ind(treatment, control,
    equal_var=False), CI on the mean difference with Welch-Satterthwaite dof,
    one-sided CI at α rather than α/2.
    """
    mean_diff = mean_t - mean_c
    se = math.sqrt(var_t / n_t + var_c / n_c)
    if se <= 0:
        raise ValueError("Zero variance in both groups — t-test undefined.")
    t_stat = mean_diff / se
    dof = (var_t / n_t + var_c / n_c) ** 2 / (
        (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    )
    if alternative == "greater":
        p = float(sps.t.sf(t_stat, df=dof))
    elif alternative == "less":
        p = float(sps.t.cdf(t_stat, df=dof))
    else:
        p = float(2.0 * sps.t.sf(abs(t_stat), df=dof))

    tail_alpha = alpha if alternative != "two-sided" else alpha / 2
    t_crit = float(sps.t.ppf(1 - tail_alpha, df=dof))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    pooled_std = math.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
    cohens_d = mean_diff / pooled_std if pooled_std > 1e-12 else 0.0
    return t_stat, p, ci_lower, ci_upper, cohens_d


# ── SQL builder + collection ──────────────────────────────────────────────────

_DOUBLE_TYPE = {
    "duckdb":   "DOUBLE",
    "postgres": "DOUBLE PRECISION",
    "mysql":    "DOUBLE",
    "bigquery": "FLOAT64",
}


def _f(col: str, backend: str, bool_cols: frozenset[str] = frozenset()) -> str:
    """Column as a double-precision expression.

    An explicit CAST, not `1.0 * col`: that multiplication is DECIMAL
    arithmetic on DuckDB/Postgres/MySQL — Σy³ of a BIGINT overflows the
    DECIMAL(38) accumulator (HUGEINT overflow) where float64 never did — and
    it is a binder error outright on a BOOLEAN column, which is what
    read_csv_auto makes of a true/false flag. The pandas path coerced both
    with astype(float); booleans take the CASE form because no backend casts
    BOOL straight to a float type.
    """
    q = quote_ident(col, backend)
    if col in bool_cols:
        return f"(CASE WHEN {q} THEN 1.0 ELSE 0.0 END)"
    return f"CAST({q} AS {_DOUBLE_TYPE.get(backend, 'DOUBLE PRECISION')})"


def _moment_selects(col: str, backend: str, prefix: str,
                    bool_cols: frozenset[str] = frozenset()) -> str:
    y = _f(col, backend, bool_cols)
    q = quote_ident(col, backend)
    return (
        f"SUM(CASE WHEN {q} IS NOT NULL THEN 1 ELSE 0 END) AS {prefix}_n,\n"
        f"SUM(CASE WHEN {q} IS NOT NULL THEN {y} ELSE 0 END) AS {prefix}_s1,\n"
        f"SUM(CASE WHEN {q} IS NOT NULL THEN {y}*{y} ELSE 0 END) AS {prefix}_s2,\n"
        f"SUM(CASE WHEN {q} IS NOT NULL THEN {y}*{y}*{y} ELSE 0 END) AS {prefix}_s3"
    )


def probe_columns(db: DBConnection, base_sql: str) -> list[str]:
    """Column names of the approved SQL without materializing any rows."""
    df = db.query(f"SELECT * FROM (\n{nestable_sql(base_sql)}\n) AS _dp_probe LIMIT 0")
    return list(df.columns)


def probe_bool_columns(db: DBConnection, base_sql: str, cols: list[str]) -> frozenset[str]:
    """Which of `cols` the approved SQL yields as BOOLEAN.

    One-row sample: the SQL builders need to know this to pick the CASE form
    over a CAST. A LIMIT 0 frame carries dtypes on DuckDB/BigQuery but not
    through pd.read_sql on Postgres/MySQL, so one row is sampled instead.
    Best-effort: a NULL sample reads as not-boolean, which just falls back to
    the CAST (i.e. the previous behaviour). Never raises.
    """
    if not cols:
        return frozenset()
    try:
        import numpy as np
        import pandas as pd
        q = f"SELECT * FROM (\n{nestable_sql(base_sql)}\n) AS _dp_probe LIMIT 1"
        df = db.query(q)
        found: set[str] = set()
        for c in cols:
            if c not in df.columns:
                continue
            dt = df[c].dtype
            if dt.kind == "b" or isinstance(dt, pd.BooleanDtype):
                found.add(c)
            elif len(df) and isinstance(df[c].iloc[0], (bool, np.bool_)):
                found.add(c)
        return frozenset(found)
    except Exception as exc:  # noqa: BLE001 — probe is advisory
        logger.info("pushdown: boolean probe unavailable (%s)", type(exc).__name__)
        return frozenset()


def compute_sufficient_stats(
    db: DBConnection,
    base_sql: str,
    metric: str,
    covariate: str,
    segment_cols: list[str],
    guardrail_metrics: list[str],
    variant_col: str = "variant",
    week_col: str = "week",
    total_rows: int = 0,
    bool_cols: frozenset[str] | None = None,
    entity_col: str = "",
) -> SufficientStats:
    """Three aggregate queries over the approved SQL → SufficientStats
    (a fourth, COUNT DISTINCT over the entity/week columns, when the extract
    has an entity column — it feeds the JOIN fan-out check)."""
    backend = db.backend
    sub = f"(\n{nestable_sql(base_sql)}\n) AS _dp_push"
    if bool_cols is None:
        bool_cols = probe_bool_columns(
            db, base_sql, [metric, covariate, *guardrail_metrics]
        )
    v = quote_ident(variant_col, backend)
    yq = quote_ident(metric, backend)
    xq = quote_ident(covariate, backend)
    y = _f(metric, backend, bool_cols)
    x = _f(covariate, backend, bool_cols)

    # Query 1 — per-variant: metric moments (incl. Σy³ for skewness), joint
    # metric+covariate moments for CUPED, and guardrail moments.
    joint = f"{yq} IS NOT NULL AND {xq} IS NOT NULL"
    parts = [
        _moment_selects(metric, backend, "m", bool_cols),
        f"SUM(CASE WHEN {yq} IS NOT NULL AND {y} > 1.0 THEN 1 ELSE 0 END) AS m_gt1",
        f"SUM(CASE WHEN {joint} THEN 1 ELSE 0 END) AS j_n",
        f"SUM(CASE WHEN {joint} THEN {y} ELSE 0 END) AS j_sy",
        f"SUM(CASE WHEN {joint} THEN {y}*{y} ELSE 0 END) AS j_syy",
        f"SUM(CASE WHEN {joint} THEN {x} ELSE 0 END) AS j_sx",
        f"SUM(CASE WHEN {joint} THEN {x}*{x} ELSE 0 END) AS j_sxx",
        f"SUM(CASE WHEN {joint} THEN {x}*{y} ELSE 0 END) AS j_sxy",
    ]
    for i, gm in enumerate(guardrail_metrics):
        parts.append(_moment_selects(gm, backend, f"g{i}", bool_cols))
    q1 = f"SELECT {v} AS _variant,\n" + ",\n".join(parts) + f"\nFROM {sub} GROUP BY {v}"
    df1 = db.query(q1)

    overall: dict[str, GroupMoments] = {}
    guardrails: dict[str, dict[str, GroupMoments]] = {gm: {} for gm in guardrail_metrics}
    metric_over_one = 0
    for _, row in df1.iterrows():
        variant = str(row["_variant"])
        metric_over_one += int(row["m_gt1"] or 0)
        overall[variant] = GroupMoments(
            n=int(row["m_n"] or 0), s1=float(row["m_s1"] or 0),
            s2=float(row["m_s2"] or 0), s3=float(row["m_s3"] or 0),
            n_j=int(row["j_n"] or 0), sy_j=float(row["j_sy"] or 0),
            syy_j=float(row["j_syy"] or 0), sx_j=float(row["j_sx"] or 0),
            sxx_j=float(row["j_sxx"] or 0), sxy_j=float(row["j_sxy"] or 0),
        )
        for i, gm in enumerate(guardrail_metrics):
            guardrails[gm][variant] = GroupMoments(
                n=int(row[f"g{i}_n"] or 0), s1=float(row[f"g{i}_s1"] or 0),
                s2=float(row[f"g{i}_s2"] or 0), s3=float(row[f"g{i}_s3"] or 0),
            )

    # Query 4 (cheap, only with an entity column) — distinct entities and
    # weeks, so the JOIN fan-out check can run without the frame.
    n_entities = 0
    n_weeks = 0
    if entity_col:
        ent = quote_ident(entity_col, backend)
        wk = (f"COUNT(DISTINCT {quote_ident(week_col, backend)})" if week_col else "0")
        try:
            df4 = db.query(f"SELECT COUNT(DISTINCT {ent}) AS _e, {wk} AS _w FROM {sub}")
            n_entities = int(df4.iloc[0]["_e"] or 0)
            n_weeks = int(df4.iloc[0]["_w"] or 0)
        except Exception as exc:  # noqa: BLE001 — advisory; the check just won't fire
            logger.info("pushdown: entity count unavailable (%s)", type(exc).__name__)
            entity_col = ""

    # Query 2 — per (all segment cols jointly, variant): metric moments.
    # Mirrors run_hte's cross-product subgroups and its dropna over
    # [metric] + segment_cols.
    by_segment: dict[str, dict[str, GroupMoments]] = {}
    segment_total = 0
    if segment_cols:
        seg_idents = [quote_ident(c, backend) for c in segment_cols]
        not_null = " AND ".join(f"{s} IS NOT NULL" for s in seg_idents + [yq])
        q2 = (
            f"SELECT {', '.join(seg_idents)}, {v} AS _variant,\n"
            + _moment_selects(metric, backend, "m", bool_cols)
            + f"\nFROM {sub} WHERE {not_null}"
            + f"\nGROUP BY {', '.join(seg_idents)}, {v}"
        )
        df2 = db.query(q2)
        for _, row in df2.iterrows():
            label = ",".join(f"{c}={row[c]}" for c in segment_cols)
            variant = str(row["_variant"])
            g = GroupMoments(
                n=int(row["m_n"] or 0), s1=float(row["m_s1"] or 0),
                s2=float(row["m_s2"] or 0), s3=float(row["m_s3"] or 0),
            )
            by_segment.setdefault(label, {})[variant] = g
            segment_total += g.n

    # Query 3 — per (week, variant): metric moments for novelty. Skipped when
    # the extract has no week column (novelty node emits a typed skip result).
    by_week: dict[str, dict[str, GroupMoments]] = {}
    if not week_col:
        return SufficientStats(
            metric=metric, covariate=covariate, total_rows=total_rows,
            overall=overall, by_segment=by_segment, by_week=by_week,
            guardrails=guardrails, segment_total=segment_total,
            entity_col=entity_col, n_entities=n_entities, n_weeks=n_weeks,
            metric_over_one=metric_over_one,
        )
    w = quote_ident(week_col, backend)
    q3 = (
        f"SELECT {w} AS _week, {v} AS _variant,\n"
        + _moment_selects(metric, backend, "m", bool_cols)
        + f"\nFROM {sub} WHERE {w} IS NOT NULL GROUP BY {w}, {v}"
    )
    try:
        df3 = db.query(q3)
        for _, row in df3.iterrows():
            by_week.setdefault(str(row["_week"]), {})[str(row["_variant"])] = GroupMoments(
                n=int(row["m_n"] or 0), s1=float(row["m_s1"] or 0),
                s2=float(row["m_s2"] or 0), s3=float(row["m_s3"] or 0),
            )
    except Exception as exc:  # noqa: BLE001 — week column may not exist; novelty skips
        logger.info("pushdown: week aggregation unavailable (%s)", type(exc).__name__)

    return SufficientStats(
        metric=metric, covariate=covariate, total_rows=total_rows,
        overall=overall, by_segment=by_segment, by_week=by_week,
        guardrails=guardrails, segment_total=segment_total,
        entity_col=entity_col, n_entities=n_entities, n_weeks=n_weeks,
        metric_over_one=metric_over_one,
    )


def preview_frame(ss: SufficientStats) -> Any:
    """Small per-variant frame standing in for query_result.

    Carries the metric/covariate/variant columns the downstream plumbing
    checks for (semantic-cache eligibility, gate payloads), holding per-variant
    means rather than user rows.
    """
    import pandas as pd
    rows = []
    for variant, g in sorted(ss.overall.items()):
        rows.append({
            "variant": variant,
            ss.metric: round(_mean(g), 6),
            ss.covariate: round(g.sx_j / g.n_j, 6) if g.n_j else 0.0,
            "n_users": g.n,
        })
    return pd.DataFrame(rows)


# ── Stats from moments (each mirrors its pandas counterpart exactly) ──────────

def _arms(groups: dict[str, GroupMoments]) -> tuple[GroupMoments, GroupMoments]:
    ctrl, trt = groups.get("control"), groups.get("treatment")
    if ctrl is None or trt is None:
        raise ValueError(
            f"variant_col must contain 'control' and 'treatment'. Found: {set(groups)}"
        )
    return ctrl, trt


def ttest_from_stats(
    ss: SufficientStats,
    alpha: float = 0.05,
    winsorize_pct: float = 0.0,
    alternative: str = "two-sided",
) -> TtestResult:
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    ctrl, trt = _arms(ss.overall)
    if ctrl.n < 2 or trt.n < 2:
        raise ValueError("Need at least 2 observations per group for t-test.")

    ctrl_skew, trt_skew = _skew(ctrl), _skew(trt)
    skewness_warning = None
    if abs(ctrl_skew) > 2.0 or abs(trt_skew) > 2.0:
        skewness_warning = (
            f"Highly skewed distribution (control skew={ctrl_skew:+.1f}, "
            f"treatment skew={trt_skew:+.1f}). "
            "Consider winsorizing or using the Mann-Whitney U test."
        )
    if winsorize_pct > 0.0:
        # Quantile clipping is not derivable from moments; be loud, not wrong.
        note = "Winsorization skipped: not available in warehouse-pushdown mode."
        skewness_warning = f"{skewness_warning} {note}" if skewness_warning else note

    t_stat, p, ci_lo, ci_hi, d = _welch(
        trt.n, _mean(trt), _var(trt), ctrl.n, _mean(ctrl), _var(ctrl),
        alpha, alternative,
    )
    return TtestResult(
        t_stat=round(t_stat, 4),
        p_value=round(p, 6),
        ci_lower=round(ci_lo, 6),
        ci_upper=round(ci_hi, 6),
        significant=bool(p < alpha),
        cohens_d=round(d, 4),
        n_control=ctrl.n,
        n_treatment=trt.n,
        control_mean=round(_mean(ctrl), 6),
        treatment_mean=round(_mean(trt), 6),
        winsorized=False,
        skewness_warning=skewness_warning,
        alternative=alternative,
    )


def cuped_from_stats(ss: SufficientStats) -> CupedResult:
    """CUPED from joint moments; mirrors stats_tools.run_cuped (pooled theta).

    run_cuped pools EVERY row that survives dropna — a third arm (holdout)
    contributes to θ, mean(X) and the before/after variances, while the ATE
    itself is control vs treatment. Same here: pooled sums run over all
    variants in `overall`, group means over the two arms.
    """
    ctrl, trt = _arms(ss.overall)
    if ctrl.n_j < 2 or trt.n_j < 2:
        raise ValueError("Need at least 2 observations per variant for CUPED.")

    groups = list(ss.overall.values())
    n = sum(g.n_j for g in groups)
    sy = sum(g.sy_j for g in groups)
    syy = sum(g.syy_j for g in groups)
    sx = sum(g.sx_j for g in groups)
    sxx = sum(g.sxx_j for g in groups)
    sxy = sum(g.sxy_j for g in groups)

    var_x = (sxx - sx * sx / n) / (n - 1)
    # One-pass variance of a constant covariate is rounding noise, not zero
    # (pandas' two-pass var() is exactly 0 and run_cuped raises). Relative
    # tolerance against the mean square: below 1e-9 the covariate is constant
    # for any purpose CUPED has, and θ = noise/noise would be reported.
    if var_x <= 1e-9 * (sxx / n if n else 0.0):
        raise ValueError(f"Covariate '{ss.covariate}' has zero variance — cannot apply CUPED.")
    cov_xy = (sxy - sx * sy / n) / (n - 1)
    theta = cov_xy / var_x

    mean_x = sx / n
    mean_y_c, mean_y_t = ctrl.sy_j / ctrl.n_j, trt.sy_j / trt.n_j
    mean_x_c, mean_x_t = ctrl.sx_j / ctrl.n_j, trt.sx_j / trt.n_j

    raw_ate = mean_y_t - mean_y_c
    # Y_adj = Y - θ(X - mean_x); group mean of Y_adj = mean_y_g - θ(mean_x_g - mean_x)
    cuped_ate = (mean_y_t - theta * (mean_x_t - mean_x)) - (mean_y_c - theta * (mean_x_c - mean_x))

    # Pooled variance before vs after adjustment:
    # var(Y - θX) = var(Y) + θ²var(X) - 2θcov(X,Y)   (constants drop out)
    var_before = (syy - sy * sy / n) / (n - 1)
    var_after = var_before + theta * theta * var_x - 2.0 * theta * cov_xy
    reduction = (1 - var_after / var_before) * 100 if var_before > 0 else 0.0

    return CupedResult(
        raw_ate=round(raw_ate, 6),
        cuped_ate=round(cuped_ate, 6),
        variance_reduction_pct=round(reduction, 2),
        theta=round(theta, 6),
    )


def hte_from_stats(
    ss: SufficientStats,
    alpha: float = 0.05,
    min_segment_size: int = 30,
) -> HteResult:
    """Per-subgroup Welch tests; mirrors stats_tools.run_hte (Bonferroni,
    significant-first sort). interaction_p_value is None — the OLS F-test
    needs row-level residuals the moments don't carry."""
    total = ss.segment_total
    results: list[SegmentResult] = []
    for label, groups in ss.by_segment.items():
        ctrl, trt = groups.get("control"), groups.get("treatment")
        if ctrl is None or trt is None:
            continue
        if ctrl.n < min_segment_size or trt.n < min_segment_size:
            continue
        try:
            t_stat, p, *_ = _welch(trt.n, _mean(trt), _var(trt),
                                   ctrl.n, _mean(ctrl), _var(ctrl),
                                   alpha, "two-sided")
        except ValueError:
            continue
        ate = _mean(trt) - _mean(ctrl)
        results.append(SegmentResult(
            segment=label,
            effect_size=round(ate, 6),
            segment_share=round((ctrl.n + trt.n) / total, 4) if total else 0.0,
            control_mean=round(_mean(ctrl), 6),
            treatment_mean=round(_mean(trt), 6),
            p_value=round(p, 6),
            significant=bool(p < alpha),
            n_control=ctrl.n,
            n_treatment=trt.n,
        ))

    if not results:
        raise ValueError(
            f"No subgroups had >= {min_segment_size} observations per variant. "
            "Try reducing min_segment_size or adding more data."
        )

    n_tests = len(results)
    if n_tests > 1:
        bonferroni_alpha = alpha / n_tests
        for r in results:
            r.significant = r.p_value < bonferroni_alpha

    results.sort(key=lambda r: (not r.significant, -abs(r.effect_size)))
    top = results[0]
    return HteResult(
        top_segment=top.segment,
        effect_size=top.effect_size,
        segment_share=top.segment_share,
        all_segments=results,
        interaction_p_value=None,
    )


def _week_to_int(val: Any) -> int | None:
    """Mirror of novelty_tools' week normalization."""
    try:
        return int(float(val))
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    for prefix in ("week_", "week", "w"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def novelty_from_stats(ss: SufficientStats) -> NoveltyResult:
    """Week-1 vs week-2 ATE; mirrors novelty_tools.detect_novelty_effect."""
    weekly: dict[int, dict[str, GroupMoments]] = {}
    for raw_week, groups in ss.by_week.items():
        wk = _week_to_int(raw_week)
        if wk is None:
            raise ValueError(
                f"week_col contains values that cannot be parsed as week numbers: "
                f"{sorted(ss.by_week)}"
            )
        merged = weekly.setdefault(wk, {})
        for variant, g in groups.items():
            if variant in merged:
                m = merged[variant]
                merged[variant] = GroupMoments(
                    n=m.n + g.n, s1=m.s1 + g.s1, s2=m.s2 + g.s2, s3=m.s3 + g.s3,
                )
            else:
                merged[variant] = g

    if not {1, 2}.issubset(weekly):
        raise ValueError(f"week_col must contain weeks 1 and 2. Found: {set(weekly)}")

    def ate_for_week(week: int) -> float:
        groups = weekly[week]
        ctrl, trt = groups.get("control"), groups.get("treatment")
        if ctrl is None or trt is None or ctrl.n < 2 or trt.n < 2:
            raise ValueError(f"Not enough data in week {week} to compute ATE.")
        return _mean(trt) - _mean(ctrl)

    week1_ate = ate_for_week(1)
    week2_ate = ate_for_week(2)
    abs1, abs2 = abs(week1_ate), abs(week2_ate)

    if abs1 == 0 and abs2 == 0:
        effect_direction = "stable"
    elif abs1 == 0:
        effect_direction = "growing" if week2_ate > 0 else "decaying"
    elif abs2 > abs1 * 1.10:
        effect_direction = "growing"
    elif abs2 < abs1 * 0.90:
        effect_direction = "decaying"
    elif (week1_ate > 0) != (week2_ate > 0) and abs2 > 0:
        effect_direction = "decaying"
    else:
        effect_direction = "stable"

    sign_reversed = abs1 > 0 and abs2 > 0 and (week1_ate > 0) != (week2_ate > 0)
    novelty_likely = (
        effect_direction == "decaying"
        and abs1 > 0
        and (abs2 < 0.5 * abs1 or sign_reversed)
    )
    return NoveltyResult(
        week1_ate=round(week1_ate, 6),
        week2_ate=round(week2_ate, 6),
        effect_direction=effect_direction,
        novelty_likely=novelty_likely,
    )


def _quote_literal(value: str, backend: str) -> str:
    """SQL string literal with standard '' escaping. MySQL additionally treats
    backslash as an escape character by default, so double those there — a
    step value ending in a backslash would otherwise swallow the closing
    quote."""
    s = str(value)
    if backend == "mysql":
        s = s.replace("\\", "\\\\")
    return "'" + s.replace("'", "''") + "'"


def funnel_from_warehouse(
    db: DBConnection,
    funnel_sql: str,
    candidate_steps: list[str],
    alpha: float = 0.05,
) -> FunnelResult | None:
    """Conditional funnel computed in-warehouse; mirrors
    funnel_tools.compute_funnel exactly.

    The pandas path pivots to one row per user (MAX(completed) per step) and
    makes step k's eligible population the users who completed step k−1. The
    same pivot is one conditional-aggregation CTE; a second aggregation
    reduces it to eligible/completed counts per (step, variant) — a handful
    of numbers regardless of funnel size. The two-proportion z-test needs
    nothing else.

    Returns None when fewer than two candidate steps exist in the data
    (mirrors compute_funnel_node's skip).
    """
    backend = db.backend
    sub = f"(\n{nestable_sql(funnel_sql)}\n) AS _dp_funnel"

    step_col = quote_ident("step", backend)
    present = {
        str(r[0]) for r in
        db.query(f"SELECT {step_col} AS _s FROM {sub} GROUP BY {step_col}").itertuples(index=False)
    }
    steps = [s for s in candidate_steps if s in present]
    if len(steps) < 2:
        return None

    v = quote_ident("variant", backend)
    uid = quote_ident("user_id", backend)
    comp = quote_ident("completed", backend)
    # A BOOLEAN completed flag cannot sit in the same CASE as the integer 0
    # on Postgres/BigQuery; the pandas pivot (aggfunc="max") accepted it.
    if "completed" in probe_bool_columns(db, funnel_sql, ["completed"]):
        comp = f"(CASE WHEN {comp} THEN 1 ELSE 0 END)"

    pivot_cols = ",\n".join(
        f"MAX(CASE WHEN {step_col} = {_quote_literal(s, backend)} "
        f"THEN {comp} ELSE 0 END) AS s{i}"
        for i, s in enumerate(steps)
    )
    count_cols = []
    for i in range(len(steps)):
        if i == 0:
            count_cols.append(f"COUNT(*) AS e0, SUM(s0) AS c0")
        else:
            count_cols.append(
                f"SUM(CASE WHEN s{i-1} = 1 THEN 1 ELSE 0 END) AS e{i}, "
                f"SUM(CASE WHEN s{i-1} = 1 THEN s{i} ELSE 0 END) AS c{i}"
            )
    q = (
        f"WITH _dp_pivot AS (\n"
        f"SELECT {uid}, {v} AS _variant,\n{pivot_cols}\n"
        f"FROM {sub} GROUP BY {uid}, {v}\n"
        f")\nSELECT _variant, {', '.join(count_cols)}\n"
        f"FROM _dp_pivot GROUP BY _variant"
    )
    counts = {str(row["_variant"]): row for _, row in db.query(q).iterrows()}
    if "control" not in counts or "treatment" not in counts:
        raise ValueError(
            f"variant_col must contain 'control' and 'treatment'. Found: {set(counts)}"
        )

    step_results: list[FunnelStep] = []
    largest_delta = 0.0
    biggest_dropoff_step = steps[0]
    for i, step in enumerate(steps):
        n_ctrl = int(counts["control"][f"e{i}"] or 0)
        n_trt  = int(counts["treatment"][f"e{i}"] or 0)
        if n_ctrl < 2 or n_trt < 2:
            raise ValueError(
                f"Not enough eligible users at step '{step}' "
                f"(ctrl={n_ctrl}, trt={n_trt})."
            )
        c_ctrl = float(counts["control"][f"c{i}"] or 0)
        c_trt  = float(counts["treatment"][f"c{i}"] or 0)
        ctrl_rate = c_ctrl / n_ctrl
        trt_rate  = c_trt / n_trt
        delta = trt_rate - ctrl_rate
        pct_change = (delta / ctrl_rate * 100) if ctrl_rate != 0 else float("inf")

        p_pool = (c_ctrl + c_trt) / (n_ctrl + n_trt)
        se = (p_pool * (1 - p_pool) * (1 / n_ctrl + 1 / n_trt)) ** 0.5
        if se == 0:
            p_value = 1.0
        else:
            z_stat = delta / se
            p_value = float(2 * (1 - sps.norm.cdf(abs(z_stat))))

        step_results.append(FunnelStep(
            step=step,
            control_rate=round(ctrl_rate, 4),
            treatment_rate=round(trt_rate, 4),
            delta=round(delta, 4),
            pct_change=round(pct_change, 2),
            p_value=round(p_value, 6),
            significant=p_value < alpha,
        ))
        if abs(delta) > abs(largest_delta):
            largest_delta = delta
            biggest_dropoff_step = step

    return FunnelResult(steps=step_results, biggest_dropoff_step=biggest_dropoff_step)


def guardrails_from_stats(
    ss: SufficientStats,
    alpha: float = 0.05,
    harm_directions: dict[str, str] | None = None,
    default_direction: str = "both",
) -> GuardrailResult:
    """Mirrors guardrail_tools.check_guardrails (two-sided Welch, keyword
    harm-direction inference, Bonferroni downgrade)."""
    from tools.guardrail_tools import _infer_harm_direction

    guardrails: list[GuardrailMetric] = []
    for metric, groups in ss.guardrails.items():
        ctrl, trt = groups.get("control"), groups.get("treatment")
        if ctrl is None or trt is None or ctrl.n < 2 or trt.n < 2:
            continue
        ctrl_mean, trt_mean = _mean(ctrl), _mean(trt)
        if ctrl_mean != 0:
            delta_pct = (trt_mean - ctrl_mean) / abs(ctrl_mean) * 100
        else:
            delta_pct = (trt_mean - ctrl_mean) * 100
        try:
            _, p_value, *_ = _welch(trt.n, trt_mean, _var(trt),
                                    ctrl.n, ctrl_mean, _var(ctrl),
                                    alpha, "two-sided")
        except ValueError:
            continue

        if harm_directions and metric in harm_directions:
            direction = harm_directions[metric]
        else:
            inferred = _infer_harm_direction(metric)
            direction = inferred if inferred != "both" else default_direction

        significant = p_value < alpha
        if direction == "increase":
            breached = significant and (trt_mean > ctrl_mean)
        elif direction == "decrease":
            breached = significant and (trt_mean < ctrl_mean)
        else:
            breached = significant

        guardrails.append(GuardrailMetric(
            metric=metric,
            control_mean=round(ctrl_mean, 6),
            treatment_mean=round(trt_mean, 6),
            delta_pct=round(delta_pct, 2),
            p_value=round(p_value, 6),
            breached=breached,
        ))

    n_tests = len(guardrails)
    if n_tests > 1:
        corrected_alpha = alpha / n_tests
        for g in guardrails:
            if g.breached and g.p_value >= corrected_alpha:
                g.breached = False

    breached_count = sum(1 for g in guardrails if g.breached)
    return GuardrailResult(
        guardrails=guardrails,
        any_breached=breached_count > 0,
        breached_count=breached_count,
    )
