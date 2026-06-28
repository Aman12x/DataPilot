"""Analyze graph nodes — analysis."""
from __future__ import annotations

import agents.analyze.node_shared as _shared
globals().update({k: v for k, v in vars(_shared).items() if not k.startswith("__")})

# ── Node 6b: load_auxiliary_data ─────────────────────────────────────────────
# Loads timeseries and funnel tables using MetricConfig — no LLM involved.
# decompose_metric / detect_anomaly / forecast_baseline need daily_df.
# compute_funnel needs funnel_df.
# These are separate from query_result (the user-level experiment DataFrame).

def _aggregate_daily_from_events(conn, mc: MetricConfig) -> pd.DataFrame:
    """Fallback: aggregate daily metric from raw events when no timeseries_table."""
    agg_map = {
        "mean":  f"AVG({mc.metric_source_col})",
        "sum":   f"SUM({mc.metric_source_col})",
        "count": f"COUNT(*)",
    }
    agg_expr = agg_map.get(mc.metric_agg, agg_map["mean"])
    segment_cols_sql = ", ".join(mc.segment_cols) if mc.segment_cols else ""
    group_by_cols    = f"{mc.date_col}" + (f", {segment_cols_sql}" if segment_cols_sql else "")
    select_extra     = (f", {segment_cols_sql}" if segment_cols_sql else "")
    return conn.query(f"""
        SELECT {mc.date_col} AS date{select_extra},
               {agg_expr} AS {mc.primary_metric}
        FROM {mc.events_table}
        GROUP BY {group_by_cols}
        ORDER BY {mc.date_col}
    """)


@observe(name="load_auxiliary_data")
def load_auxiliary_data(state: AgentState) -> dict:
    mc   = state.get("metric_config") or load_metric_config()
    conn = _db_conn(state)
    result: dict = {}

    # ── Timeseries: try pre-aggregated table first, fall back to event aggregation ──
    if mc.timeseries_table:
        try:
            daily = conn.query(f"SELECT * FROM {mc.timeseries_table} ORDER BY {mc.date_col}")
            # Aggregate DAU component columns to platform level for cleaner time series
            agg_cols = [c for c in ["dau", "new_users", "retained_users", "resurrected_users", "churned_users"]
                        if c in daily.columns]
            seg_cols_present = [c for c in mc.segment_cols if c in daily.columns]
            if agg_cols and seg_cols_present and mc.date_col in daily.columns:
                daily = (
                    daily
                    .groupby([mc.date_col] + seg_cols_present)[agg_cols]
                    .sum()
                    .reset_index()
                )
            result["daily_df"] = daily
        except Exception as exc:
            logger.warning(
                "load_auxiliary_data: %s query failed (%s) — falling back to event aggregation.",
                mc.timeseries_table, exc,
            )
            try:
                result["daily_df"] = _aggregate_daily_from_events(conn, mc)
            except Exception as exc2:
                logger.warning("load_auxiliary_data: event aggregation also failed: %s", exc2)
    else:
        try:
            result["daily_df"] = _aggregate_daily_from_events(conn, mc)
        except Exception as exc:
            logger.warning("load_auxiliary_data: event aggregation failed: %s", exc)

    # ── Funnel: optional ──────────────────────────────────────────────────────
    if mc.funnel_table:
        funnel_sql = f"""\
SELECT f.{mc.user_id_col}, ex.{mc.variant_col} AS variant, f.step, f.completed
FROM   {mc.funnel_table} f
JOIN   {mc.experiment_table} ex
       ON f.{mc.user_id_col} = ex.{mc.user_id_col}
      AND ex.{mc.week_col} = 1
"""
        try:
            result["funnel_df"] = conn.query(funnel_sql)
        except Exception as exc:
            logger.warning("load_auxiliary_data: funnel query failed: %s", exc)

    return result


# ── Node 7: decompose_metric ──────────────────────────────────────────────────

@observe(name="decompose_metric")
def decompose_metric(state: AgentState) -> dict:
    df = state.get("daily_df")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.warning("decompose_metric: no daily_df in state, skipping.")
        return {}

    mc = state.get("metric_config") or load_metric_config()

    # Use DAU-specific decomposition only when all DAU component columns are present
    dau_cols = {"new_users", "retained_users", "resurrected_users", "churned_users"}
    if dau_cols.issubset(df.columns):
        try:
            result = decomposition_tools.decompose_dau(df, date_col=mc.date_col)
            return {"decomposition_result": result}
        except Exception as exc:
            logger.warning("decompose_metric: decompose_dau failed (%s), trying generic.", exc)

    # Generic path: segment-based breakdown for any metric
    metric_col = mc.primary_metric
    segment_cols = [c for c in mc.segment_cols if c in df.columns]
    if metric_col not in df.columns or not segment_cols:
        logger.warning(
            "decompose_metric: metric_col '%s' or segment_cols %s not in daily_df, skipping.",
            metric_col, mc.segment_cols,
        )
        return {}

    try:
        result = decomposition_tools.decompose_metric(
            df, metric_col=metric_col, segment_cols=segment_cols, date_col=mc.date_col
        )
        return {"decomposition_result": result}
    except Exception as exc:
        logger.warning("decompose_metric: decompose_metric failed: %s", exc)
        return {}


# ── Node 8: detect_anomaly_node ───────────────────────────────────────────────

@observe(name="detect_anomaly")
def detect_anomaly_node(state: AgentState) -> dict:
    df     = state.get("daily_df")
    mc     = state.get("metric_config") or load_metric_config()
    date_col = mc.date_col

    # Resolve which metric column to use from the timeseries.
    # Prefer the requested/primary metric; fall back to "dau" for the built-in demo schema
    # (where primary_metric is "dau_rate" per-user but the timeseries has "dau" aggregate).
    cols = set(df.columns) if df is not None and hasattr(df, "columns") else set()
    requested = state.get("metric") or mc.primary_metric
    if requested in cols:
        metric = requested
    elif mc.primary_metric in cols:
        metric = mc.primary_metric
    elif "dau" in cols:
        metric = "dau"
        logger.info("detect_anomaly: using 'dau' as proxy for '%s' in timeseries.", requested)
    else:
        metric = requested  # caught by guard below

    if df is None or (isinstance(df, pd.DataFrame) and df.empty) or date_col not in df.columns or metric not in df.columns:
        logger.warning("detect_anomaly: required columns missing in daily_df, skipping.")
        return {}

    anomaly = anomaly_tools.detect_anomaly(df, metric_col=metric, date_col=date_col)

    dimension_cols = [c for c in mc.segment_cols if c in df.columns]
    if dimension_cols:
        # Use the first anomaly date as the before/after split so slice_and_dice
        # compares pre-experiment baseline against the experiment window rather
        # than splitting at the calendar midpoint.
        anomaly_dates = anomaly.anomaly_dates if anomaly else []
        experiment_start = anomaly_dates[0] if anomaly_dates else None
        slices = anomaly_tools.slice_and_dice(
            df, metric_col=metric, date_col=date_col,
            dimension_cols=dimension_cols, experiment_start=experiment_start,
        )
    else:
        slices = SliceResult(ranked_dimensions=[])

    return {
        "anomaly_result": anomaly,
        "slice_result":   slices,
    }


# ── Node 9: forecast_baseline_node ────────────────────────────────────────────

@observe(name="forecast_baseline")
def forecast_baseline_node(state: AgentState) -> dict:
    df     = state.get("daily_df")
    mc     = state.get("metric_config") or load_metric_config()
    date_col = mc.date_col

    # Resolve metric column using the same logic as detect_anomaly_node.
    cols = set(df.columns) if df is not None and hasattr(df, "columns") else set()
    requested = state.get("metric") or mc.primary_metric
    if requested in cols:
        metric = requested
    elif mc.primary_metric in cols:
        metric = mc.primary_metric
    elif "dau" in cols:
        metric = "dau"
        logger.info("forecast_baseline: using 'dau' as proxy for '%s' in timeseries.", requested)
    else:
        metric = requested

    if df is None or (isinstance(df, pd.DataFrame) and df.empty) or date_col not in df.columns or metric not in df.columns:
        logger.warning("forecast_baseline: required columns missing in daily_df, skipping.")
        return {}

    try:
        result = forecast_tools.forecast_baseline(df, metric_col=metric, date_col=date_col)
    except Exception as exc:
        logger.warning("forecast_baseline: failed (%s), skipping.", exc)
        return {}
    return {"forecast_result": result}


# ── Node 10: run_cuped_node ───────────────────────────────────────────────────

@observe(name="run_cuped")
def run_cuped_node(state: AgentState) -> dict:
    df        = _safe_df(state)
    mc        = state.get("metric_config") or load_metric_config()
    metric    = state.get("metric") or mc.primary_metric
    covariate = state.get("covariate") or mc.covariate
    variant   = "variant"

    if df is None:
        return {}

    if covariate == metric:
        logger.warning(
            "run_cuped: covariate '%s' is the same as metric '%s' — "
            "CUPED requires a pre-experiment covariate, not the metric itself. Skipping.",
            covariate, metric,
        )
        return {}

    for col in [metric, covariate, variant]:
        if col not in df.columns:
            logger.warning("run_cuped: column '%s' missing, skipping.", col)
            return {}

    try:
        result = stats_tools.run_cuped(
            df, metric_col=metric, covariate_col=covariate, variant_col=variant
        )
    except ValueError as exc:
        logger.warning("run_cuped: skipping — %s", exc)
        return {}
    return {"cuped_result": result}


# ── Node 11: run_ttest_node ───────────────────────────────────────────────────

@observe(name="run_ttest")
def run_ttest_node(state: AgentState) -> dict:
    df        = _safe_df(state)
    mc        = state.get("metric_config") or load_metric_config()
    metric    = state.get("metric") or mc.primary_metric
    variant   = "variant"

    if df is None:
        return {}

    # CUPED adjusts the ATE internally; run_ttest always operates on the original metric col
    use_col = metric

    if variant not in df.columns or use_col not in df.columns:
        return {}

    ctrl = df[df[variant] == "control"][use_col].dropna()
    trt  = df[df[variant] == "treatment"][use_col].dropna()

    # Auto-winsorize metrics whose names suggest heavy-tailed / revenue-style distributions.
    _SKEWED_KEYWORDS = {
        "revenue", "spend", "amount", "price", "cost", "purchase",
        "ltv", "count", "cnt", "sum", "total", "gmv", "orders",
    }
    words = re.split(r"[_\-./]+", use_col.lower())
    winsorize_pct = 0.01 if any(w in _SKEWED_KEYWORDS for w in words) else 0.0

    # Infer one-sided test direction from MetricConfig when direction is known.
    # higher_is_better → expect treatment > control → 'greater'
    # lower_is_better  → expect treatment < control → 'less'
    # Unknown / not set → safe default 'two-sided'
    direction = getattr(mc, "metric_direction", None) or ""
    if direction == "higher_is_better":
        alternative = "greater"
    elif direction == "lower_is_better":
        alternative = "less"
    else:
        alternative = "two-sided"

    result = stats_tools.run_ttest(ctrl, trt, winsorize_pct=winsorize_pct,
                                   alternative=alternative)
    return {"ttest_result": result}


# ── Node 11b: check_srm_node ──────────────────────────────────────────────────

@observe(name="check_srm")
def check_srm_node(state: AgentState) -> dict:
    """
    Sample Ratio Mismatch check.

    Runs a chi-squared goodness-of-fit test on the control/treatment split.
    An SRM means the randomization mechanism is broken and all downstream
    stats (t-test, CUPED, HTE) are potentially invalid.

    Uses ttest_result.n_control / n_treatment to avoid re-counting the DataFrame.
    Falls back to counting the DataFrame directly if ttest_result is absent.
    """
    ttest = state.get("ttest_result")
    if ttest is not None:
        n_ctrl = ttest.n_control
        n_trt  = ttest.n_treatment
    else:
        df = _safe_df(state)
        mc = state.get("metric_config") or load_metric_config()
        metric = state.get("metric") or mc.primary_metric
        if df is None or "variant" not in df.columns or metric not in df.columns:
            return {}
        grp = df.groupby("variant")[metric].count()
        n_ctrl = int(grp.get("control",  0))
        n_trt  = int(grp.get("treatment", 0))

    if n_ctrl < 1 or n_trt < 1:
        return {}

    try:
        result = stats_tools.check_srm(n_ctrl, n_trt)
    except ValueError as exc:
        logger.warning("check_srm: skipping — %s", exc)
        return {}

    if result.srm_detected:
        logger.warning(
            "SRM detected: n_control=%d, n_treatment=%d (ratio=%.3f, expected=0.500, p=%.6f)",
            n_ctrl, n_trt, result.observed_ratio, result.p_value,
        )

    return {"srm_result": result}


# ── Node 12: run_hte_node ─────────────────────────────────────────────────────

@observe(name="run_hte")
def run_hte_node(state: AgentState) -> dict:
    df      = _safe_df(state)
    mc      = state.get("metric_config") or load_metric_config()
    metric  = state.get("metric") or mc.primary_metric
    variant = "variant"

    if df is None:
        return {}

    segment_cols = [c for c in mc.segment_cols if c in df.columns]
    if not segment_cols or metric not in df.columns or variant not in df.columns:
        return {}

    result = stats_tools.run_hte(
        df, metric_col=metric, variant_col=variant, segment_cols=segment_cols
    )
    return {"hte_result": result}


# ── Node 13: detect_novelty_node ──────────────────────────────────────────────

@observe(name="detect_novelty")
def detect_novelty_node(state: AgentState) -> dict:
    df     = _safe_df(state)
    mc     = state.get("metric_config") or load_metric_config()
    metric = state.get("metric") or mc.primary_metric

    from tools.schemas import NoveltyResult as _NoveltyResult

    if df is None:
        return {}

    # Return a typed skip result so the narrative knows why this section is absent.
    missing = [col for col in [metric, "variant", "week"] if col not in df.columns]
    if missing:
        reason = f"Required column(s) missing: {', '.join(missing)}"
        logger.warning("detect_novelty: %s — skipping.", reason)
        return {"novelty_result": _NoveltyResult(
            week1_ate=0.0, week2_ate=0.0, effect_direction="unknown",
            novelty_likely=False, skipped=True, skip_reason=reason,
        )}

    try:
        result = novelty_tools.detect_novelty_effect(
            df, metric_col=metric, variant_col="variant", week_col="week"
        )
    except ValueError as exc:
        logger.warning("detect_novelty: skipping — %s", exc)
        return {"novelty_result": _NoveltyResult(
            week1_ate=0.0, week2_ate=0.0, effect_direction="unknown",
            novelty_likely=False, skipped=True, skip_reason=str(exc),
        )}
    return {"novelty_result": result}


# ── Node 14: compute_mde_node ─────────────────────────────────────────────────

@observe(name="compute_mde")
def compute_mde_node(state: AgentState) -> dict:
    df     = _safe_df(state)
    mc     = state.get("metric_config") or load_metric_config()
    metric = state.get("metric") or mc.primary_metric

    if df is None or "variant" not in df.columns or metric not in df.columns:
        return {}

    ctrl = df[df["variant"] == "control"][metric].dropna()
    trt  = df[df["variant"] == "treatment"][metric].dropna()

    if len(ctrl) < 2 or len(trt) < 2:
        return {}

    cuped = state.get("cuped_result")
    cuped_ate = cuped.cuped_ate if cuped else None

    result = mde_tools.compute_mde(
        n_control=len(ctrl),
        n_treatment=len(trt),
        baseline_mean=float(ctrl.mean()),
        baseline_std=float(ctrl.std()),
        observed_effect_abs=cuped_ate,
    )

    # Use baseline_unit_count from MetricConfig (env BASELINE_DAU as a final fallback)
    baseline_units = mc.baseline_unit_count or int(os.getenv("BASELINE_DAU", "500000"))
    impact = mde_tools.business_impact_statement(
        mde_relative_pct=result.mde_relative_pct,
        metric=metric,
        baseline_dau=baseline_units,
        revenue_per_dau=mc.revenue_per_unit,
    )

    return {
        "mde_result":      result,
        "business_impact": impact,
    }


# ── Node 14b: run_power_analysis_node ─────────────────────────────────────────

_SENSITIVITY_MDE_LEVELS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

@observe(name="run_power_analysis")
def run_power_analysis_node(state: AgentState) -> dict:
    """
    Compute required sample size and sensitivity table for experiment design.

    Queries the DB for baseline metric stats (AVG, STDDEV, daily traffic),
    then applies the inverted MDE formula across a range of effect sizes.
    Returns a PowerAnalysisResult stored in state["power_analysis_result"].
    """
    mc          = state.get("metric_config") or load_metric_config()
    metric      = state.get("metric") or mc.primary_metric
    mde_target  = float(state.get("power_mde_target_pct") or 5.0)
    alpha       = 0.05
    power_level = 0.80

    try:
        conn = _db_conn(state)
        # Query baseline stats from the historical events table.
        # COALESCE(STDDEV(...), 0) avoids NULL when all values are identical.
        stats_sql = f"""
SELECT AVG(CAST({mc.metric_source_col} AS FLOAT))    AS baseline_mean,
       COALESCE(STDDEV(CAST({mc.metric_source_col} AS FLOAT)), 0) AS baseline_std,
       COUNT(DISTINCT {mc.user_id_col})               AS total_users,
       COUNT(DISTINCT {mc.date_col})                  AS total_days
FROM {mc.events_table}
""".strip()
        stats_df = conn.query(stats_sql)
    except Exception as exc:
        logger.warning("run_power_analysis: DB stats query failed: %s", exc)
        return {}

    if stats_df is None or stats_df.empty:
        logger.warning("run_power_analysis: stats query returned no rows")
        return {}

    row = stats_df.iloc[0]
    baseline_mean = float(row.get("baseline_mean") or 0.0)
    baseline_std  = float(row.get("baseline_std")  or 0.0)
    total_users   = int(row.get("total_users")     or 0)
    total_days    = int(row.get("total_days")      or 1)

    if baseline_mean == 0 or baseline_std == 0:
        logger.warning("run_power_analysis: degenerate baseline (mean=%.4f std=%.4f)", baseline_mean, baseline_std)
        return {}

    daily_traffic = total_users / max(total_days, 1)

    # Compute sensitivity rows at each MDE level
    sensitivity: list[SensitivityRow] = []
    for mde_pct in _SENSITIVITY_MDE_LEVELS:
        try:
            n_per_arm, _ = mde_tools.required_sample_size(
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                mde_relative_pct=mde_pct,
                alpha=alpha,
                power=power_level,
            )
        except ValueError:
            continue
        total_n      = 2 * n_per_arm
        runtime_days = max(1, int(math.ceil(total_n / max(daily_traffic, 1))))
        sensitivity.append(SensitivityRow(
            mde_pct=mde_pct,
            n_per_arm=n_per_arm,
            runtime_days=runtime_days,
        ))

    if not sensitivity:
        return {}

    # Primary target
    try:
        req_n, mde_abs = mde_tools.required_sample_size(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            mde_relative_pct=mde_target,
            alpha=alpha,
            power=power_level,
        )
    except ValueError:
        logger.warning("run_power_analysis: required_sample_size failed for mde_target=%.1f", mde_target)
        return {}

    total_n      = 2 * req_n
    runtime_days = max(1, int(math.ceil(total_n / max(daily_traffic, 1))))

    result = PowerAnalysisResult(
        baseline_mean=round(baseline_mean, 6),
        baseline_std=round(baseline_std, 6),
        daily_traffic=round(daily_traffic, 1),
        mde_target_pct=mde_target,
        mde_target_abs=mde_abs,
        required_n_per_arm=req_n,
        required_total_n=total_n,
        runtime_days=runtime_days,
        alpha=alpha,
        power=power_level,
        guardrails_to_watch=mc.guardrail_metrics or [],
        sensitivity=sensitivity,
    )

    return {"power_analysis_result": result}


# ── Node 15: check_guardrails_node ────────────────────────────────────────────

@observe(name="check_guardrails")
def check_guardrails_node(state: AgentState) -> dict:
    df = _safe_df(state)
    if df is None or "variant" not in df.columns:
        return {}

    mc = state.get("metric_config") or load_metric_config()
    present = [m for m in mc.guardrail_metrics if m in df.columns]
    if not present:
        return {}

    # When primary metric is higher_is_better, unknown guardrail drops are harmful
    default_direction = (
        "decrease" if mc.metric_direction == "higher_is_better" else "increase"
    )

    result = guardrail_tools.check_guardrails(
        df,
        variant_col="variant",
        guardrail_metrics=present,
        harm_directions=mc.guardrail_harm_directions,
        default_direction=default_direction,
    )
    return {"guardrail_result": result}


# ── Node 16: compute_funnel_node ──────────────────────────────────────────────

@observe(name="compute_funnel")
def compute_funnel_node(state: AgentState) -> dict:
    df = state.get("funnel_df")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.warning("compute_funnel: no funnel_df in state, skipping.")
        return {}

    required = {"user_id", "step", "completed", "variant"}
    if not required.issubset(df.columns):
        logger.warning("compute_funnel: funnel columns missing in funnel_df, skipping.")
        return {}

    mc            = state.get("metric_config") or load_metric_config()
    present_steps = set(df["step"].dropna().unique())
    steps = [s for s in mc.funnel_steps if s in present_steps]
    if len(steps) < 2:
        return {}

    result = funnel_tools.compute_funnel(df, variant_col="variant", steps=steps)
    return {"funnel_result": result}


# ── Node 16b: describe_data_node (general analysis) ───────────────────────────

@observe(name="describe_data")
def describe_data_node(state: AgentState) -> dict:
    df = _safe_df(state)
    if df is None:
        logger.warning("describe_data: no query_result, skipping.")
        return {}
    try:
        result = describe_tools.describe_dataframe(df)
        return {"describe_result": result}
    except Exception as exc:
        logger.warning("describe_data: failed (%s), skipping.", exc)
        return {}


# ── Node 16c: find_correlations_node (general analysis) ───────────────────────

@observe(name="find_correlations")
def find_correlations_node(state: AgentState) -> dict:
    df = _safe_df(state)
    if df is None:
        logger.warning("find_correlations: no query_result, skipping.")
        return {}
    try:
        result = describe_tools.compute_correlations(df)
        return {"correlation_result": result}
    except Exception as exc:
        logger.warning("find_correlations: failed (%s), skipping.", exc)
        return {}


# ── Node 16c-2: run_regression_node ───────────────────────────────────────────

@observe(name="run_regression")
def run_regression_node(state: AgentState) -> dict:
    """OLS regression — identifies predictors of a target column.

    Skipped gracefully when:
      - No query_result available
      - Fewer than 10 rows (insufficient for regression)
      - No usable numeric features (pure categorical dataset)
    """
    df = _safe_df(state)
    if df is None:
        logger.warning("run_regression: no query_result, skipping.")
        return {}
    if len(df) < 10:
        logger.warning("run_regression: only %d rows, skipping.", len(df))
        return {}
    try:
        result = regression_tools.run_regression(
            df, task_hint=state.get("task", "")
        )
        logger.info(
            "run_regression: target=%s n=%d R²=%.4f",
            result.target, result.n_obs, result.r_squared,
        )
        return {"regression_result": result}
    except ValueError as exc:
        logger.warning("run_regression: skipped (%s).", exc)
        return {}
    except Exception as exc:
        logger.warning("run_regression: failed (%s), skipping.", exc)
        return {}


# ── Node 16c-3: detect_timeseries_node ────────────────────────────────────────

_TS_COL_PATTERNS = re.compile(
    r"\b(date|month|week|year|day|quarter|period|time|dt|timestamp)\b",
    re.IGNORECASE,
)


@observe(name="detect_timeseries")
def detect_timeseries_node(state: AgentState) -> dict:
    """Run anomaly detection + forecasting when a time column is detected.

    Reuses the existing anomaly_tools and forecast_tools (already tested for A/B
    pre-experiment context). For general mode we run them opportunistically when
    we detect a time/date column — results are injected into state so the narrative
    LLM can reference trends, anomalies, and forecasts.

    Skipped when:
      - No time-like column found
      - Fewer than 6 rows after grouping (not enough history)
    """
    df = _safe_df(state)
    if df is None:
        return {}

    # Detect a time-like column
    time_col: str | None = None
    for col in df.columns:
        if _TS_COL_PATTERNS.search(col):
            time_col = col
            break

    if time_col is None:
        logger.info("detect_timeseries: no time column found, skipping.")
        return {}

    # Build a daily_df-style frame: sort by time column, pick first numeric
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude columns that look like IDs
    _ID_PAT = re.compile(r"\b(id|_id|uuid)\b", re.IGNORECASE)
    metric_cols = [c for c in numeric_cols if not _ID_PAT.search(c)]
    if not metric_cols:
        logger.info("detect_timeseries: no suitable metric column, skipping.")
        return {}

    # Use the highest-variance numeric column as the metric
    metric_col = str(df[metric_cols].var().idxmax())

    try:
        ts_df = (
            df[[time_col, metric_col]]
            .dropna()
            .rename(columns={time_col: "date", metric_col: metric_col})
            .sort_values("date")
        )
        ts_df["date"] = pd.to_datetime(ts_df["date"], errors="coerce")
        ts_df = ts_df.dropna(subset=["date"])

        if len(ts_df) < 6:
            logger.info("detect_timeseries: only %d time points, skipping.", len(ts_df))
            return {}

        # Group to daily-ish if granular (avoid per-row noise)
        ts_df = ts_df.groupby("date")[metric_col].mean().reset_index()
        ts_df.columns = ["date", metric_col]

        result: dict = {}

        # Anomaly detection
        try:
            anomaly = anomaly_tools.detect_anomaly(ts_df, metric_col=metric_col)
            result["anomaly_result"] = anomaly
        except Exception as exc:
            logger.warning("detect_timeseries/anomaly: %s", exc)

        # Forecasting
        try:
            forecast = forecast_tools.forecast_baseline(ts_df, metric_col=metric_col)
            result["forecast_result"] = forecast
        except Exception as exc:
            logger.warning("detect_timeseries/forecast: %s", exc)

        if result:
            logger.info(
                "detect_timeseries: time_col=%s metric=%s rows=%d results=%s",
                time_col, metric_col, len(ts_df), list(result.keys()),
            )
        return result

    except Exception as exc:
        logger.warning("detect_timeseries: failed (%s), skipping.", exc)
        return {}


# ── Node 16d: generate_charts_node ────────────────────────────────────────────

@observe(name="generate_charts")
def generate_charts_node(state: AgentState) -> dict:
    """
    Deterministic chart generation — no LLM calls.
    Runs after the analysis computations but before the analysis gate so that
    charts are available in the gate's interrupt payload.
    """
    mode = state.get("analysis_mode", "ab_test")
    try:
        if mode == "general":
            describe = state.get("describe_result")
            corr     = state.get("correlation_result")
            qr       = state.get("query_result", pd.DataFrame())
            # Prefer total_records column (sum of per-group counts) or the raw
            # table row count from schema_context.  Fall back to describe.row_count
            # only as last resort — that reflects the query result size, not the
            # original dataset size, when the SQL aggregated rows.
            if isinstance(qr, pd.DataFrame) and "total_records" in qr.columns:
                n_underlying = int(qr["total_records"].sum())
            else:
                # Try parsing "TABLE: foo  -- 1,648 rows" from schema_context
                schema_ctx = state.get("schema_context", "") or ""
                m = re.search(r"TABLE:[^\n]*--\s*([\d,]+)\s*rows", schema_ctx)
                if m:
                    n_underlying = int(m.group(1).replace(",", ""))
                elif describe:
                    n_underlying = describe.row_count
                else:
                    n_underlying = len(qr)
            if describe and corr:
                specs  = generate_general_charts(describe, corr)
                charts = [s.model_dump() for s in specs]
                ti     = compute_trust_indicators(describe, None, n_underlying)
            else:
                charts = []
                ti     = compute_trust_indicators(None, None, n_underlying)
        else:
            metric  = state.get("metric", "metric")
            ttest   = state.get("ttest_result")
            cuped   = state.get("cuped_result")
            hte     = state.get("hte_result")
            novelty = state.get("novelty_result")
            funnel  = state.get("funnel_result")
            if ttest and cuped:
                specs  = generate_ab_charts(metric, ttest, cuped, hte, novelty, funnel)
                charts = [s.model_dump() for s in specs]
            else:
                charts = []
            n_rows = len(state.get("query_result", pd.DataFrame()))  # type: ignore[arg-type]
            ti     = compute_trust_indicators(None, ttest, n_rows)
        return {"charts": charts, "trust_indicators": ti.model_dump()}
    except Exception as exc:
        logger.warning("generate_charts: failed (%s), skipping.", exc)
        return {}


# ── Node 17: analysis_gate (HITL interrupt 2) ─────────────────────────────────

@observe(name="analysis_gate")
def analysis_gate(state: AgentState) -> dict:
    mode         = state.get("analysis_mode", "ab_test")
    srm_detected = False   # overridden in ab_test branch below

    if mode == "power_analysis":
        pa = state.get("power_analysis_result")
        payload = {
            "gate":                   "analysis",
            "analysis_mode":          "power_analysis",
            "power_analysis_result":  _to_dict(pa),
            "message":                "Review the power analysis. Approve or add notes.",
        }

    elif mode == "general":
        describe_res     = state.get("describe_result")
        correlation_res  = state.get("correlation_result")
        payload = {
            "gate":             "analysis",
            "analysis_mode":    "general",
            "describe_result":  _to_dict(describe_res),
            "correlation_result": _to_dict(correlation_res),
            "message":          "Review the data summary and insights. Approve or add notes.",
        }
    else:
        slice_res     = state.get("slice_result")
        slice_dims    = slice_res.ranked_dimensions if slice_res else []
        top_slice     = slice_dims[0] if slice_dims else {}

        guardrail_res = state.get("guardrail_result")
        breached      = [g for g in (guardrail_res.guardrails if guardrail_res else []) if g.breached]

        forecast_res  = state.get("forecast_result")
        cuped_res     = state.get("cuped_result")
        ttest_res     = state.get("ttest_result")
        srm_res       = state.get("srm_result")
        hte_res       = state.get("hte_result")
        novelty_res   = state.get("novelty_result")
        funnel_res    = state.get("funnel_result")
        mde_res       = state.get("mde_result")

        srm_detected = srm_res.srm_detected if srm_res else False

        verification_checklist = {
            "n_control":            ttest_res.n_control       if ttest_res     else None,
            "n_treatment":          ttest_res.n_treatment     if ttest_res     else None,
            "srm_detected":         srm_res.srm_detected      if srm_res       else None,
            "p_value":              ttest_res.p_value         if ttest_res     else None,
            "significant":          ttest_res.significant     if ttest_res     else None,
            "cohens_d":             ttest_res.cohens_d        if ttest_res     else None,
            "cuped_ate":            cuped_res.cuped_ate       if cuped_res     else None,
            "post_hoc_power":       mde_res.post_hoc_power    if mde_res       else None,
            "any_guardrail_breach": guardrail_res.any_breached if guardrail_res else None,
            "sql_warnings":         state.get("sql_validation_warnings") or [],
        }

        gate_message = (
            "⛔ SAMPLE RATIO MISMATCH DETECTED — statistical results are unreliable. "
            "You must set srm_acknowledged=true in your response to proceed, confirming "
            "you understand the results cannot be trusted."
            if srm_detected else
            "Review the analysis results. Approve or add notes/overrides."
        )

        payload = {
            "gate":                    "analysis",
            "analysis_mode":           "ab_test",
            "srm_detected":            srm_detected,
            "decomposition":           _to_dict(state.get("decomposition_result")),
            "top_anomaly_slice":       top_slice,
            "forecast_outside_ci":     forecast_res.outside_ci if forecast_res else None,
            "cuped_variance_reduction": cuped_res.variance_reduction_pct if cuped_res else None,
            "significant":             ttest_res.significant if ttest_res else None,
            "top_segment":             hte_res.top_segment if hte_res else None,
            "novelty_likely":          novelty_res.novelty_likely if novelty_res else None,
            "guardrails_breached":     guardrail_res.any_breached if guardrail_res else None,
            "breached_metrics":        [g.model_dump() for g in breached],
            "biggest_funnel_dropoff":  funnel_res.biggest_dropoff_step if funnel_res else None,
            "mde_powered":             mde_res.is_powered_for_observed_effect if mde_res else None,
            "business_impact":         state.get("business_impact"),
            "verification_checklist":  verification_checklist,
            "message":                 gate_message,
        }

    analyst_response = interrupt(payload)

    notes    = analyst_response.get("notes", "")
    approved = analyst_response.get("approved", True)
    override = dict(state.get("analyst_override") or {})
    if notes.strip():
        override["analysis_notes"] = notes.strip()

    # SRM gate: analyst must explicitly acknowledge before proceeding.
    # `srm_detected` is set at function top (False) and overridden in ab_test branch.
    if srm_detected and not analyst_response.get("srm_acknowledged"):
        logger.warning(
            "analysis_gate: SRM detected but not acknowledged — blocking approval."
        )
        return {
            "analysis_approved": False,
            "analyst_notes":     notes,
            "analyst_override":  override,
        }

    result: dict = {
        "analysis_approved": approved,
        "analyst_notes":     notes,
        "analyst_override":  override,
    }
    if srm_detected and analyst_response.get("srm_acknowledged"):
        result["srm_acknowledged"] = True
    return result
