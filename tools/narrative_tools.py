"""
tools/narrative_tools.py — Structured finding → PM narrative formatter.

Assembles all tool results into a 7-section markdown draft following the
structure defined in CLAUDE.md. No LLM call — pure template formatting.
The LLM in agents/analyze/nodes.py refines this draft.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

from typing import Any

from tools.schemas import NarrativeResult


def format_narrative(
    metric: str,
    decomposition_result: dict[str, Any],
    anomaly_result: dict[str, Any],
    cuped_result: dict[str, Any],
    ttest_result: dict[str, Any],
    hte_result: dict[str, Any],
    novelty_result: dict[str, Any],
    mde_result: dict[str, Any],
    guardrail_result: dict[str, Any],
    funnel_result: dict[str, Any],
    forecast_result: dict[str, Any],
    business_impact: str,
    analyst_notes: str = "",
    srm_result: dict[str, Any] | None = None,
) -> NarrativeResult:
    """
    Format all analysis results into a PM-ready markdown narrative.

    Args:
        metric:               Name of the primary metric (e.g. 'dau_rate').
        decomposition_result: Output of decompose_dau().
        anomaly_result:       Output of detect_anomaly().
        cuped_result:         Output of run_cuped().
        ttest_result:         Output of run_ttest().
        hte_result:           Output of run_hte().
        novelty_result:       Output of detect_novelty_effect().
        mde_result:           Output of compute_mde().
        guardrail_result:     Output of check_guardrails().
        funnel_result:        Output of compute_funnel().
        forecast_result:      Output of forecast_baseline().
        business_impact:      Output of business_impact_statement().
        analyst_notes:        Optional free-text override/annotation from analyst.

    Returns:
        {
            narrative_draft:  str,   # full 7-section markdown writeup
            recommendation:   str,   # extracted one-sentence action recommendation
        }
    """
    # Distinguish "test ran, not significant" from "test never ran".
    ttest_ran        = bool(ttest_result)
    sig              = ttest_result.get("significant", False)      if ttest_ran else None
    p_value          = ttest_result.get("p_value", None)           if ttest_ran else None
    cohens_d         = ttest_result.get("cohens_d",  None)         if ttest_ran else None
    n_control        = ttest_result.get("n_control",  0)           if ttest_ran else 0
    n_treatment      = ttest_result.get("n_treatment", 0)          if ttest_ran else 0
    ci_lower         = ttest_result.get("ci_lower",  None)         if ttest_ran else None
    ci_upper         = ttest_result.get("ci_upper",  None)         if ttest_ran else None
    winsorized       = ttest_result.get("winsorized", False)        if ttest_ran else False
    skewness_warning = ttest_result.get("skewness_warning", None)  if ttest_ran else None
    cuped_ran  = bool(cuped_result)
    cuped_ate  = cuped_result.get("cuped_ate", 0.0)       if cuped_ran else None
    var_red    = cuped_result.get("variance_reduction_pct", 0.0) if cuped_ran else None
    top_seg         = hte_result.get("top_segment", "unknown")
    seg_effect      = hte_result.get("effect_size", 0.0)
    seg_share       = hte_result.get("segment_share", 0.0)
    interaction_p   = hte_result.get("interaction_p_value", None)

    novelty_likely    = novelty_result.get("novelty_likely", False)
    effect_direction  = novelty_result.get("effect_direction", "unknown")
    week1_ate         = novelty_result.get("week1_ate", 0.0)
    week2_ate         = novelty_result.get("week2_ate", 0.0)

    powered           = mde_result.get("is_powered_for_observed_effect")
    mde_rel           = mde_result.get("mde_relative_pct", 0.0)

    any_breached      = guardrail_result.get("any_breached", False)
    breached          = [g for g in guardrail_result.get("guardrails", []) if g.get("breached")]

    biggest_dropoff   = funnel_result.get("biggest_dropoff_step", "unknown")
    dropoff_step_data = next(
        (s for s in funnel_result.get("steps", []) if s["step"] == biggest_dropoff),
        None,
    )

    forecast_outside  = forecast_result.get("outside_ci", False)
    forecast_method   = forecast_result.get("method", "rolling_mean")
    forecast_warning  = forecast_result.get("warning")

    dominant_comp     = decomposition_result.get("dominant_change_component", "unknown")
    anomaly_dates     = anomaly_result.get("anomaly_dates", [])
    anomaly_dir       = anomaly_result.get("direction", "drop")
    anomaly_severity  = anomaly_result.get("severity", 0.0)

    srm_detected      = (srm_result or {}).get("srm_detected", False)
    srm_n_ctrl        = (srm_result or {}).get("n_control",  0)
    srm_n_trt         = (srm_result or {}).get("n_treatment", 0)
    srm_ratio         = (srm_result or {}).get("observed_ratio", 0.5)
    srm_p             = (srm_result or {}).get("p_value", 1.0)

    post_hoc_power    = mde_result.get("post_hoc_power")

    # ── 1. TL;DR ────────────────────────────────────────────────────────────────
    if sig is None:
        sig_str = "significance unknown (analysis did not run)"
    else:
        sig_str = "statistically significant" if sig else "not statistically significant"

    ate_str = f"CUPED ATE={cuped_ate:+.4f}" if cuped_ate is not None else "CUPED ATE=n/a"
    p_str   = f"p={p_value:.4f}"            if p_value  is not None else "p=n/a"

    tldr = (
        f"A {anomaly_dir} in **{metric}** was detected"
        + (f" starting {anomaly_dates[0]}" if anomaly_dates else "")
        + f", driven primarily by the **{dominant_comp}** component. "
        f"The experiment effect ({ate_str}) is {sig_str} "
        f"({p_str}), concentrated in the **{top_seg}** segment."
    )

    # ── 2. What we found ────────────────────────────────────────────────────────
    if cuped_ate is not None and var_red is not None and p_value is not None:
        d_str = (
            f", Cohen's d={cohens_d:+.2f}"
            + (" (negligible)" if cohens_d is not None and abs(cohens_d) < 0.1
               else " (small)"    if cohens_d is not None and abs(cohens_d) < 0.2
               else " (medium)"   if cohens_d is not None and abs(cohens_d) < 0.5
               else " (large)"    if cohens_d is not None
               else "")
        ) if cohens_d is not None else ""
        n_str = f" [n={n_control:,} ctrl / {n_treatment:,} trt]" if n_control and n_treatment else ""
        wins_str = " _(outliers winsorized at 1%)_" if winsorized else ""
        cuped_line = (
            f"- **Experiment (CUPED):** ATE={cuped_ate:+.4f} "
            f"(variance reduction {var_red:.1f}%){d_str}, {p_str} → {sig_str.upper()}.{n_str}{wins_str}"
        )
    else:
        cuped_line = "- **Experiment:** Analysis could not be computed. Required columns may be missing."

    found_lines = [
        f"- **Decomposition:** `{dominant_comp}` is the dominant change component.",
        f"- **Anomaly:** {anomaly_dir.capitalize()} detected"
        + (f" on {anomaly_dates[0]}" if anomaly_dates else "")
        + f" (severity Z={anomaly_severity:.2f}).",
        f"- **Forecast:** Actuals are {'outside' if forecast_outside else 'within'} the "
        f"{forecast_method} confidence interval"
        + (". Confirms real signal." if forecast_outside else ". Drop may be within normal variance.")
        + (f" ⚠ {forecast_warning}" if forecast_warning else ""),
        cuped_line,
    ]

    # ── 3. Where it's concentrated ──────────────────────────────────────────────
    # Interaction p-value: confirms whether HTE heterogeneity is real (not noise).
    if interaction_p is not None:
        interact_str = (
            f" Interaction test p={interaction_p:.4f} — "
            + ("heterogeneity is **statistically confirmed**." if interaction_p < 0.05
               else "heterogeneity not yet statistically confirmed (may be sampling noise).")
        )
    else:
        interact_str = ""

    concentration_lines = [
        f"- **Top segment (HTE):** {top_seg}: effect size {seg_effect:+.4f}, "
        f"represents {seg_share:.1f}% of experiment users.{interact_str}",
    ]
    if dropoff_step_data:
        concentration_lines.append(
            f"- **Funnel:** Largest drop-off at `{biggest_dropoff}` step "
            f"(ctrl={dropoff_step_data['control_rate']:.3f}, "
            f"trt={dropoff_step_data['treatment_rate']:.3f}, "
            f"Δ={dropoff_step_data['delta']:+.3f})."
        )

    # ── 4. What else is affected (guardrails) ───────────────────────────────────
    if any_breached:
        guardrail_lines = [
            f"- ⚠️ **{g['metric']}** breached: control={g['control_mean']:.4f}, "
            f"treatment={g['treatment_mean']:.4f} "
            f"(Δ={g['delta_pct']:+.1f}%, p={g['p_value']:.4f})."
            for g in breached
        ]
    else:
        guardrail_lines = ["- All guardrail metrics within acceptable bounds."]

    # ── 5. Confidence level ──────────────────────────────────────────────────────
    confidence_lines = []

    # SRM takes priority — if randomization is broken, everything else is suspect.
    if srm_detected:
        confidence_lines.append(
            f"- ⛔ **SRM detected:** Control={srm_n_ctrl:,} / Treatment={srm_n_trt:,} "
            f"(observed ratio={srm_ratio:.3f}, expected=0.500, p={srm_p:.6f}). "
            "**Randomization may be broken — all statistical results below should be treated with extreme caution.**"
        )

    # Post-hoc power is more informative than the simple MDE comparison.
    if post_hoc_power is not None:
        pwr_pct = post_hoc_power * 100
        pwr_icon = "✅" if post_hoc_power >= 0.80 else "⚠️"
        confidence_lines.append(
            f"- {pwr_icon} **Statistical power:** {pwr_pct:.0f}% post-hoc power "
            f"(MDE={mde_rel:.1f}%)"
            + (". Adequately powered." if post_hoc_power >= 0.80
               else ". **Underpowered** — if the null is true this study would miss the effect ~"
               f"{100 - pwr_pct:.0f}% of the time. Treat the result cautiously.")
        )
    elif powered is True:
        confidence_lines.append(f"- ✅ **MDE:** Experiment is powered for the observed effect (MDE={mde_rel:.1f}%).")
    elif powered is False:
        confidence_lines.append(
            f"- ⚠️ **MDE:** Observed effect may be near or below MDE ({mde_rel:.1f}%). "
            "CUPED variance reduction is critical for detection."
        )
    else:
        confidence_lines.append(f"- **MDE:** {mde_rel:.1f}% (observed effect comparison not available).")

    confidence_lines.append(
        f"- {'✅' if not novelty_likely else '⚠️'} **Novelty effect:** "
        f"Effect is **{effect_direction}** (week1 ATE={week1_ate:+.4f}, week2={week2_ate:+.4f}). "
        f"{'Novelty decay likely.' if novelty_likely else 'Novelty ruled out.'}"
    )

    if forecast_outside:
        confidence_lines.append("- ✅ **Forecast:** Drop confirmed outside pre-experiment confidence interval.")
    else:
        confidence_lines.append("- ⚠️ **Forecast:** Drop within forecast CI. Could be natural variance.")

    confidence_lines.append(f"- **Business impact:** {business_impact}")

    # ── 6. Recommendation ───────────────────────────────────────────────────────
    p_display = f"p={p_value:.4f}" if p_value is not None else "significance unknown"

    if srm_detected:
        recommendation = (
            f"**Stop: fix randomization first.** The experiment has a Sample Ratio Mismatch "
            f"(control={srm_n_ctrl:,}, treatment={srm_n_trt:,}, p={srm_p:.6f}). "
            "All statistical conclusions are unreliable until the assignment mechanism is corrected and the experiment is re-run."
        )
    elif sig and any_breached:
        recommendation = (
            f"Roll back or pause the experiment. Significant harm detected "
            f"in **{top_seg}** ({', '.join(g['metric'] for g in breached)} guardrails breached)."
        )
    elif sig and not any_breached:
        recommendation = (
            f"Investigate root cause in **{top_seg}** segment before shipping. "
            f"Significant {metric} impact confirmed with no guardrail breaches."
        )
    elif not sig and powered is False:
        # Underpowered: give a concrete "extend the experiment" message.
        n_have = min(n_control, n_treatment) if (n_control and n_treatment) else 0
        if n_have > 0 and mde_rel > 0:
            # Rough scale-up factor: current MDE² / target-MDE² ≈ n ratio.
            # Tell the PM how much bigger the experiment needs to be.
            scale_factor = (mde_rel / 5.0) ** 2   # assumes target MDE = 5%
            n_needed = int(n_have * max(scale_factor, 1.5))
            shortfall = max(0, n_needed - n_have)
            recommendation = (
                f"Insufficient data — the experiment needs roughly {n_needed:,} users per arm "
                f"to detect a ≥5% effect (currently {n_have:,}, shortfall ~{shortfall:,}). "
                f"Extend the experiment before drawing conclusions ({p_display})."
            )
        else:
            recommendation = (
                f"Insufficient data. The experiment is underpowered for the observed effect "
                f"({p_display}). Extend the runtime to reach adequate statistical power."
            )
    else:
        recommendation = (
            f"Collect more data. Effect in **{top_seg}** is directionally negative "
            f"but not yet reliable ({p_display}). Monitor guardrails closely."
        )

    # ── 7. Caveats ──────────────────────────────────────────────────────────────
    caveats = [
        "- This analysis covers the experiment period only; pre-experiment confounders are not fully controlled for.",
        f"- HTE subgroup ({top_seg}) was identified post-hoc. Treat results as exploratory, not confirmatory.",
        "- Funnel analysis uses randomized assignment; any selection effects at upper funnel steps are not modeled.",
    ]
    if not forecast_outside:
        caveats.append("- Forecast did not flag the drop as anomalous. Consider extending the baseline window.")
    if powered is False:
        caveats.append("- Experiment may be underpowered for the blended effect; segment-level estimates are more reliable.")
    if forecast_warning:
        caveats.append(f"- Forecast warning: {forecast_warning}")
    # Data-driven: flag negligible effect size even when statistically significant
    if sig and cohens_d is not None and abs(cohens_d) < 0.1:
        caveats.append(
            f"- Effect is statistically significant but Cohen's d={cohens_d:+.2f} is negligible. "
            "Statistical significance does not imply practical importance at this scale."
        )
    # Data-driven: flag wide confidence intervals
    if ci_lower is not None and ci_upper is not None:
        ci_width  = ci_upper - ci_lower
        mid       = (ci_lower + ci_upper) / 2.0
        if abs(mid) > 1e-9 and ci_width > 2.0 * abs(mid):
            caveats.append(
                f"- The 95% CI [{ci_lower:+.4f}, {ci_upper:+.4f}] is wide relative to the "
                "estimated effect — the true effect could be substantially smaller or larger."
            )
    # Data-driven: skewness warning (distribution assumption may be violated)
    if skewness_warning:
        caveats.append(f"- ⚠️ **Skewness:** {skewness_warning}")
    # Data-driven: winsorization disclosure
    if winsorized:
        caveats.append(
            "- Outliers were winsorized at the 1st/99th percentile before testing. "
            "Results reflect the typical user, not extreme outliers."
        )
    # Data-driven: small sample warning
    n_min = min(n_control, n_treatment) if n_control and n_treatment else 0
    if 0 < n_min < 100:
        caveats.append(
            f"- Small sample: min({n_control:,} ctrl, {n_treatment:,} trt) = {n_min:,} users per arm. "
            "Results may not be stable; validate with a holdout cohort."
        )

    # Data-driven: SRM caveat (always add when detected, even if already in recommendation)
    if srm_detected:
        caveats.append(
            f"- ⛔ **SRM:** control={srm_n_ctrl:,} vs treatment={srm_n_trt:,} "
            f"(expected 50/50, p={srm_p:.6f}). "
            "Investigate assignment logs, eligibility filters, and data pipeline before re-running."
        )

    # ── Assemble markdown ────────────────────────────────────────────────────────
    analyst_section = (
        f"\n\n---\n\n**Analyst notes:** {analyst_notes}" if analyst_notes.strip() else ""
    )

    srm_banner = (
        f"\n> ⛔ **WARNING — Sample Ratio Mismatch detected** "
        f"(control={srm_n_ctrl:,} / treatment={srm_n_trt:,}, p={srm_p:.6f}). "
        "Randomization may be broken. Statistical conclusions below are unreliable.\n"
    ) if srm_detected else ""

    narrative_draft = f"""\
{srm_banner}## TL;DR

{tldr}

---

## What we found

{chr(10).join(found_lines)}

---

## Where it's concentrated

{chr(10).join(concentration_lines)}

---

## What else is affected

{chr(10).join(guardrail_lines)}

---

## Confidence level

{chr(10).join(confidence_lines)}

---

## Recommendation

{recommendation}

---

## Caveats

{chr(10).join(caveats)}{analyst_section}
"""

    return NarrativeResult(
        narrative_draft=narrative_draft,
        recommendation=recommendation,
    )
