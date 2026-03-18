"""
tools/narrative_tools.py — Structured finding → PM narrative formatter.

Assembles all tool results into a 7-section markdown draft following the
structure defined in CLAUDE.md. No LLM call — pure template formatting.
The LLM in agents/analyze/nodes.py refines this draft.

Pure Python, no LangGraph or Streamlit imports.
"""

from __future__ import annotations

from typing import Any


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
) -> dict[str, str]:
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
    sig        = ttest_result.get("significant", False)
    p_value    = ttest_result.get("p_value", 1.0)
    cuped_ate  = cuped_result.get("cuped_ate", 0.0)
    var_red    = cuped_result.get("variance_reduction_pct", 0.0)
    top_seg    = hte_result.get("top_segment", "unknown")
    seg_effect = hte_result.get("effect_size", 0.0)
    seg_share  = hte_result.get("segment_share", 0.0)

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

    # ── 1. TL;DR ────────────────────────────────────────────────────────────────
    sig_str = "statistically significant" if sig else "not statistically significant"
    tldr = (
        f"A {anomaly_dir} in **{metric}** was detected"
        + (f" starting {anomaly_dates[0]}" if anomaly_dates else "")
        + f", driven primarily by the **{dominant_comp}** component. "
        f"The experiment effect (CUPED ATE={cuped_ate:+.4f}) is {sig_str} "
        f"(p={p_value:.4f}), concentrated in the **{top_seg}** segment."
    )

    # ── 2. What we found ────────────────────────────────────────────────────────
    found_lines = [
        f"- **Decomposition:** `{dominant_comp}` is the dominant change component.",
        f"- **Anomaly:** {anomaly_dir.capitalize()} detected"
        + (f" on {anomaly_dates[0]}" if anomaly_dates else "")
        + f" (severity Z={anomaly_severity:.2f}).",
        f"- **Forecast:** Actuals are {'outside' if forecast_outside else 'within'} the "
        f"{forecast_method} confidence interval"
        + (" — confirms real signal." if forecast_outside else " — drop may be within normal variance.")
        + (f" ⚠ {forecast_warning}" if forecast_warning else ""),
        f"- **Experiment (CUPED):** ATE={cuped_ate:+.4f} "
        f"(variance reduction {var_red:.1f}%), p={p_value:.4f} → {sig_str.upper()}.",
    ]

    # ── 3. Where it's concentrated ──────────────────────────────────────────────
    concentration_lines = [
        f"- **Top segment (HTE):** `{top_seg}` — effect size {seg_effect:+.4f}, "
        f"represents {seg_share:.1f}% of experiment users.",
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

    if powered is True:
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
        f"Effect is **{effect_direction}** (week1 ATE={week1_ate:+.4f}, week2={week2_ate:+.4f}) — "
        f"{'novelty decay likely.' if novelty_likely else 'novelty ruled out.'}"
    )

    if forecast_outside:
        confidence_lines.append("- ✅ **Forecast:** Drop confirmed outside pre-experiment confidence interval.")
    else:
        confidence_lines.append("- ⚠️ **Forecast:** Drop within forecast CI — could be natural variance.")

    confidence_lines.append(f"- **Business impact:** {business_impact}")

    # ── 6. Recommendation ───────────────────────────────────────────────────────
    if sig and any_breached:
        recommendation = (
            f"Roll back or pause the experiment — statistically significant harm detected "
            f"in **{top_seg}** ({', '.join(g['metric'] for g in breached)} guardrails breached)."
        )
    elif sig and not any_breached:
        recommendation = (
            f"Investigate root cause in **{top_seg}** segment before shipping — "
            f"significant {metric} impact confirmed with no guardrail breaches."
        )
    else:
        recommendation = (
            f"Collect more data — effect in **{top_seg}** is directionally negative "
            f"but not yet significant (p={p_value:.4f}); monitor guardrails closely."
        )

    # ── 7. Caveats ──────────────────────────────────────────────────────────────
    caveats = [
        "- This analysis covers the experiment period only; pre-experiment confounders are not fully controlled for.",
        f"- HTE subgroup (`{top_seg}`) was identified post-hoc — results should be treated as exploratory, not confirmatory.",
        "- Funnel analysis uses randomized assignment; any selection effects at upper funnel steps are not modeled.",
    ]
    if not forecast_outside:
        caveats.append("- Forecast did not flag the drop as anomalous — consider extending the baseline window.")
    if powered is False:
        caveats.append("- Experiment may be underpowered for the blended effect; segment-level estimates are more reliable.")
    if forecast_warning:
        caveats.append(f"- Forecast warning: {forecast_warning}")

    # ── Assemble markdown ────────────────────────────────────────────────────────
    analyst_section = (
        f"\n\n---\n\n**Analyst notes:** {analyst_notes}" if analyst_notes.strip() else ""
    )

    narrative_draft = f"""\
## TL;DR

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

    return {
        "narrative_draft": narrative_draft,
        "recommendation":  recommendation,
    }
