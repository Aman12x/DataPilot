"""
tools/eval_tools.py — Lightweight RAGAS-inspired evaluation for DataPilot.

Three metrics, all deterministic (no LLM-as-judge):

  faithfulness   — fraction of numbers cited in the narrative that are
                   supported by actual values in the query result DataFrame
                   (within a configurable relative tolerance).
                   Catches hallucinated statistics.

  relevancy      — cosine similarity between the task embedding and the
                   narrative embedding, using the same MiniLM model already
                   loaded by semantic_cache.py.
                   Catches narratives that answer the wrong question.

  key_findings   — fraction of expected keywords / phrases that appear in
                   the narrative.
                   Catches missing top-level insights.

Usage (in-band, per production run):
    from tools.eval_tools import evaluate_run
    result = evaluate_run(task, narrative, df, ground_truth_findings)
    # result.score  → 0-1 composite suitable for eval_score column

Usage (offline, per fixture):
    from tools.eval_tools import evaluate_fixture
    result = evaluate_fixture("healthcare", narrative, df)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    faithfulness:  float        # 0-1
    relevancy:     float        # 0-1  (−1 if model unavailable)
    key_findings:  float        # 0-1  (1.0 if no expected findings provided)
    score:         float        # weighted composite
    details:       dict = field(default_factory=dict)   # per-metric breakdown


# ─── Faithfulness ──────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"""
    \$\s*(\d[\d,]*(?:\.\d+)?)       # dollar amounts: $1,234.56
    |
    (\d[\d,]*(?:\.\d+)?)\s*%        # percentages: 32.7%
    |
    \b(\d+\.\d+)\b                  # bare decimals: 32.685 (not years)
    """,
    re.VERBOSE,
)

# Numbers that look like years, IDs, bullet indices, or section numbers
_EXCLUDE_RE = re.compile(r"\b(19|20)\d{2}\b")


def _extract_narrative_numbers(narrative: str) -> list[float]:
    """
    Extract numeric values cited in the narrative that could be data-derived.
    Excludes years (1900-2099) and low integers (≤5) that are likely ordinal.
    """
    results: list[float] = []
    for m in _NUMBER_RE.finditer(narrative):
        raw = (m.group(1) or m.group(2) or m.group(3)).replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        # Skip years and tiny ordinal integers
        if _EXCLUDE_RE.search(m.group(0)):
            continue
        if v <= 5 and "." not in raw:
            continue
        results.append(v)
    return results


def _df_numeric_values(df: pd.DataFrame) -> list[float]:
    """Return all finite numeric scalars from the DataFrame."""
    values: list[float] = []
    for col in df.select_dtypes(include="number").columns:
        for v in df[col].dropna():
            fv = float(v)
            if math.isfinite(fv):
                values.append(fv)
    return values


def score_faithfulness(
    narrative: str,
    df: pd.DataFrame | None,
    tolerance: float = 0.10,
) -> dict:
    """
    Fraction of numbers in the narrative supported by a value in df within
    `tolerance` relative error.

    Returns a dict with keys: score, supported, total, unsupported_numbers.

    A score of 1.0 means every cited number is traceable to the data.
    A score < 0.7 is a strong signal of hallucination.
    """
    if df is None or df.empty:
        return {"score": 1.0, "supported": 0, "total": 0,
                "unsupported_numbers": [], "note": "no dataframe"}

    narrative_nums = _extract_narrative_numbers(narrative)
    if not narrative_nums:
        return {"score": 1.0, "supported": 0, "total": 0,
                "unsupported_numbers": [], "note": "no numbers in narrative"}

    df_vals = _df_numeric_values(df)
    if not df_vals:
        return {"score": 1.0, "supported": 0, "total": 0,
                "unsupported_numbers": [], "note": "no numeric columns in df"}

    supported = 0
    unsupported: list[float] = []

    for num in narrative_nums:
        denom = abs(num) if abs(num) > 1e-9 else 1.0
        matched = any(abs(num - dv) / denom <= tolerance for dv in df_vals)
        # Also check num/100 to handle percentage-cited-as-fraction mismatch
        # e.g. narrative says "77.4%" → num=77.4, but df stores 0.774
        # Only apply this for numbers in (0, 100] — larger values can't be percentages.
        if not matched and 0 < num <= 100:
            num_as_frac = num / 100.0
            denom_frac = abs(num_as_frac) if abs(num_as_frac) > 1e-9 else 1.0
            matched = any(abs(num_as_frac - dv) / denom_frac <= tolerance for dv in df_vals)
        if matched:
            supported += 1
        else:
            unsupported.append(num)

    total = len(narrative_nums)
    return {
        "score":               supported / total,
        "supported":           supported,
        "total":               total,
        "unsupported_numbers": unsupported,
    }


# ─── Relevancy ─────────────────────────────────────────────────────────────────

def score_relevancy(task: str, narrative: str) -> float:
    """
    Cosine similarity between MiniLM embeddings of task and narrative.
    Returns -1.0 if sentence-transformers is not installed.

    Interpretation:
      ≥ 0.70  → narrative clearly addresses the task
      0.50–0.70 → borderline
      < 0.50  → narrative has drifted from the original question
    """
    try:
        from memory.semantic_cache import cosine_similarity, embed
        task_vec = embed(task)
        narr_vec = embed(narrative)
        return float(cosine_similarity(task_vec, narr_vec))
    except Exception as exc:
        logger.debug("score_relevancy: model unavailable — %s", exc)
        return -1.0


# ─── Key findings ──────────────────────────────────────────────────────────────

def score_key_findings(narrative: str, expected_findings: Sequence[str]) -> dict:
    """
    Fraction of expected keywords/phrases present in the narrative (case-insensitive).

    expected_findings examples:
      ["diabetes", "engineering", "highest", "revenue growth"]

    Returns a dict with keys: score, found, total, missing.
    """
    if not expected_findings:
        return {"score": 1.0, "found": 0, "total": 0, "missing": []}

    lower = narrative.lower()
    found   = [f for f in expected_findings if f.lower() in lower]
    missing = [f for f in expected_findings if f.lower() not in lower]

    return {
        "score":   len(found) / len(expected_findings),
        "found":   len(found),
        "total":   len(expected_findings),
        "missing": missing,
    }


# ─── Composite scorer ──────────────────────────────────────────────────────────

# Weight faithfulness most heavily — hallucinated numbers are the worst failure.
_WEIGHTS = {"faithfulness": 0.5, "relevancy": 0.3, "key_findings": 0.2}


def evaluate_run(
    task: str,
    narrative: str,
    df: pd.DataFrame | None = None,
    ground_truth_findings: Sequence[str] | None = None,
    faithfulness_tolerance: float = 0.10,
) -> EvalResult:
    """
    Run all three RAGAS-inspired metrics and return a composite EvalResult.

    Args:
        task:                   The analyst's original question.
        narrative:              The LLM-generated report / narrative.
        df:                     The query result DataFrame (used for faithfulness).
        ground_truth_findings:  Expected keywords that must appear in the narrative.
        faithfulness_tolerance: Relative tolerance for numeric matching (default 10%).

    Returns:
        EvalResult with per-metric scores and a weighted composite.
    """
    faith_detail  = score_faithfulness(narrative, df, tolerance=faithfulness_tolerance)
    rel_score     = score_relevancy(task, narrative)
    kf_detail     = score_key_findings(narrative, ground_truth_findings or [])

    faith_score = faith_detail["score"]
    kf_score    = kf_detail["score"]

    # If relevancy model unavailable, redistribute its weight to faithfulness
    if rel_score < 0:
        w_faith = _WEIGHTS["faithfulness"] + _WEIGHTS["relevancy"]
        composite = w_faith * faith_score + _WEIGHTS["key_findings"] * kf_score
    else:
        composite = (
            _WEIGHTS["faithfulness"] * faith_score
            + _WEIGHTS["relevancy"]   * rel_score
            + _WEIGHTS["key_findings"] * kf_score
        )

    return EvalResult(
        faithfulness = faith_score,
        relevancy    = rel_score,
        key_findings = kf_score,
        score        = round(composite, 4),
        details      = {
            "faithfulness": faith_detail,
            "key_findings": kf_detail,
            "relevancy":    rel_score,
            "weights":      _WEIGHTS,
        },
    )


# ─── Fixture ground-truth registry ─────────────────────────────────────────────
# Maps fixture name → list of findings the narrative MUST mention.
# These are the most important insights a correct analysis should surface.

FIXTURE_GROUND_TRUTH: dict[str, list[str]] = {
    "healthcare": [
        "diabetes",        # highest BMI and highest readmission rate
        "hypertension",    # highest systolic BP
        "readmission",     # key outcome
        "bmi",
    ],
    "timeseries": [
        "revenue",         # growing revenue is the headline finding
        "churn",           # declining churn is the secondary finding
        "growth",          # or "increase" / "trend"
    ],
    "hr": [
        "engineering",     # highest average salary department
        "salary",          # the metric being analysed
        "lead",            # or "senior" — top level by pay
    ],
    "ab_test": [
        "android",         # top affected segment (DAU drop)
        "new",             # user_segment=new is the affected cohort
        "treatment",       # experiment arm
    ],
}


# ─── Claim accuracy (deterministic, A/B only) ──────────────────────────────────

_SIG_RE     = re.compile(r"\b(statistically\s+significant|significant(ly)?|p\s*<\s*0\.0[0-9]+)\b", re.IGNORECASE)
_NOT_SIG_RE = re.compile(r"\b(not\s+significant|no\s+(statistically\s+)?significant|insignificant)\b", re.IGNORECASE)
_LARGE_RE   = re.compile(r"\b(large|strong)\s+effect\b", re.IGNORECASE)


def score_claim_accuracy(
    narrative: str,
    ttest_result: Any,
    cuped_result: Any,
    alpha: float = 0.05,
) -> dict:
    """
    Deterministic check that A/B interpretation language matches the numbers.

    Checks:
      - "significant" language → p_value < alpha
      - "not significant" language → p_value >= alpha
      - "large/strong effect" language → abs(cohens_d) > 0.5 (skipped if field absent)

    Returns: {"score": fraction_correct, "violations": [str, ...]}
    """
    violations: list[str] = []
    checks_total = 0

    if ttest_result is None:
        return {"score": 1.0, "violations": []}

    p_value = getattr(ttest_result, "p_value", None)
    if p_value is None:
        return {"score": 1.0, "violations": []}

    # Check "significant" claims
    if _SIG_RE.search(narrative):
        checks_total += 1
        if p_value >= alpha:
            violations.append(
                f"Narrative claims significance but p_value={p_value:.4f} >= alpha={alpha}"
            )

    # Check "not significant" claims
    if _NOT_SIG_RE.search(narrative):
        checks_total += 1
        if p_value < alpha:
            violations.append(
                f"Narrative claims non-significance but p_value={p_value:.4f} < alpha={alpha}"
            )

    # Check "large/strong effect" claims
    cohens_d = getattr(ttest_result, "cohens_d", None)
    if cohens_d is None and cuped_result is not None:
        cohens_d = getattr(cuped_result, "cohens_d", None)
    if cohens_d is not None and _LARGE_RE.search(narrative):
        checks_total += 1
        if abs(cohens_d) <= 0.5:
            violations.append(
                f"Narrative claims large effect but abs(cohens_d)={abs(cohens_d):.4f} <= 0.5"
            )

    score = 1.0 - (len(violations) / checks_total) if checks_total > 0 else 1.0
    return {"score": round(score, 4), "violations": violations}


# ─── LLM judge (opt-in, Claude Haiku) ─────────────────────────────────────────

_JUDGE_PROMPT = """\
You are an expert data analyst evaluating a recommendation written by an AI analyst.

TASK (the original question):
{task}

NARRATIVE (the analysis the recommendation is based on):
{narrative}

RECOMMENDATION:
{recommendation}

Score the recommendation on three dimensions (each 0.0 to 1.0):

1. actionability: Is there a concrete, specific next step a stakeholder could act on?
   - 0.0: no clear action suggested
   - 0.5: vague direction ("investigate further", "monitor")
   - 1.0: concrete step with clear owner or mechanism

2. specificity: Does it cite specific data (numbers, segments, timeframes) rather than vague language?
   - 0.0: entirely vague ("results were positive")
   - 0.5: some specificity but missing key numbers from the narrative
   - 1.0: references specific figures/segments from the narrative

3. grounding: Does the recommendation logically follow from the narrative?
   - 0.0: contradicts or ignores the narrative
   - 0.5: weakly related
   - 1.0: clearly derived from the narrative findings

Respond ONLY with valid JSON matching this schema:
{{"actionability": <float>, "specificity": <float>, "grounding": <float>, "reasoning": "<one sentence>"}}
"""


_LLM_JUDGE_TIMEOUT = int(os.getenv("LLM_JUDGE_TIMEOUT", "30"))  # seconds


def score_recommendation(
    recommendation: str,
    narrative: str,
    task: str,
) -> dict:
    """
    Score a recommendation using Claude Haiku as a judge.

    Opt-in via ENABLE_LLM_JUDGE=true env var (called from nodes.py).
    Timeout controlled by LLM_JUDGE_TIMEOUT env var (default 30s).

    Returns:
        {"score": avg, "actionability": f, "specificity": f, "grounding": f, "reasoning": str}
    """
    import anthropic as _anthropic

    client = _anthropic.Anthropic(timeout=_LLM_JUDGE_TIMEOUT)
    prompt = _JUDGE_PROMPT.format(
        task=task or "(not specified)",
        narrative=narrative[:3000],     # cap to avoid huge tokens
        recommendation=recommendation[:500],
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    data = json.loads(raw)

    actionability = float(data.get("actionability", 0.5))
    specificity   = float(data.get("specificity",   0.5))
    grounding     = float(data.get("grounding",     0.5))
    score         = round((actionability + specificity + grounding) / 3, 4)

    return {
        "score":         score,
        "actionability": actionability,
        "specificity":   specificity,
        "grounding":     grounding,
        "reasoning":     data.get("reasoning", ""),
    }


def evaluate_fixture(
    fixture_name: str,
    task: str,
    narrative: str,
    df: pd.DataFrame | None = None,
) -> EvalResult:
    """
    Convenience wrapper that loads the ground-truth finding list for a named
    fixture and calls evaluate_run().

    fixture_name: one of "healthcare", "timeseries", "hr", "ab_test"
    """
    findings = FIXTURE_GROUND_TRUTH.get(fixture_name, [])
    return evaluate_run(task, narrative, df=df, ground_truth_findings=findings)
