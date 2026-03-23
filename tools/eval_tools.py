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

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Sequence

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
        # Also check num/100 to handle percentage-cited-as-fraction mismatch
        # e.g. narrative says "77.4%" → num=77.4, but df stores 0.774
        num_as_frac = num / 100.0
        denom_frac = abs(num_as_frac) if abs(num_as_frac) > 1e-9 else 1.0
        if any(abs(num - dv) / denom <= tolerance for dv in df_vals) or \
           any(abs(num_as_frac - dv) / denom_frac <= tolerance for dv in df_vals):
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
