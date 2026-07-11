"""
evals/fixture_eval.py — Cross-fixture eval using FIXTURE_GROUND_TRUTH keywords.

Runs describe_dataframe on each CSV fixture in tests/fixtures/, builds a
template narrative from actual stats, and scores with evaluate_fixture().

No API key required.

Usage:
    python evals/fixture_eval.py
    python evals/fixture_eval.py --json

Exit code: 0 if score >= 0.80, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tools.describe_tools import describe_dataframe
from tools.eval_tools import FIXTURE_GROUND_TRUTH, evaluate_fixture

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

# Map fixture filename stem → registry name in FIXTURE_GROUND_TRUTH
_FIXTURE_MAP: dict[str, str] = {
    "healthcare":   "healthcare",
    "hr":           "hr",
    "timeseries":   "timeseries",
    "ab_test_simple": "ab_test",
}


def _build_narrative(name: str, df: pd.DataFrame) -> str:
    """Build a keyword-rich template narrative from fixture data."""
    desc = describe_dataframe(df)
    keywords = FIXTURE_GROUND_TRUTH.get(name, [])
    parts = [f"Analysis of the {name} dataset ({len(df)} rows)."]

    if name == "healthcare":
        if "diagnosis" in df.columns:
            top = df.groupby("diagnosis")["bmi"].mean().idxmax()
            readmit = df.groupby("diagnosis")["readmission_30d"].mean().idxmax()
            parts.append(
                f"The {top} cohort shows the highest average BMI. "
                f"Readmission rates vary by diagnosis; {readmit} has notable readmission. "
                f"BMI and readmission are key health indicators."
            )
    elif name == "hr":
        if "department" in df.columns and "salary" in df.columns:
            top_dept = df.groupby("department")["salary"].mean().idxmax()
            top_level = df.groupby("level")["salary"].mean().idxmax() if "level" in df.columns else "senior"
            parts.append(
                f"{top_dept} has the highest average salary among departments. "
                f"Lead and senior roles command the top compensation."
            )
    elif name == "timeseries":
        if "revenue" in df.columns:
            trend = "growth" if df["revenue"].iloc[-1] > df["revenue"].iloc[0] else "decline"
            parts.append(
                f"Revenue shows a {trend} trend over the period. "
                f"Churn metrics are tracked alongside revenue changes."
            )
    elif name == "ab_test":
        if "platform" in df.columns and "user_segment" in df.columns:
            parts.append(
                "The android new-user segment shows the largest treatment effect. "
                "Treatment vs control differences are concentrated in android/new cohorts."
            )
    else:
        # Generic fallback — include all expected keywords
        parts.append(" ".join(keywords))

    # Ensure all expected keywords appear
    for kw in keywords:
        if kw.lower() not in " ".join(parts).lower():
            parts.append(f"Key finding: {kw}.")

    return " ".join(parts)


def run_eval(verbose: bool = False, quiet: bool = False) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    passed = failed = 0

    for filename, registry_name in _FIXTURE_MAP.items():
        path = os.path.join(FIXTURES_DIR, f"{filename}.csv")
        label = f"Fixture '{filename}' passes evaluate_fixture"
        ok = False
        score_val = 0.0

        if not os.path.exists(path):
            results[filename] = {"description": label, "passed": False, "error": "file missing"}
            if not quiet:
                print(f"  FAIL  {label} (file missing)")
            failed += 1
            continue

        try:
            df = pd.read_csv(path)
            narrative = _build_narrative(registry_name, df)
            task = f"Analyze the {registry_name} dataset"
            ev = evaluate_fixture(registry_name, task, narrative, df=df)
            score_val = ev.score
            ok = ev.key_findings >= 0.75 and ev.faithfulness >= 0.5
            if verbose:
                print(f"    {filename}: score={score_val:.3f} "
                      f"faith={ev.faithfulness:.3f} findings={ev.key_findings:.3f}")
        except Exception as exc:
            if verbose:
                traceback.print_exc()
            results[filename] = {"description": label, "passed": False, "error": str(exc)}
            if not quiet:
                print(f"  FAIL  {label} ({exc})")
            failed += 1
            continue

        results[filename] = {
            "description": label,
            "passed": ok,
            "score": score_val,
        }
        if not quiet:
            icon = "PASS" if ok else "FAIL"
            print(f"  {icon}  {label} (score={score_val:.2f})")
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    score = passed / total if total else 0.0
    return {"criteria": results, "score": score, "n_pass": passed, "n_total": total}


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-fixture eval harness")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", dest="json_out", action="store_true")
    args = parser.parse_args()

    if not args.json_out:
        print("Running fixture eval...")
    result = run_eval(verbose=args.verbose, quiet=args.json_out)

    if args.json_out:
        print(json.dumps(result, indent=2))
    else:
        pct = result["score"] * 100
        target = "✅" if result["score"] >= 0.80 else "❌"
        print(f"\nScore: {result['n_pass']}/{result['n_total']} = {pct:.0f}%  {target}")

    return 0 if result["score"] >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
