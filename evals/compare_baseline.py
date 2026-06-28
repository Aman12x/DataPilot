"""
evals/compare_baseline.py — Regression gate for offline eval scores.

Runs all fast offline evals and fails if any score drops below the committed
baseline by more than --tolerance (default 2 percentage points).

Usage:
    python evals/compare_baseline.py
    python evals/compare_baseline.py --update   # refresh evals/baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline.json")

EVAL_COMMANDS: list[tuple[str, list[str]]] = [
    ("analyze_eval",        [sys.executable, "evals/analyze_eval.py", "--skip-narrative", "--json"]),
    ("generalisability_eval", [sys.executable, "evals/generalisability_eval.py", "--json"]),
    ("transactions_eval",   [sys.executable, "evals/transactions_eval.py", "--json"]),
    ("fixture_eval",        [sys.executable, "evals/fixture_eval.py", "--json"]),
]


def _run_eval(name: str, cmd: list[str]) -> dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": ROOT}
    if name == "analyze_eval":
        # Ensure DAU DB exists
        gen = subprocess.run(
            [sys.executable, "data/generate_data.py"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        if gen.returncode != 0:
            raise RuntimeError(f"generate_data failed:\n{gen.stderr}")

    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"{name} crashed (exit {proc.returncode}):\n{proc.stderr}\n{proc.stdout}")

    # JSON eval output — find the outermost object in stdout
    stdout = proc.stdout.strip()
    if "--json" in cmd:
        # Prefer parsing from the first '{' to matching end (last line block)
        start = stdout.find("{")
        if start < 0:
            raise RuntimeError(f"{name}: no JSON in output:\n{stdout}")
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(stdout[start:])
        return result


def _load_baseline() -> dict[str, Any]:
    with open(BASELINE_PATH) as f:
        return json.load(f)


def _save_baseline(evals: dict[str, Any], tolerance: float) -> None:
    payload = {"version": 1, "tolerance": tolerance, "evals": evals}
    with open(BASELINE_PATH, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare offline eval scores to baseline")
    parser.add_argument("--update", action="store_true", help="Write current scores to baseline.json")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Max allowed score drop (default: read from baseline or 0.02)")
    args = parser.parse_args()

    current: dict[str, Any] = {}
    for name, cmd in EVAL_COMMANDS:
        print(f"Running {name}...")
        result = _run_eval(name, cmd)
        current[name] = {
            "score": round(result["score"], 4),
            "n_pass": result["n_pass"],
            "n_total": result["n_total"],
        }
        print(f"  → {result['n_pass']}/{result['n_total']} = {result['score']:.0%}")

    if args.update:
        tol = args.tolerance if args.tolerance is not None else 0.02
        _save_baseline(current, tol)
        print(f"\nBaseline updated at {BASELINE_PATH}")
        return 0

    baseline = _load_baseline()
    tolerance = args.tolerance if args.tolerance is not None else baseline.get("tolerance", 0.02)
    failures: list[str] = []

    for name, cur in current.items():
        base = baseline.get("evals", {}).get(name)
        if not base:
            failures.append(f"{name}: no baseline entry (run with --update)")
            continue
        drop = base["score"] - cur["score"]
        if drop > tolerance:
            failures.append(
                f"{name}: score dropped {drop:.1%} "
                f"({base['score']:.0%} → {cur['score']:.0%}, tolerance={tolerance:.0%})"
            )
        elif cur["n_pass"] < base["n_pass"]:
            failures.append(
                f"{name}: pass count dropped ({base['n_pass']} → {cur['n_pass']})"
            )

    if failures:
        print("\n❌ Baseline regression detected:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\n✅ All eval scores within {tolerance:.0%} of baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
