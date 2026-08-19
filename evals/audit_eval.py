"""Catch-rate eval for the narrative audit.

    python evals/audit_eval.py [--json report.json] [--threshold 0.8]

LLM-live (runs the exact audit call production makes, via
`nodes_narrative.run_narrative_audit`), so it is NOT in the per-PR gate — run
manually or from the Eval Nightly workflow.

Method: one correct narrative written against a fixed ground-truth tool
result, plus N single-sentence mutations of it — each plants exactly one
CRITICAL violation of the audit rules (wrong gap, flipped direction, invented
number). A mutation is *caught* when the audit returns a critical finding
whose quote overlaps the mutated sentence. The clean narrative is audited
too: any critical finding there is a false positive.

Reports catch rate (recall on planted errors) and false-positive count; the
threshold applies to catch rate, and a false positive on the clean narrative
also fails the run (an audit that flags correct numbers trains analysts to
approve through it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOOL_RESULTS = {
    "ttest_result": {
        "metric": "dau_flag", "n_control": 10000, "n_treatment": 10000,
        "control_mean": 0.6504, "treatment_mean": 0.6379,
        "mean_diff": -0.0125, "p_value": 0.0003, "significant": True,
        "ci_lower": -0.0192, "ci_upper": -0.0058,
    },
    "hte_result": {
        "segments": [
            {"segment": "platform=ios",     "control_mean": 0.661, "treatment_mean": 0.640, "delta": -0.021, "n_control": 4000, "n_treatment": 4010},
            {"segment": "platform=android", "control_mean": 0.645, "treatment_mean": 0.640, "delta": -0.005, "n_control": 4500, "n_treatment": 4480},
            {"segment": "platform=web",     "control_mean": 0.638, "treatment_mean": 0.627, "delta": -0.011, "n_control": 1500, "n_treatment": 1510},
        ],
        "top_segment": "platform=ios",
    },
    "guardrail_result": {
        "guardrails": [
            {"metric": "notif_optout", "control_mean": 0.0310, "treatment_mean": 0.0342, "delta_pct": 10.3, "breached": True}
        ],
        "any_breached": True,
    },
    "srm_result": {"n_control": 10000, "n_treatment": 10000, "observed_ratio": 0.5, "p_value": 0.99, "srm_detected": False},
}

CLEAN = """\
## TL;DR
Treatment reduced dau_flag by 0.0125 (0.6379 vs 0.6504), a significant decline across 10,000 users per arm.

## Findings
- The decline is concentrated on iOS: 0.640 vs 0.661 (-0.021), n=4000 ctrl / 4010 trt of 10000 / 10000 total analyzed.
- Android barely moved: 0.640 vs 0.645 (-0.005), n=4500 ctrl / 4480 trt of 10000 / 10000 total analyzed.
- Web fell 0.011 (0.627 vs 0.638), n=1500 ctrl / 1510 trt of 10000 / 10000 total analyzed.
- Guardrail breached: notif_optout rose from 0.0310 to 0.0342, a 10.3% increase.
- No sample ratio mismatch: 10,000 vs 10,000.

## Recommendation
Do not ship. The treatment lowers daily activity and raises opt-outs; iOS drives the loss.
"""

# (sentence to replace, replacement, what it violates)
MUTATIONS: list[tuple[str, str, str]] = [
    ("Treatment reduced dau_flag by 0.0125 (0.6379 vs 0.6504)",
     "Treatment reduced dau_flag by 0.0215 (0.6379 vs 0.6504)", "arithmetic: wrong gap"),
    ("Treatment reduced dau_flag by 0.0125 (0.6379 vs 0.6504), a significant decline",
     "Treatment lifted dau_flag by 0.0125 (0.6379 vs 0.6504), a significant improvement", "direction: flipped"),
    ("The decline is concentrated on iOS: 0.640 vs 0.661 (-0.021)",
     "The decline is concentrated on iOS: 0.640 vs 0.661 (-0.041)", "arithmetic: wrong segment gap"),
    ("Android barely moved: 0.640 vs 0.645 (-0.005)",
     "Android improved: 0.645 vs 0.640 (+0.005)", "direction: flipped segment"),
    ("Web fell 0.011 (0.627 vs 0.638)",
     "Web fell 0.011 (0.612 vs 0.638)", "invented number: control mean not in data"),
    ("notif_optout rose from 0.0310 to 0.0342, a 10.3% increase",
     "notif_optout rose from 0.0310 to 0.0342, a 3.1% increase", "arithmetic: wrong percent"),
    ("No sample ratio mismatch: 10,000 vs 10,000.",
     "No sample ratio mismatch: 10,000 vs 12,000.", "invented number: arm size"),
    ("iOS drives the loss.",
     "Android drives the loss.", "direction: wrong top segment"),
]


def _overlaps(quote: str, sentence: str) -> bool:
    q = " ".join(quote.split()).lower()
    s = " ".join(sentence.split()).lower()
    if not q:
        return False
    if q in s or s in q:
        return True
    # token overlap: most of the quote's distinctive tokens appear in the sentence
    qt = [t for t in q.replace("(", " ").replace(")", " ").replace(",", " ").split() if len(t) > 3]
    return bool(qt) and sum(t in s for t in qt) / len(qt) >= 0.6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the full report to this path")
    ap.add_argument("--threshold", type=float, default=0.8, help="catch rate to exit 0")
    ap.add_argument("--no-fail", action="store_true", help="always exit 0")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — this harness calls the LLM live.")
        return 2
    os.environ.setdefault("MAX_TOKENS_AUDIT", "4096")

    from agents.analyze.nodes_narrative import run_narrative_audit

    tool_json = json.dumps(TOOL_RESULTS, indent=2)

    # Clean narrative: false positives
    clean_res, _, clean_skip = run_narrative_audit(CLEAN, tool_json)
    clean_crit = [f for f in (clean_res.findings if clean_res else []) if f.severity == "critical"]
    false_positives = [{"quote": f.quote, "issue": f.issue} for f in clean_crit]

    results = []
    for original, replacement, what in MUTATIONS:
        assert original in CLEAN, original
        narrative = CLEAN.replace(original, replacement, 1)
        res, _, skipped = run_narrative_audit(narrative, tool_json)
        crit = [f for f in (res.findings if res else []) if f.severity == "critical"]
        caught = any(_overlaps(f.quote, replacement) for f in crit)
        results.append({
            "mutation": what, "planted": replacement, "caught": caught,
            "skipped": skipped, "critical_findings": [{"quote": f.quote, "issue": f.issue} for f in crit],
        })

    print("\nNarrative audit eval\n")
    for r in results:
        mark = "CAUGHT" if r["caught"] else "MISSED"
        extra = f"  ({r['skipped']})" if r["skipped"] else ""
        print(f"  {mark:<7} {r['mutation']:<40} {r['planted'][:60]}{extra}")
    catch_rate = sum(r["caught"] for r in results) / len(results)
    print(f"\n  catch rate       {catch_rate * 100:5.1f}%  ({sum(r['caught'] for r in results)}/{len(results)})")
    print(f"  false positives  {len(false_positives)} critical finding(s) on the clean narrative"
          + (f"  ({clean_skip})" if clean_skip else ""))
    for fp in false_positives:
        print(f"    - {fp['quote'][:70]!r}: {fp['issue']}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "catch_rate": catch_rate, "false_positives": false_positives,
            "clean_skipped": clean_skip, "results": results,
        }, indent=2))
    if args.no_fail:
        return 0
    return 0 if (catch_rate >= args.threshold and not false_positives) else 1


if __name__ == "__main__":
    sys.exit(main())
