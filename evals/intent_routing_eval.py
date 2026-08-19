"""Golden-task eval for intent routing.

    python evals/intent_routing_eval.py [--json report.json] [--threshold 0.85]

LLM-live (calls the real `_llm_resolve_intent` plus the heuristic override, on
the demo schema), so it is NOT in the per-PR gate — run manually or from the
Eval Nightly workflow. Scores three routing decisions per task:

  mode        analysis_mode the run lands in (general / ab_test / power_analysis)
  query_type  lookup vs exploratory — the lookup fast path skips both human
              gates and the audit, so a false "lookup" ships a bare table as
              an approved report; a false "exploratory" only costs latency
  metric      primary_metric (ab_test tasks only), where the task names one

`strict` is all applicable fields right. The threshold applies to strict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO_DB = ROOT / "data" / "dau_experiment.db"


@dataclass
class Task:
    task: str
    mode: str                       # expected analysis_mode
    query_type: str | None = None   # expected query_type (general only)
    metric: str | None = None       # expected primary_metric (ab_test only)
    note: str = ""
    alt_modes: list[str] = field(default_factory=list)   # also-acceptable modes


TASKS: list[Task] = [
    # ── plain lookups: fast path is correct ───────────────────────────────
    Task("How many users are in the experiment?", "general", "lookup"),
    Task("What was the total number of sessions last week?", "general", "lookup"),
    Task("List the top 5 platforms by number of users", "general", "lookup"),
    Task("What is the average session count?", "general", "lookup"),
    # ── opens like a lookup, is analysis: must NOT take the fast path ─────
    Task("What was the average dau_flag per user by variant?", "general", "exploratory",
         note="'by variant' is a cut; fast path would skip the gates", alt_modes=["ab_test"]),
    Task("How many users per platform, and how does that compare to last month?", "general", "exploratory"),
    Task("What is the total revenue vs last quarter?", "general", "exploratory"),
    Task("Show me session counts over time by user segment", "general", "exploratory"),
    # ── exploratory ───────────────────────────────────────────────────────
    Task("Why did DAU drop in the second week?", "general", "exploratory"),
    Task("Is there a relationship between notifications received and sessions?", "general", "exploratory"),
    Task("Find anomalies in daily active users", "general", "exploratory"),
    # ── A/B tests ─────────────────────────────────────────────────────────
    Task("Did the push-notification experiment lift DAU?", "ab_test", metric="dau_flag"),
    Task("Evaluate the A/B test: treatment vs control on session_count", "ab_test", metric="session_count"),
    Task("Did the treatment group retain better at day 7 than control?", "ab_test", metric="d7_retained"),
    Task("Run the experiment readout for notif_opened, checking guardrails", "ab_test", metric="notif_opened"),
    Task("Compare control and treatment on dau_flag, with CUPED", "ab_test", metric="dau_flag"),
    # ── power analysis ────────────────────────────────────────────────────
    Task("How many users do I need to detect a 2% lift in DAU at 80% power?", "power_analysis"),
    Task("What sample size is needed for a 5% MDE on session_count?", "power_analysis"),
    Task("How long must the experiment run to detect a 3% change in d7_retained?", "power_analysis"),
]


def _schema_context() -> str:
    from tools.db_tools import DBConnection
    return DBConnection("duckdb", path=str(DEMO_DB)).inspect_schema()


def run_task(t: Task, schema_context: str) -> dict:
    from agents.analyze import nodes_intent as ni
    from config.analysis_config import load_metric_config

    mc = load_metric_config()
    state = {
        "task": t.task,
        "schema_context": schema_context,
        "metric_config": mc,
        "db_backend": "duckdb",
        # ab_test is the default when a caller sets nothing; the node must
        # override it from the task, which is exactly what we are measuring.
        "analysis_mode": "",
    }
    out = ni.resolve_task_intent(state)
    got_mode = out.get("analysis_mode")
    got_qt = out.get("query_type")
    got_metric = out.get("metric")

    checks: dict[str, bool | None] = {
        "mode": got_mode == t.mode or got_mode in t.alt_modes,
        "query_type": (got_qt == t.query_type) if t.query_type else None,
        "metric": (got_metric == t.metric) if t.metric else None,
    }
    applicable = {k: v for k, v in checks.items() if v is not None}
    return {
        "task": t.task, "expected": {"mode": t.mode, "query_type": t.query_type, "metric": t.metric},
        "got": {"mode": got_mode, "query_type": got_qt, "metric": got_metric},
        "checks": checks, "passed": all(applicable.values()), "note": t.note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the full report to this path")
    ap.add_argument("--threshold", type=float, default=0.85, help="strict pass-rate to exit 0")
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
    if not DEMO_DB.exists():
        print(f"{DEMO_DB} missing — run data/generate_data.py")
        return 2
    os.environ.pop("MODEL", None)   # measure the shipped routing model

    schema_context = _schema_context()
    results = [run_task(t, schema_context) for t in TASKS]

    print("\nIntent routing eval\n")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        bad = [k for k, v in r["checks"].items() if v is False]
        extra = f"  wrong: {', '.join(bad)}  got={r['got']}" if bad else ""
        print(f"  {mark}  {r['task']}{extra}")

    n = len(results)
    def rate(key: str) -> float | None:
        vals = [r["checks"][key] for r in results if r["checks"][key] is not None]
        return sum(vals) / len(vals) if vals else None
    rates = {k: rate(k) for k in ("mode", "query_type", "metric")}
    strict = sum(r["passed"] for r in results) / n
    print("\nPer-field accuracy:")
    for k, v in rates.items():
        if v is not None:
            print(f"  {k:<11}{v * 100:5.1f}%")
    print(f"  strict     {strict * 100:5.1f}%  (all applicable fields)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"n": n, "strict": strict, "rates": rates, "results": results}, indent=2, default=str))
    if args.no_fail:
        return 0
    return 0 if strict >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
