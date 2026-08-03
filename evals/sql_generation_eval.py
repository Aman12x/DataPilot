"""Golden-question eval for the SQL-generation stage.

The four deterministic harnesses call the stats tools directly on DataFrames,
so intent routing, table choice, and SQL generation sit entirely outside the
regression gate — a change could regress SQL quality and every gated number
would stay green. This harness closes the SQL slice of that gap: it drives the
real `generate_sql` node (LLM live) against the demo DuckDB and scores each
question per stage rather than pass/fail end-to-end.

Stages, each scored independently:

  generated   the node returned non-empty SQL
  safe        `tools.db_tools.validate_sql` accepts it (SELECT-only etc.)
  tables      the SQL references the expected tables — and, for the wide-DB
              questions, none of the decoy tables
  executes    it runs against DuckDB read-only and returns rows
  answer      the result passes the question's shape/value predicate

Costs real LLM calls (~20 short generations), so it is NOT part of the per-PR
gate. Run manually or nightly:

    ./venv/bin/python evals/sql_generation_eval.py
    ./venv/bin/python evals/sql_generation_eval.py --json report.json
    ./venv/bin/python evals/sql_generation_eval.py --only wide  # table-choice slice

The wide-DB questions copy the demo tables into a scratch database alongside
plausible decoy tables (user_profiles, marketing_spend, …). They exist so the
future table-retrieval stage (future-work item 7) has a measurable before/after.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO_DB = str(ROOT / "data" / "dau_experiment.db")

DECOY_TABLES = [
    "user_profiles", "marketing_spend", "ab_flags",
    "sessions_archive", "orders", "support_tickets",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@dataclass
class Question:
    qid: str
    task: str
    mode: str                      # ab_test | general
    expected_tables: list[str]
    predicate: Callable[[pd.DataFrame], bool]
    predicate_desc: str
    wide: bool = False             # run against the decoy-laden wide DB
    forbidden_tables: list[str] = field(default_factory=list)


def _num(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes("number")


def _first_numeric(df: pd.DataFrame) -> float | None:
    nums = _num(df)
    if nums.empty or nums.shape[1] == 0:
        return None
    return float(nums.iloc[0, 0])


def _all_numeric_between(df: pd.DataFrame, lo: float, hi: float) -> bool:
    nums = _num(df)
    if nums.empty:
        return False
    vals = nums.to_numpy().ravel()
    return len(vals) > 0 and all(lo <= v <= hi for v in vals if pd.notna(v))


def _variant_comparison(df: pd.DataFrame) -> bool:
    """True for either shape the pipeline legitimately produces for an A/B
    question: a per-variant aggregate (>=2 rows) OR the canonical user-level
    extract (many rows, a column holding >=2 distinct group labels)."""
    if df.empty:
        return False
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() >= 2:
            return True
    return len(df) >= 2


QUESTIONS: list[Question] = [
    Question(
        "count_events", "How many rows are in the events table?", "general",
        ["events"],
        lambda df: (_first_numeric(df) or 0) == 284797,
        "single count equal to 284,797",
    ),
    Question(
        "variant_counts", "How many users are in each experiment variant?", "general",
        ["experiment"],
        lambda df: len(df) >= 2 and (_num(df).min().min() if not _num(df).empty else 0) > 1000,
        ">=2 rows with per-variant counts in the thousands",
    ),
    Question(
        "avg_dau", "What was the average daily active users per day over the whole period?", "general",
        ["metrics_daily"],
        lambda df: (_first_numeric(df) or 0) > 0,
        "a positive average",
    ),
    Question(
        # events or metrics_daily both answer this — table left open.
        "top_platform", "Which platform had the highest total DAU?", "general",
        [],
        lambda df: len(df) >= 1 and any(df[c].astype(str).isin(["web", "ios", "android"]).any() for c in df.columns if df[c].dtype == object),
        "names one of web/ios/android",
    ),
    Question(
        # The pipeline's canonical A/B SQL returns a user-level extract by
        # design (the stats layer aggregates); a per-variant aggregate is
        # equally correct.
        "dau_by_variant", "What is the average DAU rate for control versus treatment?", "ab_test",
        ["events", "experiment"],
        _variant_comparison,
        "per-variant aggregate or user-level extract with both variants",
    ),
    Question(
        "optout_by_variant", "What share of users opted out of notifications in each variant?", "ab_test",
        ["events", "experiment"],
        lambda df: len(df) >= 2,
        "one row per variant",
    ),
    Question(
        "funnel_steps", "How many users completed each funnel step?", "general",
        ["funnel"],
        lambda df: len(df) >= 3,
        "a row per funnel step (impression/click/install/d1_retain)",
    ),
    Question(
        "d7_by_variant", "What is the D7 retention rate for each experiment variant?", "ab_test",
        ["events", "experiment"],
        _variant_comparison,
        "per-variant aggregate or user-level extract with both variants",
    ),
    Question(
        # Either metrics_daily or an events aggregation answers this correctly,
        # so the table check is left open and the predicate carries the weight.
        "dau_over_time", "Show daily DAU over time for each platform.", "general",
        [],
        lambda df: len(df) > 10 and df.shape[1] >= 3,
        "many rows with date, platform, and a DAU value",
    ),
    Question(
        "sessions_new_vs_existing", "What is the average session count for new users versus existing users?", "general",
        ["events"],
        lambda df: len(df) >= 2,
        "a row per user group",
    ),
    Question(
        "notif_reach", "How many distinct users received at least one notification?", "general",
        ["events"],
        lambda df: 0 < (_first_numeric(df) or 0) <= 284797,
        "a distinct-user count below total rows",
    ),
    Question(
        "optout_by_segment", "Which user segment has the highest notification opt-out rate?", "general",
        ["events"],
        lambda df: len(df) >= 1,
        "at least one segment row",
    ),
    Question(
        "sessions_by_variant", "Compare average session counts between control and treatment.", "ab_test",
        ["events", "experiment"],
        lambda df: len(df) >= 2,
        "one row per variant",
    ),
    Question(
        "assignments_by_week", "How many users were assigned to the experiment each week?", "general",
        ["experiment"],
        lambda df: len(df) >= 1 and (_num(df).max().max() if not _num(df).empty else 0) > 0,
        "weekly assignment counts",
    ),
    Question(
        "install_rate", "What fraction of users completed the install step of the funnel?", "general",
        ["funnel"],
        lambda df: _all_numeric_between(df, 0, 1.0) or 0 < (_first_numeric(df) or 0) <= 1,
        "a fraction between 0 and 1",
    ),
    Question(
        "churn_by_date", "How many users churned on each date?", "general",
        ["metrics_daily"],
        lambda df: len(df) > 5,
        "a row per date",
    ),
    Question(
        # events.d7_retained or metrics_daily.d7_retention_rate both answer
        # this; the query may also include extra numeric columns (counts),
        # so require only that SOME numeric column looks like a rate.
        "d7_rate_by_platform", "What is the average D7 retention rate by platform?", "general",
        [],
        lambda df: len(df) >= 3 and any(
            _num(df)[c].between(0, 1).all() for c in _num(df).columns
        ),
        "three platform rows, at least one column of rates in [0, 1]",
    ),
    Question(
        "new_users_by_platform", "What are the total new users per platform?", "general",
        ["metrics_daily"],
        lambda df: len(df) >= 3,
        "a row per platform",
    ),
    # ── Wide-DB table selection: same questions, six decoy tables present ────
    Question(
        "wide_variant_counts", "How many users are in each experiment variant?", "general",
        ["experiment"],
        lambda df: len(df) >= 2,
        "uses the experiment table, not a decoy",
        wide=True, forbidden_tables=DECOY_TABLES,
    ),
    Question(
        "wide_dau_by_variant", "What is the average DAU rate for control versus treatment?", "ab_test",
        ["events", "experiment"],
        lambda df: len(df) >= 2,
        "joins events+experiment, ignores decoys",
        wide=True, forbidden_tables=DECOY_TABLES,
    ),
]


# ── Wide DB with decoy tables ────────────────────────────────────────────────

def build_wide_db(dest: str) -> None:
    """Copy the demo tables into `dest` and add plausible decoy tables."""
    con = duckdb.connect(dest)
    try:
        con.execute(f"ATTACH '{DEMO_DB}' AS demo (READ_ONLY)")
        for t in ("events", "experiment", "funnel", "metrics_daily"):
            con.execute(f"CREATE TABLE {t} AS SELECT * FROM demo.{t}")
        con.execute("DETACH demo")
        con.execute("""
            CREATE TABLE user_profiles AS
            SELECT DISTINCT user_id, 'name-' || user_id AS display_name,
                   'tier-' || (hash(user_id) % 3) AS plan_tier
            FROM events LIMIT 5000
        """)
        con.execute("""
            CREATE TABLE marketing_spend(date DATE, channel VARCHAR, spend_usd DOUBLE);
            INSERT INTO marketing_spend
            SELECT DISTINCT date, 'paid_social', 100.0 FROM events LIMIT 60
        """)
        con.execute("""
            CREATE TABLE ab_flags(flag_name VARCHAR, enabled BOOLEAN);
            INSERT INTO ab_flags VALUES ('new_onboarding', TRUE), ('dark_mode', FALSE)
        """)
        con.execute("CREATE TABLE sessions_archive AS SELECT user_id, date, session_count FROM events LIMIT 1000")
        con.execute("""
            CREATE TABLE orders(order_id INT, user_id VARCHAR, amount_usd DOUBLE);
            INSERT INTO orders SELECT 1, user_id, 9.99 FROM events LIMIT 500
        """)
        con.execute("""
            CREATE TABLE support_tickets(ticket_id INT, user_id VARCHAR, status VARCHAR);
            INSERT INTO support_tickets SELECT 1, user_id, 'open' FROM events LIMIT 200
        """)
    finally:
        con.close()


# ── Stage runners ─────────────────────────────────────────────────────────────

def _references(sql: str, table: str) -> bool:
    return re.search(rf"\b{re.escape(table)}\b", sql, re.IGNORECASE) is not None


def run_question(q: Question, db_path: str, schema_context: str) -> dict:
    from agents.analyze.nodes_sql import generate_sql
    from tools.db_tools import validate_sql

    state = {
        "task": q.task,
        "analysis_mode": q.mode,
        "db_backend": "duckdb",
        "duckdb_path": db_path,
        "schema_context": schema_context,
        "relevant_history": [],
        "user_id": "sql-eval",
    }
    stages = {"generated": False, "safe": False, "tables": False, "executes": False, "answer": False}
    detail = ""
    sql = ""
    t0 = time.perf_counter()
    try:
        out = generate_sql(state)
        sql = (out.get("generated_sql") or "").strip()
    except Exception as exc:  # noqa: BLE001 — record, don't crash the harness
        detail = f"generate_sql raised: {type(exc).__name__}"
    elapsed = time.perf_counter() - t0

    if sql:
        stages["generated"] = True
        try:
            validate_sql(sql)
            stages["safe"] = True
        except Exception as exc:  # noqa: BLE001
            detail = f"validate_sql: {exc}"

        expected_ok = all(_references(sql, t) for t in q.expected_tables)
        decoy_hit = [t for t in q.forbidden_tables if _references(sql, t)]
        stages["tables"] = expected_ok and not decoy_hit
        if not expected_ok:
            detail = detail or f"missing table(s): {[t for t in q.expected_tables if not _references(sql, t)]}"
        if decoy_hit:
            detail = detail or f"referenced decoy table(s): {decoy_hit}"

        if stages["safe"]:
            try:
                con = duckdb.connect(db_path, read_only=True)
                try:
                    df = con.execute(sql).df()
                finally:
                    con.close()
                stages["executes"] = True
                try:
                    stages["answer"] = bool(q.predicate(df))
                    if not stages["answer"]:
                        detail = detail or f"predicate failed ({q.predicate_desc}); shape={df.shape}"
                except Exception as exc:  # noqa: BLE001
                    detail = detail or f"predicate raised: {type(exc).__name__}"
            except Exception as exc:  # noqa: BLE001
                detail = detail or f"execution failed: {type(exc).__name__}"

    return {
        "qid": q.qid,
        "wide": q.wide,
        "stages": stages,
        "passed": all(stages.values()),
        "detail": detail,
        "sql": sql,
        "seconds": round(elapsed, 2),
    }


def _schema_context_for(db_path: str) -> str:
    """Build the same schema context `load_schema` would, without its caching."""
    from tools.db_tools import DBConnection

    return DBConnection("duckdb", path=db_path).inspect_schema()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the full report to this path")
    ap.add_argument("--only", choices=["demo", "wide"], help="run one slice")
    ap.add_argument("--threshold", type=float, default=0.8, help="strict pass-rate to exit 0")
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

    # Measure the shipped defaults, not local experiments: a stray
    # MAX_TOKENS_SQL=512 in .env once made this harness report truncation that
    # production would never see. Popping is not enough — node_shared calls
    # load_dotenv() at import and would re-read .env — but load_dotenv never
    # overrides an existing variable, so setting the shipped defaults here
    # (before the agents modules are imported) pins them. Values mirror
    # node_shared's defaults.
    for var, default in (
        ("MAX_TOKENS_SQL", "8192"),
        ("MAX_TOKENS_NARRATIVE", "8192"),
        ("MAX_TOKENS_AUDIT", "8192"),
    ):
        if os.environ.get(var, default) != default:
            print(f"note: overriding local {var}={os.environ[var]} — harness measures shipped defaults")
        os.environ[var] = default
    os.environ.pop("MODEL", None)

    questions = [q for q in QUESTIONS if args.only is None or (q.wide == (args.only == "wide"))]

    with tempfile.TemporaryDirectory() as tmp:
        wide_db = os.path.join(tmp, "wide.db")
        if any(q.wide for q in questions):
            build_wide_db(wide_db)

        contexts = {False: _schema_context_for(DEMO_DB)}
        if any(q.wide for q in questions):
            contexts[True] = _schema_context_for(wide_db)

        results = []
        for q in questions:
            r = run_question(q, wide_db if q.wide else DEMO_DB, contexts[q.wide])
            flag = "PASS" if r["passed"] else "FAIL"
            failed_stages = [k for k, v in r["stages"].items() if not v]
            print(f"[{flag}] {q.qid:26s} {r['seconds']:5.1f}s"
                  + (f"  failed: {','.join(failed_stages)}  {r['detail']}" if failed_stages else ""))
            results.append(r)

    n = len(results)
    stage_rates = {
        s: sum(r["stages"][s] for r in results) / n
        for s in ("generated", "safe", "tables", "executes", "answer")
    }
    strict = sum(r["passed"] for r in results) / n

    print("\n── SQL generation eval ──")
    print(f"model: {os.getenv('FAST_MODEL', 'claude-sonnet-5')}   questions: {n}")
    for s, rate in stage_rates.items():
        print(f"  {s:10s} {rate * 100:5.1f}%")
    print(f"  strict     {strict * 100:5.1f}%  (all stages)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": os.getenv("FAST_MODEL", "claude-sonnet-5"),
            "questions": n,
            "stage_rates": stage_rates,
            "strict": strict,
            "results": results,
        }, indent=2))
        print(f"report → {args.json}")

    if args.no_fail:
        return 0
    return 0 if strict >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
