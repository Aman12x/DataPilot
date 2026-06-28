"""
evals/transactions_eval.py — Golden Q&A eval for customer_transactions_10k.csv.

Validates that deterministic SQL answers match ground truth and that the
eval harness catches wrong narratives.

No API key required.

Usage:
    python evals/transactions_eval.py
    python evals/transactions_eval.py --json

Exit code: 0 if score >= 0.80, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import pandas as pd

from tools.eval_tools import score_faithfulness, score_key_findings

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "samples", "customer_transactions_10k.csv"
)
GT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "samples",
    "customer_transactions_10k_ground_truth.json",
)

# SQL templates keyed by ground-truth question id
_QA_SQL: dict[str, str] = {
    "which_category_highest_revenue": """
        SELECT product_category FROM transactions
        GROUP BY product_category
        ORDER BY SUM(total_transaction_amount_usd) DESC LIMIT 1
    """,
    "which_category_lowest_revenue": """
        SELECT product_category FROM transactions
        GROUP BY product_category
        ORDER BY SUM(total_transaction_amount_usd) ASC LIMIT 1
    """,
    "which_category_highest_return_rate": """
        SELECT product_category FROM transactions
        GROUP BY product_category
        ORDER BY AVG(is_returned) DESC LIMIT 1
    """,
    "which_category_lowest_return_rate": """
        SELECT product_category FROM transactions
        GROUP BY product_category
        ORDER BY AVG(is_returned) ASC LIMIT 1
    """,
    "which_payment_highest_fraud": """
        SELECT payment_method FROM transactions
        GROUP BY payment_method
        ORDER BY AVG(is_fraudulent) DESC LIMIT 1
    """,
    "which_payment_lowest_fraud": """
        SELECT payment_method FROM transactions
        GROUP BY payment_method
        ORDER BY AVG(is_fraudulent) ASC LIMIT 1
    """,
    "which_device_highest_fraud": """
        SELECT device_type FROM transactions
        GROUP BY device_type
        ORDER BY AVG(is_fraudulent) DESC LIMIT 1
    """,
    "which_device_lowest_fraud": """
        SELECT device_type FROM transactions
        GROUP BY device_type
        ORDER BY AVG(is_fraudulent) ASC LIMIT 1
    """,
    "which_loyalty_tier_highest_avg_txn": """
        SELECT customer_loyalty_tier FROM transactions
        GROUP BY customer_loyalty_tier
        ORDER BY AVG(total_transaction_amount_usd) DESC LIMIT 1
    """,
    "which_loyalty_tier_lowest_avg_txn": """
        SELECT customer_loyalty_tier FROM transactions
        GROUP BY customer_loyalty_tier
        ORDER BY AVG(total_transaction_amount_usd) ASC LIMIT 1
    """,
    "which_loyalty_tier_only_churned": """
        SELECT customer_loyalty_tier FROM (
            SELECT customer_loyalty_tier, AVG(is_churned_customer) AS cr
            FROM transactions GROUP BY customer_loyalty_tier
        ) WHERE cr > 0 ORDER BY cr DESC LIMIT 1
    """,
    "highest_revenue_country": """
        SELECT customer_country FROM transactions
        GROUP BY customer_country
        ORDER BY SUM(total_transaction_amount_usd) DESC LIMIT 1
    """,
    "highest_fraud_country": """
        SELECT customer_country FROM transactions
        GROUP BY customer_country
        ORDER BY AVG(is_fraudulent) DESC LIMIT 1
    """,
    "age_group_highest_avg_txn": """
        SELECT age_bucket FROM transactions
        GROUP BY age_bucket
        ORDER BY AVG(total_transaction_amount_usd) DESC LIMIT 1
    """,
    "age_group_lowest_avg_txn": """
        SELECT age_bucket FROM transactions
        GROUP BY age_bucket
        ORDER BY AVG(total_transaction_amount_usd) ASC LIMIT 1
    """,
}


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"""
        CREATE TABLE transactions AS
        SELECT *,
            CASE
                WHEN customer_age <= 25 THEN '18-25'
                WHEN customer_age <= 35 THEN '26-35'
                WHEN customer_age <= 50 THEN '36-50'
                WHEN customer_age <= 65 THEN '51-65'
                ELSE '66+'
            END AS age_bucket
        FROM read_csv_auto('{CSV_PATH}')
    """)
    return con


def _load_gt() -> dict[str, Any]:
    with open(GT_PATH) as f:
        return json.load(f)


# ── Criteria ──────────────────────────────────────────────────────────────────

def crit_qa_answers(con: duckdb.DuckDBPyConnection, gt: dict, verbose: bool) -> bool:
    """Each common_question_answers entry matches deterministic SQL."""
    answers = gt.get("common_question_answers", {})
    for key, expected in answers.items():
        if key.startswith("_"):
            continue
        sql = _QA_SQL.get(key)
        if not sql:
            continue
        actual = con.execute(sql).fetchone()[0]
        if verbose:
            print(f"    {key}: expected={expected!r} actual={actual!r}")
        if str(actual) != str(expected):
            return False
    return True


def crit_revenue_rank(con: duckdb.DuckDBPyConnection, gt: dict, verbose: bool) -> bool:
    rows = con.execute("""
        SELECT product_category FROM transactions
        GROUP BY product_category
        ORDER BY SUM(total_transaction_amount_usd) DESC
    """).fetchall()
    ranked = [r[0] for r in rows]
    expected = gt["critical_comparisons"]["revenue_rank_order"]
    if verbose:
        print(f"    ranked={ranked[:3]}... expected={expected[:3]}...")
    return ranked == expected


def crit_bronze_beats_platinum(con: duckdb.DuckDBPyConnection, gt: dict, verbose: bool) -> bool:
    rows = con.execute("""
        SELECT customer_loyalty_tier, AVG(total_transaction_amount_usd)
        FROM transactions
        WHERE customer_loyalty_tier IN ('Bronze', 'Platinum')
        GROUP BY customer_loyalty_tier
    """).fetchall()
    avgs = {r[0]: r[1] for r in rows}
    ok = avgs.get("Bronze", 0) > avgs.get("Platinum", 0)
    if verbose:
        print(f"    Bronze={avgs.get('Bronze'):.2f} Platinum={avgs.get('Platinum'):.2f} ok={ok}")
    return ok


def crit_mobile_fraud_ratio(con: duckdb.DuckDBPyConnection, gt: dict, verbose: bool) -> bool:
    rows = con.execute("""
        SELECT device_type, AVG(is_fraudulent)
        FROM transactions
        GROUP BY device_type
    """).fetchall()
    rates = {r[0]: r[1] for r in rows}
    mobile = rates.get("Mobile", 0)
    desktop = rates.get("Desktop", 0)
    if desktop <= 0:
        return False
    ratio = mobile / desktop
    expected = gt["critical_comparisons"]["mobile_vs_desktop_fraud_ratio"]
    ok = abs(ratio - expected) / expected <= 0.05
    if verbose:
        print(f"    mobile/desktop ratio={ratio:.2f} expected≈{expected} ok={ok}")
    return ok


def crit_faithfulness_passes(con: duckdb.DuckDBPyConnection, gt: dict, verbose: bool) -> bool:
    """A correct template narrative passes faithfulness against aggregate stats."""
    df = con.execute("SELECT * FROM transactions").df()
    overall = gt["overall"]
    narrative = (
        f"Overall fraud rate is {overall['fraud_rate']*100:.2f}% "
        f"({overall['fraud_count']} fraudulent transactions). "
        f"Average transaction amount is ${overall['avg_transaction_amount']:.2f}."
    )
    score = score_faithfulness(narrative, df)["score"]
    if verbose:
        print(f"    faithfulness={score:.3f}")
    return score >= 0.85


def crit_wrong_narrative_caught(con: duckdb.DuckDBPyConnection, gt: dict, verbose: bool) -> bool:
    """Eval harness flags a vague narrative that omits the correct top insight."""
    vague = "Fraud rates vary across payment channels without a clear leader."
    expected = gt["common_question_answers"]["which_payment_highest_fraud"]
    result = score_key_findings(vague, [expected])
    if verbose:
        print(f"    vague narrative key_findings={result['score']:.3f}")
    return result["score"] < 0.5


def crit_crypto_fraud_highest(con: duckdb.DuckDBPyConnection, gt: dict, verbose: bool) -> bool:
    row = con.execute("""
        SELECT payment_method FROM transactions
        GROUP BY payment_method
        ORDER BY AVG(is_fraudulent) DESC LIMIT 1
    """).fetchone()[0]
    expected = gt["common_question_answers"]["which_payment_highest_fraud"]
    if verbose:
        print(f"    highest fraud payment={row} expected={expected}")
    return row == expected


CRITERIA: list[tuple[str, str, Callable]] = [
    ("qa_answers",           "All 15 golden Q&A SQL answers match ground truth",     crit_qa_answers),
    ("revenue_rank",         "Product category revenue rank order is correct",       crit_revenue_rank),
    ("bronze_beats_platinum","Bronze avg txn value exceeds Platinum (counterintuitive)", crit_bronze_beats_platinum),
    ("mobile_fraud_ratio",   "Mobile fraud rate is ~11.4× Desktop",                   crit_mobile_fraud_ratio),
    ("crypto_fraud_highest", "Crypto is the highest-fraud payment method",           crit_crypto_fraud_highest),
    ("faithfulness_passes",  "Correct template narrative passes faithfulness",       crit_faithfulness_passes),
    ("wrong_narrative_caught","Wrong narrative is caught by faithfulness scorer",    crit_wrong_narrative_caught),
]


def run_eval(verbose: bool = False, quiet: bool = False) -> dict[str, Any]:
    gt = _load_gt()
    con = _connect()
    results: dict[str, dict[str, Any]] = {}
    passed = failed = 0

    for key, label, fn in CRITERIA:
        try:
            ok = fn(con, gt, verbose)
        except Exception:
            ok = False
            if verbose:
                traceback.print_exc()
        results[key] = {"description": label, "passed": ok}
        if not quiet:
            icon = "PASS" if ok else "FAIL"
            print(f"  {icon}  {label}")
        if ok:
            passed += 1
        else:
            failed += 1

    con.close()
    total = passed + failed
    score = passed / total if total else 0.0
    return {
        "criteria": results,
        "score": score,
        "n_pass": passed,
        "n_total": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Customer transactions golden Q&A eval")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", dest="json_out", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return 2

    if not args.json_out:
        print("Running transactions eval...")
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
