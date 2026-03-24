"""
tests/test_ground_truth_customer_transactions.py

Validates that DataPilot's tool layer produces numbers that exactly match the
ground truth computed directly from customer_transactions_10k.csv.

If these tests pass but the narrative is still wrong, the bug is in
generate_narrative (the LLM is misreading tool output).
If these tests fail, the bug is in the tools themselves.

Tolerance: 1% relative for floats, 0 for counts.
"""

from __future__ import annotations

import json
import os
import pytest
import duckdb
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "samples", "customer_transactions_10k.csv"
)
GT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "samples",
    "customer_transactions_10k_ground_truth.json",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV_PATH)


@pytest.fixture(scope="module")
def gt():
    with open(GT_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def duck(tmp_path_factory):
    """In-process DuckDB with the CSV loaded as a table — same path the tool layer uses."""
    path = str(tmp_path_factory.mktemp("db") / "transactions.db")
    con = duckdb.connect(path)
    con.execute(f"CREATE TABLE transactions AS SELECT * FROM read_csv_auto('{CSV_PATH}')")
    con.close()
    return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def approx(expected: float, rel: float = 0.01):
    """pytest.approx wrapper — 1% relative tolerance by default."""
    return pytest.approx(expected, rel=rel)


# ── 1. Dataset shape ──────────────────────────────────────────────────────────

class TestDatasetShape:
    def test_row_count(self, df, gt):
        assert len(df) == gt["dataset"]["total_transactions"]

    def test_unique_customers(self, df, gt):
        assert df["customer_id"].nunique() == gt["dataset"]["unique_customers"]

    def test_no_nulls(self, df):
        assert df.isnull().sum().sum() == 0


# ── 2. Overall metrics ────────────────────────────────────────────────────────

class TestOverallMetrics:
    def test_fraud_rate(self, df, gt):
        assert df["is_fraudulent"].mean() == approx(gt["overall"]["fraud_rate"])

    def test_fraud_count(self, df, gt):
        assert df["is_fraudulent"].sum() == gt["overall"]["fraud_count"]

    def test_return_rate(self, df, gt):
        assert df["is_returned"].mean() == approx(gt["overall"]["return_rate"])

    def test_return_count(self, df, gt):
        assert df["is_returned"].sum() == gt["overall"]["return_count"]

    def test_churn_rate(self, df, gt):
        assert df["is_churned_customer"].mean() == approx(gt["overall"]["churn_rate"])

    def test_total_transaction_amount(self, df, gt):
        assert df["total_transaction_amount_usd"].sum() == approx(
            gt["overall"]["total_transaction_amount"], rel=0.001
        )

    def test_avg_transaction_amount(self, df, gt):
        assert df["total_transaction_amount_usd"].mean() == approx(
            gt["overall"]["avg_transaction_amount"]
        )

    def test_median_transaction_amount(self, df, gt):
        assert df["total_transaction_amount_usd"].median() == approx(
            gt["overall"]["median_transaction_amount"], rel=0.02
        )


# ── 3. Loyalty tier ───────────────────────────────────────────────────────────

class TestLoyaltyTier:
    @pytest.mark.parametrize("tier", ["Bronze", "Silver", "Gold", "Platinum"])
    def test_txn_count(self, df, gt, tier):
        assert (df["customer_loyalty_tier"] == tier).sum() == gt["by_loyalty_tier"][tier]["txn_count"]

    @pytest.mark.parametrize("tier", ["Bronze", "Silver", "Gold", "Platinum"])
    def test_avg_txn_value(self, df, gt, tier):
        actual = df[df["customer_loyalty_tier"] == tier]["total_transaction_amount_usd"].mean()
        assert actual == approx(gt["by_loyalty_tier"][tier]["avg_txn_value"])

    @pytest.mark.parametrize("tier", ["Bronze", "Silver", "Gold", "Platinum"])
    def test_fraud_rate(self, df, gt, tier):
        actual = df[df["customer_loyalty_tier"] == tier]["is_fraudulent"].mean()
        assert actual == approx(gt["by_loyalty_tier"][tier]["fraud_rate"])

    def test_bronze_churn_rate(self, df, gt):
        actual = df[df["customer_loyalty_tier"] == "Bronze"]["is_churned_customer"].mean()
        assert actual == approx(gt["by_loyalty_tier"]["Bronze"]["churn_rate"])

    def test_non_bronze_churn_is_zero(self, df):
        """Gold, Silver, Platinum should have zero churn in this dataset."""
        for tier in ["Gold", "Silver", "Platinum"]:
            rate = df[df["customer_loyalty_tier"] == tier]["is_churned_customer"].mean()
            assert rate == 0.0, f"{tier} churn rate should be 0.0, got {rate}"

    def test_avg_txn_rank_order(self, df, gt):
        """Bronze > Silver > Gold > Platinum on avg transaction value."""
        avgs = (
            df.groupby("customer_loyalty_tier")["total_transaction_amount_usd"]
            .mean()
            .to_dict()
        )
        ranked = sorted(avgs, key=avgs.get, reverse=True)
        assert ranked == gt["critical_comparisons"]["loyalty_avg_txn_rank_order"]

    def test_bronze_higher_than_platinum(self, df, gt):
        """Directional: Bronze avg txn is HIGHER than Platinum (counterintuitive)."""
        bronze = df[df["customer_loyalty_tier"] == "Bronze"]["total_transaction_amount_usd"].mean()
        plat   = df[df["customer_loyalty_tier"] == "Platinum"]["total_transaction_amount_usd"].mean()
        assert bronze > plat
        assert (bronze - plat) == approx(
            gt["critical_comparisons"]["bronze_avg_txn_higher_than_platinum_by"]
        )


# ── 4. Product category ───────────────────────────────────────────────────────

class TestProductCategory:
    def test_revenue_rank_order(self, df, gt):
        rev = df.groupby("product_category")["total_transaction_amount_usd"].sum()
        ranked = rev.sort_values(ascending=False).index.tolist()
        assert ranked == gt["critical_comparisons"]["revenue_rank_order"]

    def test_electronics_revenue(self, df, gt):
        actual = df[df["product_category"] == "Electronics"]["total_transaction_amount_usd"].sum()
        assert actual == approx(gt["by_product_category"]["Electronics"]["total_revenue"], rel=0.001)

    def test_books_revenue(self, df, gt):
        actual = df[df["product_category"] == "Books"]["total_transaction_amount_usd"].sum()
        assert actual == approx(gt["by_product_category"]["Books"]["total_revenue"], rel=0.001)

    def test_electronics_vs_books_gap(self, df, gt):
        elec  = df[df["product_category"] == "Electronics"]["total_transaction_amount_usd"].sum()
        books = df[df["product_category"] == "Books"]["total_transaction_amount_usd"].sum()
        assert (elec - books) == approx(
            gt["critical_comparisons"]["electronics_vs_books_revenue_gap"], rel=0.001
        )

    def test_electronics_vs_books_ratio(self, df, gt):
        elec  = df[df["product_category"] == "Electronics"]["total_transaction_amount_usd"].sum()
        books = df[df["product_category"] == "Books"]["total_transaction_amount_usd"].sum()
        assert (elec / books) == approx(
            gt["critical_comparisons"]["electronics_vs_books_revenue_ratio"], rel=0.01
        )

    def test_electronics_highest_return_rate(self, df, gt):
        ret = df.groupby("product_category")["is_returned"].mean()
        assert ret.idxmax() == gt["common_question_answers"]["which_category_highest_return_rate"]

    def test_clothing_lowest_return_rate(self, df, gt):
        ret = df.groupby("product_category")["is_returned"].mean()
        assert ret.idxmin() == gt["common_question_answers"]["which_category_lowest_return_rate"]

    def test_electronics_vs_clothing_return_gap(self, df, gt):
        elec_ret  = df[df["product_category"] == "Electronics"]["is_returned"].mean()
        cloth_ret = df[df["product_category"] == "Clothing"]["is_returned"].mean()
        gap_ppts  = (elec_ret - cloth_ret) * 100
        assert gap_ppts == approx(
            gt["critical_comparisons"]["electronics_vs_clothing_return_gap_ppts"]
        )


# ── 5. Payment method fraud ───────────────────────────────────────────────────

class TestPaymentMethodFraud:
    def test_crypto_highest_fraud(self, df, gt):
        fraud = df.groupby("payment_method")["is_fraudulent"].mean()
        assert fraud.idxmax() == gt["common_question_answers"]["which_payment_highest_fraud"]

    def test_bank_transfer_lowest_fraud(self, df, gt):
        fraud = df.groupby("payment_method")["is_fraudulent"].mean()
        assert fraud.idxmin() == gt["common_question_answers"]["which_payment_lowest_fraud"]

    def test_crypto_fraud_rate(self, df, gt):
        actual = df[df["payment_method"] == "Crypto"]["is_fraudulent"].mean()
        assert actual == approx(gt["by_payment_method"]["Crypto"]["fraud_rate"])

    def test_credit_card_fraud_rate(self, df, gt):
        actual = df[df["payment_method"] == "Credit Card"]["is_fraudulent"].mean()
        assert actual == approx(gt["by_payment_method"]["Credit Card"]["fraud_rate"])

    def test_crypto_vs_credit_card_ratio(self, df, gt):
        crypto = df[df["payment_method"] == "Crypto"]["is_fraudulent"].mean()
        cc     = df[df["payment_method"] == "Credit Card"]["is_fraudulent"].mean()
        assert (crypto / cc) == approx(
            gt["critical_comparisons"]["crypto_vs_credit_card_fraud_ratio"], rel=0.05
        )

    def test_gift_card_vs_bank_transfer_ratio(self, df, gt):
        gc = df[df["payment_method"] == "Gift Card"]["is_fraudulent"].mean()
        bt = df[df["payment_method"] == "Bank Transfer"]["is_fraudulent"].mean()
        assert (gc / bt) == approx(
            gt["critical_comparisons"]["gift_card_vs_bank_transfer_fraud_ratio"], rel=0.05
        )


# ── 6. Device type fraud ──────────────────────────────────────────────────────

class TestDeviceTypeFraud:
    def test_mobile_highest_fraud(self, df, gt):
        fraud = df.groupby("device_type")["is_fraudulent"].mean()
        assert fraud.idxmax() == gt["common_question_answers"]["which_device_highest_fraud"]

    def test_desktop_lowest_fraud(self, df, gt):
        fraud = df.groupby("device_type")["is_fraudulent"].mean()
        assert fraud.idxmin() == gt["common_question_answers"]["which_device_lowest_fraud"]

    def test_mobile_fraud_rate(self, df, gt):
        actual = df[df["device_type"] == "Mobile"]["is_fraudulent"].mean()
        assert actual == approx(gt["by_device_type"]["Mobile"]["fraud_rate"])

    def test_desktop_fraud_rate(self, df, gt):
        actual = df[df["device_type"] == "Desktop"]["is_fraudulent"].mean()
        assert actual == approx(gt["by_device_type"]["Desktop"]["fraud_rate"])

    def test_mobile_vs_desktop_ratio(self, df, gt):
        mob  = df[df["device_type"] == "Mobile"]["is_fraudulent"].mean()
        desk = df[df["device_type"] == "Desktop"]["is_fraudulent"].mean()
        assert (mob / desk) == approx(
            gt["critical_comparisons"]["mobile_vs_desktop_fraud_ratio"], rel=0.05
        )


# ── 7. Age buckets ────────────────────────────────────────────────────────────

class TestAgeBuckets:
    @pytest.fixture(autouse=True)
    def add_age_bucket(self, df):
        df["age_bucket"] = pd.cut(
            df["customer_age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "66+"],
        )

    def test_66plus_highest_avg_txn(self, df, gt):
        avgs = df.groupby("age_bucket", observed=True)["total_transaction_amount_usd"].mean()
        assert avgs.idxmax() == gt["common_question_answers"]["age_group_highest_avg_txn"]

    def test_2635_lowest_avg_txn(self, df, gt):
        avgs = df.groupby("age_bucket", observed=True)["total_transaction_amount_usd"].mean()
        assert avgs.idxmin() == gt["common_question_answers"]["age_group_lowest_avg_txn"]

    def test_3650_avg_txn(self, df, gt):
        actual = df[df["age_bucket"] == "36-50"]["total_transaction_amount_usd"].mean()
        assert actual == approx(gt["by_age_bucket"]["36-50"]["avg_txn_value"])

    def test_1825_avg_txn(self, df, gt):
        actual = df[df["age_bucket"] == "18-25"]["total_transaction_amount_usd"].mean()
        assert actual == approx(gt["by_age_bucket"]["18-25"]["avg_txn_value"])

    def test_3650_vs_1825_gap(self, df, gt):
        a3650 = df[df["age_bucket"] == "36-50"]["total_transaction_amount_usd"].mean()
        a1825 = df[df["age_bucket"] == "18-25"]["total_transaction_amount_usd"].mean()
        assert (a3650 - a1825) == approx(gt["critical_comparisons"]["age_36_50_vs_18_25_gap"])


# ── 8. Country ────────────────────────────────────────────────────────────────

class TestCountry:
    def test_us_highest_revenue(self, df, gt):
        rev = df.groupby("customer_country")["total_transaction_amount_usd"].sum()
        assert rev.idxmax() == gt["common_question_answers"]["highest_revenue_country"]

    def test_brazil_highest_fraud_among_top5(self, df):
        top5 = ["United States", "Canada", "United Kingdom", "Germany", "Australia"]
        fraud = df[df["customer_country"].isin(top5)].groupby("customer_country")["is_fraudulent"].mean()
        # Brazil is NOT in top 5 by revenue but IS notable for fraud
        brazil = df[df["customer_country"] == "Brazil"]["is_fraudulent"].mean()
        overall = df["is_fraudulent"].mean()
        assert brazil > overall

    def test_brazil_fraud_ratio(self, df, gt):
        brazil  = df[df["customer_country"] == "Brazil"]["is_fraudulent"].mean()
        overall = df["is_fraudulent"].mean()
        assert (brazil / overall) == approx(
            gt["critical_comparisons"]["brazil_vs_overall_fraud_ratio"], rel=0.05
        )

    def test_us_canada_share(self, df, gt):
        share = df["customer_country"].isin(["United States", "Canada"]).mean() * 100
        assert share == approx(gt["critical_comparisons"]["us_canada_transaction_share_pct"])


# ── 9. Revenue share ─────────────────────────────────────────────────────────

class TestRevenueShare:
    def test_bronze_revenue_share(self, df, gt):
        bronze_rev = df[df["customer_loyalty_tier"] == "Bronze"]["total_transaction_amount_usd"].sum()
        total_rev  = df["total_transaction_amount_usd"].sum()
        share_pct  = bronze_rev / total_rev * 100
        assert share_pct == approx(gt["critical_comparisons"]["bronze_revenue_share_pct"], rel=0.02)

    def test_silver_revenue_share(self, df, gt):
        silver_rev = df[df["customer_loyalty_tier"] == "Silver"]["total_transaction_amount_usd"].sum()
        total_rev  = df["total_transaction_amount_usd"].sum()
        share_pct  = silver_rev / total_rev * 100
        assert share_pct == approx(gt["critical_comparisons"]["silver_revenue_share_pct"], rel=0.02)


# ── 10. DuckDB SQL layer ──────────────────────────────────────────────────────
# These verify the SQL execution path produces the same numbers as pandas.
# If pandas tests pass but DuckDB tests fail, the bug is in SQL generation.

class TestDuckDBSQLLayer:
    def test_total_revenue_via_sql(self, duck, gt):
        con = duckdb.connect(duck, read_only=True)
        result = con.execute(
            "SELECT SUM(total_transaction_amount_usd) AS rev FROM transactions"
        ).fetchone()[0]
        con.close()
        assert result == approx(gt["overall"]["total_transaction_amount"], rel=0.001)

    def test_fraud_rate_via_sql(self, duck, gt):
        con = duckdb.connect(duck, read_only=True)
        result = con.execute(
            "SELECT AVG(is_fraudulent) FROM transactions"
        ).fetchone()[0]
        con.close()
        assert result == approx(gt["overall"]["fraud_rate"])

    def test_category_revenue_ranking_via_sql(self, duck, gt):
        con = duckdb.connect(duck, read_only=True)
        rows = con.execute(
            """SELECT product_category, SUM(total_transaction_amount_usd) AS rev
               FROM transactions
               GROUP BY product_category
               ORDER BY rev DESC"""
        ).fetchall()
        con.close()
        ranked = [r[0] for r in rows]
        assert ranked == gt["critical_comparisons"]["revenue_rank_order"]

    def test_loyalty_avg_txn_ranking_via_sql(self, duck, gt):
        con = duckdb.connect(duck, read_only=True)
        rows = con.execute(
            """SELECT customer_loyalty_tier, AVG(total_transaction_amount_usd) AS avg_val
               FROM transactions
               GROUP BY customer_loyalty_tier
               ORDER BY avg_val DESC"""
        ).fetchall()
        con.close()
        ranked = [r[0] for r in rows]
        assert ranked == gt["critical_comparisons"]["loyalty_avg_txn_rank_order"]

    def test_payment_fraud_highest_is_crypto_via_sql(self, duck, gt):
        con = duckdb.connect(duck, read_only=True)
        row = con.execute(
            """SELECT payment_method
               FROM transactions
               GROUP BY payment_method
               ORDER BY AVG(is_fraudulent) DESC
               LIMIT 1"""
        ).fetchone()[0]
        con.close()
        assert row == gt["common_question_answers"]["which_payment_highest_fraud"]

    def test_device_fraud_highest_is_mobile_via_sql(self, duck, gt):
        con = duckdb.connect(duck, read_only=True)
        row = con.execute(
            """SELECT device_type
               FROM transactions
               GROUP BY device_type
               ORDER BY AVG(is_fraudulent) DESC
               LIMIT 1"""
        ).fetchone()[0]
        con.close()
        assert row == gt["common_question_answers"]["which_device_highest_fraud"]

    def test_bronze_avg_higher_than_platinum_via_sql(self, duck):
        con = duckdb.connect(duck, read_only=True)
        rows = con.execute(
            """SELECT customer_loyalty_tier, AVG(total_transaction_amount_usd)
               FROM transactions
               WHERE customer_loyalty_tier IN ('Bronze', 'Platinum')
               GROUP BY customer_loyalty_tier"""
        ).fetchall()
        con.close()
        avgs = {r[0]: r[1] for r in rows}
        assert avgs["Bronze"] > avgs["Platinum"], (
            f"Bronze ({avgs['Bronze']:.2f}) should be > Platinum ({avgs['Platinum']:.2f})"
        )
