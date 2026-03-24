"""
tests/test_eval_tools.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for tools/eval_tools.py.

Three modules tested:
  1. score_faithfulness  — number extraction + DataFrame matching
  2. score_relevancy     — MiniLM embedding similarity (marked @slow)
  3. score_key_findings  — keyword presence check
  4. evaluate_run        — composite scorer
  5. evaluate_fixture    — fixture registry integration

All tests that require the MiniLM model are marked @pytest.mark.slow.
Fast tests use only pandas, regex, and known string inputs.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.eval_tools import (
    EvalResult,
    FIXTURE_GROUND_TRUTH,
    _extract_narrative_numbers,
    check_magnitude_claims,
    evaluate_fixture,
    evaluate_run,
    score_faithfulness,
    score_key_findings,
    score_relevancy,
)

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ═════════════════════════════════════════════════════════════════════════════
# 1. _extract_narrative_numbers
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractNarrativeNumbers:
    def test_extracts_percentages(self):
        nums = _extract_narrative_numbers("The readmission rate was 30.6% for Diabetes patients.")
        assert 30.6 in nums

    def test_extracts_dollar_amounts(self):
        nums = _extract_narrative_numbers("Revenue grew to $334,378 by December.")
        assert 334378.0 in nums

    def test_extracts_bare_decimals(self):
        nums = _extract_narrative_numbers("Average BMI was 32.685 in the Diabetes cohort.")
        assert 32.685 in nums

    def test_excludes_years(self):
        nums = _extract_narrative_numbers("Data collected from 2015 to 2024.")
        assert 2015 not in nums
        assert 2024 not in nums

    def test_excludes_small_ordinals(self):
        """Single-digit integers used as bullet indices or ordinals should be excluded."""
        nums = _extract_narrative_numbers("3 key findings: 1. Revenue grew. 2. Churn fell.")
        assert 1 not in nums
        assert 2 not in nums
        assert 3 not in nums

    def test_empty_narrative(self):
        assert _extract_narrative_numbers("") == []

    def test_narrative_with_no_numbers(self):
        assert _extract_narrative_numbers("Revenue grew significantly last quarter.") == []

    def test_multiple_number_formats(self):
        text = "The churn rate dropped from 8.2% to 3.46%, while revenue hit $103,812."
        nums = _extract_narrative_numbers(text)
        assert 8.2 in nums
        assert 3.46 in nums
        assert 103812.0 in nums


# ═════════════════════════════════════════════════════════════════════════════
# 2. score_faithfulness
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreFaithfulness:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "bmi":           [32.685, 26.154, 23.603, 26.077],
            "readmission":   [0.3059, 0.1286, 0.0533, 0.1143],
            "systolic_bp":   [122.847, 158.057, 116.547, 119.314],
        })

    def test_perfect_faithfulness(self, df):
        """All numbers in narrative appear in DataFrame."""
        narrative = "Diabetes patients had an average BMI of 32.685 and readmission rate of 30.59%."
        result = score_faithfulness(narrative, df)
        assert result["score"] == 1.0

    def test_hallucinated_number_reduces_score(self, df):
        """A made-up number (99.99) not in the DataFrame reduces score."""
        narrative = "The average BMI was 99.99 and churn rate was 32.685."
        result = score_faithfulness(narrative, df)
        assert result["score"] < 1.0
        assert 99.99 in result["unsupported_numbers"]

    def test_within_tolerance(self, df):
        """Numbers within 10% tolerance count as supported."""
        # 33.0 is within 10% of 32.685
        narrative = "Average BMI was approximately 33.0."
        result = score_faithfulness(narrative, df, tolerance=0.10)
        assert result["score"] == 1.0

    def test_outside_tolerance_flagged(self, df):
        """Numbers >10% off from any DataFrame value are flagged."""
        narrative = "The average BMI was 50.0."  # far from any value
        result = score_faithfulness(narrative, df, tolerance=0.10)
        assert result["score"] < 1.0
        assert 50.0 in result["unsupported_numbers"]

    def test_no_numbers_in_narrative(self, df):
        """Narrative with no numbers → perfect score (nothing to be wrong about)."""
        narrative = "Revenue grew significantly across all patient groups."
        result = score_faithfulness(narrative, df)
        assert result["score"] == 1.0
        assert result["note"] == "no numbers in narrative"

    def test_none_df_returns_perfect(self):
        """No DataFrame provided → can't assess, give benefit of doubt."""
        result = score_faithfulness("BMI was 32.685.", None)
        assert result["score"] == 1.0

    def test_empty_df_returns_perfect(self):
        result = score_faithfulness("BMI was 32.685.", pd.DataFrame())
        assert result["score"] == 1.0

    def test_all_unsupported(self, df):
        """Every number in the narrative is made up."""
        narrative = "BMI was 99.0, readmission was 88.0%, BP was 999.0."
        result = score_faithfulness(narrative, df)
        assert result["score"] == 0.0
        assert result["total"] == 3
        assert len(result["unsupported_numbers"]) == 3

    def test_partial_support(self, df):
        """One real number and one hallucinated → score = 0.5."""
        narrative = "BMI was 32.685 and mystery score was 999.0."
        result = score_faithfulness(narrative, df)
        assert result["supported"] == 1
        assert result["total"] == 2
        assert abs(result["score"] - 0.5) < 0.01

    def test_returns_correct_keys(self, df):
        result = score_faithfulness("BMI was 32.685.", df)
        assert "score" in result
        assert "supported" in result
        assert "total" in result
        assert "unsupported_numbers" in result

    def test_timeseries_revenue(self):
        """Revenue values from timeseries fixture are faithfully cited."""
        ts = pd.read_csv(os.path.join(_FIXTURES, "timeseries.csv"))
        narrative = "Revenue started at $103,812 and grew to $334,378 over the period."
        result = score_faithfulness(narrative, ts)
        assert result["score"] >= 0.8, f"Expected faithfulness ≥ 0.8, got {result}"

    def test_hr_salary_faithful(self):
        """Engineering salary ~88128 should be supported by the HR fixture."""
        hr = pd.read_csv(os.path.join(_FIXTURES, "hr.csv"))
        narrative = "Engineering earned an average of $88,128, the highest across all departments."
        result = score_faithfulness(narrative, hr)
        assert result["score"] >= 0.5, f"Expected faithfulness ≥ 0.5, got {result}"


# ═════════════════════════════════════════════════════════════════════════════
# 3. score_key_findings
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreKeyFindings:
    def test_all_present(self):
        result = score_key_findings(
            "Diabetes patients had the highest BMI and readmission rate.",
            ["diabetes", "bmi", "readmission"],
        )
        assert result["score"] == 1.0
        assert result["found"] == 3
        assert result["missing"] == []

    def test_none_present(self):
        result = score_key_findings(
            "Revenue grew consistently over the period.",
            ["diabetes", "hypertension", "asthma"],
        )
        assert result["score"] == 0.0
        assert result["found"] == 0

    def test_partial_presence(self):
        result = score_key_findings(
            "Diabetes showed the highest BMI.",
            ["diabetes", "bmi", "hypertension"],
        )
        assert abs(result["score"] - 2/3) < 0.01
        assert "hypertension" in result["missing"]

    def test_case_insensitive(self):
        result = score_key_findings(
            "DIABETES patients had HIGH BMI.",
            ["diabetes", "bmi"],
        )
        assert result["score"] == 1.0

    def test_empty_expected_returns_perfect(self):
        result = score_key_findings("anything", [])
        assert result["score"] == 1.0

    def test_multiword_phrase(self):
        result = score_key_findings(
            "Engineering had the highest average salary.",
            ["highest average salary"],
        )
        assert result["score"] == 1.0

    def test_missing_list_populated(self):
        result = score_key_findings(
            "Revenue grew.",
            ["revenue", "churn", "growth"],
        )
        assert "churn" in result["missing"]
        assert "growth" in result["missing"]
        assert "revenue" not in result["missing"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. evaluate_run — composite scorer
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluateRun:
    @pytest.fixture
    def hr_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(_FIXTURES, "hr.csv"))

    def test_returns_eval_result(self, hr_df):
        result = evaluate_run(
            task="What is the salary breakdown by department?",
            narrative="Engineering had the highest average salary of $88,128.",
            df=hr_df,
            ground_truth_findings=["engineering", "salary"],
        )
        assert isinstance(result, EvalResult)

    def test_score_is_in_range(self, hr_df):
        result = evaluate_run(
            task="Salary by department",
            narrative="Engineering earned $88,128 on average, far above Marketing at $62,186.",
            df=hr_df,
            ground_truth_findings=["engineering", "marketing"],
        )
        assert 0.0 <= result.score <= 1.0

    def test_perfect_narrative_scores_high(self, hr_df):
        """A faithful, relevant narrative with all key findings should score well."""
        narrative = (
            "Engineering had the highest average salary at $88,128, followed by Sales at $66,129. "
            "Marketing had the lowest department average. Lead-level employees earned $105,518 on "
            "average, while Junior staff earned $46,532. The salary gap between levels is striking."
        )
        result = evaluate_run(
            task="What is the salary breakdown by department and level?",
            narrative=narrative,
            df=hr_df,
            ground_truth_findings=["engineering", "salary", "lead"],
        )
        assert result.faithfulness >= 0.5, f"Faithfulness too low: {result.faithfulness}"
        assert result.key_findings == 1.0

    def test_hallucinated_narrative_scores_low_faithfulness(self, hr_df):
        """A narrative with made-up numbers should score low on faithfulness."""
        narrative = (
            "Engineering averaged $999,999, Marketing averaged $888,888, "
            "and Junior staff earned $777,777."
        )
        result = evaluate_run(
            task="What is the salary by department?",
            narrative=narrative,
            df=hr_df,
            ground_truth_findings=["engineering"],
        )
        assert result.faithfulness < 0.5, f"Expected low faithfulness, got {result.faithfulness}"

    def test_missing_findings_reduces_key_findings(self, hr_df):
        """A narrative that omits key findings scores 0 on key_findings."""
        narrative = "The data shows some variation across groups."
        result = evaluate_run(
            task="What is the salary by department?",
            narrative=narrative,
            df=hr_df,
            ground_truth_findings=["engineering", "salary", "marketing"],
        )
        assert result.key_findings == 0.0

    def test_no_df_defaults_faithfulness_to_1(self):
        """When no DataFrame is provided, faithfulness defaults to 1.0."""
        result = evaluate_run(
            task="Revenue analysis",
            narrative="Revenue grew to $334,378.",
            df=None,
            ground_truth_findings=["revenue"],
        )
        assert result.faithfulness == 1.0

    def test_empty_narrative_handled_gracefully(self, hr_df):
        result = evaluate_run(task="salary analysis", narrative="", df=hr_df)
        assert 0.0 <= result.score <= 1.0

    def test_composite_weights_faithfulness_most(self, hr_df):
        """Faithfulness has weight 0.5 — zero faithfulness should drag the score down."""
        # All numbers hallucinated → faithfulness = 0
        narrative = "Department averages: Dept A $999k, Dept B $888k, Dept C $777k, Dept D $666k."
        result = evaluate_run(
            task="salary by department",
            narrative=narrative,
            df=hr_df,
            ground_truth_findings=[],
        )
        # With faith=0.0 and weight 0.5, composite ≤ 0.5
        assert result.score <= 0.5 + 0.01, f"Expected score ≤ 0.5, got {result.score}"

    def test_details_dict_populated(self, hr_df):
        result = evaluate_run("task", "narrative", hr_df)
        assert "faithfulness" in result.details
        assert "key_findings" in result.details
        assert "relevancy" in result.details
        assert "weights" in result.details


# ═════════════════════════════════════════════════════════════════════════════
# 5. evaluate_fixture — fixture registry
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluateFixture:
    def test_healthcare_fixture_loaded(self):
        assert "healthcare" in FIXTURE_GROUND_TRUTH
        assert "diabetes" in FIXTURE_GROUND_TRUTH["healthcare"]

    def test_timeseries_fixture_loaded(self):
        assert "timeseries" in FIXTURE_GROUND_TRUTH
        assert "revenue" in FIXTURE_GROUND_TRUTH["timeseries"]

    def test_hr_fixture_loaded(self):
        assert "hr" in FIXTURE_GROUND_TRUTH
        assert "engineering" in FIXTURE_GROUND_TRUTH["hr"]

    def test_ab_test_fixture_loaded(self):
        assert "ab_test" in FIXTURE_GROUND_TRUTH
        assert "android" in FIXTURE_GROUND_TRUTH["ab_test"]

    def test_evaluate_fixture_healthcare(self):
        """A narrative about Diabetes and readmission should score well for healthcare fixture."""
        hr_df = pd.read_csv(os.path.join(_FIXTURES, "healthcare.csv"))
        narrative = (
            "Diabetes patients showed the highest readmission rate at 30.6% and the "
            "highest average BMI of 32.7. Hypertension patients had elevated blood pressure. "
            "The findings highlight significant variation by diagnosis."
        )
        result = evaluate_fixture(
            "healthcare",
            task="Analyse patient outcomes by diagnosis",
            narrative=narrative,
            df=hr_df,
        )
        assert result.key_findings >= 0.75, \
            f"Expected key_findings ≥ 0.75 for good healthcare narrative, got {result.key_findings}"

    def test_evaluate_fixture_unknown_name_uses_empty_findings(self):
        """Unknown fixture name falls back to empty findings list → key_findings = 1.0."""
        result = evaluate_fixture("unknown_fixture", "task", "any narrative", df=None)
        assert result.key_findings == 1.0

    def test_evaluate_fixture_hr_wrong_narrative(self):
        """A narrative about the wrong topic should miss the hr key findings."""
        result = evaluate_fixture(
            "hr",
            task="salary by department",
            narrative="The sky is blue and revenue grew nicely last quarter.",
            df=None,
        )
        assert result.key_findings < 0.5, \
            f"Expected low key_findings for irrelevant narrative, got {result.key_findings}"


# ═════════════════════════════════════════════════════════════════════════════
# 6. score_relevancy (slow — requires MiniLM model)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestScoreRelevancy:
    def test_identical_task_and_narrative_high_relevancy(self):
        text = "What is the average salary by department?"
        sim = score_relevancy(text, text)
        assert sim >= 0.99

    def test_relevant_narrative_scores_above_threshold(self):
        task = "What is the salary breakdown by department?"
        narrative = (
            "Engineering had the highest average salary at $88,128, well above the company "
            "average. Marketing had the lowest at $62,186. The pay gap reflects different "
            "skill demands and market rates across departments."
        )
        sim = score_relevancy(task, narrative)
        assert sim >= 0.30, f"Expected relevancy ≥ 0.30 for on-topic narrative, got {sim}"

    def test_irrelevant_narrative_scores_below_relevant(self):
        task = "What is the salary breakdown by department?"
        unrelated = (
            "The weather forecast shows heavy rain tomorrow with temperatures dropping "
            "to 45°F. Consider bringing an umbrella and warm clothing."
        )
        on_topic = "Engineering earned $88,128, Marketing earned $62,186."
        sim_unrelated = score_relevancy(task, unrelated)
        sim_on_topic  = score_relevancy(task, on_topic)
        assert sim_on_topic > sim_unrelated, \
            f"On-topic ({sim_on_topic:.3f}) should be more relevant than unrelated ({sim_unrelated:.3f})"

    def test_different_domains_low_similarity(self):
        task = "Analyse DAU drop in android users."
        narrative = "Patient readmission rates were highest in Diabetes cohort."
        sim = score_relevancy(task, narrative)
        assert sim < 0.80, f"Cross-domain similarity should be low, got {sim}"

    def test_returns_float(self):
        sim = score_relevancy("task", "narrative")
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

    def test_evaluate_run_uses_relevancy(self):
        """evaluate_run composite should reflect relevancy score."""
        task = "What is the average salary by department?"
        on_topic  = "Engineering had the highest salary at $88,128. Marketing was lowest."
        off_topic = "The sky is blue and it rained yesterday."
        hr_df = pd.read_csv(os.path.join(_FIXTURES, "hr.csv"))

        result_on  = evaluate_run(task, on_topic,  hr_df)
        result_off = evaluate_run(task, off_topic, hr_df)
        assert result_on.relevancy > result_off.relevancy, \
            "On-topic narrative should have higher relevancy than off-topic"


# ═════════════════════════════════════════════════════════════════════════════
# 6. check_magnitude_claims
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckMagnitudeClaims:

    # ── "more than double" ────────────────────────────────────────────────────

    def test_more_than_double_wrong_ratio_flagged(self):
        """The exact bug: $322,996 is NOT more than double $248,410 (ratio=1.30)."""
        narrative = (
            "The Bronze segment produces $322,996 — more than double the next-largest "
            "segment (Silver at $248,410)."
        )
        result = check_magnitude_claims(narrative)
        assert len(result["violations"]) == 1
        assert "more than double" in result["violations"][0].lower()
        assert "1.3" in result["violations"][0]

    def test_more_than_double_correct_ratio_passes(self):
        """$500,000 vs $200,000 = 2.5x — genuinely more than double."""
        narrative = "Segment A generates $500,000 — more than double Segment B at $200,000."
        result = check_magnitude_claims(narrative)
        assert result["violations"] == []

    def test_more_than_double_exactly_two_flagged(self):
        """Exactly 2.0x is NOT 'more than double'."""
        narrative = "Revenue of $400,000 is more than double the $200,000 from last year."
        result = check_magnitude_claims(narrative)
        assert len(result["violations"]) == 1

    # ── "double" / "twice" ────────────────────────────────────────────────────

    def test_double_correct_passes(self):
        """$400 is double $200 — within 25% of 2.0x."""
        narrative = "Q2 revenue of $400,000 is double Q1 revenue of $200,000."
        result = check_magnitude_claims(narrative)
        assert result["violations"] == []

    def test_double_wrong_ratio_flagged(self):
        """$300 is 1.5x $200, not double."""
        narrative = "Sales of $300,000 are double the prior period at $200,000."
        result = check_magnitude_claims(narrative)
        assert len(result["violations"]) == 1

    def test_twice_correct_passes(self):
        narrative = "The treatment group converted at 10.0%, twice the control rate of 5.0%."
        result = check_magnitude_claims(narrative)
        assert result["violations"] == []

    def test_twice_wrong_flagged(self):
        narrative = "Conversion was 8.0%, twice the baseline of 5.0%."
        result = check_magnitude_claims(narrative)
        assert len(result["violations"]) == 1

    # ── "triple" / "three times" ──────────────────────────────────────────────

    def test_triple_correct_passes(self):
        narrative = "Churn of $900 is triple the $300 seen in the prior cohort."
        result = check_magnitude_claims(narrative)
        assert result["violations"] == []

    def test_triple_wrong_flagged(self):
        """$500 is 1.67x $300, not triple."""
        narrative = "Churn of $500 is triple the $300 seen in the prior cohort."
        result = check_magnitude_claims(narrative)
        assert len(result["violations"]) == 1

    # ── "N times" ─────────────────────────────────────────────────────────────

    def test_n_times_correct_passes(self):
        """$1,000 is 5 times $200 — within 25%."""
        narrative = "Electronics revenue of $1,000 is 5 times the $200 from Books."
        result = check_magnitude_claims(narrative)
        assert result["violations"] == []

    def test_n_times_wrong_flagged(self):
        """$500 is 2.5x $200, not 5 times."""
        narrative = "Electronics revenue of $500 is 5 times the $200 from Books."
        result = check_magnitude_claims(narrative)
        assert len(result["violations"]) == 1

    def test_n_times_small_multiplier_ignored(self):
        """'1 times' is noise — should not trigger a check."""
        narrative = "The result was 1 times better than expected at $200 vs $180."
        result = check_magnitude_claims(narrative)
        assert result["violations"] == []

    # ── No magnitude language ─────────────────────────────────────────────────

    def test_no_magnitude_language_passes(self):
        narrative = (
            "Revenue was $322,996 in Q1 and $248,410 in Q2. "
            "The gap reflects a shift in product mix."
        )
        result = check_magnitude_claims(narrative)
        assert result["violations"] == []

    def test_empty_narrative_passes(self):
        assert check_magnitude_claims("")["violations"] == []

    # ── Integration with score_general_claim_accuracy ─────────────────────────

    def test_integrated_into_general_claim_accuracy(self):
        """score_general_claim_accuracy must surface magnitude violations."""
        from tools.eval_tools import score_general_claim_accuracy
        narrative = (
            "Bronze generates $322,996 — more than double Silver at $248,410."
        )
        result = score_general_claim_accuracy(narrative)
        assert len(result["violations"]) >= 1
        assert any("more than double" in v.lower() for v in result["violations"])

    def test_integrated_into_ab_claim_accuracy(self):
        """score_claim_accuracy (A/B) must also surface magnitude violations."""
        from tools.eval_tools import score_claim_accuracy
        from tools.schemas import TtestResult
        tr = TtestResult(
            t_stat=2.0, p_value=0.03, ci_lower=0.01, ci_upper=0.05,
            significant=True, cohens_d=0.3, n_control=500, n_treatment=500,
        )
        narrative = (
            "The treatment group generated $322,996 — more than double the control at $248,410."
        )
        result = score_claim_accuracy(narrative, tr, None)
        assert any("more than double" in v.lower() for v in result["violations"])

    def test_correct_ratio_in_ab_narrative_passes(self):
        """A valid 'double' claim in A/B narrative should not be flagged."""
        from tools.eval_tools import score_claim_accuracy
        from tools.schemas import TtestResult
        tr = TtestResult(
            t_stat=3.0, p_value=0.001, ci_lower=0.05, ci_upper=0.15,
            significant=True, cohens_d=0.6, n_control=500, n_treatment=500,
        )
        narrative = (
            "Treatment revenue of $400,000 is double the control at $200,000. "
            "The result is statistically significant."
        )
        result = score_claim_accuracy(narrative, tr, None)
        mag_violations = [v for v in result["violations"] if "double" in v.lower()]
        assert mag_violations == []
