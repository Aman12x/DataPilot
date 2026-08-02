"""
tests/test_analysis_gate_payload.py — Gate payloads send None for skipped steps.

The gate payload crosses into the frontend, whose types declare optional
results as `<Result> | null`. `_to_dict(None)` returns {} — truthy in JS —
which crashed GeneralAnalysisGate on `corr.pairs.length` for every lookup
query (they skip find_correlations entirely). These tests pin the contract:
a skipped step is None in the payload, never {}.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256")

from agents.analyze.nodes_analysis import analysis_gate
from tools.schemas import CorrelationPair, CorrelationResult


def _gate_payload(state: dict) -> dict:
    with patch("agents.analyze.nodes_analysis.interrupt") as mock_int:
        mock_int.return_value = {"approved": True, "notes": ""}
        analysis_gate(state)
        (payload,), _ = mock_int.call_args
    return payload


class TestSkippedStepsAreNull:
    def test_general_lookup_sends_null_correlation(self):
        payload = _gate_payload({
            "analysis_mode": "general",
            "describe_result": None,
            "correlation_result": None,
        })
        assert payload["describe_result"] is None
        assert payload["correlation_result"] is None

    def test_general_present_results_stay_dicts(self):
        corr = CorrelationResult(pairs=[
            CorrelationPair(col_a="a", col_b="b", correlation=0.9, n=10),
        ])
        payload = _gate_payload({
            "analysis_mode": "general",
            "describe_result": None,
            "correlation_result": corr,
        })
        assert payload["correlation_result"]["pairs"][0]["col_a"] == "a"

    def test_power_analysis_sends_null_when_missing(self):
        payload = _gate_payload({
            "analysis_mode": "power_analysis",
            "power_analysis_result": None,
        })
        assert payload["power_analysis_result"] is None

    def test_ab_test_sends_null_decomposition(self):
        payload = _gate_payload({"analysis_mode": "ab_test"})
        assert payload["decomposition"] is None
