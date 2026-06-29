"""Tests for evals/compare_baseline.py and committed baseline.json."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "baseline.json")
EVAL_COMMANDS = ["analyze_eval", "generalisability_eval", "transactions_eval", "fixture_eval"]


class TestBaselineFile:
    def test_baseline_exists_and_is_valid_json(self):
        assert os.path.isfile(BASELINE_PATH)
        with open(BASELINE_PATH) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert 0 < data["tolerance"] <= 0.10
        assert set(data["evals"]) == set(EVAL_COMMANDS)

    @pytest.mark.parametrize("name", EVAL_COMMANDS)
    def test_each_harness_meets_pass_threshold(self, name):
        with open(BASELINE_PATH) as f:
            data = json.load(f)
        entry = data["evals"][name]
        assert entry["score"] >= 0.80
        assert entry["n_pass"] <= entry["n_total"]
        assert entry["n_pass"] / entry["n_total"] == pytest.approx(entry["score"], rel=0.01)
