"""
tests/test_audit_patching.py — patch-only audit corrections.

The audit's critical findings used to trigger full narrative regeneration:
another 8192-token draft plus another full audit per retry — a measured 88s
and ~$0.15 to fix what is typically one wrong number in one sentence. Now the
auditor's own quote → corrected_sentence pairs are applied in place with no
LLM call; only unpatchable findings reach the analyst, as a gate warning.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.analyze.nodes_narrative import _apply_audit_patches
from tools.schemas import NarrativeFinding


def _finding(quote: str, fix: str, issue: str = "wrong number") -> NarrativeFinding:
    return NarrativeFinding(
        quote=quote, corrected_sentence=fix, issue=issue, severity="critical"
    )


NARRATIVE = (
    "Treatment reduced dau_rate by 0.0125 (0.6379 vs 0.6504).\n"
    "iOS users underperform by 2.1 points relative to Android."
)


class TestApplyAuditPatches:
    def test_exact_quote_is_replaced_in_place(self):
        out, patched, unpatched = _apply_audit_patches(
            NARRATIVE,
            [_finding("underperform by 2.1 points", "underperform by 1.2 points")],
        )
        assert "1.2 points" in out and "2.1 points" not in out
        assert len(patched) == 1 and not unpatched
        # everything else untouched
        assert out.startswith("Treatment reduced dau_rate by 0.0125")

    def test_whitespace_tolerant_match(self):
        """The auditor collapses line breaks when quoting."""
        out, patched, unpatched = _apply_audit_patches(
            NARRATIVE,
            [_finding("(0.6379 vs 0.6504). iOS users", "(0.6379 vs 0.6504). Apple users")],
        )
        assert "Apple users" in out
        assert len(patched) == 1 and not unpatched

    def test_ambiguous_quote_is_left_for_the_gate(self):
        """A phrase that occurs twice (TL;DR and body) must not be patched:
        replacing the first hit would edit the correct occurrence, leave the
        wrong one, and annotate the narrative as fixed."""
        narrative = (
            "TL;DR: lift of 3.2 points.\n\n"
            "Body: the headline lift of 3.2 points is driven by iOS."
        )
        out, patched, unpatched = _apply_audit_patches(
            narrative, [_finding("3.2 points", "2.3 points")]
        )
        assert out == narrative
        assert not patched and len(unpatched) == 1

    def test_ambiguous_loose_match_is_left_for_the_gate(self):
        narrative = "lift of\n3.2 points here; lift of 3.2 points there."
        out, patched, unpatched = _apply_audit_patches(
            narrative, [_finding("lift of 3.2 points", "lift of 2.3 points")]
        )
        assert out == narrative
        assert not patched and len(unpatched) == 1

    def test_unfindable_quote_is_returned_unpatched(self):
        out, patched, unpatched = _apply_audit_patches(
            NARRATIVE, [_finding("this text is not in the narrative", "fix")]
        )
        assert out == NARRATIVE
        assert not patched and len(unpatched) == 1

    def test_empty_corrected_sentence_is_unpatchable(self):
        out, patched, unpatched = _apply_audit_patches(
            NARRATIVE, [_finding("underperform by 2.1 points", "")]
        )
        assert out == NARRATIVE
        assert not patched and len(unpatched) == 1

    def test_mixed_findings_patch_what_they_can(self):
        out, patched, unpatched = _apply_audit_patches(
            NARRATIVE,
            [
                _finding("by 0.0125", "by 0.0125 (-1.9%)"),
                _finding("nonexistent quote", "fix"),
            ],
        )
        assert "(-1.9%)" in out
        assert len(patched) == 1 and len(unpatched) == 1

    def test_replacement_with_backslashes_is_literal(self):
        """re.sub replacement escapes: '\\' or '\\1' in a fix must not explode."""
        out, patched, _ = _apply_audit_patches(
            "path is C:temp here.\nsecond   line",
            [_finding("path is C:temp   here.", r"path is C:\temp\1 here.")],
        )
        assert r"C:\temp\1" in out
        assert len(patched) == 1
