"""
tests/test_funnel_tools.py — Unit tests for tools/funnel_tools.py

Uses base_funnel_df fixture:
  - d1_retain rate drops 20pp for treatment android/new (0.45 → 0.25)
  - All other steps: no meaningful change
  - When filtered to android/new, biggest_dropoff_step == 'd1_retain'
  - Blended across all segments the signal is diluted → test with segment_filter
"""

import pytest

from tools.funnel_tools import compute_funnel

STEPS = ["impression", "click", "install", "d1_retain"]
ANDROID_NEW = {"platform": "android", "user_segment": "new"}


def test_biggest_dropoff_is_d1_retain(base_funnel_df):
    """In the android/new segment, d1_retain has the largest treatment delta."""
    result = compute_funnel(
        base_funnel_df, "variant", steps=STEPS, segment_filter=ANDROID_NEW
    )
    assert result["biggest_dropoff_step"] == "d1_retain"


def test_d1_retain_delta_negative(base_funnel_df):
    """Treatment d1_retain rate is lower than control in android/new → delta < 0."""
    result = compute_funnel(
        base_funnel_df, "variant", steps=STEPS, segment_filter=ANDROID_NEW
    )
    d1 = next(s for s in result["steps"] if s["step"] == "d1_retain")
    assert d1["delta"] < 0


def test_d1_retain_significant(base_funnel_df):
    """d1_retain drop is large enough to be statistically significant in android/new."""
    result = compute_funnel(
        base_funnel_df, "variant", steps=STEPS, segment_filter=ANDROID_NEW
    )
    d1 = next(s for s in result["steps"] if s["step"] == "d1_retain")
    assert d1["significant"] is True


def test_all_steps_returned(base_funnel_df):
    """Result contains one entry per step in the correct order."""
    result = compute_funnel(base_funnel_df, "variant", steps=STEPS)
    returned_steps = [s["step"] for s in result["steps"]]
    assert returned_steps == STEPS
