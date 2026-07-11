"""Tests for safe checkpoint serialization."""
from __future__ import annotations

import pandas as pd
import pytest

from agents.analyze.checkpoint_serde import SafeCheckpointSerde
from config.analysis_config import MetricConfig


@pytest.fixture
def serde():
    return SafeCheckpointSerde()


def test_dataframe_roundtrip(serde):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    tag, payload = serde.dumps_typed({"query_result": df})
    restored = serde.loads_typed((tag, payload))
    pd.testing.assert_frame_equal(restored["query_result"], df)


def test_metric_config_roundtrip(serde):
    mc = MetricConfig(
        primary_metric="rate",
        covariate="baseline",
        metric_direction="higher_is_better",
        guardrail_metrics=["optout"],
        segment_cols=["platform"],
    )
    tag, payload = serde.dumps_typed({"metric_config": mc})
    restored = serde.loads_typed((tag, payload))
    assert restored["metric_config"].primary_metric == "rate"


def test_rejects_pickle_checkpoints(serde):
    with pytest.raises(ValueError, match="Pickle checkpoints are disabled"):
        serde.loads_typed(("pickle", b"legacy"))


def test_interrupt_roundtrip(serde):
    from langgraph.types import Interrupt

    payload = {
        "gate": "query",
        "generated_sql": "SELECT 1",
        "message": "Review SQL",
    }
    tag, raw = serde.dumps_typed([Interrupt(value=payload, id="abc123")])
    restored = serde.loads_typed((tag, raw))
    assert len(restored) == 1
    assert isinstance(restored[0], Interrupt)
    assert restored[0].id == "abc123"
    assert restored[0].value["gate"] == "query"
