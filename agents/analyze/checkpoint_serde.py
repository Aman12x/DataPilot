"""
agents/analyze/checkpoint_serde.py — Safe JSON checkpoint serialization.

Replaces pickle for LangGraph checkpoints. DataFrames and Pydantic models
round-trip without arbitrary code execution on load.
"""
from __future__ import annotations

import io
import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from config.analysis_config import MetricConfig
from tools import schemas as schema_models

try:
    from langgraph.types import Interrupt as _Interrupt
except ImportError:  # pragma: no cover
    _Interrupt = None  # type: ignore[misc, assignment]

_SERDE_TAG = "json-v1"

# Registry for Pydantic models stored in AgentState
_PYDANTIC_MODELS: dict[str, type[BaseModel]] = {
    cls.__name__: cls
    for cls in (
        MetricConfig,
        schema_models.CupedResult,
        schema_models.TtestResult,
        schema_models.SegmentResult,
        schema_models.HteResult,
        schema_models.ComponentStats,
        schema_models.SegmentBreakdown,
        schema_models.DecompositionResult,
        schema_models.AnomalyResult,
        schema_models.SliceDimension,
        schema_models.SliceResult,
        schema_models.FunnelStep,
        schema_models.FunnelResult,
        schema_models.ForecastResult,
        schema_models.GuardrailMetric,
        schema_models.GuardrailResult,
        schema_models.NoveltyResult,
        schema_models.SrmResult,
        schema_models.MdeResult,
        schema_models.SensitivityRow,
        schema_models.PowerAnalysisResult,
        schema_models.NarrativeResult,
        schema_models.NarrativeFinding,
        schema_models.NarrativeAuditResult,
        schema_models.ColumnSummary,
        schema_models.DescribeResult,
        schema_models.CorrelationPair,
        schema_models.CorrelationResult,
        schema_models.RegressionCoef,
        schema_models.RegressionResult,
        schema_models.ChartSpec,
        schema_models.TrustIndicators,
    )
}


def _encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (datetime, date)):
        return {"__t": "datetime", "v": obj.isoformat()}
    if isinstance(obj, np.ndarray):
        return {"__t": "ndarray", "v": obj.tolist()}
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return {"__t": "numpy_scalar", "v": obj.item()}
    if isinstance(obj, pd.DataFrame):
        return {
            "__t": "dataframe",
            "v": json.loads(obj.to_json(orient="split", date_format="iso")),
        }
    if isinstance(obj, BaseModel):
        fields = {name: getattr(obj, name) for name in obj.model_fields}
        return {
            "__t": "pydantic",
            "m": obj.__class__.__name__,
            "v": _encode(fields),
        }
    if _Interrupt is not None and isinstance(obj, _Interrupt):
        return {
            "__t": "interrupt",
            "v": _encode(obj.value),
            "id": obj.id,
        }
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    raise TypeError(f"Cannot serialize checkpoint value of type {type(obj)!r}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and "__t" in obj:
        tag = obj["__t"]
        if tag == "datetime":
            return datetime.fromisoformat(obj["v"])
        if tag == "ndarray":
            return np.asarray(obj["v"])
        if tag == "numpy_scalar":
            return obj["v"]
        if tag == "dataframe":
            return pd.read_json(io.StringIO(json.dumps(obj["v"])), orient="split")
        if tag == "pydantic":
            model_cls = _PYDANTIC_MODELS.get(obj["m"])
            if model_cls is None:
                raise ValueError(f"Unknown checkpoint model: {obj['m']!r}")
            return model_cls.model_validate(_decode(obj["v"]))
        if tag == "interrupt":
            if _Interrupt is None:
                raise ValueError("Cannot restore interrupt — langgraph.types.Interrupt unavailable")
            return _Interrupt(value=_decode(obj["v"]), id=obj["id"])
    if isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


class SafeCheckpointSerde:
    """LangGraph-compatible serde using JSON instead of pickle."""

    def dumps_typed(self, obj: object) -> tuple[str, bytes]:
        payload = json.dumps(_encode(obj), separators=(",", ":")).encode("utf-8")
        return (_SERDE_TAG, payload)

    def loads_typed(self, data: tuple[str, bytes]) -> object:
        type_tag, payload = data
        if type_tag == "pickle":
            raise ValueError(
                "Pickle checkpoints are disabled for security. "
                "Delete the graph database file to start fresh."
            )
        if type_tag != _SERDE_TAG:
            raise ValueError(f"Unknown checkpoint serde tag: {type_tag!r}")
        return _decode(json.loads(payload.decode("utf-8")))
