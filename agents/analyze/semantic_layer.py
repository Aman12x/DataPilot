"""
agents/analyze/semantic_layer.py — Phase 2 helpers.

Dataset fingerprinting, pack-vs-schema drift, join-graph formatting, and
certified-pack allowlists. Kept free of LangGraph / FastAPI deps so unit
tests can import it cheaply.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from config.analysis_config import MetricConfig


def schema_hash(schema_context: str) -> str:
    """Stable short hash of a schema_context string (dialect header ok)."""
    normalised = "\n".join(
        line.rstrip() for line in (schema_context or "").splitlines() if line.strip()
    )
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def compute_dataset_fingerprint(
    *,
    connection_id: str = "",
    duckdb_path: str = "",
    metric_pack_id: str = "",
    pack_version: int | None = None,
    schema_context: str = "",
) -> str:
    """
    Fingerprint scoping semantic cache + few-shot isolation.

    Combines durable data-source identity, optional pack version, and a hash of
    the live schema so cache entries invalidate on drift.
    """
    source = connection_id or duckdb_path or "demo"
    pack = metric_pack_id or ""
    version = "" if pack_version is None else str(pack_version)
    sch = schema_hash(schema_context) if schema_context else ""
    raw = f"{source}|pack={pack}|v={version}|schema={sch}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def parse_schema_tables(schema_context: str) -> dict[str, set[str]]:
    """Return {table_lower: {col_lower, ...}} from a formatted schema string."""
    tables: dict[str, set[str]] = {}
    current: str | None = None
    for line in (schema_context or "").splitlines():
        s = line.strip()
        if s.startswith("TABLE:"):
            raw = s.split(":", 1)[1].strip()
            current = raw.split("--")[0].strip().lower()
            tables[current] = set()
        elif current and s and not s.startswith("--") and not s.upper().startswith("DIALECT"):
            col = s.split()[0].lower()
            if col:
                tables[current].add(col)
    return tables


def detect_pack_drift(mc: MetricConfig, schema_context: str) -> list[str]:
    """
    Compare MetricConfig references against live schema.

    Returns human-readable warnings (empty when aligned). Column checks are
    table-affine: only warn when the parent table exists but the column does not
    (avoids false positives for unused default cols like week_col on minimal schemas).
    """
    if not schema_context or not schema_context.strip():
        return ["Live schema is empty — cannot validate metric pack."]

    tables = parse_schema_tables(schema_context)
    if not tables:
        return ["Could not parse live schema tables."]

    known_tables = set(tables.keys())
    warnings: list[str] = []

    required_tables = [mc.events_table, mc.experiment_table]
    if mc.timeseries_table:
        required_tables.append(mc.timeseries_table)
    if mc.funnel_table:
        required_tables.append(mc.funnel_table)

    for tbl in required_tables:
        if tbl and tbl.lower() not in known_tables:
            warnings.append(f"Pack table '{tbl}' not found in live schema")

    def _need(table: str, col: str, label: str | None = None) -> None:
        if not table or not col:
            return
        tl, cl = table.lower(), col.lower()
        if tl not in known_tables:
            return  # table warning already emitted
        if cl not in tables[tl] and cl not in {c for cols in tables.values() for c in cols}:
            warnings.append(
                f"Pack column '{label or col}' not found in live schema "
                f"(expected on '{table}' or related tables)"
            )

    events = mc.events_table
    experiment = mc.experiment_table
    _need(events, mc.metric_source_col or mc.primary_metric, "metric_source_col")
    _need(events, mc.covariate, "covariate")
    _need(events, mc.user_id_col, "user_id_col")
    for col in mc.guardrail_metrics or []:
        _need(events, col)
    for col in mc.segment_cols or []:
        _need(events, col)
    _need(experiment, mc.variant_col, "variant_col")
    _need(experiment, mc.user_id_col, "user_id_col")

    # Join graph endpoints
    for join in getattr(mc, "joins", None) or []:
        if isinstance(join, dict):
            left = (join.get("left_table") or "").lower()
            right = (join.get("right_table") or "").lower()
        else:
            left = (getattr(join, "left_table", "") or "").lower()
            right = (getattr(join, "right_table", "") or "").lower()
        if left and left not in known_tables:
            warnings.append(f"Join left table '{left}' not in live schema")
        if right and right not in known_tables:
            warnings.append(f"Join right table '{right}' not in live schema")

    return warnings


def pack_allowed_tables(mc: MetricConfig) -> set[str]:
    """Tables a certified pack authorises SQL to touch (plus join graph)."""
    allowed = {
        (mc.events_table or "").lower(),
        (mc.experiment_table or "").lower(),
    }
    if mc.timeseries_table:
        allowed.add(mc.timeseries_table.lower())
    if mc.funnel_table:
        allowed.add(mc.funnel_table.lower())
    for join in getattr(mc, "joins", None) or []:
        if isinstance(join, dict):
            for key in ("left_table", "right_table"):
                v = (join.get(key) or "").lower()
                if v:
                    allowed.add(v)
        else:
            for key in ("left_table", "right_table"):
                v = (getattr(join, key, "") or "").lower()
                if v:
                    allowed.add(v)
    return {t for t in allowed if t}


def format_join_graph(joins: Iterable[Any] | None) -> str:
    """Human-readable join graph for prompt injection."""
    items = list(joins or [])
    if not items:
        return "(No declared join graph — prefer FK-safe joins on user_id / date.)"
    lines = []
    for j in items:
        if isinstance(j, dict):
            left = j.get("left_table", "?")
            right = j.get("right_table", "?")
            on = j.get("on", "?")
            note = j.get("note") or ""
        else:
            left = getattr(j, "left_table", "?")
            right = getattr(j, "right_table", "?")
            on = getattr(j, "on", "?")
            note = getattr(j, "note", "") or ""
        suffix = f"  — {note}" if note else ""
        lines.append(f"- {left} ⟷ {right} ON {on}{suffix}")
    return "\n".join(lines)


def format_synonyms(synonyms: dict[str, str] | None) -> str:
    if not synonyms:
        return ""
    return "\n".join(f"- '{k}' → {v}" for k, v in sorted(synonyms.items()))


def annotations_to_temp_file(annotations: dict[str, Any], path: str) -> str:
    """Write annotations JSON for inspect_schema(annotation_path=...)."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    return path


def strip_sql_dialect_header(schema_context: str) -> str:
    """Remove leading `-- Dialect: ...` lines for hashing / storage."""
    lines = (schema_context or "").splitlines()
    while lines and (
        lines[0].strip().lower().startswith("-- dialect")
        or not lines[0].strip()
    ):
        lines.pop(0)
    return "\n".join(lines).strip()


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_annotations_payload(annotations: dict[str, Any]) -> dict[str, dict[str, str]]:
    """
    Normalise + validate annotation JSON:
      { "table": { "col": "comment", ... }, ... }
    """
    if not isinstance(annotations, dict):
        raise ValueError("annotations must be an object keyed by table name")
    out: dict[str, dict[str, str]] = {}
    for table, cols in annotations.items():
        if not isinstance(table, str) or not _IDENT.match(table):
            raise ValueError(f"Invalid table name in annotations: {table!r}")
        if not isinstance(cols, dict):
            raise ValueError(f"annotations[{table}] must be an object of column→comment")
        clean: dict[str, str] = {}
        for col, comment in cols.items():
            if not isinstance(col, str) or not _IDENT.match(col):
                raise ValueError(f"Invalid column name in annotations: {col!r}")
            if comment is None:
                continue
            text = str(comment).strip()
            if len(text) > 500:
                text = text[:500]
            if text:
                clean[col] = text
        if clean:
            out[table] = clean
    return out
