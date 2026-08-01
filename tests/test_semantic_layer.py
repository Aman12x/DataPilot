"""
tests/test_semantic_layer.py — Phase 2 fingerprint / drift / annotations / pack scope.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256")

from agents.analyze.semantic_layer import (
    compute_dataset_fingerprint,
    detect_pack_drift,
    pack_allowed_tables,
    schema_hash,
    validate_annotations_payload,
)
from agents.analyze.nodes_metric import metric_config_gate
from config.analysis_config import MetricConfig, JoinSpec
from unittest.mock import patch


SCHEMA = """
-- Dialect: DuckDB SQL

TABLE: events
  user_id                VARCHAR
  dau_flag               INTEGER
  platform               VARCHAR
  pre_session_count      INTEGER

TABLE: experiment
  user_id                VARCHAR
  variant                VARCHAR
  assignment_date        DATE
"""


def _mc(**overrides) -> MetricConfig:
    base = dict(
        primary_metric="dau_rate",
        metric_source_col="dau_flag",
        metric_agg="mean",
        covariate="pre_session_count",
        metric_direction="higher_is_better",
        events_table="events",
        experiment_table="experiment",
        guardrail_metrics=[],
        segment_cols=["platform"],
    )
    base.update(overrides)
    return MetricConfig(**base)


class TestFingerprint:
    def test_changes_with_schema(self):
        a = compute_dataset_fingerprint(connection_id="c1", schema_context=SCHEMA)
        b = compute_dataset_fingerprint(
            connection_id="c1",
            schema_context=SCHEMA + "\nTABLE: extra\n  id INTEGER\n",
        )
        assert a != b

    def test_changes_with_pack(self):
        a = compute_dataset_fingerprint(connection_id="c1", metric_pack_id="p1", pack_version=1)
        b = compute_dataset_fingerprint(connection_id="c1", metric_pack_id="p1", pack_version=2)
        assert a != b

    def test_schema_hash_stable(self):
        assert schema_hash(SCHEMA) == schema_hash(SCHEMA)


class TestDrift:
    def test_aligned_pack_no_warnings(self):
        assert detect_pack_drift(_mc(), SCHEMA) == []

    def test_missing_table(self):
        warnings = detect_pack_drift(_mc(events_table="orders"), SCHEMA)
        assert any("orders" in w for w in warnings)

    def test_missing_column(self):
        warnings = detect_pack_drift(_mc(segment_cols=["plan_tier"]), SCHEMA)
        assert any("plan_tier" in w for w in warnings)

    def test_join_table_checked(self):
        mc = _mc(joins=[JoinSpec(left_table="events", right_table="users", on="events.user_id = users.user_id")])
        warnings = detect_pack_drift(mc, SCHEMA)
        assert any("users" in w for w in warnings)


class TestAnnotations:
    def test_validate_ok(self):
        out = validate_annotations_payload({"events": {"dau_flag": "daily active flag"}})
        assert out["events"]["dau_flag"] == "daily active flag"

    def test_validate_rejects_bad_table(self):
        with pytest.raises(ValueError):
            validate_annotations_payload({"bad-table!": {"x": "y"}})


class TestAllowlist:
    def test_includes_join_tables(self):
        mc = _mc(joins=[JoinSpec(left_table="events", right_table="dim_users", on="a=b")])
        allowed = pack_allowed_tables(mc)
        assert "events" in allowed
        assert "experiment" in allowed
        assert "dim_users" in allowed


class TestMetricGateDrift:
    def test_certified_skips_without_drift(self):
        out = metric_config_gate({
            "metric_pack_certified": True,
            "metric_config": _mc(),
            "schema_drift_warnings": [],
        })
        assert out["metric_config_approved"] is True

    def test_certified_with_drift_interrupts(self):
        with patch("agents.analyze.nodes_metric.interrupt") as mock_int:
            mock_int.return_value = {"approved": True}
            out = metric_config_gate({
                "metric_pack_certified": True,
                "force_metric_gate": True,
                "metric_config": _mc(),
                "schema_drift_warnings": ["Pack column 'plan_tier' not found"],
                "schema_context": SCHEMA,
            })
        assert out["metric_config_approved"] is True
        mock_int.assert_called_once()
        payload = mock_int.call_args[0][0]
        assert payload["schema_drift_warnings"]
        assert "drift" in payload["message"].lower()


class TestAnnotationsStore:
    def test_upsert_and_get(self, tmp_path, monkeypatch):
        auth_db = str(tmp_path / "auth.db")
        monkeypatch.setenv("AUTH_DB_PATH", auth_db)
        from auth import workspace_store
        workspace_store.init_workspace_tables(auth_db)
        c = workspace_store.create_connection(
            "u1", name="C", host="h.example.com", port=5432,
            dbname="d", username="u", password="p", path=auth_db,
        )
        ann = workspace_store.upsert_annotations(
            "u1", c.connection_id,
            annotations={"events": {"revenue": "USD revenue"}},
            synonyms={"WAU": "weekly_active"},
            path=auth_db,
        )
        assert ann.annotations["events"]["revenue"] == "USD revenue"
        assert ann.synonyms["WAU"] == "weekly_active"
        got = workspace_store.get_annotations("u1", c.connection_id, path=auth_db)
        assert got.annotations["events"]["revenue"] == "USD revenue"


class TestMemoryPackScope:
    def test_log_and_retrieve_prefers_same_pack(self, tmp_path, monkeypatch):
        mem = str(tmp_path / "mem.db")
        monkeypatch.setenv("MEMORY_DB_PATH", mem)
        from memory.store import init_db, log_run
        from memory.retriever import retrieve_relevant_history

        init_db(mem)
        log_run(
            "Did revenue increase for checkout?",
            path=mem, user_id="u1", metric="revenue",
            metric_pack_id="pack-a", connection_id="conn-a",
            audit_passed=True, eval_score=0.9, top_segment="US",
        )
        log_run(
            "Did revenue increase for checkout?",
            path=mem, user_id="u1", metric="revenue",
            metric_pack_id="pack-b", connection_id="conn-b",
            audit_passed=True, eval_score=0.9, top_segment="US",
        )
        hist = retrieve_relevant_history(
            "revenue checkout increase",
            path=mem, user_id="u1",
            metric_pack_id="pack-a", connection_id="conn-a",
            top_n=1,
        )
        assert hist
        assert hist[0]["metric_pack_id"] == "pack-a"
