"""
tests/test_metric_gate.py — Metric Config Gate + connection_id resolution.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256")

from agents.analyze.nodes_metric import metric_config_gate
from agents.analyze.node_shared import _db_conn
from config.analysis_config import MetricConfig


def _mc(**overrides) -> MetricConfig:
    base = dict(
        primary_metric="dau_rate",
        metric_source_col="dau_flag",
        metric_agg="mean",
        covariate="pre_session_count",
        metric_direction="higher_is_better",
        events_table="events",
        experiment_table="experiment",
        guardrail_metrics=["notif_optout"],
        segment_cols=["platform"],
    )
    base.update(overrides)
    return MetricConfig(**base)


class TestMetricConfigGate:
    def test_skips_for_certified_pack(self):
        state = {
            "metric_pack_certified": True,
            "metric_config": _mc(),
        }
        out = metric_config_gate(state)
        assert out["metric_config_approved"] is True

    def test_skips_for_demo_duckdb(self):
        state = {
            "db_backend": "duckdb",
            "metric_config": _mc(),
        }
        out = metric_config_gate(state)
        assert out["metric_config_approved"] is True

    def test_skips_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SKIP_METRIC_GATE", "true")
        out = metric_config_gate({"db_backend": "postgres", "metric_config": _mc()})
        assert out["metric_config_approved"] is True

    def test_decline_returns_unapproved(self):
        state = {
            "db_backend": "postgres",
            "connection_id": "00000000-0000-4000-8000-000000000001",
            "metric_config": _mc(),
        }
        with patch("agents.analyze.nodes_metric.interrupt") as mock_int:
            mock_int.return_value = {"approved": False}
            out = metric_config_gate(state)
        assert out["metric_config_approved"] is False

    def test_interrupt_and_apply_edits(self):
        state = {
            "db_backend": "duckdb",
            "duckdb_path": "/tmp/upload.db",  # upload → gate fires
            "metric_config": _mc(),
            "schema_context": "-- Dialect: DuckDB\n\nTABLE events (dau_flag INTEGER, platform VARCHAR)",
        }
        with patch("agents.analyze.nodes_metric.interrupt") as mock_int:
            mock_int.return_value = {
                "approved": True,
                "metric_config": {"primary_metric": "dau_rate", "segment_cols": ["platform"]},
            }
            out = metric_config_gate(state)
        assert out["metric_config_approved"] is True
        mock_int.assert_called_once()
        payload = mock_int.call_args[0][0]
        assert payload["gate"] == "metric"


class TestDbConnVaultResolve:
    def test_resolves_connection_id(self, tmp_path, monkeypatch):
        auth_db = str(tmp_path / "auth.db")
        monkeypatch.setenv("AUTH_DB_PATH", auth_db)
        monkeypatch.setenv("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256")

        from auth import workspace_store
        workspace_store.init_workspace_tables(auth_db)
        c = workspace_store.create_connection(
            "u1",
            name="C",
            host="vault.example.com",
            port=5432,
            dbname="analytics",
            username="reader",
            password="vault-pass",
            sslmode="require",
            path=auth_db,
        )

        # Simulate post-load_schema wipe: no inline pg_* fields
        state = {
            "db_backend": "postgres",
            "connection_id": c.connection_id,
            "user_id": "u1",
            "pg_password": "",
            "pg_host": "",
        }
        conn = _db_conn(state)
        assert conn.backend == "postgres"
        assert conn._kwargs["host"] == "vault.example.com"
        assert conn._kwargs["password"] == "vault-pass"
        assert conn._kwargs["sslmode"] == "require"
