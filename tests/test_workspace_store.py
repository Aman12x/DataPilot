"""
tests/test_workspace_store.py — Saved connections + metric packs.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ensure SECRET_KEY before crypto import
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256")

from auth import workspace_store
from backend.api.crypto_secrets import decrypt_secret, encrypt_secret
from config.analysis_config import MetricConfig


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    path = str(tmp_path / "auth.db")
    monkeypatch.setenv("AUTH_DB_PATH", path)
    workspace_store.init_workspace_tables(path)
    return path


SAMPLE_CONFIG = {
    "primary_metric": "revenue",
    "metric_source_col": "revenue_usd",
    "metric_agg": "sum",
    "covariate": "prior_week_revenue",
    "metric_direction": "higher_is_better",
    "events_table": "transactions",
    "experiment_table": "assignments",
    "guardrail_metrics": ["refund_rate"],
    "segment_cols": ["country"],
}


class TestCryptoSecrets:
    def test_roundtrip(self):
        token = encrypt_secret("s3cret!")
        assert token != "s3cret!"
        assert decrypt_secret(token) == "s3cret!"

    def test_tamper_fails(self):
        token = encrypt_secret("hello")
        with pytest.raises(ValueError):
            decrypt_secret(token[:-4] + "xxxx")


class TestConnections:
    def test_create_list_get(self, auth_db):
        c = workspace_store.create_connection(
            "user-1",
            name="Prod PG",
            host="db.example.com",
            port=5432,
            dbname="analytics",
            username="reader",
            password="hunter2",
            path=auth_db,
        )
        assert c.name == "Prod PG"
        assert c.host == "db.example.com"

        listed = workspace_store.list_connections("user-1", path=auth_db)
        assert len(listed) == 1
        assert listed[0].connection_id == c.connection_id

        # Public view never exposes password
        assert "password" not in c.to_dict()

    def test_secrets_resolve_and_ownership(self, auth_db):
        c = workspace_store.create_connection(
            "user-1",
            name="Prod",
            host="db.example.com",
            port=5432,
            dbname="analytics",
            username="reader",
            password="hunter2",
            path=auth_db,
        )
        secrets = workspace_store.get_connection_secrets("user-1", c.connection_id, path=auth_db)
        assert secrets is not None
        assert secrets.password == "hunter2"

        # Other user cannot resolve
        assert workspace_store.get_connection_secrets("user-2", c.connection_id, path=auth_db) is None
        assert workspace_store.get_connection("user-2", c.connection_id, path=auth_db) is None

    def test_soft_delete(self, auth_db):
        c = workspace_store.create_connection(
            "user-1",
            name="Temp",
            host="db.example.com",
            port=5432,
            dbname="x",
            username="u",
            password="p",
            path=auth_db,
        )
        assert workspace_store.delete_connection("user-1", c.connection_id, path=auth_db)
        assert workspace_store.get_connection("user-1", c.connection_id, path=auth_db) is None
        assert workspace_store.list_connections("user-1", path=auth_db) == []

    def test_update_password(self, auth_db):
        c = workspace_store.create_connection(
            "user-1",
            name="Prod",
            host="db.example.com",
            port=5432,
            dbname="analytics",
            username="reader",
            password="old",
            path=auth_db,
        )
        workspace_store.update_connection(
            "user-1", c.connection_id, password="new-pass", path=auth_db,
        )
        secrets = workspace_store.get_connection_secrets("user-1", c.connection_id, path=auth_db)
        assert secrets.password == "new-pass"


class TestMetricPacks:
    def test_create_and_validate(self, auth_db):
        pack = workspace_store.create_metric_pack(
            "user-1",
            name="Revenue Pack",
            config=SAMPLE_CONFIG,
            certified=True,
            path=auth_db,
        )
        assert pack.certified is True
        assert pack.config["primary_metric"] == "revenue"
        MetricConfig(**pack.config)  # still valid

    def test_invalid_config_rejected(self, auth_db):
        with pytest.raises(Exception):
            workspace_store.create_metric_pack(
                "user-1",
                name="Bad",
                config={"primary_metric": ""},
                path=auth_db,
            )

    def test_ownership(self, auth_db):
        pack = workspace_store.create_metric_pack(
            "user-1", name="P", config=SAMPLE_CONFIG, path=auth_db,
        )
        assert workspace_store.get_metric_pack("user-2", pack.pack_id, path=auth_db) is None

    def test_version_bumps_on_config_update(self, auth_db):
        pack = workspace_store.create_metric_pack(
            "user-1", name="P", config=SAMPLE_CONFIG, path=auth_db,
        )
        updated = dict(SAMPLE_CONFIG)
        updated["primary_metric"] = "orders"
        new = workspace_store.update_metric_pack(
            "user-1", pack.pack_id, config=updated, path=auth_db,
        )
        assert new.version == 2
        assert new.config["primary_metric"] == "orders"

    def test_link_to_owned_connection(self, auth_db):
        c = workspace_store.create_connection(
            "user-1", name="C", host="h", port=5432, dbname="d",
            username="u", password="p", path=auth_db,
        )
        pack = workspace_store.create_metric_pack(
            "user-1", name="P", config=SAMPLE_CONFIG,
            connection_id=c.connection_id, path=auth_db,
        )
        assert pack.connection_id == c.connection_id

        with pytest.raises(ValueError):
            workspace_store.create_metric_pack(
                "user-1", name="P2", config=SAMPLE_CONFIG,
                connection_id="00000000-0000-4000-8000-000000000000",
                path=auth_db,
            )
