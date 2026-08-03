"""Checkpointer selection: Postgres when configured, loud fallback otherwise."""
from __future__ import annotations

import logging
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
for p in (ROOT, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.api import main as api_main  # noqa: E402


@pytest.fixture(autouse=True)
def graph_db(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "graph.db"))


def test_no_database_url_selects_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    saver, pool, backend = api_main.select_checkpointer()
    assert backend == "sqlite"
    assert pool is None
    assert saver is not None


def test_postgres_failure_falls_back_loudly(monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nope:nope@127.0.0.1:1/nope")

    def boom(url):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(api_main, "_make_postgres_checkpointer", boom)

    with caplog.at_level(logging.ERROR):
        saver, pool, backend = api_main.select_checkpointer()

    assert backend.startswith("sqlite (FALLBACK")
    assert pool is None
    assert saver is not None
    assert any("SPLIT-BRAIN" in r.message for r in caplog.records)


def test_postgres_success_reports_postgres(monkeypatch):
    sentinel_saver, sentinel_pool = object(), object()
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setattr(api_main, "_make_postgres_checkpointer",
                        lambda url: (sentinel_saver, sentinel_pool))
    saver, pool, backend = api_main.select_checkpointer()
    assert backend == "postgres"
    assert saver is sentinel_saver and pool is sentinel_pool


def test_retention_skips_sqlite_checkpoint_logic_under_postgres(monkeypatch, tmp_path):
    from backend.api import retention

    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))

    report = retention.run_maintenance()
    assert report["checkpoints"] == "skipped (postgres checkpointer)"
    assert "graph_bytes_reclaimed" not in report
