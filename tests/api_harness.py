"""
tests/api_harness.py — Shared FastAPI TestClient harness for API tests.

Imported by test_api.py and test_workspace_api.py so both share one app,
one FakeGraph, and one _fake_graph_mode dict (no dual-import split-brain).
"""

from __future__ import annotations

import os
import sys
import types
import uuid
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
for p in (ROOT, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)


def _stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _stub_if_missing(name: str, **attrs):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = _stub(name, **attrs)


_stub_if_missing("anthropic")
_stub_if_missing("langfuse")
_stub_if_missing("langfuse.decorators", observe=lambda **kw: (lambda f: f))
_stub_if_missing("sentence_transformers")

if not hasattr(sys.modules.get("langfuse.decorators", object()), "observe"):
    sys.modules["langfuse.decorators"] = _stub(
        "langfuse.decorators", observe=lambda **kw: (lambda f: f)
    )


_fake_graph_mode: dict[str, str] = {"mode": "complete"}


class FakeGraph:
    """Simulates the LangGraph graph for API tests."""

    def __init__(self):
        self._known_runs: set[str] = set()
        self._gate_run_ids: set[str] = set()
        self._run_owners: dict[str, str] = {}

    def invoke(self, state_or_cmd, config, **__):
        run_id = config.get("configurable", {}).get("thread_id", "")
        mode = _fake_graph_mode["mode"]
        if mode == "crash":
            raise RuntimeError("simulated node failure")
        self._known_runs.add(run_id)
        if mode == "gate":
            self._gate_run_ids.add(run_id)
        if isinstance(state_or_cmd, dict) and state_or_cmd.get("user_id"):
            self._run_owners[run_id] = state_or_cmd["user_id"]
        return {}

    def stream(self, state_or_cmd, config, **__):
        run_id = config.get("configurable", {}).get("thread_id", "")
        mode = _fake_graph_mode["mode"]
        if mode == "crash":
            raise RuntimeError("simulated node failure")
        self._known_runs.add(run_id)
        if mode == "gate":
            self._gate_run_ids.add(run_id)
        if isinstance(state_or_cmd, dict) and state_or_cmd.get("user_id"):
            self._run_owners[run_id] = state_or_cmd["user_id"]
        yield {"generate_narrative": {}}

    def get_state(self, config, **__):
        run_id = config.get("configurable", {}).get("thread_id", "")
        if run_id not in self._known_runs:
            raise Exception("run not found")
        state = MagicMock()
        owner = self._run_owners.get(run_id, "test-user")
        state.values = {
            "task": "test",
            "narrative_draft": "hello",
            "recommendation": "ship it",
            "user_id": owner,
        }
        state.next = ()
        if run_id in self._gate_run_ids:
            interrupt_obj = MagicMock()
            interrupt_obj.value = {"gate": "intent", "payload": {"question": "What analysis?"}}
            task = MagicMock()
            task.interrupts = [interrupt_obj]
            state.tasks = [task]
        else:
            state.tasks = []
        return state


class FakeMemoryStore:
    def get_all_runs(self, **_):
        return []


@asynccontextmanager
async def test_lifespan(app):
    from api.run_manager import set_redis_client

    set_redis_client(None)
    app.state.graph = FakeGraph()
    app.state.memory_store = FakeMemoryStore()
    yield


os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough")
os.environ.setdefault("AUTH_DB_PATH", f"/tmp/test_auth_{uuid.uuid4().hex}.db")
os.environ.setdefault("MEMORY_DB_PATH", f"/tmp/test_mem_{uuid.uuid4().hex}.db")
os.environ.setdefault("UPLOAD_DIR", f"/tmp/test_uploads_{uuid.uuid4().hex}")
os.environ.setdefault("GRAPH_DB_PATH", f"/tmp/test_graph_{uuid.uuid4().hex}.db")
os.environ.setdefault("AUTH_AUTO_VERIFY_EMAIL", "true")
os.environ.setdefault("AUTH_RETURN_TOKENS", "true")
os.environ.setdefault("AUTH_RATE_MAX_ATTEMPTS", "10000")
os.environ.setdefault("ALLOW_PRIVATE_DB_HOSTS", "false")

from api.main import app  # noqa: E402

app.router.lifespan_context = test_lifespan  # type: ignore[assignment]


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def fake_mode():
    yield _fake_graph_mode
    _fake_graph_mode["mode"] = "complete"
