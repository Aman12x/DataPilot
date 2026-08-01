"""
User content must not reach log records.

resolve_task_intent logged 80 characters of the raw analyst task at INFO,
unconditionally. sentry-sdk promotes INFO records to breadcrumbs, so with
SENTRY_DSN set every one of those questions left the box.
"""
import logging

import pytest

from agents.log_safety import redact

_TASK = "Why did enterprise churn spike in EMEA during Q3?"


@pytest.fixture(autouse=True)
def _no_opt_in(monkeypatch):
    monkeypatch.delenv("LOG_USER_CONTENT", raising=False)


# ── redact ────────────────────────────────────────────────────────────────────


def test_redacted_output_contains_no_content():
    out = redact(_TASK)
    assert "churn" not in out
    assert "EMEA" not in out
    assert _TASK not in out


def test_redacted_output_keeps_length_for_diagnosis():
    assert f"len={len(_TASK)}" in redact(_TASK)


def test_same_text_redacts_identically_within_a_process():
    """Operators still need to tell two log lines apart."""
    assert redact(_TASK) == redact(_TASK)


def test_different_text_gets_a_different_reference():
    assert redact("question one") != redact("question two")


def test_reference_is_salted_so_short_text_is_not_recoverable():
    """An unsalted digest of a short task would be trivially brute-forced."""
    import hashlib

    unsalted = hashlib.sha256(b"sales").hexdigest()[:8]
    assert unsalted not in redact("sales")


def test_empty_and_none_are_distinguishable():
    assert redact("") == "<empty>"
    assert redact(None) == "<none>"


def test_opt_in_returns_the_real_text(monkeypatch):
    monkeypatch.setenv("LOG_USER_CONTENT", "true")
    assert redact(_TASK) == _TASK


def test_opt_in_is_read_at_call_time(monkeypatch):
    """A running process can be toggled without a restart."""
    assert _TASK not in redact(_TASK)
    monkeypatch.setenv("LOG_USER_CONTENT", "1")
    assert redact(_TASK) == _TASK
    monkeypatch.setenv("LOG_USER_CONTENT", "false")
    assert _TASK not in redact(_TASK)


def test_opt_in_still_respects_the_limit(monkeypatch):
    monkeypatch.setenv("LOG_USER_CONTENT", "true")
    assert len(redact("x" * 500, limit=80)) == 80


def test_non_string_input_does_not_raise():
    assert redact(12345)
    assert redact({"a": 1})


# ── The call site ─────────────────────────────────────────────────────────────


def test_lookup_heuristic_log_does_not_leak_the_task(caplog, monkeypatch):
    """Drive the real node so the assertion covers the shipped log call."""
    import json
    from types import SimpleNamespace

    from agents.analyze import nodes_intent

    # Phrased to trip _is_lookup_task, which is what reaches the log line.
    lookup_task = "How many enterprise accounts in EMEA signed up yesterday?"
    assert nodes_intent._is_lookup_task(lookup_task), "fixture no longer trips the heuristic"

    intent = {"analysis_mode": "general", "query_type": "exploratory", "ambiguous": False}

    class _FakeMessages:
        def create(self, **_):
            return SimpleNamespace(
                content=[SimpleNamespace(text=json.dumps(intent))],
                usage=SimpleNamespace(
                    input_tokens=1, output_tokens=1,
                    cache_read_input_tokens=0, cache_creation_input_tokens=0,
                ),
            )

    monkeypatch.setattr(
        nodes_intent, "_anthropic_client", lambda: SimpleNamespace(messages=_FakeMessages())
    )

    with caplog.at_level(logging.INFO):
        out = nodes_intent.resolve_task_intent({
            "task": lookup_task,
            "analysis_mode": "general",
            "run_id": "run-abc123",
        })

    assert out["query_type"] == "lookup", "heuristic did not fire; log line unreached"

    logged = caplog.text
    assert "via heuristic" in logged, "expected the heuristic log record"
    assert "EMEA" not in logged
    assert "enterprise" not in logged
    assert lookup_task not in logged
    assert "run-abc123" in logged, "run_id is the correlation key operators need"


def test_source_no_longer_slices_the_raw_task():
    """Guards against a revert to `task[:80]` in the log call."""
    import inspect

    from agents.analyze import nodes_intent

    source = inspect.getsource(nodes_intent.resolve_task_intent)
    log_lines = [ln for ln in source.splitlines() if "logger.info" in ln or "task=%s" in ln]
    assert log_lines, "expected the heuristic log line to still exist"
    assert not any("task[:80]" in ln for ln in log_lines)


def test_sentry_pii_is_explicitly_disabled():
    """Log records become breadcrumbs; request bodies must not follow them."""
    from pathlib import Path

    main_src = (Path(__file__).resolve().parents[1] / "backend" / "api" / "main.py").read_text()
    assert "send_default_pii=False" in main_src
