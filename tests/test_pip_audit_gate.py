"""
tests/test_pip_audit_gate.py — the CI dependency gate cannot pass silently.

backend/pip_audit_gate.py wraps pip-audit's JSON report. Four outcomes must
hold regardless of what pip-audit itself exits with: a new advisory fails
the gate; an ignored ID passes; a pin pip-audit *skipped* (not auditable)
fails; an empty dependency list fails.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATE = os.path.join(ROOT, "backend", "pip_audit_gate.py")


def _load(monkeypatch, stdout: str, ignore: str = ""):
    spec = importlib.util.spec_from_file_location("pip_audit_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    def _run(*a, **k):
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=1)

    monkeypatch.setattr(mod.subprocess, "run", _run)
    monkeypatch.setattr(mod, "_ignored", lambda: {l.split("#")[0].strip(): "" for l in ignore.splitlines() if l.strip()})
    return mod


def _report(deps):
    return json.dumps({"dependencies": deps, "fixes": []})


def test_new_vulnerability_fails(monkeypatch, capsys):
    mod = _load(monkeypatch, _report([
        {"name": "foo", "version": "1.0", "vulns": [{"id": "PYSEC-1", "fix_versions": ["1.1"], "aliases": []}]}
    ]))
    assert mod.main() == 1
    assert "PYSEC-1" in capsys.readouterr().out


def test_ignored_vulnerability_passes(monkeypatch):
    mod = _load(monkeypatch, _report([
        {"name": "foo", "version": "1.0", "vulns": [{"id": "PYSEC-1", "fix_versions": [], "aliases": []}]}
    ]), ignore="PYSEC-1 # triaged")
    assert mod.main() == 0


def test_skipped_pin_fails(monkeypatch, capsys):
    """A pin pip-audit could not look up (404 / non-PyPI) has skip_reason and
    no vulns; it must not read as clean."""
    mod = _load(monkeypatch, _report([
        {"name": "clean", "version": "1.0", "vulns": []},
        {"name": "torch", "version": "2.13.0+cpu", "skip_reason": "Dependency not found on PyPI"},
    ]))
    assert mod.main() == 2
    assert "torch" in capsys.readouterr().err


def test_empty_report_fails(monkeypatch):
    mod = _load(monkeypatch, _report([]))
    assert mod.main() == 2


def test_non_json_output_fails(monkeypatch):
    mod = _load(monkeypatch, "Traceback: network is down")
    assert mod.main() == 2
