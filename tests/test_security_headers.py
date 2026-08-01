"""
Content-Security-Policy and the rest of the security headers.

SecurityHeadersMiddleware set four headers but no CSP, and nothing asserted on
its output at all — so the gap was invisible. These tests cover both surfaces:
the API (strict, it only ever returns JSON/SSE) and the SPA's generated
serve.json (where a CSP actually blunts XSS).
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    # No lifespan: the middleware runs regardless, and these assertions need no
    # graph, checkpointer, or database.
    return TestClient(app, raise_server_exceptions=False)


def _csp(response) -> str:
    assert "Content-Security-Policy" in response.headers, "no CSP header"
    return response.headers["Content-Security-Policy"]


# ── API responses ─────────────────────────────────────────────────────────────


def test_api_response_carries_a_csp(client):
    policy = _csp(client.get("/no-such-route"))
    assert "default-src 'none'" in policy


def test_api_csp_forbids_framing_and_base_tag_hijacking(client):
    policy = _csp(client.get("/no-such-route"))
    assert "frame-ancestors 'none'" in policy
    assert "base-uri 'none'" in policy
    assert "form-action 'none'" in policy


def test_api_csp_allows_no_remote_origins(client):
    """A JSON API never needs to fetch anything."""
    policy = _csp(client.get("/no-such-route"))
    assert "http://" not in policy
    assert "https://" not in policy
    assert "unsafe-inline" not in policy


def test_existing_headers_still_present(client):
    headers = client.get("/no-such-route").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in headers["Permissions-Policy"]


# ── Docs pages ────────────────────────────────────────────────────────────────


def test_docs_get_a_policy_that_lets_swagger_load(client):
    """Swagger UI bootstraps from an inline script and a CDN bundle."""
    policy = _csp(client.get("/docs"))
    assert "https://cdn.jsdelivr.net" in policy
    assert "'unsafe-inline'" in policy


def test_docs_are_still_unframeable(client):
    assert "frame-ancestors 'none'" in _csp(client.get("/docs"))


def test_api_routes_do_not_inherit_the_relaxed_docs_policy(client):
    """The docs exemption must not leak to real endpoints."""
    assert "cdn.jsdelivr.net" not in _csp(client.get("/no-such-route"))
    assert "cdn.jsdelivr.net" not in _csp(client.get("/docsomething"))


# ── SPA policy generation ─────────────────────────────────────────────────────

_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="node not installed")


def _generate(tmp_path, **env) -> dict:
    subprocess.run(
        [_NODE, str(_ROOT / "frontend" / "runtime-config.js")],
        env={**os.environ, "DIST_DIR": str(tmp_path), **env},
        check=True,
        capture_output=True,
    )
    return json.loads((tmp_path / "serve.json").read_text())


def _spa_policy(config: dict) -> tuple[str, str]:
    entry = next(
        h for h in config["headers"][0]["headers"]
        if h["key"].startswith("Content-Security-Policy")
    )
    return entry["key"], entry["value"]


@_needs_node
def test_spa_connect_src_names_the_api_origin(tmp_path):
    """Without this the SPA's own CSP would block every API call."""
    _, policy = _spa_policy(_generate(tmp_path, VITE_API_URL="https://api.example.com/base"))
    assert "connect-src 'self' https://api.example.com" in policy


@_needs_node
def test_spa_policy_blocks_inline_scripts(tmp_path):
    """The build emits external modules only, so no script needs 'unsafe-inline'."""
    _, policy = _spa_policy(_generate(tmp_path, VITE_API_URL="https://api.example.com"))
    script_src = next(d for d in policy.split("; ") if d.startswith("script-src"))
    assert "'unsafe-inline'" not in script_src
    assert "'unsafe-eval'" not in policy


@_needs_node
def test_spa_policy_allows_the_google_font_it_actually_loads(tmp_path):
    """index.html links fonts.googleapis.com; the policy must match reality."""
    _, policy = _spa_policy(_generate(tmp_path, VITE_API_URL="https://api.example.com"))
    index = (_ROOT / "frontend" / "index.html").read_text()
    if "fonts.googleapis.com" in index:
        assert "https://fonts.googleapis.com" in policy
    if "fonts.gstatic.com" in index:
        assert "https://fonts.gstatic.com" in policy


@_needs_node
def test_spa_policy_hardens_the_usual_suspects(tmp_path):
    _, policy = _spa_policy(_generate(tmp_path, VITE_API_URL="https://api.example.com"))
    for directive in (
        "object-src 'none'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ):
        assert directive in policy


@_needs_node
def test_report_only_mode_switches_the_header(tmp_path):
    """Lets one deploy surface violations before enforcing."""
    key, _ = _spa_policy(
        _generate(tmp_path, VITE_API_URL="https://api.example.com", CSP_REPORT_ONLY="true")
    )
    assert key == "Content-Security-Policy-Report-Only"


@_needs_node
def test_unset_api_url_falls_back_to_self(tmp_path):
    _, policy = _spa_policy(_generate(tmp_path, VITE_API_URL=""))
    assert "connect-src 'self';" in policy


@_needs_node
def test_config_js_is_still_written(tmp_path):
    _generate(tmp_path, VITE_API_URL="https://api.example.com")
    assert "window.__DP_API__" in (tmp_path / "config.js").read_text()
