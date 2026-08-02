"""
tests/test_environment_posture.py — the deployed/local decision must fail closed.

`backend/api/environment.py` replaced five independent copies of
`ENV.lower() in ("production", "prod")`. That test was an allowlist of *strict*
environments, so an environment named `staging` got a development security
posture — insecure cookies, no HSTS, unenforced CORS, tokens in response bodies
— with nothing logged to say so.

These tests pin the inversion: unrecognised names are deployed. They boot a
clean interpreter per case because the decision is made at import time, and the
five call sites capture it at import time too.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_GOOD_KEY = "3f9a1c7e5b2d8046af13e6c9b750d2481ea637c95d0fb82e4c17a9d63b508e2f"


def _probe(env: dict[str, str], expr: str) -> str:
    """Evaluate `expr` in a fresh interpreter under `env`; return its stdout."""
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        f"print({expr})\n"
    )
    base = {
        k: v for k, v in os.environ.items()
        if k not in ("ENV", "RAILWAY_ENVIRONMENT", "AUTH_RETURN_TOKENS", "CORS_ORIGINS", "APP_URL")
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**base, "SECRET_KEY": _GOOD_KEY, **env},
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# Every name that is not unambiguously a laptop or a CI runner.
DEPLOYED_NAMES = ["production", "prod", "staging", "preview", "prod-eu", "qa", "sTaGiNg"]
LOCAL_NAMES = ["development", "dev", "local", "localhost", "test", "testing", "ci", "DEV"]


@pytest.mark.parametrize("name", DEPLOYED_NAMES)
def test_unrecognised_environment_names_are_treated_as_deployed(name):
    """The regression: `staging` used to be indistinguishable from a laptop."""
    got = _probe({"ENV": name}, "__import__('backend.api.environment', fromlist=['x']).IS_DEPLOYED")
    assert got == "True", f"ENV={name!r} must be treated as deployed"


@pytest.mark.parametrize("name", LOCAL_NAMES)
def test_known_local_names_stay_local(name):
    got = _probe({"ENV": name}, "__import__('backend.api.environment', fromlist=['x']).IS_DEPLOYED")
    assert got == "False", f"ENV={name!r} must stay local so dev is frictionless"


def test_unset_environment_is_local():
    got = _probe({}, "__import__('backend.api.environment', fromlist=['x']).IS_DEPLOYED")
    assert got == "False"


def test_railway_overrides_a_local_sounding_name():
    """A Railway environment named "development" is still a real HTTPS host."""
    got = _probe(
        {"RAILWAY_ENVIRONMENT": "development"},
        "__import__('backend.api.environment', fromlist=['x']).IS_DEPLOYED",
    )
    assert got == "True"


@pytest.mark.parametrize("name", ["staging", "preview"])
def test_cookies_are_secure_and_cross_site_on_staging(name):
    """Secure and SameSite=None move together — None is ignored without Secure."""
    got = _probe(
        {"ENV": name},
        "(__import__('backend.api.cookies', fromlist=['x'])._secure(),"
        " __import__('backend.api.cookies', fromlist=['x'])._samesite())",
    )
    assert got == "(True, 'none')"


def test_cookies_stay_insecure_locally():
    """http://127.0.0.1 cannot set a Secure cookie, so local dev must not."""
    got = _probe(
        {"ENV": "development"},
        "(__import__('backend.api.cookies', fromlist=['x'])._secure(),"
        " __import__('backend.api.cookies', fromlist=['x'])._samesite())",
    )
    assert got == "(False, 'lax')"


def test_tokens_are_not_echoed_in_response_bodies_on_staging():
    got = _probe(
        {"ENV": "staging", "CORS_ORIGINS": "https://example.com"},
        "__import__('backend.api.routes.auth', fromlist=['x'])._RETURN_TOKENS",
    )
    assert got == "False"


def test_missing_cors_origins_is_fatal_on_staging():
    """Previously this only warned, leaving allow_origins=['*'] on a public host."""
    code = "import sys; sys.path.insert(0, '.')\nimport backend.api.main\n"
    base = {
        k: v for k, v in os.environ.items()
        if k not in ("ENV", "RAILWAY_ENVIRONMENT", "CORS_ORIGINS", "APP_URL")
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**base, "SECRET_KEY": _GOOD_KEY, "ENV": "staging"},
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert result.returncode != 0
    assert "CORS_ORIGINS" in result.stderr
    # The message names the environment, so the operator knows why it applied.
    assert "staging" in result.stderr


def test_hsts_is_sent_on_staging():
    code = (
        "import sys; sys.path.insert(0, '.')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "from fastapi.testclient import TestClient\n"
        "import backend.api.main as m\n"
        # No `with` — TestClient only runs the lifespan as a context manager, and
        # this asserts on middleware, which needs no startup.
        "r = TestClient(m.app).get('/health')\n"
        "print('Strict-Transport-Security' in r.headers)\n"
    )
    base = {k: v for k, v in os.environ.items() if k not in ("ENV", "RAILWAY_ENVIRONMENT")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**base, "SECRET_KEY": _GOOD_KEY, "ENV": "staging",
             "CORS_ORIGINS": "https://example.com"},
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("True"), result.stdout


def test_no_module_recomputes_the_environment_test():
    """The five duplicated copies are what let the fix miss a site.

    Any new `in ("production", "prod")` is either a sixth copy or a fail-open
    check; both should route through `environment.is_deployed()` instead.
    """
    offenders = []
    for path in (_ROOT / "backend").rglob("*.py"):
        if path.name == "environment.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if '"production"' in line and '"prod"' in line:
                offenders.append(f"{path.relative_to(_ROOT)}:{i}")
    assert not offenders, (
        "environment-name test outside environment.py:\n  " + "\n  ".join(offenders)
    )
