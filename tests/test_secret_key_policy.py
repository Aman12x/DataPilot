"""
SECRET_KEY strength enforcement.

deps.py accepted any non-empty SECRET_KEY, so `SECRET_KEY=x` signed every JWT in
production. HS256 is HMAC-SHA256, which RFC 7518 requires be keyed with at least
256 bits.

The boot-time behaviour is checked by re-importing deps in a subprocess: the
check runs at import, and this test module has already imported it.
"""
import subprocess
import sys
import textwrap

import pytest

from backend.api.deps import (
    _MIN_SECRET_KEY_LENGTH,
    validate_secret_key,
)

_GOOD = "9f2c1b7a4e6d8f0a3c5e7b9d1f3a5c7e9b1d3f5a7c9e1b3d5f7a9c1e3b5d7f90"


# ── validate_secret_key ───────────────────────────────────────────────────────


def test_a_real_random_key_has_no_problems():
    assert validate_secret_key(_GOOD) == []


def test_short_key_is_rejected():
    problems = validate_secret_key("x")
    assert any("at least" in p for p in problems)


def test_key_one_char_below_the_minimum_is_rejected():
    """Boundary: 31 random-looking chars must still fail on length."""
    short = "abcdefghij0123456789ABCDEFGHIJK"
    assert len(short) == _MIN_SECRET_KEY_LENGTH - 1
    assert any("at least" in p for p in validate_secret_key(short))


def test_key_at_the_minimum_length_is_accepted():
    key = "abcdefghij0123456789ABCDEFGHIJKL"
    assert len(key) == _MIN_SECRET_KEY_LENGTH
    assert validate_secret_key(key) == []


def test_long_but_low_entropy_key_is_rejected():
    """Length alone is not strength."""
    problems = validate_secret_key("a" * 64)
    assert any("distinct characters" in p for p in problems)

    problems = validate_secret_key("ab" * 32)
    assert any("distinct characters" in p for p in problems)


@pytest.mark.parametrize(
    "placeholder",
    [
        "change-me-to-a-long-random-string",  # the value shipped in .env.example
        "CHANGE-ME-TO-A-LONG-RANDOM-STRING",
        "  change-me-to-a-long-random-string  ",
    ],
)
def test_known_placeholders_are_rejected(placeholder):
    problems = validate_secret_key(placeholder)
    assert any("placeholder" in p for p in problems)


def test_env_example_placeholder_would_be_rejected():
    """A copied .env must fail loudly on deploy, not sign tokens with a known key."""
    from pathlib import Path

    line = next(
        ln for ln in Path(__file__).resolve().parents[1].joinpath(".env.example")
        .read_text().splitlines()
        if ln.startswith("SECRET_KEY=")
    )
    assert validate_secret_key(line.split("=", 1)[1]) != []


def test_empty_key_reports_a_problem():
    assert validate_secret_key("") != []


# ── Boot behaviour ────────────────────────────────────────────────────────────


def _boot(env: dict[str, str]) -> subprocess.CompletedProcess:
    """Import deps in a clean interpreter with the given environment."""
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, ".")
        import backend.api.deps as d
        print("BOOTED", len(d.SECRET_KEY))
        """
    )
    import os

    base = {k: v for k, v in os.environ.items() if k not in ("SECRET_KEY", "ENV", "RAILWAY_ENVIRONMENT")}
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**base, **env},
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )


def test_weak_key_is_fatal_in_production():
    result = _boot({"ENV": "production", "SECRET_KEY": "x"})
    assert result.returncode != 0
    assert "SECRET_KEY" in result.stderr
    assert "at least" in result.stderr


def test_placeholder_key_is_fatal_in_production():
    result = _boot({"ENV": "production", "SECRET_KEY": "change-me-to-a-long-random-string"})
    assert result.returncode != 0
    assert "placeholder" in result.stderr


def test_strong_key_boots_in_production():
    result = _boot({"ENV": "production", "SECRET_KEY": _GOOD})
    assert result.returncode == 0, result.stderr
    assert "BOOTED" in result.stdout


def test_weak_key_is_fatal_on_a_non_production_railway_environment():
    """_IS_PRODUCTION only matches ("production", "prod").

    A Railway service named "staging" is still a real deployment signing real
    tokens, so the key check must not be skipped there.
    """
    result = _boot({"RAILWAY_ENVIRONMENT": "staging", "SECRET_KEY": "x"})
    assert result.returncode != 0
    assert "SECRET_KEY" in result.stderr


def test_missing_key_is_fatal_when_deployed():
    result = _boot({"ENV": "production"})
    assert result.returncode != 0
    assert "must be set" in result.stderr


def test_weak_key_only_warns_in_local_dev():
    """Local development must stay frictionless."""
    result = _boot({"ENV": "development", "SECRET_KEY": "x"})
    assert result.returncode == 0, result.stderr
    assert "BOOTED" in result.stdout


def test_missing_key_generates_one_in_local_dev():
    result = _boot({"ENV": "development"})
    assert result.returncode == 0, result.stderr
    assert "BOOTED 64" in result.stdout
