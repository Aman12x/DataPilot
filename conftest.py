import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Ensure SECRET_KEY is set before any backend module is imported.
# backend/api/deps.py exits at import time if this is missing.
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")

import pytest


@pytest.fixture(autouse=True)
def _clear_auth_rate():
    """Reset the in-memory auth rate-limiter between tests so the 10/min cap
    doesn't cause false 429s when many tests call /register or /login."""
    # The test suite adds backend/ to sys.path so the module loads as
    # api.routes.auth; the running server loads it as backend.api.routes.auth.
    # Try both to ensure the correct in-process dict is cleared.
    for mod_path in ("api.routes.auth", "backend.api.routes.auth"):
        try:
            import importlib, sys
            mod = sys.modules.get(mod_path) or importlib.import_module(mod_path)
            mod._auth_rate.clear()
        except Exception:
            pass
    yield
