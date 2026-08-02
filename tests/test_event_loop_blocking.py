"""
tests/test_event_loop_blocking.py — nothing blocking may run on the event loop.

The backend runs `--workers 1`. One synchronous SQLite read inside an `async def`
freezes every other request for its duration, and these are not rare paths:
`_workspace_of_run` ran on *every* authorised run access, and `graph.get_state`
ran inside the SSE generator.

The rule is easy to state and easy to forget, which is why it is enforced by a
scan rather than by review. An earlier hand-grep for `graph.get_state` found six
sites; this scan found fifty-six.

A sync `def` route handler is fine and deliberately not flagged — FastAPI runs
those in its own threadpool, which is why `list_runs` and the `deps.py`
dependencies were already safe.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_API = _ROOT / "backend" / "api"

# Attribute calls that reach SQLite or a checkpointer.
_BLOCKING_ATTRS = {"get_state", "update_state"}
_BLOCKING_MODULES = {"workspace_store", "org_store"}

# Names imported from a store module and then called bare.
_STORE_MODULE_HINTS = ("store", "org_store", "workspace_store")

# Startup runs before the server accepts traffic, so blocking there costs
# nothing and keeps the boot sequence readable.
_EXEMPT = {"backend/api/main.py"}


class _Finder(ast.NodeVisitor):
    def __init__(self, imported: dict[str, str]):
        self.imported = imported
        self.stack: list[str] = []
        self.hits: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node):
        self.stack.append("sync")
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.stack.append("async")
        self.generic_visit(node)
        self.stack.pop()

    def visit_Lambda(self, node):
        # A lambda handed to to_thread runs off the loop; do not flag its body.
        self.stack.append("sync")
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node):
        if self.stack and self.stack[-1] == "async":
            func = node.func
            if isinstance(func, ast.Attribute):
                base = func.value.id if isinstance(func.value, ast.Name) else ""
                if func.attr in _BLOCKING_ATTRS or base in _BLOCKING_MODULES:
                    self.hits.append((node.lineno, f"{base}.{func.attr}()"))
            elif isinstance(func, ast.Name) and func.id in self.imported:
                self.hits.append((node.lineno, f"{func.id}()"))
        self.generic_visit(node)


def _imported_store_names(tree: ast.Module) -> dict[str, str]:
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(h in node.module for h in _STORE_MODULE_HINTS):
                for alias in node.names:
                    names[alias.asname or alias.name] = node.module
    return names


def _scan(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text())
    finder = _Finder(_imported_store_names(tree))
    finder.visit(tree)
    rel = path.relative_to(_ROOT)
    return [f"{rel}:{lineno} {what}" for lineno, what in finder.hits]


def _api_files() -> list[pathlib.Path]:
    return [p for p in sorted(_API.rglob("*.py"))
            if str(p.relative_to(_ROOT)) not in _EXEMPT]


def test_no_blocking_store_calls_inside_async_functions():
    offenders: list[str] = []
    for path in _api_files():
        offenders.extend(_scan(path))
    assert not offenders, (
        "blocking call on the event loop — wrap it in asyncio.to_thread, or make "
        "the handler a sync `def` so FastAPI runs it in its threadpool:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_actually_detects_a_blocking_call(tmp_path):
    """Without this the sweep above is unfalsifiable.

    A scanner whose visitor silently stopped matching would report zero
    offenders and read exactly like a clean codebase.
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from auth.store import get_user_by_id\n"
        "async def handler(graph, run_id):\n"
        "    state = graph.get_state({})\n"
        "    return get_user_by_id('u')\n"
    )
    tree = ast.parse(sample.read_text())
    finder = _Finder(_imported_store_names(tree))
    finder.visit(tree)
    found = {what for _, what in finder.hits}
    assert found == {"graph.get_state()", "get_user_by_id()"}, found


def test_sync_handlers_are_not_flagged(tmp_path):
    """FastAPI runs a sync `def` endpoint in its threadpool — already off-loop.

    Flagging those would push people to add `async` and a `to_thread` where
    neither is needed.
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from auth.store import get_user_by_id\n"
        "def handler(graph, run_id):\n"
        "    state = graph.get_state({})\n"
        "    return get_user_by_id('u')\n"
    )
    tree = ast.parse(sample.read_text())
    finder = _Finder(_imported_store_names(tree))
    finder.visit(tree)
    assert finder.hits == []


def test_to_thread_wrapped_calls_are_not_flagged(tmp_path):
    """The fix must actually satisfy the scan, or it will be worked around."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import asyncio\n"
        "from auth.store import get_user_by_id\n"
        "async def handler(graph, run_id):\n"
        "    state = await asyncio.to_thread(graph.get_state, {})\n"
        "    return await asyncio.to_thread(get_user_by_id, 'u')\n"
    )
    tree = ast.parse(sample.read_text())
    finder = _Finder(_imported_store_names(tree))
    finder.visit(tree)
    assert finder.hits == []


@pytest.mark.parametrize("module", ["runs", "workspace", "orgs", "auth"])
def test_the_routes_that_were_fixed_stay_fixed(module):
    """Named individually so a regression points at the file, not the whole API."""
    offenders = _scan(_API / "routes" / f"{module}.py")
    assert not offenders, "\n  ".join(offenders)
