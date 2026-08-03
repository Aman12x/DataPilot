"""
The upload handler must not block the event loop.

It previously ran pd.read_csv, the temp-CSV round trip and the DuckDB load
synchronously inside `async def`. The backend runs with --workers 1, so the
entire API stopped serving for the duration of every upload.
"""
import asyncio
import inspect
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.api.routes import upload as upload_route


# ── The blocking work is isolated and callable off-loop ───────────────────────


def test_parse_and_build_are_plain_sync_functions():
    """They must stay sync so asyncio.to_thread can offload them."""
    assert not inspect.iscoroutinefunction(upload_route._parse_dataframe)
    assert not inspect.iscoroutinefunction(upload_route._build_duckdb)


def test_handler_offloads_both_blocking_stages():
    src = inspect.getsource(upload_route.upload_file)
    assert "asyncio.to_thread(_parse_dataframe" in src
    assert "asyncio.to_thread(_build_duckdb" in src
    # The synchronous forms must not have crept back in.
    assert "pd.read_csv(io.BytesIO" not in src
    assert "duckdb.connect(" not in src


def test_parse_dataframe_normalises_columns():
    raw = b"User ID,Revenue (USD)\n1,10\n2,20\n"
    df = upload_route._parse_dataframe(raw, ".csv")
    assert list(df.columns) == ["user_id", "revenue_usd"]
    assert len(df) == 2


def test_build_duckdb_writes_queryable_tables(tmp_path):
    import duckdb

    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "revenue": [10, 20, 30],
    })
    db = str(tmp_path / "u.db")
    upload_route._build_duckdb(df, db)

    con = duckdb.connect(db, read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        assert tables, "no tables written"
        total = sum(
            con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables
        )
        assert total > 0
    finally:
        con.close()


def test_build_duckdb_closes_the_connection_on_failure(tmp_path):
    """The old code only closed on the success path, leaking the file lock."""
    db = str(tmp_path / "fail.db")
    # A frame whose to_csv raises, so the failure happens mid-build.
    class _Exploding(pd.DataFrame):
        @property
        def _constructor(self):
            return _Exploding

        def to_csv(self, *a, **k):  # noqa: D102
            raise RuntimeError("boom")

    bad = _Exploding({"user_id": [1], "date": ["2024-01-01"], "revenue": [1]})
    with pytest.raises(Exception):
        upload_route._build_duckdb(bad, db)

    # If the handle leaked, opening for write would fail on a locked file.
    import duckdb

    con = duckdb.connect(db)
    con.close()


# ── The loop stays responsive while a slow upload runs ────────────────────────


class _FakeUpload:
    """Minimal stand-in for starlette's UploadFile."""

    def __init__(self, data: bytes, filename: str = "d.csv") -> None:
        self.filename = filename
        self._data = data
        self._pos = 0

    async def read(self, size: int = -1) -> bytes:
        chunk = self._data[self._pos : self._pos + size] if size >= 0 else self._data[self._pos :]
        self._pos += len(chunk)
        return chunk


def test_event_loop_keeps_serving_during_a_slow_upload(monkeypatch, tmp_path):
    """The regression test that matters: drive the real handler and watch a
    heartbeat. With the old synchronous code the loop was pinned and could not
    advance until the upload finished."""
    monkeypatch.setattr(upload_route, "_UPLOAD_DIR", str(tmp_path))

    def slow_build(df, db_path):
        time.sleep(1.0)  # stands in for a large pandas + DuckDB write

    monkeypatch.setattr(upload_route, "_build_duckdb", slow_build)

    csv = b"user_id,date,revenue\n" + b"".join(
        f"{i},2024-01-0{i % 9 + 1},{i}\n".encode() for i in range(500)
    )

    async def scenario():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        hb = asyncio.create_task(heartbeat())
        fake_request = SimpleNamespace(headers={}, client=SimpleNamespace(host="203.0.113.5"))
        result = await upload_route.upload_file(
            request=fake_request, file=_FakeUpload(csv), current_user={"user_id": "u1"}
        )
        hb.cancel()
        return ticks, result

    ticks, result = asyncio.run(scenario())
    assert result["row_count"] == 500
    # ~20 ticks fit in the 1s build. Blocking would have yielded 0-1.
    assert ticks > 5, f"event loop was blocked: only {ticks} heartbeat ticks"


def test_read_capped_rejects_before_buffering_everything():
    """Oversized bodies must be refused mid-stream, not after a full read."""
    from fastapi import HTTPException

    served = 0

    class _HugeUpload:
        async def read(self, size: int = -1) -> bytes:
            nonlocal served
            served += size
            return b"x" * size

    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload_route._read_capped(_HugeUpload(), 5 * 1024 * 1024))
    assert exc.value.status_code == 413
    # Stopped near the cap rather than consuming an unbounded stream.
    assert served <= 6 * 1024 * 1024, f"read {served} bytes before rejecting"


def test_read_capped_returns_small_bodies_whole():
    class _SmallUpload:
        def __init__(self) -> None:
            self.data = b"a,b\n1,2\n"
            self.pos = 0

        async def read(self, size: int = -1) -> bytes:
            chunk = self.data[self.pos : self.pos + size]
            self.pos += len(chunk)
            return chunk

    out = asyncio.run(upload_route._read_capped(_SmallUpload(), 1024))
    assert out == b"a,b\n1,2\n"
