"""
Retention and backups.

graph.db serialises full DataFrames into every checkpoint -- 331 MB across 507
checkpoints locally, one run accounting for 45 MB -- and nothing pruned it. It
shares a fixed-size volume with auth.db, so a full disk takes user accounts
down with it. There were also no backups at all.
"""
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from backend.api import retention

_GREGORIAN_OFFSET = 0x01B21DD213814000


def _uuid6_at(when: datetime) -> str:
    """Build a UUIDv6 encoding `when`, mirroring what LangGraph generates."""
    ts = int(when.timestamp() * 1e7) + _GREGORIAN_OFFSET
    time_high = (ts >> 28) & 0xFFFFFFFF
    time_mid = (ts >> 12) & 0xFFFF
    time_low = ts & 0x0FFF
    node = uuid.uuid4().int & 0xFFFFFFFFFFFF
    i = (time_high << 96) | (time_mid << 80) | (6 << 76) | (time_low << 64) | (0b10 << 62) | node
    return str(uuid.UUID(int=i))


def _graph_db(path, threads):
    """threads: {thread_id: [datetime, ...]}

    WAL, because that is how main.py opens graph.db -- and VACUUM behaves
    differently under WAL, which an earlier version of this file missed.
    """
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,"
        " parent_checkpoint_id TEXT, type TEXT, checkpoint BLOB, metadata BLOB)"
    )
    con.execute(
        "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT,"
        " task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
    )
    for thread_id, times in threads.items():
        for when in times:
            cid = _uuid6_at(when)
            con.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
                (thread_id, "", cid, None, "msgpack", b"x" * 512, b"{}"),
            )
            con.execute(
                "INSERT INTO writes VALUES (?,?,?,?,?,?,?,?)",
                (thread_id, "", cid, "t", 0, "c", "msgpack", b"y" * 128),
            )
    con.commit()
    con.close()


# ── UUIDv6 timestamp extraction ───────────────────────────────────────────────


def test_checkpoint_time_round_trips():
    when = datetime(2026, 3, 21, 7, 16, 4, tzinfo=timezone.utc)
    got = retention.checkpoint_time(_uuid6_at(when))
    assert got is not None
    assert abs((got - when).total_seconds()) < 1


def test_checkpoint_time_rejects_other_uuid_versions():
    assert retention.checkpoint_time(str(uuid.uuid4())) is None
    assert retention.checkpoint_time("not-a-uuid") is None
    assert retention.checkpoint_time(None) is None


# ── Checkpoint pruning ────────────────────────────────────────────────────────


def test_prunes_only_threads_past_the_cutoff(tmp_path):
    now = datetime.now(timezone.utc)
    db = str(tmp_path / "graph.db")
    _graph_db(db, {
        "old": [now - timedelta(days=90), now - timedelta(days=89)],
        "recent": [now - timedelta(days=1)],
    })

    result = retention.prune_checkpoints(db, older_than_days=30)
    assert result["threads"] == 1

    con = sqlite3.connect(db)
    remaining = {r[0] for r in con.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    writes = {r[0] for r in con.execute("SELECT DISTINCT thread_id FROM writes")}
    con.close()
    assert remaining == {"recent"}
    assert writes == {"recent"}, "writes rows were orphaned"


def test_a_resumed_thread_is_kept_on_its_newest_checkpoint(tmp_path):
    """A long-running analysis started months ago but still active must survive."""
    now = datetime.now(timezone.utc)
    db = str(tmp_path / "graph.db")
    _graph_db(db, {"long": [now - timedelta(days=200), now - timedelta(hours=2)]})

    assert retention.prune_checkpoints(db, older_than_days=30)["threads"] == 0
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 2
    con.close()


def test_prune_is_a_noop_when_nothing_is_stale(tmp_path):
    now = datetime.now(timezone.utc)
    db = str(tmp_path / "graph.db")
    _graph_db(db, {"a": [now]})
    assert retention.prune_checkpoints(db, older_than_days=30) == {
        "threads": 0, "checkpoints": 0, "writes": 0,
    }


def test_prune_handles_a_missing_database(tmp_path):
    assert retention.prune_checkpoints(str(tmp_path / "nope.db"))["threads"] == 0


def test_prune_guest_uploads_removes_only_stale_guest_dirs(tmp_path):
    import os
    import time as _time

    uploads = tmp_path / "uploads"
    old = _time.time() - 72 * 3600

    stale_guest = uploads / "guest-aaaa"
    fresh_guest = uploads / "guest-bbbb"
    user_dir = uploads / "user-cccc"
    for d in (stale_guest, fresh_guest, user_dir):
        d.mkdir(parents=True)
        (d / "u.db").write_bytes(b"x" * 10)
    # Backdate the stale guest dir and, crucially, the registered user's dir:
    # age alone must never be enough to sweep a non-guest directory.
    for d in (stale_guest, user_dir):
        os.utime(d / "u.db", (old, old))
        os.utime(d, (old, old))

    result = retention.prune_guest_uploads(str(uploads), older_than_hours=48)

    assert result["dirs"] == 1
    assert not stale_guest.exists()
    assert fresh_guest.exists()
    assert user_dir.exists()


def test_prune_guest_uploads_handles_missing_dir(tmp_path):
    assert retention.prune_guest_uploads(str(tmp_path / "nope"))["dirs"] == 0


def test_prune_survives_unparseable_checkpoint_ids(tmp_path):
    """A row with a non-UUIDv6 id must be skipped, not crash the pass."""
    now = datetime.now(timezone.utc)
    db = str(tmp_path / "graph.db")
    _graph_db(db, {"good": [now - timedelta(days=90)]})
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO checkpoints VALUES ('weird','', 'not-a-uuid', NULL,'msgpack',?,?)",
        (b"x", b"{}"),
    )
    con.commit()
    con.close()

    retention.prune_checkpoints(db, older_than_days=30)
    con = sqlite3.connect(db)
    left = {r[0] for r in con.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    con.close()
    assert "weird" in left, "unknown-age rows should be kept, not silently dropped"


# ── Run history and tokens ────────────────────────────────────────────────────


def test_prune_runs_respects_the_cutoff(tmp_path):
    db = str(tmp_path / "mem.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (run_id TEXT, timestamp TEXT)")
    now = datetime.now(timezone.utc)
    con.executemany("INSERT INTO runs VALUES (?,?)", [
        ("old", (now - timedelta(days=400)).isoformat()),
        ("new", now.isoformat()),
    ])
    con.commit(); con.close()

    assert retention.prune_runs(db, older_than_days=180) == 1
    con = sqlite3.connect(db)
    assert [r[0] for r in con.execute("SELECT run_id FROM runs")] == ["new"]
    con.close()


def test_prune_auth_tokens_clears_spent_and_expired(tmp_path):
    db = str(tmp_path / "auth.db")
    now = datetime.now(timezone.utc)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE revoked_tokens (jti TEXT PRIMARY KEY, revoked_at TEXT NOT NULL)")
    con.execute("CREATE TABLE password_reset_tokens (token TEXT, used INT, expires_at TEXT)")
    con.execute("CREATE TABLE email_verification_tokens (token TEXT, used INT, expires_at TEXT)")
    # Old enough that the refresh token it names has expired anyway.
    con.execute("INSERT INTO revoked_tokens VALUES ('a', ?)", ((now - timedelta(days=200)).isoformat(),))
    # Recent: the token is still live, so the revocation must be kept.
    con.execute("INSERT INTO revoked_tokens VALUES ('b', ?)", ((now - timedelta(days=1)).isoformat(),))
    con.execute("INSERT INTO password_reset_tokens VALUES ('t', 1, ?)", ((now + timedelta(days=1)).isoformat(),))
    con.commit(); con.close()

    removed = retention.prune_auth_tokens(db)
    assert removed["revoked_tokens"] == 1
    assert removed["password_reset_tokens"] == 1

    con = sqlite3.connect(db)
    # The still-valid revocation must survive: dropping it would un-revoke a token.
    assert [r[0] for r in con.execute("SELECT jti FROM revoked_tokens")] == ["b"]
    con.close()


# ── Backups ───────────────────────────────────────────────────────────────────


def test_backup_produces_a_readable_snapshot(tmp_path):
    src = str(tmp_path / "auth.db")
    con = sqlite3.connect(src)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE users (id TEXT)")
    con.executemany("INSERT INTO users VALUES (?)", [(str(i),) for i in range(50)])
    con.commit()

    dest = retention.backup_database(src, str(tmp_path / "backups"))
    con.close()

    assert dest and os.path.exists(dest)
    snap = sqlite3.connect(dest)
    assert snap.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 50
    snap.close()
    # VACUUM INTO output is self-contained; a stray -wal would be a trap.
    assert not os.path.exists(dest + "-wal")


def test_backup_snapshots_a_live_database(tmp_path):
    """The connection stays open and in WAL while the snapshot is taken."""
    src = str(tmp_path / "live.db")
    con = sqlite3.connect(src)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (a)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    try:
        dest = retention.backup_database(src, str(tmp_path / "b"))
        snap = sqlite3.connect(dest)
        assert snap.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
        snap.close()
    finally:
        con.close()


def test_backup_retention_keeps_only_the_newest(tmp_path):
    src = str(tmp_path / "auth.db")
    con = sqlite3.connect(src)
    con.execute("CREATE TABLE t (a)")
    con.commit(); con.close()

    out = tmp_path / "backups"
    for i in range(5):
        made = retention.backup_database(src, str(out), keep=3)
        # Distinct filenames despite second-resolution stamps.
        os.rename(made, out / f"auth-2026010{i}T000000Z.db")
    retention.backup_database(src, str(out), keep=3)

    kept = sorted(p.name for p in out.glob("auth-*.db"))
    assert len(kept) == 3, kept


def test_backup_of_a_missing_database_returns_none(tmp_path):
    assert retention.backup_database(str(tmp_path / "gone.db"), str(tmp_path)) is None


# ── Whole pass ────────────────────────────────────────────────────────────────


def test_run_maintenance_prunes_and_backs_up(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    graph = str(tmp_path / "graph.db")
    _graph_db(graph, {"old": [now - timedelta(days=90)], "new": [now]})

    auth = str(tmp_path / "auth.db")
    con = sqlite3.connect(auth); con.execute("CREATE TABLE users (id TEXT)"); con.commit(); con.close()
    mem = str(tmp_path / "mem.db")
    con = sqlite3.connect(mem); con.execute("CREATE TABLE runs (run_id TEXT, timestamp TEXT)"); con.commit(); con.close()

    monkeypatch.setenv("GRAPH_DB_PATH", graph)
    monkeypatch.setenv("AUTH_DB_PATH", auth)
    monkeypatch.setenv("MEMORY_DB_PATH", mem)
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))

    report = retention.run_maintenance()
    assert report["checkpoints"]["threads"] == 1
    assert len(report["backups"]) == 2, report["backups"]
    # graph.db is deliberately NOT backed up: it is the thing filling the disk.
    assert not any("graph" in b for b in report["backups"])


def test_run_maintenance_reclaims_space(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    graph = str(tmp_path / "graph.db")
    _graph_db(graph, {f"t{i}": [now - timedelta(days=90)] for i in range(200)})
    before = os.path.getsize(graph)

    monkeypatch.setenv("GRAPH_DB_PATH", graph)
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "none.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "none2.db"))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "b"))

    retention.run_maintenance()
    assert os.path.getsize(graph) < before, "VACUUM did not reclaim space"


def test_run_maintenance_never_raises(tmp_path, monkeypatch):
    """A failed pass must not take down the background task."""
    monkeypatch.setenv("GRAPH_DB_PATH", "/nonexistent/dir/graph.db")
    monkeypatch.setenv("AUTH_DB_PATH", "/nonexistent/dir/auth.db")
    monkeypatch.setenv("MEMORY_DB_PATH", "/nonexistent/dir/mem.db")
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "b"))
    assert isinstance(retention.run_maintenance(), dict)


# ── Regressions found by deploying and reading the logs ───────────────────────


def test_vacuum_reclaims_space_under_wal(tmp_path):
    """VACUUM alone does not shrink a WAL database with a live reader.

    The rebuilt file goes into the WAL, and the main file keeps its old size
    until a checkpoint folds it back. SQLite auto-checkpoints when the LAST
    connection closes, which is why this needs a second connection held open --
    production always has one, since the LangGraph checkpointer keeps graph.db
    open for the life of the process. The first version of vacuum() omitted the
    checkpoint and reported 0 bytes reclaimed in production while the disk
    stayed full.
    """
    db = str(tmp_path / "wal.db")
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE t (a, b)")
    writer.executemany("INSERT INTO t VALUES (?,?)", [(i, b"x" * 4000) for i in range(2000)])
    writer.commit()
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = os.path.getsize(db)
    writer.execute("DELETE FROM t")
    writer.commit()

    # Held open across the vacuum, exactly as the checkpointer holds graph.db.
    holder = sqlite3.connect(db)
    holder.execute("SELECT COUNT(*) FROM t").fetchone()
    try:
        freed = retention.vacuum(db)
    finally:
        holder.close()
        writer.close()

    assert freed > 0, "vacuum reported no reclamation under WAL"
    assert os.path.getsize(db) < before / 2


def test_revoked_tokens_are_pruned_on_revoked_at(tmp_path):
    """The table is (jti, revoked_at) -- there is no expires_at column.

    Querying one made the DELETE raise OperationalError, which was swallowed,
    so the table that grows on every /auth/refresh was never pruned at all.
    """
    db = str(tmp_path / "auth.db")
    now = datetime.now(timezone.utc)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE revoked_tokens (jti TEXT PRIMARY KEY, revoked_at TEXT NOT NULL)")
    con.execute("INSERT INTO revoked_tokens VALUES ('stale', ?)", ((now - timedelta(days=400)).isoformat(),))
    con.execute("INSERT INTO revoked_tokens VALUES ('fresh', ?)", (now.isoformat(),))
    con.commit(); con.close()

    removed = retention.prune_auth_tokens(db)
    assert "revoked_tokens" in removed, "prune silently skipped the table"
    assert removed["revoked_tokens"] == 1

    con = sqlite3.connect(db)
    assert [r[0] for r in con.execute("SELECT jti FROM revoked_tokens")] == ["fresh"]
    con.close()


def test_revocation_outlives_the_token_it_revokes():
    """Pruning too early would un-revoke a token that is still valid."""
    from backend.api.deps import REFRESH_TOKEN_EXPIRE_DAYS

    assert retention.REVOKED_TOKEN_RETENTION_DAYS > REFRESH_TOKEN_EXPIRE_DAYS


def test_schema_drift_is_logged_not_swallowed(tmp_path, caplog):
    """A missing table must be visible; a debug-level skip is how this hid."""
    import logging

    db = str(tmp_path / "auth.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE unrelated (x)")
    con.commit(); con.close()

    with caplog.at_level(logging.WARNING):
        retention.prune_auth_tokens(db)
    assert any("skipped" in r.message for r in caplog.records)


def test_vacuum_runs_when_an_earlier_pass_left_free_pages(tmp_path, monkeypatch):
    """Self-healing: reclaim space freed by a previous pass that did not vacuum.

    Production hit exactly this -- the first pass deleted 267 checkpoints but
    reclaimed nothing (the WAL bug), and the next pass had no deletions of its
    own, so the space stayed stranded.
    """
    now = datetime.now(timezone.utc)
    graph = str(tmp_path / "graph.db")
    _graph_db(graph, {f"t{i}": [now] for i in range(300)})  # all recent: nothing to prune

    con = sqlite3.connect(graph)
    con.execute("DELETE FROM checkpoints")  # free pages, no vacuum
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()

    assert retention.free_bytes(graph) > 0
    monkeypatch.setattr(retention, "VACUUM_FREE_BYTES", 1)
    monkeypatch.setenv("GRAPH_DB_PATH", graph)
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "m.db"))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "b"))

    report = retention.run_maintenance()
    assert report["checkpoints"]["checkpoints"] == 0, "nothing should have been pruned"
    assert "graph_bytes_reclaimed" in report, "vacuum did not run on stranded free pages"
    assert retention.free_bytes(graph) == 0


def test_free_bytes_is_zero_for_a_compact_database(tmp_path):
    db = str(tmp_path / "c.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (a)")
    con.commit(); con.close()
    assert retention.free_bytes(db) == 0
    assert retention.free_bytes(str(tmp_path / "missing.db")) == 0


def test_disk_report_breaks_down_volume_usage(tmp_path, monkeypatch):
    """Without per-file sizes there is no way to tell what is filling the disk."""
    graph = str(tmp_path / "graph.db")
    _graph_db(graph, {"a": [datetime.now(timezone.utc)]})
    # Pad past a megabyte so the rounded report is meaningful.
    con = sqlite3.connect(graph)
    con.executemany(
        "INSERT INTO checkpoints VALUES ('pad','',?,NULL,'msgpack',?,?)",
        [(_uuid6_at(datetime.now(timezone.utc)), b"x" * 100_000, b"{}") for _ in range(20)],
    )
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "u1.db").write_bytes(b"x" * 2_000_000)

    monkeypatch.setenv("GRAPH_DB_PATH", graph)
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "mem.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))

    rep = retention.disk_report()
    assert rep["uploads"] == pytest.approx(2.0, abs=0.2)
    assert rep["graph"] > 0
    assert rep["auth"] == 0        # missing file, not an error
    assert "graph_free" in rep


class _FakeS3:
    """Minimal boto3-shaped S3 client for off-box backup tests."""

    def __init__(self, fail_upload: bool = False):
        self.objects: dict[str, str] = {}
        self.fail_upload = fail_upload

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        if self.fail_upload:
            raise RuntimeError("network down")
        self.objects[key] = path
        self.last_extra = ExtraArgs

    def list_objects_v2(self, Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in self.objects if k.startswith(Prefix)]}

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)


def _offbox_env(monkeypatch, bucket="bkt"):
    monkeypatch.setenv("BACKUP_S3_BUCKET", bucket)
    monkeypatch.setenv("BACKUP_S3_PREFIX", "dp/")
    monkeypatch.delenv("BACKUP_S3_SSE", raising=False)


def test_offbox_disabled_without_bucket(monkeypatch):
    monkeypatch.delenv("BACKUP_S3_BUCKET", raising=False)
    assert retention.upload_backups_offbox(["/x/auth-1.db"]) == {"enabled": False}


def test_offbox_uploads_and_prunes_per_stem(monkeypatch, tmp_path):
    _offbox_env(monkeypatch)
    fake = _FakeS3()
    # Pre-existing remote copies: 8 old auth snapshots and one memory snapshot.
    for i in range(8):
        fake.objects[f"dp/auth-202601{i:02d}.db"] = "old"
    fake.objects["dp/memory-20260101.db"] = "old"
    monkeypatch.setattr(retention, "_offbox_client", lambda: fake)

    local = tmp_path / "auth-20260801.db"
    local.write_bytes(b"snapshot")
    report = retention.upload_backups_offbox([str(local)], keep=7)

    assert report["uploaded"] == ["dp/auth-20260801.db"]
    auth_keys = sorted(k for k in fake.objects if k.startswith("dp/auth-"))
    assert len(auth_keys) == 7                      # 9 total → newest 7 kept
    assert "dp/auth-20260801.db" in auth_keys       # the fresh one survives
    assert "dp/memory-20260101.db" in fake.objects  # other stems untouched
    assert report["errors"] == []


def test_offbox_upload_failure_is_recorded_not_raised(monkeypatch, tmp_path):
    _offbox_env(monkeypatch)
    monkeypatch.setattr(retention, "_offbox_client", lambda: _FakeS3(fail_upload=True))
    local = tmp_path / "auth-20260801.db"
    local.write_bytes(b"snapshot")
    report = retention.upload_backups_offbox([str(local)])
    assert report["uploaded"] == []
    assert report["errors"] and "upload" in report["errors"][0]


def test_offbox_sse_is_passed_through(monkeypatch, tmp_path):
    _offbox_env(monkeypatch)
    monkeypatch.setenv("BACKUP_S3_SSE", "AES256")
    fake = _FakeS3()
    monkeypatch.setattr(retention, "_offbox_client", lambda: fake)
    local = tmp_path / "memory-20260801.db"
    local.write_bytes(b"snapshot")
    retention.upload_backups_offbox([str(local)])
    assert fake.last_extra == {"ServerSideEncryption": "AES256"}


def test_maintenance_pass_includes_offbox_report(monkeypatch, tmp_path):
    fake = _FakeS3()
    monkeypatch.setattr(retention, "_offbox_client", lambda: fake)
    _offbox_env(monkeypatch)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    import sqlite3
    for name in ("memory.db", "auth.db"):
        sqlite3.connect(tmp_path / name).execute("CREATE TABLE t(x)").connection.commit()

    report = retention.run_maintenance()
    assert report["offbox"]["enabled"] is True
    assert len(report["offbox"]["uploaded"]) == 2   # auth + memory snapshots
