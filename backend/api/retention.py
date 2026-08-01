"""
backend/api/retention.py — prune old rows and snapshot the small databases.

Why this exists: graph.db serialises full query-result DataFrames into every
checkpoint. Locally that is 331 MB across 507 checkpoints — a single run can
account for 45 MB. Nothing pruned it, and auth.db shares the same fixed-size
volume, so a full disk takes user accounts down with it. There were also no
backups of any kind.

Two separate jobs:

  prune   — bound the growth of checkpoints, run history and auth tokens.
  backup  — snapshot the *small, irreplaceable* databases (accounts and run
            history). graph.db is deliberately not backed up: it is transient
            run state, it is the thing that grows, and copying it would eat
            the disk we are trying to protect.

Backups land on the same volume, so they protect against corruption, a bad
migration or an accidental delete — NOT against losing the disk. Off-box
copies need external storage and are out of scope here.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# 100-nanosecond intervals between the Gregorian epoch (1582-10-15) and Unix 0.
_GREGORIAN_OFFSET = 0x01B21DD213814000

CHECKPOINT_RETENTION_DAYS = float(os.getenv("CHECKPOINT_RETENTION_DAYS", "30"))
RUN_RETENTION_DAYS = float(os.getenv("RUN_RETENTION_DAYS", "180"))

try:  # source of truth for how long a refresh token stays valid
    from .deps import REFRESH_TOKEN_EXPIRE_DAYS as _REFRESH_DAYS
except Exception:  # pragma: no cover - deps validates SECRET_KEY at import
    _REFRESH_DAYS = 30
# A revocation is redundant once the JWT it names has expired on its own.
REVOKED_TOKEN_RETENTION_DAYS = float(
    os.getenv("REVOKED_TOKEN_RETENTION_DAYS", str(_REFRESH_DAYS + 7))
)
BACKUP_KEEP = int(os.getenv("BACKUP_KEEP", "7"))
RETENTION_INTERVAL_SEC = float(os.getenv("RETENTION_INTERVAL_SEC", str(24 * 3600)))


def _connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    return con


def checkpoint_time(checkpoint_id: str) -> datetime | None:
    """Extract the creation time encoded in a LangGraph UUIDv6 checkpoint id.

    The checkpoints table has no timestamp column, but ids are UUIDv6, whose
    high bits are a 60-bit Gregorian timestamp. Verified against the `ts` field
    inside the msgpack blob: 40/40 matched to sub-second precision. Reading the
    id avoids decoding every blob just to learn its age.
    """
    try:
        u = uuid.UUID(checkpoint_id)
    except (ValueError, AttributeError, TypeError):
        return None
    if u.version != 6:
        return None
    i = u.int
    ts100ns = (((i >> 96) & 0xFFFFFFFF) << 28) | (((i >> 80) & 0xFFFF) << 12) | ((i >> 64) & 0x0FFF)
    try:
        return datetime.fromtimestamp((ts100ns - _GREGORIAN_OFFSET) / 1e7, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


# ── Pruning ───────────────────────────────────────────────────────────────────

def prune_checkpoints(db_path: str, older_than_days: float = CHECKPOINT_RETENTION_DAYS) -> dict:
    """Drop threads whose most recent checkpoint predates the cutoff.

    Keyed on the newest checkpoint per thread, never the oldest: a long-running
    or resumed analysis must not be collected out from under itself while it is
    still being worked on.
    """
    if not os.path.exists(db_path):
        return {"threads": 0, "checkpoints": 0, "writes": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    con = _connect(db_path)
    try:
        newest: dict[str, datetime] = {}
        # Only ids are read here — the blobs are what make this table large.
        for thread_id, checkpoint_id in con.execute(
            "SELECT thread_id, checkpoint_id FROM checkpoints"
        ):
            ts = checkpoint_time(checkpoint_id)
            if ts is None:
                continue
            if thread_id not in newest or ts > newest[thread_id]:
                newest[thread_id] = ts

        stale = [t for t, ts in newest.items() if ts < cutoff]
        if not stale:
            return {"threads": 0, "checkpoints": 0, "writes": 0}

        removed_cp = removed_w = 0
        for batch_start in range(0, len(stale), 500):
            batch = stale[batch_start : batch_start + 500]
            marks = ",".join("?" * len(batch))
            cur = con.execute(f"DELETE FROM writes WHERE thread_id IN ({marks})", batch)
            removed_w += cur.rowcount or 0
            cur = con.execute(f"DELETE FROM checkpoints WHERE thread_id IN ({marks})", batch)
            removed_cp += cur.rowcount or 0
        con.commit()
        return {"threads": len(stale), "checkpoints": removed_cp, "writes": removed_w}
    finally:
        con.close()


def prune_runs(db_path: str, older_than_days: float = RUN_RETENTION_DAYS) -> int:
    """Delete run-history rows past the cutoff. Also drops their cached embeddings."""
    if not os.path.exists(db_path):
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    con = _connect(db_path)
    try:
        cur = con.execute("DELETE FROM runs WHERE timestamp < ?", (cutoff,))
        con.commit()
        return cur.rowcount or 0
    except sqlite3.OperationalError as exc:
        logger.debug("prune_runs skipped: %s", exc)
        return 0
    finally:
        con.close()


def prune_auth_tokens(db_path: str) -> dict:
    """Clear spent and expired tokens.

    revoked_tokens gains a row on every /auth/refresh and is read on every
    refresh, so it grows for the life of the deployment. It stores
    (jti, revoked_at) with no expiry column: a revocation only has to outlive
    the token it revokes, because verify_refresh_token rejects an expired JWT
    on its own. Hence the cutoff is the refresh-token lifetime plus a margin.
    """
    if not os.path.exists(db_path):
        return {}
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    revoked_cutoff = (now - timedelta(days=REVOKED_TOKEN_RETENTION_DAYS)).isoformat()

    removed: dict[str, int] = {}
    con = _connect(db_path)
    try:
        for table, where, params in (
            ("revoked_tokens", "revoked_at < ?", (revoked_cutoff,)),
            ("password_reset_tokens", "used = 1 OR expires_at < ?", (now_iso,)),
            ("email_verification_tokens", "used = 1 OR expires_at < ?", (now_iso,)),
        ):
            try:
                cur = con.execute(f"DELETE FROM {table} WHERE {where}", params)
                removed[table] = cur.rowcount or 0
            except sqlite3.OperationalError as exc:
                # Warning, not debug: a silent skip here is how the original
                # revoked_tokens bug hid — it assumed an expires_at column that
                # this table does not have, so the prune never ran.
                logger.warning("prune_auth_tokens: %s skipped (%s)", table, exc)
        con.commit()
        return removed
    finally:
        con.close()


def vacuum(db_path: str) -> int:
    """Reclaim freed pages. Returns bytes recovered (negative means it grew).

    The wal_checkpoint(TRUNCATE) is load-bearing, not tidiness. Under WAL —
    which is how the app opens graph.db — VACUUM writes the rebuilt database
    into the WAL, and the main file keeps its old size until a checkpoint folds
    it back. Without this the space is not returned to the filesystem and the
    reported figure is always 0.
    """
    if not os.path.exists(db_path):
        return 0
    before = os.path.getsize(db_path)
    con = _connect(db_path)
    try:
        con.execute("VACUUM")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    return before - os.path.getsize(db_path)


# ── Backups ───────────────────────────────────────────────────────────────────

def backup_database(db_path: str, backup_dir: str, keep: int = BACKUP_KEEP) -> str | None:
    """Snapshot one database, then trim to the newest `keep` copies.

    VACUUM INTO rather than copying the file: it takes a consistent snapshot of
    a live WAL database without blocking writers, and the result has no -wal
    sidecar to go missing. Copying a WAL database while it is being written to
    can capture a torn page.
    """
    if not os.path.exists(db_path):
        return None

    name = Path(db_path).stem
    out_dir = Path(backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = out_dir / f"{name}-{stamp}.db"

    con = _connect(db_path)
    try:
        con.execute("VACUUM INTO ?", (str(dest),))
    finally:
        con.close()

    existing = sorted(out_dir.glob(f"{name}-*.db"), reverse=True)
    for stale in existing[keep:]:
        try:
            stale.unlink()
        except OSError as exc:
            logger.debug("could not remove old backup %s: %s", stale, exc)
    return str(dest)


# ── Orchestration ─────────────────────────────────────────────────────────────

def _paths() -> dict[str, str]:
    return {
        "graph": os.getenv("GRAPH_DB_PATH", "memory/graph.db"),
        "memory": os.getenv("MEMORY_DB_PATH", "memory/datapilot_memory.db"),
        "auth": os.getenv("AUTH_DB_PATH", "memory/auth.db"),
    }


def backup_dir() -> str:
    """Defaults next to the databases so it lands on the same mounted volume."""
    explicit = os.getenv("BACKUP_DIR", "").strip()
    if explicit:
        return explicit
    return str(Path(_paths()["auth"]).parent / "backups")


def run_maintenance() -> dict:
    """One prune + backup pass. Blocking; call via asyncio.to_thread."""
    paths = _paths()
    report: dict[str, object] = {}

    try:
        report["checkpoints"] = prune_checkpoints(paths["graph"])
    except Exception:
        logger.warning("checkpoint prune failed", exc_info=True)
    try:
        report["runs_deleted"] = prune_runs(paths["memory"])
    except Exception:
        logger.warning("run prune failed", exc_info=True)
    try:
        report["tokens"] = prune_auth_tokens(paths["auth"])
    except Exception:
        logger.warning("token prune failed", exc_info=True)

    # VACUUM only where deletes actually happened — it rewrites the whole file.
    cp = report.get("checkpoints") or {}
    if isinstance(cp, dict) and cp.get("checkpoints"):
        try:
            report["graph_bytes_reclaimed"] = vacuum(paths["graph"])
        except Exception:
            logger.warning("graph vacuum failed", exc_info=True)

    # Only the small, irreplaceable databases. graph.db is transient run state
    # and is the thing filling the disk; snapshotting it would defeat the point.
    made = []
    for key in ("auth", "memory"):
        try:
            dest = backup_database(paths[key], backup_dir())
            if dest:
                made.append(os.path.basename(dest))
        except Exception:
            logger.warning("backup of %s failed", key, exc_info=True)
    report["backups"] = made

    logger.info("retention.pass %s", report)
    return report
