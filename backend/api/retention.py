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
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# 100-nanosecond intervals between the Gregorian epoch (1582-10-15) and Unix 0.
_GREGORIAN_OFFSET = 0x01B21DD213814000

CHECKPOINT_RETENTION_DAYS = float(os.getenv("CHECKPOINT_RETENTION_DAYS", "30"))
RUN_RETENTION_DAYS = float(os.getenv("RUN_RETENTION_DAYS", "180"))
# Guest access tokens expire after 60 minutes and cannot be refreshed, so a
# guest upload this old is unreachable by anyone — it only consumes the volume
# the databases live on.
GUEST_UPLOAD_RETENTION_HOURS = float(os.getenv("GUEST_UPLOAD_RETENTION_HOURS", "48"))

try:  # source of truth for how long a refresh token stays valid
    from .deps import REFRESH_TOKEN_EXPIRE_DAYS as _REFRESH_DAYS
except Exception:  # pragma: no cover - deps validates SECRET_KEY at import
    _REFRESH_DAYS = 30
# A revocation is redundant once the JWT it names has expired on its own.
REVOKED_TOKEN_RETENTION_DAYS = float(
    os.getenv("REVOKED_TOKEN_RETENTION_DAYS", str(_REFRESH_DAYS + 7))
)
BACKUP_KEEP = int(os.getenv("BACKUP_KEEP", "7"))
# Reclaim once this much sits in the freelist, even with no deletions this pass.
VACUUM_FREE_BYTES = int(os.getenv("VACUUM_FREE_BYTES", str(32 * 1024 * 1024)))
RETENTION_INTERVAL_SEC = float(os.getenv("RETENTION_INTERVAL_SEC", str(24 * 3600)))

# The Prod Smoke workflow registers three accounts against the deployed app on
# every run and nothing else ever removes them. Deleting rows out of the users
# table is the one prune here that could destroy something irreplaceable, so a
# candidate must satisfy *all three* conditions below, not any of them:
#
#   1. the username starts with a known probe prefix,
#   2. the email is under a reserved domain — example.com is reserved by
#      RFC 2606 and can never receive mail, so no real signup can own one,
#   3. it is older than the window, which keeps an in-flight run's accounts.
#
# Any single condition would be unsafe on its own: a real user may pick a
# colliding username, and a developer may register a throwaway example.com
# address by hand.
# e2eprobe/e2elogin/cspcheck are what prod-auth.spec.ts registers; `probe`
# covers the ad-hoc accounts a hand-run diagnostic leaves behind.
TEST_ACCOUNT_PREFIXES = tuple(
    p.strip()
    for p in os.getenv(
        "TEST_ACCOUNT_PREFIXES", "e2eprobe,e2elogin,cspcheck,probe"
    ).split(",")
    if p.strip()
)
TEST_ACCOUNT_EMAIL_SUFFIX = os.getenv("TEST_ACCOUNT_EMAIL_SUFFIX", "@example.com")
TEST_ACCOUNT_RETENTION_HOURS = float(os.getenv("TEST_ACCOUNT_RETENTION_HOURS", "48"))

# Every table outside `users` that keys rows to a user, so a prune cannot leave
# orphans behind. Split by database: the first group lives in auth.db (which is
# also where workspace_store and org_store put their tables), the second in the
# memory database.
_TEST_ACCOUNT_AUTH_TABLES = (
    "password_reset_tokens",
    "email_verification_tokens",
    "db_connections",
    "metric_packs",
    "schema_annotations",
    "workspace_members",
)
_TEST_ACCOUNT_MEMORY_TABLES = ("runs", "verified_queries")


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


def free_bytes(db_path: str) -> int:
    """Space held in the file's freelist — reclaimable by VACUUM.

    Lets a pass clean up after an earlier one that deleted rows but failed to
    return the pages, without waiting for new deletions to trigger it.
    """
    if not os.path.exists(db_path):
        return 0
    con = _connect(db_path)
    try:
        free = con.execute("PRAGMA freelist_count").fetchone()[0]
        page = con.execute("PRAGMA page_size").fetchone()[0]
        return int(free) * int(page)
    except Exception:
        return 0
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

def prune_guest_uploads(
    upload_dir: str, older_than_hours: float = GUEST_UPLOAD_RETENTION_HOURS
) -> dict:
    """Delete guest upload directories whose newest file is past the cutoff.

    Only directories named guest-* are candidates; registered users' uploads
    are never touched. The age check uses the newest file inside the directory
    so an actively-used guest session is not swept mid-analysis.
    """
    root = Path(upload_dir)
    result = {"dirs": 0, "mb": 0.0}
    if not root.exists():
        return result
    cutoff = time.time() - older_than_hours * 3600
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("guest-"):
            continue
        try:
            newest = max(
                (f.stat().st_mtime for f in d.rglob("*") if f.is_file()),
                default=d.stat().st_mtime,
            )
            if newest >= cutoff:
                continue
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
            result["dirs"] += 1
            result["mb"] = round(result["mb"] + size / 1e6, 1)
        except OSError:
            logger.warning("could not prune guest upload dir %s", d.name, exc_info=True)
    return result


def find_test_accounts(
    db_path: str, older_than_hours: float = TEST_ACCOUNT_RETENTION_HOURS
) -> list[tuple[str, str]]:
    """Smoke-test accounts eligible for deletion, as (user_id, username).

    The prefix and domain are matched in Python rather than with SQL LIKE. The
    prefixes come from an environment variable, and `_` and `%` are LIKE
    wildcards — a prefix containing either would silently widen the match, and
    the thing being widened is a DELETE against the users table.
    """
    if not os.path.exists(db_path) or not TEST_ACCOUNT_PREFIXES:
        return []
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
    ).isoformat()
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT user_id, username, email FROM users WHERE created_at < ?",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning("find_test_accounts: users unavailable (%s)", exc)
        return []
    finally:
        con.close()

    suffix = TEST_ACCOUNT_EMAIL_SUFFIX.lower()
    return [
        (user_id, username)
        for user_id, username, email in rows
        if (email or "").lower().endswith(suffix)
        and (username or "").startswith(TEST_ACCOUNT_PREFIXES)
    ]


def _delete_user_rows(db_path: str, user_ids: list[str], tables) -> dict[str, int]:
    """Delete every row keyed to these users from `tables`, reporting counts."""
    if not os.path.exists(db_path) or not user_ids:
        return {}
    placeholders = ",".join("?" * len(user_ids))
    removed: dict[str, int] = {}
    con = _connect(db_path)
    try:
        for table in tables:
            try:
                cur = con.execute(
                    f"DELETE FROM {table} WHERE user_id IN ({placeholders})",
                    user_ids,
                )
                if cur.rowcount:
                    removed[table] = cur.rowcount
            except sqlite3.OperationalError as exc:
                # Warning, not a silent skip: prune_auth_tokens once shipped a
                # prune that never ran because a missing column was swallowed.
                logger.warning("prune_test_accounts: %s skipped (%s)", table, exc)
        con.commit()
    finally:
        con.close()
    return removed


def prune_test_accounts(
    auth_path: str,
    memory_path: str,
    older_than_hours: float = TEST_ACCOUNT_RETENTION_HOURS,
) -> dict:
    """Delete accounts the Prod Smoke workflow left on the deployed app.

    Without this the workflow adds three permanent users per run. `users` is
    deleted alongside every table that keys rows to a user, so a pruned account
    cannot leave orphaned connections, packs, memberships or run history.
    """
    accounts = find_test_accounts(auth_path, older_than_hours)
    if not accounts:
        return {"accounts": 0}
    user_ids = [uid for uid, _ in accounts]
    rows = {
        **_delete_user_rows(auth_path, user_ids, (*_TEST_ACCOUNT_AUTH_TABLES, "users")),
        **_delete_user_rows(memory_path, user_ids, _TEST_ACCOUNT_MEMORY_TABLES),
    }
    # Count only — the usernames are synthetic, but this log is an INFO record
    # and INFO becomes a Sentry breadcrumb, so it is not the place to start
    # shipping identifiers.
    logger.info("Pruned %d smoke-test account(s)", len(user_ids))
    return {"accounts": len(user_ids), "rows": rows}


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


# ── Off-box backups (future-work item 3) ─────────────────────────────────────
# On-volume snapshots cover corruption and bad deletes; they do not cover
# losing the volume — the scenario people usually mean by "backups". When
# BACKUP_S3_BUCKET is set, each snapshot is also uploaded to S3-compatible
# object storage and remote copies are pruned with the same BACKUP_KEEP
# logic. Encryption is deliberately bucket-side (BACKUP_S3_SSE), never the
# app's Fernet key: a backup you can't decrypt after losing the box is not
# a backup.


def _offbox_client():
    import boto3  # imported lazily — only when off-box backups are configured

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("BACKUP_S3_ENDPOINT") or None,
        aws_access_key_id=os.getenv("BACKUP_S3_ACCESS_KEY") or None,
        aws_secret_access_key=os.getenv("BACKUP_S3_SECRET_KEY") or None,
        region_name=os.getenv("BACKUP_S3_REGION") or None,
    )


def upload_backups_offbox(local_paths: list[str], keep: int = BACKUP_KEEP) -> dict:
    """Upload fresh snapshots and prune stale remote copies.

    Returns a report dict; every failure is recorded there and logged at
    warning — an off-box hiccup must never fail the maintenance pass.
    """
    bucket = os.getenv("BACKUP_S3_BUCKET", "").strip()
    if not bucket:
        return {"enabled": False}
    prefix = os.getenv("BACKUP_S3_PREFIX", "datapilot-backups/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    sse = os.getenv("BACKUP_S3_SSE", "").strip()  # e.g. AES256 or aws:kms
    extra = {"ServerSideEncryption": sse} if sse else None

    report: dict[str, object] = {"enabled": True, "uploaded": [], "pruned": [], "errors": []}
    try:
        client = _offbox_client()
    except Exception as exc:
        logger.warning("offbox backup client unavailable: %s", exc)
        report["errors"] = [f"client: {exc}"]
        return report

    stems: set[str] = set()
    for path in local_paths:
        base = os.path.basename(path)
        stems.add(base.rsplit("-", 1)[0])
        key = prefix + base
        try:
            if extra:
                client.upload_file(path, bucket, key, ExtraArgs=extra)
            else:
                client.upload_file(path, bucket, key)
            report["uploaded"].append(key)
        except Exception as exc:
            logger.warning("offbox upload failed for %s: %s", base, exc)
            report["errors"].append(f"upload {base}: {exc}")

    # Same retention as local snapshots: newest `keep` per database stem.
    for stem in sorted(stems):
        try:
            resp = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}{stem}-")
            keys = sorted((o["Key"] for o in resp.get("Contents", [])), reverse=True)
            for stale in keys[keep:]:
                client.delete_object(Bucket=bucket, Key=stale)
                report["pruned"].append(stale)
        except Exception as exc:
            logger.warning("offbox prune failed for %s: %s", stem, exc)
            report["errors"].append(f"prune {stem}: {exc}")
    return report


def run_maintenance() -> dict:
    """One prune + backup pass. Blocking; call via asyncio.to_thread."""
    paths = _paths()
    report: dict[str, object] = {}

    if os.getenv("DATABASE_URL"):
        # Checkpoints live in Postgres; graph.db (if present) is a stale
        # leftover, not the live store. The UUIDv6-age prune and the
        # VACUUM/wal_checkpoint logic below are SQLite-specific, so skip
        # rather than report reclaimed bytes that mean nothing.
        report["checkpoints"] = "skipped (postgres checkpointer)"
    else:
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
    try:
        report["test_accounts"] = prune_test_accounts(paths["auth"], paths["memory"])
    except Exception:
        logger.warning("test account prune failed", exc_info=True)
    try:
        report["guest_uploads"] = prune_guest_uploads(
            os.getenv("UPLOAD_DIR", "tmp_uploads")
        )
    except Exception:
        logger.warning("guest upload prune failed", exc_info=True)

    # VACUUM when this pass deleted something, or when a previous one left
    # free pages behind. Keying only on "did I delete just now" stranded the
    # space freed by an earlier pass: the first production run deleted 267
    # checkpoints but reclaimed nothing, and the next pass had no deletions of
    # its own, so nothing ever went back to the filesystem.
    cp = report.get("checkpoints") or {}
    deleted_now = isinstance(cp, dict) and bool(cp.get("checkpoints"))
    if not os.getenv("DATABASE_URL") and (
        deleted_now or free_bytes(paths["graph"]) > VACUUM_FREE_BYTES
    ):
        try:
            report["graph_bytes_reclaimed"] = vacuum(paths["graph"])
        except Exception:
            logger.warning("graph vacuum failed", exc_info=True)

    # Only the small, irreplaceable databases. graph.db is transient run state
    # and is the thing filling the disk; snapshotting it would defeat the point.
    made = []
    made_paths = []
    for key in ("auth", "memory"):
        try:
            dest = backup_database(paths[key], backup_dir())
            if dest:
                made.append(os.path.basename(dest))
                made_paths.append(dest)
        except Exception:
            logger.warning("backup of %s failed", key, exc_info=True)
    report["backups"] = made

    if made_paths:
        try:
            offbox = upload_backups_offbox(made_paths)
            if offbox.get("enabled"):
                report["offbox"] = offbox
        except Exception:
            logger.warning("offbox backup failed", exc_info=True)

    # Sizes on every pass: without them there is no way to tell which file is
    # consuming a fixed-size volume, or to see growth before the disk is full.
    try:
        report["sizes_mb"] = disk_report()
    except Exception:
        logger.debug("size report failed", exc_info=True)

    logger.info("retention.pass %s", report)
    return report


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def disk_report() -> dict:
    """Per-file and per-directory usage on the volume, in MB."""
    paths = _paths()
    out: dict[str, float] = {}
    for key, p in paths.items():
        size = 0
        for suffix in ("", "-wal", "-shm"):
            try:
                size += os.path.getsize(p + suffix)
            except OSError:
                pass
        out[key] = round(size / 1e6, 1)
    out["uploads"] = round(_dir_bytes(Path(os.getenv("UPLOAD_DIR", "tmp_uploads"))) / 1e6, 1)
    out["backups"] = round(_dir_bytes(Path(backup_dir())) / 1e6, 1)
    out["graph_free"] = round(free_bytes(paths["graph"]) / 1e6, 1)
    return out
