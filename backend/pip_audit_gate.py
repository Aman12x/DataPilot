"""
CI gate: pip-audit over backend/requirements.txt, minus a tracked ignore list.

    python backend/pip_audit_gate.py

Exits non-zero when any vulnerability outside backend/pip-audit-ignore.txt is
reported. Runs pip-audit with --disable-pip --no-deps so it audits the pins
exactly as written (no resolution, no install; works on any OS regardless of
platform-specific wheels such as cuda-bindings), which is what makes it cheap
enough to run on every push next to `npm audit --audit-level=high`.

Kept as a script rather than a long `--ignore-vuln` chain in ci.yml so the
ignore list can carry the triage note next to each ID, and so `make audit`
runs the identical thing locally.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQS = os.path.join(ROOT, "backend", "requirements.txt")
IGNORE = os.path.join(ROOT, "backend", "pip-audit-ignore.txt")


def _ignored() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(IGNORE, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            vid, _, note = line.partition("#")
            out[vid.strip()] = note.strip()
    return out


def main() -> int:
    ignored = _ignored()
    proc = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--disable-pip", "--no-deps",
         "--progress-spinner", "off", "--format", "json", "-r", REQS],
        capture_output=True, text=True,
    )
    # pip-audit exits 1 when it finds anything, so the exit code alone means
    # nothing here; JSON on stdout is the signal. Anything else is a real
    # failure (network, bad requirements) and must fail the gate loudly.
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.stderr.write("\npip-audit did not produce a JSON report\n")
        return 2

    deps = report.get("dependencies", [])
    if not deps:
        sys.stderr.write("pip-audit reported zero dependencies — the requirements "
                         "file was not read; refusing to pass an empty audit\n")
        return 2

    # A pin pip-audit could not look up (not on PyPI under that exact
    # version: a local-version wheel, an extra index, a renamed package)
    # comes back with skip_reason and no vulns key. Counting that as clean
    # would silently stop auditing the pin; fail instead, the same as a
    # network failure would.
    skipped = [(d.get("name", "?"), d.get("skip_reason", "")) for d in deps if "skip_reason" in d]
    if skipped:
        sys.stderr.write(f"pip-audit skipped {len(skipped)} pin(s) it could not audit:\n")
        for name, why in skipped:
            sys.stderr.write(f"  {name}: {why}\n")
        sys.stderr.write("Every pin must be auditable; pin a PyPI release or move "
                         "the package out of backend/requirements.txt.\n")
        return 2

    seen: set[str] = set()
    new: list[tuple[str, str, str, str]] = []
    for dep in deps:
        for v in dep.get("vulns", []):
            vid = v["id"]
            seen.add(vid)
            aliases = set(v.get("aliases") or [])
            if vid in ignored or aliases & ignored.keys():
                continue
            fix = ", ".join(v.get("fix_versions") or []) or "none"
            new.append((dep["name"], dep["version"], vid, fix))

    stale = [vid for vid in ignored if vid not in seen]
    for vid in stale:
        print(f"note: ignore entry {vid} no longer matches anything — remove it "
              f"from backend/pip-audit-ignore.txt")

    known = len(seen) - len(new)
    if new:
        print(f"\n{len(new)} NEW vulnerabilit{'y' if len(new) == 1 else 'ies'} "
              f"in backend/requirements.txt ({known} known/ignored):\n")
        for name, ver, vid, fix in new:
            print(f"  {name}=={ver}  {vid}  fix: {fix}")
        print("\nBump the pin (backend/requirements.in, then recompile) or, if it "
              "genuinely cannot move yet, add the ID to backend/pip-audit-ignore.txt "
              "with a note.")
        return 1

    print(f"pip-audit: no new vulnerabilities ({known} known/ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
