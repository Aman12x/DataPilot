"""The frontend image's Node must satisfy what the build tools require.

CI and the Dockerfile pin Node separately, so they can disagree — and when they
do, nothing catches it until a deploy. That is exactly how the vite 8 bump
shipped: `ci.yml` builds on Node 22 and went green, while `frontend/Dockerfile`
was still on `node:18-alpine`, where the image build died with

    SyntaxError: The requested module 'node:util'
    does not provide an export named 'styleText'

`styleText` landed in Node 20.12. The error names neither Node nor vite, so the
cause is not obvious from the failure.

These read the requirement from `package-lock.json` rather than hard-coding a
version, so they keep working when vite raises its floor again.
"""
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DOCKERFILE = _ROOT / "frontend" / "Dockerfile"
_LOCKFILE = _ROOT / "frontend" / "package-lock.json"
_CI = _ROOT / ".github" / "workflows" / "ci.yml"

# Packages whose engines gate the production image build.
_BUILD_TOOLS = ("node_modules/vite", "node_modules/@vitejs/plugin-react")


def _dockerfile_node_majors() -> list[int]:
    text = _DOCKERFILE.read_text()
    return [int(m) for m in re.findall(r"^FROM node:(\d+)", text, re.MULTILINE)]


def _required_majors(spec: str) -> set[int]:
    """Majors allowed by an npm engines range like '^20.19.0 || >=22.12.0'."""
    allowed: set[int] = set()
    for clause in spec.split("||"):
        clause = clause.strip()
        m = re.search(r"(\d+)", clause)
        if not m:
            continue
        major = int(m.group(1))
        if clause.startswith(">="):
            allowed.update(range(major, major + 20))
        else:  # ^X.Y.Z pins the major
            allowed.add(major)
    return allowed


def _engine_specs() -> list[tuple[str, str]]:
    lock = json.loads(_LOCKFILE.read_text())
    out = []
    for key in _BUILD_TOOLS:
        engines = (lock["packages"].get(key) or {}).get("engines") or {}
        if engines.get("node"):
            out.append((key, engines["node"]))
    return out


def test_dockerfile_declares_a_node_version():
    majors = _dockerfile_node_majors()
    assert majors, "no `FROM node:<major>` found in frontend/Dockerfile"


@pytest.mark.parametrize("stage_index", [0, 1])
def test_every_dockerfile_stage_satisfies_the_build_tools(stage_index):
    """Both stages matter: the builder compiles, and the serve stage runs
    runtime-config.js at container start."""
    majors = _dockerfile_node_majors()
    if stage_index >= len(majors):
        pytest.skip("Dockerfile has fewer stages than expected")
    major = majors[stage_index]

    for name, spec in _engine_specs():
        allowed = _required_majors(spec)
        assert major in allowed, (
            f"frontend/Dockerfile stage {stage_index} uses node:{major}, but "
            f"{name} requires node {spec}. The image build fails with an error "
            f"naming neither Node nor the package."
        )


def test_ci_node_matches_the_image_node():
    """A version skew here means CI green does not predict a working image.

    That skew is what let the vite 8 bump pass every check and still break the
    deploy.
    """
    ci_majors = {int(m) for m in re.findall(r'node-version:\s*"(\d+)"', _CI.read_text())}
    if not ci_majors:
        pytest.skip("no node-version pin found in ci.yml")
    image_majors = set(_dockerfile_node_majors())
    assert ci_majors == image_majors, (
        f"ci.yml builds the frontend on Node {sorted(ci_majors)} but the image "
        f"uses Node {sorted(image_majors)}; CI cannot catch what the image hits."
    )
