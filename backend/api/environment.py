"""
backend/api/environment.py — one answer to "are we deployed?".

Five modules used to compute this independently, and four of them asked the
wrong question: `ENV.lower() in ("production", "prod")`. That is an allowlist of
*strict* environments, so it fails **open** — an environment named `staging`,
`preview`, or `prod-eu` silently got insecure cookies, unenforced CORS, no HSTS,
and tokens echoed in response bodies. Nothing logs when that happens; the app
just quietly runs a development security posture on a public hostname.

So the allowlist is inverted here: name the environments that are genuinely
local, and treat everything else as deployed. An unrecognised name now gets the
*strict* posture. The failure mode of guessing wrong flips from "staging is
wide open" to "someone's oddly-named local box needs ENV=development" — which
announces itself immediately instead of never.

There is one deliberate asymmetry: being on Railway at all means deployed, even
if the environment is named `development`. A Railway "development" environment
is still a real host on a real HTTPS domain serving real sessions.

Read at import time, matching how every caller used it. Deployment identity does
not change under a running process.
"""
from __future__ import annotations

import os

# Names that can only mean "someone's laptop or a CI runner". Anything else —
# staging, preview, prod-eu, qa, a typo — is treated as a deployment.
_LOCAL_ENVS = frozenset({
    "", "development", "dev", "local", "localhost", "test", "testing", "ci",
})

ENV: str = os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("ENV", "development")

# A Railway environment named "development" is still a deployment.
IS_DEPLOYED: bool = (
    ENV.strip().lower() not in _LOCAL_ENVS
    or bool(os.getenv("RAILWAY_ENVIRONMENT"))
)


def is_deployed() -> bool:
    """True when this process is serving from a real host.

    Gate every security posture on this, not on the environment's *name*:
    Secure/SameSite=None cookies, HSTS, CORS enforcement, and whether tokens may
    be echoed in a response body.
    """
    return IS_DEPLOYED
