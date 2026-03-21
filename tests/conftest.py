"""
tests/conftest.py — Shared fixtures for all DataPilot unit tests.

All fixtures are hand-crafted with fixed seeds — fast, self-contained,
and independent of the full DuckDB dataset.

Ground truth baked into fixtures:
  base_experiment_df:
    - 2000 users, 50/50 control/treatment, 20% android/new
    - Android/new treatment: large negative DAU effect (growing across weeks)
    - All other segments: no treatment effect
    - pre_session_count correlates with dau_rate via shared latent baseline → CUPED works
    - notif_optout 3-4x higher in android/new treatment → guardrail breach
    - d7_retained lower in android/new treatment → guardrail breach
    - session_count: no treatment effect → clean guardrail

  base_metrics_daily_df:
    - 44 days: 30 pre-experiment + 14 experiment
    - Android DAU drops ~80/day at experiment start (day 30) → detectable anomaly
    - iOS/web: stable throughout
    - new_users fraction drops for android during experiment → dominant change component

  base_funnel_df:
    - Steps: impression → click → install → d1_retain
    - d1_retain drop-off is 20pp worse for treatment android/new
    - Other segments: no funnel change
"""

import pytest
import numpy as np
import pandas as pd


# ── tmp_duckdb: minimal DB for connectivity tests ─────────────────────────────

@pytest.fixture
def tmp_duckdb(tmp_path):
    """Single-table DuckDB for DB tool tests. Fast, in-memory-equivalent."""
    import duckdb
    path = str(tmp_path / "test.db")
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE events (
            user_id       VARCHAR,
            date          DATE,
            dau_flag      INTEGER,
            session_count INTEGER
        )
    """)
    con.execute("""
        INSERT INTO events VALUES
            ('u1', '2024-01-01', 1, 3),
            ('u2', '2024-01-01', 0, 0),
            ('u3', '2024-01-02', 1, 5)
    """)
    con.close()
    return path


# ── make_experiment_df: plain factory (not a fixture) ─────────────────────────

def make_experiment_df(
    primary_metric: str = "dau_rate",
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Factory for a 2000-row user-level experiment DataFrame.

    The outcome column is named after `primary_metric`, so tests can call
    this with any metric name without pytest parametrize overhead.

    Columns: user_id, variant, platform, user_segment, week,
             <primary_metric>, pre_session_count, notif_optout, d7_retained,
             session_count

    Ground truth:
      - android/new control:   <primary_metric> ~ N(0.55, noise)
      - android/new treatment week 1: ATE ≈ -0.20
      - android/new treatment week 2: ATE ≈ -0.30  (1.5x week 1, growing)
      - all other segments:    no treatment effect
      - notif_optout:  android/new treatment ~ 3x base rate
      - d7_retained:   android/new treatment ~35% lower
      - session_count: no treatment effect (clean guardrail metric)
      - pre_session_count: correlated with <primary_metric> via shared latent baseline
    """
    rng = np.random.default_rng(rng_seed)

    def make_group(platform, segment, variant, week, n,
                   treatment_effect=0.0, notif_p=0.02, d7_p=0.45):
        # Latent user-level baseline drives both pre and post measurements.
        # This creates the pre/post correlation that CUPED exploits.
        baseline = rng.normal(0.55, 0.08, n)

        # Pre-experiment: baseline + noise (treatment not yet applied)
        pre_dau = np.clip(baseline + rng.normal(0, 0.04, n), 0.0, 1.0)
        pre_sessions = np.clip(pre_dau * 4.0 + rng.normal(0, 0.5, n), 0.0, None)

        # Post-experiment: baseline + treatment shift + noise
        post_dau = np.clip(baseline + treatment_effect + rng.normal(0, 0.04, n), 0.0, 1.0)

        return pd.DataFrame({
            "platform":          platform,
            "user_segment":      segment,
            "variant":           variant,
            "week":              week,
            primary_metric:      post_dau,   # dynamic column name
            "pre_session_count": pre_sessions,
            "notif_optout":      rng.binomial(1, notif_p, n).astype(float),
            "d7_retained":       rng.binomial(1, d7_p,    n).astype(float),
            "session_count":     rng.normal(3.0, 0.5, n),
        })

    groups = []

    # ── Affected segment: android/new ──────────────────────────────────────────
    # Control: no shift. Treatment: effect grows from week 1 to week 2.
    groups.append(make_group("android", "new", "control",   1, 100,  0.00))
    groups.append(make_group("android", "new", "control",   2, 100,  0.00))
    groups.append(make_group("android", "new", "treatment", 1, 100, -0.20, notif_p=0.09, d7_p=0.28))
    groups.append(make_group("android", "new", "treatment", 2, 100, -0.30, notif_p=0.09, d7_p=0.28))

    # ── Unaffected segments: no primary metric effect, but mild guardrail signal
    # in treatment (notif_optout slightly elevated, d7_retained slightly lower)
    # so that the blended t-test across all users is detectable for guardrail tests.
    for platform in ["android", "ios", "web"]:
        for segment in ["returning", "power"]:
            for variant in ["control", "treatment"]:
                notif_p = 0.04 if variant == "treatment" else 0.02
                d7_p    = 0.40 if variant == "treatment" else 0.45
                groups.append(make_group(platform, segment, variant, 1, 50, 0.0, notif_p=notif_p, d7_p=d7_p))
                groups.append(make_group(platform, segment, variant, 2, 50, 0.0, notif_p=notif_p, d7_p=d7_p))

    for platform in ["ios", "web"]:
        for variant in ["control", "treatment"]:
            notif_p = 0.04 if variant == "treatment" else 0.02
            d7_p    = 0.40 if variant == "treatment" else 0.45
            groups.append(make_group(platform, "new", variant, 1, 50, 0.0, notif_p=notif_p, d7_p=d7_p))
            groups.append(make_group(platform, "new", variant, 2, 50, 0.0, notif_p=notif_p, d7_p=d7_p))

    df = pd.concat(groups, ignore_index=True)
    df.insert(0, "user_id", [f"u_{i:04d}" for i in range(len(df))])
    return df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)


# ── base_experiment_df ────────────────────────────────────────────────────────

@pytest.fixture
def base_experiment_df():
    """
    2000-row user-level experiment DataFrame with 'dau_rate' as primary metric.
    Thin wrapper around make_experiment_df("dau_rate").
    See make_experiment_df for full ground-truth documentation.
    """
    return make_experiment_df("dau_rate")


# ── base_metrics_daily_df ─────────────────────────────────────────────────────

@pytest.fixture
def base_metrics_daily_df():
    """
    44 days × 3 platforms of pre-aggregated daily metrics.

    Days 0–29:  pre-experiment baseline (stable)
    Days 30–43: experiment period
      - Android: DAU drops ~80/day (16% relative) — clearly anomalous
      - iOS/web: no change
      - Android new_users fraction drops (new_users is the dominant change component)
      - Android d7_retention_rate drops, notif_optout_rate rises (guardrail signals)
    """
    rng = np.random.default_rng(42)
    EXP_START = 30
    TOTAL_DAYS = 44

    rows = []
    for day_idx in range(TOTAL_DAYS):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_idx)
        in_exp = day_idx >= EXP_START

        for platform, base_dau in [("android", 500), ("ios", 400), ("web", 300)]:
            noise = int(rng.integers(-15, 16))
            drop  = 80 if (in_exp and platform == "android") else 0
            dau   = max(base_dau - drop + noise, 50)

            # Component fractions always sum to 1.0 so new+retained+resurrected = dau.
            # During the android experiment, new_users fraction drops sharply
            # while retained_users absorbs the slack — making new the dominant delta.
            if in_exp and platform == "android":
                new_frac, retained_frac, resurrected_frac = 0.10, 0.85, 0.05
            else:
                new_frac, retained_frac, resurrected_frac = 0.25, 0.70, 0.05

            new_users         = int(dau * new_frac)
            retained_users    = int(dau * retained_frac)
            resurrected_users = int(dau * resurrected_frac)
            churned_users     = int(dau * 0.03)

            d7_base   = 0.45 - (0.05 if in_exp and platform == "android" else 0.0)
            opt_base  = 0.02 + (0.05 if in_exp and platform == "android" else 0.0)

            rows.append({
                "date":              date,
                "platform":          platform,
                "user_segment":      "all",
                "dau":               dau,
                "new_users":         new_users,
                "retained_users":    retained_users,
                "resurrected_users": resurrected_users,
                "churned_users":     churned_users,
                "d7_retention_rate": float(np.clip(d7_base  + rng.normal(0, 0.01), 0, 1)),
                "notif_optout_rate": float(np.clip(opt_base + rng.normal(0, 0.002), 0, 1)),
                "avg_session_count": float(3.5 + rng.normal(0, 0.1)),
            })

    return pd.DataFrame(rows)


# ── base_funnel_df ────────────────────────────────────────────────────────────

@pytest.fixture
def base_funnel_df():
    """
    Funnel table: one row per (user, step).
    Steps: impression → click → install → d1_retain

    Ground truth:
      - d1_retain rate drops 20pp for treatment android/new (from 0.45 → 0.25)
      - All other segments: no change across any funnel step
    """
    rng = np.random.default_rng(42)
    STEPS      = ["impression", "click", "install", "d1_retain"]
    BASE_RATES = {"impression": 1.0, "click": 0.30, "install": 0.60, "d1_retain": 0.45}
    BUG_D1_DROP = 0.20

    rows = []
    uid = 0
    for variant in ["control", "treatment"]:
        for platform, segment, n in [
            ("android", "new",       300),
            ("ios",     "returning", 200),
            ("web",     "power",     100),
        ]:
            affected = (platform == "android" and segment == "new"
                        and variant == "treatment")
            for _ in range(n):
                prev = True
                for step in STEPS:
                    rate = BASE_RATES[step]
                    if affected and step == "d1_retain":
                        rate = max(0.0, rate - BUG_D1_DROP)
                    completed = int(prev and rng.random() < rate)
                    rows.append({
                        "user_id":      f"u_{uid:04d}",
                        "variant":      variant,
                        "platform":     platform,
                        "user_segment": segment,
                        "step":         step,
                        "completed":    completed,
                    })
                    prev = bool(completed)
                uid += 1

    return pd.DataFrame(rows)
