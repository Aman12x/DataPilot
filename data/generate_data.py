"""
generate_data.py — Synthetic DAU drop dataset with known ground truth.

Ground truth:
  - Push notification bug reduces notif_opened for Android new_users in treatment
  - Effect: ~8% DAU reduction in that segment only
  - All other segments: no effect
  - Guardrails: notif_optout_rate increases in treatment, d7_retention_rate slightly decreases
  - Novelty: week 2 effect is 1.5x week 1 (bug compounds, not novelty)
  - Funnel: d1_retain drop-off worsens for treatment android new_users

Run:  python data/generate_data.py
Output: data/dau_experiment.db (DuckDB)
"""

import os
import numpy as np
import pandas as pd
import duckdb
from datetime import date, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

# ── Timeline ──────────────────────────────────────────────────────────────────
PRE_EXP_DAYS   = 28          # baseline period used for forecast / decomposition
EXP_DAYS       = 14          # 2 experiment weeks
TOTAL_DAYS     = PRE_EXP_DAYS + EXP_DAYS

START_DATE         = date(2024, 1, 1)
EXP_START_DATE     = START_DATE + timedelta(days=PRE_EXP_DAYS)
EXP_END_DATE       = EXP_START_DATE + timedelta(days=EXP_DAYS - 1)

# ── User pool ─────────────────────────────────────────────────────────────────
N_USERS = 10_000

PLATFORMS       = ["android", "ios", "web"]
PLATFORM_SHARES = [0.40, 0.35, 0.25]       # android is biggest

USER_SEGMENTS       = ["new", "returning", "power"]
SEGMENT_SHARES      = [0.20, 0.60, 0.20]   # new = 20% of users

# Affected segment: android + new  →  ~8% of total users (0.40 * 0.20 = 8%)
TREATMENT_SHARE = 0.50                      # 50/50 split

# ── DAU activity probabilities by segment ─────────────────────────────────────
BASE_DAU_PROB = {
    ("android", "new"):       0.55,
    ("android", "returning"): 0.65,
    ("android", "power"):     0.85,
    ("ios",     "new"):       0.55,
    ("ios",     "returning"): 0.65,
    ("ios",     "power"):     0.85,
    ("web",     "new"):       0.45,
    ("web",     "returning"): 0.55,
    ("web",     "power"):     0.75,
}

# Bug effect: 8% absolute DAU reduction for android new_users in treatment
BUG_DAU_REDUCTION   = 0.08
# Compounding: week 2 is 1.5x week 1
BUG_WEEK1_SCALE     = 1.0
BUG_WEEK2_SCALE     = 1.5

# ── Notification parameters ───────────────────────────────────────────────────
BASE_NOTIF_RECEIVED   = 3     # avg per day
BASE_NOTIF_OPEN_RATE  = 0.35
NOTIF_OPTOUT_RATE     = 0.02  # base daily opt-out probability
# Bug: broken notifs → lower open rate → higher opt-out for affected group
BUG_NOTIF_OPEN_DROP   = 0.60  # open rate falls to 60% of normal
BUG_OPTOUT_MULTIPLIER = 6.0   # 6x opt-out rate — high enough to be detectable across blended treatment population

# ── Session counts ─────────────────────────────────────────────────────────────
BASE_SESSIONS = {"new": 2.0, "returning": 3.5, "power": 6.0}

# ── D7 retention ─────────────────────────────────────────────────────────────
BASE_D7_RETENTION = {
    ("android", "new"):       0.35,
    ("android", "returning"): 0.55,
    ("android", "power"):     0.75,
    ("ios",     "new"):       0.38,
    ("ios",     "returning"): 0.57,
    ("ios",     "power"):     0.77,
    ("web",     "new"):       0.28,
    ("web",     "returning"): 0.48,
    ("web",     "power"):     0.68,
}
# Bug slightly reduces d7_retention for affected segment
BUG_D7_RETENTION_DROP = 0.04


def build_users() -> pd.DataFrame:
    """Create the static user pool."""
    platforms = rng.choice(PLATFORMS, size=N_USERS, p=PLATFORM_SHARES)
    segments  = rng.choice(USER_SEGMENTS, size=N_USERS, p=SEGMENT_SHARES)

    install_offsets = rng.integers(0, PRE_EXP_DAYS, size=N_USERS)
    install_dates   = [START_DATE + timedelta(days=int(d)) for d in install_offsets]

    is_new = (install_offsets >= (PRE_EXP_DAYS - 7)).astype(int)

    user_ids = [f"u_{i:06d}" for i in range(N_USERS)]
    return pd.DataFrame({
        "user_id":      user_ids,
        "platform":     platforms,
        "user_segment": segments,
        "install_date": install_dates,
        "is_new_user":  is_new,
    })


def assign_experiment(users: pd.DataFrame) -> pd.DataFrame:
    """50/50 random assignment. Returns experiment table with one row per (user, week).

    Two rows per user (week=1 and week=2) so that novelty detection can compute
    the ATE separately for each experiment week by joining with the events table.
    """
    variant = rng.choice(["control", "treatment"], size=N_USERS, p=[0.5, 0.5])
    base = pd.DataFrame({
        "user_id":         users["user_id"],
        "variant":         variant,
        "assignment_date": EXP_START_DATE,
    })
    # Duplicate rows for week 1 and week 2
    wk1 = base.copy(); wk1["week"] = 1
    wk2 = base.copy(); wk2["week"] = 2
    return pd.concat([wk1, wk2], ignore_index=True)


def is_affected(platform: str, segment: str, variant: str) -> bool:
    return platform == "android" and segment == "new" and variant == "treatment"


def build_events(users: pd.DataFrame, experiment: pd.DataFrame) -> pd.DataFrame:
    """Build the events table: one row per (user, date) for active days."""
    exp_map  = experiment.set_index("user_id")["variant"].to_dict()
    rows = []

    all_dates = [START_DATE + timedelta(days=d) for d in range(TOTAL_DAYS)]

    for _, u in users.iterrows():
        uid      = u["user_id"]
        platform = u["platform"]
        segment  = u["user_segment"]
        variant  = exp_map[uid]
        install  = u["install_date"]

        base_p   = BASE_DAU_PROB[(platform, segment)]
        base_d7  = BASE_D7_RETENTION[(platform, segment)]
        base_ses = BASE_SESSIONS[segment]
        affected = is_affected(platform, segment, variant)

        for d in all_dates:
            if d < install:
                continue

            in_exp  = d >= EXP_START_DATE
            exp_week = None
            if in_exp:
                days_in = (d - EXP_START_DATE).days
                exp_week = 1 if days_in < 7 else 2

            # DAU probability
            dau_p = base_p
            if affected and in_exp:
                scale = BUG_WEEK1_SCALE if exp_week == 1 else BUG_WEEK2_SCALE
                dau_p = max(0.0, base_p - BUG_DAU_REDUCTION * scale)

            dau_flag = int(rng.random() < dau_p)

            # Notifications (only meaningful on active days)
            notif_received = 0
            notif_opened   = 0
            notif_optout   = 0
            sessions       = 0
            d7_retained    = 0

            if dau_flag == 1:
                open_rate = BASE_NOTIF_OPEN_RATE
                optout_p  = NOTIF_OPTOUT_RATE
                if affected and in_exp:
                    open_rate = BASE_NOTIF_OPEN_RATE * BUG_NOTIF_OPEN_DROP
                    optout_p  = NOTIF_OPTOUT_RATE * BUG_OPTOUT_MULTIPLIER
                notif_received = int(rng.poisson(BASE_NOTIF_RECEIVED))
                notif_opened   = int(rng.binomial(notif_received, open_rate))
                notif_optout   = int(rng.random() < optout_p)

                sessions = max(1, int(rng.poisson(base_ses)))

                d7 = base_d7
                if affected and in_exp:
                    d7 = max(0.0, base_d7 - BUG_D7_RETENTION_DROP)
                d7_retained = int(rng.random() < d7)

            is_new_flag = 1 if (d - install).days < 7 else 0

            rows.append({
                "user_id":        uid,
                "date":           d,
                "platform":       platform,
                "user_segment":   segment,
                "is_new_user":    is_new_flag,
                "dau_flag":       dau_flag,
                "session_count":  sessions,
                "notif_received": notif_received,
                "notif_opened":   notif_opened,
                "notif_optout":   notif_optout,
                "d7_retained":    d7_retained,
                "install_date":   install,
            })

    return pd.DataFrame(rows)


def build_funnel(users: pd.DataFrame, experiment: pd.DataFrame) -> pd.DataFrame:
    """
    Funnel: impression → click → install → d1_retain
    Drop-off worsens at d1_retain for treatment android new_users.
    """
    exp_map  = experiment.set_index("user_id")["variant"].to_dict()
    STEPS = ["impression", "click", "install", "d1_retain"]

    # Baseline completion rates per step
    BASE_RATES = {
        "impression": 1.00,
        "click":      0.30,
        "install":    0.60,   # of clickers
        "d1_retain":  0.45,   # of installers
    }
    BUG_D1_RETAIN_DROP = 0.10  # absolute drop for affected segment

    rows = []
    for _, u in users.iterrows():
        uid      = u["user_id"]
        platform = u["platform"]
        segment  = u["user_segment"]
        variant  = exp_map[uid]
        affected = is_affected(platform, segment, variant)
        funnel_date = EXP_START_DATE

        completed_prev = True
        for step in STEPS:
            rate = BASE_RATES[step]
            if affected and step == "d1_retain":
                rate = max(0.0, rate - BUG_D1_RETAIN_DROP)
            completed = int(completed_prev and rng.random() < rate)
            rows.append({
                "user_id":   uid,
                "date":      funnel_date,
                "step":      step,
                "completed": completed,
            })
            completed_prev = bool(completed)

    return pd.DataFrame(rows)


def build_metrics_daily(events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-aggregate metrics_daily from the events table.
    Includes retained / resurrected / churned using a 28-day activity window.
    """
    # Active user sets per date
    active_by_date: dict[date, set] = {}
    for d, grp in events.groupby("date"):
        active_by_date[d] = set(grp["user_id"])

    all_dates = sorted(active_by_date.keys())

    # platform × segment combinations
    combos = [
        (p, s)
        for p in PLATFORMS
        for s in USER_SEGMENTS
    ]

    rows = []
    user_meta = users.set_index("user_id")

    for d in all_dates:
        active_today = active_by_date.get(d, set())

        # 28-day prior window
        prior_window = set()
        for pd_ in all_dates:
            if timedelta(0) < (d - pd_) <= timedelta(days=28):
                prior_window |= active_by_date.get(pd_, set())

        # Users active >28 days ago (for resurrected)
        ancient = set()
        for pd_ in all_dates:
            if (d - pd_) > timedelta(days=28):
                ancient |= active_by_date.get(pd_, set())

        # Users active 28 days ago (for churn)
        day_28_ago = d - timedelta(days=28)
        active_28d_ago = active_by_date.get(day_28_ago, set())

        install_dates = user_meta["install_date"]

        for platform, segment in combos:
            seg_users = set(
                user_meta[
                    (user_meta["platform"] == platform) &
                    (user_meta["user_segment"] == segment)
                ].index
            )

            today_seg       = active_today & seg_users
            prior_seg       = prior_window & seg_users
            ancient_seg     = ancient & seg_users
            active_28d_seg  = active_28d_ago & seg_users

            dau              = len(today_seg)
            new_users        = len([u for u in today_seg
                                    if (d - install_dates.get(u, d)).days < 7])
            retained_users   = len(today_seg & prior_seg)
            resurrected_users = len(today_seg - prior_seg & ancient_seg)
            churned_users    = len(active_28d_seg - active_today)

            # Rates from events for this platform/segment/date
            day_events = events[
                (events["date"] == d) &
                (events["platform"] == platform) &
                (events["user_segment"] == segment)
            ]

            if len(day_events) > 0:
                d7_ret_rate   = day_events["d7_retained"].mean()
                optout_rate   = day_events["notif_optout"].mean()
                avg_sessions  = day_events["session_count"].mean()
            else:
                d7_ret_rate   = 0.0
                optout_rate   = 0.0
                avg_sessions  = 0.0

            rows.append({
                "date":              d,
                "platform":          platform,
                "user_segment":      segment,
                "dau":               dau,
                "new_users":         new_users,
                "retained_users":    retained_users,
                "resurrected_users": resurrected_users,
                "churned_users":     churned_users,
                "d7_retention_rate": round(d7_ret_rate, 4),
                "notif_optout_rate": round(optout_rate, 4),
                "avg_session_count": round(avg_sessions, 4),
            })

    return pd.DataFrame(rows)


def write_to_duckdb(
    events: pd.DataFrame,
    funnel: pd.DataFrame,
    experiment: pd.DataFrame,
    metrics_daily: pd.DataFrame,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)

    con = duckdb.connect(path)

    con.execute("""
        CREATE TABLE events AS SELECT * FROM events
    """)
    con.execute("""
        CREATE TABLE funnel AS SELECT * FROM funnel
    """)
    con.execute("""
        CREATE TABLE experiment AS SELECT * FROM experiment
    """)
    con.execute("""
        CREATE TABLE metrics_daily AS SELECT * FROM metrics_daily
    """)

    con.close()


def print_ground_truth_check(events: pd.DataFrame, experiment: pd.DataFrame) -> None:
    """Print summary stats proving the android/new_user DAU gap is detectable."""
    exp_map = experiment.set_index("user_id")[["variant"]].copy()
    exp_events = events[events["date"] >= EXP_START_DATE].copy()
    exp_events = exp_events.merge(exp_map, on="user_id")

    print("\n" + "="*60)
    print("GROUND TRUTH VERIFICATION")
    print("="*60)

    # ── 1. Overall DAU by variant (all users) ─────────────────────────────────
    overall = (
        exp_events.groupby(["date", "variant"])["dau_flag"]
        .sum()
        .reset_index()
        .groupby("variant")["dau_flag"]
        .mean()
    )
    print("\n[1] Mean daily active users by variant (all segments):")
    for v, val in overall.items():
        print(f"    {v:12s}: {val:,.1f}")

    # ── 2. Android new_users — the affected segment ───────────────────────────
    affected = exp_events[
        (exp_events["platform"] == "android") &
        (exp_events["user_segment"] == "new")
    ]
    aff_summary = (
        affected.groupby(["date", "variant"])["dau_flag"]
        .sum()
        .reset_index()
        .groupby("variant")["dau_flag"]
        .mean()
    )
    print("\n[2] Mean daily active users — android / new_users (AFFECTED):")
    ctrl = aff_summary.get("control", 0)
    trt  = aff_summary.get("treatment", 0)
    gap  = (ctrl - trt) / ctrl * 100 if ctrl > 0 else 0
    for v, val in aff_summary.items():
        print(f"    {v:12s}: {val:,.1f}")
    print(f"    DAU gap      : {gap:.1f}%  (expected ~8%)")

    # ── 3. Other segments — should show no effect ─────────────────────────────
    unaffected = exp_events[
        ~((exp_events["platform"] == "android") &
          (exp_events["user_segment"] == "new"))
    ]
    unaff_summary = (
        unaffected.groupby(["date", "variant"])["dau_flag"]
        .sum()
        .reset_index()
        .groupby("variant")["dau_flag"]
        .mean()
    )
    print("\n[3] Mean daily active users — all OTHER segments (should be ~equal):")
    for v, val in unaff_summary.items():
        print(f"    {v:12s}: {val:,.1f}")

    # ── 4. Guardrails ─────────────────────────────────────────────────────────
    print("\n[4] Guardrail metrics — android / new_users:")
    for metric in ["notif_optout", "d7_retained"]:
        g = (
            affected.groupby("variant")[metric]
            .mean()
        )
        ctrl_v = g.get("control", 0)
        trt_v  = g.get("treatment", 0)
        delta  = (trt_v - ctrl_v) / ctrl_v * 100 if ctrl_v > 0 else 0
        direction = "UP" if delta > 0 else "DOWN"
        print(f"    {metric:18s}: control={ctrl_v:.4f}, treatment={trt_v:.4f}, "
              f"delta={delta:+.1f}% [{direction}]")

    # ── 5. Novelty effect check (week1 vs week2) ──────────────────────────────
    print("\n[5] Week-over-week treatment effect — android / new_users:")
    for wk, label in [(1, "Week 1 (days 1-7)"), (2, "Week 2 (days 8-14)")]:
        wk_start = EXP_START_DATE + timedelta(days=(wk - 1) * 7)
        wk_end   = wk_start + timedelta(days=6)
        wk_data  = affected[
            (affected["date"] >= wk_start) & (affected["date"] <= wk_end)
        ]
        wk_sum = (
            wk_data.groupby(["date", "variant"])["dau_flag"]
            .sum()
            .reset_index()
            .groupby("variant")["dau_flag"]
            .mean()
        )
        ctrl_w = wk_sum.get("control", 0)
        trt_w  = wk_sum.get("treatment", 0)
        ate    = trt_w - ctrl_w
        print(f"    {label}: ATE = {ate:+.2f}  (control={ctrl_w:.1f}, treatment={trt_w:.1f})")

    print("\n" + "="*60)
    print("All tables written to data/dau_experiment.db")
    print("="*60 + "\n")


def main():
    print("Generating synthetic dataset (seed=42)...")

    print("  Building user pool...")
    users = build_users()

    print("  Assigning experiment variants...")
    experiment = assign_experiment(users)

    print("  Building events table (this may take ~30s)...")
    events = build_events(users, experiment)

    print("  Building funnel table...")
    funnel = build_funnel(users, experiment)

    print("  Aggregating metrics_daily...")
    metrics_daily = build_metrics_daily(events, users)

    db_path = os.path.join(os.path.dirname(__file__), "dau_experiment.db")
    print(f"  Writing to {db_path}...")
    write_to_duckdb(events, funnel, experiment, metrics_daily, db_path)

    print("\nTable row counts:")
    con = duckdb.connect(db_path, read_only=True)
    for tbl in ["events", "funnel", "experiment", "metrics_daily"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:20s}: {n:,}")
    con.close()

    print_ground_truth_check(events, experiment)


if __name__ == "__main__":
    main()
