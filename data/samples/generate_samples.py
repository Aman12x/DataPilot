"""
Generate 5 realistic CSV sample datasets for DataPilot demos.
Each file uses non-uniform distributions, realistic correlations, and business logic.
Seed: 42
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

rng = np.random.default_rng(42)
OUT_DIR = Path("/Users/amansingh/Desktop/datapilot/data/samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_id(prefix, n, total):
    """Return zero-padded ID strings, e.g. USR-00001."""
    return [f"{prefix}-{i:05d}" for i in range(1, total + 1)]


def weighted_choice(rng, choices, weights, size):
    """Multinomial draw returning an array of string labels."""
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    idx = rng.choice(len(choices), size=size, p=probs)
    return np.array(choices)[idx]


def date_range_sample(rng, start, days, size):
    """Uniform sample of dates within [start, start+days)."""
    offsets = rng.integers(0, days, size=size)
    base = pd.Timestamp(start)
    return [base + pd.Timedelta(days=int(d)) for d in offsets]


def seasonal_multiplier(dates):
    """Return a per-row seasonal multiplier based on month (sine wave)."""
    months = np.array([d.month for d in dates])
    # Peak in Dec (month 12), trough in Jul/Aug — retail pattern
    return 1.0 + 0.15 * np.sin(2 * np.pi * (months - 3) / 12)


def weekday_multiplier(dates, weekday_boost=1.10, weekend_mult=0.85):
    """Weekday rows get a boost; weekend rows a reduction."""
    dow = np.array([d.weekday() for d in dates])  # 0=Mon … 6=Sun
    m = np.where(dow < 5, weekday_boost, weekend_mult)
    return m


# ===========================================================================
# 1. ecommerce_ab_test.csv  (~15 000 rows)
# ===========================================================================

def generate_ecommerce_ab_test():
    N = 15_000
    print("Generating ecommerce_ab_test.csv …")

    user_ids = fmt_id("USR", N, N)

    # Date range: 90 days starting 2024-10-01
    dates = date_range_sample(rng, "2024-10-01", 90, N)

    # Assignment date is up to 7 days before the observation date (simulate multi-day tracking)
    assignment_offsets = rng.integers(0, 8, size=N)
    assignment_dates = [d - pd.Timedelta(days=int(o)) for d, o in zip(dates, assignment_offsets)]

    # Week within the experiment (1 or 2)
    weeks = rng.choice([1, 2], size=N, p=[0.5, 0.5])

    variant = rng.choice(["control", "treatment"], size=N, p=[0.5, 0.5])
    is_treatment = (variant == "treatment").astype(float)

    device = weighted_choice(rng, ["mobile", "desktop", "tablet"], [0.55, 0.35, 0.10], N)
    user_segment = weighted_choice(rng, ["new", "returning", "power"], [0.30, 0.45, 0.25], N)
    country = weighted_choice(rng, ["US", "UK", "CA", "DE", "AU", "FR", "other"],
                              [0.40, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07], N)
    category = weighted_choice(rng, ["electronics", "clothing", "home", "sports", "beauty"],
                               [0.25, 0.30, 0.20, 0.15, 0.10], N)

    # Session duration (gamma): shape=2, scale=8 → mean ~16 min
    session_duration = rng.gamma(shape=2.0, scale=8.0, size=N)
    # Weekend sessions are shorter
    session_duration *= weekday_multiplier(dates, weekday_boost=1.05, weekend_mult=0.92)

    # Revenue: log-normal base, correlated with session_duration and user_segment
    segment_mult = np.where(user_segment == "power", 1.5,
                   np.where(user_segment == "returning", 1.1, 0.75))
    log_mu = np.log(45) + 0.03 * np.log1p(session_duration) + np.log(segment_mult)
    revenue_base = rng.lognormal(mean=log_mu, sigma=0.6)
    # Treatment effect +12%
    revenue_usd = revenue_base * (1 + 0.12 * is_treatment)
    # Seasonal boost
    revenue_usd *= seasonal_multiplier(dates)
    revenue_usd = np.round(revenue_usd, 2)

    # Orders: Poisson(~2.1) with treatment +8%
    lam_orders = 2.1 * (1 + 0.08 * is_treatment) * segment_mult / 1.1
    orders = rng.poisson(lam=lam_orders.clip(0.5, 10))

    # Avg order value: log-normal ~$48, correlated with revenue
    aov_mu = np.log(48) + 0.2 * (np.log(revenue_usd + 1) - np.log(46))
    avg_order_value = np.round(rng.lognormal(mean=aov_mu, sigma=0.35), 2)

    df = pd.DataFrame({
        "user_id": user_ids,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "variant": variant,
        "week": weeks,
        "assignment_date": [d.strftime("%Y-%m-%d") for d in assignment_dates],
        "revenue_usd": revenue_usd,
        "orders": orders,
        "avg_order_value": avg_order_value,
        "session_duration_min": np.round(session_duration, 2),
        "device": device,
        "user_segment": user_segment,
        "country": country,
        "category": category,
    })

    out = OUT_DIR / "ecommerce_ab_test.csv"
    df.to_csv(out, index=False)
    print(f"  Written {len(df):,} rows → {out}")
    return df


# ===========================================================================
# 2. saas_churn_analysis.csv  (~12 000 rows)
# ===========================================================================

def generate_saas_churn():
    N_CUSTOMERS = 1_000
    MONTHS = 12
    N = N_CUSTOMERS * MONTHS  # 12 000
    print("Generating saas_churn_analysis.csv …")

    customer_ids = fmt_id("CUS", N_CUSTOMERS, N_CUSTOMERS)

    # Per-customer attributes
    plan = weighted_choice(rng, ["starter", "pro", "enterprise"], [0.40, 0.35, 0.25], N_CUSTOMERS)
    industry = weighted_choice(rng, ["SaaS", "fintech", "healthcare", "ecommerce", "other"],
                               [0.20, 0.15, 0.12, 0.18, 0.35], N_CUSTOMERS)
    company_size = weighted_choice(rng, ["1-10", "11-50", "51-200", "200+"],
                                   [0.40, 0.30, 0.20, 0.10], N_CUSTOMERS)

    # Seat count by plan
    seat_count = np.where(
        plan == "starter", rng.integers(1, 4, N_CUSTOMERS),
        np.where(plan == "pro", rng.integers(3, 16, N_CUSTOMERS),
                 rng.integers(10, 201, N_CUSTOMERS))
    )

    # Tenure from exponential (mean 18)
    tenure_months = np.round(rng.exponential(scale=18, size=N_CUSTOMERS)).clip(1, 120).astype(int)

    # MRR: log-normal per plan
    mrr_mu = np.where(plan == "starter", np.log(49),
              np.where(plan == "pro", np.log(199), np.log(999)))
    mrr_sigma = np.where(plan == "starter", 0.25, np.where(plan == "pro", 0.30, 0.35))
    mrr_base = rng.lognormal(mean=mrr_mu, sigma=mrr_sigma)

    # Monthly churn probability per plan
    churn_prob_base = np.where(plan == "starter", 0.08,
                      np.where(plan == "pro", 0.04, 0.015))

    # NPS base by plan
    nps_mu = np.where(plan == "enterprise", 45, np.where(plan == "pro", 35, 25))
    nps_base = rng.normal(loc=nps_mu, scale=10).clip(-100, 100)

    # Sessions per week base (Poisson lambda) by plan
    sessions_lam = np.where(plan == "starter", 3,
                   np.where(plan == "pro", 7, 14)).astype(float)

    # Features used base
    features_lam = np.where(plan == "starter", 4,
                   np.where(plan == "pro", 10, 20)).astype(float)

    # Expand to monthly rows
    cust_idx = np.repeat(np.arange(N_CUSTOMERS), MONTHS)
    month_idx = np.tile(np.arange(MONTHS), N_CUSTOMERS)  # 0..11
    base_month = pd.Timestamp("2024-01-01")
    month_labels = [(base_month + pd.DateOffset(months=int(m))).strftime("%Y-%m")
                    for m in month_idx]

    # Churn flag: customer who churns exits at some month
    # Draw churn month for each customer (geometric-like)
    churned_ever = rng.random(N_CUSTOMERS) < (churn_prob_base * MONTHS * 1.5)  # ~fraction that churn
    churn_month = rng.integers(1, MONTHS, size=N_CUSTOMERS)  # month they churn (1-indexed)

    # Build row-level churned flag
    churned_row = np.zeros(N, dtype=int)
    for i in range(N_CUSTOMERS):
        if churned_ever[i]:
            row_start = i * MONTHS
            m = churn_month[i]
            churned_row[row_start + m] = 1  # churned in that month

    # sessions_per_week: Poisson, lower for churners
    churn_session_penalty = np.where(churned_row == 1, 0.5, 1.0)
    s_lam = (sessions_lam[cust_idx] * churn_session_penalty).clip(0.1)
    sessions_per_week = rng.poisson(lam=s_lam)

    # features_used: Poisson, correlated with sessions
    f_lam = (features_lam[cust_idx] * (sessions_per_week / sessions_lam[cust_idx].clip(1)) * 0.7 + 2).clip(1)
    features_used_count = rng.poisson(lam=f_lam)

    # support_tickets: higher for churners
    ticket_lam = 0.8 + 1.5 * churned_row
    support_tickets = rng.poisson(lam=ticket_lam)

    # NPS per row: add some noise
    nps_score = np.round(rng.normal(loc=nps_base[cust_idx], scale=5)).clip(-100, 100).astype(int)

    # MRR with small monthly drift (growth for non-churners)
    growth = 1.0 + 0.005 * month_idx  # 0.5% MoM growth
    mrr_usd = np.round(mrr_base[cust_idx] * growth, 2)

    df = pd.DataFrame({
        "customer_id": np.array(customer_ids)[cust_idx],
        "month": month_labels,
        "plan": plan[cust_idx],
        "mrr_usd": mrr_usd,
        "churned": churned_row,
        "sessions_per_week": sessions_per_week,
        "features_used_count": features_used_count,
        "support_tickets": support_tickets,
        "nps_score": nps_score,
        "company_size": company_size[cust_idx],
        "industry": industry[cust_idx],
        "tenure_months": tenure_months[cust_idx] + month_idx,
        "seat_count": seat_count[cust_idx],
    })

    out = OUT_DIR / "saas_churn_analysis.csv"
    df.to_csv(out, index=False)
    print(f"  Written {len(df):,} rows → {out}")
    return df


# ===========================================================================
# 3. media_ctr_experiment.csv  (~20 000 rows)
# ===========================================================================

def generate_media_ctr():
    N = 20_000
    print("Generating media_ctr_experiment.csv …")

    user_ids = fmt_id("MED", N, N)
    dates = date_range_sample(rng, "2024-08-01", 60, N)
    assignment_offsets = rng.integers(0, 5, size=N)
    assignment_dates = [d - pd.Timedelta(days=int(o)) for d, o in zip(dates, assignment_offsets)]
    weeks = rng.choice([1, 2], size=N, p=[0.5, 0.5])

    variant = rng.choice(["control", "treatment"], size=N, p=[0.5, 0.5])
    is_treatment = (variant == "treatment").astype(float)

    content_type = weighted_choice(rng, ["video", "article", "podcast", "newsletter"],
                                   [0.45, 0.30, 0.15, 0.10], N)
    platform = weighted_choice(rng, ["mobile_app", "web", "smart_tv"],
                               [0.50, 0.30, 0.20], N)
    age_group = weighted_choice(rng, ["18-24", "25-34", "35-44", "45-54", "55+"],
                                [0.20, 0.35, 0.25, 0.12, 0.08], N)
    region = weighted_choice(rng, ["NA", "EU", "APAC", "LATAM"],
                             [0.45, 0.30, 0.20, 0.05], N)
    premium_user = rng.choice([0, 1], size=N, p=[0.75, 0.25])

    # Impressions: Poisson(~12), slightly higher for premium
    imp_lam = 12 + 3 * premium_user
    impressions = rng.poisson(lam=imp_lam).clip(1, 100)

    # CTR: control 4%, treatment 4.8%, with novelty decay in week 2
    # Week 1 treatment lift: +1.0pp, Week 2: +0.6pp
    week_novelty = np.where(weeks == 1, 1.0, 0.6)
    base_ctr = 0.04
    treatment_lift = 0.008 * week_novelty  # 0.8% max lift
    ctr_true = base_ctr + is_treatment * treatment_lift

    # Platform and content modifiers
    platform_mod = np.where(platform == "mobile_app", 1.1,
                   np.where(platform == "smart_tv", 0.85, 1.0))
    content_mod = np.where(content_type == "video", 1.15,
                  np.where(content_type == "newsletter", 0.75, 1.0))
    ctr_true = (ctr_true * platform_mod * content_mod).clip(0.005, 0.30)

    # Clicks: binomial draw
    clicks = rng.binomial(n=impressions, p=ctr_true)
    ctr = np.round(np.where(impressions > 0, clicks / impressions, 0.0), 4)

    # Watch time: gamma correlated with clicks (more clicks → more watch)
    watch_base = rng.gamma(shape=1.5, scale=4.0, size=N)
    click_boost = 1 + 0.8 * np.log1p(clicks)
    watch_time_min = np.round(watch_base * click_boost, 2)

    df = pd.DataFrame({
        "user_id": user_ids,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "variant": variant,
        "week": weeks,
        "assignment_date": [d.strftime("%Y-%m-%d") for d in assignment_dates],
        "impressions": impressions,
        "clicks": clicks,
        "ctr": ctr,
        "watch_time_min": watch_time_min,
        "content_type": content_type,
        "platform": platform,
        "age_group": age_group,
        "region": region,
        "premium_user": premium_user,
    })

    out = OUT_DIR / "media_ctr_experiment.csv"
    df.to_csv(out, index=False)
    print(f"  Written {len(df):,} rows → {out}")
    return df


# ===========================================================================
# 4. clinical_trial.csv  (~10 000 rows)
# ===========================================================================

def generate_clinical_trial():
    N = 10_000
    print("Generating clinical_trial.csv …")

    patient_ids = fmt_id("PAT", N, N)
    enrollment_dates = date_range_sample(rng, "2023-06-01", 120, N)
    # assignment_date == enrollment_date for RCT
    assignment_dates = enrollment_dates[:]
    weeks = rng.choice([1, 2], size=N, p=[0.5, 0.5])

    treatment_group = rng.choice(["control", "treatment"], size=N, p=[0.5, 0.5])
    is_treatment = (treatment_group == "treatment").astype(float)

    baseline_severity = weighted_choice(rng, ["mild", "moderate", "severe"],
                                        [0.35, 0.40, 0.25], N)
    gender = rng.choice(["M", "F"], size=N, p=[0.48, 0.52])
    bmi_category = weighted_choice(rng, ["underweight", "normal", "overweight", "obese"],
                                   [0.05, 0.45, 0.30, 0.20], N)
    region = weighted_choice(rng, ["North", "South", "East", "West"],
                             [0.28, 0.25, 0.24, 0.23], N)

    # Age: normal ~52 std=14, clipped 18-85
    age = np.round(rng.normal(loc=52, scale=14, size=N)).clip(18, 85).astype(int)

    # Comorbidity count: Poisson ~1.2
    comorbidity_count = rng.poisson(lam=1.2, size=N)

    # Recovery score: control mean=52 std=18, treatment mean=61 std=16
    # HTE: severe baseline → smaller treatment effect
    severity_mod = np.where(baseline_severity == "severe", 0.7,
                   np.where(baseline_severity == "moderate", 1.0, 1.15))
    control_mean = 52.0
    treatment_effect = 9.0 * severity_mod  # HTE

    rec_mean = control_mean + is_treatment * treatment_effect
    rec_std = np.where(is_treatment == 1, 16.0, 18.0)
    recovery_score = np.round(rng.normal(loc=rec_mean, scale=rec_std)).clip(0, 100).astype(int)

    # Side effect count: Poisson (control ~0.8, treatment ~1.4)
    se_lam = np.where(is_treatment == 1, 1.4, 0.8)
    side_effect_count = rng.poisson(lam=se_lam)

    # Adherence: beta distribution (high for both, treatment slightly lower)
    # control: beta(10, 1.5) → mean ~0.87; treatment: beta(8, 1.8) → mean ~0.82
    adh_a = np.where(is_treatment == 1, 8.0, 10.0)
    adh_b = np.where(is_treatment == 1, 1.8, 1.5)
    adherence_pct = np.round(rng.beta(a=adh_a, b=adh_b) * 100, 1)

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "enrollment_date": [d.strftime("%Y-%m-%d") for d in enrollment_dates],
        "treatment_group": treatment_group,
        "week": weeks,
        "assignment_date": [d.strftime("%Y-%m-%d") for d in assignment_dates],
        "recovery_score": recovery_score,
        "side_effect_count": side_effect_count,
        "adherence_pct": adherence_pct,
        "baseline_severity": baseline_severity,
        "age": age,
        "gender": gender,
        "bmi_category": bmi_category,
        "region": region,
        "comorbidity_count": comorbidity_count,
    })

    out = OUT_DIR / "clinical_trial.csv"
    df.to_csv(out, index=False)
    print(f"  Written {len(df):,} rows → {out}")
    return df


# ===========================================================================
# 5. logistics_ops.csv  (~18 000 rows)
# ===========================================================================

def generate_logistics_ops():
    N = 18_000
    print("Generating logistics_ops.csv …")

    shipment_ids = fmt_id("SHP", N, N)
    dates = date_range_sample(rng, "2024-01-01", 180, N)

    hubs = ["Chicago", "New_York", "Los_Angeles", "Houston",
            "Phoenix", "Philadelphia", "San_Antonio", "Dallas"]

    origin_hub = weighted_choice(rng, hubs,
                                 [0.18, 0.20, 0.16, 0.12, 0.08, 0.10, 0.08, 0.08], N)
    # Destination: avoid same as origin
    dest_hub_raw = weighted_choice(rng, hubs,
                                   [0.15, 0.18, 0.17, 0.13, 0.09, 0.11, 0.09, 0.08], N)
    destination_hub = np.where(dest_hub_raw == origin_hub,
                               np.array(hubs)[rng.integers(0, 8, N)],
                               dest_hub_raw)

    carrier = weighted_choice(rng, ["FastShip", "MegaFreight", "QuickPost", "LocalEx"],
                              [0.35, 0.30, 0.20, 0.15], N)
    priority = weighted_choice(rng, ["standard", "express", "overnight"],
                               [0.60, 0.30, 0.10], N)
    product_category = weighted_choice(rng,
                                       ["electronics", "perishables", "clothing",
                                        "furniture", "industrial"],
                                       [0.20, 0.15, 0.25, 0.10, 0.30], N)

    # Season from date
    def get_season(d):
        m = d.month
        if m in [12, 1, 2]:   return "winter"
        elif m in [3, 4, 5]:   return "spring"
        elif m in [6, 7, 8]:   return "summer"
        else:                   return "autumn"
    season = np.array([get_season(d) for d in dates])

    # Distance: log-normal ~800km
    # Cross-country pairs → higher; local → lower
    cross_country = (origin_hub != destination_hub).astype(float)
    dist_mu = np.log(800) + 0.3 * cross_country
    distance_km = np.round(rng.lognormal(mean=dist_mu, sigma=0.5)).clip(50, 5000).astype(int)

    # Weight: log-normal ~12kg
    cat_weight_mu = np.where(product_category == "furniture", np.log(40),
                    np.where(product_category == "industrial", np.log(25),
                    np.where(product_category == "electronics", np.log(8),
                    np.where(product_category == "perishables", np.log(15),
                             np.log(5)))))
    weight_kg = np.round(rng.lognormal(mean=cat_weight_mu, sigma=0.6), 2).clip(0.1, 500)

    # Cost: correlated with distance and weight
    cost_mu = np.log(10) + 0.6 * np.log(distance_km / 800 + 0.1) + 0.4 * np.log(weight_kg + 1)
    # Priority surcharge
    priority_mult = np.where(priority == "overnight", 2.5,
                    np.where(priority == "express", 1.5, 1.0))
    cost_usd = np.round(rng.lognormal(mean=cost_mu, sigma=0.35) * priority_mult, 2)

    # Delivery days promised (based on priority)
    delivery_days_promised = np.where(priority == "overnight", 1,
                             np.where(priority == "express",
                                      rng.integers(2, 4, N),
                                      rng.integers(3, 8, N)))

    # Carrier quality affects on-time rate
    carrier_otd = {"FastShip": 0.88, "MegaFreight": 0.82, "QuickPost": 0.75, "LocalEx": 0.70}
    # Season affects OTD: winter worse
    season_otd = {"spring": 0.03, "summer": 0.02, "autumn": 0.00, "winter": -0.08}

    otd_prob = np.array([carrier_otd[c] for c in carrier])
    otd_prob += np.array([season_otd[s] for s in season])
    otd_prob = otd_prob.clip(0.05, 0.98)
    on_time_delivery = rng.binomial(1, otd_prob)

    # Actual delivery days
    # On-time: actual ≤ promised; delayed: actual = promised + Poisson(1-3)
    delay_days = rng.poisson(lam=1.5, size=N)
    on_time_actual = (delivery_days_promised - rng.integers(0, 2, N)).clip(1)
    delayed_actual = delivery_days_promised + delay_days + 1
    delivery_days_actual = np.where(on_time_delivery == 1, on_time_actual, delayed_actual)

    # Delay reason (only for delayed shipments)
    delay_reason_choices = ["weather", "capacity", "customs", "mechanical", "other"]
    delay_reason_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    all_delay_reasons = weighted_choice(rng, delay_reason_choices, delay_reason_weights, N)
    delay_reason = np.where(on_time_delivery == 0, all_delay_reasons, "")

    df = pd.DataFrame({
        "shipment_id": shipment_ids,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "origin_hub": origin_hub,
        "destination_hub": destination_hub,
        "carrier": carrier,
        "delivery_days_actual": delivery_days_actual,
        "delivery_days_promised": delivery_days_promised,
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "cost_usd": cost_usd,
        "on_time_delivery": on_time_delivery,
        "delay_reason": delay_reason,
        "product_category": product_category,
        "season": season,
        "priority": priority,
    })

    out = OUT_DIR / "logistics_ops.csv"
    df.to_csv(out, index=False)
    print(f"  Written {len(df):,} rows → {out}")
    return df


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DataPilot Sample Dataset Generator")
    print("=" * 60)

    generate_ecommerce_ab_test()
    generate_saas_churn()
    generate_media_ctr()
    generate_clinical_trial()
    generate_logistics_ops()

    print("=" * 60)
    print("All 5 files written successfully.")
    print("=" * 60)

    # Quick sanity-check summary
    for fname in [
        "ecommerce_ab_test.csv",
        "saas_churn_analysis.csv",
        "media_ctr_experiment.csv",
        "clinical_trial.csv",
        "logistics_ops.csv",
    ]:
        p = OUT_DIR / fname
        df = pd.read_csv(p)
        print(f"\n{fname}  ({len(df):,} rows x {df.shape[1]} cols)")
        print(f"  Columns: {list(df.columns)}")
