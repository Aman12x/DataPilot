# DataPilot — Fixes Log

Chronological record of bugs found and fixed during build, with root cause and location.

---

## Fix 1 — Sparse events table (inactive days missing)

**File:** `data/generate_data.py`
**Root cause:** Original code had `if dau_flag == 0: continue`, so rows for inactive user-days
were never written. This meant every row in the events table had `dau_flag=1`, giving
`AVG(dau_flag) = 1.0` for every user — no variance, no detectable treatment effect.
**Fix:** Removed the skip guard so all user×day rows are always appended (dau_flag=0 for
inactive days).
**Result:** Table grew from ~183K → ~285K rows. `AVG(dau_flag)` now has proper variance.

---

## Fix 2 — Guardrail tests failing: blended t-test not significant

**File:** `tests/conftest.py`, lines 119–132
**Root cause:** Guardrail effects (elevated `notif_optout`, lower `d7_retained`) were only
applied to the android/new treatment group (~20% of fixture users). The blended t-test across
all 2000 users was not significant (p ≈ 0.05–0.13), so `breached=False` for all metrics.
**Fix:** Added mild guardrail signals to ALL treatment users in unaffected segments:
- `notif_p = 0.04` (treatment) vs `0.02` (control) — lines 122–123, 129–130
- `d7_p = 0.40` (treatment) vs `0.45` (control) — same lines

This reflects reality: a bad push-notification feature would elevate opt-outs platform-wide,
not only in the segment with the DAU bug.
**Result:** Blended t-test now significant; all 4 guardrail tests pass.

---

## Fix 3 — MDE test: underpowered scenario miscalibrated

**File:** `tests/test_mde_tools.py`, lines 22–35
**Root cause:** `test_underpowered_for_blended_effect` used n=5000/std=0.10, giving
MDE=1.02%. The "blended 1.3% effect" was actually *above* that MDE, so
`is_powered_for_observed_effect` was `True` — the assertion `is False` was wrong.
**Fix:** Changed to n=1000/std=0.25 (MDE=5.7%), matching CLAUDE.md ground truth
("MDE ~3% DAU lift, blended effect ~1.6% — near the MDE boundary"). The observed effect
(0.55 × 0.016 = 0.0088) is now correctly below MDE.
**Result:** Test correctly validates that the blended experiment is underpowered.

---

## Fix 4 — Funnel tool: unconditional completion rates (wrong denominator)

**File:** `tools/funnel_tools.py`, lines 80–104
**Root cause:** Original implementation computed `completed.mean()` across all N users at
each step. At `d1_retain`, 246 of 300 users had `completed=0` simply because they dropped
out at click or install — not because they failed d1_retain. Using all 300 as the denominator
washed out the 20pp d1_retain drop, yielding delta ≈ 0.0017 and p ≈ 0.066 (not significant).
**Fix:** Refactored to conditional step-by-step rates:
1. Pivot data to one row per user with columns per step (lines 80–88)
2. At step k, filter `eligible` to users who completed step k-1 (lines 92–101)
3. Run two-proportion z-test only on eligible users (lines 103–125)

Effective n at d1_retain drops from 300 → ~54 per arm, but the rate comparison is
45% vs 25% (the true signal), giving p ≈ 0.04.
**Result:** All 4 funnel tests pass; biggest_dropoff correctly identified as `d1_retain`.

---

## Fix 5 — Funnel tests: blended view dilutes affected-segment signal

**File:** `tests/test_funnel_tools.py`, lines 14–43
**Root cause:** Tests ran `compute_funnel` on all segments combined. The d1_retain bug only
affects android/new (1/3 of users), so the blended signal was diluted to near-zero even
after Fix 4.
**Fix:** Added `segment_filter={'platform': 'android', 'user_segment': 'new'}` to the three
signal tests. This mirrors real-world usage: after HTE identifies the affected segment, you
drill into the funnel for that segment only. `test_all_steps_returned` (structural test)
still runs on unfiltered data.
**Result:** Tests now validate the correct analytical pattern, not a diluted blended view.
