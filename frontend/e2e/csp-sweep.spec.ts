/**
 * csp-sweep.spec.ts — every screen the app can put on the page.
 *
 * `csp-render.spec.ts` covers the three surfaces that were called out as
 * unverified. This file exists for the opposite reason: to justify *tightening*
 * the policy. A tightening is only as good as its coverage, and three screens
 * are not enough to conclude that `style-src 'unsafe-inline'` is dead weight.
 *
 * So this walks the unauthenticated pages, the mode picker, both analysis modes,
 * all seven HITL gates, the chain-of-thought stream, the finished view with its
 * disclosure expanded, history with a run expanded, and each modal — asserting
 * zero violations on every one.
 *
 * The gates matter most. They are the largest body of UI in the app, they only
 * appear mid-run behind an SSE event, and nothing had ever rendered one under a
 * real policy.
 */
import { test, expect, type Page } from "@playwright/test";

import {
  DONE_STATE,
  expectNoViolations,
  startAnalysis,
  stubApi,
  watchCsp,
} from "./csp-helpers";

// ── Gate payloads ─────────────────────────────────────────────────────────────
// Populated rather than minimal: an empty payload renders an empty gate, which
// would pass without exercising the tables, badges, and diff views that carry
// most of the styling.

const METRIC_CONFIG = {
  primary_metric: "revenue",
  metric_source_col: "revenue_usd",
  metric_agg: "sum",
  covariate: "prior_week_revenue",
  metric_direction: "higher_is_better",
  events_table: "transactions",
  experiment_table: "assignments",
  user_id_col: "user_id",
  date_col: "event_date",
  variant_col: "variant",
  guardrail_metrics: ["refund_rate", "latency_p95"],
  segment_cols: ["country", "plan"],
};

const GATES: Record<string, Record<string, unknown>> = {
  intent: {
    message: "One detail is missing before the analysis can start.",
    question: "Which date range should the comparison cover?",
    task: "Did variant B lift signups?",
  },
  semantic_cache: {
    message: "A very similar analysis already exists.",
    hit_type: "near",
    similarity: 0.93,
    narrative_draft: "## Summary\n\nA prior run found a **21%** lift.",
    recommendation: "Reuse the earlier result.",
  },
  metric: {
    message: "Confirm the metric definition.",
    metric_config: METRIC_CONFIG,
    metric_pack_id: "pack-1",
    source: "metric_pack",
    schema_drift_warnings: [
      "Column `prior_week_revenue` was not found in the current schema.",
    ],
  },
  query: {
    message: "Review the generated SQL.",
    generated_sql:
      "SELECT variant, SUM(revenue_usd) AS revenue\nFROM transactions\nJOIN assignments USING (user_id)\nGROUP BY variant",
    sql_validation_warnings: ["No date filter — this scans the full table."],
  },
  analysis: {
    message: "Review the statistical results.",
    srm_detected: false,
    significant: true,
    top_segment: "Enterprise",
    novelty_likely: false,
    guardrails_breached: true,
    breached_metrics: [
      { metric: "refund_rate", direction: "increased" },
      { metric: "latency_p95", direction: "increased" },
    ],
    mde_powered: true,
    business_impact: "≈ $412k annualised",
    cuped_variance_reduction: 0.31,
    biggest_funnel_dropoff: "checkout → payment",
  },
  narrative: {
    message: "Review the narrative. Approve, or add notes to trigger a revision.",
    narrative_draft: DONE_STATE.narrative_draft,
    recommendation: DONE_STATE.recommendation,
  },
};

const GENERAL_ANALYSIS_GATE = {
  message: "Review what the data looks like.",
  describe_result: {
    row_count: 48_213,
    col_count: 3,
    columns: [
      {
        name: "revenue_usd", dtype: "float64", non_null: 48_000, null_count: 213,
        mean: 42.5, std: 11.2, min: 0.0, median: 40.1, max: 990.0,
      },
      {
        name: "country", dtype: "object", non_null: 48_213, null_count: 0,
        n_unique: 27, top_values: ["US", "GB", "DE"],
      },
      {
        name: "signed_up_at", dtype: "datetime64[ns]", non_null: 48_213, null_count: 0,
      },
    ],
  },
  correlation_result: {
    pairs: [
      { col_a: "revenue_usd", col_b: "sessions", correlation: 0.72 },
      { col_a: "revenue_usd", col_b: "refunds", correlation: -0.31 },
    ],
  },
};

const DECK_DATA = {
  verdict: "positive" as const,
  headline: "Variant B is the clear winner",
  hero_metric: "+21% signups",
  confidence: "High",
  recommendation: "Ship variant B to 100%.",
  evidence: [
    "Significant at p < 0.01 across 48,213 users",
    "Holds in every segment except Self-serve",
  ],
  watch_out: "Refund rate rose 0.4pp — worth a follow-up.",
};

const POWER_ANALYSIS = {
  baseline_mean: 42.5,
  baseline_std: 11.2,
  daily_traffic: 4_000,
  mde_target_pct: 3.0,
  mde_target_abs: 1.275,
  required_n_per_arm: 18_400,
  required_total_n: 36_800,
  runtime_days: 10,
  alpha: 0.05,
  power: 0.8,
  guardrails_to_watch: ["refund_rate", "latency_p95"],
  sensitivity: [
    { mde_pct: 1.0, n_per_arm: 165_000, runtime_days: 83 },
    { mde_pct: 3.0, n_per_arm: 18_400, runtime_days: 10 },
    { mde_pct: 5.0, n_per_arm: 6_700, runtime_days: 4 },
  ],
};

const STEPS = [
  { type: "step", node: "resolve_intent", label: "Understanding the question", status: "completed" },
  { type: "step", node: "generate_sql", label: "Writing SQL", status: "completed",
    detail: "SELECT variant, SUM(revenue_usd) FROM transactions GROUP BY variant" },
  { type: "step", node: "run_analysis", label: "Running statistics", status: "completed" },
];

async function gateScreen(page: Page, gate: string, payload: Record<string, unknown>, mode: "ab" | "explore" = "ab") {
  await stubApi(page, { events: [...STEPS, { type: "gate", gate, payload }] });
  await startAnalysis(page, mode);
}

// ── Styling actually survives the policy ──────────────────────────────────────

test("inline style props still take effect under the tightened policy", async ({ page }) => {
  // "No violations" is also true of a page that rendered nothing, so this
  // checks the other direction: styles declared through React's style prop are
  // actually computed. If style-src had started blocking them the app would
  // render *unstyled* rather than broken, which is the failure mode a
  // violations-only assertion cannot see.
  //
  // The finished view, not the mode picker: that screen is styled entirely by
  // CSS classes and has zero inline styles, so it would pass vacuously.
  await stubApi(page);
  await startAnalysis(page);
  await page.getByText(/Additional details/i).click();
  // Wait for a *drawn* bar, not just a mounted <svg> — reading computed styles
  // before Recharts has painted races and reports nothing applied.
  await expect(
    page.locator("svg.recharts-surface .recharts-bar-rectangle path").first(),
  ).toBeVisible({ timeout: 30_000 });

  const result = await page.evaluate(() => {
    const styled = Array.from(document.querySelectorAll<HTMLElement>("[style]"));
    const unapplied: string[] = [];
    for (const el of styled.slice(0, 60)) {
      for (const decl of (el.getAttribute("style") || "").split(";")) {
        const [prop, value] = decl.split(":").map((x) => x && x.trim());
        // Properties whose computed form is comparable without normalising.
        if (!prop || !value || !/^(display|flex-direction|min-height|font-size)$/.test(prop)) continue;
        if (!getComputedStyle(el).getPropertyValue(prop).trim()) {
          unapplied.push(`${el.tagName}.${prop} declared ${value}, computed nothing`);
        }
      }
    }
    const bar = document.querySelector("svg.recharts-surface .recharts-bar-rectangle path");
    return {
      styledCount: styled.length,
      unapplied,
      barFill: bar ? getComputedStyle(bar).fill : "",
    };
  });

  // Measured at 47 on this screen under both the loose and tightened policies —
  // identical, which is what made the tightening safe to make.
  expect(result.styledCount, "no inline-styled elements — wrong screen or nothing rendered")
    .toBeGreaterThan(20);
  expect(result.unapplied, `declared but not applied: ${result.unapplied.join(" | ")}`).toEqual([]);
  // The chart colour comes from the ChartSpec through to the SVG. Two bar charts
  // render, so accept either declared colour rather than depending on DOM order.
  expect(["rgb(122, 162, 247)", "rgb(224, 175, 104)"]).toContain(result.barFill);
});

// ── Unauthenticated screens ───────────────────────────────────────────────────

const PUBLIC_ROUTES = [
  ["/login", /sign in/i],
  ["/forgot-password", /reset|password/i],
  ["/reset-password?token=abc123", /password/i],
  ["/verify-email?token=abc123", /verif/i],
] as const;

for (const [route, marker] of PUBLIC_ROUTES) {
  test(`no CSP violations on ${route}`, async ({ page }) => {
    const violations = await watchCsp(page);
    // Signed out: /auth/me must fail, or AuthGuard redirects away from /login.
    await page.route("**/auth/me", (r) => r.fulfill({ status: 401, body: "{}" }));
    await page.goto(route);
    await expect(page.getByText(marker).first()).toBeVisible({ timeout: 15_000 });
    await expectNoViolations(page, violations, route);
  });
}

test("no CSP violations on the register form", async ({ page }) => {
  const violations = await watchCsp(page);
  await page.route("**/auth/me", (r) => r.fulfill({ status: 401, body: "{}" }));
  await page.goto("/login");
  await page.getByRole("button", { name: /create account/i }).first().click();
  await expect(page.getByRole("textbox").first()).toBeVisible();
  await expectNoViolations(page, violations, "register form");
});

// ── Authenticated screens ─────────────────────────────────────────────────────

test("no CSP violations on the mode picker", async ({ page }) => {
  const violations = await watchCsp(page);
  await stubApi(page);
  await page.goto("/");
  await expect(page.getByRole("button", { name: /A\/B Testing/i })).toBeVisible({ timeout: 15_000 });
  await expectNoViolations(page, violations, "mode picker");
});

test("no CSP violations on the A/B task form", async ({ page }) => {
  const violations = await watchCsp(page);
  await stubApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: /A\/B Testing/i }).click();
  await page.getByRole("button", { name: /Design Experiment/i }).click();
  await expect(page.getByRole("button", { name: /Calculate Sample Size|Run Analysis/i })).toBeVisible();
  await expectNoViolations(page, violations, "power-analysis form");
});

test("no CSP violations on the explore task form", async ({ page }) => {
  const violations = await watchCsp(page);
  await stubApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: /Explore & Understand/i }).click();
  await expect(page.getByRole("textbox").first()).toBeVisible();
  await expectNoViolations(page, violations, "explore form");
});

test("no CSP violations in the chain-of-thought step list", async ({ page }) => {
  const violations = await watchCsp(page);
  // Read from the finished view rather than mid-run: the stub answers the SSE
  // request with a complete body, so the stream always closes, and a run with no
  // terminal frame lands on "Connection lost" instead of the progress view.
  // Same component either way — it renders in both.
  await stubApi(page, { events: [...STEPS, { type: "done", state: DONE_STATE }] });
  await startAnalysis(page);
  await page.getByText(/Show processing details/i).click();
  await expect(page.getByText(/Writing SQL/i).first()).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText(/SELECT variant/i).first()).toBeVisible();
  await expectNoViolations(page, violations, "chain-of-thought");
});

// ── Every HITL gate ───────────────────────────────────────────────────────────

for (const [gate, payload] of Object.entries(GATES)) {
  test(`no CSP violations on the ${gate} gate`, async ({ page }) => {
    const violations = await watchCsp(page);
    await gateScreen(page, gate, payload);
    await expect(page.getByText(payload.message as string).first())
      .toBeVisible({ timeout: 20_000 });
    await expectNoViolations(page, violations, `${gate} gate`);
  });
}

test("no CSP violations on the general-analysis gate", async ({ page }) => {
  // Same gate name as the A/B one; the explore mode picks the other component.
  const violations = await watchCsp(page);
  await gateScreen(page, "analysis", GENERAL_ANALYSIS_GATE, "explore");
  await expect(page.getByText(GENERAL_ANALYSIS_GATE.message).first())
    .toBeVisible({ timeout: 20_000 });
  await expectNoViolations(page, violations, "general-analysis gate");
});

// ── Finished view, history, modals ────────────────────────────────────────────

test("no CSP violations on the finished view with details expanded", async ({ page }) => {
  const violations = await watchCsp(page);
  await stubApi(page);
  await startAnalysis(page);
  await page.getByText(/Additional details/i).click();
  await expect(page.locator("svg.recharts-surface").first()).toBeVisible({ timeout: 30_000 });
  await expectNoViolations(page, violations, "finished view");
});

test("no CSP violations on the stakeholder deck", async ({ page }) => {
  // Rendered instead of the narrative when the run produced deck_data, so a
  // finished-view test without it never touches this component.
  const violations = await watchCsp(page);
  await stubApi(page, {
    events: [{ type: "done", state: { ...DONE_STATE, deck_data: DECK_DATA } }],
  });
  await startAnalysis(page);
  await expect(page.getByText(DECK_DATA.headline).first()).toBeVisible({ timeout: 20_000 });
  await expectNoViolations(page, violations, "stakeholder deck");
});

test("no CSP violations on the power-analysis result", async ({ page }) => {
  // Its own layout: stat blocks and a sensitivity table nothing else renders.
  const violations = await watchCsp(page);
  await stubApi(page, {
    events: [{
      type: "done",
      state: {
        ...DONE_STATE,
        analysis_mode: "power_analysis",
        charts: [],
        power_analysis_result: POWER_ANALYSIS,
      },
    }],
  });
  await startAnalysis(page);
  // Like the charts, this sits behind the disclosure (hasDetails is true for a
  // power-analysis run), so the summary view alone never renders it.
  await page.getByText(/Additional details/i).click();
  await expect(page.getByText(/users per arm/i).first()).toBeVisible({ timeout: 20_000 });
  await expectNoViolations(page, violations, "power-analysis result");
});

test("no CSP violations on history with a run expanded", async ({ page }) => {
  const violations = await watchCsp(page);
  await stubApi(page);
  await page.goto("/history");
  await page.getByText("Did variant B lift signups?").click();
  await expect(page.getByRole("button", { name: /pdf report/i })).toBeVisible({ timeout: 15_000 });
  await expectNoViolations(page, violations, "history detail");
});

const MODALS = [
  ["Sources", /Manage database connections/i],
  ["Metrics", /Define your metrics once/i],
  ["Schema", /Teach DataPilot your column meanings/i],
] as const;

for (const [modal, marker] of MODALS) {
  test(`no CSP violations in the ${modal} modal`, async ({ page }) => {
    const violations = await watchCsp(page);
    await stubApi(page);
    await page.goto("/");
    await page.getByRole("button", { name: modal, exact: true }).click();
    // Assert on the modal's own copy — a bare wait would pass whether or not
    // the overlay ever mounted.
    await expect(page.getByText(marker).first()).toBeVisible({ timeout: 15_000 });
    await expectNoViolations(page, violations, `${modal} modal`);
  });
}

// ── The three surfaces the sweep used to stop short of ───────────────────────
// PackStudio's inner flow, AnnotationStudio with a live connection, and
// MembersPanel. Same discipline as everywhere else: stub the API, assert on
// the component's own copy so the screen provably rendered, then check for
// violations.

test("no CSP violations in PackStudio's template → form → save flow", async ({ page }) => {
  const violations = await watchCsp(page);
  const pack = {
    pack_id: "p-1", name: "DAU", description: "", certified: true,
    connection_id: null, version: 1,
    config: {
      primary_metric: "dau_rate", metric_source_col: "dau_flag", metric_agg: "mean",
      covariate: "pre_exp_dau", metric_direction: "higher_is_better",
      events_table: "events", experiment_table: "experiment", user_id_col: "user_id",
      date_col: "date", variant_col: "variant", week_col: "week",
      guardrail_metrics: ["notif_optout"], segment_cols: ["platform"],
    },
  };
  // Path-keyed stub serves GET and POST alike; the merged body satisfies both.
  await stubApi(page, { routes: { "/metric-packs": { metric_packs: [pack], ...pack } } });
  await page.goto("/");
  await page.getByRole("button", { name: "Metrics", exact: true }).click();
  await expect(page.getByText(/Define your metrics once/i).first()).toBeVisible({ timeout: 15_000 });

  await page.getByRole("button", { name: "DAU / engagement" }).click();
  await page.getByRole("button", { name: /Data mapping/ }).click();
  await expect(page.getByText("Value column").first()).toBeVisible();
  await page.getByRole("button", { name: /Create & certify/ }).click();
  // Saved-pack list re-renders from the stubbed GET — the flow completed.
  await expect(page.getByText("Certified").first()).toBeVisible({ timeout: 15_000 });
  await expectNoViolations(page, violations, "PackStudio inner flow");
});

test("no CSP violations in AnnotationStudio with a live connection", async ({ page }) => {
  const violations = await watchCsp(page);
  await stubApi(page, {
    routes: {
      "/connections/conn-ok/annotations": {
        connection_id: "conn-ok",
        annotations: { events: { revenue_usd: "Gross revenue in USD" } },
        synonyms: { WAU: "weekly_active_users" },
        updated_at: "2026-08-02T00:00:00Z",
      },
    },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Schema", exact: true }).click();
  await expect(page.getByText(/Teach DataPilot your column meanings/i).first()).toBeVisible({ timeout: 15_000 });
  // Populated state, not the empty state: the saved annotation and synonym
  // rows rendered. React-controlled inputs carry values as properties, not
  // attributes, so read them rather than using a CSS value selector.
  const inputValues = () =>
    page.locator("input").evaluateAll((els) => els.map((el) => (el as HTMLInputElement).value));
  await expect.poll(inputValues, { timeout: 15_000 }).toContain("Gross revenue in USD");
  await expect.poll(inputValues).toContain("weekly_active_users");
  await expectNoViolations(page, violations, "AnnotationStudio live connection");
});

test("no CSP violations in MembersPanel", async ({ page }) => {
  const violations = await watchCsp(page);
  await stubApi(page, {
    routes: {
      "/workspaces": {
        workspaces: [{ workspace_id: "ws-1", name: "Acme", role: "owner" }],
      },
      "/workspaces/ws-1/members": {
        members: [
          { user_id: "u-1", username: "ana", email: "ana@acme.test", role: "owner", created_at: "2026-08-01" },
          { user_id: "u-2", username: "bo", email: "bo@acme.test", role: "analyst", created_at: "2026-08-02" },
        ],
      },
    },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Team" }).click();
  await expect(page.getByText("ana@acme.test")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText("bo@acme.test")).toBeVisible();
  await expectNoViolations(page, violations, "MembersPanel");
});
