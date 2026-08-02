/**
 * csp-helpers.ts — shared rig for the CSP specs.
 *
 * Both specs run against the production build served with the generated
 * `dist/serve.json` (see playwright.csp.config.mjs) and stub the API entirely:
 * the policy governs what the *page* is allowed to do, so real markup, real
 * bundle, and real headers are what matter — real data is not.
 */
import { expect, type Page } from "@playwright/test";

export const API = "http://127.0.0.1:8899";
export const ORIGIN = "http://127.0.0.1:4173";
export const RUN_ID = "11111111-2222-3333-4444-555555555555";

/** Collect CSP violations from both the console and the DOM event. */
export async function watchCsp(page: Page): Promise<string[]> {
  const violations: string[] = [];
  page.on("console", (m) => {
    if (/Refused to|Content Security Policy|violates/i.test(m.text())) violations.push(m.text());
  });
  page.on("pageerror", (e) => violations.push(`pageerror: ${e.message}`));
  await page.addInitScript(() => {
    document.addEventListener("securitypolicyviolation", (e) => {
      const w = window as unknown as { __csp?: string[] };
      (w.__csp = w.__csp || []).push(
        `${e.violatedDirective} <- ${e.blockedURI || "(inline)"} @ ${e.sourceFile || "?"}:${e.lineNumber || 0}`,
      );
    });
  });
  return violations;
}

export async function domViolations(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { __csp?: string[] }).__csp || []);
}

/** Assert both violation channels are empty, naming what was being exercised. */
export async function expectNoViolations(page: Page, consoleHits: string[], what: string) {
  const dom = await domViolations(page);
  expect(dom, `blocked by CSP while exercising ${what}: ${dom.join(" | ")}`).toEqual([]);
  expect(consoleHits, `console CSP violations during ${what}: ${consoleHits.join(" | ")}`).toEqual([]);
}

// A minimal but genuinely valid PDF, so the browser handles a real one.
export const PDF_BYTES = Buffer.from(
  "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n" +
    "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n" +
    "3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n" +
    "trailer<</Root 1 0 R>>\n%%EOF\n",
  "utf8",
);

// Every chart type ChartCard supports, so a violation in one renderer cannot
// hide behind another. Error bars and dual series are separate code paths.
export const CHARTS = [
  {
    chart_type: "bar", title: "Signups by variant", insight: "Variant **B** leads.",
    data: [
      { variant: "A", signups: 120, lo: 8, hi: 8 },
      { variant: "B", signups: 168, lo: 9, hi: 9 },
    ],
    x_key: "variant", y_key: "signups", color: "#7aa2f7",
    error_bar_low: "lo", error_bar_high: "hi",
    x_label: "Variant", y_label: "Signups",
  },
  {
    chart_type: "line", title: "Daily conversion", insight: "Trending up.",
    data: Array.from({ length: 14 }, (_, i) => ({
      day: `2026-07-${String(i + 1).padStart(2, "0")}`,
      rate: 0.1 + i * 0.004,
      baseline: 0.1,
    })),
    x_key: "day", y_key: "rate", y_key2: "baseline",
    color: "#9ece6a", color2: "#f7768e",
  },
  {
    chart_type: "scatter", title: "Spend vs revenue", insight: "Positive.",
    data: Array.from({ length: 30 }, (_, i) => ({ spend: i * 3, revenue: i * 7 + (i % 5) })),
    x_key: "spend", y_key: "revenue", color: "#bb9af7",
  },
  {
    chart_type: "bar_horizontal", title: "Lift by segment", insight: "Enterprise leads.",
    data: [
      { segment: "Enterprise", lift: 0.21 },
      { segment: "SMB", lift: 0.08 },
      { segment: "Self-serve", lift: -0.03 },
    ],
    x_key: "segment", y_key: "lift", color: "#e0af68",
  },
];

export const DONE_STATE = {
  run_id: RUN_ID,
  narrative_draft: "## Summary\n\nVariant B lifted signups by **21%**.\n\n- point one\n- point two",
  recommendation: "Ship variant B.",
  analysis_mode: "ab_test",
  charts: CHARTS,
  trust_indicators: {
    n_data_points: 48_213,
    confidence_level: "high",
    confidence_reason: "Sample size and balance both check out.",
  },
};

export const HISTORY_RUN = {
  run_id: RUN_ID,
  task: "Did variant B lift signups?",
  timestamp: "2026-07-30T12:00:00Z",
  analysis_mode: "ab_test",
  metric: "signups",
  eval_score: 0.86,
  username: "cspuser",
};

export interface StubOptions {
  /** SSE frames to emit, in order. Defaults to the finished A/B analysis. */
  events?: unknown[];
  /** Extra path → JSON body overrides. */
  routes?: Record<string, unknown>;
}

/** Stub every API call the app makes, including the cross-origin SSE stream. */
export async function stubApi(page: Page, opts: StubOptions = {}) {
  // The real API is cross-origin and the client sends credentials, so the stub
  // has to satisfy CORS properly: a wildcard Allow-Origin is rejected outright
  // on a credentialed request, the app falls back to /login, and the failure
  // looks nothing like the CORS problem it is.
  const cors = {
    "Access-Control-Allow-Origin": ORIGIN,
    "Access-Control-Allow-Credentials": "true",
  };
  const json = (body: unknown) => ({
    status: 200,
    contentType: "application/json",
    headers: cors,
    body: JSON.stringify(body),
  });
  const events = opts.events ?? [{ type: "done", state: DONE_STATE }];

  await page.route(`${API}/**`, async (route) => {
    const p = new URL(route.request().url()).pathname;
    const method = route.request().method();

    if (method === "OPTIONS") {
      return route.fulfill({
        status: 204,
        headers: { ...cors, "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Methods": "*" },
      });
    }
    if (opts.routes && p in opts.routes) return route.fulfill(json(opts.routes[p]));

    if (p === "/auth/me") return route.fulfill(json({ username: "cspuser", user_id: "u-1" }));
    if (p === "/workspaces") return route.fulfill(json({ workspaces: [] }));
    if (p === "/metric-packs") return route.fulfill(json({ metric_packs: [] }));
    if (p === "/samples") return route.fulfill(json([]));
    if (p === "/connections")
      return route.fulfill(json({
        connections: [
          {
            connection_id: "conn-ok", name: "Prod warehouse", backend: "postgres",
            host: "db.example.com", port: 5432, dbname: "analytics", username: "reader",
            sslmode: "require", last_test_ok: true,
            last_tested_at: "2026-08-02T00:00:00Z", last_test_error: null, project_id: null,
          },
          {
            connection_id: "conn-bad", name: "Staging", backend: "mysql",
            host: "stage.example.com", port: 3306, dbname: "app", username: "root",
            sslmode: "prefer", last_test_ok: false,
            last_tested_at: "2026-08-01T00:00:00Z",
            last_test_error: "authentication failed for user root", project_id: null,
          },
        ],
      }));
    if (p === "/runs" && method === "POST") return route.fulfill(json({ run_id: RUN_ID }));
    if (p === "/runs") return route.fulfill(json([HISTORY_RUN]));
    if (p === `/runs/${RUN_ID}/stream-token`) return route.fulfill(json({ stream_token: "s" }));
    if (p === `/runs/${RUN_ID}/pdf-token`) return route.fulfill(json({ pdf_token: "p" }));
    if (p === `/runs/${RUN_ID}/detail`)
      return route.fulfill(json({
        task: HISTORY_RUN.task,
        narrative: DONE_STATE.narrative_draft,
        recommendation: DONE_STATE.recommendation,
      }));
    if (p === `/runs/${RUN_ID}/stream`) {
      return route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        headers: { ...cors, "Cache-Control": "no-cache" },
        body: events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""),
      });
    }
    if (p === `/runs/${RUN_ID}/pdf`) {
      return route.fulfill({
        status: 200,
        contentType: "application/pdf",
        headers: {
          ...cors,
          "Content-Disposition": `attachment; filename="datapilot-${RUN_ID.slice(0, 8)}.pdf"`,
        },
        body: PDF_BYTES,
      });
    }
    return route.fulfill(json({}));
  });
}

/** Walk the mode picker to the task form and start the (stubbed) run. */
export async function startAnalysis(page: Page, mode: "ab" | "explore" = "ab") {
  await page.goto("/");
  if (mode === "explore") {
    await page.getByRole("button", { name: /Explore & Understand/i }).click();
  } else {
    await page.getByRole("button", { name: /A\/B Testing/i }).click();
    await page.getByRole("button", { name: /Interpret Results/i }).click();
  }
  await page.getByRole("textbox").first().fill("Did variant B lift signups?");
  await page.getByRole("button", { name: /Run Analysis|Explore Data/i }).click();
}
