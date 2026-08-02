/**
 * csp-render.spec.ts — the two surfaces the CSP sweep never reached.
 *
 * `prod-auth.spec.ts` walks the authenticated routes and asserts zero CSP
 * violations, but a fresh account has no completed run, so it never renders a
 * chart and never clicks the PDF button. Those were the two documented gaps:
 * Recharts injects styles at runtime, and the PDF opens a cross-origin URL in a
 * new tab — both are exactly the kind of thing a policy written from reading
 * the code gets wrong.
 *
 * Runs against the production build served with the generated `dist/serve.json`
 * (see playwright.csp.config.mjs). The API is stubbed: the policy governs what
 * the *page* may do, so real markup, real bundle, and real headers are what
 * matter — real data is not.
 */
import { test, expect, type Page } from "@playwright/test";

const API = "http://127.0.0.1:8899";
const RUN_ID = "11111111-2222-3333-4444-555555555555";
const ORIGIN = "http://127.0.0.1:4173";

// Every chart type ChartCard supports, so a violation in one renderer cannot
// hide behind another. Error bars and dual series are separate code paths.
const CHARTS = [
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

const DONE_STATE = {
  run_id: RUN_ID,
  narrative_draft: "## Summary\n\nVariant B lifted signups by **21%**.",
  recommendation: "Ship variant B.",
  analysis_mode: "ab_test",
  charts: CHARTS,
  trust_indicators: {
    n_data_points: 48_213,
    confidence_level: "high",
    confidence_reason: "Sample size and balance both check out.",
  },
};

// A minimal but genuinely valid PDF, so the browser handles a real one.
const PDF_BYTES = Buffer.from(
  "%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n" +
    "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n" +
    "3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n" +
    "trailer<</Root 1 0 R>>\n%%EOF\n",
  "utf8",
);

const HISTORY_RUN = {
  run_id: RUN_ID,
  task: "Did variant B lift signups?",
  timestamp: "2026-07-30T12:00:00Z",
  analysis_mode: "ab_test",
  metric: "signups",
  eval_score: 0.86,
  username: "cspuser",
};

type Violation = string;

/** Collect CSP violations from both the console and the DOM event. */
async function watchCsp(page: Page): Promise<Violation[]> {
  const violations: Violation[] = [];
  page.on("console", (m) => {
    if (/Refused to|Content Security Policy|violates/i.test(m.text())) violations.push(m.text());
  });
  page.on("pageerror", (e) => violations.push(`pageerror: ${e.message}`));
  await page.addInitScript(() => {
    document.addEventListener("securitypolicyviolation", (e) => {
      const w = window as unknown as { __csp?: string[] };
      (w.__csp = w.__csp || []).push(
        `${e.violatedDirective} <- ${e.blockedURI || "(inline)"} @ ${e.sourceFile || "?"}`,
      );
    });
  });
  return violations;
}

async function domViolations(page: Page): Promise<string[]> {
  return page.evaluate(() => (window as unknown as { __csp?: string[] }).__csp || []);
}

/** Stub every API call the app makes, including the cross-origin SSE stream. */
async function stubApi(page: Page, origin: string) {
  // The real API is cross-origin and the client sends credentials, so the stub
  // has to satisfy CORS properly: a wildcard Allow-Origin is rejected outright
  // on a credentialed request, the app falls back to /login, and the failure
  // looks nothing like the CORS problem it is.
  const cors = {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Credentials": "true",
  };
  const json = (body: unknown) => ({
    status: 200,
    contentType: "application/json",
    headers: cors,
    body: JSON.stringify(body),
  });

  await page.route(`${API}/**`, async (route) => {
    const url = new URL(route.request().url());
    const p = url.pathname;

    if (route.request().method() === "OPTIONS") {
      return route.fulfill({
        status: 204,
        headers: {
          ...cors,
          "Access-Control-Allow-Headers": "*",
          "Access-Control-Allow-Methods": "*",
        },
      });
    }
    if (p === "/auth/me") return route.fulfill(json({ username: "cspuser", user_id: "u-1" }));
    if (p === "/workspaces") return route.fulfill(json({ workspaces: [] }));
    if (p === "/metric-packs") return route.fulfill(json({ metric_packs: [] }));
    if (p === "/samples") return route.fulfill(json([]));
    if (p === "/connections") return route.fulfill(json({ connections: [] }));
    if (p === "/runs" && route.request().method() === "POST")
      return route.fulfill(json({ run_id: RUN_ID }));
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
        body:
          `data: ${JSON.stringify({ type: "step", node: "generate_narrative", label: "Writing", status: "completed" })}\n\n` +
          `data: ${JSON.stringify({ type: "done", state: DONE_STATE })}\n\n`,
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
async function startAnalysis(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: /A\/B Testing/i }).click();
  await page.getByRole("button", { name: /Interpret Results/i }).click();
  await page.getByRole("textbox").first().fill("Did variant B lift signups?");
  await page.getByRole("button", { name: /Run Analysis/i }).click();
  // Charts live behind the "Additional details" disclosure, which is why the
  // route-level sweep never rendered one even on an account that had runs.
  await page.getByText(/Additional details/i).click();
}

test.describe("CSP under real rendering", () => {
  test("the served policy is enforced, not report-only", async ({ page }) => {
    // If this ever regresses to report-only the other tests still pass while
    // proving nothing, so it is asserted separately and first.
    const res = await page.goto("/index.html");
    const headers = res!.headers();
    expect(headers["content-security-policy"], "no enforced CSP on the SPA").toBeTruthy();
    expect(headers["content-security-policy-report-only"]).toBeUndefined();
    expect(headers["content-security-policy"]).toContain(API);
  });

  test("the violation detector actually detects a violation", async ({ page }) => {
    // Without this, every "no violations" result below is unfalsifiable — a
    // detector that never fires and a policy that never blocks look identical.
    // Tightening style-src in runtime-config.js and re-running this file was
    // *not* enough to fail the chart test, which is how the gap was found: React
    // and Recharts set styles through the CSSOM, and CSP does not govern that.
    const violations = await watchCsp(page);
    await page.goto("/index.html");
    await page.evaluate(() => {
      // script-src 'self' — an injected inline script is unambiguously blocked.
      const el = document.createElement("script");
      el.textContent = "window.__pwned = true;";
      document.body.appendChild(el);
    });
    await expect
      .poll(async () => (await domViolations(page)).length, { timeout: 5_000 })
      .toBeGreaterThan(0);
    expect(await page.evaluate(() => (window as unknown as { __pwned?: boolean }).__pwned))
      .toBeUndefined();
    expect(violations.length, "console channel saw nothing").toBeGreaterThan(0);
  });

  test("rendered charts produce no CSP violations", async ({ page }) => {
    const violations = await watchCsp(page);
    await stubApi(page, ORIGIN);

    await startAnalysis(page);

    // Recharts renders to SVG; wait for one chart per spec to actually exist.
    await expect(page.locator("svg.recharts-surface")).toHaveCount(CHARTS.length, {
      timeout: 30_000,
    });
    // A drawn bar, not just a mounted <svg> — an empty chart shell would satisfy
    // the count above while proving nothing about what the policy allowed.
    await expect(
      page.locator("svg.recharts-surface .recharts-bar-rectangle path").first(),
    ).toBeVisible();

    // Inline styles are the one thing this policy has to permit for charts:
    // React style props compile to style attributes and Recharts sets more at
    // runtime. If style-src had dropped 'unsafe-inline' these would be gone and
    // the charts would render unstyled rather than not at all.
    const inlineStyled = await page.locator("svg.recharts-surface [style]").count();
    expect(inlineStyled, "no inline styles survived inside the chart").toBeGreaterThan(0);

    const dom = await domViolations(page);
    expect(dom, `blocked by CSP while rendering charts: ${dom.join(" | ")}`).toEqual([]);
    expect(violations, `console CSP violations: ${violations.join(" | ")}`).toEqual([]);
  });

  test("the CSV export blob is not blocked", async ({ page }) => {
    // Same screen as the charts: it builds a blob: URL and clicks an anchor.
    const violations = await watchCsp(page);
    await stubApi(page, ORIGIN);

    await startAnalysis(page);
    await expect(page.locator("svg.recharts-surface").first()).toBeVisible({ timeout: 30_000 });

    const download = page.waitForEvent("download", { timeout: 15_000 });
    await page.getByRole("button", { name: /csv/i }).first().click();
    expect((await download).suggestedFilename()).toContain(".csv");

    const dom = await domViolations(page);
    expect(dom, `blocked by CSP during CSV export: ${dom.join(" | ")}`).toEqual([]);
    expect(violations).toEqual([]);
  });

  test("the PDF download opens without a CSP violation", async ({ page, context }) => {
    const violations = await watchCsp(page);
    await stubApi(page, ORIGIN);

    await page.goto("/history");
    await page.getByText("Did variant B lift signups?").click();
    const pdfButton = page.getByRole("button", { name: /pdf report/i });
    await expect(pdfButton).toBeVisible({ timeout: 15_000 });

    // window.open on a cross-origin URL that answers with
    // `Content-Disposition: attachment` — Chrome may surface it as a popup that
    // downloads and closes, or as a download on the current page. Accept either;
    // the assertion under test is that nothing was refused.
    const settled = Promise.race([
      context.waitForEvent("page", { timeout: 15_000 }).then((p) => `popup:${p.url()}`),
      page.waitForEvent("download", { timeout: 15_000 }).then((d) => `download:${d.suggestedFilename()}`),
    ]).catch(() => "none");

    await pdfButton.click();
    const outcome = await settled;
    expect(outcome, "the PDF request never left the page").not.toBe("none");

    const dom = await domViolations(page);
    expect(dom, `blocked by CSP during PDF download: ${dom.join(" | ")}`).toEqual([]);
    expect(violations, `console CSP violations: ${violations.join(" | ")}`).toEqual([]);
  });
});
