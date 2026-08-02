/**
 * csp-render.spec.ts — the surfaces the route-level sweep never reached.
 *
 * `prod-auth.spec.ts` walks the authenticated routes and asserts zero CSP
 * violations, but a fresh account has no completed run, so it never rendered a
 * chart and never clicked the PDF button. Charts additionally sit behind the
 * "Additional details" disclosure, so even an account *with* runs would not have
 * reached them.
 *
 * The two detector tests below are not ceremony. Tightening the policy and
 * re-running an earlier version of this file did *not* fail anything, which is
 * how the gap was found: a detector that never fires and a policy that never
 * blocks are indistinguishable from a green result.
 */
import { test, expect } from "@playwright/test";

import {
  API,
  CHARTS,
  domViolations,
  expectNoViolations,
  startAnalysis,
  stubApi,
  watchCsp,
} from "./csp-helpers";

test.describe("CSP under real rendering", () => {
  test("the served policy is enforced, not report-only", async ({ page }) => {
    // If this regresses to report-only every other test still passes while
    // proving nothing, so it is asserted separately and first.
    const res = await page.goto("/index.html");
    const headers = res!.headers();
    expect(headers["content-security-policy"], "no enforced CSP on the SPA").toBeTruthy();
    expect(headers["content-security-policy-report-only"]).toBeUndefined();
    expect(headers["content-security-policy"]).toContain(API);
  });

  test("an inline script is blocked and reported", async ({ page }) => {
    const violations = await watchCsp(page);
    await page.goto("/index.html");
    await page.evaluate(() => {
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

  test("an inline style attribute is blocked and reported", async ({ page }) => {
    // The script check above does not cover style-src: separate directive,
    // separate fallback. This is what makes the "no violations" results on
    // chart rendering mean anything, since those depend on style-src-attr
    // being permissive enough for what React and Recharts do.
    const violations = await watchCsp(page);
    await page.goto("/index.html");
    await page.evaluate(() => {
      const el = document.createElement("div");
      el.id = "csp-style-probe";
      // setAttribute goes through the HTML attribute path, which style-src-attr
      // governs. Assigning el.style.color would go through the CSSOM instead,
      // which CSP does not police at all — and that distinction is the whole
      // reason 'unsafe-inline' turned out not to be load-bearing here.
      el.setAttribute("style", "color: rgb(1, 2, 3)");
      document.body.appendChild(el);
    });
    const colour = await page.evaluate(
      () => getComputedStyle(document.getElementById("csp-style-probe")!).color,
    );
    expect(colour, "the style attribute applied — style-src is not enforcing")
      .not.toBe("rgb(1, 2, 3)");
    await expect
      .poll(async () => (await domViolations(page)).filter((v) => /style/i.test(v)).length, {
        timeout: 5_000,
      })
      .toBeGreaterThan(0);
    expect(violations.length, "console channel saw nothing").toBeGreaterThan(0);
  });

  test("rendered charts produce no CSP violations", async ({ page }) => {
    const violations = await watchCsp(page);
    await stubApi(page);
    await startAnalysis(page);
    await page.getByText(/Additional details/i).click();

    // One chart per spec must actually exist.
    await expect(page.locator("svg.recharts-surface")).toHaveCount(CHARTS.length, {
      timeout: 30_000,
    });
    // A drawn bar, not just a mounted <svg> — an empty chart shell would satisfy
    // the count above while proving nothing about what the policy allowed.
    await expect(
      page.locator("svg.recharts-surface .recharts-bar-rectangle path").first(),
    ).toBeVisible();
    // And the styling landed, so a silently unstyled chart cannot pass either.
    const filled = await page.evaluate(() => {
      const bar = document.querySelector("svg.recharts-surface .recharts-bar-rectangle path");
      return bar ? getComputedStyle(bar).fill : "";
    });
    expect(filled, "the bar rendered with no fill").not.toBe("");
    expect(filled).not.toBe("none");

    await expectNoViolations(page, violations, "chart rendering");
  });

  test("the CSV export blob is not blocked", async ({ page }) => {
    const violations = await watchCsp(page);
    await stubApi(page);
    await startAnalysis(page);
    await page.getByText(/Additional details/i).click();
    await expect(page.locator("svg.recharts-surface").first()).toBeVisible({ timeout: 30_000 });

    const download = page.waitForEvent("download", { timeout: 15_000 });
    await page.getByRole("button", { name: /csv/i }).first().click();
    expect((await download).suggestedFilename()).toContain(".csv");

    await expectNoViolations(page, violations, "CSV export");
  });

  test("the PDF download opens without a CSP violation", async ({ page, context }) => {
    const violations = await watchCsp(page);
    await stubApi(page);

    await page.goto("/history");
    await page.getByText("Did variant B lift signups?").click();
    const pdfButton = page.getByRole("button", { name: /pdf report/i });
    await expect(pdfButton).toBeVisible({ timeout: 15_000 });

    // window.open on a cross-origin URL answering with
    // `Content-Disposition: attachment` — Chrome may surface it as a popup that
    // downloads and closes, or as a download on the current page. Accept either;
    // the assertion under test is that nothing was refused.
    const settled = Promise.race([
      context.waitForEvent("page", { timeout: 15_000 }).then((p) => `popup:${p.url()}`),
      page.waitForEvent("download", { timeout: 15_000 }).then((d) => `download:${d.suggestedFilename()}`),
    ]).catch(() => "none");

    await pdfButton.click();
    expect(await settled, "the PDF request never left the page").not.toBe("none");

    await expectNoViolations(page, violations, "PDF download");
  });
});
