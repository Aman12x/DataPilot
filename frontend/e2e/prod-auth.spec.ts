/**
 * Diagnostic e2e against the deployed app (not the local dev stack).
 *
 * Run with:
 *   npx playwright test --config=playwright.prod.config.mjs
 *
 * Captures console output, page errors and failed/4xx-5xx requests so a UI
 * auth failure can be told apart from an API one -- the API returns 201/200
 * for register/login when driven directly with curl.
 */
import { test, expect, Page } from "@playwright/test";

type Diag = { console: string[]; pageErrors: string[]; failed: string[]; responses: string[] };

function attach(page: Page): Diag {
  const d: Diag = { console: [], pageErrors: [], failed: [], responses: [] };
  page.on("console", (m) => d.console.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => d.pageErrors.push(String(e)));
  page.on("requestfailed", (r) =>
    d.failed.push(`${r.method()} ${r.url()} -- ${r.failure()?.errorText}`),
  );
  page.on("response", async (r) => {
    if (r.status() >= 400) {
      let body = "";
      try { body = (await r.text()).slice(0, 300); } catch { /* opaque */ }
      d.responses.push(`${r.status()} ${r.request().method()} ${r.url()} :: ${body}`);
    }
  });
  return d;
}

function dump(name: string, d: Diag) {
  const section = (label: string, lines: string[]) =>
    lines.length ? `\n--- ${label} ---\n${lines.join("\n")}` : "";
  console.log(
    `\n===== ${name} =====` +
      section("console", d.console) +
      section("page errors", d.pageErrors) +
      section("failed requests", d.failed) +
      section("http >=400", d.responses) +
      "\n",
  );
}

test("login page renders and its form is usable", async ({ page }) => {
  const d = attach(page);
  await page.goto("/login", { waitUntil: "networkidle" });
  await page.screenshot({ path: "e2e-out/01-login.png", fullPage: true });
  dump("login page load", d);

  // The whole app is a single #root mount; an empty root means a JS failure.
  const rootHtml = await page.locator("#root").innerHTML();
  expect(rootHtml.length, "#root is empty -- the SPA did not mount").toBeGreaterThan(50);

  await expect(page.getByRole("button", { name: /sign in|log ?in/i }).first()).toBeVisible();
});

test("signup through the UI reaches the app", async ({ page }) => {
  const d = attach(page);
  const stamp = Date.now();
  const creds = {
    username: `e2eprobe${stamp}`,
    email: `e2eprobe${stamp}@example.com`,
    password: "Str0ngTestPass!x",
  };

  await page.goto("/login", { waitUntil: "networkidle" });

  // Switch to the register tab.
  const registerTab = page.getByRole("button", { name: /sign ?up|register|create account/i }).first();
  if (await registerTab.count()) await registerTab.click();

  // FormField renders <label> without htmlFor and <input> without id, so the
  // two are not associated and getByLabel cannot find these fields.
  await page.getByPlaceholder("john_doe").fill(creds.username);
  await page.getByPlaceholder("you@example.com").fill(creds.email);
  const pwds = page.locator('input[type="password"]');
  await pwds.nth(0).fill(creds.password);
  if ((await pwds.count()) > 1) await pwds.nth(1).fill(creds.password);

  await page.screenshot({ path: "e2e-out/02-signup-filled.png", fullPage: true });
  await page.getByRole("button", { name: /sign ?up|create account/i }).last().click();

  await page.waitForTimeout(8000);
  await page.screenshot({ path: "e2e-out/03-after-signup.png", fullPage: true });
  dump("signup", d);

  console.log("URL after signup:", page.url());
  console.log("visible text:", (await page.locator("body").innerText()).slice(0, 600));
});

test("login through the UI with a freshly created account", async ({ page, request }) => {
  const d = attach(page);
  const stamp = Date.now();
  const email = `e2elogin${stamp}@example.com`;
  const password = "Str0ngTestPass!x";

  // Create the account via the API so this test isolates the *login* UI.
  const reg = await request.post("https://datapilot.singhaman.dev/auth/register", {
    data: { username: `e2elogin${stamp}`, email, password },
  });
  expect(reg.status(), `register API failed: ${await reg.text()}`).toBe(201);

  await page.goto("/login", { waitUntil: "networkidle" });
  await page.getByPlaceholder("you@example.com").fill(email);
  await page.locator('input[type="password"]').first().fill(password);
  await page.getByRole("button", { name: /sign in|log ?in/i }).last().click();

  await page.waitForTimeout(8000);
  await page.screenshot({ path: "e2e-out/04-after-login.png", fullPage: true });
  dump("login", d);

  console.log("URL after login:", page.url());
  console.log("visible text:", (await page.locator("body").innerText()).slice(0, 600));
});
