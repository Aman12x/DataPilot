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

  // Every field must be reachable by its label: screen readers rely on the
  // association, and clicking a label should focus its input.
  for (const name of ["Username or email", "Password"]) {
    const field = page.getByLabel(name, { exact: true });
    await expect(field, `"${name}" is not associated with an input`).toHaveCount(1);
    const id = await field.getAttribute("id");
    expect(id, `"${name}" input has no id`).toBeTruthy();
  }
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

  // getByLabel doubles as a regression test: FormField previously rendered the
  // label and input as unassociated siblings, so these queries found nothing.
  await page.getByLabel("Username", { exact: true }).fill(creds.username);
  await page.getByLabel("Email", { exact: true }).fill(creds.email);
  await page.getByLabel("Password", { exact: true }).fill(creds.password);
  const confirm = page.getByLabel("Confirm password", { exact: true });
  if (await confirm.count()) await confirm.fill(creds.password);

  await page.screenshot({ path: "e2e-out/02-signup-filled.png", fullPage: true });
  // Target the form's submit button structurally, not by label. The register
  // *tab* is also a button and reads "Create Account", while the submit reads
  // "Create your account" -- which /create account/i does not match. So a
  // name-based .last() selected the tab, and the tab's onClick resets
  // passwordConfirm and submits nothing: the run failed 30s later at "Sign out",
  // with no request ever sent. Labels here have already changed once.
  await page.locator('form button[type="submit"]').click();

  // Wait on the post-auth shell, not a fixed sleep: a timeout-based wait made
  // this suite report success while the browser was still sitting on /login.
  const signedIn = page.getByRole("button", { name: /sign out/i });
  await expect(signedIn, "signup did not reach the authenticated app").toBeVisible({
    timeout: 30_000,
  });

  await page.screenshot({ path: "e2e-out/03-after-signup.png", fullPage: true });
  dump("signup", d);
  expect(page.url(), "still on /login after signup").not.toContain("/login");
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
  await page.getByLabel("Username or email", { exact: true }).fill(email);
  await page.getByLabel("Password", { exact: true }).fill(password);
  // Same structural selector as the signup test: /sign in/i matches both the
  // "Sign In" tab and the "Sign in to DataPilot" submit, so .last() was correct
  // only by DOM ordering.
  await page.locator('form button[type="submit"]').click();

  const signedIn = page.getByRole("button", { name: /sign out/i });
  await expect(signedIn, "login did not reach the authenticated app").toBeVisible({
    timeout: 30_000,
  });

  await page.screenshot({ path: "e2e-out/04-after-login.png", fullPage: true });
  dump("login", d);
  expect(page.url(), "still on /login after login").not.toContain("/login");
  await expect(page.getByText(email.split("@")[0]), "signed-in user not shown").toBeVisible();
});


test("no CSP violations across the authenticated app", async ({ page, request }) => {
  const d = attach(page);
  const violations: string[] = [];
  page.on("console", (m) => {
    if (/Refused to|Content Security Policy|violates/i.test(m.text())) violations.push(m.text());
  });
  // The DOM event catches violations the console formats differently.
  await page.addInitScript(() => {
    document.addEventListener("securitypolicyviolation", (e) => {
      const w = window as unknown as { __csp?: string[] };
      (w.__csp = w.__csp || []).push(`${e.violatedDirective} <- ${e.blockedURI || "(inline)"}`);
    });
  });

  const stamp = Date.now();
  const email = `cspcheck${stamp}@example.com`;
  const password = "Str0ngTestPass!x";
  const reg = await request.post("https://datapilot.singhaman.dev/auth/register", {
    data: { username: `cspcheck${stamp}`, email, password },
  });
  expect(reg.status()).toBe(201);

  await page.goto("/login", { waitUntil: "networkidle" });
  await page.getByLabel("Username or email", { exact: true }).fill(email);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.locator('form button[type="submit"]').click();
  await expect(page.getByRole("button", { name: /sign out/i })).toBeVisible({ timeout: 30_000 });

  for (const route of ["/", "/history", "/"]) {
    await page.goto(route, { waitUntil: "networkidle" });
    await expect(page.locator("#root")).not.toBeEmpty();
  }

  const dom = await page.evaluate(() => (window as unknown as { __csp?: string[] }).__csp || []);
  dump("csp sweep", d);
  expect(violations, `console CSP violations: ${violations.join(" | ")}`).toEqual([]);
  expect(dom, `blocked by CSP: ${dom.join(" | ")}`).toEqual([]);
});
