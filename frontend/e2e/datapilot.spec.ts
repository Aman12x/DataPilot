import { expect, test, type Page } from "@playwright/test";

async function mockAuthenticatedShell(page: Page) {
  await page.route("**/api/auth/me", async route => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ username: "e2e_user", email: "e2e@example.com" }),
    });
  });

  await page.route("**/api/samples", async route => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([]),
    });
  });
}

test("unauthenticated and stale guest sessions are redirected to login", async ({ page }) => {
  await page.goto("/");

  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("button", { name: /Continue as Guest/i })).toHaveCount(0);

  await page.evaluate(() => localStorage.setItem("access_token", "guest"));
  await page.goto("/");

  await expect(page).toHaveURL(/\/login$/);
  await expect.poll(() => page.evaluate(() => localStorage.getItem("access_token"))).toBeNull();
});

test("registration stores tokens and opens the authenticated analysis shell", async ({ page }) => {
  await page.route("**/api/auth/register", async route => {
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({
        access_token: "access-e2e",
        refresh_token: "refresh-e2e",
        user: { username: "new_user" },
      }),
    });
  });
  await mockAuthenticatedShell(page);

  await page.goto("/login");
  await page.getByRole("button", { name: "Create Account" }).click();
  await page.getByPlaceholder("john_doe").fill("new_user");
  await page.getByPlaceholder("you@example.com").fill("new@example.com");
  await page.getByPlaceholder("••••••••").fill("correct horse battery staple");
  await page.getByRole("button", { name: /Create Account/ }).last().click();

  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByText("What would you like to do?")).toBeVisible();
  await expect.poll(() => page.evaluate(() => localStorage.getItem("access_token"))).toBe("access-e2e");
  await expect(page.getByText("e2e_user")).toBeVisible();
});

test("upload-driven analysis posts the uploaded DuckDB id and consumes SSE completion", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("access_token", "access-e2e");
    localStorage.setItem("refresh_token", "refresh-e2e");
  });
  await mockAuthenticatedShell(page);

  await page.route("**/api/upload", async route => {
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({
        upload_id: "11111111-1111-4111-8111-111111111111",
        columns: ["date", "variant", "conversions"],
        row_count: 2,
        preview: [{ date: "2026-01-01", variant: "control", conversions: 10 }],
      }),
    });
  });

  let runRequest: Record<string, unknown> | undefined;
  await page.route("**/api/runs", async route => {
    runRequest = route.request().postDataJSON() as Record<string, unknown>;
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({ run_id: "run-e2e" }),
    });
  });

  await page.route("**/api/runs/run-e2e/stream?**", async route => {
    const done = {
      type: "done",
      state: {
        run_id: "run-e2e",
        narrative_draft: "The uploaded dataset completed successfully.",
        recommendation: "Ship it.",
        charts: [],
        trust_indicators: {
          n_data_points: 2,
          confidence_level: "medium",
          confidence_reason: "E2E mocked result",
        },
        analysis_mode: "general",
      },
    };
    await route.fulfill({
      status: 200,
      headers: {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
      },
      body: `id: 1\ndata: ${JSON.stringify({ type: "step", node: "load_schema", label: "Reading database schema", status: "completed" })}\n\nid: 2\ndata: ${JSON.stringify(done)}\n\n`,
    });
  });

  await page.goto("/");
  await page.getByText("Explore & Understand").click();
  await page.getByRole("textbox").fill("Find conversion patterns in this upload.");
  await page.getByLabel("Upload CSV / Excel").check();
  await page.locator('input[type="file"]').setInputFiles({
    name: "experiment.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("date,variant,conversions\n2026-01-01,control,10\n2026-01-01,treatment,12\n"),
  });

  await expect(page.getByText("experiment.csv")).toBeVisible();
  await expect(page.getByText("2 rows")).toBeVisible();

  await page.getByRole("button", { name: /Explore Data/ }).click();

  await expect.poll(() => runRequest).toMatchObject({
    task: "Find conversion patterns in this upload.",
    db_backend: "duckdb",
    analysis_mode: "general",
    duckdb_path: "11111111-1111-4111-8111-111111111111",
  });
  await expect(page.getByText("Analysis Complete")).toBeVisible();
  await expect(page.getByText("The uploaded dataset completed successfully.")).toBeVisible();
});

test("postgres mode blocks submission until required connection fields are present", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("access_token", "access-e2e"));
  await mockAuthenticatedShell(page);

  await page.goto("/");
  await page.getByText("Explore & Understand").click();
  await page.getByRole("textbox").fill("Analyze a warehouse database.");
  await page.getByRole("combobox").selectOption("postgres");

  const submit = page.getByRole("button", { name: /Explore Data/ });
  await expect(submit).toBeDisabled();

  await page.getByPlaceholder("mydb").fill("analytics");
  await page.getByPlaceholder("postgres").fill("readonly_user");
  await expect(submit).toBeEnabled();
});
