import { test, expect } from "@playwright/test";
import path from "path";

const FIXTURE_CSV = path.join(process.cwd(), "e2e", "fixtures", "saas_churn.csv");

test.describe("Analysis flow", () => {
  test("guest can log in, upload CSV, and start a run", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByRole("button", { name: "Continue as Guest" })).toBeVisible();

    const guestResponsePromise = page.waitForResponse(
      (resp) => resp.url().includes("/auth/guest") && resp.request().method() === "POST",
    );
    await page.getByRole("button", { name: "Continue as Guest" }).click();
    const guestResponse = await guestResponsePromise;
    expect(
      guestResponse.ok(),
      `Guest login failed (${guestResponse.status()}): ${await guestResponse.text()}`,
    ).toBeTruthy();

    await expect(page).toHaveURL("/");

    await page.getByRole("button", { name: /Explore & Understand/i }).click();
    await expect(page.getByRole("heading", { name: /Explore/i })).toBeVisible();

    await page.getByText("Upload CSV / Excel").click();
    await page.getByRole("button", { name: /Choose file/i }).click();
    await page.locator('input[type="file"]').setInputFiles(FIXTURE_CSV);

    await expect(page.getByText(/rows/i)).toBeVisible({ timeout: 30_000 });

    await page.locator("textarea").fill(
      "What factors predict churn in this dataset?"
    );
    await page.getByRole("button", { name: /Explore Data/i }).click();

    // Run started — pipeline progress, a HITL gate, or an analysis error screen
    await expect(
      page.getByText(
        /One quick question|Review the generated SQL|Processing your response|Something went wrong|Connection lost/i
      )
    ).toBeVisible({ timeout: 120_000 });
  });

  test("upload path reaches intent gate when LLM is configured", async ({ page }) => {
    test.skip(
      !process.env.ANTHROPIC_API_KEY,
      "Set ANTHROPIC_API_KEY to run the full intent-gate E2E"
    );

    await page.goto("/login");
    const guestResponsePromise = page.waitForResponse(
      (resp) => resp.url().includes("/auth/guest") && resp.request().method() === "POST",
    );
    await page.getByRole("button", { name: "Continue as Guest" }).click();
    const guestResponse = await guestResponsePromise;
    expect(guestResponse.ok()).toBeTruthy();
    await expect(page).toHaveURL("/");

    await page.getByRole("button", { name: /Explore & Understand/i }).click();

    await page.getByText("Upload CSV / Excel").click();
    await page.locator('input[type="file"]').setInputFiles(FIXTURE_CSV);
    await expect(page.getByText(/rows/i)).toBeVisible({ timeout: 30_000 });

    await page.locator("textarea").fill("analyze the data");
    await page.getByRole("button", { name: /Explore Data/i }).click();

    await expect(page.getByText("One quick question")).toBeVisible({ timeout: 180_000 });
    await expect(page.getByText("DataPilot needs a bit more context")).toBeVisible();
  });
});
