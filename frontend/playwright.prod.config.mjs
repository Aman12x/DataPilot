// @ts-check
// Diagnostic config: drives the DEPLOYED app, so no local webServer is started.
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: /prod-auth\.spec\.ts/,
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL: process.env.PROD_APP_URL || "https://datapilotapp.singhaman.dev",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
