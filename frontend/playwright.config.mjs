// @ts-check
import { defineConfig, devices } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: "./e2e",
  // Convention: e2e/prod-*.spec.ts drives the DEPLOYED app and talks to the
  // live API, so those specs cannot run against the local stack this config
  // starts. They have their own config (playwright.prod.config.mjs). Without
  // this exclusion CI collected prod-auth.spec.ts here and failed on a session
  // cookie that was never set locally.
  testIgnore: /prod-.*\.spec\.ts/,
  timeout: 180_000,
  expect: { timeout: 120_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
  webServer: {
    // Same-origin via Vite proxy — HttpOnly auth cookies work without CORS.
    // The launcher waits for backend /health before starting Vite.
    command: `node ${path.join(__dirname, "e2e", "start-e2e-servers.mjs")}`,
    url: "http://127.0.0.1:5173",
    timeout: 180_000,
    reuseExistingServer: false,
  },
});
