// @ts-check
import { defineConfig, devices } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: "./e2e",
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
