// @ts-check
/**
 * Runs e2e/csp-*.spec.ts against the *production build* served with the CSP that
 * `runtime-config.js` generates.
 *
 * Separate from the default config because the default one starts the Vite dev
 * server, which sends no CSP and injects inline scripts that the production
 * `script-src 'self'` would block. Testing the real policy needs the real build.
 *
 *   cd frontend && npx playwright test --config=playwright.csp.config.mjs
 *
 * No backend runs: the specs stub every API call with page.route, so the page,
 * its bundle, and its headers are real while the data is not. That is the right
 * split for a CSP test — the policy governs what the *page* is allowed to do.
 */
import { defineConfig, devices } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// A cross-origin API base, matching the Railway deploy shape: connect-src has to
// name it explicitly, and a same-origin value would hide that whole class of bug.
const API_URL = "http://127.0.0.1:8899";
const PORT = 4173;

export default defineConfig({
  testDir: "./e2e",
  testMatch: /csp-.*\.spec\.ts/,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  use: {
    ...devices["Desktop Chrome"],
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "on-first-retry",
  },
  metadata: { apiUrl: API_URL },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
  webServer: {
    command:
      `npm run build && ` +
      `VITE_API_URL=${API_URL} DIST_DIR=${path.join(__dirname, "dist")} node runtime-config.js && ` +
      `CSP_SERVE_PORT=${PORT} node ${path.join(__dirname, "e2e", "serve-dist.mjs")}`,
    url: `http://127.0.0.1:${PORT}/index.html`,
    cwd: __dirname,
    timeout: 180_000,
    reuseExistingServer: false,
  },
});
