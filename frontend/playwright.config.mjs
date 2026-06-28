// @ts-check
import { defineConfig, devices } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");

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
  webServer: [
    {
      command:
        `cd ${repoRoot} && SECRET_KEY=e2e-test-secret-key-for-playwright ` +
        `AUTH_DB_PATH=/tmp/datapilot-e2e-auth.db ` +
        `MEMORY_DB_PATH=/tmp/datapilot-e2e-mem.db ` +
        `UPLOAD_DIR=/tmp/datapilot-e2e-uploads ` +
        `GRAPH_DB_PATH=/tmp/datapilot-e2e-graph.db ` +
        `PYTHONPATH=. python3 -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`,
      url: "http://127.0.0.1:8000/health",
      timeout: 180_000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5173",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
    },
  ],
});
