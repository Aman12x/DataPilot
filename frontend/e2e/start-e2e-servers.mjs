import { spawn } from "node:child_process";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.join(__dirname, "..");
const repoRoot = path.join(frontendRoot, "..");
const pythonBin = process.env.PYTHON ?? "python3";

const children = [];

function spawnLogged(name, command, args, options) {
  const child = spawn(command, args, {
    ...options,
    stdio: ["ignore", "pipe", "pipe"],
  });
  children.push(child);
  child.stdout.on("data", (chunk) => process.stdout.write(`[${name}] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[${name}] ${chunk}`));
  child.on("exit", (code, signal) => {
    if (shuttingDown) return;
    console.error(`[${name}] exited code=${code} signal=${signal}`);
    shutdown(code ?? 1);
  });
  return child;
}

function waitForHealth(url, timeoutMs = 180_000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(url, (res) => {
        res.resume();
        if (res.statusCode && res.statusCode < 500) {
          resolve();
          return;
        }
        retry();
      });
      req.on("error", retry);
      req.setTimeout(2_000, () => {
        req.destroy();
        retry();
      });
    };

    const retry = () => {
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Timed out waiting for ${url}`));
        return;
      }
      setTimeout(tick, 500);
    };

    tick();
  });
}

let shuttingDown = false;
function shutdown(code = 0) {
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) child.kill("SIGTERM");
  }
  setTimeout(() => process.exit(code), 250).unref();
}

process.on("SIGINT", () => shutdown(130));
process.on("SIGTERM", () => shutdown(143));

spawnLogged(
  "backend",
  pythonBin,
  ["-m", "uvicorn", "backend.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
  {
    cwd: repoRoot,
    env: {
      ...process.env,
      ENV: "development",
      SECRET_KEY: process.env.SECRET_KEY ?? "e2e-test-secret-key-for-playwright",
      CORS_ORIGINS: "http://127.0.0.1:5173",
      AUTH_DB_PATH: "/tmp/datapilot-e2e-auth.db",
      MEMORY_DB_PATH: "/tmp/datapilot-e2e-mem.db",
      UPLOAD_DIR: "/tmp/datapilot-e2e-uploads",
      GRAPH_DB_PATH: "/tmp/datapilot-e2e-graph.db",
      MODEL: process.env.E2E_MODEL ?? "claude-haiku-4-5-20251001",
      FAST_MODEL: process.env.FAST_MODEL ?? "claude-haiku-4-5-20251001",
      PYTHONPATH: repoRoot,
    },
  },
);

try {
  await waitForHealth("http://127.0.0.1:8000/health");
  spawnLogged(
    "frontend",
    "npm",
    ["run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
    { cwd: frontendRoot, env: process.env },
  );
} catch (error) {
  console.error(error);
  shutdown(1);
}
