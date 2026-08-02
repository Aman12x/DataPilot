/**
 * serve-dist.mjs — serve the production build with the headers `serve` would.
 *
 * The SPA's CSP is not a static file: `runtime-config.js` writes it into
 * `dist/serve.json` at container start because `connect-src` has to name the API
 * origin, which is only known from VITE_API_URL. Nothing outside production ever
 * applied that policy — the Vite dev server sends no CSP at all, and it injects
 * inline scripts that `script-src 'self'` would rightly block, so the dev server
 * cannot be used to test the production policy under any circumstances.
 *
 * This reads the generated `dist/serve.json` and applies its headers verbatim,
 * so the policy under test is the one the generator actually emits — a
 * mis-generated header fails here rather than in production.
 */
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = process.env.DIST_DIR || path.join(__dirname, "..", "dist");
const PORT = Number(process.env.CSP_SERVE_PORT || 4173);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".png": "image/png",
  ".woff2": "font/woff2",
};

const config = JSON.parse(await readFile(path.join(DIST, "serve.json"), "utf8"));
// One `source: "**"` rule today. Applied to everything rather than re-implementing
// serve's glob matching, which would be a second thing that can disagree.
const headers = config.headers.flatMap((rule) => rule.headers);

createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  let filePath = path.join(DIST, url.pathname);
  let body;
  try {
    body = await readFile(filePath);
  } catch {
    // SPA fallback — client-side routes like /history have no file on disk.
    filePath = path.join(DIST, "index.html");
    body = await readFile(filePath);
  }
  for (const { key, value } of headers) res.setHeader(key, value);
  res.setHeader("Content-Type", TYPES[path.extname(filePath)] || "application/octet-stream");
  res.end(body);
}).listen(PORT, "127.0.0.1", () => {
  console.log(`[serve-dist] ${DIST} on http://127.0.0.1:${PORT} with serve.json headers`);
});
