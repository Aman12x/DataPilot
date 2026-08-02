/**
 * Writes the two runtime-generated files into the built SPA before `serve`
 * starts:
 *
 *   dist/config.js   — window.__DP_API__, the API base URL
 *   dist/serve.json  — security headers, including a CSP whose connect-src
 *                      must name the API origin
 *
 * The CSP cannot be a static file: the backend lives on a different origin that
 * is only known from VITE_API_URL at container start.
 *
 * Set CSP_REPORT_ONLY=true to emit Content-Security-Policy-Report-Only instead,
 * which reports violations without blocking — use it for one deploy to confirm
 * nothing legitimate trips before enforcing.
 */
const fs = require("fs");
const path = require("path");

const DIST = process.env.DIST_DIR || "/app/dist";
const apiUrl = process.env.VITE_API_URL || "";

function apiOrigin(raw) {
  if (!raw) return "";
  try {
    return new URL(raw).origin;
  } catch {
    // Relative base (same-origin deployment) — 'self' already covers it.
    return "";
  }
}

function buildCsp(origin) {
  const connect = ["'self'", origin].filter(Boolean).join(" ");
  return [
    "default-src 'self'",
    // The build emits external modules only; no inline <script> to allow.
    "script-src 'self'",
    // No 'unsafe-inline'. The assumption that React style props need it is
    // wrong: React and Recharts write through the CSSOM (node.style.foo = …),
    // which CSP does not police. Only literal style= attributes in parsed HTML
    // and <style> blocks are, and the build emits neither — verified by
    // e2e/csp-sweep.spec.ts across every screen, gate, and modal.
    "style-src 'self' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    // blob:/data: cover chart rendering and the PDF download.
    "img-src 'self' data: blob:",
    `connect-src ${connect}`,
    "object-src 'none'",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join("; ");
}

function main() {
  const origin = apiOrigin(apiUrl);
  const reportOnly = String(process.env.CSP_REPORT_ONLY || "").toLowerCase() === "true";
  const cspHeader = reportOnly
    ? "Content-Security-Policy-Report-Only"
    : "Content-Security-Policy";

  fs.writeFileSync(
    path.join(DIST, "config.js"),
    "window.__DP_API__=" + JSON.stringify(apiUrl) + ";",
  );

  const serveConfig = {
    headers: [
      {
        source: "**",
        headers: [
          { key: cspHeader, value: buildCsp(origin) },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
        ],
      },
    ],
  };

  fs.writeFileSync(
    path.join(DIST, "serve.json"),
    JSON.stringify(serveConfig, null, 2),
  );

  if (!origin && apiUrl) {
    console.warn(
      `[runtime-config] VITE_API_URL=${apiUrl} is not an absolute URL; ` +
        "connect-src falls back to 'self'.",
    );
  }
  console.log(
    `[runtime-config] api=${apiUrl || "(unset)"} csp=${reportOnly ? "report-only" : "enforced"}`,
  );
}

if (require.main === module) main();

module.exports = { apiOrigin, buildCsp };
