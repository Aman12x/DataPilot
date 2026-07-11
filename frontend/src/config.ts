declare global {
  interface Window {
    __DP_API__?: string;
  }
}

/** Resolve API base URL: runtime config (Railway) → build-time Vite env → dev proxy. */
export function resolveApiBase(): string {
  const runtime = window.__DP_API__?.trim();
  if (runtime) return runtime.replace(/\/$/, "");

  const built = import.meta.env.VITE_API_URL?.trim();
  if (built) return built.replace(/\/$/, "");

  return "/api";
}

export const API_BASE = resolveApiBase();
