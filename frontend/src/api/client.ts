import axios from "axios";
import { API_BASE } from "../config";

const client = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
});

// On 401: attempt cookie-based refresh once, then redirect to login.
// Auth probe routes (/auth/me, etc.) must reject quietly — no refresh or redirect.
const AUTH_PROBE_PATHS = ["/auth/me", "/auth/refresh", "/auth/login", "/auth/guest", "/auth/register"];

client.interceptors.response.use(
  (res) => res,
  async (err) => {
    const original = err.config;
    if (err.response?.status === 401 && !original._retry) {
      original._retry = true;
      if (AUTH_PROBE_PATHS.some((p) => original.url?.includes(p))) {
        return Promise.reject(err);
      }
      try {
        await axios.post(`${API_BASE}/auth/refresh`, {}, { withCredentials: true });
        return client(original);
      } catch {
        if (!window.location.pathname.startsWith("/login")) {
          window.location.href = "/login";
        }
        return Promise.reject(err);
      }
    }
    return Promise.reject(err);
  }
);

export default client;
export { API_BASE };

export interface UploadResult {
  upload_id: string;
  columns:   string[];
  row_count: number;
  preview:   Record<string, unknown>[];
}

export async function logout(): Promise<void> {
  try { await client.post("/auth/logout", {}); } catch { /* ignore */ }
}

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await client.post<UploadResult>("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function checkAuth(): Promise<boolean> {
  try {
    await client.get("/auth/me");
    return true;
  } catch {
    return false;
  }
}

/** Lightweight connectivity check — no auth required. */
export async function pingBackend(): Promise<{ ok: boolean; detail: string }> {
  try {
    const r = await axios.get(`${API_BASE}/health`, { timeout: 10_000 });
    return { ok: r.status === 200, detail: "Backend is reachable." };
  } catch {
    return {
      ok: false,
      detail: `Cannot reach backend at ${API_BASE}. Set VITE_API_URL on the frontend service `
        + "and CORS_ORIGINS on the backend, then redeploy both.",
    };
  }
}
