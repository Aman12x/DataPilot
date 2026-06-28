import axios from "axios";

// In dev: Vite proxies /api → 127.0.0.1:8000 (see vite.config.ts).
// In prod: VITE_API_URL must be set at build time to the backend's public URL,
//          e.g. https://datapilot-backend.up.railway.app
if (import.meta.env.PROD && !import.meta.env.VITE_API_URL) {
  throw new Error(
    "[DataPilot] VITE_API_URL is not set. " +
    "Add it as a Railway environment variable on the frontend service " +
    "and redeploy so it is baked into the production bundle."
  );
}

export const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

const client = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
});

// On 401: attempt cookie-based refresh once, then redirect to login.
client.interceptors.response.use(
  (res) => res,
  async (err) => {
    const original = err.config;
    if (err.response?.status === 401 && !original._retry) {
      original._retry = true;
      if (original.url?.includes("/auth/refresh") || original.url?.includes("/auth/login")) {
        return Promise.reject(err);
      }
      try {
        await axios.post(`${API_BASE}/auth/refresh`, {}, { withCredentials: true });
        return client(original);
      } catch {
        window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

export default client;

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
