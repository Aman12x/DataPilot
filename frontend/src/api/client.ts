import axios from "axios";

// In dev: Vite proxies /api → localhost:8000 (see vite.config.ts).
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

const client = axios.create({ baseURL: API_BASE });

// Attach access token to every request.
client.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// On 401: attempt refresh once, then redirect to login.
client.interceptors.response.use(
  (res) => res,
  async (err) => {
    const original = err.config;
    if (err.response?.status === 401 && !original._retry) {
      original._retry = true;
      try {
        const refresh_token = localStorage.getItem("refresh_token");
        if (!refresh_token) throw new Error("No refresh token");
        const { data } = await axios.post(`${API_BASE}/auth/refresh`, { refresh_token });
        localStorage.setItem("access_token", data.access_token);
        original.headers.Authorization = `Bearer ${data.access_token}`;
        return client(original);
      } catch {
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
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
  const refresh_token = localStorage.getItem("refresh_token");
  if (refresh_token) {
    try { await client.post("/auth/logout", { refresh_token }); } catch { /* ignore */ }
  }
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await client.post<UploadResult>("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}
