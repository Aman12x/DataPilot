/** Extract a human-readable error message from an Axios error response. */
export function extractApiError(err: unknown, fallback: string): string {
  const ax = err as {
    message?: string;
    code?: string;
    response?: { status?: number; data?: { detail?: unknown } };
  };

  const detail = ax?.response?.data?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((item) => (typeof item === "object" && item && "msg" in item ? String((item as { msg: unknown }).msg) : ""))
      .filter(Boolean);
    if (msgs.length) return msgs.join(". ");
  }

  if (!ax?.response) {
    if (ax?.code === "ERR_NETWORK" || ax?.message === "Network Error") {
      return "Could not reach the server. Check VITE_API_URL on the frontend and CORS_ORIGINS on the backend.";
    }
    if (ax?.message) return ax.message;
  }

  const status = ax?.response?.status;
  if (status === 429) return "Too many attempts. Please wait a minute and try again.";
  if (status && status >= 500) return "Server error. Check backend logs on Railway.";

  return fallback;
}
