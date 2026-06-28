/**
 * useTokenRefresh — proactively refresh the HttpOnly session cookie.
 *
 * Calls POST /auth/refresh every 50 minutes so SSE streams stay authenticated
 * without reading the JWT from JavaScript.
 */
import { useEffect, useRef } from "react";
import client from "../api/client";

const REFRESH_INTERVAL_MS = 50 * 60 * 1000;

export function useTokenRefresh(onRefreshed: () => void) {
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const refresh = async () => {
      try {
        await client.post("/auth/refresh", {});
        onRefreshed();
      } catch {
        window.location.href = "/login";
      }
    };

    timerRef.current = setInterval(refresh, REFRESH_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [onRefreshed]);

  return {};
}
