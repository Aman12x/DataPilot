/**
 * useTokenRefresh — schedules a proactive access-token refresh.
 *
 * Decodes the JWT exp claim (no secret needed — we're just reading the
 * payload, not verifying it). Schedules a refresh 5 minutes before expiry
 * so EventSource connections are never open with an about-to-expire token.
 *
 * When the refresh succeeds, `onRefreshed` is called so callers can
 * increment a reconnect trigger and reopen any EventSource connections.
 */
import { useEffect, useRef } from "react";
import axios from "axios";
import { API_BASE } from "../api/client";

function _tokenExpiresAt(token: string): number | null {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return typeof payload.exp === "number" ? payload.exp * 1000 : null;
  } catch {
    return null;
  }
}

export function useTokenRefresh(onRefreshed: () => void) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const schedule = () => {
    if (timerRef.current) clearTimeout(timerRef.current);

    const token = localStorage.getItem("access_token");
    if (!token) return;

    const expiresAt = _tokenExpiresAt(token);
    if (!expiresAt) return;

    // Refresh 5 minutes before expiry (but not if already expired)
    const msUntilRefresh = expiresAt - Date.now() - 5 * 60 * 1000;
    if (msUntilRefresh <= 0) {
      // Token already expired or expiring imminently — refresh now
      _doRefresh(onRefreshed, schedule);
      return;
    }

    timerRef.current = setTimeout(() => {
      _doRefresh(onRefreshed, schedule);
    }, msUntilRefresh);
  };

  useEffect(() => {
    schedule();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { reschedule: schedule };
}

async function _doRefresh(onRefreshed: () => void, reschedule: () => void) {
  const refresh_token = localStorage.getItem("refresh_token");
  if (!refresh_token) return;
  try {
    const { data } = await axios.post(`${API_BASE}/auth/refresh`, { refresh_token });
    localStorage.setItem("access_token", data.access_token);
    onRefreshed();   // caller reopens EventSource with fresh token
    reschedule();    // schedule the next refresh
  } catch {
    // Refresh failed — redirect to login
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    window.location.href = "/login";
  }
}
