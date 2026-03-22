import { useEffect, useRef, useState } from "react";
import { API_BASE } from "../api/client";
import type { ChartSpec } from "../components/ChartCard";

export interface GateEvent {
  type: "gate";
  gate: string;
  payload: Record<string, unknown>;
}

export interface TrustIndicators {
  n_data_points:     number;
  confidence_level:  "high" | "medium" | "low";
  confidence_reason: string;
}

export interface DoneEvent {
  type: "done";
  state: {
    narrative_draft:  string;
    recommendation:   string;
    run_id:           string;
    charts:           ChartSpec[];
    trust_indicators: TrustIndicators;
    analysis_mode:    string;
  };
}

/**
 * Subscribe to a run's SSE stream.
 *
 * reconnectTrigger: increment this after a successful resume POST to open a
 * fresh EventSource with the current access token.  This handles both the
 * explicit reconnect-after-resume case and token expiry between gates
 * (the new EventSource reads the latest token from localStorage).
 */
export function useSSE(runId: string | null, reconnectTrigger: number = 0) {
  const [gate, setGate] = useState<GateEvent | null>(null);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [error, setError] = useState<string>("");
  // True once a gate event has been received on the *current* connection.
  // Suppresses the onerror that fires when server intentionally closes after a gate.
  const gateReceivedRef = useRef(false);

  // Reset all state whenever the run changes (new run or start over).
  useEffect(() => {
    setGate(null);
    setDone(null);
    setError("");
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    gateReceivedRef.current = false;

    // Always read the latest token so a refreshed token is used on reconnect.
    const token = localStorage.getItem("access_token");
    if (!token) {
      setError("Not authenticated. Please log in.");
      return;
    }
    const es = new EventSource(`${API_BASE}/runs/${runId}/stream?token=${encodeURIComponent(token)}`);

    es.onmessage = (e) => {
      let msg: unknown;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return; // malformed frame — ignore and keep connection open
      }
      if (typeof msg !== "object" || msg === null) return;
      const m = msg as Record<string, unknown>;
      if (m.type === "gate") {
        gateReceivedRef.current = true;
        setGate(m as unknown as GateEvent);
      }
      if (m.type === "done") {
        setDone(m as unknown as DoneEvent);
        es.close();
      }
      if (m.type === "error") {
        setError(m.message as string);
        es.close();
      }
    };

    es.onerror = () => {
      // Server closes the SSE intentionally after sending a gate event.
      // EventSource will auto-reconnect; don't surface this as an error.
      if (gateReceivedRef.current) return;
      setError("Connection lost. Please refresh.");
      es.close();
    };

    return () => es.close();
  }, [runId, reconnectTrigger]);

  return { gate, done, error, setGate, setDone, setError };
}
