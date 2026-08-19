import { useEffect, useRef, useState } from "react";
import client, { API_BASE } from "../api/client";
import type { ChartSpec } from "../components/ChartCard";

export interface StepEvent {
  type: "step";
  node: string;
  label: string;
  detail?: string;
  status: "completed";
}

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

export interface SensitivityRow {
  mde_pct:      number;
  n_per_arm:    number;
  runtime_days: number;
}

export interface PowerAnalysisResult {
  baseline_mean:       number;
  baseline_std:        number;
  daily_traffic:       number;
  mde_target_pct:      number;
  mde_target_abs:      number;
  required_n_per_arm:  number;
  required_total_n:    number;
  runtime_days:        number;
  alpha:               number;
  power:               number;
  guardrails_to_watch: string[];
  sensitivity:         SensitivityRow[];
}

export interface DeckData {
  verdict:        "positive" | "neutral" | "negative";
  headline:       string;
  hero_metric:    string;
  confidence:     string;
  recommendation: string;
  evidence:       string[];
  watch_out:      string;
}

export interface DoneEvent {
  type: "done";
  state: {
    narrative_draft:      string;
    recommendation:       string;
    run_id:               string;
    charts:               ChartSpec[];
    trust_indicators:     TrustIndicators;
    analysis_mode:        string;
    power_analysis_result?: PowerAnalysisResult;
    deck_data?:           DeckData;
  };
}

/**
 * Subscribe to a run's SSE stream.
 *
 * reconnectTrigger: increment this after a successful resume POST to open a
 * fresh EventSource with a new short-lived stream token.
 */
export function useSSE(runId: string | null, reconnectTrigger: number = 0) {
  const [gate,  setGate]  = useState<GateEvent | null>(null);
  const [done,  setDone]  = useState<DoneEvent | null>(null);
  const [error, setError] = useState<string>("");
  const [steps, setSteps] = useState<StepEvent[]>([]);
  // Live narrative draft, streamed token-batch by token-batch while
  // generate_narrative runs. It is cleared when a gate arrives (the gate
  // shows the finished draft from its payload) so that the revision after a
  // narrative-gate decline starts from empty. We cannot rely on the server's
  // narrative_start for that: the resumed worker emits it within ms of the
  // resume POST, before the browser has minted a stream token and opened the
  // new EventSource, and a reconnect reads the Redis stream from `$` — so in
  // production the reset was lost and revision deltas appended to the old
  // draft under "Writing narrative…". narrative_start is kept as a
  // belt-and-braces reset for the in-memory (replayed) path.
  const [narrativeDraft, setNarrativeDraft] = useState<string>("");
  const gateReceivedRef = useRef(false);
  // Last SSE id seen (the server stamps every event with its Redis stream
  // id). Passed as ?last_id= when we open a fresh EventSource on resume or
  // token refresh, so the server replays what was published during the
  // reconnect window instead of starting from "$" and dropping it.
  const lastEventIdRef = useRef<string>("");

  useEffect(() => {
    setGate(null);
    setDone(null);
    setError("");
    setSteps([]);
    setNarrativeDraft("");
    lastEventIdRef.current = "";
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    gateReceivedRef.current = false;

    let es: EventSource | null = null;
    let cancelled = false;

    (async () => {
      try {
        const { data } = await client.get<{ stream_token: string }>(
          `/runs/${runId}/stream-token`
        );
        if (cancelled) return;

        const resume = lastEventIdRef.current && lastEventIdRef.current !== "$"
          ? `&last_id=${encodeURIComponent(lastEventIdRef.current)}`
          : "";
        es = new EventSource(
          `${API_BASE}/runs/${runId}/stream?stream_token=${encodeURIComponent(data.stream_token)}${resume}`
        );

        es.onmessage = (e) => {
          if (e.lastEventId) lastEventIdRef.current = e.lastEventId;
          let msg: unknown;
          try {
            msg = JSON.parse(e.data);
          } catch {
            return;
          }
          if (typeof msg !== "object" || msg === null) return;
          const m = msg as Record<string, unknown>;
          if (m.type === "step") {
            setSteps(prev => [...prev, m as unknown as StepEvent]);
          }
          if (m.type === "narrative_start") {
            setNarrativeDraft("");
          }
          if (m.type === "narrative_delta" && typeof m.text === "string") {
            setNarrativeDraft(prev => prev + m.text);
          }
          if (m.type === "gate") {
            gateReceivedRef.current = true;
            setNarrativeDraft("");
            setGate(m as unknown as GateEvent);
          }
          if (m.type === "done") {
            setDone(m as unknown as DoneEvent);
            es?.close();
          }
          if (m.type === "error") {
            setError(m.message as string);
            es?.close();
          }
        };

        es.onerror = () => {
          if (gateReceivedRef.current) return;
          setError("Connection lost. Please refresh.");
          es?.close();
        };
      } catch {
        if (!cancelled) setError("Could not connect to analysis stream.");
      }
    })();

    return () => {
      cancelled = true;
      es?.close();
    };
  }, [runId, reconnectTrigger]);

  return { gate, done, error, steps, narrativeDraft, setGate, setDone, setError };
}
