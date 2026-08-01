import { useState } from "react";
import { gateCard, gateTitle, gateMessage, gateTextarea, gateActions, gateBtnApprove } from "./shared";

interface Props {
  payload: {
    message: string;
    srm_detected?: boolean;
    significant: boolean | null;
    top_segment: string | null;
    novelty_likely: boolean | null;
    guardrails_breached: boolean | null;
    breached_metrics: Array<{ metric: string; direction: string }>;
    mde_powered: boolean | null;
    business_impact: string | null;
    cuped_variance_reduction: number | null;
    biggest_funnel_dropoff: string | null;
  };
  onSubmit: (value: object) => void;
  submitting?: boolean;
}

type Tone = "good" | "warn" | "neutral";

function Stat({ label, value, tone }: { label: string; value: string | null; tone?: Tone }) {
  if (value == null) return null;
  const color = tone === "good" ? "var(--dp-success)" : tone === "warn" ? "var(--dp-danger)" : "var(--dp-ink)";
  const icon  = tone === "good" ? "✓" : tone === "warn" ? "⚠" : "·";
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 8 }}>
      <span style={{ color, fontSize: 13, marginTop: 1, flexShrink: 0 }}>{icon}</span>
      <div>
        <span style={{ color: "var(--dp-ink-muted)", fontSize: 12 }}>{label} </span>
        <span style={{ color, fontWeight: 600, fontSize: 13 }}>{value}</span>
      </div>
    </div>
  );
}

export default function AnalysisGate({ payload, onSubmit, submitting }: Props) {
  const [notes, setNotes] = useState("");
  const [srmAcked, setSrmAcked] = useState(false);
  const { significant: sig, novelty_likely: novel, guardrails_breached: guard, mde_powered: mde } = payload;
  const srmDetected = !!payload.srm_detected;
  const canApprove  = !srmDetected || srmAcked;

  return (
    <div style={{ ...gateCard, maxWidth: 640 }}>
      <h3 style={gateTitle}>Review Analysis Results</h3>
      <p style={{ ...gateMessage, marginBottom: 16 }}>{payload.message}</p>

      <div style={s.stats}>
        <Stat label="Effect:"          value={sig == null ? null : sig ? "Statistically significant" : "Not statistically significant"} tone={sig == null ? "neutral" : sig ? "good" : "warn"} />
        <Stat label="Top segment:"     value={payload.top_segment} tone="neutral" />
        <Stat label="Novelty check:"   value={novel == null ? null : novel ? "Novelty effect likely — treat results cautiously" : "Novelty ruled out"} tone={novel == null ? "neutral" : novel ? "warn" : "good"} />
        <Stat label="Guardrails:"      value={guard == null ? null : guard ? "One or more secondary metrics breached" : "No guardrails breached"} tone={guard == null ? "neutral" : guard ? "warn" : "good"} />
        <Stat label="Sample power:"    value={mde == null ? null : mde ? "Adequately powered for observed effect" : "Underpowered — interpret with caution"} tone={mde == null ? "neutral" : mde ? "good" : "warn"} />
        <Stat label="Business impact:" value={payload.business_impact} tone="neutral" />
        <Stat label="Variance reduction (CUPED):" value={payload.cuped_variance_reduction != null ? `${payload.cuped_variance_reduction.toFixed(1)}%` : null} tone="neutral" />
        <Stat label="Biggest funnel drop:" value={payload.biggest_funnel_dropoff} tone={payload.biggest_funnel_dropoff ? "warn" : "neutral"} />
      </div>

      {payload.breached_metrics?.length > 0 && (
        <div style={s.breachedBox}>
          <div style={s.breachedLabel}>Breached guardrails</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" as const, marginTop: 6 }}>
            {payload.breached_metrics.map((m, i) => (
              <span key={i} style={s.badge}>{m.metric} ({m.direction})</span>
            ))}
          </div>
        </div>
      )}

      {srmDetected && (
        <label style={s.srmAck} className="fade-in">
          <input
            type="checkbox"
            checked={srmAcked}
            onChange={(e) => setSrmAcked(e.target.checked)}
            disabled={submitting}
            style={{ accentColor: "var(--dp-danger)", width: 15, height: 15, flexShrink: 0 }}
          />
          <span style={{ color: "var(--dp-danger)", fontSize: 13, lineHeight: 1.4 }}>
            I understand the statistical results are unreliable due to sample ratio mismatch
            and wish to proceed anyway.
          </span>
        </label>
      )}

      <textarea
        style={gateTextarea}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Optional: add notes or flag anything you want changed in the report…"
        rows={3}
        disabled={submitting}
      />

      <div style={gateActions}>
        <button
          style={{ ...gateBtnApprove, opacity: canApprove ? 1 : 0.45 }}
          onClick={() => onSubmit({ approved: true, notes, ...(srmDetected ? { srm_acknowledged: srmAcked } : {}) })}
          disabled={submitting || !canApprove}
          title={!canApprove ? "Acknowledge the SRM warning above to proceed" : undefined}
        >
          {submitting ? "Submitting…" : "Approve & Continue"}
        </button>
        <button
          style={s.btnReject}
          onClick={() => onSubmit({ approved: false, notes })}
          disabled={submitting}
          title="Flag concerns — the report will note these issues"
        >
          Request Changes
        </button>
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  stats:        { background: "var(--dp-surface-2)", borderRadius: 8, padding: "14px 18px", marginBottom: 16, border: "1px solid var(--dp-line)" },
  breachedBox:  { background: "var(--dp-danger)11", border: "1px solid var(--dp-danger)33", borderRadius: 8, padding: "12px 16px", marginBottom: 14 },
  breachedLabel:{ color: "var(--dp-danger)", fontSize: 12, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.06em" },
  badge:        { background: "var(--dp-danger)22", color: "var(--dp-danger)", borderRadius: 6, padding: "3px 10px", fontSize: 12, fontWeight: 500 },
  btnReject:    { padding: "10px 22px", background: "transparent", color: "var(--dp-danger)", border: "1px solid var(--dp-danger)44", borderRadius: 8, cursor: "pointer", fontSize: 14 },
  srmAck:       { display: "flex", alignItems: "flex-start", gap: 10, background: "var(--dp-danger)11", border: "1px solid var(--dp-danger)44", borderRadius: 8, padding: "12px 14px", marginBottom: 12, cursor: "pointer" },
};
