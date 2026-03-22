import { useState } from "react";

interface Props {
  payload: {
    message: string;
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
  const color = tone === "good" ? "#a6e3a1" : tone === "warn" ? "#f38ba8" : "#cdd6f4";
  const icon  = tone === "good" ? "✓" : tone === "warn" ? "⚠" : "·";
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 8 }}>
      <span style={{ color, fontSize: 13, marginTop: 1, flexShrink: 0 }}>{icon}</span>
      <div>
        <span style={{ color: "#585b70", fontSize: 12 }}>{label} </span>
        <span style={{ color, fontWeight: 600, fontSize: 13 }}>{value}</span>
      </div>
    </div>
  );
}

export default function AnalysisGate({ payload, onSubmit, submitting }: Props) {
  const [notes, setNotes] = useState("");

  const sig   = payload.significant;
  const novel = payload.novelty_likely;
  const guard = payload.guardrails_breached;
  const mde   = payload.mde_powered;

  return (
    <div style={styles.card}>
      <h3 style={styles.title}>Review Analysis Results</h3>
      <p style={styles.message}>{payload.message}</p>

      <div style={styles.stats}>
        <Stat
          label="Effect:"
          value={sig == null ? null : sig ? "Statistically significant" : "Not statistically significant"}
          tone={sig == null ? "neutral" : sig ? "good" : "warn"}
        />
        <Stat
          label="Top segment:"
          value={payload.top_segment}
          tone="neutral"
        />
        <Stat
          label="Novelty check:"
          value={novel == null ? null : novel ? "Novelty effect likely — treat results cautiously" : "Novelty ruled out"}
          tone={novel == null ? "neutral" : novel ? "warn" : "good"}
        />
        <Stat
          label="Guardrails:"
          value={guard == null ? null : guard ? "One or more secondary metrics breached" : "No guardrails breached"}
          tone={guard == null ? "neutral" : guard ? "warn" : "good"}
        />
        <Stat
          label="Sample power:"
          value={mde == null ? null : mde ? "Adequately powered for observed effect" : "Underpowered — interpret with caution"}
          tone={mde == null ? "neutral" : mde ? "good" : "warn"}
        />
        <Stat
          label="Business impact:"
          value={payload.business_impact}
          tone="neutral"
        />
        <Stat
          label="Variance reduction (CUPED):"
          value={payload.cuped_variance_reduction != null ? `${payload.cuped_variance_reduction.toFixed(1)}%` : null}
          tone="neutral"
        />
        <Stat
          label="Biggest funnel drop:"
          value={payload.biggest_funnel_dropoff}
          tone={payload.biggest_funnel_dropoff ? "warn" : "neutral"}
        />
      </div>

      {payload.breached_metrics?.length > 0 && (
        <div style={styles.breachedBox}>
          <div style={styles.breachedLabel}>Breached guardrails</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" as const, marginTop: 6 }}>
            {payload.breached_metrics.map((m, i) => (
              <span key={i} style={styles.badge}>{m.metric} ({m.direction})</span>
            ))}
          </div>
        </div>
      )}

      <textarea
        style={styles.textarea}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Optional: add notes or flag anything you want changed in the report…"
        rows={3}
        disabled={submitting}
      />

      <div style={styles.actions}>
        <button style={styles.btnApprove} onClick={() => onSubmit({ approved: true, notes })} disabled={submitting}>
          {submitting ? "Submitting…" : "Approve & Continue"}
        </button>
        <button
          style={styles.btnReject}
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

const styles: Record<string, React.CSSProperties> = {
  card:         { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "28px 32px", maxWidth: 640, margin: "0 auto", boxShadow: "0 8px 40px #00000044" },
  title:        { color: "#cdd6f4", marginTop: 0, fontSize: 18, fontWeight: 700 },
  message:      { color: "#a6adc8", marginBottom: 16, fontSize: 14 },
  stats:        { background: "#181825", borderRadius: 8, padding: "14px 18px", marginBottom: 16, border: "1px solid #313244" },
  breachedBox:  { background: "#f38ba811", border: "1px solid #f38ba833", borderRadius: 8, padding: "12px 16px", marginBottom: 14 },
  breachedLabel:{ color: "#f38ba8", fontSize: 12, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.06em" },
  badge:        { background: "#f38ba822", color: "#f38ba8", borderRadius: 6, padding: "3px 10px", fontSize: 12, fontWeight: 500 },
  textarea:     { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "9px 12px", fontSize: 13, resize: "vertical" as const, boxSizing: "border-box" as const },
  actions:      { display: "flex", gap: 12, marginTop: 18 },
  btnApprove:   { padding: "10px 22px", background: "#a6e3a1", color: "#1e1e2e", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 14 },
  btnReject:    { padding: "10px 22px", background: "transparent", color: "#f38ba8", border: "1px solid #f38ba844", borderRadius: 8, cursor: "pointer", fontSize: 14 },
};
