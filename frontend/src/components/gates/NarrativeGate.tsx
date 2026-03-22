import { useState } from "react";
import Markdown from "../Markdown";

interface Props {
  payload: {
    narrative_draft: string;
    recommendation: string;
    message: string;
  };
  onSubmit: (value: object) => void;
  submitting?: boolean;
}

export default function NarrativeGate({ payload, onSubmit, submitting }: Props) {
  const [notes, setNotes]             = useState("");
  const [recOverride, setRecOverride] = useState("");

  return (
    <div style={styles.card}>
      <h3 style={styles.title}>Review Your Report</h3>
      <p style={styles.message}>{payload.message}</p>

      <div style={styles.section}>
        <div style={styles.label}>Recommendation</div>
        <div style={styles.rec}>{payload.recommendation}</div>
      </div>

      <div style={styles.section}>
        <div style={styles.label}>Full Report</div>
        <div style={styles.narrativeWrapper}>
          <Markdown content={payload.narrative_draft} />
        </div>
      </div>

      <input
        style={styles.input}
        value={recOverride}
        onChange={(e) => setRecOverride(e.target.value)}
        placeholder="Override recommendation (optional)…"
        disabled={submitting}
      />

      <textarea
        style={styles.textarea}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Notes for revision (leave blank to approve as-is)…"
        rows={3}
        disabled={submitting}
      />

      <div style={styles.actions}>
        <button
          style={styles.btnApprove}
          onClick={() => onSubmit({ approved: true, notes, recommendation_override: recOverride })}
          disabled={submitting}
        >
          {submitting ? "Submitting…" : "Approve & Finish"}
        </button>
        <button
          style={styles.btnRevise}
          onClick={() => onSubmit({ approved: false, notes, recommendation_override: recOverride })}
          disabled={submitting}
          title="Ask DataPilot to revise the report based on your notes"
        >
          Request Changes
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card:             { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "28px 32px", maxWidth: 720, margin: "0 auto", boxShadow: "0 8px 40px #00000044" },
  title:            { color: "#cdd6f4", marginTop: 0, fontSize: 18, fontWeight: 700 },
  message:          { color: "#a6adc8", marginBottom: 16, fontSize: 14 },
  section:          { marginBottom: 18 },
  label:            { color: "#585b70", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 6 },
  rec:              { color: "#a6e3a1", fontWeight: 600, fontSize: 15, lineHeight: 1.5 },
  narrativeWrapper: { background: "#181825", borderRadius: 8, padding: "16px 20px", maxHeight: 340, overflowY: "auto" as const, border: "1px solid #313244" },
  input:            { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "9px 12px", fontSize: 13, marginBottom: 10, boxSizing: "border-box" as const },
  textarea:         { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "9px 12px", fontSize: 13, resize: "vertical" as const, boxSizing: "border-box" as const },
  actions:          { display: "flex", gap: 12, marginTop: 18 },
  btnApprove:       { padding: "10px 22px", background: "#a6e3a1", color: "#1e1e2e", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 14 },
  btnRevise:        { padding: "10px 22px", background: "transparent", color: "#cba6f7", border: "1px solid #cba6f744", borderRadius: 8, cursor: "pointer", fontSize: 14 },
};
