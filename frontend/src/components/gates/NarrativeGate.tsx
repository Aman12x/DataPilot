import { useState } from "react";
import Markdown from "../Markdown";
import { gateCard, gateTitle, gateMessage, gateTextarea, gateActions, gateBtnApprove } from "./shared";

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
    <div style={{ ...gateCard, maxWidth: 720 }}>
      <h3 style={gateTitle}>Review Your Report</h3>
      <p style={{ ...gateMessage, marginBottom: 16 }}>{payload.message}</p>

      <div style={s.section}>
        <div style={s.label}>Recommendation</div>
        <div style={s.rec}>{payload.recommendation}</div>
      </div>

      <div style={s.section}>
        <div style={s.label}>Full Report</div>
        <div style={s.narrativeWrapper}><Markdown content={payload.narrative_draft} /></div>
      </div>

      <input
        style={s.input}
        value={recOverride}
        onChange={(e) => setRecOverride(e.target.value)}
        placeholder="Override recommendation (optional)…"
        disabled={submitting}
      />

      <textarea
        style={gateTextarea}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Notes for revision (leave blank to approve as-is)…"
        rows={3}
        disabled={submitting}
      />

      <div style={gateActions}>
        <button
          style={gateBtnApprove}
          onClick={() => onSubmit({ approved: true, notes, recommendation_override: recOverride })}
          disabled={submitting}
        >
          {submitting ? "Submitting…" : "Approve & Finish"}
        </button>
        <button
          style={s.btnRevise}
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

const s: Record<string, React.CSSProperties> = {
  section:          { marginBottom: 18 },
  label:            { color: "#585b70", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 6 },
  rec:              { color: "#a6e3a1", fontWeight: 600, fontSize: 15, lineHeight: 1.5 },
  narrativeWrapper: { background: "#181825", borderRadius: 8, padding: "16px 20px", maxHeight: 340, overflowY: "auto" as const, border: "1px solid #313244" },
  input:            { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "9px 12px", fontSize: 13, marginBottom: 10, boxSizing: "border-box" as const },
  btnRevise:        { padding: "10px 22px", background: "transparent", color: "#cba6f7", border: "1px solid #cba6f744", borderRadius: 8, cursor: "pointer", fontSize: 14 },
};
