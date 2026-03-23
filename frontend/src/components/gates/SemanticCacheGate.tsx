import Markdown from "../Markdown";
import { gateCard, gateTitle, gateMessage, gateActions, gateBtnApprove, gateBtnSecondary } from "./shared";

interface Props {
  payload: {
    message: string;
    hit_type: string;
    similarity: number;
    narrative_draft: string;
    recommendation: string;
  };
  onSubmit: (value: object) => void;
  submitting?: boolean;
}

export default function SemanticCacheGate({ payload, onSubmit, submitting }: Props) {
  return (
    <div style={{ ...gateCard, maxWidth: 680 }}>
      <div style={s.cacheTag}>⚡ Similar result found</div>
      <h3 style={gateTitle}>Use a Cached Result?</h3>
      <p style={{ ...gateMessage, marginBottom: 14 }}>{payload.message}</p>

      <div style={s.metaRow}>
        <span style={s.metaBadge}>{(payload.similarity * 100).toFixed(0)}% match</span>
        <span style={s.metaType}>{payload.hit_type === "exact" ? "Exact match" : "Close match"}</span>
      </div>

      {payload.recommendation && (
        <div style={s.recRow}>
          <div style={s.label}>Previous recommendation</div>
          <div style={s.rec}>{payload.recommendation}</div>
        </div>
      )}

      {payload.narrative_draft && (
        <div style={s.previewSection}>
          <div style={s.label}>Report preview</div>
          <div style={s.preview}><Markdown content={payload.narrative_draft} /></div>
        </div>
      )}

      <div style={gateActions}>
        <button style={gateBtnApprove} onClick={() => onSubmit({ approved: true })} disabled={submitting}>
          {submitting ? "Loading…" : "Use cached result"}
        </button>
        <button style={gateBtnSecondary} onClick={() => onSubmit({ approved: false })} disabled={submitting}>
          Run fresh analysis
        </button>
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  cacheTag:       { fontSize: 11, fontWeight: 700, color: "#f9e2af", background: "#f9e2af11", border: "1px solid #f9e2af33", borderRadius: 20, padding: "3px 10px", display: "inline-block", marginBottom: 10 },
  metaRow:        { display: "flex", gap: 10, alignItems: "center", marginBottom: 16 },
  metaBadge:      { background: "#89b4fa22", color: "#89b4fa", fontSize: 12, fontWeight: 700, padding: "4px 10px", borderRadius: 20, border: "1px solid #89b4fa44" },
  metaType:       { color: "#585b70", fontSize: 13 },
  recRow:         { marginBottom: 14 },
  label:          { color: "#585b70", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 6 },
  rec:            { color: "#a6e3a1", fontWeight: 600, fontSize: 14 },
  previewSection: { marginBottom: 18 },
  preview:        { background: "#181825", borderRadius: 8, padding: "14px 18px", maxHeight: 280, overflowY: "auto" as const, border: "1px solid #313244" },
};
