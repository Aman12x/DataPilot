import Markdown from "../Markdown";

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
    <div style={styles.card}>
      <div style={styles.cacheTag}>⚡ Similar result found</div>
      <h3 style={styles.title}>Use a Cached Result?</h3>
      <p style={styles.message}>{payload.message}</p>

      <div style={styles.metaRow}>
        <span style={styles.metaBadge}>{(payload.similarity * 100).toFixed(0)}% match</span>
        <span style={styles.metaType}>{payload.hit_type === "exact" ? "Exact match" : "Close match"}</span>
      </div>

      {payload.recommendation && (
        <div style={styles.recRow}>
          <div style={styles.label}>Previous recommendation</div>
          <div style={styles.rec}>{payload.recommendation}</div>
        </div>
      )}

      {payload.narrative_draft && (
        <div style={styles.previewSection}>
          <div style={styles.label}>Report preview</div>
          <div style={styles.preview}>
            <Markdown content={payload.narrative_draft} />
          </div>
        </div>
      )}

      <div style={styles.actions}>
        <button style={styles.btnPrimary} onClick={() => onSubmit({ approved: true })} disabled={submitting}>
          {submitting ? "Loading…" : "Use cached result"}
        </button>
        <button style={styles.btnSecondary} onClick={() => onSubmit({ approved: false })} disabled={submitting}>
          Run fresh analysis
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card:           { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "28px 32px", maxWidth: 680, margin: "0 auto", boxShadow: "0 8px 40px #00000044" },
  cacheTag:       { fontSize: 11, fontWeight: 700, color: "#f9e2af", background: "#f9e2af11", border: "1px solid #f9e2af33", borderRadius: 20, padding: "3px 10px", display: "inline-block", marginBottom: 10 },
  title:          { color: "#cdd6f4", marginTop: 0, fontSize: 18, fontWeight: 700 },
  message:        { color: "#a6adc8", fontSize: 14, marginBottom: 14 },
  metaRow:        { display: "flex", gap: 10, alignItems: "center", marginBottom: 16 },
  metaBadge:      { background: "#89b4fa22", color: "#89b4fa", fontSize: 12, fontWeight: 700, padding: "4px 10px", borderRadius: 20, border: "1px solid #89b4fa44" },
  metaType:       { color: "#585b70", fontSize: 13 },
  recRow:         { marginBottom: 14 },
  label:          { color: "#585b70", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 6 },
  rec:            { color: "#a6e3a1", fontWeight: 600, fontSize: 14 },
  previewSection: { marginBottom: 18 },
  preview:        { background: "#181825", borderRadius: 8, padding: "14px 18px", maxHeight: 280, overflowY: "auto" as const, border: "1px solid #313244" },
  actions:        { display: "flex", gap: 12 },
  btnPrimary:     { padding: "10px 22px", background: "#a6e3a1", color: "#1e1e2e", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 14 },
  btnSecondary:   { padding: "10px 22px", background: "transparent", color: "#a6adc8", border: "1px solid #45475a", borderRadius: 8, cursor: "pointer", fontSize: 14 },
};
