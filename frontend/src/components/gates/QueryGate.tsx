import { useState } from "react";

interface Props {
  payload: {
    generated_sql: string;
    message: string;
    sql_validation_warnings: string[];
  };
  onSubmit: (value: object) => void;
  submitting?: boolean;
}

export default function QueryGate({ payload, onSubmit, submitting }: Props) {
  const [sql, setSql] = useState(payload.generated_sql);

  return (
    <div style={styles.card}>
      <h3 style={styles.title}>Review the SQL Query</h3>
      <p style={styles.message}>{payload.message}</p>

      {payload.sql_validation_warnings?.length > 0 && (
        <div style={styles.warnings}>
          {payload.sql_validation_warnings.map((w, i) => (
            <div key={i} style={styles.warning}>⚠ {w}</div>
          ))}
        </div>
      )}

      <div style={styles.sqlLabel}>You can edit the SQL before running it</div>
      <textarea
        style={styles.sql}
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        rows={8}
        spellCheck={false}
        disabled={submitting}
      />

      <div style={styles.actions}>
        <button style={styles.btnApprove} onClick={() => onSubmit({ approved: true, sql })} disabled={submitting}>
          {submitting ? "Running…" : "Approve & Run →"}
        </button>
        <button
          style={styles.btnReject}
          onClick={() => onSubmit({ approved: false, sql })}
          disabled={submitting}
          title="Ask DataPilot to generate a different SQL query"
        >
          Regenerate
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card:       { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "28px 32px", maxWidth: 720, margin: "0 auto", boxShadow: "0 8px 40px #00000044" },
  title:      { color: "#cdd6f4", marginTop: 0, fontSize: 18, fontWeight: 700 },
  message:    { color: "#a6adc8", marginBottom: 14, fontSize: 14 },
  warnings:   { background: "#fab38711", border: "1px solid #fab38744", borderRadius: 8, padding: "10px 14px", marginBottom: 14 },
  warning:    { color: "#fab387", fontSize: 13, marginBottom: 2 },
  sqlLabel:   { color: "#585b70", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 6 },
  sql:        { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "12px 14px", fontFamily: "monospace", fontSize: 13, resize: "vertical" as const, boxSizing: "border-box" as const },
  actions:    { display: "flex", gap: 12, marginTop: 18 },
  btnApprove: { padding: "10px 22px", background: "#a6e3a1", color: "#1e1e2e", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 14 },
  btnReject:  { padding: "10px 22px", background: "transparent", color: "#a6adc8", border: "1px solid #45475a", borderRadius: 8, cursor: "pointer", fontSize: 14 },
};
