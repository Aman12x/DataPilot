import { useState } from "react";
import { gateCard, gateTitle, gateMessage, gateActions, gateBtnApprove, gateBtnSecondary } from "./shared";

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
    <div style={{ ...gateCard, maxWidth: 720 }}>
      <h3 style={gateTitle}>Review the SQL Query</h3>
      <p style={{ ...gateMessage, marginBottom: 14 }}>{payload.message}</p>

      {payload.sql_validation_warnings?.length > 0 && (
        <div style={s.warnings}>
          {payload.sql_validation_warnings.map((w, i) => (
            <div key={i} style={s.warning}>⚠ {w}</div>
          ))}
        </div>
      )}

      <div style={s.sqlLabel}>You can edit the SQL before running it</div>
      <textarea
        style={s.sql}
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        rows={8}
        spellCheck={false}
        disabled={submitting}
      />

      <div style={gateActions}>
        <button style={gateBtnApprove} onClick={() => onSubmit({ approved: true, sql })} disabled={submitting}>
          {submitting ? "Running…" : "Approve & Run →"}
        </button>
        <button
          style={gateBtnSecondary}
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

const s: Record<string, React.CSSProperties> = {
  warnings: { background: "#fab38711", border: "1px solid #fab38744", borderRadius: 8, padding: "10px 14px", marginBottom: 14 },
  warning:  { color: "#fab387", fontSize: 13, marginBottom: 2 },
  sqlLabel: { color: "#585b70", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 6 },
  sql:      { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "12px 14px", fontFamily: "monospace", fontSize: 13, resize: "vertical" as const, boxSizing: "border-box" as const },
};
