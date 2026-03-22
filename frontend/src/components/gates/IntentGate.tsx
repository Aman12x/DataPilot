import { useState } from "react";

interface Props {
  payload: {
    question: string;
    task: string;
    message: string;
  };
  onSubmit: (value: object) => void;
  submitting?: boolean;
}

export default function IntentGate({ payload, onSubmit, submitting }: Props) {
  const [answer, setAnswer] = useState("");

  return (
    <div style={styles.card}>
      <div style={styles.tag}>One quick question</div>
      <h3 style={styles.title}>DataPilot needs a bit more context</h3>
      <p style={styles.message}>{payload.message}</p>
      <p style={styles.question}>{payload.question}</p>
      <textarea
        style={styles.textarea}
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        placeholder="Your answer…"
        rows={3}
        disabled={submitting}
        autoFocus
      />
      <button
        style={{ ...styles.btn, ...(!answer.trim() || submitting ? styles.btnDisabled : {}) }}
        onClick={() => onSubmit({ answer })}
        disabled={!answer.trim() || submitting}
      >
        {submitting ? "Submitting…" : "Continue →"}
      </button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card:        { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "28px 32px", maxWidth: 560, margin: "0 auto", boxShadow: "0 8px 40px #00000044" },
  tag:         { fontSize: 11, fontWeight: 700, color: "#89b4fa", background: "#89b4fa11", border: "1px solid #89b4fa33", borderRadius: 20, padding: "3px 10px", display: "inline-block", marginBottom: 10 },
  title:       { color: "#cdd6f4", marginTop: 0, fontSize: 18, fontWeight: 700 },
  message:     { color: "#a6adc8", marginBottom: 10, fontSize: 14 },
  question:    { color: "#cdd6f4", fontWeight: 600, fontSize: 15, marginBottom: 16, lineHeight: 1.5 },
  textarea:    { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "10px 12px", fontSize: 14, resize: "vertical" as const, boxSizing: "border-box" as const, marginBottom: 14 },
  btn:         { padding: "11px 24px", background: "linear-gradient(135deg, #89b4fa, #74c7ec)", color: "#1e1e2e", border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 14 },
  btnDisabled: { opacity: 0.45, cursor: "not-allowed" as const },
};
