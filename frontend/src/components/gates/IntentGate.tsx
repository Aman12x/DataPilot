import { useState } from "react";
import { gateCard, gateTitle, gateMessage, gateTextarea, gateActions, gateBtnApprove } from "./shared";

interface Props {
  payload: { question: string; task: string; message: string };
  onSubmit: (value: object) => void;
  submitting?: boolean;
}

export default function IntentGate({ payload, onSubmit, submitting }: Props) {
  const [answer, setAnswer] = useState("");

  return (
    <div style={{ ...gateCard, maxWidth: 560 }}>
      <div style={s.tag}>One quick question</div>
      <h3 style={gateTitle}>DataPilot needs a bit more context</h3>
      <p style={{ ...gateMessage, marginBottom: 10 }}>{payload.message}</p>
      <p style={s.question}>{payload.question}</p>
      <textarea
        style={{ ...gateTextarea, marginBottom: 14 }}
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        placeholder="Your answer…"
        rows={3}
        disabled={submitting}
        autoFocus
      />
      <button
        style={{ ...gateBtnApprove, ...(!answer.trim() || submitting ? s.disabled : {}) }}
        onClick={() => onSubmit({ answer })}
        disabled={!answer.trim() || submitting}
      >
        {submitting ? "Submitting…" : "Continue →"}
      </button>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  tag:      { fontSize: 11, fontWeight: 700, color: "#89b4fa", background: "#89b4fa11", border: "1px solid #89b4fa33", borderRadius: 20, padding: "3px 10px", display: "inline-block", marginBottom: 10 },
  question: { color: "#cdd6f4", fontWeight: 600, fontSize: 15, marginBottom: 16, lineHeight: 1.5 },
  disabled: { opacity: 0.45, cursor: "not-allowed" as const },
};
