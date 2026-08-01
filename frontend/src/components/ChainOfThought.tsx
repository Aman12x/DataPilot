import { useEffect, useState } from "react";
import type { StepEvent } from "../hooks/useSSE";

interface Props {
  steps: StepEvent[];
  isRunning: boolean;
}

export default function ChainOfThought({ steps, isRunning }: Props) {
  const [open, setOpen] = useState(false);

  // Auto-open while the run is active so users see live progress without clicking.
  // Once the run finishes they can collapse it; don't force re-open on completion.
  useEffect(() => {
    if (isRunning) setOpen(true);
  }, [isRunning]);

  if (steps.length === 0 && !isRunning) return null;

  return (
    <div style={s.wrap}>
      <button style={s.toggle} onClick={() => setOpen(v => !v)}>
        <span style={s.toggleIcon}>{open ? "▲" : "▼"}</span>
        {open ? "Hide processing details" : "Show processing details"}
        {steps.length > 0 && (
          <span style={s.badge}>{steps.length}</span>
        )}
      </button>

      {open && (
        <div style={s.list}>
          {steps.map((step, i) => {
            const isCurrent = isRunning && i === steps.length - 1;
            return (
              <div key={i} style={s.row}>
                <span style={isCurrent ? s.dotRunning : s.dotDone}>
                  {isCurrent ? "●" : "✓"}
                </span>
                <span style={s.label}>{step.label}</span>
                {step.detail && (
                  <span style={s.detail}>{step.detail}</span>
                )}
                {isCurrent && (
                  <span style={s.runningLabel}>[running…]</span>
                )}
              </div>
            );
          })}
          {isRunning && steps.length === 0 && (
            <div style={s.row}>
              <span style={s.dotRunning}>●</span>
              <span style={s.label}>Starting…</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  wrap:         { marginTop: 12, borderRadius: 8, border: "1px solid var(--dp-line)", overflow: "hidden" },
  toggle:       { width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", background: "var(--dp-surface-2)", border: "none", cursor: "pointer", color: "var(--dp-ink-muted)", fontSize: 12, textAlign: "left" as const },
  toggleIcon:   { fontSize: 10, color: "var(--dp-ink-faint)" },
  badge:        { marginLeft: "auto", background: "var(--dp-line)", color: "var(--dp-ink-secondary)", fontSize: 11, padding: "1px 7px", borderRadius: 10 },
  list:         { padding: "10px 14px 12px", background: "var(--dp-bg)", display: "flex", flexDirection: "column" as const, gap: 4, maxHeight: 260, overflowY: "auto" as const },
  row:          { display: "flex", alignItems: "baseline", gap: 8, fontFamily: "monospace", fontSize: 12 },
  dotDone:      { color: "var(--dp-success)", flexShrink: 0, width: 12 },
  dotRunning:   { color: "var(--dp-accent)", flexShrink: 0, width: 12, animation: "pulse 1.2s ease-in-out infinite" },
  label:        { color: "var(--dp-ink-secondary)", minWidth: 180 },
  detail:       { color: "var(--dp-ink-muted)", fontSize: 11 },
  runningLabel: { color: "var(--dp-ink-faint)", fontSize: 11, fontStyle: "italic" as const },
};
