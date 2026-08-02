/**
 * PipelineProgress — step indicator shown while a run is active.
 *
 * Steps visible to the user mirror the graph nodes that produce meaningful
 * output. The current step is highlighted; completed steps show a check;
 * pending steps are dimmed.
 */

import { IconCheck } from "./icons";

interface Step {
  id: string;
  label: string;
}

const AB_STEPS: Step[] = [
  { id: "schema",    label: "Load schema" },
  { id: "sql",       label: "Generate SQL" },
  { id: "query",     label: "Execute query" },
  { id: "stats",     label: "Run statistics" },
  { id: "narrative", label: "Write narrative" },
];

const GENERAL_STEPS: Step[] = [
  { id: "schema",    label: "Load schema" },
  { id: "sql",       label: "Generate SQL" },
  { id: "query",     label: "Execute query" },
  { id: "describe",  label: "Describe data" },
  { id: "narrative", label: "Write insights" },
];

/** Map gate name → the step that is currently interrupting. */
const GATE_TO_STEP: Record<string, string> = {
  intent:        "schema",
  semantic_cache:"schema",
  metric:        "schema",
  query:         "sql",
  analysis:      "stats",
  narrative:     "narrative",
};

function stepIndex(steps: Step[], gate: string | null): number {
  if (!gate) return -1;   // running between gates — unknown step
  const id = GATE_TO_STEP[gate];
  return steps.findIndex((s) => s.id === id);
}

interface Props {
  gate:         string | null;  // current gate name (null = running)
  lastGate?:    string | null;  // last gate the user approved (for between-gate progress)
  analysisMode: string;
  done?:        boolean;
}

export default function PipelineProgress({ gate, lastGate, analysisMode, done }: Props) {
  // Mode not yet known — show a compact "detecting" placeholder
  if (!analysisMode && !done) {
    return (
      <div style={s.wrapper} className="fade-in">
        <div style={s.detectRow}>
          <span style={s.dot} />
          <span style={{ color: "var(--dp-ink-secondary)", fontSize: 14 }}>Detecting analysis type…</span>
        </div>
        <p style={s.status}>
          <span style={s.dot} /> Starting up. This may take a moment.
        </p>
      </div>
    );
  }

  const steps   = analysisMode === "general" ? GENERAL_STEPS : AB_STEPS;
  const running = !done && gate === null;

  // When running between gates: use lastGate to determine how far we've come.
  // Steps up through lastGate's step are complete; the next step is active/running.
  const lastStep = lastGate ? stepIndex(steps, lastGate) : -1;
  const current  = done          ? steps.length
                 : gate !== null ? stepIndex(steps, gate)
                 : lastStep >= 0 ? lastStep + 1
                 : -1;

  return (
    <div style={s.wrapper} className="fade-in">
      <div style={s.track}>
        {steps.map((step, i) => {
          const isComplete = done || i < current;
          const isActive   = !done && i === current;

          return (
            <div key={step.id} style={s.stepRow}>
              {i > 0 && (
                <div style={{
                  ...s.connector,
                  background: isComplete ? "var(--dp-accent)" : "var(--dp-line)",
                }} />
              )}

              <div style={{
                ...s.node,
                ...(isComplete ? s.nodeComplete : isActive ? s.nodeActive : s.nodePending),
              }}>
                {isComplete ? <IconCheck size={13} /> : i + 1}
              </div>

              <div style={{
                ...s.label,
                color: isComplete ? "var(--dp-ink-secondary)"
                     : isActive   ? "var(--dp-ink)"
                     : "var(--dp-ink-faint)",
                fontWeight: isActive ? 600 : 400,
              }}>
                {step.label}
                {isActive && !running && (
                  <span style={s.waitBadge}>Waiting for review</span>
                )}
                {isActive && running && (
                  <span style={s.runBadge}>In progress</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {running && (
        <p style={s.status}>
          <span style={s.dot} /> Analysis in progress. This may take a minute.
        </p>
      )}
      {done && (
        <p style={{ ...s.status, color: "var(--dp-success)" }}>
          <IconCheck /> Analysis complete
        </p>
      )}
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  wrapper:      { padding: "28px 24px", maxWidth: 400, margin: "0 auto" },
  track:        { display: "flex", flexDirection: "column", gap: 0 },
  stepRow:      { display: "flex", alignItems: "center", gap: 14, position: "relative", paddingBottom: 6 },
  connector:    { position: "absolute", left: 13, top: -18, width: 2, height: 20 },
  node:         { width: 28, height: 28, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 600, flexShrink: 0, transition: "all 0.2s", fontVariantNumeric: "tabular-nums" },
  nodeComplete: { background: "var(--dp-accent)", border: "1px solid var(--dp-accent)", color: "#FFFFFF" },
  nodeActive:   { background: "var(--dp-surface)", border: "2px solid var(--dp-accent)", color: "var(--dp-accent)" },
  nodePending:  { background: "var(--dp-surface)", border: "1px solid var(--dp-line-strong)", color: "var(--dp-ink-faint)" },
  label:        { fontSize: 14, transition: "color 0.2s", display: "flex", alignItems: "center", gap: 8 },
  waitBadge:    { fontSize: 11, background: "var(--dp-accent-soft)", color: "var(--dp-accent)", padding: "2px 8px", borderRadius: 4, fontWeight: 500 },
  runBadge:     { fontSize: 11, background: "var(--dp-surface-2)", color: "var(--dp-ink-muted)", padding: "2px 8px", borderRadius: 4, fontWeight: 500, animation: "pulse 1.6s infinite" },
  status:       { marginTop: 22, color: "var(--dp-ink-secondary)", fontSize: 13, display: "flex", alignItems: "center", gap: 8 },
  dot:          { display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: "var(--dp-accent)", animation: "pulse 1.4s ease-in-out infinite" },
  detectRow:    { display: "flex", alignItems: "center", gap: 10, padding: "12px 0" },
};
