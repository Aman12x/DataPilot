/**
 * PipelineProgress — animated step indicator shown while a run is active.
 *
 * Steps visible to the user mirror the graph nodes that produce meaningful
 * output.  The current step pulses; completed steps show a tick; pending
 * steps are dimmed.
 */

interface Step {
  id: string;
  label: string;
  icon: string;
}

const AB_STEPS: Step[] = [
  { id: "schema",    label: "Load schema",    icon: "🗄️" },
  { id: "sql",       label: "Generate SQL",   icon: "✏️" },
  { id: "query",     label: "Execute query",  icon: "⚡" },
  { id: "stats",     label: "Run statistics", icon: "📊" },
  { id: "narrative", label: "Write narrative",icon: "📝" },
];

const GENERAL_STEPS: Step[] = [
  { id: "schema",      label: "Load schema",    icon: "🗄️" },
  { id: "sql",         label: "Generate SQL",   icon: "✏️" },
  { id: "query",       label: "Execute query",  icon: "⚡" },
  { id: "describe",    label: "Describe data",  icon: "🔍" },
  { id: "narrative",   label: "Write insights", icon: "💡" },
];

/** Map gate name → the step that is currently interrupting. */
const GATE_TO_STEP: Record<string, string> = {
  intent:        "schema",
  semantic_cache:"schema",
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
          <span style={{ color: "#a6adc8", fontSize: 14 }}>Detecting analysis type…</span>
        </div>
        <p style={s.status}>
          <span style={s.dot} /> Starting up — this may take a moment
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
          const isPending  = !isComplete && !isActive;

          return (
            <div key={step.id} style={s.stepRow}>
              {/* Connector line */}
              {i > 0 && (
                <div style={{
                  ...s.connector,
                  background: isComplete ? "#89b4fa" : "#313244",
                }} />
              )}

              {/* Node */}
              <div style={{
                ...s.node,
                ...(isComplete ? s.nodeComplete : {}),
                ...(isActive   ? s.nodeActive   : {}),
                ...(isPending  ? s.nodePending  : {}),
              }}>
                {isComplete ? "✓" : step.icon}
              </div>

              {/* Label */}
              <div style={{
                ...s.label,
                color: isComplete ? "#a6e3a1"
                     : isActive   ? "#cdd6f4"
                     : "#585b70",
                fontWeight: isActive ? 600 : 400,
              }}>
                {step.label}
                {isActive && !running && (
                  <span style={s.waitBadge}>waiting for you</span>
                )}
                {isActive && running && (
                  <span style={s.runBadge}>processing…</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom status text */}
      {running && (
        <p style={s.status}>
          <span style={s.dot} /> Analysis in progress — this may take a minute
        </p>
      )}
      {done && <p style={{ ...s.status, color: "#a6e3a1" }}>✓ Analysis complete</p>}
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  wrapper:      { padding: "32px 24px", maxWidth: 400, margin: "0 auto" },
  track:        { display: "flex", flexDirection: "column", gap: 0 },
  stepRow:      { display: "flex", alignItems: "center", gap: 14, position: "relative", paddingBottom: 4 },
  connector:    { position: "absolute", left: 17, top: -20, width: 2, height: 24, borderRadius: 1 },
  node:         { width: 36, height: 36, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0, transition: "all 0.3s" },
  nodeComplete: { background: "#1a3a2a", border: "2px solid #a6e3a1", color: "#a6e3a1", fontSize: 14 },
  nodeActive:   { background: "#1a2a3a", border: "2px solid #89b4fa", color: "#89b4fa", animation: "glow 1.8s ease-in-out infinite" },
  nodePending:  { background: "#181825", border: "2px solid #313244", color: "#45475a" },
  label:        { fontSize: 14, transition: "color 0.3s", display: "flex", alignItems: "center", gap: 8 },
  waitBadge:    { fontSize: 10, background: "#89b4fa22", color: "#89b4fa", padding: "2px 7px", borderRadius: 10, animation: "pulse 2s infinite" },
  runBadge:     { fontSize: 10, background: "#f9e2af22", color: "#f9e2af", padding: "2px 7px", borderRadius: 10, animation: "pulse 1.2s infinite" },
  status:       { marginTop: 24, color: "#a6adc8", fontSize: 13, display: "flex", alignItems: "center", gap: 8 },
  dot:          { display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#f9e2af", animation: "pulse 1.2s ease-in-out infinite" },
  detectRow:    { display: "flex", alignItems: "center", gap: 10, padding: "12px 0" },
};
