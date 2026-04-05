import type { DeckData } from "../hooks/useSSE";

interface Props {
  deck:          DeckData;
  onViewReport:  () => void;
}

const VERDICT_CONFIG = {
  positive: { label: "Ship It",    color: "#a6e3a1", bg: "#a6e3a115", icon: "▲" },
  negative: { label: "Don't Ship", color: "#f38ba8", bg: "#f38ba815", icon: "▼" },
  neutral:  { label: "Investigate", color: "#f9e2af", bg: "#f9e2af15", icon: "◆" },
};

export default function StakeholderDeck({ deck, onViewReport }: Props) {
  const cfg = VERDICT_CONFIG[deck.verdict] ?? VERDICT_CONFIG.neutral;

  return (
    <div style={s.card} className="fade-in">
      {/* Header row */}
      <div style={s.header}>
        <div style={{ ...s.verdictBadge, color: cfg.color, background: cfg.bg }}>
          <span style={{ fontSize: 11 }}>{cfg.icon}</span>
          &nbsp;{cfg.label}
        </div>
        <button style={s.fullReportBtn} onClick={onViewReport}>
          Full Report ↓
        </button>
      </div>

      {/* Headline */}
      <p style={s.headline}>{deck.headline}</p>

      {/* Hero metric + confidence */}
      <div style={s.heroRow}>
        <div style={{ ...s.heroBox, borderColor: cfg.color + "44" }}>
          <div style={{ ...s.heroMetric, color: cfg.color }}>{deck.hero_metric}</div>
          <div style={s.heroLabel}>primary metric</div>
        </div>
        <div style={s.confidencePill}>{deck.confidence}</div>
      </div>

      {/* Evidence */}
      <div style={s.section}>
        <div style={s.sectionLabel}>Key Evidence</div>
        <ul style={s.evidenceList}>
          {deck.evidence.map((e, i) => (
            <li key={i} style={s.evidenceItem}>
              <span style={s.bullet}>▸</span>
              <span>{e}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Recommendation */}
      <div style={{ ...s.recBox, borderColor: cfg.color + "55" }}>
        <span style={{ ...s.recLabel, color: cfg.color }}>Recommendation</span>
        <p style={s.recText}>{deck.recommendation}</p>
      </div>

      {/* Watch out */}
      {deck.watch_out && (
        <div style={s.watchOut}>
          <span style={s.watchIcon}>⚠</span>
          <span style={s.watchText}>{deck.watch_out}</span>
        </div>
      )}
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  card:          { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "28px 32px", maxWidth: 760 },
  header:        { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 },
  verdictBadge:  { display: "inline-flex", alignItems: "center", gap: 5, padding: "5px 14px", borderRadius: 20, fontWeight: 700, fontSize: 13, letterSpacing: "0.04em" },
  fullReportBtn: { background: "transparent", border: "1px solid #45475a", color: "#cdd6f4", padding: "6px 16px", borderRadius: 8, cursor: "pointer", fontSize: 13 },
  headline:      { color: "#cdd6f4", fontSize: 17, fontWeight: 600, lineHeight: 1.5, margin: "0 0 22px" },
  heroRow:       { display: "flex", alignItems: "center", gap: 16, marginBottom: 24 },
  heroBox:       { background: "#181825", border: "2px solid", borderRadius: 10, padding: "16px 24px", minWidth: 140, textAlign: "center" as const },
  heroMetric:    { fontSize: 28, fontWeight: 800, letterSpacing: "-0.5px" },
  heroLabel:     { color: "#585b70", fontSize: 11, fontWeight: 600, textTransform: "uppercase" as const, letterSpacing: "0.06em", marginTop: 4 },
  confidencePill:{ background: "#181825", border: "1px solid #313244", borderRadius: 8, padding: "10px 16px", color: "#a6adc8", fontSize: 13, lineHeight: 1.5 },
  section:       { marginBottom: 20 },
  sectionLabel:  { color: "#585b70", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 10 },
  evidenceList:  { listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column" as const, gap: 7 },
  evidenceItem:  { display: "flex", gap: 10, color: "#cdd6f4", fontSize: 14, lineHeight: 1.5 },
  bullet:        { color: "#6c7086", flexShrink: 0, paddingTop: 1 },
  recBox:        { background: "#181825", border: "1px solid", borderRadius: 10, padding: "14px 18px", marginBottom: 16 },
  recLabel:      { fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", display: "block", marginBottom: 6 },
  recText:       { color: "#cdd6f4", fontSize: 14, fontWeight: 500, margin: 0, lineHeight: 1.6 },
  watchOut:      { display: "flex", gap: 8, alignItems: "flex-start" },
  watchIcon:     { color: "#f9e2af", fontSize: 13, flexShrink: 0, paddingTop: 1 },
  watchText:     { color: "#a6adc8", fontSize: 13, lineHeight: 1.5 },
};
