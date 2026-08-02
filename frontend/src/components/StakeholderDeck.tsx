import type { DeckData } from "../hooks/useSSE";
import { IconAlert } from "./icons";
import { normalizeTypography } from "../utils/markdown";

interface Props {
  deck:          DeckData;
  onViewReport:  () => void;
}

const VERDICT_CONFIG = {
  positive: { label: "Ship It",    color: "var(--dp-success)", bg: "var(--dp-success-soft)" },
  negative: { label: "Don't Ship", color: "var(--dp-danger)", bg: "var(--dp-danger-soft)" },
  neutral:  { label: "Investigate", color: "var(--dp-warning)", bg: "var(--dp-warning-soft)" },
};

export default function StakeholderDeck({ deck, onViewReport }: Props) {
  const cfg = VERDICT_CONFIG[deck.verdict] ?? VERDICT_CONFIG.neutral;

  return (
    <div style={s.card} className="fade-in">
      {/* Header row */}
      <div style={s.header}>
        <div style={{ ...s.verdictBadge, color: cfg.color, background: cfg.bg }}>
          {cfg.label}
        </div>
        <button style={s.fullReportBtn} onClick={onViewReport}>
          View full report
        </button>
      </div>

      {/* Headline */}
      <p style={s.headline}>{normalizeTypography(deck.headline)}</p>

      {/* Hero metric + confidence */}
      <div style={s.heroRow}>
        <div style={{ ...s.heroBox, borderColor: cfg.color + "44" }}>
          <div style={{ ...s.heroMetric, color: cfg.color }}>{normalizeTypography(deck.hero_metric)}</div>
          <div style={s.heroLabel}>primary metric</div>
        </div>
        <div style={s.confidencePill}>{normalizeTypography(deck.confidence)}</div>
      </div>

      {/* Evidence */}
      <div style={s.section}>
        <div style={s.sectionLabel}>Key Evidence</div>
        <ul style={s.evidenceList}>
          {deck.evidence.map((e, i) => (
            <li key={i} style={s.evidenceItem}>
              <span style={s.bullet} aria-hidden="true" />
              <span>{normalizeTypography(e)}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Recommendation */}
      <div style={{ ...s.recBox, borderColor: cfg.color + "55" }}>
        <span style={{ ...s.recLabel, color: cfg.color }}>Recommendation</span>
        <p style={s.recText}>{normalizeTypography(deck.recommendation)}</p>
      </div>

      {/* Watch out */}
      {deck.watch_out && (
        <div style={s.watchOut}>
          <span style={s.watchIcon}><IconAlert /></span>
          <span style={s.watchText}>{normalizeTypography(deck.watch_out)}</span>
        </div>
      )}
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  card:          { background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 10, padding: "28px 32px", maxWidth: 760 },
  header:        { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 },
  verdictBadge:  { display: "inline-flex", alignItems: "center", gap: 5, padding: "4px 12px", borderRadius: 4, fontWeight: 600, fontSize: 12, letterSpacing: "0.04em", textTransform: "uppercase" as const },
  fullReportBtn: { background: "transparent", border: "1px solid var(--dp-ink-faint)", color: "var(--dp-ink)", padding: "6px 16px", borderRadius: 8, cursor: "pointer", fontSize: 13 },
  headline:      { color: "var(--dp-ink)", fontSize: 17, fontWeight: 600, lineHeight: 1.5, margin: "0 0 22px" },
  heroRow:       { display: "flex", alignItems: "center", gap: 16, marginBottom: 24 },
  heroBox:       { background: "var(--dp-surface)", border: "1px solid", borderRadius: 8, padding: "16px 24px", minWidth: 140, textAlign: "center" as const },
  heroMetric:    { fontSize: 28, fontWeight: 700, letterSpacing: "-0.5px", fontVariantNumeric: "tabular-nums" as const },
  heroLabel:     { color: "var(--dp-ink-muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase" as const, letterSpacing: "0.06em", marginTop: 4 },
  confidencePill:{ background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", borderRadius: 8, padding: "10px 16px", color: "var(--dp-ink-secondary)", fontSize: 13, lineHeight: 1.5 },
  section:       { marginBottom: 20 },
  sectionLabel:  { color: "var(--dp-ink-muted)", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 10 },
  evidenceList:  { listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column" as const, gap: 7 },
  evidenceItem:  { display: "flex", gap: 10, color: "var(--dp-ink)", fontSize: 14, lineHeight: 1.5 },
  bullet:        { width: 5, height: 5, borderRadius: "50%", background: "var(--dp-ink-faint)", flexShrink: 0, marginTop: 8 },
  recBox:        { background: "var(--dp-surface-2)", border: "1px solid", borderRadius: 10, padding: "14px 18px", marginBottom: 16 },
  recLabel:      { fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", display: "block", marginBottom: 6 },
  recText:       { color: "var(--dp-ink)", fontSize: 14, fontWeight: 500, margin: 0, lineHeight: 1.6 },
  watchOut:      { display: "flex", gap: 8, alignItems: "flex-start" },
  watchIcon:     { color: "var(--dp-warning)", fontSize: 13, flexShrink: 0, paddingTop: 1 },
  watchText:     { color: "var(--dp-ink-secondary)", fontSize: 13, lineHeight: 1.5 },
};
