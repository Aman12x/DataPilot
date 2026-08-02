import { BrandMark } from "./AppShell";

export const authShared: Record<string, React.CSSProperties> = {
  errorBox: {
    background: "var(--dp-danger-soft)",
    border: "1px solid rgba(194, 59, 74, 0.25)",
    color: "var(--dp-danger)",
    borderRadius: 6,
    padding: "10px 14px",
    fontSize: 13,
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  btn: {
    padding: "11px 0",
    background: "var(--dp-accent)",
    color: "var(--dp-accent-ink)",
    border: "none",
    borderRadius: 6,
    fontWeight: 600,
    fontSize: 14,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    marginTop: 4,
    width: "100%",
  },
  linkBtn: {
    marginTop: 16,
    background: "none",
    border: "none",
    color: "var(--dp-ink-muted)",
    fontSize: 13,
    cursor: "pointer",
    textAlign: "center",
    width: "100%",
  },
};

export default function AuthCard({ children, tagline }: { children: React.ReactNode; tagline?: string }) {
  return (
    <div className="dp-auth-page">
      <aside className="dp-auth-aside">
        <BrandMark size="lg" />
        <div className="dp-auth-aside-inner">
          <h1>Analysis your team can stand behind.</h1>
          <p>
            DataPilot runs the statistics and writes the report. You review the
            SQL, the method, and the story before anything ships.
          </p>
          <ul className="dp-auth-points">
            <li>Connect Postgres, MySQL, BigQuery, or a CSV</li>
            <li>Certified metric definitions shared across the team</li>
            <li>Every analysis is reviewed and reproducible</li>
          </ul>
        </div>
      </aside>
      <div className="dp-auth-main">
        <div className="dp-auth-card fade-in">
          <div style={{ marginBottom: 8 }}>
            <BrandMark />
          </div>
          {tagline && (
            <p style={{ color: "var(--dp-ink-muted)", fontSize: 14, marginBottom: 22, lineHeight: 1.5 }}>
              {tagline}
            </p>
          )}
          {children}
        </div>
      </div>
    </div>
  );
}
