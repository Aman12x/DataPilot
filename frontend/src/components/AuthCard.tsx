import { BrandMark } from "./AppShell";

export const authShared: Record<string, React.CSSProperties> = {
  errorBox: {
    background: "var(--dp-danger-soft)",
    border: "1px solid rgba(194, 59, 74, 0.28)",
    color: "var(--dp-danger)",
    borderRadius: 10,
    padding: "10px 14px",
    fontSize: 13,
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  btn: {
    padding: "12px 0",
    background: "var(--dp-accent)",
    color: "var(--dp-accent-ink)",
    border: "none",
    borderRadius: 10,
    fontWeight: 700,
    fontSize: 15,
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
        <div className="dp-auth-aside-inner">
          <BrandMark size="lg" />
          <h1>Decisions from your warehouse — with humans in the loop.</h1>
          <p>
            Connect data, define metrics once, and let DataPilot run the analysis
            while you approve the SQL and the story.
          </p>
        </div>
      </aside>
      <div className="dp-auth-main">
        <div className="dp-auth-card slide-up">
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
