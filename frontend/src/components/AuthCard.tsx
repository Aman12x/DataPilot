export const authShared: Record<string, React.CSSProperties> = {
  errorBox: {
    background: "#f38ba811", border: "1px solid #f38ba844", color: "#f38ba8",
    borderRadius: 8, padding: "10px 14px", fontSize: 13, display: "flex", alignItems: "center", gap: 8,
  },
  btn: {
    padding: "12px 0", background: "linear-gradient(135deg, #89b4fa, #74c7ec)",
    color: "#1e1e2e", border: "none", borderRadius: 8,
    fontWeight: 700, fontSize: 15, cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "center", gap: 8, marginTop: 4,
    width: "100%",
  },
  linkBtn: {
    marginTop: 16, background: "none", border: "none",
    color: "#585b70", fontSize: 13, cursor: "pointer", textAlign: "center", width: "100%",
  },
};

const s: Record<string, React.CSSProperties> = {
  outer:    { minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#11111b", overflow: "hidden", position: "relative", padding: "24px 16px" },
  orb1:     { position: "absolute", width: 520, height: 520, borderRadius: "50%", background: "radial-gradient(circle, rgba(137,180,250,0.12) 0%, transparent 70%)", top: -140, left: -140, pointerEvents: "none" },
  orb2:     { position: "absolute", width: 420, height: 420, borderRadius: "50%", background: "radial-gradient(circle, rgba(203,166,247,0.1) 0%, transparent 70%)", bottom: -100, right: -80, pointerEvents: "none" },
  logoRow:  { display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 12 },
  logoIcon: { fontSize: 26, color: "#89b4fa" },
  logoText: { fontSize: 26, fontWeight: 700, color: "#cdd6f4", letterSpacing: "-0.04em" },
  tagline:  { color: "#585b70", fontSize: 14, textAlign: "center", marginBottom: 28, lineHeight: 1.5 },
};

export default function AuthCard({ children, tagline }: { children: React.ReactNode; tagline?: string }) {
  return (
    <div style={s.outer}>
      <div style={s.orb1} className="dp-orb" />
      <div style={s.orb2} className="dp-orb" />
      <div className="dp-card dp-auth-card slide-up">
        <div style={s.logoRow}>
          <span style={s.logoIcon}>✦</span>
          <span style={s.logoText}>DataPilot</span>
        </div>
        {tagline && <p style={s.tagline}>{tagline}</p>}
        {children}
      </div>
    </div>
  );
}
