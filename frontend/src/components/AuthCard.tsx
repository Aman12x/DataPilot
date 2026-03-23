export const authShared: Record<string, React.CSSProperties> = {
  errorBox: {
    background: "#f38ba811", border: "1px solid #f38ba844", color: "#f38ba8",
    borderRadius: 6, padding: "10px 14px", fontSize: 13, display: "flex", alignItems: "center", gap: 8,
  },
  btn: {
    padding: "12px 0", background: "linear-gradient(135deg, #89b4fa, #74c7ec)",
    color: "#1e1e2e", border: "none", borderRadius: 8,
    fontWeight: 700, fontSize: 15, cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "center", gap: 8, marginTop: 4,
  },
  linkBtn: {
    marginTop: 16, background: "none", border: "none",
    color: "#585b70", fontSize: 13, cursor: "pointer", textAlign: "center", width: "100%",
  },
};

const s: Record<string, React.CSSProperties> = {
  outer:    { minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#11111b", overflow: "hidden", position: "relative" },
  orb1:     { position: "absolute", width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, #89b4fa18 0%, transparent 70%)", top: -120, left: -120, pointerEvents: "none" },
  orb2:     { position: "absolute", width: 400, height: 400, borderRadius: "50%", background: "radial-gradient(circle, #cba6f718 0%, transparent 70%)", bottom: -100, right: -80, pointerEvents: "none" },
  card:     { position: "relative", zIndex: 1, background: "#1e1e2e", border: "1px solid #313244", borderRadius: 16, padding: "36px 32px", width: 380, boxShadow: "0 24px 64px #00000066" },
  logoRow:  { display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 12 },
  logoIcon: { fontSize: 24, color: "#89b4fa" },
  logoText: { fontSize: 24, fontWeight: 700, color: "#cdd6f4", letterSpacing: "-0.5px" },
  tagline:  { color: "#585b70", fontSize: 13, textAlign: "center", marginBottom: 24 },
};

export default function AuthCard({ children, tagline }: { children: React.ReactNode; tagline?: string }) {
  return (
    <div style={s.outer}>
      <div style={s.orb1} />
      <div style={s.orb2} />
      <div style={s.card} className="slide-up">
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
