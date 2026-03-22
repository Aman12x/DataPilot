import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import client from "../api/client";

export default function ResetPassword() {
  const navigate          = useNavigate();
  const [params]          = useSearchParams();
  const token             = params.get("token") ?? "";

  const [password,  setPassword]  = useState("");
  const [confirm,   setConfirm]   = useState("");
  const [loading,   setLoading]   = useState(false);
  const [done,      setDone]      = useState(false);
  const [error,     setError]     = useState("");

  if (!token) {
    return (
      <div style={s.outer}>
        <div style={s.card}>
          <div style={s.errorBox}>
            <span>⚠</span> Invalid reset link. Please request a new one.
          </div>
          <button style={s.linkBtn} onClick={() => navigate("/forgot-password")}>
            Request new link
          </button>
        </div>
      </div>
    );
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirm) {
      setError("Passwords don't match.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      await client.post("/auth/reset-password", { token, password });
      setDone(true);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail;
      setError(detail ?? "Reset failed. The link may have expired.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={s.outer}>
      <div style={s.orb1} />
      <div style={s.orb2} />

      <div style={s.card} className="slide-up">
        <div style={s.logoRow}>
          <span style={s.logoIcon}>✦</span>
          <span style={s.logoText}>DataPilot</span>
        </div>

        {done ? (
          <div style={s.successBox} className="fade-in">
            <div style={s.successIcon}>✓</div>
            <p style={s.successTitle}>Password updated</p>
            <p style={s.successSub}>Your new password is set. Sign in to continue.</p>
            <button style={s.btn} onClick={() => navigate("/login")}>
              Sign In →
            </button>
          </div>
        ) : (
          <>
            <p style={s.heading}>Set a new password</p>
            <p style={s.sub}>Must be at least 8 characters.</p>

            <form onSubmit={submit} style={s.form}>
              <div style={s.group}>
                <label style={s.label}>New password</label>
                <input
                  style={s.input}
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  autoFocus
                  autoComplete="new-password"
                />
              </div>
              <div style={s.group}>
                <label style={s.label}>Confirm password</label>
                <input
                  style={s.input}
                  type="password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="••••••••"
                  required
                  autoComplete="new-password"
                />
              </div>

              {error && (
                <div style={s.errorBox} className="fade-in">
                  <span>⚠</span> {error}
                </div>
              )}

              <button style={s.btn} type="submit" disabled={loading}>
                {loading
                  ? <><span style={s.spinner} /> Updating…</>
                  : "Update password →"}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  outer:        { minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#11111b", overflow: "hidden", position: "relative" },
  orb1:         { position: "absolute", width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, #89b4fa18 0%, transparent 70%)", top: -120, left: -120, pointerEvents: "none" },
  orb2:         { position: "absolute", width: 400, height: 400, borderRadius: "50%", background: "radial-gradient(circle, #cba6f718 0%, transparent 70%)", bottom: -100, right: -80, pointerEvents: "none" },
  card:         { position: "relative", zIndex: 1, background: "#1e1e2e", border: "1px solid #313244", borderRadius: 16, padding: "36px 32px", width: 380, boxShadow: "0 24px 64px #00000066" },
  logoRow:      { display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 20 },
  logoIcon:     { fontSize: 24, color: "#89b4fa" },
  logoText:     { fontSize: 24, fontWeight: 700, color: "#cdd6f4", letterSpacing: "-0.5px" },
  heading:      { color: "#cdd6f4", fontSize: 18, fontWeight: 600, textAlign: "center", marginBottom: 6 },
  sub:          { color: "#585b70", fontSize: 13, textAlign: "center", marginBottom: 24 },
  form:         { display: "flex", flexDirection: "column", gap: 16 },
  group:        { display: "flex", flexDirection: "column", gap: 6 },
  label:        { fontSize: 12, fontWeight: 600, color: "#a6adc8", letterSpacing: "0.04em", textTransform: "uppercase" },
  input:        { padding: "10px 12px", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, fontSize: 14 },
  errorBox:     { background: "#f38ba811", border: "1px solid #f38ba844", color: "#f38ba8", borderRadius: 6, padding: "10px 14px", fontSize: 13, display: "flex", alignItems: "center", gap: 8 },
  btn:          { padding: "12px 0", background: "linear-gradient(135deg, #89b4fa, #74c7ec)", color: "#1e1e2e", border: "none", borderRadius: 8, fontWeight: 700, fontSize: 15, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, marginTop: 4 },
  spinner:      { width: 16, height: 16, border: "2px solid #1e1e2e44", borderTop: "2px solid #1e1e2e", borderRadius: "50%", animation: "spin 0.7s linear infinite", display: "inline-block" },
  linkBtn:      { marginTop: 16, background: "none", border: "none", color: "#89b4fa", fontSize: 13, cursor: "pointer", textAlign: "center", width: "100%" },
  successBox:   { textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 12 },
  successIcon:  { fontSize: 40, color: "#a6e3a1" },
  successTitle: { color: "#cdd6f4", fontSize: 18, fontWeight: 600 },
  successSub:   { color: "#a6adc8", fontSize: 13, lineHeight: 1.6 },
};
