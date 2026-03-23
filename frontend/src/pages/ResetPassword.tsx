import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import client from "../api/client";
import AuthCard, { authShared } from "../components/AuthCard";
import FormField from "../components/FormField";
import Spinner from "../components/Spinner";
import { extractApiError } from "../utils/error";

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
      <AuthCard>
        <div style={authShared.errorBox}><span>⚠</span> Invalid reset link. Please request a new one.</div>
        <button style={{ ...authShared.linkBtn, color: "#89b4fa" }} onClick={() => navigate("/forgot-password")}>
          Request new link
        </button>
      </AuthCard>
    );
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirm) { setError("Passwords don't match."); return; }
    if (password.length < 8)  { setError("Password must be at least 8 characters."); return; }
    setLoading(true);
    setError("");
    try {
      await client.post("/auth/reset-password", { token, password });
      setDone(true);
    } catch (err) {
      setError(extractApiError(err, "Reset failed. The link may have expired."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthCard>
      {done ? (
        <div style={s.successBox} className="fade-in">
          <div style={s.successIcon}>✓</div>
          <p style={s.successTitle}>Password updated</p>
          <p style={s.successSub}>Your new password is set. Sign in to continue.</p>
          <button style={authShared.btn} onClick={() => navigate("/login")}>Sign In →</button>
        </div>
      ) : (
        <>
          <p style={s.heading}>Set a new password</p>
          <p style={s.sub}>Must be at least 8 characters.</p>

          <form onSubmit={submit} style={s.form}>
            <FormField label="New password"     type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" autoFocus />
            <FormField label="Confirm password" type="password" value={confirm}  onChange={(e) => setConfirm(e.target.value)}  placeholder="••••••••" />

            {error && (
              <div style={authShared.errorBox} className="fade-in">
                <span>⚠</span> {error}
              </div>
            )}

            <button style={authShared.btn} type="submit" disabled={loading}>
              {loading ? <><Spinner variant="button" /> Updating…</> : "Update password →"}
            </button>
          </form>
        </>
      )}
    </AuthCard>
  );
}

const s: Record<string, React.CSSProperties> = {
  form:        { display: "flex", flexDirection: "column", gap: 16 },
  heading:     { color: "#cdd6f4", fontSize: 18, fontWeight: 600, textAlign: "center", marginBottom: 6 },
  sub:         { color: "#585b70", fontSize: 13, textAlign: "center", marginBottom: 24 },
  successBox:  { textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 12 },
  successIcon: { fontSize: 40, color: "#a6e3a1" },
  successTitle:{ color: "#cdd6f4", fontSize: 18, fontWeight: 600 },
  successSub:  { color: "#a6adc8", fontSize: 13, lineHeight: 1.6 },
};
