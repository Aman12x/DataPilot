import { useState } from "react";
import { useNavigate } from "react-router-dom";
import client from "../api/client";
import AuthCard, { authShared } from "../components/AuthCard";
import FormField from "../components/FormField";
import Spinner from "../components/Spinner";
import { extractApiError } from "../utils/error";

export default function ForgotPassword() {
  const navigate       = useNavigate();
  const [email,   setEmail]   = useState("");
  const [loading, setLoading] = useState(false);
  const [sent,    setSent]    = useState(false);
  const [error,   setError]   = useState("");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await client.post("/auth/forgot-password", { email });
      setSent(true);
    } catch (err) {
      setError(extractApiError(err, "Something went wrong. Please try again."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthCard>
      {sent ? (
        <div style={s.successBox} className="fade-in">
          <div style={s.successIcon}>✉</div>
          <p style={s.successTitle}>Check your email</p>
          <p style={s.successSub}>
            If that address is registered, we've sent a reset link. It expires in 1 hour.
          </p>
          <button style={authShared.btn} onClick={() => navigate("/login")}>
            Back to Sign In
          </button>
        </div>
      ) : (
        <>
          <p style={s.heading}>Reset your password</p>
          <p style={s.sub}>Enter your email and we'll send you a reset link.</p>

          <form onSubmit={submit} style={s.form}>
            <FormField
              label="Email address"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoFocus
            />

            {error && (
              <div style={authShared.errorBox} className="fade-in">
                <span>⚠</span> {error}
              </div>
            )}

            <button style={authShared.btn} type="submit" disabled={loading}>
              {loading ? <><Spinner variant="button" /> Sending…</> : "Send reset link →"}
            </button>
          </form>

          <button style={authShared.linkBtn} onClick={() => navigate("/login")}>
            ← Back to Sign In
          </button>
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
  successIcon: { fontSize: 40, color: "#89b4fa" },
  successTitle:{ color: "#cdd6f4", fontSize: 18, fontWeight: 600 },
  successSub:  { color: "#a6adc8", fontSize: 13, lineHeight: 1.6 },
};
