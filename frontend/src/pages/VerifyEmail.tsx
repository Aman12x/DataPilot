import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import client from "../api/client";
import AuthCard, { authShared } from "../components/AuthCard";
import Spinner from "../components/Spinner";
import { IconAlert, IconCheck } from "../components/icons";
import { extractApiError } from "../utils/error";

export default function VerifyEmail() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    const token = params.get("token");
    if (!token) {
      setStatus("error");
      setError("Missing verification token.");
      return;
    }

    let cancelled = false;
    client.post("/auth/verify-email", { token })
      .then(() => {
        if (!cancelled) {
          setStatus("ok");
          setTimeout(() => navigate("/", { replace: true }), 1500);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setStatus("error");
          setError(extractApiError(err, "Verification failed."));
        }
      });

    return () => { cancelled = true; };
  }, [params, navigate]);

  return (
    <AuthCard tagline="Email verification">
      {status === "loading" && (
        <div style={s.center}>
          <Spinner variant="button" />
          <p style={s.text}>Verifying your email…</p>
        </div>
      )}
      {status === "ok" && (
        <div style={s.success} className="fade-in">
          <div style={s.icon}><IconCheck /></div>
          <p style={s.title}>Email verified</p>
          <p style={s.text}>Redirecting you to DataPilot…</p>
        </div>
      )}
      {status === "error" && (
        <div className="fade-in">
          <div style={authShared.errorBox} role="alert" aria-live="polite">
            <IconAlert /> {error}
          </div>
          <button style={authShared.btn} type="button" onClick={() => navigate("/login")}>
            Back to Sign In
          </button>
        </div>
      )}
    </AuthCard>
  );
}

const s: Record<string, React.CSSProperties> = {
  center:  { display: "flex", flexDirection: "column", alignItems: "center", gap: 16, padding: "24px 0" },
  success: { textAlign: "center", padding: "16px 0" },
  icon:    { fontSize: 28, color: "var(--dp-success)", marginBottom: 12 },
  title:   { fontSize: 18, fontWeight: 600, color: "var(--dp-ink)", marginBottom: 8 },
  text:    { color: "var(--dp-ink-secondary)", fontSize: 14, lineHeight: 1.6 },
};
