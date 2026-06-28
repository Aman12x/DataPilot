import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import client, { checkAuth } from "../api/client";
import AuthCard, { authShared } from "../components/AuthCard";
import FormField from "../components/FormField";
import Spinner from "../components/Spinner";
import { extractApiError } from "../utils/error";

const PASSWORD_HINT = "At least 8 characters with a letter and a number.";

function passwordsMatch(a: string, b: string) {
  return a.length > 0 && a === b;
}

export default function Login() {
  const navigate = useNavigate();
  const [tab,     setTab]     = useState<"login" | "register">("login");
  const [form,    setForm]    = useState({ login: "", username: "", email: "", password: "", passwordConfirm: "" });
  const [error,   setError]   = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingEmail, setPendingEmail] = useState("");

  useEffect(() => {
    checkAuth().then((ok) => { if (ok) navigate("/", { replace: true }); });
  }, [navigate]);

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const continueAsGuest = async () => {
    setLoading(true);
    setError("");
    try {
      await client.post("/auth/guest");
      navigate("/");
    } catch (err) {
      setError(extractApiError(err, "Could not start guest session"));
    } finally {
      setLoading(false);
    }
  };

  const resendVerification = async () => {
    if (!pendingEmail) return;
    setLoading(true);
    setError("");
    try {
      await client.post("/auth/resend-verification", { email: pendingEmail });
      setError("");
    } catch (err) {
      setError(extractApiError(err, "Could not resend verification email."));
    } finally {
      setLoading(false);
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (tab === "register" && form.password !== form.passwordConfirm) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      const url  = tab === "login" ? "/auth/login" : "/auth/register";
      const body = tab === "login"
        ? { login: form.login, password: form.password }
        : { username: form.username, email: form.email, password: form.password };
      const { data } = await client.post(url, body);

      if (tab === "register" && data.verify_pending) {
        setPendingEmail(form.email);
        return;
      }

      navigate("/");
    } catch (err) {
      setError(extractApiError(err, "Authentication failed"));
    } finally {
      setLoading(false);
    }
  };

  if (pendingEmail) {
    return (
      <AuthCard tagline="Almost there">
        <div style={s.pendingBox} className="fade-in">
          <div style={s.pendingIcon}>✉</div>
          <p style={s.pendingTitle}>Check your email</p>
          <p style={s.pendingSub}>
            We sent a verification link to <strong style={{ color: "#cdd6f4" }}>{pendingEmail}</strong>.
            Click the link to activate your account, then sign in.
          </p>
          {error && (
            <div style={authShared.errorBox} role="alert" aria-live="polite">
              <span>⚠</span> {error}
            </div>
          )}
          <button style={authShared.btn} type="button" disabled={loading} onClick={resendVerification}>
            {loading ? <><Spinner variant="button" /> Sending…</> : "Resend verification email"}
          </button>
          <button
            style={authShared.linkBtn}
            type="button"
            onClick={() => { setPendingEmail(""); setTab("login"); }}
          >
            Back to Sign In
          </button>
        </div>
      </AuthCard>
    );
  }

  const loadingLabel = tab === "login" ? "Signing in…" : "Creating account…";

  return (
    <AuthCard tagline="AI-powered experiment & data analyst">
      <div style={s.tabs}>
        {(["login", "register"] as const).map((t) => (
          <button
            key={t}
            style={tab === t ? s.tabActive : s.tab}
            onClick={() => { setTab(t); setError(""); setForm((f) => ({ ...f, passwordConfirm: "" })); }}
            type="button"
          >
            {t === "login" ? "Sign In" : "Create Account"}
          </button>
        ))}
      </div>

      <form onSubmit={submit} style={s.form}>
        {tab === "login" ? (
          <FormField
            label="Username or email"
            type="text"
            value={form.login}
            onChange={set("login")}
            placeholder="you@example.com"
            autoComplete="username"
          />
        ) : (
          <>
            <FormField
              label="Username"
              type="text"
              value={form.username}
              onChange={set("username")}
              placeholder="john_doe"
              autoComplete="username"
            />
            <FormField
              label="Email"
              type="email"
              value={form.email}
              onChange={set("email")}
              placeholder="you@example.com"
              autoComplete="email"
            />
          </>
        )}
        <div>
          <FormField
            label="Password"
            type="password"
            value={form.password}
            onChange={set("password")}
            placeholder="••••••••"
            autoComplete={tab === "register" ? "new-password" : "current-password"}
          />
          {tab === "register" && (
            <p style={s.hint}>{PASSWORD_HINT}</p>
          )}
          {tab === "login" && (
            <button type="button" style={s.forgotLink} onClick={() => navigate("/forgot-password")}>
              Forgot password?
            </button>
          )}
        </div>

        {tab === "register" && (
          <FormField
            label="Confirm password"
            type="password"
            value={form.passwordConfirm}
            onChange={set("passwordConfirm")}
            placeholder="••••••••"
            autoComplete="new-password"
          />
        )}

        {tab === "register" && form.passwordConfirm && !passwordsMatch(form.password, form.passwordConfirm) && (
          <p style={s.matchErr} role="alert">Passwords do not match.</p>
        )}

        {error && (
          <div style={authShared.errorBox} role="alert" aria-live="polite" className="fade-in">
            <span>⚠</span> {error}
          </div>
        )}

        <button style={authShared.btn} type="submit" disabled={loading}>
          {loading ? <><Spinner variant="button" /> {loadingLabel}</> : tab === "login" ? "Sign In →" : "Create Account →"}
        </button>
      </form>

      <div style={s.divider}>
        <div style={s.dividerLine} />
        <span style={{ color: "#45475a", fontSize: 12, flexShrink: 0 }}>or</span>
        <div style={s.dividerLine} />
      </div>
      <button
        style={s.guestBtn}
        className="dp-guest-btn"
        onClick={continueAsGuest}
        type="button"
        disabled={loading}
      >
        {loading ? "Starting guest session…" : "Continue as Guest"}
      </button>
    </AuthCard>
  );
}

const s: Record<string, React.CSSProperties> = {
  tabs:       { display: "flex", background: "#181825", borderRadius: 8, padding: 3, marginBottom: 24, gap: 3 },
  tab:        { flex: 1, padding: "8px 0", background: "transparent", color: "#585b70", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 500, transition: "all 0.2s" },
  tabActive:  { flex: 1, padding: "8px 0", background: "#313244", color: "#cdd6f4", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.2s" },
  form:       { display: "flex", flexDirection: "column", gap: 16 },
  hint:       { color: "#585b70", fontSize: 12, marginTop: 6, lineHeight: 1.4 },
  matchErr:   { color: "#f38ba8", fontSize: 12, marginTop: -8 },
  forgotLink: { background: "none", border: "none", color: "#585b70", fontSize: 12, cursor: "pointer", marginTop: 4, padding: 0, textAlign: "right" as const, width: "100%", display: "block" },
  divider:    { display: "flex", alignItems: "center", gap: 12, margin: "18px 0 0" },
  dividerLine:{ flex: 1, height: 1, background: "#313244" },
  guestBtn:   { width: "100%", marginTop: 10, padding: "10px 0", background: "transparent", color: "#585b70", border: "1px solid #313244", borderRadius: 8, cursor: "pointer", fontSize: 13, fontWeight: 500 },
  pendingBox: { textAlign: "center" },
  pendingIcon:{ fontSize: 36, marginBottom: 12 },
  pendingTitle:{ fontSize: 18, fontWeight: 600, color: "#cdd6f4", marginBottom: 8 },
  pendingSub: { color: "#a6adc8", fontSize: 14, lineHeight: 1.6, marginBottom: 20 },
};
