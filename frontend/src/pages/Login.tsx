import { useState } from "react";
import { useNavigate } from "react-router-dom";
import client from "../api/client";

export default function Login() {
  const navigate = useNavigate();
  const [tab,     setTab]     = useState<"login" | "register">("login");
  const [form,    setForm]    = useState({ login: "", username: "", email: "", password: "" });
  const [error,   setError]   = useState("");
  const [loading, setLoading] = useState(false);

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const url  = tab === "login" ? "/auth/login" : "/auth/register";
      const body = tab === "login"
        ? { login: form.login, password: form.password }
        : { username: form.username, email: form.email, password: form.password };
      const { data } = await client.post(url, body);
      localStorage.setItem("access_token",  data.access_token);
      localStorage.setItem("refresh_token", data.refresh_token);
      navigate("/");
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail;
      setError(detail ?? "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={s.outer}>
      {/* Animated background orbs */}
      <div style={s.orb1} />
      <div style={s.orb2} />

      <div style={s.card} className="slide-up">
        {/* Logo */}
        <div style={s.logoRow}>
          <span style={s.logoIcon}>✦</span>
          <span style={s.logoText}>DataPilot</span>
        </div>
        <p style={s.tagline}>AI-powered experiment & data analyst</p>

        {/* Tabs */}
        <div style={s.tabs}>
          {(["login", "register"] as const).map((t) => (
            <button
              key={t}
              style={tab === t ? s.tabActive : s.tab}
              onClick={() => { setTab(t); setError(""); }}
              type="button"
            >
              {t === "login" ? "Sign In" : "Create Account"}
            </button>
          ))}
        </div>

        <form onSubmit={submit} style={s.form}>
          {tab === "login" ? (
            <Field label="Username or email" type="text"     value={form.login}    onChange={set("login")}    placeholder="you@example.com" />
          ) : (
            <>
              <Field label="Username"  type="text"  value={form.username} onChange={set("username")} placeholder="john_doe" />
              <Field label="Email"     type="email" value={form.email}    onChange={set("email")}    placeholder="you@example.com" />
            </>
          )}
          <div>
            <Field label="Password" type="password" value={form.password} onChange={set("password")} placeholder="••••••••" />
            {tab === "login" && (
              <button
                type="button"
                style={s.forgotLink}
                onClick={() => navigate("/forgot-password")}
              >
                Forgot password?
              </button>
            )}
          </div>

          {error && (
            <div style={s.errorBox} className="fade-in">
              <span style={s.errorIcon}>⚠</span> {error}
            </div>
          )}

          <button style={s.btn} type="submit" disabled={loading}>
            {loading
              ? <><span style={s.spinner} /> Signing in…</>
              : tab === "login" ? "Sign In →" : "Create Account →"}
          </button>
        </form>
      </div>
    </div>
  );
}

function Field({ label, type, value, onChange, placeholder }: {
  label: string; type: string; value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder: string;
}) {
  return (
    <div style={f.group}>
      <label style={f.label}>{label}</label>
      <input
        style={f.input}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required
        autoComplete={type === "password" ? "current-password" : "on"}
      />
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  outer:    {
    minHeight: "100vh",
    display: "flex", alignItems: "center", justifyContent: "center",
    background: "#11111b",
    overflow: "hidden", position: "relative",
  },
  orb1:     {
    position: "absolute", width: 500, height: 500, borderRadius: "50%",
    background: "radial-gradient(circle, #89b4fa18 0%, transparent 70%)",
    top: -120, left: -120, pointerEvents: "none",
  },
  orb2:     {
    position: "absolute", width: 400, height: 400, borderRadius: "50%",
    background: "radial-gradient(circle, #cba6f718 0%, transparent 70%)",
    bottom: -100, right: -80, pointerEvents: "none",
  },
  card:     {
    position: "relative", zIndex: 1,
    background: "#1e1e2e",
    border: "1px solid #313244",
    borderRadius: 16, padding: "36px 32px", width: 380,
    boxShadow: "0 24px 64px #00000066",
  },
  logoRow:  { display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 6 },
  logoIcon: { fontSize: 24, color: "#89b4fa" },
  logoText: { fontSize: 24, fontWeight: 700, color: "#cdd6f4", letterSpacing: "-0.5px" },
  tagline:  { color: "#585b70", fontSize: 13, textAlign: "center", marginBottom: 28 },
  tabs:     { display: "flex", background: "#181825", borderRadius: 8, padding: 3, marginBottom: 24, gap: 3 },
  tab:      { flex: 1, padding: "8px 0", background: "transparent", color: "#585b70", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 500, transition: "all 0.2s" },
  tabActive:{ flex: 1, padding: "8px 0", background: "#313244", color: "#cdd6f4", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.2s" },
  form:     { display: "flex", flexDirection: "column", gap: 16 },
  errorBox: { background: "#f38ba811", border: "1px solid #f38ba844", color: "#f38ba8", borderRadius: 6, padding: "10px 14px", fontSize: 13, display: "flex", alignItems: "center", gap: 8 },
  errorIcon:{ fontSize: 15 },
  btn:      {
    padding: "12px 0", background: "linear-gradient(135deg, #89b4fa, #74c7ec)",
    color: "#1e1e2e", border: "none", borderRadius: 8,
    fontWeight: 700, fontSize: 15, cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
    marginTop: 4,
  },
  spinner:    { width: 16, height: 16, border: "2px solid #1e1e2e44", borderTop: "2px solid #1e1e2e", borderRadius: "50%", animation: "spin 0.7s linear infinite", display: "inline-block" },
  forgotLink: { background: "none", border: "none", color: "#585b70", fontSize: 12, cursor: "pointer", marginTop: 4, padding: 0, textAlign: "right" as const, width: "100%", display: "block" },
};

const f: Record<string, React.CSSProperties> = {
  group:  { display: "flex", flexDirection: "column", gap: 6 },
  label:  { fontSize: 12, fontWeight: 600, color: "#a6adc8", letterSpacing: "0.04em", textTransform: "uppercase" },
  input:  { padding: "10px 12px", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, fontSize: 14, transition: "border-color 0.2s" },
};
