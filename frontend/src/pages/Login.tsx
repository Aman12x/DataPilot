import { useState } from "react";
import { useNavigate } from "react-router-dom";
import client from "../api/client";
import AuthCard, { authShared } from "../components/AuthCard";
import FormField from "../components/FormField";
import Spinner from "../components/Spinner";
import { extractApiError } from "../utils/error";

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
    } catch (err) {
      setError(extractApiError(err, "Authentication failed"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthCard tagline="AI-powered experiment & data analyst">
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
          <FormField label="Username or email" type="text" value={form.login} onChange={set("login")} placeholder="you@example.com" />
        ) : (
          <>
            <FormField label="Username" type="text"  value={form.username} onChange={set("username")} placeholder="john_doe" />
            <FormField label="Email"    type="email" value={form.email}    onChange={set("email")}    placeholder="you@example.com" />
          </>
        )}
        <div>
          <FormField label="Password" type="password" value={form.password} onChange={set("password")} placeholder="••••••••" />
          {tab === "login" && (
            <button type="button" style={s.forgotLink} onClick={() => navigate("/forgot-password")}>
              Forgot password?
            </button>
          )}
        </div>

        {error && (
          <div style={authShared.errorBox} className="fade-in">
            <span>⚠</span> {error}
          </div>
        )}

        <button style={authShared.btn} type="submit" disabled={loading}>
          {loading ? <><Spinner variant="button" /> Signing in…</> : tab === "login" ? "Sign In →" : "Create Account →"}
        </button>
      </form>
    </AuthCard>
  );
}

const s: Record<string, React.CSSProperties> = {
  tabs:      { display: "flex", background: "#181825", borderRadius: 8, padding: 3, marginBottom: 24, gap: 3 },
  tab:       { flex: 1, padding: "8px 0", background: "transparent", color: "#585b70", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 500, transition: "all 0.2s" },
  tabActive: { flex: 1, padding: "8px 0", background: "#313244", color: "#cdd6f4", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.2s" },
  form:      { display: "flex", flexDirection: "column", gap: 16 },
  forgotLink:{ background: "none", border: "none", color: "#585b70", fontSize: 12, cursor: "pointer", marginTop: 4, padding: 0, textAlign: "right" as const, width: "100%", display: "block" },
};
