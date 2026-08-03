import { useEffect, useState } from "react";
import client from "../api/client";
import { extractApiError } from "../utils/error";

interface Member {
  user_id: string;
  username: string;
  email: string;
  role: string;
  created_at: string;
}

interface Props {
  open: boolean;
  workspaceId: string;
  workspaceName?: string;
  canManage: boolean;
  onClose: () => void;
}

export default function MembersPanel({ open, workspaceId, workspaceName, canManage, onClose }: Props) {
  const [members, setMembers] = useState<Member[]>([]);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<"analyst" | "owner">("analyst");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [okMsg, setOkMsg] = useState("");

  const load = () => {
    if (!workspaceId) return;
    setLoading(true);
    setError("");
    client.get<{ members: Member[] }>(`/workspaces/${workspaceId}/members`)
      .then((r) => setMembers(r.data.members || []))
      .catch((err) => setError(extractApiError(err, "Could not load members")))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (open && workspaceId) load();
  }, [open, workspaceId]);

  const add = async () => {
    if (!email.trim()) return;
    setSaving(true);
    setError("");
    setOkMsg("");
    try {
      await client.post(`/workspaces/${workspaceId}/members`, {
        email: email.trim().toLowerCase(),
        role,
      });
      setEmail("");
      setOkMsg("Member added. They’ll see shared connections, packs, and history.");
      load();
    } catch (err) {
      setError(extractApiError(err, "Could not add member"));
    } finally {
      setSaving(false);
    }
  };

  const remove = async (userId: string) => {
    if (!confirm("Remove this member from the workspace?")) return;
    try {
      await client.delete(`/workspaces/${workspaceId}/members/${userId}`);
      load();
    } catch (err) {
      setError(extractApiError(err, "Could not remove member"));
    }
  };

  if (!open) return null;

  return (
    <div style={s.backdrop} onClick={onClose} role="presentation">
      <div style={s.modal} className="fade-in" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Workspace members">
        <div style={s.header}>
          <div>
            <div style={s.kicker}>Workspace</div>
            <h2 style={s.title}>{workspaceName || "Team members"}</h2>
            <p style={s.sub}>Owners manage connections and packs. Analysts run analyses and read shared history.</p>
          </div>
          <button className="dp-btn dp-btn-link" onClick={onClose}>Close</button>
        </div>

        {error && <div style={s.error}>{error}</div>}
        {okMsg && <div style={s.ok}>{okMsg}</div>}

        {canManage && (
          <div style={s.addRow}>
            <input
              style={s.input}
              type="email"
              placeholder="teammate@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <select style={s.select} value={role} onChange={(e) => setRole(e.target.value as "analyst" | "owner")}>
              <option value="analyst">Analyst</option>
              <option value="owner">Owner</option>
            </select>
            <button
              className="dp-btn dp-btn-primary"
              style={{ padding: "8px 14px" }}
              disabled={saving || !email.trim()}
              onClick={add}
            >
              {saving ? "Adding…" : "Add"}
            </button>
          </div>
        )}

        {loading ? (
          <p style={s.muted}>Loading members…</p>
        ) : (
          <div style={s.list}>
            {members.map((m) => (
              <div key={m.user_id} style={s.row}>
                <div>
                  <div style={s.name}>{m.username || m.email || m.user_id.slice(0, 8)}</div>
                  <div style={s.meta}>{m.email} · {m.role === "owner" ? "Owner" : "Analyst"}</div>
                </div>
                {canManage && (
                  <button className="dp-btn dp-btn-link" style={{ color: "var(--dp-danger)" }} onClick={() => remove(m.user_id)}>
                    Remove
                  </button>
                )}
              </div>
            ))}
            {members.length === 0 && <p style={s.muted}>No members found.</p>}
          </div>
        )}
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  backdrop: { position: "fixed", inset: 0, background: "#000000aa", zIndex: 80, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 },
  modal: { width: "min(560px, 100%)", background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 16, padding: 20 },
  header: { display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 16 },
  kicker: { color: "var(--dp-accent)", fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" },
  title: { color: "var(--dp-ink)", fontSize: 20, fontWeight: 700, margin: "4px 0" },
  sub: { color: "var(--dp-ink-muted)", fontSize: 13, margin: 0 },
  error: { background: "var(--dp-danger)11", border: "1px solid var(--dp-danger)44", color: "var(--dp-danger)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 13 },
  ok: { background: "var(--dp-success)11", border: "1px solid var(--dp-success)44", color: "var(--dp-success)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 13 },
  addRow: { display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" },
  input: { flex: "1 1 180px", background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 8, color: "var(--dp-ink)", padding: "8px 10px", fontSize: 13 },
  select: { background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 8, color: "var(--dp-ink)", padding: "8px 10px", fontSize: 13 },
  list: { display: "flex", flexDirection: "column", gap: 8 },
  row: { display: "flex", justifyContent: "space-between", alignItems: "center", background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", borderRadius: 10, padding: "12px 14px" },
  name: { color: "var(--dp-ink)", fontSize: 14, fontWeight: 600 },
  meta: { color: "var(--dp-ink-muted)", fontSize: 12, marginTop: 2 },
  muted: { color: "var(--dp-ink-muted)", fontSize: 13 },
};
