/**
 * ConnectionsPanel — the "Data sources" modal.
 *
 * The one place to see, test, edit, and remove saved database connections.
 * Mutations are owner-only (canEdit); testing is open to every member because
 * the backend test endpoint is not owner-gated.
 */
import { useEffect, useState } from "react";
import client, { type WorkspaceSummary } from "../api/client";
import { type SavedConnection } from "../types/analysis";
import { extractApiError } from "../utils/error";

type VerifiedQuery = {
  vq_id: string;
  source: "gate" | "contributed";
  name: string;
  task: string;
  sql: string;
  connection_id: string;
};
import Spinner from "./Spinner";
import { IconAlert, IconCheck, IconDot } from "./icons";

// ── Shared helpers (also used by TaskInput and AnnotationStudio) ──────────────

/** Human label for where a connection points: BigQuery is project/dataset. */
export function connectionLabel(c: SavedConnection): string {
  if (c.backend === "bigquery") return `${c.project_id || "?"}/${c.dbname}`;
  return `${c.host}/${c.dbname}`;
}

function relTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const mins = Math.round((Date.now() - then) / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

/** Dot + text health state. Never color alone. */
export function ConnectionHealthBadge({ conn }: { conn: SavedConnection }) {
  if (conn.last_test_ok === true) {
    return (
      <span style={{ ...hb.badge, color: "var(--dp-success)" }} title={relTime(conn.last_tested_at)}>
        <IconCheck size={12} /> Connected{conn.last_tested_at ? ` ${relTime(conn.last_tested_at)}` : ""}
      </span>
    );
  }
  if (conn.last_test_ok === false) {
    return (
      <span
        style={{ ...hb.badge, color: "var(--dp-danger)" }}
        title={conn.last_test_error || "The last connection test failed."}
      >
        <IconAlert size={12} /> Test failed{conn.last_tested_at ? ` ${relTime(conn.last_tested_at)}` : ""}
      </span>
    );
  }
  return (
    <span style={{ ...hb.badge, color: "var(--dp-ink-muted)" }} title="This connection has not been tested yet.">
      <IconDot size={12} /> Not tested
    </span>
  );
}

const hb: Record<string, React.CSSProperties> = {
  badge: { display: "inline-flex", alignItems: "center", gap: 5, fontSize: 12, fontWeight: 500, whiteSpace: "nowrap" },
};

// ── Panel ─────────────────────────────────────────────────────────────────────

const SSL_MODES = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"];

interface FormState {
  name: string;
  backend: "postgres" | "mysql" | "bigquery";
  host: string;
  port: string;
  dbname: string;
  username: string;
  password: string;
  sslmode: string;
  projectId: string;
}

const EMPTY_FORM: FormState = {
  name: "", backend: "postgres", host: "", port: "5432", dbname: "",
  username: "", password: "", sslmode: "prefer", projectId: "",
};

interface Props {
  open: boolean;
  onClose: () => void;
  /** Called after any create/update/delete so callers can refetch. */
  onChanged?: () => void;
  /** When false, panel is read-only apart from Test (analysts). Auto-detected if omitted. */
  canEdit?: boolean;
}

type Msg = { kind: "ok" | "error"; text: string } | null;

export default function ConnectionsPanel({ open, onClose, onChanged, canEdit }: Props) {
  const [connections, setConnections] = useState<SavedConnection[]>([]);
  const [loading, setLoading] = useState(false);
  const [editable, setEditable] = useState(canEdit !== false);
  const [msg, setMsg] = useState<Msg>(null);
  const [drift, setDrift] = useState<string[]>([]);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [mode, setMode] = useState<"list" | "add" | "edit">("list");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>({ ...EMPTY_FORM });

  const load = () => {
    setLoading(true);
    client.get<{ connections: SavedConnection[] }>("/connections")
      .then((r) => setConnections(r.data.connections || []))
      .catch(() => setMsg({ kind: "error", text: "Could not load connections." }))
      .finally(() => setLoading(false));
    client.get<{ verified_queries: VerifiedQuery[] }>("/verified-queries")
      .then((r) => setVqs(r.data.verified_queries || []))
      .catch(() => setVqs([]));
  };

  const [vqs, setVqs] = useState<VerifiedQuery[]>([]);
  const [vqForm, setVqForm] = useState({ name: "", task: "", sql: "" });
  const [vqOpen, setVqOpen] = useState(false);
  const [vqBusy, setVqBusy] = useState(false);

  const addVq = async () => {
    if (!vqForm.task.trim() || !vqForm.sql.trim()) {
      setMsg({ kind: "error", text: "A canonical query needs both the question and the SQL." });
      return;
    }
    setVqBusy(true);
    try {
      await client.post("/verified-queries", vqForm);
      setVqForm({ name: "", task: "", sql: "" });
      setVqOpen(false);
      setMsg({ kind: "ok", text: "Canonical query saved — future analyses will learn from it." });
      load();
    } catch (err) {
      setMsg({ kind: "error", text: extractApiError(err, "Could not save the query.") });
    } finally {
      setVqBusy(false);
    }
  };

  const deleteVq = async (vq: VerifiedQuery) => {
    if (!confirm(`Remove "${vq.name || vq.task}" from verified queries?`)) return;
    try {
      await client.delete(`/verified-queries/${vq.vq_id}`);
      load();
    } catch (err) {
      setMsg({ kind: "error", text: extractApiError(err, "Could not remove the query.") });
    }
  };

  useEffect(() => {
    if (!open) return;
    setMsg(null);
    setDrift([]);
    setMode("list");
    setForm({ ...EMPTY_FORM });
    load();
    if (canEdit !== undefined) {
      setEditable(canEdit);
      return;
    }
    client.get<{ workspaces: WorkspaceSummary[] }>("/workspaces")
      .then((r) => {
        const list = r.data.workspaces || [];
        const saved = (() => {
          try { return localStorage.getItem("datapilot.workspace_id"); } catch { return null; }
        })();
        const ws = list.find((w) => w.workspace_id === saved) || list[0];
        setEditable(!ws || ws.role === "owner");
      })
      .catch(() => setEditable(true));
  }, [open, canEdit]);

  if (!open) return null;

  const set = (k: keyof FormState) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>,
  ) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const isBq = form.backend === "bigquery";
  const editingConn = editingId ? connections.find((c) => c.connection_id === editingId) : null;

  const formValid = isBq
    ? !!(form.projectId && form.dbname && (form.password || mode === "edit"))
    : !!(form.host && form.dbname && form.username);

  const startAdd = () => {
    setMode("add");
    setEditingId(null);
    setForm({ ...EMPTY_FORM });
    setMsg(null);
    setDrift([]);
  };

  const startEdit = (c: SavedConnection) => {
    setMode("edit");
    setEditingId(c.connection_id);
    setForm({
      name: c.name,
      backend: (c.backend as FormState["backend"]) || "postgres",
      host: c.host || "",
      port: String(c.port || (c.backend === "mysql" ? 3306 : 5432)),
      dbname: c.dbname || "",
      username: c.username || "",
      password: "",
      sslmode: c.sslmode || "prefer",
      projectId: c.project_id || "",
    });
    setMsg(null);
    setDrift([]);
  };

  const backToList = () => {
    setMode("list");
    setEditingId(null);
    setMsg(null);
  };

  const formPayload = () => {
    if (isBq) {
      return {
        name: form.name.trim() || `${form.dbname}@${form.projectId}`,
        backend: "bigquery",
        project_id: form.projectId.trim(),
        dbname: form.dbname.trim(),
        password: form.password,
      };
    }
    return {
      name: form.name.trim() || `${form.dbname}@${form.host}`,
      backend: form.backend,
      host: form.host.trim(),
      port: parseInt(form.port) || (form.backend === "mysql" ? 3306 : 5432),
      dbname: form.dbname.trim(),
      username: form.username.trim(),
      password: form.password,
      sslmode: form.sslmode,
    };
  };

  const testSaved = async (c: SavedConnection) => {
    setTestingId(c.connection_id);
    setMsg(null);
    setDrift([]);
    try {
      const { data } = await client.post<{
        success: boolean; error?: string; table_count?: number; drift_warnings?: string[];
      }>(`/connections/${c.connection_id}/test`);
      if (data.success) {
        setMsg({ kind: "ok", text: `${c.name}: connected, ${data.table_count ?? 0} tables visible.` });
      } else {
        setMsg({ kind: "error", text: `${c.name}: ${data.error || "connection test failed."}` });
      }
      if (data.drift_warnings?.length) setDrift(data.drift_warnings);
    } catch {
      setMsg({ kind: "error", text: `${c.name}: connection test failed.` });
    } finally {
      setTestingId(null);
      load();
      onChanged?.();
    }
  };

  const testForm = async () => {
    setBusy(true);
    setMsg(null);
    try {
      if (mode === "edit" && !form.password && editingId) {
        // No new credential entered: exercise the stored one.
        const { data } = await client.post<{ success: boolean; error?: string; table_count?: number }>(
          `/connections/${editingId}/test`,
        );
        setMsg(data.success
          ? { kind: "ok", text: `Connected, ${data.table_count ?? 0} tables visible (using the saved credential).` }
          : { kind: "error", text: data.error || "Connection test failed." });
      } else {
        const { data } = await client.post<{ success: boolean; error?: string; table_count?: number }>(
          "/connections/test-ephemeral",
          formPayload(),
        );
        setMsg(data.success
          ? { kind: "ok", text: `Connected, ${data.table_count ?? 0} tables visible.` }
          : { kind: "error", text: data.error || "Connection test failed." });
      }
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setMsg({ kind: "error", text: detail || "Connection test failed." });
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    setBusy(true);
    setMsg(null);
    try {
      if (mode === "add") {
        await client.post("/connections", { ...formPayload(), test: true });
        setMsg({ kind: "ok", text: "Connection saved and tested." });
      } else if (editingId) {
        const payload: Record<string, unknown> = { name: form.name.trim() || undefined };
        if (isBq) {
          payload.project_id = form.projectId.trim();
          payload.dbname = form.dbname.trim();
        } else {
          payload.host = form.host.trim();
          payload.port = parseInt(form.port) || undefined;
          payload.dbname = form.dbname.trim();
          payload.username = form.username.trim();
          payload.sslmode = form.sslmode;
        }
        if (form.password) payload.password = form.password;
        await client.patch(`/connections/${editingId}`, payload);
        // The update reset stored health; re-test so the badge is fresh.
        await client.post(`/connections/${editingId}/test`).catch(() => {});
        setMsg({ kind: "ok", text: "Connection updated and re-tested." });
      }
      setMode("list");
      setEditingId(null);
      load();
      onChanged?.();
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setMsg({ kind: "error", text: detail || "Could not save the connection." });
    } finally {
      setBusy(false);
    }
  };

  const del = async (c: SavedConnection) => {
    if (!window.confirm(`Delete the connection "${c.name}"? Saved annotations for it will no longer be reachable.`)) return;
    setBusy(true);
    try {
      await client.delete(`/connections/${c.connection_id}`);
      setMsg({ kind: "ok", text: `Deleted ${c.name}.` });
      load();
      onChanged?.();
    } catch {
      setMsg({ kind: "error", text: `Could not delete ${c.name}.` });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={s.backdrop} onClick={onClose} role="presentation">
      <div style={s.modal} className="fade-in" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Data sources">
        <div style={s.header}>
          <div>
            <div style={s.kicker}>Data sources</div>
            <h3 style={s.title}>Manage database connections</h3>
            <p style={s.sub}>
              Saved connections are shared with your workspace. Credentials are encrypted
              and never shown again after saving.
            </p>
          </div>
          <button className="dp-btn dp-btn-link" onClick={onClose}>Close</button>
        </div>

        {msg && <div style={msg.kind === "ok" ? s.ok : s.error}>{msg.text}</div>}
        {drift.length > 0 && (
          <div style={s.warn}>
            <strong style={{ display: "block", marginBottom: 4 }}>Schema notes</strong>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {drift.map((w) => <li key={w}>{w}</li>)}
            </ul>
          </div>
        )}

        {mode === "list" ? (
          <>
            {loading ? (
              <div style={s.loadingRow}><Spinner variant="inline" /> Loading connections</div>
            ) : connections.length === 0 ? (
              <div style={s.empty}>
                <p style={{ ...s.muted, marginBottom: editable ? 12 : 0 }}>
                  No database connections yet. Connect Postgres, MySQL, or BigQuery once
                  and the whole workspace can analyze it.
                </p>
                {editable && (
                  <button className="dp-btn dp-btn-primary" onClick={startAdd}>Add connection</button>
                )}
              </div>
            ) : (
              <>
                <div style={s.list}>
                  {connections.map((c) => (
                    <div key={c.connection_id} style={s.row}>
                      <div style={s.rowMain}>
                        <div style={s.rowName}>
                          {c.name}
                          <span style={s.backendChip}>{c.backend}</span>
                        </div>
                        <div style={s.rowDetail}>{connectionLabel(c)}</div>
                        <ConnectionHealthBadge conn={c} />
                        {c.last_test_ok === false && c.last_test_error && (
                          <div style={s.rowError}>{c.last_test_error}</div>
                        )}
                      </div>
                      <div style={s.rowActions}>
                        <button
                          className="dp-btn dp-btn-ghost"
                          style={s.tinyBtn}
                          onClick={() => testSaved(c)}
                          disabled={testingId !== null || busy}
                        >
                          {testingId === c.connection_id ? <><Spinner variant="inline" /> Testing</> : "Test"}
                        </button>
                        {editable && (
                          <>
                            <button className="dp-btn dp-btn-ghost" style={s.tinyBtn} onClick={() => startEdit(c)} disabled={busy}>
                              Edit
                            </button>
                            <button
                              className="dp-btn dp-btn-ghost"
                              style={{ ...s.tinyBtn, color: "var(--dp-danger)", borderColor: "var(--dp-danger)44" }}
                              onClick={() => del(c)}
                              disabled={busy}
                            >
                              Delete
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
                {editable && (
                  <div style={s.listFoot}>
                    <button className="dp-btn dp-btn-primary" onClick={startAdd}>Add connection</button>
                  </div>
                )}
              </>
            )}
            {!editable && connections.length > 0 && (
              <p style={{ ...s.muted, marginTop: 10 }}>
                You are an analyst in this workspace, so connections are read-only. Ask an owner to make changes.
              </p>
            )}

            <div style={{ borderTop: "1px solid var(--dp-line)", marginTop: 18, paddingTop: 14 }}>
              <div style={s.sectionHead}>
                <span style={s.sectionTitle}>Canonical queries</span>
                {editable && !vqOpen && (
                  <button className="dp-btn dp-btn-ghost" style={{ fontSize: 12 }} onClick={() => setVqOpen(true)}>
                    + Add
                  </button>
                )}
              </div>
              <p style={{ ...s.muted, margin: "4px 0 10px" }}>
                Teach DataPilot how your team writes queries. Approved analyses are learned
                automatically; add a few hand-picked exemplars here — future SQL follows
                their tables, joins, and naming.
              </p>
              {vqs.length === 0 && !vqOpen && (
                <p style={s.muted}>Nothing saved yet. Complete an analysis, or add an exemplar.</p>
              )}
              {vqs.map((v) => (
                <div key={v.vq_id} style={{ display: "flex", alignItems: "baseline", gap: 8, padding: "6px 0", borderBottom: "1px solid var(--dp-line)" }}>
                  <span style={{ color: "var(--dp-ink)", fontSize: 13, fontWeight: 600, flexShrink: 0 }}>
                    {v.name || v.task.slice(0, 48)}
                  </span>
                  <span style={{ ...s.muted, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={v.task}>
                    {v.task}
                  </span>
                  <span style={{ ...s.muted, flexShrink: 0 }}>
                    {v.source === "contributed" ? "exemplar" : "from a run"}
                  </span>
                  {editable && (
                    <button
                      className="dp-btn dp-btn-link"
                      style={{ color: "var(--dp-danger)", fontSize: 12, padding: 0 }}
                      onClick={() => deleteVq(v)}
                    >
                      Remove
                    </button>
                  )}
                </div>
              ))}
              {vqOpen && (
                <div style={{ display: "grid", gap: 8, marginTop: 10 }}>
                  <label style={s.field}>
                    <span style={s.label}>Name</span>
                    <input style={s.input} value={vqForm.name} placeholder="Weekly revenue, the official way"
                           onChange={(e) => setVqForm((f) => ({ ...f, name: e.target.value }))} />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>The question it answers</span>
                    <input style={s.input} value={vqForm.task} placeholder="What is weekly revenue by product line?"
                           onChange={(e) => setVqForm((f) => ({ ...f, task: e.target.value }))} />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>The SQL your team considers correct</span>
                    <textarea
                      style={{ ...s.input, fontFamily: "ui-monospace, monospace", fontSize: 12, minHeight: 90, resize: "vertical" }}
                      value={vqForm.sql}
                      spellCheck={false}
                      onChange={(e) => setVqForm((f) => ({ ...f, sql: e.target.value }))}
                    />
                  </label>
                  <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                    <button className="dp-btn dp-btn-ghost" onClick={() => setVqOpen(false)} disabled={vqBusy}>Cancel</button>
                    <button className="dp-btn dp-btn-primary" onClick={addVq} disabled={vqBusy}>
                      {vqBusy ? "Saving…" : "Save exemplar"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div>
            <div style={s.sectionHead}>
              <span style={s.sectionTitle}>
                {mode === "add" ? "Add connection" : `Edit ${editingConn?.name ?? "connection"}`}
              </span>
              <button className="dp-btn dp-btn-link" onClick={backToList}>Back to list</button>
            </div>

            <div style={s.formGrid}>
              <label style={s.field}>
                <span style={s.label}>Name (optional)</span>
                <input style={s.input} value={form.name} onChange={set("name")} placeholder="Production warehouse" />
              </label>
              <label style={s.field}>
                <span style={s.label}>Backend</span>
                {mode === "edit" ? (
                  <input style={{ ...s.input, color: "var(--dp-ink-muted)" }} value={form.backend} disabled />
                ) : (
                  <select style={s.input} value={form.backend} onChange={(e) => {
                    const backend = e.target.value as FormState["backend"];
                    setForm((f) => ({
                      ...f,
                      backend,
                      port: backend === "mysql" ? (f.port === "5432" ? "3306" : f.port) : (f.port === "3306" ? "5432" : f.port),
                    }));
                  }}>
                    <option value="postgres">PostgreSQL</option>
                    <option value="mysql">MySQL / MariaDB</option>
                    <option value="bigquery">Google BigQuery</option>
                  </select>
                )}
              </label>

              {isBq ? (
                <>
                  <label style={s.field}>
                    <span style={s.label}>Project ID</span>
                    <input style={s.input} value={form.projectId} onChange={set("projectId")} placeholder="my-gcp-project" autoComplete="off" />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>Dataset</span>
                    <input style={s.input} value={form.dbname} onChange={set("dbname")} placeholder="analytics" autoComplete="off" />
                  </label>
                  <label style={{ ...s.field, gridColumn: "1 / -1" }}>
                    <span style={s.label}>
                      Service account JSON{mode === "edit" ? " (leave blank to keep the current key)" : ""}
                    </span>
                    <textarea
                      style={{ ...s.input, fontFamily: "var(--dp-mono)", fontSize: 12, minHeight: 100, resize: "vertical" }}
                      value={form.password}
                      onChange={set("password")}
                      placeholder='{"type":"service_account","project_id":"...","private_key":"..."}'
                      autoComplete="off"
                      spellCheck={false}
                    />
                  </label>
                </>
              ) : (
                <>
                  <label style={s.field}>
                    <span style={s.label}>Host</span>
                    <input style={s.input} value={form.host} onChange={set("host")} placeholder="db.example.com" autoComplete="off" />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>Port</span>
                    <input style={s.input} value={form.port} onChange={set("port")} placeholder={form.backend === "mysql" ? "3306" : "5432"} autoComplete="off" />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>Database</span>
                    <input style={s.input} value={form.dbname} onChange={set("dbname")} placeholder="analytics" autoComplete="off" />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>User</span>
                    <input style={s.input} value={form.username} onChange={set("username")} placeholder={form.backend === "mysql" ? "root" : "postgres"} autoComplete="off" />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>
                      Password{mode === "edit" ? " (leave blank to keep the current one)" : ""}
                    </span>
                    <input style={s.input} type="password" value={form.password} onChange={set("password")} autoComplete="new-password" />
                  </label>
                  <label style={s.field}>
                    <span style={s.label}>SSL mode</span>
                    <select style={s.input} value={form.sslmode} onChange={set("sslmode")}>
                      {SSL_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                    <span style={{ color: "var(--dp-ink-muted)", fontSize: 11 }}>
                      Not sure? Leave the default.
                    </span>
                  </label>
                </>
              )}
            </div>

            <div style={s.formActions}>
              <button className="dp-btn dp-btn-ghost" onClick={testForm} disabled={busy || !formValid}>
                {busy ? <><Spinner variant="inline" /> Working</> : "Test connection"}
              </button>
              <button className="dp-btn dp-btn-primary" onClick={save} disabled={busy || !formValid}>
                {mode === "add" ? "Save connection" : "Save changes"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  backdrop: { position: "fixed", inset: 0, background: "#000000aa", zIndex: 80, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 },
  modal: { width: "min(760px, 100%)", maxHeight: "90vh", overflow: "auto", background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 16, padding: 20 },
  header: { display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 16 },
  kicker: { color: "var(--dp-accent)", fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase" },
  title: { color: "var(--dp-ink)", fontSize: 20, fontWeight: 600, margin: "4px 0", fontFamily: "var(--dp-display)" },
  sub: { color: "var(--dp-ink-muted)", fontSize: 13, margin: 0, maxWidth: 520 },
  error: { background: "var(--dp-danger)11", border: "1px solid var(--dp-danger)44", color: "var(--dp-danger)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 13 },
  ok: { background: "var(--dp-success)11", border: "1px solid var(--dp-success)44", color: "var(--dp-success)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 13 },
  warn: { background: "var(--dp-warning)11", border: "1px solid var(--dp-warning)44", color: "var(--dp-warning)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 12 },
  muted: { color: "var(--dp-ink-muted)", fontSize: 13 },
  loadingRow: { color: "var(--dp-ink-muted)", fontSize: 13, display: "flex", alignItems: "center", gap: 8, padding: "16px 0" },
  empty: { padding: "20px 0 8px" },
  list: { display: "flex", flexDirection: "column", gap: 8 },
  row: { display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, border: "1px solid var(--dp-line)", borderRadius: 8, padding: "12px 14px", background: "var(--dp-surface)" },
  rowMain: { minWidth: 0, display: "flex", flexDirection: "column", gap: 3 },
  rowName: { color: "var(--dp-ink)", fontSize: 14, fontWeight: 600, display: "flex", alignItems: "center", gap: 8 },
  backendChip: { background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", color: "var(--dp-ink-secondary)", borderRadius: 4, padding: "1px 7px", fontSize: 11, fontWeight: 500 },
  rowDetail: { color: "var(--dp-ink-muted)", fontSize: 12, fontFamily: "var(--dp-mono)" },
  rowError: { color: "var(--dp-danger)", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 420 },
  rowActions: { display: "flex", gap: 6, flexShrink: 0 },
  listFoot: { display: "flex", justifyContent: "flex-end", marginTop: 12 },
  tinyBtn: { padding: "5px 10px", fontSize: 12 },
  sectionHead: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  sectionTitle: { color: "var(--dp-ink)", fontSize: 14, fontWeight: 600 },
  formGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 },
  field: { display: "flex", flexDirection: "column", gap: 4 },
  label: { color: "var(--dp-ink-secondary)", fontSize: 11, fontWeight: 600 },
  input: { background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 6, color: "var(--dp-ink)", padding: "8px 10px", fontSize: 13, width: "100%", fontFamily: "inherit" },
  formActions: { display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 16 },
};
