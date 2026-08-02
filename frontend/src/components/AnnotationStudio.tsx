import { useEffect, useState } from "react";
import { IconClose } from "./icons";
import client from "../api/client";
import { extractApiError } from "../utils/error";
import type { SavedConnection } from "../types/analysis";

type ColComment = { table: string; column: string; comment: string };
type SynonymRow = { term: string; column: string };

interface AnnotationsResponse {
  connection_id: string;
  annotations: Record<string, Record<string, string>>;
  synonyms: Record<string, string>;
  updated_at?: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  /** Prefill a connection when opened from the analysis form. */
  initialConnectionId?: string;
  canEdit?: boolean;
}

function annotationsToRows(ann: Record<string, Record<string, string>>): ColComment[] {
  const rows: ColComment[] = [];
  for (const [table, cols] of Object.entries(ann || {})) {
    for (const [column, comment] of Object.entries(cols || {})) {
      rows.push({ table, column, comment });
    }
  }
  return rows.length ? rows : [{ table: "", column: "", comment: "" }];
}

function rowsToAnnotations(rows: ColComment[]): Record<string, Record<string, string>> {
  const out: Record<string, Record<string, string>> = {};
  for (const r of rows) {
    const table = r.table.trim();
    const column = r.column.trim();
    const comment = r.comment.trim();
    if (!table || !column || !comment) continue;
    if (!out[table]) out[table] = {};
    out[table][column] = comment;
  }
  return out;
}

function synonymsToRows(syn: Record<string, string>): SynonymRow[] {
  const rows = Object.entries(syn || {}).map(([term, column]) => ({ term, column }));
  return rows.length ? rows : [{ term: "", column: "" }];
}

function rowsToSynonyms(rows: SynonymRow[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const r of rows) {
    const term = r.term.trim();
    const column = r.column.trim();
    if (term && column) out[term] = column;
  }
  return out;
}

export default function AnnotationStudio({
  open,
  onClose,
  initialConnectionId = "",
  canEdit,
}: Props) {
  const [connections, setConnections] = useState<SavedConnection[]>([]);
  const [connectionId, setConnectionId] = useState("");
  const [rows, setRows] = useState<ColComment[]>([{ table: "", column: "", comment: "" }]);
  const [synRows, setSynRows] = useState<SynonymRow[]>([{ term: "", column: "" }]);
  const [tables, setTables] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState("");
  const [okMsg, setOkMsg] = useState("");
  const [updatedAt, setUpdatedAt] = useState("");
  const [editable, setEditable] = useState(canEdit !== false);
  const [drift, setDrift] = useState<string[]>([]);

  useEffect(() => {
    if (!open) return;
    setError("");
    setOkMsg("");
    setDrift([]);
    if (canEdit !== undefined) setEditable(canEdit);

    client.get<{ connections: SavedConnection[] }>("/connections")
      .then((r) => {
        const list = r.data.connections || [];
        setConnections(list);
        const preferred = initialConnectionId
          || (list.find((c) => c.connection_id === connectionId)?.connection_id)
          || list[0]?.connection_id
          || "";
        setConnectionId(preferred);
      })
      .catch(() => setError("Could not load connections"));

    if (canEdit === undefined) {
      client.get<{ workspaces: { workspace_id: string; role: string }[] }>("/workspaces")
        .then((r) => {
          const list = r.data.workspaces || [];
          const saved = (() => {
            try { return localStorage.getItem("datapilot.workspace_id"); } catch { return null; }
          })();
          const ws = list.find((w) => w.workspace_id === saved) || list[0];
          setEditable(!ws || ws.role === "owner");
        })
        .catch(() => setEditable(true));
    }
  }, [open, initialConnectionId, canEdit]);

  useEffect(() => {
    if (!open || !connectionId) {
      setRows([{ table: "", column: "", comment: "" }]);
      setSynRows([{ term: "", column: "" }]);
      setTables([]);
      setUpdatedAt("");
      return;
    }
    setLoading(true);
    setError("");
    client.get<AnnotationsResponse>(`/connections/${connectionId}/annotations`)
      .then((r) => {
        setRows(annotationsToRows(r.data.annotations || {}));
        setSynRows(synonymsToRows(r.data.synonyms || {}));
        setUpdatedAt(r.data.updated_at || "");
      })
      .catch((err) => setError(extractApiError(err, "Could not load annotations")))
      .finally(() => setLoading(false));
  }, [open, connectionId]);

  const loadTablesFromTest = async () => {
    if (!connectionId) return;
    setTesting(true);
    setError("");
    setDrift([]);
    try {
      const { data } = await client.post<{
        success?: boolean;
        tables?: string[];
        drift_warnings?: string[];
        error?: string;
      }>(`/connections/${connectionId}/test`);
      if (!data.success) {
        setError(data.error || "Connection test failed");
      } else {
        setTables(data.tables || []);
        setDrift(data.drift_warnings || []);
        setOkMsg(
          data.tables?.length
            ? `Loaded ${data.tables.length} table name(s) from live schema.`
            : "Connection ok, but no table names returned.",
        );
      }
    } catch (err) {
      setError(extractApiError(err, "Could not test connection"));
    } finally {
      setTesting(false);
    }
  };

  const save = async () => {
    if (!connectionId) return;
    setSaving(true);
    setError("");
    setOkMsg("");
    try {
      const { data } = await client.put<AnnotationsResponse>(
        `/connections/${connectionId}/annotations`,
        {
          annotations: rowsToAnnotations(rows),
          synonyms: rowsToSynonyms(synRows),
        },
      );
      setRows(annotationsToRows(data.annotations || {}));
      setSynRows(synonymsToRows(data.synonyms || {}));
      setUpdatedAt(data.updated_at || "");
      setOkMsg("Annotations saved. They will be included in schema context on the next analysis.");
    } catch (err) {
      setError(extractApiError(err, "Could not save annotations"));
    } finally {
      setSaving(false);
    }
  };

  const setRow = (idx: number, patch: Partial<ColComment>) => {
    setRows((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  };

  const setSyn = (idx: number, patch: Partial<SynonymRow>) => {
    setSynRows((prev) => prev.map((r, i) => (i === idx ? { ...r, ...patch } : r)));
  };

  if (!open) return null;

  return (
    <div style={s.backdrop} onClick={onClose} role="presentation">
      <div style={s.modal} className="fade-in" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Schema annotations">
        <div style={s.header}>
          <div>
            <div style={s.kicker}>Schema annotations</div>
            <h2 style={s.title}>Teach DataPilot your column meanings</h2>
            <p style={s.sub}>
              Comments and business synonyms are injected into the schema context so SQL and metrics
              match how your team talks about the data.
            </p>
          </div>
          <button className="dp-btn dp-btn-link" onClick={onClose}>Close</button>
        </div>

        {error && <div style={s.error}>{error}</div>}
        {okMsg && <div style={s.ok}>{okMsg}</div>}
        {drift.length > 0 && (
          <div style={s.warn}>
            <div style={{ fontWeight: 700, marginBottom: 4 }}>Schema notes</div>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {drift.map((w) => <li key={w}>{w}</li>)}
            </ul>
          </div>
        )}

        {connections.length === 0 ? (
          <p style={s.muted}>Save a Postgres connection first, then come back to annotate it.</p>
        ) : (
          <>
            <div style={s.connRow}>
              <label style={s.field}>
                <span style={s.label}>Connection</span>
                <select
                  style={s.input}
                  value={connectionId}
                  onChange={(e) => setConnectionId(e.target.value)}
                >
                  {connections.map((c) => (
                    <option key={c.connection_id} value={c.connection_id}>
                      {c.name} ({c.host}/{c.dbname})
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="dp-btn dp-btn-ghost"
                style={{ padding: "8px 12px", alignSelf: "end" }}
                onClick={loadTablesFromTest}
                disabled={!connectionId || testing}
              >
                {testing ? "Testing…" : "Load tables"}
              </button>
            </div>

            {tables.length > 0 && (
              <div style={s.tableChips}>
                {tables.slice(0, 24).map((t) => (
                  <button
                    key={t}
                    type="button"
                    style={s.chip}
                    disabled={!editable}
                    onClick={() => setRows((prev) => [...prev, { table: t, column: "", comment: "" }])}
                    title="Add annotation row for this table"
                  >
                    {t}
                  </button>
                ))}
              </div>
            )}

            {!editable && (
              <p style={s.muted}>You are an analyst, so annotations are read-only. Ask an owner to edit.</p>
            )}

            {loading ? (
              <p style={s.muted}>Loading annotations…</p>
            ) : (
              <>
                <div style={s.sectionHead}>
                  <span style={s.sectionTitle}>Column comments</span>
                  {editable && (
                    <button
                      className="dp-btn dp-btn-ghost"
                      style={s.tinyBtn}
                      onClick={() => setRows((prev) => [...prev, { table: "", column: "", comment: "" }])}
                    >
                      + Row
                    </button>
                  )}
                </div>
                <div style={s.gridHead}>
                  <span>Table</span><span>Column</span><span>Meaning</span><span />
                </div>
                {rows.map((r, idx) => (
                  <div key={idx} style={s.gridRow}>
                    <input
                      style={s.input}
                      placeholder="events"
                      value={r.table}
                      disabled={!editable}
                      onChange={(e) => setRow(idx, { table: e.target.value })}
                    />
                    <input
                      style={s.input}
                      placeholder="revenue_usd"
                      value={r.column}
                      disabled={!editable}
                      onChange={(e) => setRow(idx, { column: e.target.value })}
                    />
                    <input
                      style={s.input}
                      placeholder="Gross revenue in USD"
                      value={r.comment}
                      disabled={!editable}
                      onChange={(e) => setRow(idx, { comment: e.target.value })}
                    />
                    {editable && (
                      <button
                        className="dp-btn dp-btn-link"
                        style={{ color: "var(--dp-danger)" }}
                        onClick={() => setRows((prev) => prev.filter((_, i) => i !== idx))}
                      >
                        <IconClose />
                      </button>
                    )}
                  </div>
                ))}

                <div style={{ ...s.sectionHead, marginTop: 18 }}>
                  <span style={s.sectionTitle}>Business synonyms</span>
                  {editable && (
                    <button
                      className="dp-btn dp-btn-ghost"
                      style={s.tinyBtn}
                      onClick={() => setSynRows((prev) => [...prev, { term: "", column: "" }])}
                    >
                      + Synonym
                    </button>
                  )}
                </div>
                <p style={s.hint}>Map how PMs talk (“WAU”) to physical columns (`weekly_active_users`).</p>
                <div style={s.synHead}>
                  <span>Business term</span><span>Physical column</span><span />
                </div>
                {synRows.map((r, idx) => (
                  <div key={idx} style={s.synRow}>
                    <input
                      style={s.input}
                      placeholder="WAU"
                      value={r.term}
                      disabled={!editable}
                      onChange={(e) => setSyn(idx, { term: e.target.value })}
                    />
                    <input
                      style={s.input}
                      placeholder="weekly_active_users"
                      value={r.column}
                      disabled={!editable}
                      onChange={(e) => setSyn(idx, { column: e.target.value })}
                    />
                    {editable && (
                      <button
                        className="dp-btn dp-btn-link"
                        style={{ color: "var(--dp-danger)" }}
                        onClick={() => setSynRows((prev) => prev.filter((_, i) => i !== idx))}
                      >
                        <IconClose />
                      </button>
                    )}
                  </div>
                ))}

                {updatedAt && (
                  <p style={s.muted}>Last updated {new Date(updatedAt).toLocaleString()}</p>
                )}

                {editable && (
                  <div style={s.actions}>
                    <button
                      className="dp-btn dp-btn-primary"
                      style={{ padding: "10px 18px" }}
                      disabled={saving || !connectionId}
                      onClick={save}
                    >
                      {saving ? "Saving…" : "Save annotations"}
                    </button>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  backdrop: { position: "fixed", inset: 0, background: "#000000aa", zIndex: 80, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 },
  modal: { width: "min(860px, 100%)", maxHeight: "90vh", overflow: "auto", background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 16, padding: 20 },
  header: { display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 16 },
  kicker: { color: "var(--dp-warning)", fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" },
  title: { color: "var(--dp-ink)", fontSize: 20, fontWeight: 700, margin: "4px 0" },
  sub: { color: "var(--dp-ink-muted)", fontSize: 13, margin: 0, maxWidth: 560 },
  error: { background: "var(--dp-danger)11", border: "1px solid var(--dp-danger)44", color: "var(--dp-danger)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 13 },
  ok: { background: "var(--dp-success)11", border: "1px solid var(--dp-success)44", color: "var(--dp-success)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 13 },
  warn: { background: "var(--dp-warning)11", border: "1px solid var(--dp-warning)44", color: "var(--dp-warning)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 12 },
  muted: { color: "var(--dp-ink-muted)", fontSize: 13 },
  hint: { color: "var(--dp-ink-muted)", fontSize: 11, margin: "0 0 8px" },
  connRow: { display: "flex", gap: 10, alignItems: "end", marginBottom: 12, flexWrap: "wrap" },
  field: { display: "flex", flexDirection: "column", gap: 4, flex: "1 1 240px" },
  label: { color: "var(--dp-ink-secondary)", fontSize: 11, fontWeight: 600 },
  input: { background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 8, color: "var(--dp-ink)", padding: "8px 10px", fontSize: 13, width: "100%" },
  tableChips: { display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 },
  chip: { background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", color: "var(--dp-ink-secondary)", borderRadius: 8, padding: "4px 10px", fontSize: 11, cursor: "pointer" },
  sectionHead: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 },
  sectionTitle: { color: "var(--dp-ink)", fontSize: 13, fontWeight: 700 },
  tinyBtn: { padding: "4px 8px", fontSize: 11 },
  gridHead: { display: "grid", gridTemplateColumns: "1fr 1fr 1.4fr 32px", gap: 8, color: "var(--dp-ink-muted)", fontSize: 11, marginBottom: 6, padding: "0 2px" },
  gridRow: { display: "grid", gridTemplateColumns: "1fr 1fr 1.4fr 32px", gap: 8, marginBottom: 8, alignItems: "center" },
  synHead: { display: "grid", gridTemplateColumns: "1fr 1fr 32px", gap: 8, color: "var(--dp-ink-muted)", fontSize: 11, marginBottom: 6 },
  synRow: { display: "grid", gridTemplateColumns: "1fr 1fr 32px", gap: 8, marginBottom: 8, alignItems: "center" },
  actions: { display: "flex", justifyContent: "flex-end", marginTop: 16 },
};
