import { useEffect, useState } from "react";
import client, { type WorkspaceSummary } from "../api/client";
import { extractApiError } from "../utils/error";

export interface MetricPack {
  pack_id: string;
  name: string;
  description: string;
  certified: boolean;
  connection_id: string | null;
  version?: number;
  config: MetricConfigForm;
}

type MetricConfigForm = {
  primary_metric: string;
  metric_source_col: string;
  metric_agg: string;
  covariate: string;
  metric_direction: "higher_is_better" | "lower_is_better";
  events_table: string;
  experiment_table: string;
  user_id_col: string;
  date_col: string;
  variant_col: string;
  week_col: string;
  guardrail_metrics: string[];
  segment_cols: string[];
};

const TEMPLATES: { id: string; label: string; name: string; description: string; config: MetricConfigForm }[] = [
  {
    id: "revenue",
    label: "Revenue experiment",
    name: "Revenue",
    description: "Ecommerce / checkout revenue with CUPED",
    config: {
      primary_metric: "revenue",
      metric_source_col: "revenue_usd",
      metric_agg: "sum",
      covariate: "prior_week_revenue",
      metric_direction: "higher_is_better",
      events_table: "transactions",
      experiment_table: "experiment_assignments",
      user_id_col: "user_id",
      date_col: "transaction_date",
      variant_col: "variant",
      week_col: "experiment_week",
      guardrail_metrics: ["refund_rate", "cart_abandonment_rate"],
      segment_cols: ["platform", "country"],
    },
  },
  {
    id: "dau",
    label: "DAU / engagement",
    name: "DAU",
    description: "Daily active users with retention guardrails",
    config: {
      primary_metric: "dau_rate",
      metric_source_col: "dau_flag",
      metric_agg: "mean",
      covariate: "pre_exp_dau",
      metric_direction: "higher_is_better",
      events_table: "events",
      experiment_table: "experiment",
      user_id_col: "user_id",
      date_col: "date",
      variant_col: "variant",
      week_col: "week",
      guardrail_metrics: ["notif_optout", "d7_retained", "session_count"],
      segment_cols: ["platform", "country"],
    },
  },
  {
    id: "retention",
    label: "Retention",
    name: "D7 Retention",
    description: "D7 retention experiment mapping",
    config: {
      primary_metric: "d7_retained",
      metric_source_col: "d7_retained",
      metric_agg: "mean",
      covariate: "pre_exp_retention",
      metric_direction: "higher_is_better",
      events_table: "events",
      experiment_table: "experiment",
      user_id_col: "user_id",
      date_col: "date",
      variant_col: "variant",
      week_col: "week",
      guardrail_metrics: ["notif_optout_rate", "session_count"],
      segment_cols: ["platform", "country"],
    },
  },
];

const EMPTY: MetricConfigForm = TEMPLATES[0].config;

const FIELDS: { key: keyof MetricConfigForm; label: string; hint?: string }[] = [
  { key: "primary_metric", label: "Primary metric", hint: "e.g. revenue" },
  { key: "metric_source_col", label: "Source column", hint: "Raw DB column" },
  { key: "metric_agg", label: "Aggregation", hint: "mean | sum | count" },
  { key: "covariate", label: "CUPED covariate" },
  { key: "metric_direction", label: "Direction" },
  { key: "events_table", label: "Events table" },
  { key: "experiment_table", label: "Experiment table" },
  { key: "user_id_col", label: "User ID column" },
  { key: "date_col", label: "Date column" },
  { key: "variant_col", label: "Variant column" },
  { key: "week_col", label: "Week column" },
];

interface Props {
  open: boolean;
  onClose: () => void;
  onChanged?: () => void;
  /** When false, studio is read-only (analysts). Auto-detected if omitted. */
  canEdit?: boolean;
}

export default function PackStudio({ open, onClose, onChanged, canEdit }: Props) {
  const [packs, setPacks] = useState<MetricPack[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [form, setForm] = useState<MetricConfigForm>({ ...EMPTY });
  const [guardrails, setGuardrails] = useState(EMPTY.guardrail_metrics.join(", "));
  const [segments, setSegments] = useState(EMPTY.segment_cols.join(", "));
  const [editable, setEditable] = useState(canEdit !== false);

  const load = () => {
    setLoading(true);
    client.get<{ metric_packs: MetricPack[] }>("/metric-packs")
      .then((r) => setPacks(r.data.metric_packs || []))
      .catch(() => setError("Could not load metric packs"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!open) return;
    setError("");
    load();
    resetForm();
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

  const resetForm = (tpl = TEMPLATES[0]) => {
    setEditingId(null);
    setName(tpl.name);
    setDescription(tpl.description);
    setForm({ ...tpl.config });
    setGuardrails(tpl.config.guardrail_metrics.join(", "));
    setSegments(tpl.config.segment_cols.join(", "));
  };

  const startEdit = async (packId: string) => {
    setError("");
    try {
      const { data } = await client.get<MetricPack>(`/metric-packs/${packId}`);
      setEditingId(data.pack_id);
      setName(data.name);
      setDescription(data.description || "");
      setForm({ ...EMPTY, ...data.config });
      setGuardrails((data.config.guardrail_metrics || []).join(", "));
      setSegments((data.config.segment_cols || []).join(", "));
    } catch (err) {
      setError(extractApiError(err, "Could not load pack"));
    }
  };

  const buildConfig = (): MetricConfigForm => ({
    ...form,
    guardrail_metrics: guardrails.split(",").map((s) => s.trim()).filter(Boolean),
    segment_cols: segments.split(",").map((s) => s.trim()).filter(Boolean),
  });

  const save = async (asCertified: boolean) => {
    if (!name.trim()) {
      setError("Name is required");
      return;
    }
    setSaving(true);
    setError("");
    const config = buildConfig();
    try {
      if (editingId) {
        await client.patch(`/metric-packs/${editingId}`, {
          name: name.trim(),
          description: description.trim(),
          config,
          certified: asCertified,
        });
      } else {
        await client.post("/metric-packs", {
          name: name.trim(),
          description: description.trim(),
          config,
          certified: asCertified,
        });
      }
      load();
      onChanged?.();
      resetForm();
    } catch (err) {
      setError(extractApiError(err, "Could not save pack"));
    } finally {
      setSaving(false);
    }
  };

  const remove = async (packId: string) => {
    if (!confirm("Delete this metric pack?")) return;
    try {
      await client.delete(`/metric-packs/${packId}`);
      if (editingId === packId) resetForm();
      load();
      onChanged?.();
    } catch (err) {
      setError(extractApiError(err, "Could not delete pack"));
    }
  };

  if (!open) return null;

  return (
    <div style={s.backdrop} onClick={onClose} role="presentation">
      <div style={s.modal} className="fade-in" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Metric packs">
        <div style={s.header}>
          <div>
            <div style={s.kicker}>Metric packs</div>
            <h2 style={s.title}>Define your metrics once</h2>
            <p style={s.sub}>Certified packs skip the mapping gate and keep analyses consistent for the team.</p>
          </div>
          <button className="dp-btn dp-btn-link" onClick={onClose}>Close</button>
        </div>

        {error && <div style={s.error}>{error}</div>}

        <div style={s.body}>
          <aside style={s.listPane}>
            <div style={s.listHeader}>
              <span style={s.listTitle}>Saved packs</span>
              {editable && (
                <button className="dp-btn dp-btn-ghost" style={s.tinyBtn} onClick={() => resetForm()}>+ New</button>
              )}
            </div>
            {loading && <p style={s.muted}>Loading…</p>}
            {!loading && packs.length === 0 && (
              <p style={s.muted}>No packs yet. Start from a template on the right.</p>
            )}
            {packs.map((p) => (
              <button
                key={p.pack_id}
                style={{ ...s.packRow, ...(editingId === p.pack_id ? s.packRowActive : {}) }}
                onClick={() => startEdit(p.pack_id)}
              >
                <span style={s.packName}>{p.name}</span>
                <span style={s.packMeta}>
                  {p.certified ? "Certified" : "Draft"}
                  {p.version ? ` (v${p.version})` : ""}
                </span>
                {editable && (
                  <span
                    style={s.deleteLink}
                    onClick={(e) => { e.stopPropagation(); remove(p.pack_id); }}
                  >
                    Delete
                  </span>
                )}
              </button>
            ))}
          </aside>

          <section style={s.formPane}>
            {!editable && (
              <p style={s.muted}>You are an analyst in this workspace, so packs are read-only. Ask an owner to certify definitions.</p>
            )}
            {editable && (
              <div style={s.templates}>
                {TEMPLATES.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    style={s.tplBtn}
                    onClick={() => resetForm(t)}
                    disabled={!!editingId}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            )}

            <label style={s.field}>
              <span style={s.label}>Name</span>
              <input style={s.input} value={name} onChange={(e) => setName(e.target.value)} placeholder="Revenue" disabled={!editable} />
            </label>
            <label style={s.field}>
              <span style={s.label}>Description</span>
              <input style={s.input} value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Checkout experiment metrics" disabled={!editable} />
            </label>

            <div style={s.grid}>
              {FIELDS.map(({ key, label, hint }) => (
                <label key={key} style={s.field}>
                  <span style={s.label}>{label}</span>
                  {key === "metric_direction" ? (
                    <select
                      style={s.input}
                      value={form.metric_direction}
                      disabled={!editable}
                      onChange={(e) => setForm((f) => ({
                        ...f,
                        metric_direction: e.target.value as MetricConfigForm["metric_direction"],
                      }))}
                    >
                      <option value="higher_is_better">higher_is_better</option>
                      <option value="lower_is_better">lower_is_better</option>
                    </select>
                  ) : key === "metric_agg" ? (
                    <select
                      style={s.input}
                      value={form.metric_agg}
                      disabled={!editable}
                      onChange={(e) => setForm((f) => ({ ...f, metric_agg: e.target.value }))}
                    >
                      <option value="mean">mean</option>
                      <option value="sum">sum</option>
                      <option value="count">count</option>
                    </select>
                  ) : (
                    <input
                      style={s.input}
                      value={String(form[key] ?? "")}
                      disabled={!editable}
                      onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
                      placeholder={hint}
                    />
                  )}
                </label>
              ))}
              <label style={{ ...s.field, gridColumn: "1 / -1" }}>
                <span style={s.label}>Guardrail metrics (comma-separated)</span>
                <input style={s.input} value={guardrails} disabled={!editable} onChange={(e) => setGuardrails(e.target.value)} />
              </label>
              <label style={{ ...s.field, gridColumn: "1 / -1" }}>
                <span style={s.label}>Segment columns (comma-separated)</span>
                <input style={s.input} value={segments} disabled={!editable} onChange={(e) => setSegments(e.target.value)} />
              </label>
            </div>

            {editable && (
              <>
                <div style={s.actions}>
                  <button
                    className="dp-btn dp-btn-ghost"
                    disabled={saving}
                    onClick={() => save(false)}
                  >
                    {saving ? "Saving…" : editingId ? "Save draft" : "Create draft"}
                  </button>
                  <button
                    className="dp-btn dp-btn-primary"
                    disabled={saving}
                    onClick={() => save(true)}
                    style={{ padding: "10px 18px" }}
                  >
                    {saving ? "Saving…" : editingId ? "Save & certify" : "Create & certify"}
                  </button>
                </div>
                <p style={s.hint}>Certified packs skip the Metric Config Gate for everyone in the workspace.</p>
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  backdrop: { position: "fixed", inset: 0, background: "#000000aa", zIndex: 80, display: "flex", alignItems: "center", justifyContent: "center", padding: 16 },
  modal: { width: "min(960px, 100%)", maxHeight: "90vh", overflow: "auto", background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 16, padding: 20 },
  header: { display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 16 },
  kicker: { color: "var(--dp-accent)", fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" },
  title: { color: "var(--dp-ink)", fontSize: 20, fontWeight: 700, margin: "4px 0" },
  sub: { color: "var(--dp-ink-muted)", fontSize: 13, margin: 0, maxWidth: 520 },
  error: { background: "var(--dp-danger)11", border: "1px solid var(--dp-danger)44", color: "var(--dp-danger)", borderRadius: 8, padding: "10px 12px", marginBottom: 12, fontSize: 13 },
  body: { display: "flex", flexWrap: "wrap", gap: 16 },
  listPane: { flex: "1 1 220px", maxWidth: 280, borderRight: "1px solid var(--dp-line)", paddingRight: 12 },
  listHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  listTitle: { color: "var(--dp-ink-secondary)", fontSize: 12, fontWeight: 700 },
  tinyBtn: { padding: "4px 8px", fontSize: 11 },
  muted: { color: "var(--dp-ink-muted)", fontSize: 12, lineHeight: 1.4 },
  packRow: { width: "100%", textAlign: "left", background: "transparent", border: "1px solid transparent", borderRadius: 8, padding: "10px 10px", cursor: "pointer", marginBottom: 6, display: "grid", gap: 2 },
  packRowActive: { background: "var(--dp-line)55", borderColor: "var(--dp-accent)44" },
  packName: { color: "var(--dp-ink)", fontSize: 13, fontWeight: 600 },
  packMeta: { color: "var(--dp-ink-muted)", fontSize: 11 },
  deleteLink: { color: "var(--dp-danger)", fontSize: 11, marginTop: 4 },
  formPane: { flex: "2 1 360px", display: "flex", flexDirection: "column", gap: 10 },
  templates: { display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 4 },
  tplBtn: { background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", color: "var(--dp-ink-secondary)", borderRadius: 8, padding: "5px 10px", fontSize: 11, cursor: "pointer" },
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 },
  field: { display: "flex", flexDirection: "column", gap: 4 },
  label: { color: "var(--dp-ink-secondary)", fontSize: 11, fontWeight: 600 },
  input: { background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 8, color: "var(--dp-ink)", padding: "8px 10px", fontSize: 13 },
  actions: { display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 8, flexWrap: "wrap" },
  hint: { color: "var(--dp-ink-muted)", fontSize: 11, margin: 0, textAlign: "right" },
};
