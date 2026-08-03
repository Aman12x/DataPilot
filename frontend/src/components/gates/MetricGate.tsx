import { useState } from "react";
import { gateCard, gateTitle, gateMessage, gateActions, gateBtnApprove, gateBtnSecondary } from "./shared";

type MetricConfigForm = {
  primary_metric: string;
  metric_source_col: string;
  metric_agg: string;
  covariate: string;
  metric_direction: string;
  events_table: string;
  experiment_table: string;
  user_id_col: string;
  date_col: string;
  variant_col: string;
  guardrail_metrics: string[];
  segment_cols: string[];
};

interface Props {
  payload: {
    metric_config: MetricConfigForm;
    metric_pack_id?: string;
    source?: string;
    message: string;
    schema_drift_warnings?: string[];
  };
  onSubmit: (value: object) => void;
  submitting?: boolean;
}

const FIELDS: { key: keyof MetricConfigForm; label: string; help: string }[] = [
  { key: "primary_metric", label: "Metric name", help: "What you're measuring — appears in reports." },
  { key: "metric_source_col", label: "Value column", help: "Database column holding the raw numbers." },
  { key: "covariate", label: "Pre-experiment covariate (optional)", help: "Same metric from before the experiment; cancels out noise (CUPED)." },
  { key: "events_table", label: "Events table", help: "One row per event or transaction." },
  { key: "experiment_table", label: "Experiment table", help: "Which user got which variant." },
  { key: "user_id_col", label: "User ID column", help: "Identifies the user in both tables." },
  { key: "date_col", label: "Date column", help: "When each event happened." },
  { key: "variant_col", label: "Variant column", help: "Group labels, e.g. control / treatment." },
];

const AGG_PHRASE: Record<string, string> = {
  mean: "averaged per user",
  sum: "totalled per user",
  count: "counted per user",
};

export default function MetricGate({ payload, onSubmit, submitting }: Props) {
  const [form, setForm] = useState<MetricConfigForm>(() => ({
    ...payload.metric_config,
    guardrail_metrics: [...(payload.metric_config.guardrail_metrics || [])],
    segment_cols: [...(payload.metric_config.segment_cols || [])],
  }));
  const [guardrails, setGuardrails] = useState(
    (payload.metric_config.guardrail_metrics || []).join(", "),
  );
  const [segments, setSegments] = useState(
    (payload.metric_config.segment_cols || []).join(", "),
  );
  const [editing, setEditing] = useState(false);

  const set = (key: keyof MetricConfigForm) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [key]: e.target.value }));

  const approve = () => {
    const config = {
      ...form,
      guardrail_metrics: guardrails.split(",").map((s) => s.trim()).filter(Boolean),
      segment_cols: segments.split(",").map((s) => s.trim()).filter(Boolean),
    };
    onSubmit({ approved: true, metric_config: config });
  };

  const aggPhrase = AGG_PHRASE[form.metric_agg] || form.metric_agg;
  const directionPhrase = form.metric_direction === "lower_is_better" ? "lower is better" : "higher is better";
  const guardrailList = guardrails.split(",").map((g) => g.trim()).filter(Boolean);

  return (
    <div style={{ ...gateCard, maxWidth: 640 }}>
      <div style={s.tag}>Metric mapping</div>
      <h3 style={gateTitle}>Confirm how your data is defined</h3>
      <p style={{ ...gateMessage, marginBottom: 8 }}>{payload.message}</p>
      <p style={s.meta}>
        Source: <strong>{payload.source === "pack" ? "your saved metric pack" : "detected from your data"}</strong>
      </p>

      {(payload.schema_drift_warnings?.length ?? 0) > 0 && (
        <div style={s.driftBox}>
          <div style={s.driftTitle}>Heads up — your database has changed</div>
          <ul style={s.driftList}>
            {payload.schema_drift_warnings!.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div style={s.summaryBox}>
        <p style={s.summaryLine}>
          Measuring <strong>{form.primary_metric || "—"}</strong>: the{" "}
          <code style={s.code}>{form.metric_source_col || "—"}</code> column, {aggPhrase},
          where {directionPhrase}.
        </p>
        <p style={s.summaryLine}>
          Comparing groups from <code style={s.code}>{form.variant_col || "—"}</code> in{" "}
          <code style={s.code}>{form.experiment_table || "—"}</code>, matched to events in{" "}
          <code style={s.code}>{form.events_table || "—"}</code> by{" "}
          <code style={s.code}>{form.user_id_col || "—"}</code>.
        </p>
        {guardrailList.length > 0 && (
          <p style={s.summaryLine}>
            Also watching that {guardrailList.map((g, i) => (
              <span key={g}>
                {i > 0 && (i === guardrailList.length - 1 ? " and " : ", ")}
                <code style={s.code}>{g}</code>
              </span>
            ))} don't get worse.
          </p>
        )}
      </div>

      <button
        type="button"
        style={s.editToggle}
        onClick={() => setEditing((v) => !v)}
        aria-expanded={editing}
      >
        {editing ? "▾ Adjust mapping" : "▸ Adjust mapping"}
        {!editing && <span style={s.editSub}> — change table or column names if the summary looks wrong</span>}
      </button>

      {editing && (
        <div style={s.grid}>
          {FIELDS.map(({ key, label, help }) => (
            <label key={key} style={s.field}>
              <span style={s.label}>{label}</span>
              <input
                style={s.input}
                value={String(form[key] ?? "")}
                onChange={set(key)}
                disabled={submitting}
              />
              <span style={s.help}>{help}</span>
            </label>
          ))}
          <label style={s.field}>
            <span style={s.label}>How to combine values</span>
            <select
              style={s.input}
              value={form.metric_agg}
              disabled={submitting}
              onChange={(e) => setForm((f) => ({ ...f, metric_agg: e.target.value }))}
            >
              <option value="mean">Average per user (rates, yes/no flags)</option>
              <option value="sum">Total (amounts like revenue)</option>
              <option value="count">Count of events</option>
            </select>
            <span style={s.help}>How each user's rows roll up into one number.</span>
          </label>
          <label style={s.field}>
            <span style={s.label}>What does success look like?</span>
            <select
              style={s.input}
              value={form.metric_direction}
              disabled={submitting}
              onChange={(e) => setForm((f) => ({ ...f, metric_direction: e.target.value }))}
            >
              <option value="higher_is_better">Higher is better (revenue, retention…)</option>
              <option value="lower_is_better">Lower is better (churn, refunds…)</option>
            </select>
            <span style={s.help}>Which direction counts as a win.</span>
          </label>
          <label style={{ ...s.field, gridColumn: "1 / -1" }}>
            <span style={s.label}>Guardrail metrics</span>
            <input
              style={s.input}
              value={guardrails}
              onChange={(e) => setGuardrails(e.target.value)}
              disabled={submitting}
            />
            <span style={s.help}>Metrics that must not get worse. Separate with commas.</span>
          </label>
          <label style={{ ...s.field, gridColumn: "1 / -1" }}>
            <span style={s.label}>Segment columns</span>
            <input
              style={s.input}
              value={segments}
              onChange={(e) => setSegments(e.target.value)}
              disabled={submitting}
            />
            <span style={s.help}>Break results down by these, e.g. platform or country. Separate with commas.</span>
          </label>
        </div>
      )}

      <div style={gateActions}>
        <button style={gateBtnSecondary} disabled={submitting} onClick={() => onSubmit({ approved: false })}>
          Re-detect from data
        </button>
        <button style={gateBtnApprove} disabled={submitting} onClick={approve}>
          {submitting ? "Submitting…" : "Confirm mapping"}
        </button>
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  tag:  { fontSize: 11, fontWeight: 700, color: "var(--dp-success)", background: "var(--dp-success)11", border: "1px solid var(--dp-success)33", borderRadius: 20, padding: "3px 10px", display: "inline-block", marginBottom: 10 },
  meta: { color: "var(--dp-ink-muted)", fontSize: 12, marginBottom: 16 },
  driftBox: { background: "var(--dp-warning)11", border: "1px solid var(--dp-warning)44", borderRadius: 10, padding: "10px 14px", marginBottom: 16 },
  driftTitle: { color: "var(--dp-warning)", fontSize: 12, fontWeight: 700, marginBottom: 6 },
  driftList: { margin: 0, paddingLeft: 18, color: "#bac2de", fontSize: 12, lineHeight: 1.5 },
  summaryBox: { background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", borderRadius: 10, padding: "12px 14px", marginBottom: 10 },
  summaryLine: { color: "var(--dp-ink-secondary)", fontSize: 13, lineHeight: 1.6, margin: "0 0 6px" },
  code: { background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 4, padding: "1px 5px", fontSize: 12, fontFamily: "ui-monospace, monospace", color: "var(--dp-ink)" },
  editToggle: { background: "transparent", border: "none", color: "var(--dp-ink-secondary)", fontSize: 12, fontWeight: 600, textAlign: "left", cursor: "pointer", padding: "4px 0", marginBottom: 8 },
  editSub: { color: "var(--dp-ink-muted)", fontWeight: 400 },
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 18 },
  field:{ display: "flex", flexDirection: "column", gap: 4 },
  label:{ color: "var(--dp-ink-secondary)", fontSize: 11, fontWeight: 600 },
  input:{ background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 8, color: "var(--dp-ink)", padding: "8px 10px", fontSize: 13 },
  help: { color: "var(--dp-ink-muted)", fontSize: 11, lineHeight: 1.4 },
};
