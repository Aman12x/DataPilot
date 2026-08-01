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

const FIELDS: { key: keyof MetricConfigForm; label: string; hint?: string }[] = [
  { key: "primary_metric", label: "Primary metric alias", hint: "e.g. dau_rate, revenue" },
  { key: "metric_source_col", label: "Source column", hint: "Raw column in the events table" },
  { key: "metric_agg", label: "Aggregation", hint: "mean | sum | count" },
  { key: "covariate", label: "CUPED covariate" },
  { key: "metric_direction", label: "Direction", hint: "higher_is_better | lower_is_better" },
  { key: "events_table", label: "Events table" },
  { key: "experiment_table", label: "Experiment table" },
  { key: "user_id_col", label: "User ID column" },
  { key: "date_col", label: "Date column" },
  { key: "variant_col", label: "Variant column" },
];

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

  return (
    <div style={{ ...gateCard, maxWidth: 640 }}>
      <div style={s.tag}>Metric mapping</div>
      <h3 style={gateTitle}>Confirm how your data is defined</h3>
      <p style={{ ...gateMessage, marginBottom: 8 }}>{payload.message}</p>
      <p style={s.meta}>
        Source: <strong>{payload.source === "pack" ? "Metric pack" : "Inferred from schema"}</strong>
        {payload.metric_pack_id ? ` · pack ${payload.metric_pack_id.slice(0, 8)}…` : ""}
      </p>

      {(payload.schema_drift_warnings?.length ?? 0) > 0 && (
        <div style={s.driftBox}>
          <div style={s.driftTitle}>Schema drift</div>
          <ul style={s.driftList}>
            {payload.schema_drift_warnings!.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div style={s.grid}>
        {FIELDS.map(({ key, label, hint }) => (
          <label key={key} style={s.field}>
            <span style={s.label}>{label}</span>
            <input
              style={s.input}
              value={String(form[key] ?? "")}
              onChange={set(key)}
              placeholder={hint}
              disabled={submitting}
            />
          </label>
        ))}
        <label style={{ ...s.field, gridColumn: "1 / -1" }}>
          <span style={s.label}>Guardrail metrics (comma-separated)</span>
          <input
            style={s.input}
            value={guardrails}
            onChange={(e) => setGuardrails(e.target.value)}
            disabled={submitting}
          />
        </label>
        <label style={{ ...s.field, gridColumn: "1 / -1" }}>
          <span style={s.label}>Segment columns (comma-separated)</span>
          <input
            style={s.input}
            value={segments}
            onChange={(e) => setSegments(e.target.value)}
            disabled={submitting}
          />
        </label>
      </div>

      <div style={gateActions}>
        <button style={gateBtnSecondary} disabled={submitting} onClick={() => onSubmit({ approved: false })}>
          Re-check
        </button>
        <button style={gateBtnApprove} disabled={submitting} onClick={approve}>
          {submitting ? "Submitting…" : "Confirm mapping →"}
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
  grid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 18 },
  field:{ display: "flex", flexDirection: "column", gap: 4 },
  label:{ color: "var(--dp-ink-secondary)", fontSize: 11, fontWeight: 600 },
  input:{ background: "var(--dp-bg)", border: "1px solid var(--dp-line)", borderRadius: 8, color: "var(--dp-ink)", padding: "8px 10px", fontSize: 13 },
};
