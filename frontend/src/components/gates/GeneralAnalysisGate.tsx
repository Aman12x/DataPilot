import { useState } from "react";
import { gateCard, gateTitle, gateTextarea, gateActions, gateBtnApprove } from "./shared";

interface ColumnSummary {
  name: string; dtype: string; non_null: number; null_count: number;
  mean?: number; std?: number; min?: number; median?: number; max?: number;
  n_unique?: number; top_values?: string[];
}
interface DescribeResult { row_count: number; col_count: number; columns: ColumnSummary[]; }
interface CorrelationPair { col_a: string; col_b: string; correlation: number; }
interface CorrelationResult { pairs: CorrelationPair[]; }

interface Props {
  payload: {
    message: string;
    describe_result:    DescribeResult   | null;
    correlation_result: CorrelationResult | null;
  };
  onSubmit:   (value: object) => void;
  submitting?: boolean;
}

function ColCard({ col }: { col: ColumnSummary }) {
  const isNumeric = col.mean !== undefined;
  return (
    <div style={s.colCard}>
      <div style={s.colName}>{col.name} <span style={s.dtype}>{col.dtype}</span></div>
      {col.null_count > 0 && <div style={s.nullBadge}>{col.null_count} nulls</div>}
      {isNumeric ? (
        <div style={s.colStats}>
          <span>min {fmt(col.min)}</span>
          <span>med {fmt(col.median)}</span>
          <span>max {fmt(col.max)}</span>
          <span>σ {fmt(col.std)}</span>
        </div>
      ) : (
        <div style={s.colStats}>
          <span>{col.n_unique} unique</span>
          {col.top_values?.slice(0, 3).map((v, i) => (
            <span key={i} style={s.topVal}>{v}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function fmt(v: number | undefined): string {
  if (v === undefined || v === null) return "—";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toPrecision(4);
}

function corrColor(r: number): string {
  const abs = Math.abs(r);
  if (abs >= 0.7) return r > 0 ? "#a6e3a1" : "#f38ba8";
  if (abs >= 0.4) return "#fab387";
  return "#a6adc8";
}

export default function GeneralAnalysisGate({ payload, onSubmit, submitting }: Props) {
  const [notes, setNotes] = useState("");
  const { describe_result: desc, correlation_result: corr } = payload;

  return (
    <div style={{ ...gateCard, maxWidth: 760 }}>
      <h3 style={gateTitle}>Review Data Summary</h3>
      <p style={{ color: "#a6adc8", marginBottom: 16, fontSize: 14 }}>{payload.message}</p>

      {desc && (
        <>
          <div style={s.meta}>
            <span style={s.metaItem}>{desc.row_count.toLocaleString()} rows</span>
            <span style={s.metaItem}>{desc.col_count} columns</span>
          </div>
          <div style={s.colGrid}>
            {desc.columns.map((col) => <ColCard key={col.name} col={col} />)}
          </div>
        </>
      )}

      {corr && corr.pairs.length > 0 && (
        <>
          <h4 style={s.subTitle}>Top Correlations</h4>
          <div style={s.corrList}>
            {corr.pairs.slice(0, 8).map((p, i) => (
              <div key={i} style={s.corrRow}>
                <span style={s.corrCols}>{p.col_a} × {p.col_b}</span>
                <span style={{ ...s.corrVal, color: corrColor(p.correlation) }}>r = {p.correlation.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </>
      )}

      <textarea
        style={{ ...gateTextarea, marginTop: 8 }}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Optional: add notes or flag data quality issues…"
        rows={3}
        disabled={submitting}
      />
      <div style={gateActions}>
        <button style={gateBtnApprove} onClick={() => onSubmit({ approved: true, notes })} disabled={submitting}>
          {submitting ? "Submitting…" : "Approve & Generate Insights"}
        </button>
        <button style={s.btnReject} onClick={() => onSubmit({ approved: false, notes })} disabled={submitting}
          title="Flag data quality issues or ask for changes">
          Request Changes
        </button>
      </div>
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  meta:      { display: "flex", gap: 16, marginBottom: 16 },
  metaItem:  { background: "#313244", color: "#89b4fa", padding: "4px 12px", borderRadius: 4, fontSize: 13, fontWeight: 600 },
  colGrid:   { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 8, marginBottom: 20 },
  colCard:   { background: "#181825", borderRadius: 6, padding: "10px 12px", border: "1px solid #313244" },
  colName:   { color: "#cdd6f4", fontWeight: 600, fontSize: 13, marginBottom: 4 },
  dtype:     { color: "#585b70", fontWeight: 400, fontSize: 11, marginLeft: 6 },
  nullBadge: { color: "#f38ba8", fontSize: 11, marginBottom: 4 },
  colStats:  { display: "flex", flexWrap: "wrap" as const, gap: 6, fontSize: 11, color: "#a6adc8" },
  topVal:    { background: "#313244", borderRadius: 3, padding: "1px 5px" },
  subTitle:  { color: "#cdd6f4", marginTop: 20, marginBottom: 8, fontSize: 15 },
  corrList:  { background: "#181825", borderRadius: 6, padding: "8px 12px", marginBottom: 16 },
  corrRow:   { display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #313244" },
  corrCols:  { color: "#a6adc8", fontSize: 13 },
  corrVal:   { fontWeight: 700, fontSize: 13 },
  btnReject: { padding: "10px 22px", background: "transparent", color: "#f38ba8", border: "1px solid #f38ba844", borderRadius: 8, cursor: "pointer", fontSize: 14 },
};
