/**
 * ChartCard.tsx — Renders a single ChartSpec using recharts.
 *
 * Supports: bar, bar_horizontal, line, scatter.
 * Shows title, chart, and a 1-sentence plain-language insight below.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ChartSpec {
  chart_type:     string;
  title:          string;
  insight:        string;
  data:           Record<string, unknown>[];
  x_key:          string;
  y_key:          string;
  y_key2?:        string | null;
  color:          string;
  color2?:        string | null;
  error_bar_low?: string | null;
  error_bar_high?: string | null;
  x_label?:       string | null;
  y_label?:       string | null;
}

interface Props {
  spec: ChartSpec;
}

// Tooltip styling on design tokens
const tooltipStyle = {
  background: "var(--dp-surface)",
  border: "1px solid var(--dp-line)",
  borderRadius: 6,
  color: "var(--dp-ink)",
  fontSize: 12,
  boxShadow: "var(--dp-shadow)",
};

// Parse **bold** markdown in insight text into <strong> tags
function InsightText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <p style={css.insight}>
      {parts.map((part, i) =>
        part.startsWith("**") && part.endsWith("**")
          ? <strong key={i}>{part.slice(2, -2)}</strong>
          : part,
      )}
    </p>
  );
}

export default function ChartCard({ spec }: Props) {
  const { chart_type, title, insight, data, x_key, y_key, y_key2, color, color2,
          error_bar_low, error_bar_high, x_label, y_label } = spec;

  const hasErrorBar = !!(error_bar_low && error_bar_high);

  function renderChart() {
    // ── Horizontal bar ──────────────────────────────────────────────────────
    if (chart_type === "bar_horizontal") {
      return (
        <ResponsiveContainer width="100%" height={Math.max(180, data.length * 38)}>
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--dp-line)" horizontal={false} />
            <XAxis
              type="number"
              dataKey={x_key}
              tick={{ fill: "var(--dp-ink-muted)", fontSize: 11 }}
              axisLine={{ stroke: "var(--dp-line)" }}
              tickLine={false}
              label={x_label ? { value: x_label, position: "insideBottom", offset: -2, fill: "var(--dp-ink-faint)", fontSize: 11 } : undefined}
            />
            <YAxis
              type="category"
              dataKey={y_key}
              width={140}
              tick={{ fill: "var(--dp-ink-secondary)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--dp-line)" }} />
            <Bar dataKey={x_key} fill={color} radius={[0, 4, 4, 0]}>
              {hasErrorBar && (
                <ErrorBar dataKey={error_bar_low!} width={4} strokeWidth={1.5} stroke={color} direction="x" />
              )}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      );
    }

    // ── Line chart ──────────────────────────────────────────────────────────
    if (chart_type === "line") {
      return (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ left: 10, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--dp-line)" />
            <XAxis
              dataKey={x_key}
              tick={{ fill: "var(--dp-ink-muted)", fontSize: 11 }}
              axisLine={{ stroke: "var(--dp-line)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "var(--dp-ink-muted)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              label={y_label ? { value: y_label, angle: -90, position: "insideLeft", fill: "var(--dp-ink-faint)", fontSize: 11 } : undefined}
            />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey={y_key} stroke={color} strokeWidth={2} dot={{ r: 3, fill: color }} />
            {y_key2 && <Line type="monotone" dataKey={y_key2} stroke={color2 ?? "#B45309"} strokeWidth={2} dot={{ r: 3, fill: color2 ?? "#B45309" }} />}
          </LineChart>
        </ResponsiveContainer>
      );
    }

    // ── Scatter chart ────────────────────────────────────────────────────────
    if (chart_type === "scatter") {
      return (
        <ResponsiveContainer width="100%" height={200}>
          <ScatterChart margin={{ left: 10, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--dp-line)" />
            <XAxis dataKey={x_key} tick={{ fill: "var(--dp-ink-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--dp-line)" }} tickLine={false}
              label={x_label ? { value: x_label, position: "insideBottom", offset: -2, fill: "var(--dp-ink-faint)", fontSize: 11 } : undefined}
            />
            <YAxis dataKey={y_key} tick={{ fill: "var(--dp-ink-muted)", fontSize: 11 }} axisLine={false} tickLine={false}
              label={y_label ? { value: y_label, angle: -90, position: "insideLeft", fill: "var(--dp-ink-faint)", fontSize: 11 } : undefined}
            />
            <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3", stroke: "var(--dp-ink-faint)" }} />
            <Scatter data={data} fill={color} />
          </ScatterChart>
        </ResponsiveContainer>
      );
    }

    // ── Default: vertical bar (single or grouped) ────────────────────────────
    const isGrouped = !!y_key2;
    return (
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ left: 10, right: 16, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--dp-line)" vertical={false} />
          <XAxis
            dataKey={x_key}
            tick={{ fill: "var(--dp-ink-muted)", fontSize: 11 }}
            axisLine={{ stroke: "var(--dp-line)" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "var(--dp-ink-muted)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            label={y_label ? { value: y_label, angle: -90, position: "insideLeft", fill: "var(--dp-ink-faint)", fontSize: 10 } : undefined}
          />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--dp-line)" }} />
          {isGrouped && <Legend wrapperStyle={{ fontSize: 11, color: "var(--dp-ink-secondary)" }} />}
          <Bar dataKey={y_key} fill={color} radius={[4, 4, 0, 0]} maxBarSize={32}>
            {hasErrorBar && (
              <ErrorBar
                dataKey={(d: Record<string, number>) => [d[error_bar_low!], d[error_bar_high!]] as [number, number]}
                width={4}
                strokeWidth={1.5}
                stroke={color}
              />
            )}
            {!isGrouped && data.length <= 6 && (
              <LabelList dataKey={y_key} position="top" style={{ fill: "var(--dp-ink-muted)", fontSize: 10 }} />
            )}
          </Bar>
          {isGrouped && y_key2 && (
            <Bar dataKey={y_key2} fill={color2 ?? "#B45309"} radius={[4, 4, 0, 0]} maxBarSize={32} />
          )}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <div style={css.card}>
      <p style={css.title}>{title}</p>
      <div style={css.chartWrap}>{renderChart()}</div>
      <InsightText text={insight} />
    </div>
  );
}

const css: Record<string, React.CSSProperties> = {
  card:      { background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 8, padding: "18px 18px 14px", display: "flex", flexDirection: "column", gap: 10 },
  title:     { color: "var(--dp-ink)", fontSize: 13, fontWeight: 600, margin: 0 },
  chartWrap: { width: "100%", overflow: "hidden" },
  insight:   { color: "var(--dp-ink-secondary)", fontSize: 12, lineHeight: 1.6, margin: 0, borderTop: "1px solid var(--dp-line)", paddingTop: 10 },
};
