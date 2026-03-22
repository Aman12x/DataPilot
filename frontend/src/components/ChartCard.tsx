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

// Very light tooltip styling that matches the dark theme
const tooltipStyle = {
  background: "#1e1e2e",
  border: "1px solid #313244",
  borderRadius: 6,
  color: "#cdd6f4",
  fontSize: 12,
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
            <CartesianGrid strokeDasharray="3 3" stroke="#313244" horizontal={false} />
            <XAxis
              type="number"
              dataKey={x_key}
              tick={{ fill: "#585b70", fontSize: 11 }}
              axisLine={{ stroke: "#313244" }}
              tickLine={false}
              label={x_label ? { value: x_label, position: "insideBottom", offset: -2, fill: "#45475a", fontSize: 11 } : undefined}
            />
            <YAxis
              type="category"
              dataKey={y_key}
              width={140}
              tick={{ fill: "#a6adc8", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#313244" }} />
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
            <CartesianGrid strokeDasharray="3 3" stroke="#313244" />
            <XAxis
              dataKey={x_key}
              tick={{ fill: "#585b70", fontSize: 11 }}
              axisLine={{ stroke: "#313244" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#585b70", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              label={y_label ? { value: y_label, angle: -90, position: "insideLeft", fill: "#45475a", fontSize: 11 } : undefined}
            />
            <Tooltip contentStyle={tooltipStyle} />
            <Line type="monotone" dataKey={y_key} stroke={color} strokeWidth={2} dot={{ r: 3, fill: color }} />
            {y_key2 && <Line type="monotone" dataKey={y_key2} stroke={color2 ?? "#cba6f7"} strokeWidth={2} dot={{ r: 3, fill: color2 ?? "#cba6f7" }} />}
          </LineChart>
        </ResponsiveContainer>
      );
    }

    // ── Scatter chart ────────────────────────────────────────────────────────
    if (chart_type === "scatter") {
      return (
        <ResponsiveContainer width="100%" height={200}>
          <ScatterChart margin={{ left: 10, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#313244" />
            <XAxis dataKey={x_key} tick={{ fill: "#585b70", fontSize: 11 }} axisLine={{ stroke: "#313244" }} tickLine={false}
              label={x_label ? { value: x_label, position: "insideBottom", offset: -2, fill: "#45475a", fontSize: 11 } : undefined}
            />
            <YAxis dataKey={y_key} tick={{ fill: "#585b70", fontSize: 11 }} axisLine={false} tickLine={false}
              label={y_label ? { value: y_label, angle: -90, position: "insideLeft", fill: "#45475a", fontSize: 11 } : undefined}
            />
            <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: "3 3", stroke: "#45475a" }} />
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
          <CartesianGrid strokeDasharray="3 3" stroke="#313244" vertical={false} />
          <XAxis
            dataKey={x_key}
            tick={{ fill: "#585b70", fontSize: 11 }}
            axisLine={{ stroke: "#313244" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#585b70", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            label={y_label ? { value: y_label, angle: -90, position: "insideLeft", fill: "#45475a", fontSize: 10 } : undefined}
          />
          <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#313244" }} />
          {isGrouped && <Legend wrapperStyle={{ fontSize: 11, color: "#a6adc8" }} />}
          <Bar dataKey={y_key} fill={color} radius={[4, 4, 0, 0]} maxBarSize={48}>
            {hasErrorBar && (
              <ErrorBar
                dataKey={(d: Record<string, number>) => [d[error_bar_low!], d[error_bar_high!]] as [number, number]}
                width={4}
                strokeWidth={1.5}
                stroke={color}
              />
            )}
            {!isGrouped && data.length <= 6 && (
              <LabelList dataKey={y_key} position="top" style={{ fill: "#585b70", fontSize: 10 }} />
            )}
          </Bar>
          {isGrouped && y_key2 && (
            <Bar dataKey={y_key2} fill={color2 ?? "#cba6f7"} radius={[4, 4, 0, 0]} maxBarSize={48} />
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
  card:      { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 12, padding: "18px 18px 14px", display: "flex", flexDirection: "column", gap: 10 },
  title:     { color: "#cdd6f4", fontSize: 13, fontWeight: 700, margin: 0 },
  chartWrap: { width: "100%", overflow: "hidden" },
  insight:   { color: "#a6adc8", fontSize: 12, lineHeight: 1.6, margin: 0, borderTop: "1px solid #313244", paddingTop: 10 },
};
