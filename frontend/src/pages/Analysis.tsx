import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import client, { API_BASE, uploadFile, type UploadResult, logout } from "../api/client";
import { useSSE, type DoneEvent, type TrustIndicators, type PowerAnalysisResult, type SensitivityRow } from "../hooks/useSSE";
import { useTokenRefresh } from "../hooks/useTokenRefresh";
import PipelineProgress from "../components/PipelineProgress";
import Markdown from "../components/Markdown";
import ChartCard, { type ChartSpec } from "../components/ChartCard";
import IntentGate from "../components/gates/IntentGate";
import SemanticCacheGate from "../components/gates/SemanticCacheGate";
import QueryGate from "../components/gates/QueryGate";
import AnalysisGate from "../components/gates/AnalysisGate";
import GeneralAnalysisGate from "../components/gates/GeneralAnalysisGate";
import NarrativeGate from "../components/gates/NarrativeGate";
import { type Mode, type PgCreds, type Sample, MODE_META } from "../types/analysis";
import { stripMarkdown, sanitiseNarrative } from "../utils/markdown";
import { extractApiError } from "../utils/error";

// ── ModeSelect — the two-button landing ───────────────────────────────────────

function ModeSelect({ onSelect, username, onHistory, onSignOut }: {
  onSelect: (mode: Mode) => void;
  username: string;
  onHistory: () => void;
  onSignOut: () => void;
}) {
  const [showAbSub, setShowAbSub] = useState(false);

  return (
    <div style={ms.page}>
      <div style={ms.orb1} /><div style={ms.orb2} />

      <div style={ms.topBar}>
        <span style={ms.logo}>✦ DataPilot</span>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {username && <span style={ms.username}>{username}</span>}
          <button style={ms.navBtn} onClick={onHistory}>History</button>
          <button style={ms.signOutBtn} onClick={onSignOut}>Sign out</button>
        </div>
      </div>

      <div style={ms.hero} className="fade-in">
        <h1 style={ms.heroTitle}>What would you like to do?</h1>
        <p style={ms.heroSub}>Choose your analysis type to get started</p>
      </div>

      <div style={ms.cards} className="slide-up">

        <button style={{ ...ms.card, ...ms.cardGeneral }} onClick={() => onSelect("general")}>
          <div style={ms.cardIcon}>💡</div>
          <div style={ms.cardTitle}>Explore & Understand</div>
          <div style={ms.cardDesc}>Find patterns, trends, and insights in any dataset</div>
          <ul style={ms.featureList}>
            <li style={ms.feature}><span style={{ color: "#cba6f7" }}>✓</span> Descriptive statistics</li>
            <li style={ms.feature}><span style={{ color: "#cba6f7" }}>✓</span> Correlation analysis</li>
            <li style={ms.feature}><span style={{ color: "#cba6f7" }}>✓</span> Trend &amp; anomaly detection</li>
            <li style={ms.feature}><span style={{ color: "#cba6f7" }}>✓</span> Driver &amp; segment analysis</li>
          </ul>
          <div style={ms.useCases}>Health data · SaaS metrics · Ops logs · Finance</div>
          <div style={{ ...ms.cta, background: "#cba6f7", color: "#1e1e2e" }}>Start Exploring →</div>
        </button>

        {!showAbSub ? (
          <button style={{ ...ms.card, ...ms.cardAB }} onClick={() => setShowAbSub(true)}>
            <div style={ms.cardIcon}>🧪</div>
            <div style={ms.cardTitle}>A/B Testing</div>
            <div style={ms.cardDesc}>Design experiments or analyse results from running tests</div>
            <ul style={ms.featureList}>
              <li style={ms.feature}><span style={{ color: "#89b4fa" }}>✓</span> Sample size &amp; power calculation</li>
              <li style={ms.feature}><span style={{ color: "#89b4fa" }}>✓</span> t-test, CUPED &amp; HTE</li>
              <li style={ms.feature}><span style={{ color: "#89b4fa" }}>✓</span> Guardrails &amp; novelty check</li>
              <li style={ms.feature}><span style={{ color: "#89b4fa" }}>✓</span> Segment sensitivity analysis</li>
            </ul>
            <div style={ms.useCases}>Product · Clinical trials · Marketing · Pricing</div>
            <div style={{ ...ms.cta, background: "#89b4fa", color: "#1e1e2e" }}>Select →</div>
          </button>
        ) : (
          <div style={{ ...ms.card, ...ms.cardAB, cursor: "default" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <button style={ms.backSubBtn} onClick={() => setShowAbSub(false)}>← Back</button>
              <span style={{ color: "#89b4fa", fontWeight: 700, fontSize: 15 }}>🧪 A/B Testing</span>
            </div>
            <p style={{ color: "#585b70", fontSize: 13, margin: "4px 0 16px" }}>Choose what you want to do:</p>

            <button style={ms.subCard} onClick={() => onSelect("power_analysis")}>
              <div style={{ fontSize: 20 }}>📐</div>
              <div>
                <div style={{ color: "#cdd6f4", fontWeight: 700, fontSize: 14 }}>Design Experiment</div>
                <div style={{ color: "#585b70", fontSize: 12, marginTop: 2 }}>Sample size, runtime &amp; sensitivity table</div>
              </div>
              <span style={{ color: "#89b4fa", marginLeft: "auto", fontSize: 16 }}>→</span>
            </button>

            <button style={{ ...ms.subCard, marginTop: 8 }} onClick={() => onSelect("ab_test")}>
              <div style={{ fontSize: 20 }}>📊</div>
              <div>
                <div style={{ color: "#cdd6f4", fontWeight: 700, fontSize: 14 }}>Interpret Results</div>
                <div style={{ color: "#585b70", fontSize: 12, marginTop: 2 }}>Analyse a completed experiment end-to-end</div>
              </div>
              <span style={{ color: "#89b4fa", marginLeft: "auto", fontSize: 16 }}>→</span>
            </button>
          </div>
        )}
      </div>

      <p style={ms.footer}>Not sure? Start with Explore — DataPilot will figure out the right analysis.</p>
    </div>
  );
}

const ms: Record<string, React.CSSProperties> = {
  page:       { minHeight: "100vh", background: "#11111b", padding: "0 20px 60px", position: "relative", overflow: "hidden" },
  orb1:       { position: "fixed", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, #cba6f708 0%, transparent 70%)", top: -200, left: -200, pointerEvents: "none" },
  orb2:       { position: "fixed", width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, #89b4fa08 0%, transparent 70%)", bottom: -150, right: -100, pointerEvents: "none" },
  topBar:     { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 4px", maxWidth: 760, margin: "0 auto" },
  logo:       { color: "#89b4fa", fontWeight: 700, fontSize: 18, letterSpacing: "-0.3px" },
  username:   { color: "#45475a", fontSize: 13 },
  navBtn:     { background: "transparent", border: "1px solid #313244", color: "#a6adc8", padding: "5px 12px", borderRadius: 6, cursor: "pointer", fontSize: 12 },
  signOutBtn: { background: "transparent", border: "none", color: "#45475a", fontSize: 12, cursor: "pointer" },
  hero:       { textAlign: "center", padding: "48px 0 36px", maxWidth: 560, margin: "0 auto" },
  heroTitle:  { color: "#cdd6f4", fontSize: 30, fontWeight: 700, marginBottom: 10, letterSpacing: "-0.5px" },
  heroSub:    { color: "#585b70", fontSize: 15 },
  cards:      { display: "flex", gap: 20, maxWidth: 760, margin: "0 auto", flexWrap: "wrap" as const, justifyContent: "center" },
  card:       { flex: "1 1 320px", maxWidth: 360, background: "#1e1e2e", border: "1px solid #313244", borderRadius: 16, padding: "28px 28px 24px", cursor: "pointer", textAlign: "left" as const, transition: "transform 0.15s, border-color 0.15s, box-shadow 0.15s", display: "flex", flexDirection: "column" as const, gap: 12 },
  cardGeneral:{ borderColor: "#cba6f733" },
  cardAB:     { borderColor: "#89b4fa33" },
  cardIcon:   { fontSize: 32 },
  cardTitle:  { color: "#cdd6f4", fontSize: 20, fontWeight: 700 },
  cardDesc:   { color: "#585b70", fontSize: 14, lineHeight: 1.5 },
  featureList:{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column" as const, gap: 6 },
  feature:    { color: "#a6adc8", fontSize: 13, display: "flex", alignItems: "center", gap: 8 },
  useCases:   { color: "#45475a", fontSize: 11, borderTop: "1px solid #313244", paddingTop: 12, marginTop: 4 },
  cta:        { borderRadius: 8, padding: "10px 0", fontWeight: 700, fontSize: 14, textAlign: "center" as const, marginTop: 4 },
  footer:     { textAlign: "center", color: "#45475a", fontSize: 12, marginTop: 28, maxWidth: 500, margin: "28px auto 0" },
  subCard:    { display: "flex", alignItems: "center", gap: 12, padding: "12px 14px", background: "#181825", border: "1px solid #313244", borderRadius: 10, cursor: "pointer", textAlign: "left" as const, width: "100%", transition: "border-color 0.15s" },
  backSubBtn: { background: "transparent", border: "none", color: "#45475a", fontSize: 12, cursor: "pointer", padding: 0 },
};

// ── TaskInput ──────────────────────────────────────────────────────────────────

function TaskInput({ mode, onSubmit, onBack, startError }: {
  mode: Mode;
  onSubmit: (task: string, db: string, pg?: PgCreds, uploadId?: string) => void;
  onBack: () => void;
  startError?: string;
}) {
  const meta = MODE_META[mode];
  const [task,           setTask]           = useState("");
  const [db,             setDb]             = useState("duckdb");
  const [pg,             setPg]             = useState<PgCreds>({ host: "localhost", port: "5432", dbname: "", user: "", password: "" });
  const [useUpload,      setUseUpload]      = useState(false);
  const [uploading,      setUploading]      = useState(false);
  const [uploadResult,   setUploadResult]   = useState<UploadResult | null>(null);
  const [uploadError,    setUploadError]    = useState("");
  const [uploadFileName, setUploadFileName] = useState("");
  const [samples,        setSamples]        = useState<Sample[]>([]);
  const [loadingSample,  setLoadingSample]  = useState<string | null>(null);
  const [submitting,     setSubmitting]     = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    client.get<Sample[]>("/samples")
      .then(r => setSamples(r.data.filter(s => s.mode === mode)))
      .catch(() => {});
  }, [mode]);

  const setP = (k: keyof PgCreds) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setPg(p => ({ ...p, [k]: e.target.value }));

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true); setUploadError(""); setUploadResult(null); setUploadFileName(file.name);
    try {
      setUploadResult(await uploadFile(file));
    } catch (err) {
      setUploadError(extractApiError(err, "Upload failed."));
      setUploadFileName("");
    } finally { setUploading(false); }
  };

  const loadSample = async (sample: Sample) => {
    setLoadingSample(sample.name);
    setUploadError("");
    try {
      const res  = await fetch(`${API_BASE}/samples/${sample.name}`);
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      const file = new File([blob], sample.name, { type: "text/csv" });
      const result = await uploadFile(file);
      setUploadResult(result);
      setUploadFileName(sample.name);
      setUseUpload(true);
      setTask(sample.suggested_task);
    } catch {
      setUploadError("Could not load sample. Try again.");
    } finally { setLoadingSample(null); }
  };

  const pgValid     = db !== "postgres" || (pg.host && pg.dbname && pg.user);
  const uploadReady = !useUpload || !!uploadResult;
  const canSubmit   = task.trim() && pgValid && uploadReady && !uploading && !loadingSample && !submitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      if (useUpload && uploadResult) await onSubmit(task, "duckdb", undefined, uploadResult.upload_id);
      else await onSubmit(task, db, db === "postgres" ? pg : undefined);
    } finally { setSubmitting(false); }
  };

  return (
    <div style={s.page} className="fade-in">
      <div style={s.header}>
        <button style={s.backBtn} onClick={onBack}>← Back</button>
        <span style={{ ...s.modeBadge, borderColor: meta.accent + "44", color: meta.accent, background: meta.accent + "11" }}>
          {meta.badge}
        </span>
      </div>

      <div style={s.inputCard} className="slide-up">
        <h2 style={s.heading}>{meta.heading}</h2>
        <p style={s.sub}>{meta.sub}</p>

        {samples.length > 0 && (
          <div style={s.samplesSection}>
            <div style={s.sectionLabel}>Try a sample dataset</div>
            <div style={s.samplesGrid} className="samples-grid">
              {samples.map(s2 => (
                <button
                  key={s2.name}
                  style={{ ...s.sampleCard, ...(loadingSample === s2.name ? s.sampleLoading : {}), borderColor: meta.accent + "33" }}
                  onClick={() => loadSample(s2)}
                  disabled={!!loadingSample}
                  type="button"
                >
                  <span style={s.sampleIcon}>{s2.icon}</span>
                  <div>
                    <div style={s.sampleLabel}>{s2.label}</div>
                    <div style={s.sampleDomain}>{s2.domain}</div>
                  </div>
                  {loadingSample === s2.name && <span style={s.sampleSpinner} />}
                </button>
              ))}
            </div>
          </div>
        )}

        <textarea
          style={s.taskInput}
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder={meta.placeholder}
          rows={4}
          autoFocus
        />

        <div style={s.section}>
          <div style={s.sectionLabel}>Data source</div>
          <div style={s.sourceRow}>
            <label style={s.sourceOption}>
              <input type="radio" checked={!useUpload} onChange={() => { setUseUpload(false); setUploadResult(null); setUploadFileName(""); }} />
              <span>Database</span>
            </label>
            <label style={s.sourceOption}>
              <input type="radio" checked={useUpload} onChange={() => setUseUpload(true)} />
              <span>Upload CSV / Excel</span>
            </label>
          </div>

          {useUpload ? (
            <div style={s.uploadBox}>
              <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" style={{ display: "none" }} onChange={handleFileChange} />
              <button style={s.uploadBtn} onClick={() => fileRef.current?.click()} disabled={uploading || !!loadingSample}>
                {uploading ? "⏳ Uploading…" : "📂 Choose file"}
              </button>
              {uploadError && <p style={{ color: "#f38ba8", fontSize: 13, margin: 0 }}>{uploadError}</p>}
              {uploadResult && (
                <div style={s.uploadSuccess} className="fade-in">
                  <span style={{ color: "#a6e3a1" }}>✓</span>
                  {uploadFileName && <span style={{ color: "#89b4fa", fontSize: 12, fontFamily: "monospace" }}>{uploadFileName}</span>}
                  <span style={{ color: "#585b70" }}>·</span>
                  <span style={{ color: "#cdd6f4" }}>{uploadResult.row_count.toLocaleString()} rows</span>
                  <span style={{ color: "#585b70" }}>·</span>
                  <span style={{ color: "#a6adc8" }}>{uploadResult.columns.length} columns</span>
                  <span style={{ color: "#585b70", fontSize: 12 }}>
                    ({uploadResult.columns.slice(0, 5).join(", ")}{uploadResult.columns.length > 5 ? "…" : ""})
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div style={s.dbRow}>
              <select style={s.select} value={db} onChange={(e) => setDb(e.target.value)}>
                <option value="duckdb">Built-in sample data</option>
                <option value="postgres">PostgreSQL</option>
              </select>
            </div>
          )}

          {db === "postgres" && !useUpload && (
            <div style={s.pgGrid} className="fade-in">
              {([
                { k: "host",     label: "Host",     placeholder: "localhost", type: "text" },
                { k: "port",     label: "Port",     placeholder: "5432",      type: "text" },
                { k: "dbname",   label: "Database", placeholder: "mydb",      type: "text" },
                { k: "user",     label: "User",     placeholder: "postgres",  type: "text" },
                { k: "password", label: "Password", placeholder: "••••••",    type: "password" },
              ] as const).map(({ k, label, placeholder, type }) => (
                <div key={k} style={s.pgField}>
                  <label style={s.pgLabel}>{label}</label>
                  <input style={s.pgInput} type={type} placeholder={placeholder} value={pg[k]} onChange={setP(k)} />
                </div>
              ))}
            </div>
          )}
        </div>

        {startError && <div style={s.errorBox} className="fade-in">⚠ {startError}</div>}

        <button
          style={{ ...s.runBtn, background: mode === "general"
            ? "linear-gradient(135deg, #cba6f7, #b4befe)"
            : "linear-gradient(135deg, #89b4fa, #74c7ec)",
            opacity: canSubmit ? 1 : 0.45,
          }}
          onClick={handleSubmit}
          disabled={!canSubmit}
        >
          {submitting
            ? <><span style={s.btnSpinner} /> Starting…</>
            : mode === "general" ? "Explore Data →"
            : mode === "power_analysis" ? "Calculate Sample Size →"
            : "Run Analysis →"}
        </button>
      </div>
    </div>
  );
}

// ── GateBar ────────────────────────────────────────────────────────────────────

function GateBar({ task, mode, onStartOver }: { task: string; mode: string; onStartOver: () => void }) {
  const truncated = task.length > 90 ? task.slice(0, 90) + "…" : task;
  const accent = mode === "general" ? "#cba6f7" : "#89b4fa";
  const label  = mode === "general" ? "💡 Explore" : mode === "power_analysis" ? "📐 Design" : "🧪 A/B Test";
  return (
    <div style={gb.bar}>
      <div style={gb.inner}>
        <span style={{ ...gb.badge, color: accent, background: accent + "11", borderColor: accent + "33" }}>{label}</span>
        <span style={gb.task}>"{truncated}"</span>
        <button style={gb.startOver} onClick={onStartOver}>✕ Start over</button>
      </div>
    </div>
  );
}

const gb: Record<string, React.CSSProperties> = {
  bar:       { background: "#181825", borderBottom: "1px solid #313244", padding: "10px 20px" },
  inner:     { maxWidth: 760, margin: "0 auto", display: "flex", alignItems: "center", gap: 12 },
  badge:     { fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 20, border: "1px solid", flexShrink: 0 },
  task:      { color: "#585b70", fontSize: 13, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" as const },
  startOver: { background: "transparent", border: "1px solid #45475a", color: "#585b70", borderRadius: 6, padding: "4px 10px", fontSize: 12, cursor: "pointer", flexShrink: 0 },
};

// ── TrustBanner ────────────────────────────────────────────────────────────────

const _TRUST_CONFIG = {
  high:   { color: "#a6e3a1", bg: "#1a2820", border: "#a6e3a133", icon: "✓", label: "High confidence" },
  medium: { color: "#f9e2af", bg: "#201e10", border: "#f9e2af33", icon: "⚠", label: "Medium confidence" },
  low:    { color: "#f38ba8", bg: "#20101a", border: "#f38ba833", icon: "!", label: "Low confidence" },
};

function TrustBanner({ trust }: { trust: TrustIndicators }) {
  const cfg = _TRUST_CONFIG[trust.confidence_level] ?? _TRUST_CONFIG.medium;
  return (
    <div style={{ ...s.trustBanner, background: cfg.bg, borderColor: cfg.border }} className="slide-up">
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: cfg.color, fontWeight: 700, fontSize: 15 }}>{cfg.icon}</span>
        <span style={{ color: cfg.color, fontWeight: 700, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>{cfg.label}</span>
        <span style={{ color: "#45475a", fontSize: 12 }}>·</span>
        <span style={{ color: "#585b70", fontSize: 12 }}>{trust.n_data_points.toLocaleString()} data points</span>
      </div>
      <p style={{ color: "#a6adc8", fontSize: 12, margin: 0, lineHeight: 1.5 }}>{trust.confidence_reason}</p>
    </div>
  );
}

// ── ChartsGrid ─────────────────────────────────────────────────────────────────

function ChartsGrid({ charts }: { charts: ChartSpec[] }) {
  if (!charts || charts.length === 0) return null;
  return (
    <div style={s.chartsSection} className="slide-up">
      <div style={s.chartsSectionLabel}>Data at a glance</div>
      <div style={s.chartsGrid} className="charts-grid">
        {charts.map((spec, i) => <ChartCard key={i} spec={spec} />)}
      </div>
    </div>
  );
}

// ── PowerAnalysisSummary ────────────────────────────────────────────────────────

function PowerAnalysisSummary({ pa }: { pa: PowerAnalysisResult }) {
  return (
    <div style={pa_s.card} className="slide-up">
      <div style={pa_s.headline}>
        <div style={pa_s.stat}>
          <span style={pa_s.statNum}>{pa.required_n_per_arm.toLocaleString()}</span>
          <span style={pa_s.statLabel}>users per arm</span>
        </div>
        <div style={pa_s.divider} />
        <div style={pa_s.stat}>
          <span style={pa_s.statNum}>{pa.required_total_n.toLocaleString()}</span>
          <span style={pa_s.statLabel}>total users</span>
        </div>
        <div style={pa_s.divider} />
        <div style={pa_s.stat}>
          <span style={pa_s.statNum}>{pa.runtime_days}</span>
          <span style={pa_s.statLabel}>days runtime</span>
        </div>
        <div style={pa_s.divider} />
        <div style={pa_s.stat}>
          <span style={pa_s.statNum}>{pa.mde_target_pct}%</span>
          <span style={pa_s.statLabel}>target MDE</span>
        </div>
      </div>

      {pa.sensitivity && pa.sensitivity.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <div style={pa_s.tableLabel}>Sensitivity table</div>
          <table style={pa_s.table}>
            <thead>
              <tr>
                <th style={pa_s.th}>MDE (%)</th>
                <th style={pa_s.th}>N per arm</th>
                <th style={pa_s.th}>Runtime (days)</th>
              </tr>
            </thead>
            <tbody>
              {pa.sensitivity.map((row: SensitivityRow) => {
                const isTarget = row.mde_pct === pa.mde_target_pct;
                return (
                  <tr key={row.mde_pct} style={isTarget ? pa_s.trHighlight : {}}>
                    <td style={{ ...pa_s.td, color: isTarget ? "#89b4fa" : "#cdd6f4", fontWeight: isTarget ? 700 : 400 }}>
                      {row.mde_pct}%{isTarget ? " ◀" : ""}
                    </td>
                    <td style={{ ...pa_s.td, color: isTarget ? "#89b4fa" : "#a6adc8" }}>{row.n_per_arm.toLocaleString()}</td>
                    <td style={{ ...pa_s.td, color: isTarget ? "#89b4fa" : "#a6adc8" }}>{row.runtime_days}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const pa_s: Record<string, React.CSSProperties> = {
  card:        { background: "#181825", border: "1px solid #313244", borderRadius: 10, padding: "20px 24px", marginBottom: 16 },
  headline:    { display: "flex", gap: 0, alignItems: "stretch" },
  stat:        { flex: 1, display: "flex", flexDirection: "column" as const, alignItems: "center", padding: "8px 0" },
  statNum:     { color: "#89b4fa", fontSize: 26, fontWeight: 700, letterSpacing: "-0.5px" },
  statLabel:   { color: "#585b70", fontSize: 11, marginTop: 4, textTransform: "uppercase" as const, letterSpacing: "0.06em" },
  divider:     { width: 1, background: "#313244", margin: "4px 0" },
  tableLabel:  { color: "#45475a", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: 10 },
  table:       { width: "100%", borderCollapse: "collapse" as const },
  th:          { color: "#585b70", fontSize: 11, fontWeight: 600, textTransform: "uppercase" as const, padding: "6px 10px", textAlign: "left" as const, borderBottom: "1px solid #313244" },
  td:          { padding: "8px 10px", fontSize: 13, borderBottom: "1px solid #1e1e2e" },
  trHighlight: { background: "#89b4fa0a" },
};

// ── FinishedView ───────────────────────────────────────────────────────────────

const DETAILS_MARKER = "<!-- details -->";

function splitNarrative(raw: string): { brief: string; details: string } {
  const clean = sanitiseNarrative(raw);
  const idx   = clean.indexOf(DETAILS_MARKER);
  if (idx === -1) return { brief: clean, details: "" };
  return { brief: clean.slice(0, idx).trim(), details: clean.slice(idx + DETAILS_MARKER.length).trim() };
}

function FinishedView({ state, runId, onNewAnalysis }: { state: DoneEvent["state"]; runId: string; onNewAnalysis: () => void }) {
  const navigate = useNavigate();
  const [copied,      setCopied]      = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const isPowerAnalysis = state.analysis_mode === "power_analysis";
  const { brief, details } = splitNarrative(state.narrative_draft);
  const hasDetails = details.length > 0 || (isPowerAnalysis && !!state.power_analysis_result);
  const hasCharts  = state.charts && state.charts.length > 0;
  const hasTrust   = state.trust_indicators && state.trust_indicators.confidence_level;

  const downloadPdf = () => {
    const token  = localStorage.getItem("access_token") ?? "";
    const params = new URLSearchParams({ token, narrative: sanitiseNarrative(state.narrative_draft), recommendation: state.recommendation });
    window.open(`${API_BASE}/runs/${runId}/pdf?${params}`, "_blank");
  };

  const copyText = () => {
    navigator.clipboard.writeText(stripMarkdown(sanitiseNarrative(state.narrative_draft)));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={s.finPage} className="fade-in">
      <div style={s.finInner}>
        <div style={s.finHeader}>
          <div style={s.finTitle}>
            <span style={{ color: "#a6e3a1", fontSize: 20 }}>✓</span>
            <h2 style={{ color: "#cdd6f4", margin: 0, fontSize: 20 }}>Analysis Complete</h2>
          </div>
          <div style={s.finActions} className="fin-actions">
            {(hasDetails || hasCharts) && (
              <button
                style={showDetails ? s.btnDetailsActive : s.btnDetails}
                onClick={() => setShowDetails(v => !v)}
              >
                {showDetails ? "▲ Hide details" : "▼ Additional details"}
              </button>
            )}
            <button style={s.btnSec} onClick={copyText}>{copied ? "Copied!" : "Copy text"}</button>
            <button style={s.btnSec} onClick={downloadPdf}>↓ PDF</button>
            <button style={s.btnSec} onClick={() => navigate("/history")}>History</button>
            <button style={s.btnPri} onClick={onNewAnalysis}>+ New Analysis</button>
          </div>
        </div>

        {hasTrust && <TrustBanner trust={state.trust_indicators} />}

        <div style={s.narrativeCard} className="slide-up">
          <Markdown content={brief} />
        </div>

        {showDetails && (
          <div className="fade-in">
            {isPowerAnalysis && state.power_analysis_result && (
              <PowerAnalysisSummary pa={state.power_analysis_result} />
            )}
            {hasCharts && <ChartsGrid charts={state.charts} />}
            {details && (
              <div style={{ ...s.narrativeCard, marginTop: 14 }}>
                <Markdown content={details} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main Analysis page ─────────────────────────────────────────────────────────

export default function Analysis() {
  const navigate = useNavigate();

  const [selectedMode,     setSelectedMode]     = useState<Mode | null>(null);
  const [runId,            setRunId]            = useState<string | null>(null);
  const [taskText,         setTaskText]         = useState("");
  const [analysisMode,     setAnalysisMode]     = useState("");
  const [lastGate,         setLastGate]         = useState<string | null>(null);
  const [reconnectTrigger, setReconnectTrigger] = useState(0);
  const [submitting,       setSubmitting]       = useState(false);
  const [submitError,      setSubmitError]      = useState("");
  const [startError,       setStartError]       = useState("");
  const [username,         setUsername]         = useState("");

  const { gate, done, error, setGate, setError } = useSSE(runId, reconnectTrigger);

  useTokenRefresh(() => setReconnectTrigger(n => n + 1));

  useEffect(() => {
    client.get<{ username: string }>("/auth/me")
      .then(r => setUsername(r.data.username))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const mode = (gate?.payload as Record<string, unknown> | undefined)?.analysis_mode;
    if (typeof mode === "string" && (mode === "ab_test" || mode === "general" || mode === "power_analysis")) {
      setAnalysisMode(mode);
    }
  }, [gate]);

  const startOver = () => {
    setRunId(null); setSelectedMode(null); setTaskText(""); setAnalysisMode("");
    setLastGate(null); setSubmitError(""); setStartError(""); setGate(null);
  };

  const startRun = async (task: string, db_backend: string, pg?: PgCreds, uploadId?: string) => {
    setStartError("");
    try {
      const body: Record<string, unknown> = { task, db_backend };
      if (selectedMode) body.analysis_mode = selectedMode;
      if (uploadId) body.duckdb_path = uploadId;
      if (pg) { body.pg_host = pg.host; body.pg_port = parseInt(pg.port) || 5432; body.pg_dbname = pg.dbname; body.pg_user = pg.user; body.pg_password = pg.password; }
      const { data } = await client.post("/runs", body);
      setTaskText(task);
      setAnalysisMode(selectedMode ?? "");
      setRunId(data.run_id);
    } catch (err) {
      const msg = extractApiError(err, "Failed to start analysis.");
      setStartError(msg);
      throw err;
    }
  };

  const resume = async (value: object) => {
    if (!runId || !gate || submitting) return;
    setSubmitting(true); setSubmitError("");
    try {
      await client.post(`/runs/${runId}/resume`, { gate: gate.gate, value });
      setLastGate(gate.gate);
      setGate(null);
      setReconnectTrigger(n => n + 1);
    } catch (err) {
      setSubmitError(extractApiError(err, "Failed to submit."));
    } finally { setSubmitting(false); }
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  if (error) {
    const isConnectionLost = error.startsWith("Connection lost");
    return (
      <div style={s.center}>
        <div className="fade-in" style={{ textAlign: "center", maxWidth: 380 }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>{isConnectionLost ? "📡" : "⚠️"}</div>
          <p style={{ color: "#f38ba8", marginBottom: 8, fontSize: 15, fontWeight: 600 }}>
            {isConnectionLost ? "Connection lost" : "Something went wrong"}
          </p>
          <p style={{ color: "#585b70", marginBottom: 20, fontSize: 13 }}>{error}</p>
          <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
            {isConnectionLost && runId && (
              <button
                style={{ ...s.runBtn, background: "#313244", color: "#cdd6f4", width: "auto", padding: "10px 24px" }}
                onClick={() => { setError(""); setReconnectTrigger(n => n + 1); }}
              >
                ↻ Retry connection
              </button>
            )}
            <button
              style={{ ...s.runBtn, background: "transparent", border: "1px solid #45475a", color: "#a6adc8", width: "auto", padding: "10px 24px" }}
              onClick={startOver}
            >
              Start Over
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!selectedMode && !runId)
    return (
      <ModeSelect
        onSelect={setSelectedMode}
        username={username}
        onHistory={() => navigate("/history")}
        onSignOut={async () => { await logout(); navigate("/login"); }}
      />
    );

  if (selectedMode && !runId)
    return <TaskInput mode={selectedMode} onSubmit={startRun} onBack={() => setSelectedMode(null)} startError={startError} />;

  const errBanner = submitError ? (
    <div style={s.floatError} className="fade-in">
      ⚠ {submitError}
      <button style={s.dismissBtn} onClick={() => setSubmitError("")}>✕</button>
    </div>
  ) : null;

  const payload    = gate?.payload as Record<string, unknown> | undefined;
  const activeMode = (analysisMode || selectedMode || "ab_test") as string;

  const renderGate = (el: React.ReactNode) => (
    <div style={s.gatePage}>
      <GateBar task={taskText} mode={activeMode} onStartOver={startOver} />
      {errBanner}
      <div style={s.gateScroll}>
        <div style={s.gateContent}>{el}</div>
      </div>
    </div>
  );

  if (gate?.gate === "intent")
    return renderGate(<IntentGate payload={payload as Parameters<typeof IntentGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  if (gate?.gate === "semantic_cache")
    return renderGate(<SemanticCacheGate payload={payload as Parameters<typeof SemanticCacheGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  if (gate?.gate === "query")
    return renderGate(<QueryGate payload={payload as Parameters<typeof QueryGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  if (gate?.gate === "analysis") {
    if (payload?.analysis_mode === "general")
      return renderGate(<GeneralAnalysisGate payload={payload as Parameters<typeof GeneralAnalysisGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
    return renderGate(<AnalysisGate payload={payload as Parameters<typeof AnalysisGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  }
  if (gate?.gate === "narrative")
    return renderGate(<NarrativeGate payload={payload as Parameters<typeof NarrativeGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);

  if (done) return <FinishedView state={done.state} runId={runId!} onNewAnalysis={startOver} />;

  return (
    <div style={s.gatePage}>
      <GateBar task={taskText} mode={activeMode} onStartOver={startOver} />
      <div style={s.gateScroll}>
        <div style={s.center}>
          <div style={{ width: "100%", maxWidth: 420 }} className="fade-in">
            <div style={s.progressCard}>
              <PipelineProgress gate={gate?.gate ?? null} lastGate={lastGate} analysisMode={analysisMode} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  page:       { minHeight: "100vh", background: "#11111b", padding: "0 20px 40px" },
  center:     { flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 20px" },

  header:     { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 0", maxWidth: 640, margin: "0 auto" },
  backBtn:    { background: "transparent", border: "none", color: "#585b70", fontSize: 13, cursor: "pointer", padding: 0 },
  modeBadge:  { fontSize: 12, fontWeight: 600, padding: "4px 12px", borderRadius: 20, border: "1px solid" },

  inputCard:  { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "32px 32px 28px", maxWidth: 640, margin: "0 auto", boxShadow: "0 8px 40px #00000044" },
  heading:    { color: "#cdd6f4", fontSize: 22, fontWeight: 700, marginBottom: 4 },
  sub:        { color: "#585b70", fontSize: 13, marginBottom: 20 },
  taskInput:  { width: "100%", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, padding: "12px 14px", fontSize: 14, resize: "vertical" as const, lineHeight: 1.6, marginBottom: 20, boxSizing: "border-box" as const },

  samplesSection: { marginBottom: 20 },
  samplesGrid:    { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 },
  sampleCard:     { display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", background: "#181825", border: "1px solid #313244", borderRadius: 8, cursor: "pointer", textAlign: "left" as const, transition: "border-color 0.15s", position: "relative" as const },
  sampleLoading:  { opacity: 0.6 },
  sampleIcon:     { fontSize: 18, flexShrink: 0 },
  sampleLabel:    { color: "#cdd6f4", fontSize: 12, fontWeight: 600 },
  sampleDomain:   { color: "#585b70", fontSize: 11, marginTop: 1 },
  sampleSpinner:  { position: "absolute" as const, right: 10, width: 12, height: 12, border: "2px solid #45475a", borderTop: "2px solid #89b4fa", borderRadius: "50%", animation: "spin 0.7s linear infinite" },

  section:      { marginBottom: 20 },
  sectionLabel: { color: "#a6adc8", fontSize: 11, fontWeight: 600, textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: 8 },
  sourceRow:    { display: "flex", gap: 20, marginBottom: 12 },
  sourceOption: { display: "flex", alignItems: "center", gap: 6, color: "#a6adc8", fontSize: 13, cursor: "pointer" },
  dbRow:        { display: "flex" },
  select:       { padding: "9px 12px", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, fontSize: 13, cursor: "pointer" },

  uploadBox:    { display: "flex", flexDirection: "column" as const, gap: 8, padding: 14, background: "#181825", borderRadius: 8, border: "1px dashed #45475a" },
  uploadBtn:    { alignSelf: "flex-start", padding: "8px 16px", background: "#313244", color: "#cdd6f4", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13 },
  uploadSuccess: { display: "flex", gap: 8, alignItems: "center", fontSize: 13, flexWrap: "wrap" as const },

  pgGrid:  { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 10, padding: 14, background: "#181825", borderRadius: 8, border: "1px solid #313244" },
  pgField: { display: "flex", flexDirection: "column" as const, gap: 4 },
  pgLabel: { fontSize: 11, color: "#585b70", fontWeight: 600, textTransform: "uppercase" as const },
  pgInput: { padding: "8px 10px", background: "#1e1e2e", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 6, fontSize: 13 },

  errorBox:   { background: "#f38ba811", border: "1px solid #f38ba844", color: "#f38ba8", borderRadius: 8, padding: "10px 14px", fontSize: 13, marginBottom: 14 },
  runBtn:     { width: "100%", padding: "13px 0", color: "#1e1e2e", border: "none", borderRadius: 8, fontWeight: 700, fontSize: 15, cursor: "pointer", marginTop: 4, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 },
  btnSpinner: { width: 14, height: 14, border: "2px solid #1e1e2e44", borderTop: "2px solid #1e1e2e", borderRadius: "50%", animation: "spin 0.7s linear infinite", display: "inline-block", flexShrink: 0 },

  gatePage:   { minHeight: "100vh", background: "#11111b", display: "flex", flexDirection: "column" as const },
  gateScroll: { flex: 1, display: "flex", flexDirection: "column" as const },
  gateContent:{ padding: "36px 20px 60px", flex: 1 },

  progressCard: { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 14, padding: "16px 24px", boxShadow: "0 8px 40px #00000044" },

  floatError: { position: "fixed", top: 20, left: "50%", transform: "translateX(-50%)", background: "#f38ba8", color: "#1e1e2e", padding: "10px 20px", borderRadius: 8, display: "flex", gap: 12, alignItems: "center", zIndex: 1000, fontSize: 14, boxShadow: "0 4px 20px #00000055", whiteSpace: "nowrap" as const },
  dismissBtn: { background: "transparent", border: "none", color: "#1e1e2e", cursor: "pointer", fontWeight: 700, fontSize: 16, padding: 0 },

  finPage:    { minHeight: "100vh", background: "#11111b", padding: "32px 20px 60px" },
  finInner:   { maxWidth: 820, margin: "0 auto" },
  finHeader:  { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap" as const, gap: 12 },
  finTitle:   { display: "flex", alignItems: "center", gap: 10 },
  finActions: { display: "flex", gap: 8 },
  btnSec:          { padding: "7px 16px", background: "#313244", color: "#cdd6f4", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13 },
  btnPri:          { padding: "7px 16px", background: "#89b4fa", color: "#1e1e2e", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 600 },
  btnDetails:      { padding: "7px 16px", background: "transparent", color: "#89b4fa", border: "1px solid #89b4fa44", borderRadius: 6, cursor: "pointer", fontSize: 13 },
  btnDetailsActive:{ padding: "7px 16px", background: "#89b4fa1a", color: "#89b4fa", border: "1px solid #89b4fa66", borderRadius: 6, cursor: "pointer", fontSize: 13 },

  trustBanner: { border: "1px solid", borderRadius: 10, padding: "12px 16px", marginBottom: 16, display: "flex", flexDirection: "column" as const, gap: 6 },

  chartsSection:      { marginBottom: 24, marginTop: 14 },
  chartsSectionLabel: { color: "#45475a", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 12 },
  chartsGrid:         { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 },

  narrativeCard: { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 12, padding: "28px 32px", lineHeight: 1.7 },
};
