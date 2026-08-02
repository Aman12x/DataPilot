import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import client, {
  API_BASE, uploadFile, type UploadResult, logout,
  getActiveWorkspaceId, setActiveWorkspaceId, type WorkspaceSummary,
} from "../api/client";
import { downloadRunPdf } from "../utils/downloadPdf";
import { useSSE, type DoneEvent, type StepEvent, type TrustIndicators, type PowerAnalysisResult, type SensitivityRow } from "../hooks/useSSE";
import { useTokenRefresh } from "../hooks/useTokenRefresh";
import PipelineProgress from "../components/PipelineProgress";
import Markdown from "../components/Markdown";
import ChartCard, { type ChartSpec } from "../components/ChartCard";
import ChainOfThought from "../components/ChainOfThought";
import IntentGate from "../components/gates/IntentGate";
import SemanticCacheGate from "../components/gates/SemanticCacheGate";
import MetricGate from "../components/gates/MetricGate";
import QueryGate from "../components/gates/QueryGate";
import AnalysisGate from "../components/gates/AnalysisGate";
import GeneralAnalysisGate from "../components/gates/GeneralAnalysisGate";
import NarrativeGate from "../components/gates/NarrativeGate";
import ErrorBoundary from "../components/ErrorBoundary";
import { IconAlert, IconArrowLeft, IconCheck, IconChevronDown, IconChevronRight, IconChevronUp, IconClose, IconDownload } from "../components/icons";
import {
  type Mode, type PgCreds, type BqCreds, type Sample, type SavedConnection,
  type MetricPackSummary, type RunOptions, MODE_META,
} from "../types/analysis";
import { stripMarkdown, sanitiseNarrative, normalizeTypography } from "../utils/markdown";
import { extractApiError } from "../utils/error";
import StakeholderDeck from "../components/StakeholderDeck";
import PackStudio from "../components/PackStudio";
import MembersPanel from "../components/MembersPanel";
import AnnotationStudio from "../components/AnnotationStudio";
import type { DeckData } from "../hooks/useSSE";
import AppShell from "../components/AppShell";

// ── Fallback samples (shown even when backend is offline) ─────────────────────
const FALLBACK_SAMPLES: Sample[] = [
  { name: "ecommerce_ab_test.csv",       label: "E-commerce A/B Test",    domain: "E-commerce", icon: "EC", mode: "ab_test",  suggested_task: "Did the new checkout flow increase revenue and orders? Identify which customer segments and devices benefited most." },
  { name: "clinical_trial.csv",          label: "Clinical Trial",          domain: "Healthcare", icon: "HC", mode: "ab_test",  suggested_task: "Did Drug A improve recovery scores compared to the control group? Check for side effects and subgroup differences by age and severity." },
  { name: "media_ctr_experiment.csv",    label: "Media CTR Experiment",    domain: "Media",      icon: "MD", mode: "ab_test",  suggested_task: "Did the new content recommendation algorithm improve click-through rate and watch time? Break down results by platform, content type, and age group." },
  { name: "saas_churn_analysis.csv",     label: "SaaS Churn Analysis",     domain: "SaaS",       icon: "SA", mode: "general", suggested_task: "What factors most strongly predict customer churn? Which industries and plan types have the highest churn rates?" },
  { name: "logistics_ops.csv",           label: "Logistics Operations",    domain: "Logistics",  icon: "LG", mode: "general", suggested_task: "Why are deliveries being delayed? Which carriers and routes have the worst on-time performance?" },
  { name: "customer_transactions_10k.csv", label: "Retail Transactions",   domain: "Retail",     icon: "RT", mode: "general", suggested_task: "Which product categories and customer segments drive the most revenue? What patterns exist in fraud, returns, and churn?" },
];

// ── ModeSelect — product home ─────────────────────────────────────────────────

function ModeSelect({ onSelect, username, onHistory, onSignOut, workspaces, workspaceId, onWorkspaceChange }: {
  onSelect: (mode: Mode) => void;
  username: string;
  onHistory: () => void;
  onSignOut: () => void;
  workspaces: WorkspaceSummary[];
  workspaceId: string;
  onWorkspaceChange: (id: string) => void;
}) {
  const [showAbSub, setShowAbSub] = useState(false);
  const [showPacks, setShowPacks] = useState(false);
  const [showMembers, setShowMembers] = useState(false);
  const [showAnnotations, setShowAnnotations] = useState(false);
  const activeWs = workspaces.find((w) => w.workspace_id === workspaceId);
  const canManage = activeWs?.role === "owner";

  return (
    <AppShell
      nav={(
        <>
          {workspaces.length > 0 && (
            <select
              aria-label="Workspace"
              value={workspaceId}
              onChange={(e) => onWorkspaceChange(e.target.value)}
              className="dp-shell-select"
            >
              {workspaces.map((w) => (
                <option key={w.workspace_id} value={w.workspace_id}>
                  {w.name}{w.role === "owner" ? "" : ` (${w.role})`}
                </option>
              ))}
            </select>
          )}
          {username && <span className="dp-shell-meta">{username}</span>}
          <button className="dp-btn dp-btn-ghost" onClick={() => setShowPacks(true)}>Metrics</button>
          <button className="dp-btn dp-btn-ghost" onClick={() => setShowAnnotations(true)}>Schema</button>
          {workspaceId && (
            <button className="dp-btn dp-btn-ghost" onClick={() => setShowMembers(true)}>Team</button>
          )}
          <button className="dp-btn dp-btn-ghost" onClick={onHistory}>History</button>
          <button className="dp-btn dp-btn-link" onClick={onSignOut}>Sign out</button>
        </>
      )}
    >
      <PackStudio open={showPacks} onClose={() => setShowPacks(false)} canEdit={!!canManage || !workspaceId} />
      <AnnotationStudio
        open={showAnnotations}
        onClose={() => setShowAnnotations(false)}
        canEdit={!!canManage || !workspaceId}
      />
      <MembersPanel
        open={showMembers}
        workspaceId={workspaceId}
        workspaceName={activeWs?.name}
        canManage={!!canManage}
        onClose={() => setShowMembers(false)}
      />

      <section className="dp-home-hero">
        <div className="dp-home-copy fade-in">
          <h1 className="dp-home-brand">What would you like to know?</h1>
          <p className="dp-home-sub">
            Connect a warehouse or upload a file, then run an analysis with
            review at every critical step.
          </p>

          <div className="dp-home-actions">
            <button className="dp-mode-action" onClick={() => onSelect("general")}>
              <div>
                <strong>Explore & Understand</strong>
                <span>Patterns, drivers, and anomalies across any dataset</span>
              </div>
              <em><IconChevronRight /></em>
            </button>

            {!showAbSub ? (
              <button className="dp-mode-action" onClick={() => setShowAbSub(true)}>
                <div>
                  <strong>A/B Testing</strong>
                  <span>Design experiments or interpret completed tests</span>
                </div>
                <em><IconChevronRight /></em>
              </button>
            ) : (
              <div className="dp-home-subpanel fade-in">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                  <strong style={{ fontSize: 13, color: "var(--dp-ink)" }}>A/B Testing</strong>
                  <button className="dp-btn dp-btn-link" onClick={() => setShowAbSub(false)}>Back</button>
                </div>
                <button onClick={() => onSelect("power_analysis")}>
                  <div>
                    <strong>Design Experiment</strong>
                    <small>Sample size, runtime, and sensitivity</small>
                  </div>
                  <em style={{ marginLeft: "auto", color: "var(--dp-ink-faint)", fontStyle: "normal" }}><IconChevronRight /></em>
                </button>
                <button onClick={() => onSelect("ab_test")}>
                  <div>
                    <strong>Interpret Results</strong>
                    <small>Full statistical readout for a finished test</small>
                  </div>
                  <em style={{ marginLeft: "auto", color: "var(--dp-ink-faint)", fontStyle: "normal" }}><IconChevronRight /></em>
                </button>
              </div>
            )}
          </div>

          <p className="dp-home-footnote">Not sure where to start? Choose Explore and DataPilot will route the analysis.</p>
        </div>
      </section>
    </AppShell>
  );
}

// ── TaskInput ──────────────────────────────────────────────────────────────────

function TaskInput({ mode, onSubmit, onBack, startError }: {
  mode: Mode;
  onSubmit: (task: string, db: string, pg?: PgCreds, uploadId?: string, opts?: RunOptions) => void;
  onBack: () => void;
  startError?: string;
}) {
  const meta = MODE_META[mode];
  const [task,           setTask]           = useState("");
  const [db,             setDb]             = useState("duckdb");
  const [pg,             setPg]             = useState<PgCreds>({ host: "localhost", port: "5432", dbname: "", user: "", password: "" });
  const [bq,             setBq]             = useState<BqCreds>({ projectId: "", dataset: "", credentialsJson: "" });
  const [useUpload,      setUseUpload]      = useState(false);
  const [uploading,      setUploading]      = useState(false);
  const [uploadResult,   setUploadResult]   = useState<UploadResult | null>(null);
  const [uploadError,    setUploadError]    = useState("");
  const [uploadFileName, setUploadFileName] = useState("");
  const [samples,        setSamples]        = useState<Sample[]>(FALLBACK_SAMPLES.filter(s => s.mode === mode));
  const [loadingSample,  setLoadingSample]  = useState<string | null>(null);
  const [submitting,     setSubmitting]     = useState(false);
  const [connections,    setConnections]    = useState<SavedConnection[]>([]);
  const [packs,          setPacks]          = useState<MetricPackSummary[]>([]);
  const [connectionId,   setConnectionId]   = useState("");
  const [metricPackId,   setMetricPackId]   = useState("");
  const [savingConn,     setSavingConn]     = useState(false);
  const [connMsg,        setConnMsg]        = useState("");
  const [showPacks,      setShowPacks]      = useState(false);
  const [showAnnotations, setShowAnnotations] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const reloadPacks = () => {
    client.get<{ metric_packs: MetricPackSummary[] }>("/metric-packs")
      .then(r => setPacks(r.data.metric_packs || []))
      .catch(() => {});
  };

  useEffect(() => {
    setSamples(FALLBACK_SAMPLES.filter(s => s.mode === mode));
    client.get<Sample[]>("/samples")
      .then(r => setSamples(r.data.filter(s => s.mode === mode)))
      .catch(() => {});
    client.get<{ connections: SavedConnection[] }>("/connections")
      .then(r => setConnections(r.data.connections || []))
      .catch(() => {});
    reloadPacks();
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

  const usingSavedConn = !!connectionId && !useUpload;
  const sqlBackend = db === "postgres" || db === "mysql";
  const pgValid = !sqlBackend || usingSavedConn || !!(pg.host && pg.dbname && pg.user);
  const bqValid = db !== "bigquery" || usingSavedConn || !!(bq.projectId && bq.dataset && bq.credentialsJson);
  const uploadReady = !useUpload || !!uploadResult;
  const canSubmit   = task.trim() && pgValid && bqValid && uploadReady && !uploading && !loadingSample && !submitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      const opts: RunOptions = {};
      if (metricPackId) opts.metricPackId = metricPackId;
      if (usingSavedConn) opts.connectionId = connectionId;
      if (db === "bigquery" && !usingSavedConn) opts.bq = bq;
      if (useUpload && uploadResult) {
        await onSubmit(task, "duckdb", undefined, uploadResult.upload_id, opts);
      } else if (usingSavedConn) {
        await onSubmit(task, db === "duckdb" ? "postgres" : db, undefined, undefined, opts);
      } else {
        await onSubmit(task, db, sqlBackend ? pg : undefined, undefined, opts);
      }
    } finally { setSubmitting(false); }
  };

  const saveCurrentConnection = async () => {
    setSavingConn(true); setConnMsg("");
    try {
      let payload: Record<string, unknown>;
      if (db === "bigquery") {
        if (!bq.projectId || !bq.dataset || !bq.credentialsJson) return;
        payload = {
          name: `${bq.dataset}@${bq.projectId}`,
          backend: "bigquery",
          project_id: bq.projectId,
          dbname: bq.dataset,
          password: bq.credentialsJson,
          test: true,
        };
      } else {
        if (!pg.host || !pg.dbname || !pg.user) return;
        const backend = db === "mysql" ? "mysql" : "postgres";
        payload = {
          name: `${pg.dbname}@${pg.host}`,
          backend,
          host: pg.host,
          port: parseInt(pg.port) || (backend === "mysql" ? 3306 : 5432),
          dbname: pg.dbname,
          username: pg.user,
          password: pg.password,
          sslmode: "prefer",
          test: true,
        };
      }
      const { data } = await client.post<SavedConnection>("/connections", payload);
      setConnections((c) => [data, ...c]);
      setConnectionId(data.connection_id);
      setDb(data.backend || db);
      setConnMsg("Connection saved and tested.");
    } catch (err) {
      setConnMsg(extractApiError(err, "Could not save connection."));
    } finally { setSavingConn(false); }
  };

  return (
    <div className="dp-workspace fade-in">
      <div className="dp-workspace-header">
        <button className="dp-btn dp-btn-link" onClick={onBack} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><IconArrowLeft /> Back</button>
        <span className="dp-badge" style={{ color: meta.accent, borderColor: "transparent", background: "var(--dp-accent-soft)" }}>
          {meta.badge}
        </span>
      </div>

      <div className="dp-panel slide-up">
        <h2>{meta.heading}</h2>
        <p>{meta.sub}</p>

        {samples.length > 0 && (
          <div style={s.samplesSection}>
            <div style={s.sectionLabel}>Try a sample dataset</div>
            <div style={s.samplesGrid} className="samples-grid">
              {samples.map(s2 => (
                <button
                  key={s2.name}
                  className="dp-sample-card"
                  style={{ ...s.sampleCard, ...(loadingSample === s2.name ? s.sampleLoading : {}) }}
                  onClick={() => loadSample(s2)}
                  disabled={!!loadingSample}
                  type="button"
                >
                  <span style={s.sampleIcon}>{(/^[A-Za-z0-9]/.test(s2.icon) ? s2.icon : s2.domain).slice(0, 2).toUpperCase()}</span>
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
          className="dp-textarea"
          style={{ ...s.taskInput, border: undefined, background: undefined, padding: undefined, borderRadius: undefined }}
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
                {uploading ? "Uploading…" : "Choose file"}
              </button>
              {uploadError && <p style={{ color: "var(--dp-danger)", fontSize: 13, margin: 0 }}>{uploadError}</p>}
              {uploadResult && (
                <div style={s.uploadSuccess} className="fade-in">
                  <span style={{ color: "var(--dp-success)", display: "inline-flex" }}><IconCheck /></span>
                  {uploadFileName && <span style={{ color: "var(--dp-accent)", fontSize: 12, fontFamily: "var(--dp-mono)" }}>{uploadFileName}</span>}
                  <span style={{ color: "var(--dp-ink-faint)" }}>|</span>
                  <span style={{ color: "var(--dp-ink)" }}>{uploadResult.row_count.toLocaleString()} rows</span>
                  <span style={{ color: "var(--dp-ink-faint)" }}>|</span>
                  <span style={{ color: "var(--dp-ink-secondary)" }}>{uploadResult.columns.length} columns</span>
                  <span style={{ color: "var(--dp-ink-muted)", fontSize: 12 }}>
                    ({uploadResult.columns.slice(0, 5).join(", ")}{uploadResult.columns.length > 5 ? "…" : ""})
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div style={s.dbRow}>
              <select
                style={s.select}
                value={connectionId ? `conn:${connectionId}` : db}
                onChange={(e) => {
                  const v = e.target.value;
                  if (v.startsWith("conn:")) {
                    const id = v.slice(5);
                    const c = connections.find(x => x.connection_id === id);
                    setConnectionId(id);
                    setDb(c?.backend || "postgres");
                  } else {
                    setConnectionId("");
                    setDb(v);
                    if (v === "mysql") setPg(p => ({ ...p, port: p.port === "5432" ? "3306" : p.port }));
                    if (v === "postgres") setPg(p => ({ ...p, port: p.port === "3306" ? "5432" : p.port }));
                  }
                }}
              >
                <option value="duckdb">Built-in sample data</option>
                <option value="postgres">PostgreSQL</option>
                <option value="mysql">MySQL / MariaDB</option>
                <option value="bigquery">Google BigQuery</option>
                {connections.map((c) => (
                  <option key={c.connection_id} value={`conn:${c.connection_id}`}>
                    {c.name} ({c.backend})
                    {c.backend === "bigquery"
                      ? ` (${c.project_id || "?"}/${c.dbname})`
                      : ` (${c.host}/${c.dbname})`}
                    {c.last_test_ok === false ? " (last test failed)" : ""}
                  </option>
                ))}
              </select>
            </div>
          )}

          {(db === "postgres" || db === "mysql") && !useUpload && !connectionId && (
            <div style={s.pgGrid} className="fade-in">
              {([
                { k: "host",     label: "Host",     placeholder: "localhost", type: "text" },
                { k: "port",     label: "Port",     placeholder: db === "mysql" ? "3306" : "5432", type: "text" },
                { k: "dbname",   label: "Database", placeholder: "mydb",      type: "text" },
                { k: "user",     label: "User",     placeholder: db === "mysql" ? "root" : "postgres", type: "text" },
                { k: "password", label: "Password", placeholder: "••••••",    type: "password" },
              ] as const).map(({ k, label, placeholder, type }) => (
                <div key={k} style={s.pgField}>
                  <label style={s.pgLabel}>{label}</label>
                  <input style={s.pgInput} type={type} placeholder={placeholder} value={pg[k]} onChange={setP(k)} />
                </div>
              ))}
              <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "center" }}>
                <button
                  type="button"
                  className="dp-btn dp-btn-ghost"
                  style={{ padding: "6px 12px", fontSize: 12 }}
                  onClick={saveCurrentConnection}
                  disabled={savingConn || !pg.host || !pg.dbname || !pg.user}
                >
                  {savingConn ? "Testing & saving…" : "Save connection"}
                </button>
                {connMsg && <span style={{ fontSize: 12, color: connMsg.includes("saved") ? "var(--dp-success)" : "var(--dp-danger)" }}>{connMsg}</span>}
              </div>
            </div>
          )}

          {db === "bigquery" && !useUpload && !connectionId && (
            <div style={s.pgGrid} className="fade-in">
              <div style={s.pgField}>
                <label style={s.pgLabel}>Project ID</label>
                <input
                  style={s.pgInput}
                  type="text"
                  placeholder="my-gcp-project"
                  value={bq.projectId}
                  onChange={(e) => setBq(b => ({ ...b, projectId: e.target.value }))}
                  autoComplete="off"
                />
              </div>
              <div style={s.pgField}>
                <label style={s.pgLabel}>Dataset</label>
                <input
                  style={s.pgInput}
                  type="text"
                  placeholder="analytics"
                  value={bq.dataset}
                  onChange={(e) => setBq(b => ({ ...b, dataset: e.target.value }))}
                  autoComplete="off"
                />
              </div>
              <div style={{ ...s.pgField, gridColumn: "1 / -1" }}>
                <label style={s.pgLabel}>Service account JSON</label>
                <textarea
                  style={{ ...s.pgInput, fontFamily: "ui-monospace, monospace", fontSize: 12, minHeight: 110, resize: "vertical" }}
                  placeholder='{"type":"service_account","project_id":"...","private_key":"..."}'
                  value={bq.credentialsJson}
                  onChange={(e) => setBq(b => ({ ...b, credentialsJson: e.target.value }))}
                  autoComplete="off"
                  spellCheck={false}
                />
                <div style={{ fontSize: 11, color: "var(--dp-ink-muted)", marginTop: 4 }}>
                  Paste a GCP service-account key (BigQuery Job User + Data Viewer). Prefer saving to the vault.
                </div>
              </div>
              <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "center" }}>
                <button
                  type="button"
                  className="dp-btn dp-btn-ghost"
                  style={{ padding: "6px 12px", fontSize: 12 }}
                  onClick={saveCurrentConnection}
                  disabled={savingConn || !bq.projectId || !bq.dataset || !bq.credentialsJson}
                >
                  {savingConn ? "Testing & saving…" : "Save connection"}
                </button>
                {connMsg && <span style={{ fontSize: 12, color: connMsg.includes("saved") ? "var(--dp-success)" : "var(--dp-danger)" }}>{connMsg}</span>}
              </div>
            </div>
          )}

          {connectionId && !useUpload && (
            <div style={{ marginTop: 10, display: "flex", justifyContent: "flex-end" }}>
              <button
                type="button"
                className="dp-btn dp-btn-link"
                style={{ fontSize: 12, padding: 0 }}
                onClick={() => setShowAnnotations(true)}
              >
                Annotate schema
              </button>
            </div>
          )}

          <div style={{ marginTop: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={s.sectionLabel}>Metric pack (optional)</div>
              <button
                type="button"
                className="dp-btn dp-btn-link"
                style={{ fontSize: 12, padding: 0 }}
                onClick={() => setShowPacks(true)}
              >
                {packs.length ? "Manage packs" : "Define metrics"}
              </button>
            </div>
            {packs.length > 0 ? (
              <>
                <select
                  style={s.select}
                  value={metricPackId}
                  onChange={(e) => setMetricPackId(e.target.value)}
                >
                  <option value="">Infer from schema</option>
                  {packs.map((p) => (
                    <option key={p.pack_id} value={p.pack_id}>
                      {p.name}{p.certified ? " (certified)" : ""}
                    </option>
                  ))}
                </select>
                <p style={{ color: "var(--dp-ink-muted)", fontSize: 11, margin: "6px 0 0" }}>
                  Certified packs skip the metric confirmation gate and constrain SQL to your definitions.
                </p>
              </>
            ) : (
              <p style={{ color: "var(--dp-ink-muted)", fontSize: 12, margin: "4px 0 0" }}>
                No packs yet. Define your revenue and DAU mapping once so the team does not re-confirm every run.
              </p>
            )}
          </div>
        </div>

        <PackStudio
          open={showPacks}
          onClose={() => setShowPacks(false)}
          onChanged={() => { reloadPacks(); }}
        />
        <AnnotationStudio
          open={showAnnotations}
          onClose={() => setShowAnnotations(false)}
          initialConnectionId={connectionId}
        />

        {startError && <div style={s.errorBox} className="fade-in">{startError}</div>}

        <button
          className="dp-btn dp-btn-primary"
          style={{ ...s.runBtn, opacity: canSubmit ? 1 : 0.45 }}
          onClick={handleSubmit}
          disabled={!canSubmit}
        >
          {submitting
            ? <><span style={s.btnSpinner} /> Starting…</>
            : mode === "general" ? "Explore Data"
            : mode === "power_analysis" ? "Calculate Sample Size"
            : "Run Analysis"}
        </button>
      </div>
    </div>
  );
}

// ── GateBar ────────────────────────────────────────────────────────────────────

function GateBar({ task, mode, onStartOver }: { task: string; mode: string; onStartOver: () => void }) {
  const truncated = task.length > 90 ? task.slice(0, 90) + "…" : task;
  const label  = mode === "general" ? "Explore" : mode === "power_analysis" ? "Design" : "A/B Test";
  return (
    <div className="dp-gate-bar">
      <div style={gb.inner}>
        <span className="dp-badge">{label}</span>
        <span className="dp-gate-task" title={task}>"{truncated}"</span>
        <button className="dp-btn dp-btn-ghost" onClick={onStartOver}>Start over</button>
      </div>
    </div>
  );
}

const gb: Record<string, React.CSSProperties> = {
  inner: { width: "min(100%, 820px)", margin: "0 auto", display: "flex", alignItems: "center", gap: 12 },
};

// ── TrustBanner ────────────────────────────────────────────────────────────────

const _TRUST_CONFIG = {
  high:   { color: "var(--dp-success)", bg: "var(--dp-success-soft)", border: "rgba(31,138,91,0.25)", high: true,  label: "High confidence" },
  medium: { color: "var(--dp-warning)", bg: "var(--dp-warning-soft)", border: "rgba(154,103,0,0.25)", high: false, label: "Medium confidence" },
  low:    { color: "var(--dp-danger)", bg: "var(--dp-danger-soft)", border: "rgba(194,59,74,0.25)", high: false, label: "Low confidence" },
};

function TrustBanner({ trust }: { trust: TrustIndicators }) {
  const cfg = _TRUST_CONFIG[trust.confidence_level] ?? _TRUST_CONFIG.medium;
  return (
    <div style={{ ...s.trustBanner, background: cfg.bg, borderColor: cfg.border }} className="slide-up">
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ color: cfg.color, fontSize: 14, display: "inline-flex" }}>{cfg.high ? <IconCheck /> : <IconAlert />}</span>
        <span style={{ color: cfg.color, fontWeight: 600, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>{cfg.label}</span>
        <span style={{ color: "var(--dp-ink-faint)", fontSize: 12 }}>|</span>
        <span style={{ color: "var(--dp-ink-muted)", fontSize: 12 }}>{trust.n_data_points.toLocaleString()} data points</span>
      </div>
      <p style={{ color: "var(--dp-ink-secondary)", fontSize: 12, margin: 0, lineHeight: 1.5 }}>{normalizeTypography(trust.confidence_reason)}</p>
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
                    <td style={{ ...pa_s.td, color: isTarget ? "var(--dp-accent)" : "var(--dp-ink)", fontWeight: isTarget ? 700 : 400 }}>
                      {row.mde_pct}%{isTarget ? " (target)" : ""}
                    </td>
                    <td style={{ ...pa_s.td, color: isTarget ? "var(--dp-accent)" : "var(--dp-ink-secondary)" }}>{row.n_per_arm.toLocaleString()}</td>
                    <td style={{ ...pa_s.td, color: isTarget ? "var(--dp-accent)" : "var(--dp-ink-secondary)" }}>{row.runtime_days}</td>
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
  card:        { background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", borderRadius: 10, padding: "20px 24px", marginBottom: 16 },
  headline:    { display: "flex", gap: 0, alignItems: "stretch" },
  stat:        { flex: 1, display: "flex", flexDirection: "column" as const, alignItems: "center", padding: "8px 0" },
  statNum:     { color: "var(--dp-accent)", fontSize: 26, fontWeight: 700, letterSpacing: "-0.5px" },
  statLabel:   { color: "var(--dp-ink-muted)", fontSize: 11, marginTop: 4, textTransform: "uppercase" as const, letterSpacing: "0.06em" },
  divider:     { width: 1, background: "var(--dp-line)", margin: "4px 0" },
  tableLabel:  { color: "var(--dp-ink-faint)", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: 10 },
  table:       { width: "100%", borderCollapse: "collapse" as const },
  th:          { color: "var(--dp-ink-muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase" as const, padding: "6px 10px", textAlign: "left" as const, borderBottom: "1px solid var(--dp-line)" },
  td:          { padding: "8px 10px", fontSize: 13, borderBottom: "1px solid var(--dp-surface)" },
  trHighlight: { background: "var(--dp-accent)0a" },
};

// ── FinishedView ───────────────────────────────────────────────────────────────

const DETAILS_MARKER = "<!-- details -->";

function splitNarrative(raw: string): { brief: string; details: string } {
  const clean = sanitiseNarrative(raw);
  const idx   = clean.indexOf(DETAILS_MARKER);
  if (idx === -1) return { brief: clean, details: "" };
  return { brief: clean.slice(0, idx).trim(), details: clean.slice(idx + DETAILS_MARKER.length).trim() };
}

function FinishedView({ state, runId, steps, onNewAnalysis, onFollowUp }: {
  state: DoneEvent["state"];
  runId: string;
  steps: StepEvent[];
  onNewAnalysis: () => void;
  onFollowUp: (task: string) => void;
}) {
  const navigate = useNavigate();
  const [copied,       setCopied]      = useState(false);
  const [showDetails,  setShowDetails] = useState(false);
  const [showFullReport, setShowFullReport] = useState(false);
  const [followUp,     setFollowUp]    = useState("");
  const [submitting,   setSubmitting]  = useState(false);

  const deck = state.deck_data as DeckData | undefined;
  const hasDeck = deck && deck.headline && deck.hero_metric;

  const isPowerAnalysis = state.analysis_mode === "power_analysis";
  const { brief, details } = splitNarrative(state.narrative_draft);
  const hasDetails = details.length > 0 || (isPowerAnalysis && !!state.power_analysis_result);
  const hasCharts  = state.charts && state.charts.length > 0;
  const hasTrust   = state.trust_indicators && state.trust_indicators.confidence_level;

  const downloadPdf = async () => {
    try {
      await downloadRunPdf(runId);
    } catch {
      alert("Could not generate PDF.");
    }
  };

  const downloadCsv = () => {
    if (!state.charts?.length) return;
    const rows: string[] = [];
    for (const chart of state.charts) {
      rows.push(`# ${chart.title}`);
      if (!chart.data.length) { rows.push(""); continue; }
      const keys = Object.keys(chart.data[0]);
      rows.push(keys.join(","));
      for (const row of chart.data) {
        rows.push(keys.map(k => {
          const v = String((row as Record<string, unknown>)[k] ?? "");
          return v.includes(",") ? `"${v}"` : v;
        }).join(","));
      }
      rows.push("");
    }
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = "datapilot-analysis.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  const copyText = () => {
    navigator.clipboard.writeText(stripMarkdown(sanitiseNarrative(state.narrative_draft)));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFollowUp = async () => {
    const q = followUp.trim();
    if (!q || submitting) return;
    setSubmitting(true);
    try {
      onFollowUp(q);
      setFollowUp("");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={s.finPage} className="fade-in">
      <div style={s.finInner}>
        <div style={s.finHeader}>
          <div style={s.finTitle}>
            <span style={{ color: "var(--dp-success)", fontSize: 18, display: "inline-flex" }}><IconCheck /></span>
            <h2 style={{ color: "var(--dp-ink)", margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: "-0.01em" }}>Analysis complete</h2>
          </div>
          <div style={s.finActions} className="fin-actions">
            {(hasDetails || hasCharts) && (
              <button
                style={showDetails ? s.btnDetailsActive : s.btnDetails}
                onClick={() => setShowDetails(v => !v)}
              >
                {showDetails ? <><IconChevronUp /> Hide details</> : <><IconChevronDown /> Additional details</>}
              </button>
            )}
            <button style={s.btnSec} onClick={copyText}>{copied ? "Copied!" : "Copy text"}</button>
            <button style={s.btnSec} onClick={downloadPdf}><IconDownload /> Download PDF</button>
            {hasCharts && <button style={s.btnSec} onClick={downloadCsv}><IconDownload /> Export CSV</button>}
            <button style={s.btnSec} onClick={() => navigate("/history")}>History</button>
            <button style={s.btnPri} onClick={onNewAnalysis}>New Analysis</button>
          </div>
        </div>

        {hasTrust && <TrustBanner trust={state.trust_indicators} />}

        {/* Stakeholder deck — shown first; full report revealed on demand */}
        {hasDeck && !showFullReport && (
          <StakeholderDeck deck={deck!} onViewReport={() => setShowFullReport(true)} />
        )}

        {/* Full narrative — always shown when no deck, or when user clicked through */}
        {(!hasDeck || showFullReport) && (
          <>
            {hasDeck && showFullReport && (
              <button
                style={s.btnBackToDeck}
                onClick={() => setShowFullReport(false)}
              >
                Back to summary
              </button>
            )}
            <div style={s.narrativeCard} className="slide-up">
              <Markdown content={brief} />
            </div>
          </>
        )}

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

        <ChainOfThought steps={steps} isRunning={false} />

        {/* Follow-up input */}
        <div style={s.followUpBox} className="slide-up">
          <div style={s.followUpLabel}>Ask a follow-up question</div>
          <div style={s.followUpRow}>
            <textarea
              style={s.followUpInput}
              value={followUp}
              onChange={e => setFollowUp(e.target.value)}
              placeholder="e.g. Show me only Android users…"
              rows={2}
              onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleFollowUp(); } }}
            />
            <button
              style={{ ...s.followUpBtn, opacity: followUp.trim() && !submitting ? 1 : 0.45 }}
              onClick={handleFollowUp}
              disabled={!followUp.trim() || submitting}
            >
              {submitting ? "…" : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Exchange — one turn in the conversation thread ────────────────────────────

interface Exchange {
  task:         string;
  runId:        string;
  parentRunId:  string;
  result:       DoneEvent["state"] | null;
  steps:        import("../hooks/useSSE").StepEvent[];
  analysisMode: string;
}

// ── ActiveRun — SSE-connected single exchange ─────────────────────────────────

function ActiveRun({
  exchange,
  onDone,
  onError,
}: {
  exchange:  Exchange;
  onDone:    (state: DoneEvent["state"], steps: StepEvent[]) => void;
  onError:   (msg: string) => void;
}) {
  const [lastGate,         setLastGate]         = useState<string | null>(null);
  const [reconnectTrigger, setReconnectTrigger] = useState(0);
  const [submitting,       setSubmitting]       = useState(false);
  const [submitError,      setSubmitError]      = useState("");

  const { gate, done, error, steps, setGate } = useSSE(exchange.runId, reconnectTrigger);

  useTokenRefresh(() => setReconnectTrigger(n => n + 1));

  useEffect(() => {
    if (done) onDone(done.state, steps);
  }, [done]);

  useEffect(() => {
    if (error) onError(error);
  }, [error]);

  const analysisMode = (
    (gate?.payload as Record<string, unknown> | undefined)?.analysis_mode as string | undefined
  ) || exchange.analysisMode || "ab_test";

  const resume = async (value: object) => {
    if (!gate || submitting) return;
    setSubmitting(true); setSubmitError("");
    try {
      await client.post(`/runs/${exchange.runId}/resume`, { gate: gate.gate, value });
      setLastGate(gate.gate);
      setGate(null);
      setReconnectTrigger(n => n + 1);
    } catch (err) {
      setSubmitError(extractApiError(err, "Failed to submit."));
    } finally { setSubmitting(false); }
  };

  const errBanner = submitError ? (
    <div style={s.floatError} className="fade-in">
      <IconAlert /> {submitError}
      <button style={s.dismissBtn} onClick={() => setSubmitError("")} aria-label="Dismiss"><IconClose /></button>
    </div>
  ) : null;

  const payload = gate?.payload as Record<string, unknown> | undefined;

  const isRunning = !done && !gate;
  const processingLabel = submitting || (lastGate && !gate) ? "Processing your response…" : undefined;

  const renderGate = (el: React.ReactNode) => (
    <>
      {errBanner}
      <div style={s.gateContent}>
        <ErrorBoundary
          resetKey={gate?.gate}
          hint="The analysis is still paused at this gate. Reload the page to show it again."
        >
          {el}
        </ErrorBoundary>
      </div>
    </>
  );

  if (gate?.gate === "intent")
    return renderGate(<IntentGate payload={payload as Parameters<typeof IntentGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  if (gate?.gate === "semantic_cache")
    return renderGate(<SemanticCacheGate payload={payload as Parameters<typeof SemanticCacheGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  if (gate?.gate === "metric")
    return renderGate(<MetricGate payload={payload as Parameters<typeof MetricGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  if (gate?.gate === "query")
    return renderGate(<QueryGate payload={payload as Parameters<typeof QueryGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  if (gate?.gate === "analysis") {
    if (payload?.analysis_mode === "general")
      return renderGate(<GeneralAnalysisGate payload={payload as Parameters<typeof GeneralAnalysisGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
    return renderGate(<AnalysisGate payload={payload as Parameters<typeof AnalysisGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);
  }
  if (gate?.gate === "narrative")
    return renderGate(<NarrativeGate payload={payload as Parameters<typeof NarrativeGate>[0]["payload"]} onSubmit={resume} submitting={submitting} />);

  return (
    <div style={s.center}>
      <div style={{ width: "100%", maxWidth: 420 }} className="fade-in">
        {processingLabel && (
          <div style={s.processingBanner} className="fade-in">
            <span style={s.processingDot} />
            {processingLabel}
          </div>
        )}
        <div style={s.progressCard} className="dp-card">
          <PipelineProgress gate={gate?.gate ?? null} lastGate={lastGate} analysisMode={analysisMode} />
        </div>
        <ChainOfThought steps={steps} isRunning={isRunning} />
      </div>
    </div>
  );
}

// ── CollapsedExchange — past exchange shown as summary ────────────────────────

function CollapsedExchange({ exchange, onExpand }: { exchange: Exchange; onExpand: () => void }) {
  const rec = exchange.result?.recommendation ?? "";
  const truncTask = exchange.task.length > 80 ? exchange.task.slice(0, 80) + "…" : exchange.task;
  return (
    <div style={s.collapsedExchange}>
      <div style={s.collapsedLeft}>
        <span style={{ color: "var(--dp-ink-faint)", fontSize: 12 }}>Q:</span>
        <span style={{ color: "var(--dp-ink-secondary)", fontSize: 13 }}>{truncTask}</span>
      </div>
      {rec && <span style={s.collapsedRec}>{rec.length > 80 ? rec.slice(0, 80) + "…" : rec}</span>}
      <button style={s.expandBtn} onClick={onExpand}>Expand</button>
    </div>
  );
}

// ── Main Analysis page ─────────────────────────────────────────────────────────

export default function Analysis() {
  const navigate = useNavigate();

  const [selectedMode, setSelectedMode] = useState<Mode | null>(null);
  const [thread,       setThread]       = useState<Exchange[]>([]);
  const [expanded,     setExpanded]     = useState<Set<number>>(new Set());
  const [startError,   setStartError]   = useState("");
  const [username,     setUsername]     = useState("");
  const [runError,     setRunError]     = useState("");
  const [workspaces,   setWorkspaces]   = useState<WorkspaceSummary[]>([]);
  const [workspaceId,  setWorkspaceId]  = useState(getActiveWorkspaceId() || "");

  useTokenRefresh(() => {});

  useEffect(() => {
    client.get<{ username: string }>("/auth/me")
      .then(r => setUsername(r.data.username))
      .catch(() => {});
    client.get<{ workspaces: WorkspaceSummary[] }>("/workspaces")
      .then((r) => {
        const list = r.data.workspaces || [];
        setWorkspaces(list);
        if (!list.length) return;
        const saved = getActiveWorkspaceId();
        const active = list.find((w) => w.workspace_id === saved)?.workspace_id
          || list[0].workspace_id;
        setWorkspaceId(active);
        setActiveWorkspaceId(active);
      })
      .catch(() => {});
  }, []);

  const handleWorkspaceChange = (id: string) => {
    setWorkspaceId(id);
    setActiveWorkspaceId(id);
  };

  const startOver = () => {
    setThread([]); setSelectedMode(null); setStartError(""); setRunError(""); setExpanded(new Set());
  };

  const startRun = async (
    task: string,
    db_backend: string,
    pg?: PgCreds,
    uploadId?: string,
    parentRunId = "",
    opts?: RunOptions,
  ) => {
    setStartError("");
    try {
      const body: Record<string, unknown> = { task, db_backend };
      if (selectedMode) body.analysis_mode = selectedMode;
      if (uploadId) body.duckdb_path = uploadId;
      if (parentRunId) body.parent_run_id = parentRunId;
      if (opts?.connectionId) body.connection_id = opts.connectionId;
      if (opts?.metricPackId) body.metric_pack_id = opts.metricPackId;
      if (pg) {
        const defaultPort = db_backend === "mysql" ? 3306 : 5432;
        body.pg_host = pg.host;
        body.pg_port = parseInt(pg.port) || defaultPort;
        body.pg_dbname = pg.dbname;
        body.pg_user = pg.user;
        body.pg_password = pg.password;
      }
      if (opts?.bq) {
        body.bq_project_id = opts.bq.projectId;
        body.bq_dataset = opts.bq.dataset;
        body.bq_credentials_json = opts.bq.credentialsJson;
      }
      const { data } = await client.post("/runs", body);
      const newExchange: Exchange = {
        task, runId: data.run_id, parentRunId,
        result: null, steps: [], analysisMode: (selectedMode ?? ""),
      };
      setThread(prev => {
        const next = [...prev, newExchange];
        setExpanded(e => { const s2 = new Set(e); s2.add(next.length - 1); return s2; });
        return next;
      });
    } catch (err) {
      const msg = extractApiError(err, "Failed to start analysis.");
      setStartError(msg);
      throw err;
    }
  };

  const handleExchangeDone = (idx: number) => (state: DoneEvent["state"], steps: StepEvent[]) => {
    setThread(prev => prev.map((ex, i) => i === idx ? { ...ex, result: state, steps } : ex));
  };

  const handleExchangeError = (idx: number) => (msg: string) => {
    setRunError(msg);
  };

  const handleFollowUp = async (task: string) => {
    const lastDone = [...thread].reverse().find(ex => ex.result !== null);
    const parentRunId = lastDone?.runId ?? "";
    await startRun(task, "duckdb", undefined, undefined, parentRunId);
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  if (runError) {
    const isConnectionLost = runError.startsWith("Connection lost");
    return (
      <div style={s.center}>
        <div className="fade-in" style={{ textAlign: "center", maxWidth: 380 }}>
          <div style={{ fontSize: 28, marginBottom: 12, color: "var(--dp-danger)" }}><IconAlert /></div>
          <p style={{ color: "var(--dp-danger)", marginBottom: 8, fontSize: 15, fontWeight: 600 }}>
            {isConnectionLost ? "Connection lost" : "Something went wrong"}
          </p>
          <p style={{ color: "var(--dp-ink-muted)", marginBottom: 20, fontSize: 13 }}>{runError}</p>
          <button
            style={{ ...s.runBtn, background: "transparent", border: "1px solid var(--dp-ink-faint)", color: "var(--dp-ink-secondary)", width: "auto", padding: "10px 24px" }}
            onClick={startOver}
          >
            Start Over
          </button>
        </div>
      </div>
    );
  }

  if (!selectedMode && thread.length === 0)
    return (
      <ModeSelect
        onSelect={setSelectedMode}
        username={username}
        onHistory={() => navigate("/history")}
        onSignOut={async () => { await logout(); navigate("/login"); }}
        workspaces={workspaces}
        workspaceId={workspaceId}
        onWorkspaceChange={handleWorkspaceChange}
      />
    );

  if (selectedMode && thread.length === 0)
    return (
      <TaskInput
        mode={selectedMode}
        onSubmit={(task, db, pg, uploadId, opts) => startRun(task, db, pg, uploadId, "", opts)}
        onBack={() => setSelectedMode(null)}
        startError={startError}
      />
    );

  // Active run: show thread with last exchange running
  const activeMode = (thread[0]?.analysisMode || selectedMode || "ab_test") as string;
  const lastIdx = thread.length - 1;
  const lastExchange = thread[lastIdx];
  const isLastRunning = lastExchange && lastExchange.result === null;

  return (
    <div style={s.gatePage}>
      <GateBar task={lastExchange?.task ?? ""} mode={activeMode} onStartOver={startOver} />
      <div style={s.gateScroll}>
        {/* Past exchanges (all except last) */}
        {thread.slice(0, -1).map((ex, i) => {
          const isExpandedEx = expanded.has(i);
          return (
            <div key={ex.runId} style={s.pastExchange}>
              {isExpandedEx ? (
                <div>
                  <div style={s.exchangeTaskRow}>
                    <span style={s.exchangeQLabel}>Q:</span>
                    <span style={s.exchangeTask}>{ex.task}</span>
                    <button style={s.collapseBtn} onClick={() => setExpanded(prev => { const s2 = new Set(prev); s2.delete(i); return s2; })}>Collapse</button>
                  </div>
                  {ex.result && (
                    <>
                      <div style={{ ...s.narrativeCard, marginTop: 10 }}>
                        <Markdown content={sanitiseNarrative(ex.result.narrative_draft).split("<!-- details -->")[0].trim()} />
                      </div>
                      <ChainOfThought steps={ex.steps} isRunning={false} />
                    </>
                  )}
                </div>
              ) : (
                <CollapsedExchange exchange={ex} onExpand={() => setExpanded(prev => new Set([...prev, i]))} />
              )}
            </div>
          );
        })}

        {/* Current (last) exchange */}
        {lastExchange && (
          <div style={s.currentExchange}>
            {lastExchange.result ? (
              <FinishedView
                state={lastExchange.result}
                runId={lastExchange.runId}
                steps={lastExchange.steps}
                onNewAnalysis={startOver}
                onFollowUp={handleFollowUp}
              />
            ) : (
              <ActiveRun
                exchange={lastExchange}
                onDone={handleExchangeDone(lastIdx)}
                onError={handleExchangeError(lastIdx)}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  page:       { minHeight: "100vh", background: "var(--dp-bg)", padding: "0 20px 40px" },
  center:     { flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 20px" },

  header:     { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 0", maxWidth: 640, margin: "0 auto" },
  backBtn:    { background: "transparent", border: "none", color: "var(--dp-ink-muted)", fontSize: 13, cursor: "pointer", padding: 0 },
  modeBadge:  { fontSize: 12, fontWeight: 600, padding: "4px 12px", borderRadius: 20, border: "1px solid" },

  inputCard:  { background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 14, padding: "32px 32px 28px", maxWidth: 640, margin: "0 auto", boxShadow: "0 8px 40px rgba(11,31,42,0.06)" },
  heading:    { color: "var(--dp-ink)", fontSize: 22, fontWeight: 700, marginBottom: 4 },
  sub:        { color: "var(--dp-ink-muted)", fontSize: 13, marginBottom: 20 },
  taskInput:  { width: "100%", background: "var(--dp-surface-2)", color: "var(--dp-ink)", border: "1px solid var(--dp-line)", borderRadius: 8, padding: "12px 14px", fontSize: 14, resize: "vertical" as const, lineHeight: 1.6, marginBottom: 20, boxSizing: "border-box" as const },

  samplesSection: { marginBottom: 20 },
  samplesGrid:    { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 },
  sampleCard:     { display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", borderRadius: 8, cursor: "pointer", textAlign: "left" as const, transition: "border-color 0.15s", position: "relative" as const },
  sampleLoading:  { opacity: 0.6 },
  sampleIcon:     { width: 30, height: 30, borderRadius: 4, background: "var(--dp-accent-soft)", color: "var(--dp-accent)", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 600, letterSpacing: "0.02em", flexShrink: 0 },
  sampleLabel:    { color: "var(--dp-ink)", fontSize: 12, fontWeight: 600 },
  sampleDomain:   { color: "var(--dp-ink-muted)", fontSize: 11, marginTop: 1 },
  sampleSpinner:  { position: "absolute" as const, right: 10, width: 12, height: 12, border: "2px solid var(--dp-ink-faint)", borderTop: "2px solid var(--dp-accent)", borderRadius: "50%", animation: "spin 0.7s linear infinite" },

  section:      { marginBottom: 20 },
  sectionLabel: { color: "var(--dp-ink-secondary)", fontSize: 11, fontWeight: 600, textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: 8 },
  sourceRow:    { display: "flex", gap: 20, marginBottom: 12 },
  sourceOption: { display: "flex", alignItems: "center", gap: 6, color: "var(--dp-ink-secondary)", fontSize: 13, cursor: "pointer" },
  dbRow:        { display: "flex" },
  select:       { padding: "9px 12px", background: "var(--dp-surface-2)", color: "var(--dp-ink)", border: "1px solid var(--dp-line)", borderRadius: 8, fontSize: 13, cursor: "pointer" },

  uploadBox:    { display: "flex", flexDirection: "column" as const, gap: 8, padding: 14, background: "var(--dp-surface-2)", borderRadius: 8, border: "1px dashed var(--dp-ink-faint)" },
  uploadBtn:    { alignSelf: "flex-start", padding: "8px 16px", background: "var(--dp-line)", color: "var(--dp-ink)", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13 },
  uploadSuccess: { display: "flex", gap: 8, alignItems: "center", fontSize: 13, flexWrap: "wrap" as const },

  pgGrid:  { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 10, padding: 14, background: "var(--dp-surface-2)", borderRadius: 8, border: "1px solid var(--dp-line)" },
  pgField: { display: "flex", flexDirection: "column" as const, gap: 4 },
  pgLabel: { fontSize: 11, color: "var(--dp-ink-muted)", fontWeight: 600, textTransform: "uppercase" as const },
  pgInput: { padding: "8px 10px", background: "var(--dp-surface)", color: "var(--dp-ink)", border: "1px solid var(--dp-line)", borderRadius: 6, fontSize: 13 },

  errorBox:   { background: "var(--dp-danger)11", border: "1px solid var(--dp-danger)44", color: "var(--dp-danger)", borderRadius: 8, padding: "10px 14px", fontSize: 13, marginBottom: 14 },
  runBtn:     { width: "100%", padding: "13px 0", color: "var(--dp-surface)", border: "none", borderRadius: 8, fontWeight: 700, fontSize: 15, cursor: "pointer", marginTop: 4, display: "flex", alignItems: "center", justifyContent: "center", gap: 8 },
  btnSpinner: { width: 14, height: 14, border: "2px solid var(--dp-surface)44", borderTop: "2px solid var(--dp-surface)", borderRadius: "50%", animation: "spin 0.7s linear infinite", display: "inline-block", flexShrink: 0 },

  gatePage:   { minHeight: "100vh", background: "var(--dp-bg)", display: "flex", flexDirection: "column" as const },
  gateScroll: { flex: 1, display: "flex", flexDirection: "column" as const },
  gateContent:{ padding: "36px 20px 60px", flex: 1 },

  progressCard:      { background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 14, padding: "16px 24px", boxShadow: "0 8px 40px rgba(11,31,42,0.06)" },
  processingBanner:  { display: "flex", alignItems: "center", gap: 10, color: "var(--dp-ink-secondary)", fontSize: 13, padding: "10px 16px", background: "var(--dp-surface-2)", borderRadius: 8, marginBottom: 12, border: "1px solid var(--dp-line)" },
  processingDot:     { width: 8, height: 8, borderRadius: "50%", background: "var(--dp-accent)", animation: "pulse 1.2s ease-in-out infinite", flexShrink: 0 },

  floatError: { position: "fixed", top: 20, left: "50%", transform: "translateX(-50%)", background: "var(--dp-danger)", color: "var(--dp-surface)", padding: "10px 20px", borderRadius: 8, display: "flex", gap: 12, alignItems: "center", zIndex: 1000, fontSize: 14, boxShadow: "0 4px 20px rgba(11,31,42,0.12)", whiteSpace: "nowrap" as const },
  dismissBtn: { background: "transparent", border: "none", color: "var(--dp-surface)", cursor: "pointer", fontWeight: 700, fontSize: 16, padding: 0 },

  finPage:    { minHeight: "100vh", background: "var(--dp-bg)", padding: "32px 20px 60px" },
  finInner:   { maxWidth: 820, margin: "0 auto" },
  finHeader:  { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap" as const, gap: 12 },
  finTitle:   { display: "flex", alignItems: "center", gap: 10 },
  finActions: { display: "flex", gap: 8 },
  btnSec:          { padding: "7px 14px", background: "var(--dp-surface)", color: "var(--dp-ink)", border: "1px solid var(--dp-line-strong)", borderRadius: 6, cursor: "pointer", fontSize: 13, display: "inline-flex", alignItems: "center", gap: 6 },
  btnPri:          { padding: "7px 14px", background: "var(--dp-accent)", color: "#FFFFFF", border: "none", borderRadius: 6, cursor: "pointer", fontSize: 13, fontWeight: 600 },
  btnDetails:      { padding: "7px 14px", background: "transparent", color: "var(--dp-accent)", border: "1px solid var(--dp-accent)44", borderRadius: 6, cursor: "pointer", fontSize: 13, display: "inline-flex", alignItems: "center", gap: 6 },
  btnDetailsActive:{ padding: "7px 14px", background: "var(--dp-accent-soft)", color: "var(--dp-accent)", border: "1px solid var(--dp-accent)66", borderRadius: 6, cursor: "pointer", fontSize: 13, display: "inline-flex", alignItems: "center", gap: 6 },

  trustBanner: { border: "1px solid", borderRadius: 10, padding: "12px 16px", marginBottom: 16, display: "flex", flexDirection: "column" as const, gap: 6 },

  chartsSection:      { marginBottom: 24, marginTop: 14 },
  chartsSectionLabel: { color: "var(--dp-ink-faint)", fontSize: 11, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.08em", marginBottom: 12 },
  chartsGrid:         { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 },

  narrativeCard:  { background: "var(--dp-surface)", border: "1px solid var(--dp-line)", borderRadius: 8, padding: "28px 32px", lineHeight: 1.7 },
  btnBackToDeck:  { background: "transparent", border: "none", color: "var(--dp-accent)", fontSize: 13, cursor: "pointer", padding: "0 0 12px", display: "block" },

  followUpBox:   { marginTop: 20, padding: "16px 20px", background: "var(--dp-surface-2)", border: "1px solid var(--dp-line)", borderRadius: 10 },
  followUpLabel: { color: "var(--dp-ink-muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase" as const, letterSpacing: "0.06em", marginBottom: 10 },
  followUpRow:   { display: "flex", gap: 10, alignItems: "flex-end" },
  followUpInput: { flex: 1, background: "var(--dp-bg)", color: "var(--dp-ink)", border: "1px solid var(--dp-line)", borderRadius: 8, padding: "10px 12px", fontSize: 13, resize: "none" as const, lineHeight: 1.5, fontFamily: "inherit" },
  followUpBtn:   { padding: "10px 18px", background: "var(--dp-accent)", color: "#FFFFFF", border: "none", borderRadius: 6, fontWeight: 600, fontSize: 13, cursor: "pointer", flexShrink: 0, height: 42 },

  pastExchange:      { borderBottom: "1px solid var(--dp-surface-2)", padding: "12px 20px" },
  currentExchange:   { flex: 1 },
  exchangeTaskRow:   { display: "flex", alignItems: "center", gap: 8 },
  exchangeQLabel:    { color: "var(--dp-ink-faint)", fontSize: 12, flexShrink: 0 },
  exchangeTask:      { color: "var(--dp-ink-secondary)", fontSize: 13, flex: 1 },
  collapseBtn:       { background: "transparent", border: "none", color: "var(--dp-ink-faint)", fontSize: 12, cursor: "pointer", flexShrink: 0 },

  collapsedExchange: { display: "flex", alignItems: "center", gap: 12, padding: "10px 0" },
  collapsedLeft:     { display: "flex", gap: 6, alignItems: "center", flex: 1, minWidth: 0 },
  collapsedRec:      { color: "var(--dp-ink-muted)", fontSize: 12, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" as const },
  expandBtn:         { background: "transparent", border: "1px solid var(--dp-line)", color: "var(--dp-ink-muted)", fontSize: 12, borderRadius: 6, padding: "3px 10px", cursor: "pointer", flexShrink: 0 },
};
