import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import client, {
  logout,
  getActiveWorkspaceId, setActiveWorkspaceId, type WorkspaceSummary,
} from "../api/client";
import { downloadRunPdf } from "../utils/downloadPdf";
import Markdown from "../components/Markdown";
import Spinner from "../components/Spinner";
import AppShell from "../components/AppShell";

interface Run {
  run_id:    string;
  task:      string;
  timestamp: string;
  eval_score?: number;
  metric?:   string;
  analysis_mode?: string;
  user_id?: string;
  username?: string;
  workspace_id?: string;
}

interface RunDetail {
  narrative:      string;
  recommendation: string;
  task:           string;
}

export default function History() {
  const navigate              = useNavigate();
  const [runs, setRuns]       = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");
  const [expanded,    setExpanded]    = useState<string | null>(null);
  const [detail,      setDetail]      = useState<Record<string, RunDetail>>({});
  const [loadingDetail, setLoadingDetail] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [workspaceId, setWorkspaceId] = useState(getActiveWorkspaceId() || "");

  const loadRuns = () => {
    setLoading(true);
    setError("");
    client.get("/runs?limit=20")
      .then(({ data }) => setRuns(Array.isArray(data) ? data : []))
      .catch(() => setError("Could not load history"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
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
      .catch(() => {})
      .finally(() => loadRuns());
  }, []);

  const handleWorkspaceChange = (id: string) => {
    setWorkspaceId(id);
    setActiveWorkspaceId(id);
    setExpanded(null);
    setDetail({});
    loadRuns();
  };

  const toggleExpand = async (run: Run) => {
    if (expanded === run.run_id) { setExpanded(null); return; }
    setExpanded(run.run_id);
    if (detail[run.run_id]) return;
    setLoadingDetail(run.run_id);
    try {
      const { data } = await client.get(`/runs/${run.run_id}/detail`);
      setDetail(d => ({ ...d, [run.run_id]: data }));
    } catch {
      setDetail(d => ({ ...d, [run.run_id]: { narrative: "_Could not load analysis._", recommendation: "", task: run.task } }));
    } finally { setLoadingDetail(null); }
  };

  const downloadPdf = async (e: React.MouseEvent, run: Run) => {
    e.stopPropagation();
    setDownloading(run.run_id);
    try {
      await downloadRunPdf(run.run_id);
    } catch {
      alert("Could not generate PDF.");
    } finally { setDownloading(null); }
  };

  const scoreColor = (sc: number) =>
    sc >= 0.8 ? "var(--dp-success)" : sc >= 0.6 ? "var(--dp-warning)" : "var(--dp-danger)";
  const scoreLabel = (sc: number) => sc >= 0.8 ? "High quality" : sc >= 0.6 ? "Good" : "Needs review";

  return (
    <AppShell
      brandTo="/"
      nav={(
        <>
          {workspaces.length > 0 && (
            <select
              aria-label="Workspace"
              value={workspaceId}
              onChange={(e) => handleWorkspaceChange(e.target.value)}
              className="dp-shell-select"
            >
              {workspaces.map((w) => (
                <option key={w.workspace_id} value={w.workspace_id}>{w.name}</option>
              ))}
            </select>
          )}
          <button className="dp-btn dp-btn-primary" onClick={() => navigate("/")}>New analysis</button>
          <button className="dp-btn dp-btn-ghost" onClick={async () => { await logout(); navigate("/login"); }}>Sign out</button>
        </>
      )}
    >
      <div className="fade-in" style={{ maxWidth: 820, margin: "0 auto" }}>
        <div style={{ marginBottom: 28 }}>
          <h1 style={{
            fontSize: 24, fontWeight: 600,
            letterSpacing: "-0.01em", color: "var(--dp-ink)", margin: 0,
          }}>
            Analysis history
          </h1>
          <p style={{ color: "var(--dp-ink-muted)", fontSize: 14, marginTop: 6 }}>
            {workspaces.length ? "Shared workspace runs" : "Your past DataPilot runs"}
          </p>
        </div>

        {loading && (
          <div style={s.emptyState}>
            <Spinner variant="page" />
            <p style={s.emptyText}>Loading history…</p>
          </div>
        )}
        {error && <div style={s.errorBox}>{error}</div>}
        {!loading && !error && runs.length === 0 && (
          <div style={s.emptyState} className="dp-panel">
            <p style={s.emptyText}>No analyses yet</p>
            <p style={s.emptySub}>Run your first analysis to see it here</p>
            <button className="dp-btn dp-btn-primary" style={{ marginTop: 8 }} onClick={() => navigate("/")}>
              Start analysing
            </button>
          </div>
        )}

        <div style={s.cardList}>
          {runs.map((r, idx) => {
            const isOpen = expanded === r.run_id;
            const det    = detail[r.run_id];
            return (
              <div
                key={r.run_id}
                className={`dp-card dp-history-card fade-in${isOpen ? " dp-card-open" : ""}`}
                style={{ ...s.card, animationDelay: `${idx * 40}ms` }}
                onClick={() => toggleExpand(r)}
              >
                <div style={s.cardTop}>
                  <span className="dp-badge">
                    {r.analysis_mode === "general" ? "Explore" : "A/B Test"}
                  </span>

                  {r.eval_score != null && (
                    <div style={s.scoreChip}>
                      <div style={{ ...s.scoreDot, background: scoreColor(r.eval_score) }} />
                      <span style={{ color: scoreColor(r.eval_score), fontSize: 12, fontWeight: 600 }}>
                        {(r.eval_score * 100).toFixed(0)}% {scoreLabel(r.eval_score)}
                      </span>
                    </div>
                  )}
                </div>

                <p style={s.task}>{r.task}</p>

                <div style={s.meta}>
                  <span style={s.metaItem}>
                    {new Date(r.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </span>
                  {r.username && <span style={s.metaItem}>by {r.username}</span>}
                  {r.metric && <span style={s.metricBadge}>{r.metric}</span>}
                  <span style={s.runId}>{r.run_id.slice(0, 8)}</span>
                  <span style={{ ...s.metaItem, marginLeft: "auto" }}>{isOpen ? "Collapse" : "View analysis"}</span>
                </div>

                {isOpen && (
                  <div style={s.narrativeSection} className="fade-in" onClick={e => e.stopPropagation()}>
                    <div style={s.narrativeDivider} />
                    {loadingDetail === r.run_id ? (
                      <div style={s.loadingRow}><Spinner variant="inline" /> Loading analysis…</div>
                    ) : det ? (
                      <>
                        {det.recommendation && (
                          <div style={s.recBanner}>
                            <div style={s.recLabel}>Recommendation</div>
                            <p style={s.recText}>{det.recommendation}</p>
                          </div>
                        )}
                        <div style={s.narrativeBody}>
                          <Markdown content={det.narrative} />
                        </div>
                        <div style={s.actions}>
                          <button
                            className="dp-btn dp-btn-ghost"
                            onClick={(e) => downloadPdf(e, r)}
                            disabled={downloading === r.run_id}
                          >
                            {downloading === r.run_id
                              ? <><Spinner variant="inline" /> Loading…</>
                              : "PDF report"}
                          </button>
                        </div>
                      </>
                    ) : null}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </AppShell>
  );
}

const s: Record<string, React.CSSProperties> = {
  errorBox:    { background: "var(--dp-danger-soft)", border: "1px solid rgba(194,59,74,0.28)", color: "var(--dp-danger)", borderRadius: 10, padding: "12px 16px", marginBottom: 20 },
  emptyState:  { textAlign: "center", padding: "64px 20px", display: "flex", flexDirection: "column", alignItems: "center", gap: 10 },
  emptyText:   { color: "var(--dp-ink)", fontSize: 17, fontWeight: 600 },
  emptySub:    { color: "var(--dp-ink-muted)", fontSize: 14 },
  cardList:    { display: "flex", flexDirection: "column", gap: 12 },
  card:        { padding: "18px 20px", cursor: "pointer" },
  cardTop:     { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  scoreChip:   { display: "flex", alignItems: "center", gap: 6 },
  scoreDot:    { width: 8, height: 8, borderRadius: "50%" },
  task:        { color: "var(--dp-ink)", fontWeight: 600, fontSize: 15, marginBottom: 10, lineHeight: 1.4 },
  meta:        { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" },
  metaItem:    { color: "var(--dp-ink-muted)", fontSize: 12 },
  metricBadge: { background: "var(--dp-accent-soft)", color: "var(--dp-accent)", fontSize: 11, padding: "2px 8px", borderRadius: 4, fontWeight: 600 },
  runId:       { color: "var(--dp-ink-faint)", fontSize: 11, fontFamily: "var(--dp-mono)" },
  narrativeSection: { marginTop: 0 },
  narrativeDivider: { height: 1, background: "var(--dp-line)", margin: "16px 0" },
  loadingRow:       { color: "var(--dp-ink-muted)", fontSize: 13, display: "flex", alignItems: "center", gap: 8, padding: "8px 0" },
  recBanner:        { background: "var(--dp-info-soft)", border: "1px solid rgba(45,98,196,0.22)", borderRadius: 6, padding: "12px 16px", marginBottom: 16 },
  recLabel:         { color: "var(--dp-info)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 },
  recText:          { color: "var(--dp-ink)", fontWeight: 600, fontSize: 14, lineHeight: 1.5, margin: 0 },
  narrativeBody:    { background: "var(--dp-surface-2)", borderRadius: 6, padding: "16px 20px", lineHeight: 1.7, marginBottom: 16 },
  actions:          { display: "flex", gap: 8 },
};
