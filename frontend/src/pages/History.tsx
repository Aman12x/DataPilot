import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import client, { API_BASE, logout } from "../api/client";
import Markdown from "../components/Markdown";
import Spinner from "../components/Spinner";

interface Run {
  run_id:    string;
  task:      string;
  timestamp: string;
  eval_score?: number;
  metric?:   string;
  analysis_mode?: string;
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

  useEffect(() => {
    client.get("/runs?limit=20")
      .then(({ data }) => setRuns(data))
      .catch(() => setError("Could not load history"))
      .finally(() => setLoading(false));
  }, []);

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
      const d = detail[run.run_id] ?? (await client.get(`/runs/${run.run_id}/detail`)).data;
      const token  = localStorage.getItem("access_token") ?? "";
      const params = new URLSearchParams({ token, narrative: d.narrative ?? "", recommendation: d.recommendation ?? "", task: d.task ?? run.task });
      window.open(`${API_BASE}/runs/${run.run_id}/pdf?${params}`, "_blank");
    } catch {
      alert("Could not load run details for PDF.");
    } finally { setDownloading(null); }
  };

  const scoreColor = (sc: number) => sc >= 0.8 ? "#a6e3a1" : sc >= 0.6 ? "#f9e2af" : "#f38ba8";
  const scoreLabel = (sc: number) => sc >= 0.8 ? "High quality" : sc >= 0.6 ? "Good" : "Needs review";

  return (
    <div style={s.outer}>
      <div style={s.orb1} />
      <div style={s.orb2} />

      <div style={s.inner}>
        <div style={s.header} className="fade-in">
          <div style={s.titleGroup}>
            <span style={s.logoIcon}>✦</span>
            <div>
              <h1 style={s.title}>Analysis History</h1>
              <p style={s.subtitle}>Your past DataPilot runs</p>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button style={s.newBtn} onClick={() => navigate("/")}>+ New Analysis</button>
            <button style={s.logoutBtn} onClick={async () => { await logout(); navigate("/login"); }}>Sign out</button>
          </div>
        </div>

        {loading && (
          <div style={s.emptyState} className="fade-in">
            <Spinner variant="page" />
            <p style={s.emptyText}>Loading history…</p>
          </div>
        )}
        {error && <div style={s.errorBox} className="fade-in"><span>⚠</span> {error}</div>}
        {!loading && !error && runs.length === 0 && (
          <div style={s.emptyState} className="fade-in">
            <div style={s.emptyIcon}>📊</div>
            <p style={s.emptyText}>No analyses yet</p>
            <p style={s.emptySub}>Run your first analysis to see it here</p>
            <button style={s.newBtn} onClick={() => navigate("/")}>Start Analysing</button>
          </div>
        )}

        <div style={s.cardList}>
          {runs.map((r, idx) => {
            const isOpen = expanded === r.run_id;
            const det    = detail[r.run_id];
            return (
              <div
                key={r.run_id}
                style={{ ...s.card, animationDelay: `${idx * 40}ms`, ...(isOpen ? s.cardOpen : {}) }}
                className="fade-in"
                onClick={() => toggleExpand(r)}
              >
                <div style={s.cardTop}>
                  <span style={{
                    ...s.modeBadge,
                    background:  r.analysis_mode === "general" ? "#cba6f722" : "#89b4fa22",
                    color:       r.analysis_mode === "general" ? "#cba6f7"   : "#89b4fa",
                    borderColor: r.analysis_mode === "general" ? "#cba6f744" : "#89b4fa44",
                  }}>
                    {r.analysis_mode === "general" ? "💡 General" : "🧪 A/B Test"}
                  </span>

                  {r.eval_score != null && (
                    <div style={s.scoreChip}>
                      <div style={{ ...s.scoreDot, background: scoreColor(r.eval_score) }} />
                      <span style={{ color: scoreColor(r.eval_score), fontSize: 12, fontWeight: 600 }}>
                        {(r.eval_score * 100).toFixed(0)}% — {scoreLabel(r.eval_score)}
                      </span>
                    </div>
                  )}
                </div>

                <p style={s.task}>{r.task}</p>

                <div style={s.meta}>
                  <span style={s.metaItem}>
                    🕐 {new Date(r.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </span>
                  {r.metric && <span style={s.metricBadge}>{r.metric}</span>}
                  <span style={s.runId}>{r.run_id.slice(0, 8)}</span>
                  <span style={{ ...s.metaItem, marginLeft: "auto" }}>{isOpen ? "▲ collapse" : "▼ view analysis"}</span>
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
                            style={s.pdfBtn}
                            onClick={(e) => downloadPdf(e, r)}
                            disabled={downloading === r.run_id}
                          >
                            {downloading === r.run_id
                              ? <><Spinner variant="inline" /> Loading…</>
                              : "↓ PDF Report"}
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
    </div>
  );
}

const s: Record<string, React.CSSProperties> = {
  outer:       { minHeight: "100vh", background: "#11111b", padding: "40px 20px", position: "relative", overflow: "hidden" },
  orb1:        { position: "fixed", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, #89b4fa0d 0%, transparent 70%)", top: -200, right: -200, pointerEvents: "none" },
  orb2:        { position: "fixed", width: 400, height: 400, borderRadius: "50%", background: "radial-gradient(circle, #cba6f70d 0%, transparent 70%)", bottom: -150, left: -100, pointerEvents: "none" },
  inner:       { maxWidth: 760, margin: "0 auto", position: "relative", zIndex: 1 },

  header:      { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 },
  titleGroup:  { display: "flex", alignItems: "center", gap: 14 },
  logoIcon:    { fontSize: 28, color: "#89b4fa" },
  title:       { color: "#cdd6f4", fontSize: 22, fontWeight: 700, margin: 0 },
  subtitle:    { color: "#585b70", fontSize: 13, marginTop: 2 },

  newBtn:      { padding: "10px 20px", background: "linear-gradient(135deg, #89b4fa, #74c7ec)", color: "#1e1e2e", border: "none", borderRadius: 8, fontWeight: 700, cursor: "pointer", fontSize: 13 },
  logoutBtn:   { padding: "10px 16px", background: "transparent", color: "#585b70", border: "1px solid #313244", borderRadius: 8, fontWeight: 500, cursor: "pointer", fontSize: 13 },

  errorBox:    { background: "#f38ba811", border: "1px solid #f38ba844", color: "#f38ba8", borderRadius: 8, padding: "12px 16px", display: "flex", alignItems: "center", gap: 8, marginBottom: 20 },

  emptyState:  { textAlign: "center" as const, padding: "80px 20px", display: "flex", flexDirection: "column" as const, alignItems: "center", gap: 12 },
  emptyIcon:   { fontSize: 48, marginBottom: 8 },
  emptyText:   { color: "#cdd6f4", fontSize: 18, fontWeight: 600 },
  emptySub:    { color: "#585b70", fontSize: 14 },

  cardList:    { display: "flex", flexDirection: "column" as const, gap: 12 },
  card:        { background: "#1e1e2e", border: "1px solid #313244", borderRadius: 12, padding: "18px 20px", cursor: "pointer", transition: "border-color 0.2s, box-shadow 0.2s" },
  cardOpen:    { borderColor: "#89b4fa55", boxShadow: "0 4px 24px #89b4fa11" },

  cardTop:     { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  modeBadge:   { fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 20, border: "1px solid" },
  scoreChip:   { display: "flex", alignItems: "center", gap: 6 },
  scoreDot:    { width: 8, height: 8, borderRadius: "50%" },

  task:        { color: "#cdd6f4", fontWeight: 600, fontSize: 15, marginBottom: 10, lineHeight: 1.4 },

  meta:        { display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" as const },
  metaItem:    { color: "#585b70", fontSize: 12 },
  metricBadge: { background: "#313244", color: "#cba6f7", fontSize: 11, padding: "2px 8px", borderRadius: 4, fontWeight: 500 },
  runId:       { color: "#45475a", fontSize: 11, fontFamily: "monospace" },

  narrativeSection: { marginTop: 0 },
  narrativeDivider: { height: 1, background: "#313244", margin: "16px 0" },
  loadingRow:       { color: "#585b70", fontSize: 13, display: "flex", alignItems: "center", gap: 8, padding: "8px 0" },
  recBanner:        { background: "linear-gradient(135deg, #1a2035, #1a2820)", border: "1px solid #89b4fa33", borderRadius: 8, padding: "12px 16px", marginBottom: 16 },
  recLabel:         { color: "#89b4fa", fontSize: 10, fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: "0.1em", marginBottom: 4 },
  recText:          { color: "#cdd6f4", fontWeight: 600, fontSize: 14, lineHeight: 1.5, margin: 0 },
  narrativeBody:    { background: "#181825", borderRadius: 8, padding: "16px 20px", lineHeight: 1.7, marginBottom: 16 },
  actions:          { display: "flex", gap: 8 },
  pdfBtn:           { padding: "6px 14px", background: "transparent", color: "#89b4fa", border: "1px solid #89b4fa44", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 500, display: "flex", alignItems: "center", gap: 6 },
};
