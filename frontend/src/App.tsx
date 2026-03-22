import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import Login from "./pages/Login";
import ForgotPassword from "./pages/ForgotPassword";
import ResetPassword from "./pages/ResetPassword";
import Analysis from "./pages/Analysis";
import History from "./pages/History";

function AuthGuard({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const token = localStorage.getItem("access_token");
  if (!token) return <Navigate to="/login" state={{ from: location }} replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <style>{globalCss}</style>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/" element={<AuthGuard><Analysis /></AuthGuard>} />
        <Route path="/history" element={<AuthGuard><History /></AuthGuard>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

const globalCss = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #11111b; color: #cdd6f4; }

  @keyframes spin    { to { transform: rotate(360deg); } }
  @keyframes fadeIn  { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes slideUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse   { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  @keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
  @keyframes glow    { 0%,100% { box-shadow: 0 0 6px #89b4fa55; } 50% { box-shadow: 0 0 18px #89b4faaa; } }
  @keyframes gradMove{ 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

  .fade-in  { animation: fadeIn  0.35s ease both; }
  .slide-up { animation: slideUp 0.4s ease both; }

  input:focus, textarea:focus, select:focus {
    outline: none;
    border-color: #89b4fa !important;
    box-shadow: 0 0 0 2px #89b4fa22;
  }
  button:active { transform: scale(0.97); }
  button { transition: opacity 0.15s, transform 0.1s; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #181825; }
  ::-webkit-scrollbar-thumb { background: #45475a; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #585b70; }

  /* Markdown narrative styles */
  .md h1 { color: #cdd6f4; font-size: 1.15rem; font-weight: 700; margin: 0 0 0.8em; padding-bottom: 6px; border-bottom: 1px solid #45475a; }
  .md h2 { color: #89b4fa; font-size: 1rem; margin: 1.2em 0 0.4em; padding-bottom: 4px; border-bottom: 1px solid #313244; }
  .md h3 { color: #cba6f7; font-size: 0.95rem; margin: 1em 0 0.3em; }
  .md p  { color: #cdd6f4; line-height: 1.65; margin-bottom: 0.6em; }
  .md ul { padding-left: 1.4em; margin-bottom: 0.6em; }
  .md li { color: #a6adc8; line-height: 1.6; margin-bottom: 2px; }
  .md strong { color: #cdd6f4; font-weight: 600; }
  .md em    { color: #f9e2af; font-style: normal; }
  .md code  { background: #313244; color: #a6e3a1; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }

  /* ── Mobile layout ─────────────────────────────────── */
  @media (max-width: 600px) {
    /* Sample cards: single column */
    .samples-grid { grid-template-columns: 1fr !important; }

    /* Finished view action buttons: wrap */
    .fin-actions { flex-wrap: wrap !important; }
    .fin-actions button { flex: 1 1 auto; min-width: 80px; }

    /* Charts grid: single column on mobile */
    .charts-grid { grid-template-columns: 1fr !important; }

    /* Reduce card padding on small screens */
    .slide-up { padding-left: 0 !important; padding-right: 0 !important; }
  }
`;
