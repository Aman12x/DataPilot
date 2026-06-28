import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import Login from "./pages/Login";
import ForgotPassword from "./pages/ForgotPassword";
import ResetPassword from "./pages/ResetPassword";
import VerifyEmail from "./pages/VerifyEmail";
import Analysis from "./pages/Analysis";
import History from "./pages/History";
import { checkAuth } from "./api/client";
import { useEffect, useState } from "react";

function AuthGuard({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const [status, setStatus] = useState<"loading" | "ok" | "denied">("loading");

  useEffect(() => {
    let cancelled = false;
    checkAuth().then((ok) => {
      if (!cancelled) setStatus(ok ? "ok" : "denied");
    });
    return () => { cancelled = true; };
  }, []);

  if (status === "loading") {
    return (
      <div className="dp-loading-screen fade-in">
        <div className="dp-spinner" />
        <p className="dp-loading-text">Loading DataPilot…</p>
      </div>
    );
  }
  if (status === "denied") {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return <>{children}</>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/verify-email" element={<VerifyEmail />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/" element={<AuthGuard><Analysis /></AuthGuard>} />
        <Route path="/history" element={<AuthGuard><History /></AuthGuard>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
