import type { ReactNode } from "react";
import { Link } from "react-router-dom";

export function BrandMark({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const scale = size === "lg" ? 1.15 : size === "sm" ? 0.85 : 1;
  return (
    <span className="dp-shell-brand" style={{ transform: `scale(${scale})`, transformOrigin: "left center" }}>
      <span className="dp-mark" aria-hidden="true" />
      <span className="dp-wordmark">DataPilot</span>
    </span>
  );
}

export default function AppShell({
  children,
  nav,
  narrow = false,
  brandTo = "/",
}: {
  children: ReactNode;
  nav?: ReactNode;
  narrow?: boolean;
  brandTo?: string;
}) {
  return (
    <div className="dp-shell">
      <header className="dp-shell-bar">
        <Link to={brandTo} className="dp-shell-brand" aria-label="DataPilot home">
          <span className="dp-mark" aria-hidden="true" />
          <span className="dp-wordmark">DataPilot</span>
        </Link>
        {nav && <nav className="dp-shell-nav">{nav}</nav>}
      </header>
      <main className={`dp-shell-main${narrow ? " dp-shell-main--narrow" : ""}`}>
        {children}
      </main>
    </div>
  );
}
