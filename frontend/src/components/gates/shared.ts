import type React from "react";

export const gateCard: React.CSSProperties = {
  background: "var(--dp-surface)",
  border: "1px solid var(--dp-line)",
  borderRadius: 10,
  padding: "28px 32px",
  margin: "0 auto",
  boxShadow: "var(--dp-shadow)",
  maxWidth: 720,
};

export const gateTitle: React.CSSProperties = {
  color: "var(--dp-ink)",
  marginTop: 0,
  fontSize: 18,
  fontWeight: 600,
  letterSpacing: "-0.01em",
};

export const gateMessage: React.CSSProperties = {
  color: "var(--dp-ink-secondary)",
  fontSize: 14,
  lineHeight: 1.6,
};

export const gateTextarea: React.CSSProperties = {
  width: "100%",
  background: "var(--dp-surface)",
  color: "var(--dp-ink)",
  border: "1px solid var(--dp-line)",
  borderRadius: 6,
  padding: "11px 14px",
  fontSize: 14,
  fontFamily: "inherit",
  resize: "vertical",
  boxSizing: "border-box",
  lineHeight: 1.5,
};

export const gateActions: React.CSSProperties = {
  display: "flex", gap: 12, marginTop: 20, flexWrap: "wrap",
};

export const gateBtnApprove: React.CSSProperties = {
  padding: "10px 22px",
  background: "var(--dp-accent)",
  color: "var(--dp-accent-ink)",
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
  fontWeight: 600,
  fontSize: 14,
};

export const gateBtnSecondary: React.CSSProperties = {
  padding: "10px 22px",
  background: "transparent",
  color: "var(--dp-ink-secondary)",
  border: "1px solid var(--dp-line-strong)",
  borderRadius: 6,
  cursor: "pointer",
  fontSize: 14,
  fontWeight: 500,
};

export const gateBtnClass = "dp-btn";
