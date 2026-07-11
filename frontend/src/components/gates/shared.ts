import type React from "react";

export const gateCard: React.CSSProperties = {
  background: "#1e1e2e", border: "1px solid #313244",
  borderRadius: 16, padding: "32px 36px",
  margin: "0 auto", boxShadow: "0 12px 40px rgba(0,0,0,0.35)",
  maxWidth: 720,
};

export const gateTitle: React.CSSProperties = {
  color: "#cdd6f4", marginTop: 0, fontSize: 20, fontWeight: 700, letterSpacing: "-0.02em",
};

export const gateMessage: React.CSSProperties = {
  color: "#a6adc8", fontSize: 14, lineHeight: 1.6,
};

export const gateTextarea: React.CSSProperties = {
  width: "100%", background: "#181825", color: "#cdd6f4",
  border: "1px solid #313244", borderRadius: 8,
  padding: "11px 14px", fontSize: 14, fontFamily: "inherit",
  resize: "vertical", boxSizing: "border-box", lineHeight: 1.5,
};

export const gateActions: React.CSSProperties = {
  display: "flex", gap: 12, marginTop: 20, flexWrap: "wrap",
};

export const gateBtnApprove: React.CSSProperties = {
  padding: "11px 24px",
  background: "linear-gradient(135deg, #a6e3a1, #94e2d5)",
  color: "#1e1e2e",
  border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 14,
};

export const gateBtnSecondary: React.CSSProperties = {
  padding: "11px 24px", background: "transparent",
  color: "#a6adc8", border: "1px solid #45475a",
  borderRadius: 8, cursor: "pointer", fontSize: 14, fontWeight: 500,
};

export const gateBtnClass = "dp-btn";
