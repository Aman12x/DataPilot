import type React from "react";

export const gateCard: React.CSSProperties = {
  background: "#1e1e2e", border: "1px solid #313244",
  borderRadius: 14, padding: "28px 32px",
  margin: "0 auto", boxShadow: "0 8px 40px #00000044",
};

export const gateTitle: React.CSSProperties = {
  color: "#cdd6f4", marginTop: 0, fontSize: 18, fontWeight: 700,
};

export const gateMessage: React.CSSProperties = {
  color: "#a6adc8", fontSize: 14,
};

export const gateTextarea: React.CSSProperties = {
  width: "100%", background: "#181825", color: "#cdd6f4",
  border: "1px solid #313244", borderRadius: 8,
  padding: "9px 12px", fontSize: 13,
  resize: "vertical", boxSizing: "border-box",
};

export const gateActions: React.CSSProperties = {
  display: "flex", gap: 12, marginTop: 18,
};

export const gateBtnApprove: React.CSSProperties = {
  padding: "10px 22px", background: "#a6e3a1", color: "#1e1e2e",
  border: "none", borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 14,
};

export const gateBtnSecondary: React.CSSProperties = {
  padding: "10px 22px", background: "transparent",
  color: "#a6adc8", border: "1px solid #45475a",
  borderRadius: 8, cursor: "pointer", fontSize: 14,
};
