/** Design tokens — Catppuccin Mocha palette */
export const colors = {
  base:     "#11111b",
  mantle:   "#181825",
  crust:    "#1e1e2e",
  surface0: "#313244",
  surface1: "#45475a",
  surface2: "#585b70",
  text:     "#cdd6f4",
  subtext0: "#a6adc8",
  subtext1: "#585b70",
  blue:     "#89b4fa",
  lavender: "#b4befe",
  mauve:    "#cba6f7",
  green:    "#a6e3a1",
  yellow:   "#f9e2af",
  red:      "#f38ba8",
  teal:     "#94e2d5",
} as const;

export const radii = {
  sm: 6,
  md: 8,
  lg: 12,
  xl: 16,
  pill: 999,
} as const;

export const shadows = {
  card:    "0 8px 32px rgba(0, 0, 0, 0.35)",
  cardLg:  "0 16px 48px rgba(0, 0, 0, 0.45)",
  glowBlue: "0 0 0 1px rgba(137, 180, 250, 0.15), 0 8px 32px rgba(137, 180, 250, 0.08)",
  glowMauve: "0 0 0 1px rgba(203, 166, 247, 0.15), 0 8px 32px rgba(203, 166, 247, 0.08)",
} as const;
