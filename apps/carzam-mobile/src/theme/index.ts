// Centralized design tokens. Liquid-glass + dark-first.
export const colors = {
  bg: "#050507",
  bgElev: "#0d0d12",
  text: "#ffffff",
  textMuted: "rgba(255,255,255,0.65)",
  textDim: "rgba(255,255,255,0.45)",
  border: "rgba(255,255,255,0.10)",
  glassFill: "rgba(255,255,255,0.06)",
  glassFillStrong: "rgba(255,255,255,0.12)",
  accent: "#ff7e1a",
  accentHi: "#ffb47a",
  accentLo: "#c81848",
  success: "#6fdc8c",
  danger: "#ff5e5e",
  state: {
    idle: "#7eb8ff",
    accel: "#ff7e1a",
    decel: "#ffce54",
  },
};

export const radius = {
  sm: 10,
  md: 14,
  lg: 22,
  xl: 28,
  pill: 999,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
};

export const fontSize = {
  caption: 11,
  small: 12,
  body: 14,
  title: 18,
  large: 22,
  hero: 28,
};

export const accentGradient = ["#ffb47a", "#ff7e1a", "#ff3d3d", "#8b0e2e"] as const;
