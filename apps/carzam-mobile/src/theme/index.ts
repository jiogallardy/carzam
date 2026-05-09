// Centralized design tokens. iOS 26 Liquid Glass + dark-first.
export const colors = {
  bg: "#050507",
  bgElev: "#0d0d12",
  text: "#ffffff",
  textMuted: "rgba(255,255,255,0.68)",
  textDim: "rgba(255,255,255,0.42)",
  border: "rgba(255,255,255,0.12)",
  glassFill: "rgba(255,255,255,0.06)",
  glassFillStrong: "rgba(255,255,255,0.14)",
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
  sm: 12,
  md: 16,
  lg: 24,
  xl: 32,
  pill: 999,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 18,
  xl: 26,
  xxl: 36,
};

// iOS 26 type ramp. `display` = oversized hero numbers, `hero` = large title.
export const fontSize = {
  caption: 11,
  small: 13,
  body: 15,
  title: 19,
  large: 24,
  hero: 34,
  display: 56,
};

export const accentGradient = ["#ffb47a", "#ff7e1a", "#ff3d3d", "#8b0e2e"] as const;
