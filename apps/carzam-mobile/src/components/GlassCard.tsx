import { BlurView } from "expo-blur";
import { LinearGradient } from "expo-linear-gradient";
import { ReactNode } from "react";
import { StyleSheet, View, ViewStyle } from "react-native";

import { colors, radius } from "@/theme";

type Props = {
  children: ReactNode;
  intensity?: number;
  tone?: "neutral" | "warm" | "cool";
  style?: ViewStyle | ViewStyle[];
};

const TONE_OVERLAYS: Record<NonNullable<Props["tone"]>, [string, string]> = {
  neutral: ["rgba(255,255,255,0.14)", "rgba(255,255,255,0.02)"],
  warm: ["rgba(255,160,90,0.18)", "rgba(255,80,60,0.04)"],
  cool: ["rgba(120,150,255,0.16)", "rgba(60,80,180,0.03)"],
};

export function GlassCard({ children, intensity = 55, tone = "neutral", style }: Props) {
  const overlay = TONE_OVERLAYS[tone];
  return (
    <View style={[styles.outer, style]}>
      <BlurView intensity={intensity} tint="dark" style={StyleSheet.absoluteFill} />
      <LinearGradient
        colors={overlay}
        start={{ x: 0.1, y: 0 }}
        end={{ x: 0.9, y: 1 }}
        style={StyleSheet.absoluteFill}
      />
      {/* Top edge specular (light hits the top of the glass) */}
      <View style={styles.specularTop} />
      {/* Bottom edge dark line for depth */}
      <View style={styles.shadowBottom} />
      <View style={styles.content}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  outer: {
    borderRadius: radius.lg,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: "rgba(20,20,28,0.32)",
    shadowColor: "#000",
    shadowOpacity: 0.35,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 10 },
  },
  specularTop: {
    position: "absolute",
    left: "6%",
    right: "6%",
    top: 0,
    height: 1.5,
    backgroundColor: "rgba(255,255,255,0.45)",
    borderRadius: 999,
  },
  shadowBottom: {
    position: "absolute",
    left: "10%",
    right: "10%",
    bottom: 0,
    height: 1,
    backgroundColor: "rgba(0,0,0,0.35)",
  },
  content: {
    padding: 0,
  },
});
