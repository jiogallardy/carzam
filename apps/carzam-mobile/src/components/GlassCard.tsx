import { BlurView } from "expo-blur";
import { LinearGradient } from "expo-linear-gradient";
import { ReactNode } from "react";
import { StyleSheet, View, ViewStyle } from "react-native";

import { colors, radius } from "@/theme";

type Props = {
  children: ReactNode;
  intensity?: number;
  style?: ViewStyle | ViewStyle[];
};

export function GlassCard({ children, intensity = 30, style }: Props) {
  return (
    <View style={[styles.outer, style]}>
      <BlurView intensity={intensity} tint="dark" style={StyleSheet.absoluteFill} />
      <LinearGradient
        colors={["rgba(255,255,255,0.10)", "rgba(255,255,255,0.02)"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={StyleSheet.absoluteFill}
      />
      <View style={styles.specular} />
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
    backgroundColor: "rgba(20,20,28,0.30)",
  },
  specular: {
    position: "absolute",
    left: "8%",
    right: "8%",
    top: 0,
    height: 1,
    backgroundColor: "rgba(255,255,255,0.30)",
  },
  content: {
    padding: 0,
  },
});
