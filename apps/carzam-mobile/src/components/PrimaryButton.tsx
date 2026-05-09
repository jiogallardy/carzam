import { LinearGradient } from "expo-linear-gradient";
import { ActivityIndicator, Pressable, StyleSheet, Text, ViewStyle } from "react-native";

import { colors, radius } from "@/theme";

type Props = {
  label: string;
  onPress?: () => void;
  loading?: boolean;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "accent";
  style?: ViewStyle | ViewStyle[];
};

export function PrimaryButton({ label, onPress, loading, disabled, variant = "primary", style }: Props) {
  const isAccent = variant === "accent";
  const isGhost = variant === "ghost";
  const inner = (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={({ pressed }) => [
        styles.base,
        isGhost && styles.ghost,
        !isAccent && !isGhost && styles.primary,
        pressed && { opacity: 0.85 },
        disabled && { opacity: 0.5 },
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={isAccent || isGhost ? "#fff" : "#000"} />
      ) : (
        <Text style={[styles.label, (isAccent || isGhost) && styles.labelLight]}>{label}</Text>
      )}
    </Pressable>
  );

  if (!isAccent) return inner;
  return (
    <LinearGradient
      colors={[colors.accentHi, colors.accent, colors.accentLo]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={[styles.base, styles.accentWrap, style]}
    >
      <Pressable
        onPress={onPress}
        disabled={disabled || loading}
        style={({ pressed }) => [styles.accentInner, pressed && { opacity: 0.85 }]}
      >
        {loading ? <ActivityIndicator color="#fff" /> : <Text style={[styles.label, styles.labelLight]}>{label}</Text>}
      </Pressable>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  base: {
    height: 50,
    borderRadius: radius.md,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 18,
  },
  primary: { backgroundColor: "#ffffff" },
  ghost: {
    backgroundColor: colors.glassFill,
    borderWidth: 1,
    borderColor: colors.border,
  },
  accentWrap: {
    padding: 0,
  },
  accentInner: {
    height: 50,
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  label: { fontSize: 15, fontWeight: "700", color: "#050507", letterSpacing: -0.2 },
  labelLight: { color: "#ffffff" },
});
