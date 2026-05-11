import { Pressable, StyleSheet, Text, View } from "react-native";

import { CarIcon } from "@/components/CarIcon";
import type { MicPermissionStatus } from "@/hooks/useMicPermission";
import { colors, fontSize, radius, spacing } from "@/theme";

/**
 * UI shown when we don't (yet) have microphone access. Three states:
 *
 *   undetermined — first time. Big "Allow Microphone" button calls
 *                  requestPermissionsAsync, which triggers the iOS prompt.
 *   denied       — declined once, OS still allows re-prompting. Same button
 *                  text, but tap re-fires the request (iOS may show prompt
 *                  again or just silently resolve).
 *   blocked      — declined permanently. Button deep-links to Settings; the
 *                  hook auto-rechecks status when the app returns to fg.
 *
 * The component is intentionally non-blocking — the caller decides what to
 * render around it (it doesn't own the close button or navigation).
 */
interface MicPermissionGateProps {
  status: MicPermissionStatus;
  onRequest: () => void;
  onOpenSettings: () => void;
}

export function MicPermissionGate({
  status,
  onRequest,
  onOpenSettings,
}: MicPermissionGateProps) {
  const blocked = status === "blocked";
  const denied = status === "denied";

  const title = blocked
    ? "Microphone access is off"
    : denied
    ? "We need your microphone"
    : "Carzam needs your microphone";

  const body = blocked
    ? "iOS won't let us ask again from here. Open Settings, flip Microphone on, then come back."
    : denied
    ? "Tap Allow on the iOS prompt so Carzam can listen to the engine. If no prompt appears, open Settings."
    : "Carzam listens to engine sounds to identify the car. We never record continuously — only for 5 seconds when you tap.";

  const ctaText = blocked ? "Open Settings" : "Allow Microphone";
  const onCta = blocked ? onOpenSettings : onRequest;

  return (
    <View style={styles.wrap}>
      <View style={styles.iconWrap}>
        <CarIcon size={64} color={colors.text} />
      </View>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.body}>{body}</Text>

      <Pressable
        onPress={onCta}
        style={({ pressed }) => [styles.cta, pressed && { opacity: 0.85 }]}
      >
        <Text style={styles.ctaText}>{ctaText}</Text>
      </Pressable>

      {/* Secondary action: even in "denied" state offer Settings as escape hatch */}
      {denied && (
        <Pressable onPress={onOpenSettings} hitSlop={12} style={styles.secondary}>
          <Text style={styles.secondaryText}>Or open Settings →</Text>
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.xl,
  },
  iconWrap: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: colors.glassFillStrong,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.lg,
  },
  title: {
    color: colors.text,
    fontSize: fontSize.large,
    fontWeight: "800",
    textAlign: "center",
    letterSpacing: -0.5,
  },
  body: {
    color: colors.textMuted,
    fontSize: fontSize.body,
    textAlign: "center",
    lineHeight: 22,
    marginTop: spacing.sm,
    marginBottom: spacing.xl,
    maxWidth: 320,
  },
  cta: {
    backgroundColor: "#fff",
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: radius.md,
    minWidth: 220,
    alignItems: "center",
  },
  ctaText: { color: "#000", fontWeight: "700", fontSize: fontSize.body },
  secondary: { marginTop: spacing.md, paddingVertical: 8 },
  secondaryText: { color: colors.textMuted, fontSize: fontSize.small, fontWeight: "600" },
});
