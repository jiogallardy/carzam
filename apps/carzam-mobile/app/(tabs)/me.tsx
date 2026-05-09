import { Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { AmbientBackground } from "@/components/AmbientBackground";
import { GlassCard } from "@/components/GlassCard";
import { useAuthStore } from "@/store/auth";
import { colors, fontSize, radius, spacing } from "@/theme";

export default function MeScreen() {
  const insets = useSafeAreaInsets();
  const user = useAuthStore((s) => s.user);
  const signOut = useAuthStore((s) => s.signOut);

  return (
    <View style={{ flex: 1 }}>
      <AmbientBackground />
      <View style={[styles.safe, { paddingTop: insets.top }]}>
        <View style={styles.header}>
          <Text style={styles.title}>Me</Text>
        </View>

        <View style={styles.body}>
          <GlassCard>
            <View style={styles.profileInner}>
              <View style={styles.avatar}>
                <Text style={styles.avatarText}>
                  {(user?.display_name ?? user?.email ?? "?").slice(0, 1).toUpperCase()}
                </Text>
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.name}>{user?.display_name ?? "Carzam user"}</Text>
                <Text style={styles.email}>{user?.email ?? "—"}</Text>
              </View>
            </View>
          </GlassCard>

          <Pressable
            onPress={signOut}
            style={({ pressed }) => [styles.signOut, pressed && { opacity: 0.85 }]}
          >
            <Text style={styles.signOutText}>Sign out</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.sm },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8 },
  body: { padding: spacing.lg, gap: spacing.lg },

  profileInner: { padding: spacing.lg, flexDirection: "row", alignItems: "center", gap: spacing.md },
  avatar: {
    width: 56,
    height: 56,
    borderRadius: 999,
    backgroundColor: colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: { color: "#fff", fontWeight: "800", fontSize: 24 },
  name: { color: colors.text, fontSize: fontSize.title, fontWeight: "700" },
  email: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 2 },

  signOut: {
    marginTop: spacing.md,
    height: 50,
    borderRadius: radius.md,
    backgroundColor: "rgba(255,94,94,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,94,94,0.40)",
    alignItems: "center",
    justifyContent: "center",
  },
  signOutText: { color: colors.danger, fontWeight: "700" },
});
