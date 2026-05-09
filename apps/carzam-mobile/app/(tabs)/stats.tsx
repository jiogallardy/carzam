import { useQuery } from "@tanstack/react-query";
import { ActivityIndicator, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { fetchStats } from "@/api/discoveries";
import { AmbientBackground } from "@/components/AmbientBackground";
import { GlassCard } from "@/components/GlassCard";
import { colors, fontSize, spacing } from "@/theme";

export default function StatsScreen() {
  const insets = useSafeAreaInsets();
  const { data, isLoading } = useQuery({ queryKey: ["stats"], queryFn: fetchStats });

  return (
    <View style={{ flex: 1 }}>
      <AmbientBackground />
      <View style={[styles.safe, { paddingTop: insets.top }]}>
        <View style={styles.header}>
          <Text style={styles.title}>Stats</Text>
          <Text style={styles.sub}>How you're doing.</Text>
        </View>

        <View style={styles.body}>
          {isLoading ? (
            <ActivityIndicator color="#fff" />
          ) : data ? (
            <>
              <GlassCard style={styles.bigCard}>
                <View style={styles.bigCardInner}>
                  <Text style={styles.bigPct}>{data.completion_percent}%</Text>
                  <Text style={styles.bigLabel}>Collection Complete</Text>
                  <Text style={styles.bigSub}>{data.n_discovered} of {data.n_total} cars discovered</Text>
                </View>
              </GlassCard>

              <View style={styles.row}>
                <Stat label="Recognitions" value={data.total_recognitions.toString()} />
                <Stat label="Discovered" value={`${data.n_discovered}`} />
                <Stat label="Remaining" value={`${data.n_total - data.n_discovered}`} />
              </View>
            </>
          ) : (
            <Text style={{ color: colors.textMuted }}>No stats yet.</Text>
          )}
        </View>
      </View>
    </View>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={{ flex: 1 }}>
      <GlassCard>
        <View style={styles.statInner}>
          <Text style={styles.statValue}>{value}</Text>
          <Text style={styles.statLabel}>{label}</Text>
        </View>
      </GlassCard>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.sm },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 2 },
  body: { padding: spacing.lg, gap: spacing.lg },

  bigCard: {},
  bigCardInner: { padding: spacing.xl, alignItems: "center" },
  bigPct: { color: colors.text, fontSize: 64, fontWeight: "200", letterSpacing: -2 },
  bigLabel: { color: colors.text, fontSize: fontSize.body, fontWeight: "700", marginTop: 4 },
  bigSub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 6 },

  row: { flexDirection: "row", gap: spacing.md },
  statInner: { padding: spacing.md, alignItems: "center" },
  statValue: { color: colors.text, fontSize: fontSize.large, fontWeight: "800" },
  statLabel: { color: colors.textMuted, fontSize: fontSize.caption, marginTop: 2, textTransform: "uppercase", letterSpacing: 0.5 },
});
