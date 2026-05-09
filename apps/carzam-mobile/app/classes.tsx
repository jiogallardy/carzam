import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { useMemo } from "react";
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { listCarClasses, listDiscoveries } from "@/api/discoveries";
import type { CarClass, Discovery } from "@/api/types";
import { AmbientBackground } from "@/components/AmbientBackground";
import { GlassCard } from "@/components/GlassCard";
import { colors, fontSize, spacing } from "@/theme";

export default function ClassesScreen() {
  const router = useRouter();
  const classesQ = useQuery({ queryKey: ["car-classes"], queryFn: listCarClasses });
  const discQ = useQuery({ queryKey: ["discoveries"], queryFn: listDiscoveries });

  const items = useMemo(() => {
    const found = new Map((discQ.data ?? []).map((d) => [d.car_class.id, d]));
    return (classesQ.data ?? []).map((c) => ({ klass: c, discovery: found.get(c.id) ?? null }));
  }, [classesQ.data, discQ.data]);

  const loading = classesQ.isLoading || discQ.isLoading;

  return (
    <View style={{ flex: 1 }}>
      <AmbientBackground />
      <SafeAreaView style={{ flex: 1 }} edges={["top"]}>
        <View style={styles.header}>
          <Pressable onPress={() => router.back()} hitSlop={10}>
            <Text style={styles.back}>‹ Back</Text>
          </Pressable>
          <Text style={styles.title}>Cars Carzam knows</Text>
          <Text style={styles.sub}>
            The classes the model can recognize today. Submit clips of cars not on this list and we'll consider adding them.
          </Text>
        </View>

        {loading ? (
          <View style={styles.center}><ActivityIndicator color="#fff" /></View>
        ) : (
          <FlatList
            data={items}
            keyExtractor={(it) => it.klass.id}
            contentContainerStyle={{ padding: spacing.lg, paddingBottom: 140 }}
            ItemSeparatorComponent={() => <View style={{ height: spacing.md }} />}
            renderItem={({ item }) => <Row klass={item.klass} discovery={item.discovery} />}
          />
        )}
      </SafeAreaView>
    </View>
  );
}

function Row({ klass, discovery }: { klass: CarClass; discovery: Discovery | null }) {
  return (
    <GlassCard>
      <View style={styles.row}>
        <View style={{ flex: 1 }}>
          <View style={styles.rowHead}>
            <Text style={styles.rowName}>{klass.display_name}</Text>
            {discovery ? (
              <View style={styles.foundPill}>
                <Text style={styles.foundPillText}>FOUND</Text>
              </View>
            ) : (
              <View style={styles.lockedPill}>
                <Text style={styles.lockedPillText}>LOCKED</Text>
              </View>
            )}
          </View>
          {klass.engine_family ? (
            <Text style={styles.engine}>{klass.engine_family}</Text>
          ) : null}
          {klass.description ? (
            <Text style={styles.desc} numberOfLines={3}>{klass.description}</Text>
          ) : null}
          {discovery ? (
            <Text style={styles.stats}>
              {discovery.total_matches}× matched · best {Math.round(discovery.best_confidence * 100)}%
            </Text>
          ) : null}
        </View>
      </View>
    </GlassCard>
  );
}

const styles = StyleSheet.create({
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.sm },
  back: { color: colors.text, fontSize: fontSize.body, fontWeight: "600", marginBottom: spacing.sm },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 4, lineHeight: 18 },

  center: { flex: 1, alignItems: "center", justifyContent: "center", paddingBottom: 100 },

  row: { padding: spacing.lg },
  rowHead: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 4 },
  rowName: { color: colors.text, fontSize: fontSize.title, fontWeight: "700", letterSpacing: -0.3, flexShrink: 1, marginRight: spacing.md },
  engine: { color: colors.accentHi, fontSize: fontSize.small, fontWeight: "700", letterSpacing: 0.3, marginBottom: spacing.sm },
  desc: { color: colors.textMuted, fontSize: fontSize.small, lineHeight: 20 },
  stats: { color: colors.success, fontSize: fontSize.caption, fontWeight: "700", marginTop: 8 },

  foundPill: { backgroundColor: "rgba(111,220,140,0.15)", borderColor: "rgba(111,220,140,0.40)", borderWidth: 1, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999 },
  foundPillText: { color: colors.success, fontSize: 9, fontWeight: "800", letterSpacing: 1 },
  lockedPill: { backgroundColor: "rgba(255,255,255,0.06)", borderColor: colors.border, borderWidth: 1, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999 },
  lockedPillText: { color: colors.textMuted, fontSize: 9, fontWeight: "800", letterSpacing: 1 },
});
