import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { useMemo } from "react";
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { listCarClasses, listDiscoveries } from "@/api/discoveries";
import type { CarClass, Discovery } from "@/api/types";
import { AmbientBackground } from "@/components/AmbientBackground";
import { GlassCard } from "@/components/GlassCard";
import { colors, fontSize, radius, spacing } from "@/theme";

export default function FoundScreen() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const classesQ = useQuery({ queryKey: ["car-classes"], queryFn: listCarClasses });
  const discQ = useQuery({ queryKey: ["discoveries"], queryFn: listDiscoveries });

  const cells = useMemo(() => {
    const classes = classesQ.data ?? [];
    const found = new Map((discQ.data ?? []).map((d) => [d.car_class.id, d]));
    return classes.map((c) => ({ klass: c, found: found.get(c.id) ?? null }));
  }, [classesQ.data, discQ.data]);

  const loading = classesQ.isLoading || discQ.isLoading;
  const totalFound = (discQ.data ?? []).filter((d) => d.car_class.id !== "other").length;
  const totalAvail = (classesQ.data ?? []).length;
  const pct = totalAvail ? Math.round((100 * totalFound) / totalAvail) : 0;

  return (
    <View style={{ flex: 1 }}>
      <AmbientBackground />
      <View style={[styles.safe, { paddingTop: insets.top }]}>
        <View style={styles.header}>
          <View style={{ flexDirection: "row", alignItems: "flex-start", justifyContent: "space-between" }}>
            <View style={{ flex: 1 }}>
              <Text style={styles.title}>Found</Text>
              <Text style={styles.sub}>{totalFound} / {totalAvail} discovered · {pct}%</Text>
            </View>
            <Pressable
              onPress={() => router.push("/classes")}
              style={({ pressed }) => [styles.browseBtn, pressed && { opacity: 0.85 }]}
            >
              <Text style={styles.browseBtnText}>Browse all</Text>
            </Pressable>
          </View>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${pct}%` }]} />
          </View>
        </View>

        {loading ? (
          <View style={styles.center}><ActivityIndicator color="#fff" /></View>
        ) : (
          <FlatList
            data={cells}
            keyExtractor={(c) => c.klass.id}
            numColumns={2}
            columnWrapperStyle={{ gap: spacing.md, paddingHorizontal: spacing.lg }}
            ItemSeparatorComponent={() => <View style={{ height: spacing.md }} />}
            contentContainerStyle={{ paddingTop: spacing.md, paddingBottom: 140 }}
            renderItem={({ item }) => <Cell klass={item.klass} discovery={item.found} />}
          />
        )}
      </View>
    </View>
  );
}

function Cell({ klass, discovery }: { klass: CarClass; discovery: Discovery | null }) {
  const found = discovery !== null;
  return (
    <View style={{ flex: 1 }}>
      <GlassCard>
        <View style={[styles.cell, !found && styles.cellLocked]}>
          <Text style={[styles.cellName, !found && styles.cellNameLocked]} numberOfLines={2}>
            {found ? klass.display_name : "???"}
          </Text>
          <Text style={styles.cellEngine} numberOfLines={1}>
            {found ? klass.engine_family ?? "—" : "Locked"}
          </Text>
          {found && (
            <Text style={styles.cellMatches}>
              {discovery!.total_matches}× · best {Math.round(discovery!.best_confidence * 100)}%
            </Text>
          )}
        </View>
      </GlassCard>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.md },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 2 },
  progressTrack: {
    marginTop: spacing.md,
    height: 6,
    borderRadius: 999,
    backgroundColor: colors.glassFill,
    overflow: "hidden",
  },
  progressFill: { height: "100%", backgroundColor: colors.accent, borderRadius: 999 },
  browseBtn: {
    backgroundColor: colors.glassFillStrong,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
  },
  browseBtnText: { color: colors.text, fontWeight: "700", fontSize: 13 },

  center: { flex: 1, alignItems: "center", justifyContent: "center", paddingBottom: 100 },

  cell: {
    aspectRatio: 1,
    padding: spacing.md,
    justifyContent: "flex-end",
  },
  cellLocked: { opacity: 0.5 },
  cellName: { color: colors.text, fontSize: fontSize.body, fontWeight: "700", letterSpacing: -0.2 },
  cellNameLocked: { color: colors.textDim },
  cellEngine: { color: colors.textMuted, fontSize: fontSize.caption, marginTop: 2 },
  cellMatches: { color: colors.accent, fontSize: fontSize.caption, marginTop: 6, fontWeight: "700" },
});
