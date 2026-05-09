import { useQuery } from "@tanstack/react-query";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { ActivityIndicator, FlatList, Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { listCars } from "@/api/cars";
import type { OwnedCar } from "@/api/types";
import { AmbientBackground } from "@/components/AmbientBackground";
import { GlassCard } from "@/components/GlassCard";
import { colors, fontSize, radius, spacing } from "@/theme";

export default function GarageScreen() {
  const router = useRouter();
  const { data, isLoading, error, refetch, isRefetching } = useQuery({
    queryKey: ["cars"],
    queryFn: listCars,
  });

  return (
    <View style={{ flex: 1 }}>
      <AmbientBackground />
      <SafeAreaView style={styles.safe} edges={["top"]}>
        <View style={styles.header}>
          <View>
            <Text style={styles.title}>Garage</Text>
            <Text style={styles.sub}>{data ? `${data.length} ${data.length === 1 ? "car" : "cars"}` : "—"}</Text>
          </View>
          <Pressable onPress={() => router.push("/add-car")} style={({ pressed }) => [styles.addBtn, pressed && { opacity: 0.85 }]}>
            <Text style={styles.addBtnText}>+ Add</Text>
          </Pressable>
        </View>

        {isLoading && (
          <View style={styles.center}>
            <ActivityIndicator color="#fff" />
          </View>
        )}

        {error && (
          <View style={styles.center}>
            <Text style={styles.errorText}>Couldn't load your cars.</Text>
            <Pressable onPress={() => refetch()} style={styles.retryBtn}>
              <Text style={styles.retryText}>Retry</Text>
            </Pressable>
          </View>
        )}

        {data && (
          <FlatList
            data={data}
            keyExtractor={(c) => c.id}
            contentContainerStyle={{ paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: 140 }}
            ItemSeparatorComponent={() => <View style={{ height: spacing.md }} />}
            ListEmptyComponent={<EmptyState onAdd={() => router.push("/add-car")} />}
            refreshing={isRefetching}
            onRefresh={refetch}
            renderItem={({ item }) => <CarRow car={item} onPress={() => router.push(`/car/${item.id}`)} />}
          />
        )}
      </SafeAreaView>
    </View>
  );
}

function CarRow({ car, onPress }: { car: OwnedCar; onPress: () => void }) {
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [{ opacity: pressed ? 0.85 : 1 }]}>
      <GlassCard>
        <LinearGradient
          colors={["rgba(255,126,26,0.18)", "rgba(106,58,255,0.10)"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.cardInner}
        >
          <View>
            <Text style={styles.carName}>{car.year} {car.make} {car.model}</Text>
            <Text style={styles.carMeta}>
              {[car.trim, car.color, car.mods_notes].filter(Boolean).join(" · ") || "—"}
            </Text>
          </View>
        </LinearGradient>
      </GlassCard>
    </Pressable>
  );
}

function EmptyState({ onAdd }: { onAdd: () => void }) {
  return (
    <View style={styles.empty}>
      <Text style={styles.emptyTitle}>No cars yet</Text>
      <Text style={styles.emptySub}>Add a car to track your audio recordings and personal stats.</Text>
      <Pressable onPress={onAdd} style={styles.emptyBtn}>
        <Text style={styles.emptyBtnText}>Add my first car</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 2 },
  addBtn: {
    backgroundColor: colors.glassFillStrong,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radius.pill,
  },
  addBtnText: { color: colors.text, fontWeight: "700", fontSize: 13 },

  cardInner: { padding: spacing.lg },
  carName: { color: colors.text, fontSize: fontSize.title, fontWeight: "700", letterSpacing: -0.3 },
  carMeta: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 4 },

  center: { flex: 1, alignItems: "center", justifyContent: "center", paddingBottom: 100 },
  errorText: { color: colors.danger, fontSize: fontSize.body, marginBottom: spacing.md },
  retryBtn: {
    backgroundColor: colors.glassFillStrong,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: radius.md,
  },
  retryText: { color: colors.text, fontWeight: "700" },

  empty: { padding: spacing.xl, alignItems: "center", marginTop: 60 },
  emptyTitle: { color: colors.text, fontSize: fontSize.large, fontWeight: "700", marginBottom: 4 },
  emptySub: { color: colors.textMuted, fontSize: fontSize.body, textAlign: "center", marginBottom: spacing.lg },
  emptyBtn: {
    backgroundColor: "#fff",
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: radius.md,
  },
  emptyBtnText: { color: "#000", fontWeight: "700" },
});
