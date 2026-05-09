import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { deleteCar, listCars } from "@/api/cars";
import { AmbientBackground } from "@/components/AmbientBackground";
import { GlassCard } from "@/components/GlassCard";
import { colors, fontSize, radius, spacing } from "@/theme";

export default function CarDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const qc = useQueryClient();
  const carsQ = useQuery({ queryKey: ["cars"], queryFn: listCars });
  const car = carsQ.data?.find((c) => c.id === id);

  const del = useMutation({
    mutationFn: () => deleteCar(id!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["cars"] });
      router.back();
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : "Delete failed.";
      Alert.alert("Couldn't delete", msg);
    },
  });

  const onDelete = () =>
    Alert.alert("Delete car?", "This removes the car and any clips attached to it.", [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => del.mutate() },
    ]);

  return (
    <View style={{ flex: 1 }}>
      <AmbientBackground />
      <SafeAreaView style={{ flex: 1 }} edges={["top"]}>
        <View style={styles.header}>
          <Pressable onPress={() => router.back()} hitSlop={10}>
            <Text style={styles.back}>‹ Back</Text>
          </Pressable>
        </View>
        <ScrollView contentContainerStyle={{ padding: spacing.lg, paddingBottom: 140 }}>
          {!car ? (
            <Text style={{ color: colors.textMuted }}>Car not found.</Text>
          ) : (
            <>
              <Text style={styles.title}>{car.year} {car.make} {car.model}</Text>
              <Text style={styles.sub}>{[car.trim, car.color].filter(Boolean).join(" · ") || "—"}</Text>

              <GlassCard style={{ marginTop: spacing.lg }}>
                <View style={{ padding: spacing.lg, gap: 6 }}>
                  <DetailRow label="Year" value={String(car.year)} />
                  <DetailRow label="Make" value={car.make} />
                  <DetailRow label="Model" value={car.model} />
                  {car.trim ? <DetailRow label="Trim" value={car.trim} /> : null}
                  {car.color ? <DetailRow label="Color" value={car.color} /> : null}
                  {car.mods_notes ? <DetailRow label="Notes" value={car.mods_notes} /> : null}
                </View>
              </GlassCard>

              <Pressable
                onPress={() => router.push(`/record-for-car/${id}`)}
                style={({ pressed }) => [styles.recordBtn, pressed && { opacity: 0.85 }]}
              >
                <Text style={styles.recordBtnText}>+ Record audio of this car</Text>
              </Pressable>

              <Pressable
                onPress={onDelete}
                style={({ pressed }) => [styles.delete, pressed && { opacity: 0.85 }]}
              >
                <Text style={styles.deleteText}>{del.isPending ? "Deleting…" : "Delete car"}</Text>
              </Pressable>
            </>
          )}
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowValue} numberOfLines={3}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.sm },
  back: { color: colors.text, fontSize: fontSize.body, fontWeight: "600" },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 4 },
  row: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 4 },
  rowLabel: { color: colors.textMuted, fontSize: fontSize.small, fontWeight: "600" },
  rowValue: { color: colors.text, fontSize: fontSize.body, flexShrink: 1, textAlign: "right", marginLeft: spacing.md },
  recordBtn: {
    marginTop: spacing.lg,
    height: 50,
    borderRadius: radius.md,
    backgroundColor: "rgba(255,126,26,0.15)",
    borderWidth: 1,
    borderColor: "rgba(255,126,26,0.45)",
    alignItems: "center",
    justifyContent: "center",
  },
  recordBtnText: { color: colors.accentHi, fontWeight: "700" },
  delete: {
    marginTop: spacing.md,
    height: 50,
    borderRadius: radius.md,
    backgroundColor: "rgba(255,94,94,0.10)",
    borderWidth: 1,
    borderColor: "rgba(255,94,94,0.40)",
    alignItems: "center",
    justifyContent: "center",
  },
  deleteText: { color: colors.danger, fontWeight: "700" },
});
