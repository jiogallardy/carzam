import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { listModelClasses, listModels } from "@/api/recognize";
import type { CarClass, ModelInfo } from "@/api/types";
import { AmbientBackground } from "@/components/AmbientBackground";
import { GlassCard } from "@/components/GlassCard";
import { useAuthStore } from "@/store/auth";
import { useModelStore } from "@/store/model";
import { colors, fontSize, radius, spacing } from "@/theme";

export default function MeScreen() {
  const insets = useSafeAreaInsets();
  const user = useAuthStore((s) => s.user);
  const signOut = useAuthStore((s) => s.signOut);

  const activeModelId = useModelStore((s) => s.activeModelId);
  const setActiveModelId = useModelStore((s) => s.setActiveModelId);

  // Tracked separately from `activeModelId` so a user can browse classes
  // without committing to switching their default model.
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);

  const modelsQ = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    staleTime: 60_000,
  });

  return (
    <View style={{ flex: 1 }}>
      <AmbientBackground />
      <View style={[styles.safe, { paddingTop: insets.top }]}>
        <View style={styles.header}>
          <Text style={styles.title}>Me</Text>
        </View>

        <ScrollView contentContainerStyle={styles.body}>
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

          <View>
            <Text style={styles.sectionLabel}>MODEL</Text>
            <Text style={styles.sectionHint}>
              Pick which checkpoint runs on your next Carzam recording.
              {"\n"}New checkpoints uploaded to R2 (model/&lt;id&gt;/) appear here.
            </Text>
            <GlassCard style={{ marginTop: spacing.sm }}>
              {modelsQ.isLoading && (
                <View style={styles.modelLoading}>
                  <ActivityIndicator color="#fff" />
                </View>
              )}
              {modelsQ.error && (
                <View style={styles.modelLoading}>
                  <Text style={styles.modelError}>Couldn't load models. Pull to retry.</Text>
                </View>
              )}
              {modelsQ.data && (
                <View>
                  <ModelRow
                    label="Server default"
                    sub={`(${modelsQ.data.default_id})`}
                    selected={activeModelId === null}
                    badge={null}
                    expanded={false}
                    onPress={() => setActiveModelId(null)}
                    onToggleExpand={() => {}}
                  />
                  {modelsQ.data.models.map((m) => (
                    <View key={m.id}>
                      <ModelRow
                        label={m.label}
                        sub={m.is_default ? "default" : null}
                        selected={activeModelId === m.id}
                        badge={m.is_loaded ? "hot" : null}
                        expanded={expandedModelId === m.id}
                        onPress={() => {
                          setActiveModelId(m.id);
                          setExpandedModelId(expandedModelId === m.id ? null : m.id);
                        }}
                        onToggleExpand={() =>
                          setExpandedModelId(expandedModelId === m.id ? null : m.id)
                        }
                      />
                      {expandedModelId === m.id && <ModelClassesPanel modelId={m.id} />}
                    </View>
                  ))}
                </View>
              )}
            </GlassCard>
          </View>

          <Pressable
            onPress={signOut}
            style={({ pressed }) => [styles.signOut, pressed && { opacity: 0.85 }]}
          >
            <Text style={styles.signOutText}>Sign out</Text>
          </Pressable>
        </ScrollView>
      </View>
    </View>
  );
}

function ModelRow({
  label,
  sub,
  selected,
  badge,
  expanded,
  onPress,
  onToggleExpand,
  isFirst,
}: {
  label: string;
  sub: string | null;
  selected: boolean;
  badge: string | null;
  expanded: boolean;
  onPress: () => void;
  onToggleExpand: () => void;
  isFirst?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.modelRow,
        !isFirst && styles.modelRowSep,
        pressed && { opacity: 0.85 },
      ]}
    >
      <View style={{ flex: 1 }}>
        <View style={styles.modelRowTitle}>
          <Text style={[styles.modelRowLabel, selected && styles.modelRowLabelOn]} numberOfLines={1}>
            {label}
          </Text>
          {badge && (
            <View style={styles.modelBadge}>
              <Text style={styles.modelBadgeText}>{badge}</Text>
            </View>
          )}
        </View>
        {sub && <Text style={styles.modelRowSub}>{sub}</Text>}
      </View>
      <Pressable
        hitSlop={10}
        onPress={onToggleExpand}
        style={styles.expandToggle}
      >
        <Text style={[styles.expandChevron, expanded && styles.expandChevronOn]}>
          {expanded ? "▾" : "▸"}
        </Text>
      </Pressable>
      <View style={[styles.radio, selected && styles.radioOn]}>
        {selected && <View style={styles.radioInner} />}
      </View>
    </Pressable>
  );
}

/**
 * Renders the car classes a model can identify as a wrapped list of pills.
 * Lazy-fetches on mount (only when expanded) and caches via react-query
 * so toggling open/closed doesn't re-hit the server.
 */
function ModelClassesPanel({ modelId }: { modelId: string }) {
  const q = useQuery({
    queryKey: ["models", modelId, "classes"],
    queryFn: () => listModelClasses(modelId),
    staleTime: 5 * 60_000,
  });
  if (q.isLoading) {
    return (
      <View style={styles.classesPanel}>
        <ActivityIndicator color="#fff" size="small" />
      </View>
    );
  }
  if (q.error || !q.data) {
    return (
      <View style={styles.classesPanel}>
        <Text style={styles.modelError}>Couldn't load classes for this model.</Text>
      </View>
    );
  }
  return (
    <View style={styles.classesPanel}>
      <Text style={styles.classesHeading}>
        Identifies {q.data.length} {q.data.length === 1 ? "car" : "cars"}
      </Text>
      <View style={styles.pillRow}>
        {q.data.map((c) => (
          <View key={c.id} style={styles.pill}>
            <Text style={styles.pillText} numberOfLines={1}>
              {c.display_name}
            </Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1 },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.sm },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8 },
  body: { padding: spacing.lg, gap: spacing.lg, paddingBottom: 140 },

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

  sectionLabel: {
    color: colors.textDim,
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 1.6,
    marginLeft: spacing.xs,
  },
  sectionHint: {
    color: colors.textMuted,
    fontSize: fontSize.small,
    marginTop: 6,
    marginLeft: spacing.xs,
    lineHeight: 18,
  },

  modelLoading: { padding: spacing.lg, alignItems: "center" },
  modelError: { color: colors.danger, fontSize: fontSize.small },

  modelRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    padding: spacing.md,
  },
  modelRowSep: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(255,255,255,0.10)",
  },
  modelRowTitle: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  modelRowLabel: { color: colors.text, fontSize: fontSize.body, fontWeight: "600" },
  modelRowLabelOn: { fontWeight: "800" },
  modelRowSub: { color: colors.textMuted, fontSize: 11, marginTop: 2 },

  modelBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 999,
    backgroundColor: "rgba(111,220,140,0.15)",
    borderWidth: 1,
    borderColor: "rgba(111,220,140,0.45)",
  },
  modelBadgeText: { color: colors.success, fontSize: 10, fontWeight: "800", letterSpacing: 0.6 },

  radio: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1.5,
    borderColor: colors.textDim,
    alignItems: "center",
    justifyContent: "center",
  },
  radioOn: { borderColor: colors.accent },
  radioInner: {
    width: 11,
    height: 11,
    borderRadius: 999,
    backgroundColor: colors.accent,
  },

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

  expandToggle: {
    width: 28,
    height: 28,
    alignItems: "center",
    justifyContent: "center",
    marginRight: spacing.xs,
  },
  expandChevron: {
    color: colors.textDim,
    fontSize: 14,
    fontWeight: "700",
  },
  expandChevronOn: { color: colors.text },

  classesPanel: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.xs,
    paddingBottom: spacing.md,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "rgba(255,255,255,0.06)",
    backgroundColor: "rgba(0,0,0,0.15)",
  },
  classesHeading: {
    color: colors.textMuted,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 1.4,
    fontWeight: "700",
    marginBottom: spacing.sm,
  },
  pillRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
  },
  pill: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.06)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.10)",
  },
  pillText: {
    color: colors.text,
    fontSize: 12,
    fontWeight: "600",
  },
});
