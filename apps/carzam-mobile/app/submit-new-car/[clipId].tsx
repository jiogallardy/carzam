import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useState } from "react";
import {
  Alert,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { submitAsNewCar } from "@/api/recognize";
import { GlassCard } from "@/components/GlassCard";
import { PrimaryButton } from "@/components/PrimaryButton";
import { colors, fontSize, spacing } from "@/theme";

export default function SubmitNewCarScreen() {
  const { clipId } = useLocalSearchParams<{ clipId: string }>();
  const router = useRouter();
  const qc = useQueryClient();

  const [year, setYear] = useState("");
  const [make, setMake] = useState("");
  const [model, setModel] = useState("");
  const [trim, setTrim] = useState("");
  const [notes, setNotes] = useState("");

  const m = useMutation({
    mutationFn: (body: {
      year: number;
      make: string;
      model: string;
      trim?: string | null;
      notes?: string | null;
    }) => submitAsNewCar(clipId!, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["discoveries"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
      Alert.alert(
        "Submitted",
        "Thanks — your clip is in the review queue. If we get enough audio of this car we'll add it as a class.",
        [{ text: "OK", onPress: () => router.back() }],
      );
    },
    onError: (e: unknown) => {
      Alert.alert("Save failed", e instanceof Error ? e.message : String(e));
    },
  });

  const onSubmit = () => {
    Keyboard.dismiss();
    const yr = parseInt(year, 10);
    if (!yr || yr < 1900 || yr > 2100) return Alert.alert("Year required", "e.g. 2024");
    if (!make.trim()) return Alert.alert("Make required");
    if (!model.trim()) return Alert.alert("Model required");
    m.mutate({
      year: yr,
      make: make.trim(),
      model: model.trim(),
      trim: trim.trim() || null,
      notes: notes.trim() || null,
    });
  };

  return (
    <View style={{ flex: 1, backgroundColor: colors.bg }}>
      <SafeAreaView style={{ flex: 1 }} edges={["top", "bottom"]}>
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : undefined}
          style={{ flex: 1 }}
          keyboardVerticalOffset={20}
        >
          <View style={styles.header}>
            <View>
              <Text style={styles.eyebrow}>SUBMIT AS NEW CAR</Text>
              <Text style={styles.title}>Tell us what it is</Text>
              <Text style={styles.sub}>
                We'll review the clip and the model gets better at this car next time.
              </Text>
            </View>
            <Pressable onPress={() => router.back()} hitSlop={10}>
              <Text style={styles.cancel}>Cancel</Text>
            </Pressable>
          </View>

          <ScrollView contentContainerStyle={styles.body} keyboardShouldPersistTaps="handled">
            <Field label="Year" value={year} onChangeText={setYear} placeholder="2024" keyboardType="number-pad" maxLength={4} />
            <Field label="Make" value={make} onChangeText={setMake} placeholder="Ford" autoCapitalize="words" />
            <Field label="Model" value={model} onChangeText={setModel} placeholder="Mustang GT" autoCapitalize="words" />
            <Field label="Trim (optional)" value={trim} onChangeText={setTrim} placeholder="Dark Horse" autoCapitalize="words" />
            <Field
              label="Notes (engine, mods, anything that helps)"
              value={notes}
              onChangeText={setNotes}
              placeholder="5.0 Coyote, Borla S-Type, idle"
              multiline
            />
          </ScrollView>

          <View style={styles.footer}>
            <PrimaryButton
              variant="accent"
              label={m.isPending ? "Submitting…" : "Submit clip"}
              onPress={onSubmit}
              loading={m.isPending}
            />
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
}

function Field(props: {
  label: string;
  value: string;
  onChangeText: (v: string) => void;
  placeholder?: string;
  keyboardType?: "default" | "number-pad";
  autoCapitalize?: "none" | "words";
  maxLength?: number;
  multiline?: boolean;
}) {
  return (
    <View style={{ marginBottom: spacing.md }}>
      <Text style={styles.fieldLabel}>{props.label}</Text>
      <GlassCard>
        <TextInput
          value={props.value}
          onChangeText={props.onChangeText}
          placeholder={props.placeholder}
          placeholderTextColor={colors.textDim}
          keyboardType={props.keyboardType}
          autoCapitalize={props.autoCapitalize}
          maxLength={props.maxLength}
          multiline={props.multiline}
          style={[styles.input, props.multiline && { minHeight: 90, paddingTop: 14 }]}
        />
      </GlassCard>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
    gap: spacing.md,
  },
  eyebrow: { color: colors.accentHi, fontSize: 10, letterSpacing: 1.8, fontWeight: "700" },
  title: { color: colors.text, fontSize: fontSize.large, fontWeight: "800", letterSpacing: -0.5, marginTop: 4 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 4, lineHeight: 18 },
  cancel: { color: colors.textMuted, fontSize: fontSize.body, fontWeight: "600", paddingTop: 4 },

  body: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: 60 },
  fieldLabel: {
    color: colors.textMuted,
    fontSize: fontSize.caption,
    fontWeight: "700",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 6,
  },
  input: {
    color: colors.text,
    fontSize: fontSize.body,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },

  footer: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, paddingBottom: spacing.md },
});
