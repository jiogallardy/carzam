import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useEffect, useState } from "react";
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
import { PICKER_OTHER, SearchablePicker, pickerValue } from "@/components/SearchablePicker";
import { ENGINES, MAKES, modelsFor } from "@/data/vehicles";
import { colors, fontSize, spacing } from "@/theme";

export default function SubmitNewCarScreen() {
  const { clipId } = useLocalSearchParams<{ clipId: string }>();
  const router = useRouter();
  const qc = useQueryClient();

  const [year, setYear] = useState("");
  const [make, setMake] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [trim, setTrim] = useState<string | null>(null);
  const [engine, setEngine] = useState<string | null>(null);
  const [notes, setNotes] = useState("");

  useEffect(() => {
    setModel(null);
  }, [make]);

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
    const makeStr = pickerValue(make);
    if (!makeStr) return Alert.alert("Make required");
    const modelStr = pickerValue(model);
    if (!modelStr) return Alert.alert("Model required");

    const engineStr = pickerValue(engine);
    const composedNotes = [
      engineStr ? `Engine: ${engineStr}` : null,
      notes.trim() || null,
    ].filter(Boolean).join("\n") || null;

    m.mutate({
      year: yr,
      make: makeStr,
      model: modelStr,
      trim: pickerValue(trim),
      notes: composedNotes,
    });
  };

  const modelOptions = make && make !== PICKER_OTHER ? modelsFor(make) : [];
  const modelHint = !make ? "Pick a make first" : modelOptions.length === 0 ? "Type your own model below" : undefined;

  return (
    <View style={{ flex: 1, backgroundColor: colors.bg }}>
      <SafeAreaView style={{ flex: 1 }} edges={["top", "bottom"]}>
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : undefined}
          style={{ flex: 1 }}
          keyboardVerticalOffset={20}
        >
          <View style={styles.header}>
            <View style={{ flex: 1 }}>
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
            <View style={{ marginBottom: spacing.md }}>
              <Text style={styles.fieldLabel}>YEAR</Text>
              <GlassCard>
                <TextInput
                  value={year}
                  onChangeText={setYear}
                  placeholder="2024"
                  placeholderTextColor={colors.textDim}
                  keyboardType="number-pad"
                  maxLength={4}
                  style={styles.input}
                />
              </GlassCard>
            </View>

            <SearchablePicker
              label="Make"
              value={make}
              options={MAKES}
              placeholder="Pick a make"
              onChange={setMake}
            />

            <SearchablePicker
              label="Model"
              value={model}
              options={modelOptions}
              placeholder={make ? "Pick a model" : "Pick a make first"}
              disabled={!make}
              onChange={setModel}
              hint={modelHint}
            />

            <SearchablePicker
              label="Engine (helps the next model)"
              value={engine}
              options={ENGINES}
              placeholder="V8 NA, I6 Turbo, …"
              onChange={setEngine}
            />

            <View style={{ marginBottom: spacing.md }}>
              <Text style={styles.fieldLabel}>TRIM (OPTIONAL)</Text>
              <GlassCard>
                <TextInput
                  value={trim ?? ""}
                  onChangeText={(t) => setTrim(t || null)}
                  placeholder="e.g. Dark Horse, Black Series"
                  placeholderTextColor={colors.textDim}
                  autoCapitalize="words"
                  style={styles.input}
                />
              </GlassCard>
            </View>

            <View style={{ marginBottom: spacing.md }}>
              <Text style={styles.fieldLabel}>NOTES (OPTIONAL)</Text>
              <GlassCard>
                <TextInput
                  value={notes}
                  onChangeText={setNotes}
                  placeholder="5.0 Coyote, Borla S-Type, idle"
                  placeholderTextColor={colors.textDim}
                  multiline
                  style={[styles.input, { minHeight: 90, paddingTop: 14 }]}
                />
              </GlassCard>
            </View>
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
  fieldLabel: { color: colors.textMuted, fontSize: fontSize.caption, fontWeight: "700", letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 6 },
  input: { color: colors.text, fontSize: fontSize.body, paddingHorizontal: 14, paddingVertical: 14 },

  footer: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, paddingBottom: spacing.md },
});
