import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "expo-router";
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

import { addCar } from "@/api/cars";
import { GlassCard } from "@/components/GlassCard";
import { PrimaryButton } from "@/components/PrimaryButton";
import { PICKER_OTHER, SearchablePicker, pickerValue } from "@/components/SearchablePicker";
import { ENGINES, MAKES, modelsFor } from "@/data/vehicles";
import { colors, fontSize, spacing } from "@/theme";

export default function AddCarModal() {
  const router = useRouter();
  const qc = useQueryClient();

  const [year, setYear] = useState("");
  const [make, setMake] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [trim, setTrim] = useState<string | null>(null);
  const [engine, setEngine] = useState<string | null>(null);
  const [color, setColor] = useState("");
  const [notes, setNotes] = useState("");

  // If the user changes the make, clear the model so the cascading list resets.
  useEffect(() => {
    setModel(null);
  }, [make]);

  const m = useMutation({
    mutationFn: addCar,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["cars"] });
      router.back();
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : "Failed to save.";
      Alert.alert("Save failed", msg);
    },
  });

  const onSubmit = () => {
    Keyboard.dismiss();
    const yr = parseInt(year, 10);
    if (!yr || yr < 1900 || yr > 2100) return Alert.alert("Year required", "Enter a year (e.g. 2021).");
    const makeStr = pickerValue(make);
    if (!makeStr) return Alert.alert("Make required");
    const modelStr = pickerValue(model);
    if (!modelStr) return Alert.alert("Model required");

    const trimStr = pickerValue(trim);
    const engineStr = pickerValue(engine);

    // Prepend engine to mods_notes since the API doesn't have a dedicated engine column.
    const composedNotes = [
      engineStr ? `Engine: ${engineStr}` : null,
      notes.trim() || null,
    ].filter(Boolean).join("\n") || null;

    m.mutate({
      year: yr,
      make: makeStr,
      model: modelStr,
      trim: trimStr,
      color: color.trim() || null,
      mods_notes: composedNotes,
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
            <Text style={styles.title}>Add a car</Text>
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
                  placeholder="2021"
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
              label="Engine (optional)"
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
                  placeholder="e.g. Type R, GT3 RS"
                  placeholderTextColor={colors.textDim}
                  autoCapitalize="words"
                  style={styles.input}
                />
              </GlassCard>
            </View>

            <View style={{ marginBottom: spacing.md }}>
              <Text style={styles.fieldLabel}>COLOR (OPTIONAL)</Text>
              <GlassCard>
                <TextInput
                  value={color}
                  onChangeText={setColor}
                  placeholder="Nardo Grey"
                  placeholderTextColor={colors.textDim}
                  autoCapitalize="words"
                  style={styles.input}
                />
              </GlassCard>
            </View>

            <View style={{ marginBottom: spacing.md }}>
              <Text style={styles.fieldLabel}>MODS / NOTES (OPTIONAL)</Text>
              <GlassCard>
                <TextInput
                  value={notes}
                  onChangeText={setNotes}
                  placeholder="Catback, tune, etc."
                  placeholderTextColor={colors.textDim}
                  multiline
                  style={[styles.input, { minHeight: 90, paddingTop: 14 }]}
                />
              </GlassCard>
            </View>
          </ScrollView>

          <View style={styles.footer}>
            <PrimaryButton variant="accent" label={m.isPending ? "Saving…" : "Save car"} onPress={onSubmit} loading={m.isPending} />
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  title: { color: colors.text, fontSize: fontSize.large, fontWeight: "800", letterSpacing: -0.5 },
  cancel: { color: colors.textMuted, fontSize: fontSize.body, fontWeight: "600" },

  body: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: 60 },
  fieldLabel: { color: colors.textMuted, fontSize: fontSize.caption, fontWeight: "700", letterSpacing: 0.5, textTransform: "uppercase", marginBottom: 6 },
  input: { color: colors.text, fontSize: fontSize.body, paddingHorizontal: 14, paddingVertical: 14 },

  footer: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, paddingBottom: spacing.md },
});
