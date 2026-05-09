import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "expo-router";
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

import { addCar } from "@/api/cars";
import { GlassCard } from "@/components/GlassCard";
import { PrimaryButton } from "@/components/PrimaryButton";
import { colors, fontSize, radius, spacing } from "@/theme";

export default function AddCarModal() {
  const router = useRouter();
  const qc = useQueryClient();
  const [year, setYear] = useState("");
  const [make, setMake] = useState("");
  const [model, setModel] = useState("");
  const [trim, setTrim] = useState("");
  const [color, setColor] = useState("");
  const [notes, setNotes] = useState("");

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
    if (!make.trim()) return Alert.alert("Make required");
    if (!model.trim()) return Alert.alert("Model required");
    m.mutate({
      year: yr,
      make: make.trim(),
      model: model.trim(),
      trim: trim.trim() || null,
      color: color.trim() || null,
      mods_notes: notes.trim() || null,
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
            <Text style={styles.title}>Add a car</Text>
            <Pressable onPress={() => router.back()} hitSlop={10}>
              <Text style={styles.cancel}>Cancel</Text>
            </Pressable>
          </View>

          <ScrollView contentContainerStyle={styles.body} keyboardShouldPersistTaps="handled">
            <Field label="Year" value={year} onChangeText={setYear} placeholder="2021" keyboardType="number-pad" maxLength={4} />
            <Field label="Make" value={make} onChangeText={setMake} placeholder="Honda" autoCapitalize="words" />
            <Field label="Model" value={model} onChangeText={setModel} placeholder="Civic Type R" autoCapitalize="words" />
            <Field label="Trim (optional)" value={trim} onChangeText={setTrim} placeholder="—" autoCapitalize="words" />
            <Field label="Color (optional)" value={color} onChangeText={setColor} placeholder="—" autoCapitalize="words" />
            <Field label="Mods / notes (optional)" value={notes} onChangeText={setNotes} placeholder="Catback, tune, etc." multiline />
          </ScrollView>

          <View style={styles.footer}>
            <PrimaryButton variant="accent" label={m.isPending ? "Saving…" : "Save car"} onPress={onSubmit} loading={m.isPending} />
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
  input: {
    color: colors.text,
    fontSize: fontSize.body,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },

  footer: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, paddingBottom: spacing.md },
});
