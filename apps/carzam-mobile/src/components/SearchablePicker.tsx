import { BlurView } from "expo-blur";
import { useEffect, useMemo, useState } from "react";
import {
  FlatList,
  Keyboard,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { colors, fontSize, radius, spacing } from "@/theme";

const OTHER_VALUE = "__other__";

type Props = {
  /** Visible label above the field. */
  label: string;
  /** Current selection (one of `options`, the literal "__other__", a free-text custom value, or null). */
  value: string | null;
  /** Available options. "Other" is appended automatically. */
  options: readonly string[];
  /** Disabled = greyed out, taps no-op. Use this for a model field before a make is picked. */
  disabled?: boolean;
  /** Placeholder shown when no value is selected. */
  placeholder?: string;
  /** Fired when the user picks something. Pass the literal "__other__" sentinel for free-text mode. */
  onChange: (value: string) => void;
  /** Optional helper text. */
  hint?: string;
};

export function SearchablePicker({ label, value, options, disabled, placeholder, onChange, hint }: Props) {
  const [open, setOpen] = useState(false);
  const isCustom = value !== null && value !== OTHER_VALUE && !options.includes(value);
  const isOther = value === OTHER_VALUE || isCustom;

  const display = value === null
    ? (placeholder ?? "Select…")
    : isCustom
      ? value
      : value === OTHER_VALUE
        ? "Other (custom)"
        : value;

  return (
    <View style={{ marginBottom: spacing.md }}>
      <Text style={styles.label}>{label}</Text>

      <Pressable
        onPress={() => !disabled && setOpen(true)}
        style={[styles.field, disabled && { opacity: 0.45 }]}
      >
        <Text style={[styles.fieldText, value === null && styles.placeholder]}>
          {display}
        </Text>
        <Text style={styles.chevron}>›</Text>
      </Pressable>

      {isOther && (
        <View style={[styles.field, { marginTop: 6 }]}>
          <TextInput
            style={styles.customInput}
            value={isCustom ? value! : ""}
            onChangeText={(t) => onChange(t.length === 0 ? OTHER_VALUE : t)}
            placeholder="Type your own…"
            placeholderTextColor={colors.textDim}
            autoCapitalize="words"
          />
        </View>
      )}

      {hint && <Text style={styles.hint}>{hint}</Text>}

      <PickerSheet
        title={label}
        visible={open}
        options={options}
        currentValue={isCustom ? OTHER_VALUE : value}
        onClose={() => setOpen(false)}
        onPick={(v) => {
          onChange(v);
          setOpen(false);
        }}
      />
    </View>
  );
}

function PickerSheet({
  title,
  visible,
  options,
  currentValue,
  onClose,
  onPick,
}: {
  title: string;
  visible: boolean;
  options: readonly string[];
  currentValue: string | null;
  onClose: () => void;
  onPick: (v: string) => void;
}) {
  const [query, setQuery] = useState("");
  useEffect(() => {
    if (!visible) setQuery("");
  }, [visible]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const matched = q ? options.filter((o) => o.toLowerCase().includes(q)) : options;
    // Always pin "Other" to the end so the user has to scroll past everything to reach it.
    return [...matched, OTHER_VALUE];
  }, [options, query]);

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={styles.modalRoot}>
        <Pressable style={styles.scrim} onPress={onClose} />
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : undefined}
          style={styles.sheetWrap}
          keyboardVerticalOffset={20}
        >
          <View style={styles.sheet}>
            <BlurView intensity={70} tint="dark" style={StyleSheet.absoluteFillObject} />
            <View style={styles.sheetVeil} />
            <SafeAreaView edges={["bottom"]} style={{ flex: 1 }}>
              <View style={styles.handle} />
              <View style={styles.sheetHeader}>
                <Text style={styles.sheetTitle}>{title}</Text>
                <Pressable onPress={onClose} hitSlop={10}>
                  <Text style={styles.sheetClose}>Close</Text>
                </Pressable>
              </View>

              <View style={styles.searchWrap}>
                <Text style={styles.searchIcon}>⌕</Text>
                <TextInput
                  style={styles.search}
                  value={query}
                  onChangeText={setQuery}
                  placeholder="Search…"
                  placeholderTextColor={colors.textDim}
                  autoCapitalize="none"
                  autoCorrect={false}
                />
              </View>

              <FlatList
                data={filtered}
                keyExtractor={(item) => item}
                keyboardShouldPersistTaps="handled"
                contentContainerStyle={{ paddingBottom: 24 }}
                renderItem={({ item }) => {
                  const isOther = item === OTHER_VALUE;
                  const selected = currentValue === item;
                  return (
                    <Pressable
                      onPress={() => {
                        Keyboard.dismiss();
                        onPick(item);
                      }}
                      style={({ pressed }) => [
                        styles.row,
                        isOther && styles.rowOther,
                        pressed && { backgroundColor: "rgba(255,255,255,0.06)" },
                      ]}
                    >
                      <Text style={[styles.rowText, isOther && styles.rowOtherText]}>
                        {isOther ? "Other (type your own)" : item}
                      </Text>
                      {selected && <Text style={styles.checkmark}>✓</Text>}
                    </Pressable>
                  );
                }}
                ListEmptyComponent={
                  <View style={{ paddingVertical: spacing.xl, alignItems: "center" }}>
                    <Text style={{ color: colors.textDim }}>No matches</Text>
                  </View>
                }
              />
            </SafeAreaView>
          </View>
        </KeyboardAvoidingView>
      </View>
    </Modal>
  );
}

export const PICKER_OTHER = OTHER_VALUE;

/** Helper: turn a picker output back into a real string for the API.
 *  The literal sentinel becomes null (the user picked Other but didn't type anything). */
export function pickerValue(v: string | null): string | null {
  if (v === null || v === OTHER_VALUE) return null;
  return v;
}

const styles = StyleSheet.create({
  label: {
    color: colors.textMuted,
    fontSize: fontSize.caption,
    fontWeight: "700",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 6,
  },
  field: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 14,
    paddingVertical: 14,
    borderRadius: radius.md,
    backgroundColor: "rgba(20,20,28,0.45)",
    borderWidth: 1,
    borderColor: colors.border,
  },
  fieldText: { color: colors.text, fontSize: fontSize.body, flex: 1 },
  placeholder: { color: colors.textDim },
  chevron: { color: colors.textMuted, fontSize: 22, transform: [{ rotate: "90deg" }], marginLeft: spacing.sm },

  customInput: {
    flex: 1,
    color: colors.text,
    fontSize: fontSize.body,
  },

  hint: { color: colors.textDim, fontSize: fontSize.caption, marginTop: 6 },

  modalRoot: { flex: 1, justifyContent: "flex-end" },
  scrim: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(0,0,0,0.6)" },
  sheetWrap: { height: "78%" },
  sheet: {
    flex: 1,
    backgroundColor: "rgba(20,20,28,0.85)",
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    overflow: "hidden",
    borderTopWidth: 1,
    borderColor: colors.border,
  },
  sheetVeil: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(5,5,7,0.55)" },
  handle: {
    alignSelf: "center",
    width: 36,
    height: 4,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.18)",
    marginTop: spacing.sm,
  },
  sheetHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  sheetTitle: { color: colors.text, fontSize: fontSize.title, fontWeight: "800", letterSpacing: -0.3 },
  sheetClose: { color: colors.textMuted, fontSize: fontSize.body, fontWeight: "600" },

  searchWrap: {
    flexDirection: "row",
    alignItems: "center",
    marginHorizontal: spacing.lg,
    marginBottom: spacing.sm,
    paddingHorizontal: 12,
    backgroundColor: "rgba(255,255,255,0.06)",
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  searchIcon: { color: colors.textMuted, fontSize: 18, marginRight: 8 },
  search: { flex: 1, color: colors.text, fontSize: fontSize.body, paddingVertical: 12 },

  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.lg,
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "rgba(255,255,255,0.06)",
  },
  rowOther: { backgroundColor: "rgba(255,126,26,0.06)" },
  rowText: { color: colors.text, fontSize: fontSize.body },
  rowOtherText: { color: colors.accentHi, fontWeight: "700" },
  checkmark: { color: colors.success, fontSize: 18, fontWeight: "800" },
});
