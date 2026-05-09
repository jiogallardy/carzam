import { useQueryClient } from "@tanstack/react-query";
import { Audio } from "expo-av";
import * as Haptics from "expo-haptics";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Animated, {
  Easing,
  cancelAnimation,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from "react-native-reanimated";
import { SafeAreaView } from "react-native-safe-area-context";

import { recognizeAudio, submitAsNewCar } from "@/api/recognize";
import type { RecognizeResponse } from "@/api/types";
import { CarIcon } from "@/components/CarIcon";
import { GlassCard } from "@/components/GlassCard";
import { colors, fontSize, radius, spacing } from "@/theme";

const RECORD_SECONDS = 5;
const SAMPLE_RATE = 16_000;

const recordingOptions: Audio.RecordingOptions = {
  isMeteringEnabled: true,
  android: {
    extension: ".wav",
    outputFormat: Audio.AndroidOutputFormat.DEFAULT,
    audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
    sampleRate: SAMPLE_RATE,
    numberOfChannels: 1,
    bitRate: 256_000,
  },
  ios: {
    extension: ".wav",
    outputFormat: Audio.IOSOutputFormat.LINEARPCM,
    audioQuality: Audio.IOSAudioQuality.HIGH,
    sampleRate: SAMPLE_RATE,
    numberOfChannels: 1,
    bitRate: 256_000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  web: { mimeType: "audio/wav", bitsPerSecond: 128_000 },
};

type Phase = "idle" | "recording" | "uploading" | "result" | "error";

export default function CarzamModal() {
  const router = useRouter();
  const qc = useQueryClient();
  const [phase, setPhase] = useState<Phase>("idle");
  const [secondsLeft, setSecondsLeft] = useState(RECORD_SECONDS);
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-start recording on mount
  useEffect(() => {
    start();
    return () => cleanup();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function start() {
    try {
      setErrorMsg(null);
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        setErrorMsg("Mic permission denied. Enable it in Settings to use Carzam.");
        setPhase("error");
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync(recordingOptions);
      recordingRef.current = rec;
      await rec.startAsync();
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
      setPhase("recording");
      setSecondsLeft(RECORD_SECONDS);
      // tick the visible countdown
      tickRef.current = setInterval(() => {
        setSecondsLeft((s) => {
          if (s <= 1) return 0;
          return s - 1;
        });
      }, 1000);
      // schedule auto-stop
      setTimeout(() => stopAndRecognize(), RECORD_SECONDS * 1000);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Couldn't start recording.";
      setErrorMsg(msg);
      setPhase("error");
    }
  }

  async function stopAndRecognize() {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    const rec = recordingRef.current;
    if (!rec) return;
    try {
      setPhase("uploading");
      await rec.stopAndUnloadAsync();
      const uri = rec.getURI();
      recordingRef.current = null;
      if (!uri) throw new Error("Recording produced no file.");
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
      const res = await recognizeAudio(uri);
      setResult(res);
      setPhase("result");
      Haptics.notificationAsync(
        res.discovery_unlocked ? Haptics.NotificationFeedbackType.Success : Haptics.NotificationFeedbackType.Warning,
      ).catch(() => {});
      // refresh dependent queries
      qc.invalidateQueries({ queryKey: ["discoveries"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Recognition failed.";
      setErrorMsg(msg);
      setPhase("error");
    }
  }

  function cleanup() {
    if (tickRef.current) clearInterval(tickRef.current);
    if (recordingRef.current) {
      recordingRef.current.stopAndUnloadAsync().catch(() => {});
      recordingRef.current = null;
    }
  }

  return (
    <View style={styles.root}>
      <SafeAreaView style={styles.safe} edges={["top", "bottom"]}>
        <View style={styles.topBar}>
          <Pressable onPress={() => { cleanup(); router.back(); }} style={styles.close}>
            <Text style={styles.closeText}>×</Text>
          </Pressable>
        </View>

        {(phase === "recording" || phase === "uploading") && (
          <ListeningView
            secondsLeft={phase === "recording" ? secondsLeft : 0}
            uploading={phase === "uploading"}
          />
        )}

        {phase === "result" && result && (
          <ResultView
            result={result}
            onDone={() => { cleanup(); router.back(); }}
            onWrong={async () => {
              Alert.alert(
                "Submit as new car",
                "Tap and we'll save this clip for review with the specs you enter next. (Coming soon — just close for now.)",
              );
            }}
          />
        )}

        {phase === "error" && (
          <View style={styles.errorWrap}>
            <Text style={styles.errorTitle}>Something went wrong</Text>
            <Text style={styles.errorBody}>{errorMsg}</Text>
            <Pressable onPress={() => { cleanup(); router.back(); }} style={styles.errorBtn}>
              <Text style={styles.errorBtnText}>Close</Text>
            </Pressable>
          </View>
        )}
      </SafeAreaView>

      <BackgroundGlow />
    </View>
  );
}

function BackgroundGlow() {
  return (
    <LinearGradient
      colors={["rgba(255,80,30,0.55)", "rgba(106,58,255,0.20)", "transparent"]}
      start={{ x: 0.5, y: 0.0 }}
      end={{ x: 0.5, y: 0.8 }}
      style={StyleSheet.absoluteFillObject}
      pointerEvents="none"
    />
  );
}

function ListeningView({ secondsLeft, uploading }: { secondsLeft: number; uploading: boolean }) {
  const r1 = useSharedValue(0.4);
  const r2 = useSharedValue(0.4);
  const r3 = useSharedValue(0.4);

  useEffect(() => {
    r1.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1);
    r2.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1, false);
    r3.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1, false);
    const t1 = setTimeout(() => (r2.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1)), 600);
    const t2 = setTimeout(() => (r3.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1)), 1200);
    return () => {
      clearTimeout(t1); clearTimeout(t2);
      cancelAnimation(r1); cancelAnimation(r2); cancelAnimation(r3);
    };
  }, [r1, r2, r3]);

  const s1 = useAnimatedStyle(() => ({
    transform: [{ scale: 0.9 + r1.value * 1.4 }],
    opacity: 1 - r1.value,
  }));
  const s2 = useAnimatedStyle(() => ({
    transform: [{ scale: 0.9 + r2.value * 1.4 }],
    opacity: 1 - r2.value,
  }));
  const s3 = useAnimatedStyle(() => ({
    transform: [{ scale: 0.9 + r3.value * 1.4 }],
    opacity: 1 - r3.value,
  }));

  return (
    <View style={lvStyles.wrap}>
      <Text style={lvStyles.heading}>{uploading ? "Uploading" : "Listening"}</Text>
      <Text style={lvStyles.timer}>{uploading ? "—" : `0:0${secondsLeft}`}</Text>

      <View style={lvStyles.pulseStack}>
        <Animated.View style={[lvStyles.ring, s1]} />
        <Animated.View style={[lvStyles.ring, s2]} />
        <Animated.View style={[lvStyles.ring, s3]} />
        <LinearGradient
          colors={["#ffd9b8", "#ff7e1a", "#ff3d3d", "#8b0e2e"]}
          start={{ x: 0.3, y: 0.2 }}
          end={{ x: 0.7, y: 1 }}
          style={lvStyles.core}
        >
          {uploading ? <ActivityIndicator color="#fff" size="large" /> : <CarIcon size={56} color="#fff" />}
        </LinearGradient>
      </View>

      <Text style={lvStyles.hint}>
        {uploading
          ? "Sending to the model…"
          : "Hold the phone near the engine.\nAuto-stops in 5 seconds."}
      </Text>
    </View>
  );
}

function ResultView({
  result,
  onDone,
  onWrong,
}: {
  result: RecognizeResponse;
  onDone: () => void;
  onWrong: () => void;
}) {
  const matched = result.predicted_class_id !== null;
  return (
    <View style={rvStyles.wrap}>
      <Text style={rvStyles.eyebrow}>{matched ? "MATCH FOUND" : "LOW CONFIDENCE"}</Text>
      <Text style={rvStyles.title} numberOfLines={2}>
        {result.predicted_display_name ?? "Couldn't place this one"}
      </Text>
      <Text style={rvStyles.sub}>
        {result.predicted_state} · {Math.round(result.confidence * 100)}% confidence
      </Text>

      <GlassCard style={{ marginTop: spacing.lg }}>
        <View style={rvStyles.top3Inner}>
          <Text style={rvStyles.top3Label}>Top 3 candidates</Text>
          {result.top3.map((c, i) => (
            <View
              key={c.car_class_id}
              style={[rvStyles.row, i > 0 && { borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.06)" }]}
            >
              <Text style={[rvStyles.rowName, i === 0 && rvStyles.rowNameWin]}>{c.display_name}</Text>
              <Text style={[rvStyles.rowConf, i === 0 && rvStyles.rowConfWin]}>
                {Math.round(c.confidence * 100)}%
              </Text>
            </View>
          ))}
        </View>
      </GlassCard>

      {matched && (
        <View style={rvStyles.discovery}>
          <View>
            <Text style={rvStyles.dT1}>{result.discovery_unlocked ? "NEW DISCOVERY" : "ALREADY DISCOVERED"}</Text>
            <Text style={rvStyles.dT2}>
              {result.n_discovered} of {result.n_total} found
            </Text>
          </View>
          <View style={rvStyles.dPct}>
            <Text style={rvStyles.dPctText}>{result.completion_percent}%</Text>
          </View>
        </View>
      )}

      <View style={rvStyles.actions}>
        <Pressable onPress={matched ? onWrong : onWrong} style={[rvStyles.btn, rvStyles.btnGhost]}>
          <Text style={rvStyles.btnGhostText}>{matched ? "Wrong?" : "Add as new car"}</Text>
        </Pressable>
        <Pressable onPress={onDone} style={[rvStyles.btn, rvStyles.btnPrimary]}>
          <Text style={rvStyles.btnPrimaryText}>Done</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  safe: { flex: 1, paddingHorizontal: spacing.lg },
  topBar: { flexDirection: "row", justifyContent: "flex-end", marginBottom: spacing.sm },
  close: {
    width: 44, height: 44,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.glassFillStrong,
    borderWidth: 1,
    borderColor: colors.border,
  },
  closeText: { color: colors.text, fontSize: 24, lineHeight: 24, fontWeight: "300" },
  errorWrap: { flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.xl },
  errorTitle: { color: colors.text, fontSize: fontSize.large, fontWeight: "800", marginBottom: spacing.sm },
  errorBody: { color: colors.textMuted, fontSize: fontSize.body, textAlign: "center", marginBottom: spacing.lg },
  errorBtn: { backgroundColor: "#fff", paddingHorizontal: 22, paddingVertical: 12, borderRadius: radius.md },
  errorBtnText: { color: "#000", fontWeight: "700" },
});

const lvStyles = StyleSheet.create({
  wrap: { flex: 1, alignItems: "center", justifyContent: "center", paddingTop: 40 },
  heading: { color: colors.textMuted, fontSize: 11, textTransform: "uppercase", letterSpacing: 2.5, fontWeight: "700" },
  timer: { color: colors.text, fontSize: 64, fontWeight: "200", letterSpacing: -2, marginTop: 6, marginBottom: 32 },
  pulseStack: { width: 220, height: 220, alignItems: "center", justifyContent: "center" },
  ring: {
    position: "absolute",
    width: 200,
    height: 200,
    borderRadius: 100,
    borderWidth: 1.5,
    borderColor: "rgba(255,126,26,0.55)",
  },
  core: {
    width: 110, height: 110,
    borderRadius: 55,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#ff7e1a",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 30,
  },
  hint: { color: colors.textMuted, fontSize: fontSize.small, textAlign: "center", marginTop: 32, lineHeight: 20 },
});

const rvStyles = StyleSheet.create({
  wrap: { flex: 1, paddingTop: spacing.md },
  eyebrow: { color: colors.accentHi, fontSize: 11, textTransform: "uppercase", letterSpacing: 2, fontWeight: "700" },
  title: { color: colors.text, fontSize: 30, fontWeight: "800", letterSpacing: -0.8, marginTop: 6 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 4 },

  top3Inner: { padding: spacing.md },
  top3Label: { color: colors.textMuted, fontSize: 10, textTransform: "uppercase", letterSpacing: 1.5, fontWeight: "700", marginBottom: 8 },
  row: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 8 },
  rowName: { color: colors.text, fontSize: fontSize.body },
  rowNameWin: { fontWeight: "700" },
  rowConf: { color: colors.textMuted, fontSize: fontSize.body, fontWeight: "600" },
  rowConfWin: { color: colors.accentHi },

  discovery: {
    marginTop: spacing.md,
    padding: spacing.md,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.18)",
    backgroundColor: "rgba(255,126,26,0.18)",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  dT1: { color: colors.text, fontSize: 11, letterSpacing: 1.2, fontWeight: "700" },
  dT2: { color: colors.text, fontSize: fontSize.title, fontWeight: "800", marginTop: 2 },
  dPct: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 14,
    backgroundColor: "rgba(0,0,0,0.20)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.15)",
  },
  dPctText: { color: colors.text, fontWeight: "800", fontSize: 16 },

  actions: { flexDirection: "row", gap: spacing.md, marginTop: "auto", paddingBottom: spacing.lg },
  btn: { flex: 1, height: 50, borderRadius: radius.md, alignItems: "center", justifyContent: "center" },
  btnPrimary: { backgroundColor: "#fff" },
  btnPrimaryText: { color: "#000", fontWeight: "700" },
  btnGhost: { backgroundColor: colors.glassFillStrong, borderWidth: 1, borderColor: colors.border },
  btnGhostText: { color: colors.text, fontWeight: "700" },
});
