import { useQueryClient } from "@tanstack/react-query";
import { Audio } from "expo-av";
import * as Haptics from "expo-haptics";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  AppState,
  InteractionManager,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import Animated, {
  Easing,
  cancelAnimation,
  interpolate,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSpring,
  withTiming,
} from "react-native-reanimated";
import { SafeAreaView } from "react-native-safe-area-context";

import { recognizeAudio, submitAsNewCar } from "@/api/recognize";
import type { RecognizeResponse } from "@/api/types";
import { CarIcon } from "@/components/CarIcon";
import { GlassCard } from "@/components/GlassCard";
import { MicPermissionGate } from "@/components/MicPermissionGate";
import { useMicPermission } from "@/hooks/useMicPermission";
import { useModelStore } from "@/store/model";
import { colors, fontSize, radius, spacing } from "@/theme";

const RECORD_SECONDS = 5;
const SAMPLE_RATE = 16_000;
// Below this peak dBFS we treat the clip as silence and refuse to upload.
// iOS metering reports 0 dB = clipping, -160 dB = digital silence. A pocket
// recording with no engine typically peaks around -55 to -65 dB; a car at
// idle nearby is -25 to -15 dB.
const SILENCE_PEAK_DBFS = -45;

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

type Phase =
  | "idle"
  | "permission" // mic perms not yet granted — show MicPermissionGate
  | "recording"
  | "uploading"
  | "result"
  | "error";

export default function CarzamModal() {
  const router = useRouter();
  const qc = useQueryClient();
  const activeModelId = useModelStore((s) => s.activeModelId);
  const mic = useMicPermission();
  const [phase, setPhase] = useState<Phase>("idle");
  const [secondsLeft, setSecondsLeft] = useState(RECORD_SECONDS);
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const peakDbRef = useRef<number>(-160);
  const liveDbRef = useRef<number>(-160);
  const [liveDb, setLiveDb] = useState<number>(-160);
  // Once we've kicked off recording for this modal session, don't auto-start
  // it again even if mic state flips back to granted (which can happen after
  // returning from Settings while phase===result/error).
  const startedRef = useRef(false);

  // Drive phase from permission status. We DON'T touch the Audio API until
  // we know we have permission — calling requestPermissionsAsync inside a
  // useEffect race was the source of the "no prompt ever shown" bug.
  useEffect(() => {
    if (mic.status === "unknown") return; // still reading initial state

    if (mic.status === "granted") {
      // Auto-start once when permission is granted and we haven't yet started.
      // For all other phases (recording / uploading / result / error) we leave
      // the existing flow alone — the user might have returned from Settings
      // mid-result and we don't want to clobber it.
      if (!startedRef.current && (phase === "idle" || phase === "permission")) {
        startedRef.current = true;
        // Same scheduling guard as before: wait for interactions + foreground.
        let cancelled = false;
        const handle = InteractionManager.runAfterInteractions(() => {
          if (cancelled) return;
          if (AppState.currentState === "active") {
            start();
          } else {
            const sub = AppState.addEventListener("change", (next) => {
              if (next === "active") {
                sub.remove();
                if (!cancelled) start();
              }
            });
          }
        });
        return () => {
          cancelled = true;
          handle.cancel?.();
        };
      }
    } else {
      // Anything other than granted — show the permission gate unless we're
      // already showing a result/error from a prior successful run.
      if (phase === "idle" || phase === "permission") {
        setPhase("permission");
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mic.status]);

  useEffect(() => () => cleanup(), []);

  async function start() {
    try {
      setErrorMsg(null);

      // Audio session: switch into a recording-friendly mode BEFORE creating
      // the Recording. Doing it after sometimes silently produces empty
      // files on iOS when the app was previously playing audio.
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      // Recording.
      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync(recordingOptions);
      // Stream metering at ~10 Hz so the live core can react to mic level
      // and so we can compute a peak for the silence gate.
      rec.setProgressUpdateInterval(100);
      rec.setOnRecordingStatusUpdate((status) => {
        if (status.isRecording && typeof status.metering === "number") {
          const db = status.metering;
          liveDbRef.current = db;
          setLiveDb(db);
          if (db > peakDbRef.current) peakDbRef.current = db;
        }
      });
      peakDbRef.current = -160;
      liveDbRef.current = -160;
      recordingRef.current = rec;
      await rec.startAsync();
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
      setPhase("recording");
      setSecondsLeft(RECORD_SECONDS);

      // visible countdown
      tickRef.current = setInterval(() => {
        setSecondsLeft((s) => (s <= 1 ? 0 : s - 1));
      }, 1000);
      // auto-stop
      setTimeout(() => stopAndRecognize(), RECORD_SECONDS * 1000);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Couldn't start recording.";
      console.warn("[carzam] start() failed:", msg, e);
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

      // Silence gate. Refuse to upload if the loudest moment of the clip
      // never even cleared the noise floor — otherwise the model just
      // hallucinates a confident answer (typically "Lamborghini").
      if (peakDbRef.current < SILENCE_PEAK_DBFS) {
        throw new Error(
          "Couldn't hear an engine. Hold the phone closer to the car and try again.",
        );
      }

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
      const res = await recognizeAudio(uri, activeModelId);
      if (res.silence) {
        throw new Error(
          "Couldn't hear an engine. Hold the phone closer to the car and try again.",
        );
      }
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
          <Pressable
            onPress={() => { cleanup(); router.push("/(tabs)/me"); }}
            style={styles.modelChip}
            hitSlop={8}
          >
            <Text style={styles.modelChipLabel}>MODEL</Text>
            <Text style={styles.modelChipValue} numberOfLines={1}>
              {activeModelId ?? "default"}
            </Text>
          </Pressable>
          <Pressable onPress={() => { cleanup(); router.back(); }} style={styles.close}>
            <Text style={styles.closeText}>×</Text>
          </Pressable>
        </View>

        {phase === "permission" && mic.status !== "unknown" && mic.status !== "granted" && (
          <MicPermissionGate
            status={mic.status}
            onRequest={async () => {
              const next = await mic.request();
              // If the user just granted, the status-driven effect above will
              // pick it up and auto-start recording.
              if (next === "blocked") {
                // We attempted, OS won't re-prompt — surface the Settings CTA
                // by letting the gate re-render in "blocked" state.
              }
            }}
            onOpenSettings={mic.openSettings}
          />
        )}

        {(phase === "recording" || phase === "uploading") && (
          <ListeningView
            secondsLeft={phase === "recording" ? secondsLeft : 0}
            uploading={phase === "uploading"}
            liveDb={liveDb}
          />
        )}

        {phase === "result" && result && (
          <ResultView
            result={result}
            onDone={() => { cleanup(); router.back(); }}
            onWrong={() => {
              cleanup();
              router.replace(`/submit-new-car/${result.clip_id}`);
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

function ListeningView({
  secondsLeft,
  uploading,
  liveDb,
}: {
  secondsLeft: number;
  uploading: boolean;
  liveDb: number;
}) {
  // 4 rings, staggered phase, continuous sine ease via withRepeat+linear timing.
  const t = useSharedValue(0);
  const sweep = useSharedValue(0);
  const level = useSharedValue(0); // 0..1 audio reactive level

  useEffect(() => {
    t.value = withRepeat(
      withTiming(1, { duration: 3200, easing: Easing.linear }),
      -1,
      false,
    );
    sweep.value = withRepeat(
      withTiming(1, { duration: 5200, easing: Easing.linear }),
      -1,
      false,
    );
    return () => {
      cancelAnimation(t);
      cancelAnimation(sweep);
    };
  }, [t, sweep]);

  // Smooth the audio level toward the target. -60 dB ≈ silence, -10 dB ≈ loud.
  useEffect(() => {
    const norm = Math.max(0, Math.min(1, (liveDb + 60) / 50));
    level.value = withSpring(norm, { damping: 14, stiffness: 110, mass: 0.6 });
  }, [liveDb, level]);

  // Four staggered rings. Each is its own useAnimatedStyle so we obey Rules of Hooks.
  const r1 = useAnimatedStyle(() => {
    "worklet";
    const p = (t.value + 0.0) % 1;
    const reactive = 0.85 + level.value * 0.5;
    return {
      transform: [{ scale: (0.55 + p * 1.15) * reactive }],
      opacity: (1 - p) * (0.35 + level.value * 0.65),
    };
  });
  const r2 = useAnimatedStyle(() => {
    "worklet";
    const p = (t.value + 0.25) % 1;
    const reactive = 0.85 + level.value * 0.5;
    return {
      transform: [{ scale: (0.55 + p * 1.15) * reactive }],
      opacity: (1 - p) * (0.35 + level.value * 0.65),
    };
  });
  const r3 = useAnimatedStyle(() => {
    "worklet";
    const p = (t.value + 0.5) % 1;
    const reactive = 0.85 + level.value * 0.5;
    return {
      transform: [{ scale: (0.55 + p * 1.15) * reactive }],
      opacity: (1 - p) * (0.35 + level.value * 0.65),
    };
  });
  const r4 = useAnimatedStyle(() => {
    "worklet";
    const p = (t.value + 0.75) % 1;
    const reactive = 0.85 + level.value * 0.5;
    return {
      transform: [{ scale: (0.55 + p * 1.15) * reactive }],
      opacity: (1 - p) * (0.35 + level.value * 0.65),
    };
  });

  const coreStyle = useAnimatedStyle(() => {
    "worklet";
    const breathe = 1 + Math.sin(t.value * Math.PI * 2) * 0.03;
    const reactive = 1 + level.value * 0.18;
    return {
      transform: [{ scale: breathe * reactive }],
    };
  });

  const sweepStyle = useAnimatedStyle(() => {
    "worklet";
    return {
      transform: [{ rotate: `${sweep.value * 360}deg` }],
      opacity: 0.55 + level.value * 0.35,
    };
  });

  const meterStyle = useAnimatedStyle(() => {
    "worklet";
    return {
      width: `${interpolate(level.value, [0, 1], [8, 100])}%`,
      opacity: 0.35 + level.value * 0.55,
    };
  });

  return (
    <View style={lvStyles.wrap}>
      <Text style={lvStyles.heading}>{uploading ? "Identifying" : "Listening"}</Text>
      <Text style={lvStyles.timer}>{uploading ? "—" : `0:0${secondsLeft}`}</Text>

      <View style={lvStyles.pulseStack}>
        <Animated.View style={[lvStyles.ring, lvStyles.ringOuter, r1]} />
        <Animated.View style={[lvStyles.ring, lvStyles.ringMid, r2]} />
        <Animated.View style={[lvStyles.ring, lvStyles.ringInner, r3]} />
        <Animated.View style={[lvStyles.ring, lvStyles.ringFaint, r4]} />

        <Animated.View style={[lvStyles.coreWrap, coreStyle]}>
          <LinearGradient
            colors={["#ffd9b8", "#ff7e1a", "#ff3d3d", "#8b0e2e"]}
            start={{ x: 0.25, y: 0.15 }}
            end={{ x: 0.75, y: 1 }}
            style={lvStyles.core}
          >
            <Animated.View style={[StyleSheet.absoluteFillObject, sweepStyle]}>
              <LinearGradient
                colors={["transparent", "rgba(255,255,255,0.55)", "transparent"]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={lvStyles.sweep}
              />
            </Animated.View>
            {uploading ? (
              <ActivityIndicator color="#fff" size="large" />
            ) : (
              <CarIcon size={56} color="#fff" />
            )}
          </LinearGradient>
        </Animated.View>
      </View>

      {/* Live audio meter — gives the user a visible signal that the mic is hearing them */}
      <View style={lvStyles.meterTrack}>
        <Animated.View style={[lvStyles.meterFill, meterStyle]}>
          <LinearGradient
            colors={["#ffd9b8", "#ff7e1a", "#ff3d3d"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={StyleSheet.absoluteFillObject}
          />
        </Animated.View>
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
      {result.model_id && (
        <Text style={rvStyles.byline}>via {result.model_id}</Text>
      )}

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
  topBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.sm,
  },
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
  modelChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: colors.glassFillStrong,
    borderWidth: 1,
    borderColor: colors.border,
    maxWidth: "70%",
  },
  modelChipLabel: {
    color: colors.textDim,
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.4,
  },
  modelChipValue: {
    color: colors.text,
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: -0.2,
  },
  errorWrap: { flex: 1, alignItems: "center", justifyContent: "center", padding: spacing.xl },
  errorTitle: { color: colors.text, fontSize: fontSize.large, fontWeight: "800", marginBottom: spacing.sm },
  errorBody: { color: colors.textMuted, fontSize: fontSize.body, textAlign: "center", marginBottom: spacing.lg },
  errorBtn: { backgroundColor: "#fff", paddingHorizontal: 22, paddingVertical: 12, borderRadius: radius.md },
  errorBtnText: { color: "#000", fontWeight: "700" },
});

const lvStyles = StyleSheet.create({
  wrap: { flex: 1, alignItems: "center", justifyContent: "center", paddingTop: 40 },
  heading: { color: colors.textMuted, fontSize: 11, textTransform: "uppercase", letterSpacing: 3, fontWeight: "700" },
  timer: { color: colors.text, fontSize: 72, fontWeight: "200", letterSpacing: -2.5, marginTop: 8, marginBottom: 28 },
  pulseStack: { width: 280, height: 280, alignItems: "center", justifyContent: "center" },
  ring: {
    position: "absolute",
    width: 240,
    height: 240,
    borderRadius: 120,
    borderWidth: 1.5,
  },
  ringOuter: { borderColor: "rgba(255,126,26,0.55)" },
  ringMid: { borderColor: "rgba(255,180,122,0.45)", borderWidth: 1.25 },
  ringInner: { borderColor: "rgba(255,61,61,0.40)", borderWidth: 1 },
  ringFaint: { borderColor: "rgba(255,255,255,0.18)", borderWidth: 1 },
  coreWrap: {
    shadowColor: "#ff7e1a",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.85,
    shadowRadius: 36,
  },
  core: {
    width: 130, height: 130,
    borderRadius: 65,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  sweep: {
    width: "200%",
    height: "200%",
    position: "absolute",
    top: "-50%",
    left: "-50%",
  },
  meterTrack: {
    marginTop: 28,
    width: "70%",
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(255,255,255,0.08)",
    overflow: "hidden",
  },
  meterFill: {
    height: "100%",
    borderRadius: 2,
    overflow: "hidden",
  },
  hint: { color: colors.textMuted, fontSize: fontSize.small, textAlign: "center", marginTop: 18, lineHeight: 20 },
});

const rvStyles = StyleSheet.create({
  wrap: { flex: 1, paddingTop: spacing.md },
  eyebrow: { color: colors.accentHi, fontSize: 11, textTransform: "uppercase", letterSpacing: 2, fontWeight: "700" },
  title: { color: colors.text, fontSize: 30, fontWeight: "800", letterSpacing: -0.8, marginTop: 6 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 4 },
  byline: { color: colors.textDim, fontSize: 11, marginTop: 4, letterSpacing: 0.4, fontWeight: "600" },

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
