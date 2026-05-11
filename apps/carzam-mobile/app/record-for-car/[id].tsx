import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Audio } from "expo-av";
import * as DocumentPicker from "expo-document-picker";
import * as Haptics from "expo-haptics";
import { LinearGradient } from "expo-linear-gradient";
import { useLocalSearchParams, useRouter } from "expo-router";
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

import { listCars } from "@/api/cars";
import { uploadClipForCar } from "@/api/clips";
import { CarIcon } from "@/components/CarIcon";
import { MicPermissionGate } from "@/components/MicPermissionGate";
import { useMicPermission } from "@/hooks/useMicPermission";
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

type Phase = "idle" | "recording" | "uploading" | "done" | "error";

export default function RecordForCarScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const qc = useQueryClient();

  const carsQ = useQuery({ queryKey: ["cars"], queryFn: listCars });
  const car = carsQ.data?.find((c) => c.id === id);
  const mic = useMicPermission();

  const [phase, setPhase] = useState<Phase>("idle");
  const [secondsLeft, setSecondsLeft] = useState(RECORD_SECONDS);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => cleanup();
  }, []);

  function cleanup() {
    if (tickRef.current) clearInterval(tickRef.current);
    if (recordingRef.current) {
      recordingRef.current.stopAndUnloadAsync().catch(() => {});
      recordingRef.current = null;
    }
  }

  async function startRecording() {
    try {
      setErrorMsg(null);
      // Gate on mic permission. If not yet granted, request it and bail —
      // the user can tap Record again after they grant.
      if (mic.status !== "granted") {
        const next = await mic.request();
        if (next !== "granted") {
          // Don't render an error message here — the MicPermissionGate UI
          // (see below) shows the right CTA based on next ("denied" =>
          // "Allow Microphone", "blocked" => "Open Settings").
          return;
        }
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });
      const rec = new Audio.Recording();
      await rec.prepareToRecordAsync(recordingOptions);
      recordingRef.current = rec;
      await rec.startAsync();
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
      setPhase("recording");
      setSecondsLeft(RECORD_SECONDS);
      tickRef.current = setInterval(() => {
        setSecondsLeft((s) => (s <= 1 ? 0 : s - 1));
      }, 1000);
      setTimeout(() => stopAndUpload(), RECORD_SECONDS * 1000);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Couldn't start recording.";
      setErrorMsg(msg);
      setPhase("error");
    }
  }

  async function stopAndUpload() {
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
      await uploadClipForCar(id!, uri);
      qc.invalidateQueries({ queryKey: ["cars"] });
      qc.invalidateQueries({ queryKey: ["car", id] });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      setPhase("done");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Upload failed.";
      setErrorMsg(msg);
      setPhase("error");
    }
  }

  async function pickFromFiles() {
    try {
      setErrorMsg(null);
      const res = await DocumentPicker.getDocumentAsync({
        type: ["audio/wav", "audio/x-wav", "audio/*"],
        copyToCacheDirectory: true,
        multiple: false,
      });
      if (res.canceled) return;
      const asset = res.assets?.[0];
      if (!asset?.uri) throw new Error("No file selected.");
      setPhase("uploading");
      await uploadClipForCar(id!, asset.uri);
      qc.invalidateQueries({ queryKey: ["cars"] });
      qc.invalidateQueries({ queryKey: ["car", id] });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      setPhase("done");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Upload failed.";
      setErrorMsg(msg);
      setPhase("error");
    }
  }

  return (
    <View style={{ flex: 1, backgroundColor: colors.bg }}>
      <SafeAreaView style={{ flex: 1, paddingHorizontal: spacing.lg }} edges={["top", "bottom"]}>
        <View style={styles.topBar}>
          <Pressable onPress={() => { cleanup(); router.back(); }} style={styles.close}>
            <Text style={styles.closeText}>×</Text>
          </Pressable>
        </View>

        <View style={styles.head}>
          <Text style={styles.eyebrow}>RECORD AUDIO FOR</Text>
          <Text style={styles.title} numberOfLines={2}>
            {car ? `${car.year} ${car.make} ${car.model}` : "Loading…"}
          </Text>
          {car?.trim ? <Text style={styles.sub}>{car.trim}</Text> : null}
        </View>

        {phase === "idle" && (
          mic.status === "denied" || mic.status === "blocked" ? (
            <MicPermissionGate
              status={mic.status}
              onRequest={() => void mic.request()}
              onOpenSettings={mic.openSettings}
            />
          ) : (
            <View style={styles.center}>
              <Pressable onPress={startRecording}>
                <LinearGradient
                  colors={["#ffd9b8", "#ff7e1a", "#ff3d3d", "#8b0e2e"]}
                  start={{ x: 0.3, y: 0.2 }}
                  end={{ x: 0.7, y: 1 }}
                  style={styles.bigButton}
                >
                  <CarIcon size={56} color="#fff" />
                </LinearGradient>
              </Pressable>
              <Text style={styles.bigHint}>Tap to record 5 seconds</Text>

              <Pressable onPress={pickFromFiles} style={styles.altLink}>
                <Text style={styles.altLinkText}>or pick a WAV from Files</Text>
              </Pressable>
            </View>
          )
        )}

        {(phase === "recording" || phase === "uploading") && (
          <ListeningView
            secondsLeft={phase === "recording" ? secondsLeft : 0}
            uploading={phase === "uploading"}
          />
        )}

        {phase === "done" && (
          <View style={styles.center}>
            <View style={[styles.bigButton, { backgroundColor: "transparent", borderWidth: 2, borderColor: colors.success }]}>
              <Text style={{ color: colors.success, fontSize: 56 }}>✓</Text>
            </View>
            <Text style={styles.doneTitle}>Saved</Text>
            <Text style={styles.doneSub}>Clip uploaded for {car?.make} {car?.model}.</Text>
            <Pressable onPress={() => { cleanup(); router.back(); }} style={[styles.actionBtn, { backgroundColor: "#fff" }]}>
              <Text style={{ color: "#000", fontWeight: "700" }}>Done</Text>
            </Pressable>
          </View>
        )}

        {phase === "error" && (
          <View style={styles.center}>
            <Text style={styles.errTitle}>Couldn't save the clip</Text>
            <Text style={styles.errBody}>{errorMsg}</Text>
            <Pressable
              onPress={() => { setErrorMsg(null); setPhase("idle"); }}
              style={[styles.actionBtn, { backgroundColor: colors.glassFillStrong, borderWidth: 1, borderColor: colors.border }]}
            >
              <Text style={{ color: colors.text, fontWeight: "700" }}>Try again</Text>
            </Pressable>
          </View>
        )}
      </SafeAreaView>
    </View>
  );
}

function ListeningView({ secondsLeft, uploading }: { secondsLeft: number; uploading: boolean }) {
  const r1 = useSharedValue(0.4);
  const r2 = useSharedValue(0.4);
  const r3 = useSharedValue(0.4);
  useEffect(() => {
    r1.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1);
    const t1 = setTimeout(() => (r2.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1)), 600);
    const t2 = setTimeout(() => (r3.value = withRepeat(withTiming(1, { duration: 2400, easing: Easing.out(Easing.quad) }), -1)), 1200);
    return () => { clearTimeout(t1); clearTimeout(t2); cancelAnimation(r1); cancelAnimation(r2); cancelAnimation(r3); };
  }, [r1, r2, r3]);
  const s1 = useAnimatedStyle(() => ({ transform: [{ scale: 0.9 + r1.value * 1.4 }], opacity: 1 - r1.value }));
  const s2 = useAnimatedStyle(() => ({ transform: [{ scale: 0.9 + r2.value * 1.4 }], opacity: 1 - r2.value }));
  const s3 = useAnimatedStyle(() => ({ transform: [{ scale: 0.9 + r3.value * 1.4 }], opacity: 1 - r3.value }));

  return (
    <View style={styles.center}>
      <Text style={styles.eyebrow}>{uploading ? "UPLOADING" : "LISTENING"}</Text>
      <Text style={styles.timer}>{uploading ? "—" : `0:0${secondsLeft}`}</Text>
      <View style={styles.pulseStack}>
        <Animated.View style={[styles.ring, s1]} />
        <Animated.View style={[styles.ring, s2]} />
        <Animated.View style={[styles.ring, s3]} />
        <LinearGradient
          colors={["#ffd9b8", "#ff7e1a", "#ff3d3d", "#8b0e2e"]}
          start={{ x: 0.3, y: 0.2 }}
          end={{ x: 0.7, y: 1 }}
          style={styles.core}
        >
          {uploading ? <ActivityIndicator color="#fff" size="large" /> : <CarIcon size={56} color="#fff" />}
        </LinearGradient>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  topBar: { flexDirection: "row", justifyContent: "flex-end", marginBottom: spacing.sm },
  close: { width: 44, height: 44, borderRadius: 22, alignItems: "center", justifyContent: "center", backgroundColor: colors.glassFillStrong, borderWidth: 1, borderColor: colors.border },
  closeText: { color: colors.text, fontSize: 24, lineHeight: 24, fontWeight: "300" },

  head: { paddingTop: spacing.sm },
  eyebrow: { color: colors.accentHi, fontSize: 11, letterSpacing: 2, fontWeight: "700" },
  title: { color: colors.text, fontSize: fontSize.hero, fontWeight: "800", letterSpacing: -0.8, marginTop: 4 },
  sub: { color: colors.textMuted, fontSize: fontSize.small, marginTop: 4 },

  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  bigButton: { width: 130, height: 130, borderRadius: 65, alignItems: "center", justifyContent: "center", shadowColor: "#ff7e1a", shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.5, shadowRadius: 30 },
  bigHint: { color: colors.text, fontSize: fontSize.body, fontWeight: "600", marginTop: spacing.xl },
  altLink: { marginTop: spacing.lg, paddingVertical: spacing.sm, paddingHorizontal: spacing.md },
  altLinkText: { color: colors.textMuted, fontSize: fontSize.small, textDecorationLine: "underline" },

  timer: { color: colors.text, fontSize: 64, fontWeight: "200", letterSpacing: -2, marginTop: 6, marginBottom: 32 },
  pulseStack: { width: 220, height: 220, alignItems: "center", justifyContent: "center" },
  ring: { position: "absolute", width: 200, height: 200, borderRadius: 100, borderWidth: 1.5, borderColor: "rgba(255,126,26,0.55)" },
  core: { width: 110, height: 110, borderRadius: 55, alignItems: "center", justifyContent: "center", shadowColor: "#ff7e1a", shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.7, shadowRadius: 30 },

  doneTitle: { color: colors.text, fontSize: fontSize.large, fontWeight: "800", marginTop: spacing.lg },
  doneSub: { color: colors.textMuted, fontSize: fontSize.body, marginTop: 4, textAlign: "center" },

  errTitle: { color: colors.text, fontSize: fontSize.large, fontWeight: "800", marginBottom: spacing.sm },
  errBody: { color: colors.textMuted, fontSize: fontSize.body, textAlign: "center", marginBottom: spacing.lg, paddingHorizontal: spacing.lg },

  actionBtn: { marginTop: spacing.lg, paddingHorizontal: 28, paddingVertical: 14, borderRadius: radius.md },
});
