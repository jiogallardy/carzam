import { BlurView } from "expo-blur";
import { useEffect } from "react";
import { StyleSheet, View } from "react-native";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from "react-native-reanimated";

import { colors } from "@/theme";

// Three drifting blurry orbs that fill the viewport. Provides the "liquid"
// the glass surfaces refract. We render the orbs and then put a BlurView
// over the whole stack to produce the iOS-26 liquid-glass feel.
export function AmbientBackground() {
  const t1 = useSharedValue(0);
  const t2 = useSharedValue(0);
  const t3 = useSharedValue(0);

  useEffect(() => {
    t1.value = withRepeat(withSequence(withTiming(1, { duration: 18000 }), withTiming(0, { duration: 18000 })), -1);
    t2.value = withRepeat(withSequence(withTiming(1, { duration: 22000 }), withTiming(0, { duration: 22000 })), -1);
    t3.value = withRepeat(withSequence(withTiming(1, { duration: 26000 }), withTiming(0, { duration: 26000 })), -1);
  }, [t1, t2, t3]);

  const s1 = useAnimatedStyle(() => ({ transform: [{ translateX: 50 * t1.value }, { translateY: -40 * t1.value }] }));
  const s2 = useAnimatedStyle(() => ({ transform: [{ translateX: -40 * t2.value }, { translateY: 50 * t2.value }] }));
  const s3 = useAnimatedStyle(() => ({ transform: [{ translateX: 30 * t3.value }, { translateY: -30 * t3.value }] }));

  return (
    <View pointerEvents="none" style={styles.root}>
      <Animated.View style={[styles.blob, styles.b1, s1]} />
      <Animated.View style={[styles.blob, styles.b2, s2]} />
      <Animated.View style={[styles.blob, styles.b3, s3]} />
      {/* Heavy blur over the whole stack — this is what makes the orbs read
          as "liquid" rather than as flat colored circles. */}
      <BlurView intensity={70} tint="dark" style={StyleSheet.absoluteFillObject} />
      {/* Subtle dark veil so the blurred orbs don't wash everything out */}
      <View style={styles.veil} />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { ...StyleSheet.absoluteFillObject, backgroundColor: colors.bg, overflow: "hidden" },
  blob: {
    position: "absolute",
    borderRadius: 999,
    opacity: 0.85,
  },
  b1: { top: -120, left: -80, width: 420, height: 420, backgroundColor: "#ff5e2e" },
  b2: { top: 220, right: -140, width: 480, height: 480, backgroundColor: "#6a3aff" },
  b3: { bottom: -140, left: 60, width: 400, height: 400, backgroundColor: "#00b8d4" },
  veil: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(5,5,7,0.55)" },
});
