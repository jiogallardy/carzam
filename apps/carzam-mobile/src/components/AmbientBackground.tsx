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

// Three drifting blurry orbs that fill the viewport. Provides the "liquid" the
// glass surfaces refract.
export function AmbientBackground() {
  const t1 = useSharedValue(0);
  const t2 = useSharedValue(0);
  const t3 = useSharedValue(0);

  useEffect(() => {
    t1.value = withRepeat(withSequence(withTiming(1, { duration: 18000 }), withTiming(0, { duration: 18000 })), -1);
    t2.value = withRepeat(withSequence(withTiming(1, { duration: 22000 }), withTiming(0, { duration: 22000 })), -1);
    t3.value = withRepeat(withSequence(withTiming(1, { duration: 26000 }), withTiming(0, { duration: 26000 })), -1);
  }, [t1, t2, t3]);

  const s1 = useAnimatedStyle(() => ({ transform: [{ translateX: 40 * t1.value }, { translateY: -30 * t1.value }] }));
  const s2 = useAnimatedStyle(() => ({ transform: [{ translateX: -30 * t2.value }, { translateY: 40 * t2.value }] }));
  const s3 = useAnimatedStyle(() => ({ transform: [{ translateX: 25 * t3.value }, { translateY: -25 * t3.value }] }));

  return (
    <View pointerEvents="none" style={styles.root}>
      <Animated.View style={[styles.blob, styles.b1, s1]} />
      <Animated.View style={[styles.blob, styles.b2, s2]} />
      <Animated.View style={[styles.blob, styles.b3, s3]} />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { ...StyleSheet.absoluteFillObject, backgroundColor: colors.bg, overflow: "hidden" },
  blob: {
    position: "absolute",
    borderRadius: 999,
    opacity: 0.55,
    // RN doesn't support CSS filter:blur on iOS reliably for shadows of this
    // scale. We approximate with large size + opacity. For production polish
    // we can swap in BlurView clipped to a circle.
  },
  b1: { top: -80, left: -40, width: 360, height: 360, backgroundColor: "#ff5e2e" },
  b2: { top: 200, right: -100, width: 420, height: 420, backgroundColor: "#6a3aff" },
  b3: { bottom: -100, left: 80, width: 360, height: 360, backgroundColor: "#00b8d4" },
});
