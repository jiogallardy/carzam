import { BlurView } from "expo-blur";
import { LinearGradient } from "expo-linear-gradient";
import { Tabs, useRouter } from "expo-router";
import { useEffect } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import Animated, {
  Easing,
  cancelAnimation,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from "react-native-reanimated";
import { SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";
import Svg, { Circle, Path } from "react-native-svg";

import { CarIcon } from "@/components/CarIcon";
import { colors, radius } from "@/theme";

type IconName = "garage" | "found" | "stats" | "me";

function TabIcon({ name, color }: { name: IconName; color: string }) {
  const stroke = { stroke: color, strokeWidth: 2, fill: "none" as const, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  return (
    <Svg width={22} height={22} viewBox="0 0 24 24">
      {name === "garage" && (
        <>
          <Path d="M3 10l9-7 9 7v11H3z" {...stroke} />
          <Path d="M9 21V13h6v8" {...stroke} />
        </>
      )}
      {name === "found" && (
        <>
          <Circle cx={12} cy={12} r={9} {...stroke} />
          <Path d="M3 12h18" {...stroke} />
          <Path d="M12 3a14 14 0 010 18" {...stroke} />
          <Path d="M12 3a14 14 0 000 18" {...stroke} />
        </>
      )}
      {name === "stats" && (
        <>
          <Path d="M3 21V9" {...stroke} />
          <Path d="M9 21V3" {...stroke} />
          <Path d="M15 21v-7" {...stroke} />
          <Path d="M21 21v-13" {...stroke} />
        </>
      )}
      {name === "me" && (
        <>
          <Circle cx={12} cy={8} r={4} {...stroke} />
          <Path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8" {...stroke} />
        </>
      )}
    </Svg>
  );
}

function CenterCarzamButton({ onPress }: { onPress: () => void }) {
  const t = useSharedValue(0);

  useEffect(() => {
    t.value = withRepeat(
      withTiming(1, { duration: 2200, easing: Easing.inOut(Easing.sin) }),
      -1,
      true,
    );
    return () => cancelAnimation(t);
  }, [t]);

  const haloStyle = useAnimatedStyle(() => ({
    transform: [{ scale: 1 + t.value * 0.35 }],
    opacity: 0.55 - t.value * 0.45,
  }));

  const buttonStyle = useAnimatedStyle(() => ({
    transform: [{ scale: 1 + t.value * 0.04 }],
  }));

  return (
    <Pressable onPress={onPress} hitSlop={12} style={styles.centerWrap}>
      {/* Soft expanding halo behind the button */}
      <Animated.View pointerEvents="none" style={[styles.centerHalo, haloStyle]} />
      <Animated.View style={buttonStyle}>
        <LinearGradient
          colors={["#ffd9b8", "#ff7e1a", "#ff3d3d", "#8b0e2e"]}
          start={{ x: 0.3, y: 0.2 }}
          end={{ x: 0.7, y: 1 }}
          style={styles.centerOuter}
        >
          <View style={styles.centerSpecular} />
          <CarIcon size={36} color="#fff" />
        </LinearGradient>
      </Animated.View>
    </Pressable>
  );
}

export default function TabsLayout() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const barBottom = Math.max(insets.bottom, 14);

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        sceneStyle: { backgroundColor: colors.bg },
      }}
      tabBar={({ state, descriptors, navigation }) => {
        const order: IconName[] = ["garage", "found", "stats", "me"];
        const routesByName = Object.fromEntries(state.routes.map((r) => [r.name, r]));
        const visible: { name: string; icon: IconName; label: string; isFocused: boolean }[] = [
          { name: "index", icon: "garage", label: "Garage", isFocused: state.index === state.routes.findIndex((r) => r.name === "index") },
          { name: "found", icon: "found", label: "Found", isFocused: state.index === state.routes.findIndex((r) => r.name === "found") },
          { name: "stats", icon: "stats", label: "Stats", isFocused: state.index === state.routes.findIndex((r) => r.name === "stats") },
          { name: "me", icon: "me", label: "Me", isFocused: state.index === state.routes.findIndex((r) => r.name === "me") },
        ];

        return (
          <View pointerEvents="box-none" style={[styles.barWrap, { bottom: barBottom }]}>
            <BlurView intensity={70} tint="dark" style={StyleSheet.absoluteFillObject} />
            <View style={styles.barOverlay} />
            <View style={styles.specular} />
            <View style={styles.barRow}>
              {visible.slice(0, 2).map(({ name, icon, label, isFocused }) => (
                <Pressable
                  key={name}
                  onPress={() => navigation.navigate(routesByName[name].name as never)}
                  style={styles.tab}
                >
                  <TabIcon name={icon} color={isFocused ? colors.text : colors.textDim} />
                  <Text style={[styles.tabLabel, { color: isFocused ? colors.text : colors.textDim }]}>{label}</Text>
                </Pressable>
              ))}
              <View style={styles.tab}>
                <CenterCarzamButton onPress={() => router.push("/carzam")} />
              </View>
              {visible.slice(2).map(({ name, icon, label, isFocused }) => (
                <Pressable
                  key={name}
                  onPress={() => navigation.navigate(routesByName[name].name as never)}
                  style={styles.tab}
                >
                  <TabIcon name={icon} color={isFocused ? colors.text : colors.textDim} />
                  <Text style={[styles.tabLabel, { color: isFocused ? colors.text : colors.textDim }]}>{label}</Text>
                </Pressable>
              ))}
            </View>
          </View>
        );
      }}
    >
      <Tabs.Screen name="index" options={{ title: "Garage" }} />
      <Tabs.Screen name="found" options={{ title: "Found" }} />
      <Tabs.Screen name="stats" options={{ title: "Stats" }} />
      <Tabs.Screen name="me" options={{ title: "Me" }} />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  barWrap: {
    position: "absolute",
    left: 14,
    right: 14,
    height: 64,
    borderRadius: 32,
    overflow: "visible",
    backgroundColor: "rgba(20,20,28,0.40)",
    borderWidth: 1,
    borderColor: colors.border,
    shadowColor: "#000",
    shadowOpacity: 0.5,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 12 },
  },
  barOverlay: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 32,
    backgroundColor: "rgba(255,255,255,0.04)",
  },
  specular: {
    position: "absolute",
    top: 0,
    left: "8%",
    right: "8%",
    height: 1,
    backgroundColor: "rgba(255,255,255,0.20)",
  },
  barRow: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 3,
  },
  tabLabel: { fontSize: 10, fontWeight: "600", letterSpacing: 0.3 },
  centerWrap: {
    alignItems: "center",
    justifyContent: "center",
  },
  centerHalo: {
    position: "absolute",
    top: -38,
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: "rgba(255,126,26,0.45)",
  },
  centerOuter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    marginTop: -38,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 4,
    borderColor: "#050507",
    shadowColor: "#ff7e1a",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.65,
    shadowRadius: 24,
  },
  centerSpecular: {
    position: "absolute",
    top: 6,
    left: 12,
    right: 12,
    height: 18,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.55)",
    opacity: 0.7,
  },
});
