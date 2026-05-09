import * as AppleAuthentication from "expo-apple-authentication";
import { LinearGradient } from "expo-linear-gradient";
import { useState } from "react";
import { Alert, Image, Platform, Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { signInWithApple, signInWithGoogle } from "@/api/auth";
import { AmbientBackground } from "@/components/AmbientBackground";
import { CarIcon } from "@/components/CarIcon";
import { useAuthStore } from "@/store/auth";
import { colors, fontSize, radius, spacing } from "@/theme";

// Lazy-import google sign-in so a missing native module doesn't break Metro at boot.
async function googleSignInFlow() {
  const { GoogleSignin } = await import("@react-native-google-signin/google-signin");
  GoogleSignin.configure({
    iosClientId: process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID,
  });
  await GoogleSignin.hasPlayServices({ showPlayServicesUpdateDialog: false }).catch(() => {});
  const result = await GoogleSignin.signIn();
  // GoogleSignin response shape varies between versions — check both.
  // @ts-ignore
  const idToken: string | undefined = result.idToken ?? result.data?.idToken;
  if (!idToken) throw new Error("Google sign-in didn't return an ID token.");
  return idToken;
}

export default function SignInScreen() {
  const signIn = useAuthStore((s) => s.signIn);
  const [busy, setBusy] = useState<"google" | "apple" | null>(null);

  const onGoogle = async () => {
    if (busy) return;
    setBusy("google");
    try {
      const idToken = await googleSignInFlow();
      const auth = await signInWithGoogle(idToken);
      await signIn(auth.jwt, auth.user);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Sign-in failed.";
      if (!String(msg).toLowerCase().includes("cancel")) Alert.alert("Sign-in failed", msg);
    } finally {
      setBusy(null);
    }
  };

  const onApple = async () => {
    if (busy) return;
    setBusy("apple");
    try {
      const cred = await AppleAuthentication.signInAsync({
        requestedScopes: [
          AppleAuthentication.AppleAuthenticationScope.FULL_NAME,
          AppleAuthentication.AppleAuthenticationScope.EMAIL,
        ],
      });
      if (!cred.identityToken) throw new Error("Apple did not return an identity token.");
      const fullName = [cred.fullName?.givenName, cred.fullName?.familyName]
        .filter(Boolean)
        .join(" ") || null;
      const auth = await signInWithApple({
        idToken: cred.identityToken,
        authorizationCode: cred.authorizationCode ?? null,
        fullName,
        email: cred.email ?? null,
      });
      await signIn(auth.jwt, auth.user);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Sign-in failed.";
      if (!String(msg).toLowerCase().includes("cancel")) Alert.alert("Sign-in failed", msg);
    } finally {
      setBusy(null);
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: colors.bg }}>
      <AmbientBackground />
      <SafeAreaView style={styles.safe}>
        <View style={styles.brand}>
          <LinearGradient
            colors={["#ffb47a", "#ff7e1a", "#ff3d3d", "#8b0e2e"]}
            start={{ x: 0.3, y: 0.2 }}
            end={{ x: 0.7, y: 1 }}
            style={styles.brandIcon}
          >
            <CarIcon size={48} color="#fff" />
          </LinearGradient>
          <Text style={styles.brandTitle}>Carzam</Text>
          <Text style={styles.brandSub}>Identify cars by their engine sound.</Text>
        </View>

        <View style={styles.actions}>
          {Platform.OS === "ios" && (
            <AppleAuthentication.AppleAuthenticationButton
              buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_IN}
              buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.WHITE}
              cornerRadius={radius.md}
              style={styles.appleButton}
              onPress={onApple}
            />
          )}
          <Pressable
            style={({ pressed }) => [styles.googleButton, pressed && { opacity: 0.85 }]}
            onPress={onGoogle}
            disabled={busy !== null}
          >
            <View style={styles.googleG}>
              <Text style={styles.googleGText}>G</Text>
            </View>
            <Text style={styles.googleText}>
              {busy === "google" ? "Signing in…" : "Sign in with Google"}
            </Text>
          </Pressable>
          <Text style={styles.legal}>
            By continuing you agree to be a beta tester. No account data is shared with anyone.
          </Text>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, paddingHorizontal: spacing.xl, justifyContent: "space-between" },
  brand: { flex: 1, alignItems: "center", justifyContent: "center" },
  brandIcon: {
    width: 110,
    height: 110,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.xl,
    shadowColor: "#ff7e1a",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.6,
    shadowRadius: 24,
  },
  brandTitle: {
    fontSize: 44,
    fontWeight: "800",
    color: colors.text,
    letterSpacing: -1.5,
    marginBottom: spacing.xs,
  },
  brandSub: { color: colors.textMuted, fontSize: fontSize.body },
  actions: { paddingBottom: spacing.xxl, gap: spacing.md },
  appleButton: { height: 50 },
  googleButton: {
    height: 50,
    borderRadius: radius.md,
    backgroundColor: "#fff",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  googleG: {
    width: 22,
    height: 22,
    borderRadius: 4,
    backgroundColor: "#4285F4",
    alignItems: "center",
    justifyContent: "center",
  },
  googleGText: { color: "#fff", fontWeight: "800", fontSize: 14 },
  googleText: { color: "#000", fontWeight: "600", fontSize: 16 },
  legal: { color: colors.textDim, fontSize: 11, textAlign: "center", marginTop: spacing.md },
});
