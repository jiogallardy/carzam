import { Audio } from "expo-av";
import { useCallback, useEffect, useRef, useState } from "react";
import { AppState, Linking } from "react-native";

/**
 * Microphone-permission state machine.
 *
 *   unknown    — initial, before we've asked the OS for status
 *   granted    — go ahead, recording is allowed
 *   undetermined — never been requested; tapping "Allow" will trigger the
 *                  iOS system prompt
 *   denied     — user previously denied, but iOS still lets us call
 *                requestPermissionsAsync (will re-prompt the first time)
 *   blocked    — denied and the OS won't re-prompt; user must open Settings
 *
 * The distinction between `denied` and `blocked` exists because once a user
 * has tapped "Don't allow" in the system prompt on iOS, the OS silently
 * resolves all future requestPermissionsAsync calls without showing UI.
 * `canAskAgain === false` is the signal — we surface that as `blocked` so the
 * caller knows to deep-link to Settings instead of calling request again.
 */
export type MicPermissionStatus =
  | "unknown"
  | "granted"
  | "undetermined"
  | "denied"
  | "blocked";

export interface MicPermission {
  status: MicPermissionStatus;
  /** Ask the OS to grant permission. On iOS this only shows a prompt the first
   *  time (and on the first denial); after that it returns silently. Returns
   *  the resulting status. */
  request: () => Promise<MicPermissionStatus>;
  /** Open the iOS/Android system Settings page for this app so the user can
   *  flip the toggle themselves. Resolves immediately; permission status will
   *  be re-checked automatically when the app comes back to the foreground. */
  openSettings: () => Promise<void>;
  /** Re-read the current OS-level status (e.g. after returning from Settings). */
  refresh: () => Promise<MicPermissionStatus>;
}

function classify(resp: {
  granted: boolean;
  canAskAgain: boolean;
  status: string;
}): MicPermissionStatus {
  if (resp.granted) return "granted";
  // status is one of "granted" | "denied" | "undetermined" in expo-av v15
  if (resp.status === "undetermined") return "undetermined";
  return resp.canAskAgain ? "denied" : "blocked";
}

export function useMicPermission(): MicPermission {
  const [status, setStatus] = useState<MicPermissionStatus>("unknown");
  // Track whether we've already attempted a request in this session — we use
  // this together with AppState to re-check after the user comes back from
  // the Settings app.
  const requestedRef = useRef(false);

  const refresh = useCallback(async (): Promise<MicPermissionStatus> => {
    try {
      const resp = await Audio.getPermissionsAsync();
      const next = classify(resp);
      setStatus(next);
      return next;
    } catch (e) {
      console.warn("[mic] getPermissionsAsync failed:", e);
      // Treat failure as undetermined so the user still sees an actionable UI.
      setStatus("undetermined");
      return "undetermined";
    }
  }, []);

  const request = useCallback(async (): Promise<MicPermissionStatus> => {
    requestedRef.current = true;
    try {
      const resp = await Audio.requestPermissionsAsync();
      const next = classify(resp);
      setStatus(next);
      return next;
    } catch (e) {
      // Almost always means NSMicrophoneUsageDescription is missing — but
      // since our app.json sets it, the only way to hit this is a stale
      // build. Surface as "blocked" so the UI shows the Settings CTA.
      console.warn("[mic] requestPermissionsAsync threw:", e);
      setStatus("blocked");
      return "blocked";
    }
  }, []);

  const openSettings = useCallback(async (): Promise<void> => {
    try {
      await Linking.openSettings();
    } catch (e) {
      console.warn("[mic] openSettings failed:", e);
    }
  }, []);

  // 1) Read current status on mount.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // 2) When the app returns to foreground (e.g. user came back from
  //    Settings), re-check. This is the only way to detect a permission
  //    change without the user manually retrying.
  useEffect(() => {
    const sub = AppState.addEventListener("change", (next) => {
      if (next === "active") void refresh();
    });
    return () => sub.remove();
  }, [refresh]);

  return { status, request, openSettings, refresh };
}
