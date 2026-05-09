# carzam-mobile

React Native (Expo SDK 52) app for Carzam. Targets iOS first (Android works but unconfigured).

## Prereqs

- **Node 20+ + pnpm** (`brew install pnpm`)
- **Apple Developer account** (for Sign in with Apple + on-device install). Free tier is fine for sideloading dev builds.
- **Google Cloud project** with an iOS OAuth client. The bundle ID below must match.
- **EAS account** (free): `pnpm dlx eas-cli login` then `eas init` from this dir.

## First-time setup

```bash
cd apps/carzam-mobile
pnpm install

# 1. Apple Sign-In
#    In your Apple Developer account → Identifiers → register an App ID
#    "com.carzam.app" with the "Sign In with Apple" capability enabled.
#    The free tier won't let you ship Apple Sign-In to TestFlight, but it will
#    work for on-device dev builds via your developer team.

# 2. Google Sign-In
#    Google Cloud Console → APIs & Services → Credentials →
#    Create Credentials → OAuth client ID → iOS → Bundle ID = com.carzam.app
#    Copy the iOS client ID and reversed client ID.

# 3. Fill the env file
cp .env.example .env
#    Then put EXPO_PUBLIC_API_URL=https://api.carzam.app (or wherever your
#    Coolify-hosted backend lives) and EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID.

# 4. Update app.json
#    Replace `REPLACE_WITH_REVERSED_CLIENT_ID` with your reversed Google client ID
#    (looks like com.googleusercontent.apps.000-xxx).
#    `eas init` will fill in the EAS projectId for you.

# 5. EAS init (first time only)
pnpm dlx eas-cli login
pnpm dlx eas-cli init
```

## Build a dev client and install it on your phone

The default Expo Go *can't* run this app — `expo-apple-authentication` and
`@react-native-google-signin/google-signin` aren't bundled in Expo Go. You
need a custom dev client. EAS builds it in the cloud:

```bash
pnpm build:dev:ios
# (= eas build --profile development --platform ios)
```

When it finishes (~10 minutes the first time), EAS gives you a link. Open it
on your iPhone in Safari → tap the install button → trust the developer cert
in **Settings → General → VPN & Device Management**.

Then, from your laptop:

```bash
pnpm start    # starts Metro on port 8081 with --dev-client
```

Open the dev client app on your phone, scan the QR, the JS loads, you're in.
JS-only changes hot-reload. Native changes require another `eas build`.

## Folder layout

```
app/                 expo-router file-based routes
  _layout.tsx        root: providers, auth gate
  (auth)/sign-in     Google + Apple buttons
  (tabs)/_layout     bottom nav, raised Carzam center button
  (tabs)/index       Garage (My Cars list)
  (tabs)/found       Discovered grid + completion bar
  (tabs)/stats       big completion %, totals
  (tabs)/me          profile + sign-out
  carzam.tsx         full-screen modal: record + classify + result
  add-car.tsx        modal form
  car/[id].tsx       car detail
src/
  api/               typed API client per resource
  components/        GlassCard, AmbientBackground, CarIcon, PrimaryButton
  store/             zustand auth store (token + user, persisted via SecureStore)
  theme/             colors, radii, spacing tokens
```

## What the Carzam button does

1. Tap the orange center button → opens `/carzam` as a full-screen modal.
2. Modal auto-starts a 5s recording at 16kHz mono WAV.
3. On stop, uploads multipart to `POST /recognize`.
4. Shows top-1 + top-3 + state. If first-ever match of that class →
   "NEW DISCOVERY" banner. If low-confidence → "Add as new car" CTA.
5. Closing the modal returns you to whichever tab you were on.

## Sign-out

`Me → Sign out` clears the JWT from SecureStore and bounces you back to
`/sign-in`. The same path runs automatically if any API call returns 401.
