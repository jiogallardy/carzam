# Carzam — ship it

This is the cookbook for taking the code in this repo and getting:
1. The backend running on your Hetzner box (`carzam-server`, `5.78.203.232`).
2. A signed iOS dev-client `.ipa` installed on your iPhone.

If something doesn't match here, the code is the source of truth — this doc gets stale.

---

## Part 1 — Deploy the backend

### 1.1 Cloudflare R2

1. Cloudflare dashboard → **R2** → enable (will require a card).
2. **Create bucket** named `carzam-clips`. Default settings.
3. **Manage R2 API Tokens** → **Create API token**:
   - Permission: **Object Read & Write**
   - Bucket: `carzam-clips`
   - TTL: forever
4. Copy the **Access Key ID**, **Secret Access Key**, and your **Account ID** (the dashboard URL has it: `https://dash.cloudflare.com/<account-id>/r2`).

### 1.2 Apple Sign-In setup

1. https://developer.apple.com → Identifiers → **+** → App IDs → App
2. Bundle ID: **`com.carzam.app`** (Explicit). Description: `Carzam`.
3. Capabilities: enable **Sign In with Apple**.
4. Note your **Team ID** (top right of the developer page).

### 1.3 Google Sign-In setup

1. https://console.cloud.google.com → create or pick a project.
2. **APIs & Services → Credentials → + Create Credentials → OAuth client ID**
3. Application type: **iOS**
4. Bundle ID: **`com.carzam.app`**
5. Copy the **Client ID** (looks like `0000-xxxx.apps.googleusercontent.com`) and the **Reversed Client ID**.

### 1.4 Push the repo to GitHub

```bash
# from monorepo root (only if you haven't already)
gh repo create carzam --source=. --push --private
```

### 1.5 Coolify wiring

Coolify dashboard → `carzam-server` is already added (you did this earlier). Now:

1. **+ New Project** → name `Carzam` → Server: `carzam-server`.
2. **+ New Resource → PostgreSQL 16** in the project. Wait for it to come up — Coolify exposes its hostname/port/credentials as `DATABASE_URL`.
3. **+ New Resource → Application → Public Repository** (or **GitHub App** if you've connected GitHub).
   - Repo URL: your `carzam` repo.
   - Branch: `main`
   - **Build Pack: Dockerfile**
   - **Dockerfile Path:** `apps/carzam-api/Dockerfile`
   - **Base directory:** `/`
4. **Environment Variables** — paste these (substituting real values):
   ```
   DATABASE_URL=<the postgresql URL Coolify gave you, but change postgresql:// → postgresql+asyncpg://>
   JWT_SECRET=<openssl rand -hex 64>
   JWT_TTL_DAYS=30
   GOOGLE_CLIENT_ID_IOS=<from 1.3>
   APPLE_TEAM_ID=<from 1.2>
   APPLE_BUNDLE_ID=com.carzam.app
   R2_ACCOUNT_ID=<from 1.1>
   R2_ACCESS_KEY_ID=<from 1.1>
   R2_SECRET_ACCESS_KEY=<from 1.1>
   R2_BUCKET=carzam-clips
   CHECKPOINT_PATH=/data/checkpoint.pt
   CLASSES_PATH=/data/classes.json
   ```
5. **Networking → Add Domain:** `api.carzam.app` (or whatever subdomain you want). Point that DNS A record at `5.78.203.232` first. Coolify will issue Let's Encrypt automatically.
6. **Deploy.**
7. Verify: `curl https://api.carzam.app/health` should return `{"status":"ok","model_run_id":"20260507_174723"}`.

If the build OOMs on the CX21 (2GB RAM is tight), rescale the box to CX22 (4GB) — Hetzner does this online in ~30 seconds.

---

## Part 2 — Build + flash the iOS app

### 2.1 First-time prerequisites on your Mac

```bash
brew install pnpm
xcode-select --install      # if not installed
sudo xcodebuild -license accept
```

You also need:
- An **Apple ID** signed into Xcode (Xcode → Settings → Accounts → +).
- An **Expo (EAS) account**: https://expo.dev/signup.

### 2.2 Configure the mobile app

```bash
cd apps/carzam-mobile

cp .env.example .env
# Edit .env:
#   EXPO_PUBLIC_API_URL=https://api.carzam.app
#   EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID=<from 1.3>
```

Open `app.json` and replace `REPLACE_WITH_REVERSED_CLIENT_ID` with your reversed Google client ID (from step 1.3 — looks like `com.googleusercontent.apps.0000-xxxx`).

```bash
pnpm install
pnpm dlx eas-cli login
pnpm dlx eas-cli init       # creates EAS projectId, fills it into app.json
```

### 2.3 Build the dev client

```bash
pnpm build:dev:ios
# = eas build --profile development --platform ios
```

EAS will:
- Ask which Apple Developer team to use → pick yours.
- Generate / pick a provisioning profile and signing cert (you can let EAS handle it).
- Build in the cloud (~10–15 min the first time).

When it finishes, EAS prints a URL that opens an installation page. Open it on your iPhone in Safari, tap **Install**.

After install: **iPhone Settings → General → VPN & Device Management → trust the developer**.

### 2.4 Flash + run

```bash
# From your Mac, in apps/carzam-mobile:
pnpm start
```

Open the Carzam dev client on your iPhone (it has a generic Expo icon) → it'll connect to the Metro server on your Mac automatically (same WiFi). The JS bundle loads and you're in.

### 2.5 First-run flow

1. Sign-in screen → tap **Sign in with Apple** or **Sign in with Google**.
2. You land on **Garage**, empty.
3. Tap **+ Add** → enter `2018 / BMW / M3` → Save. You're back on the list.
4. Tap the orange **Carzam** button in the center of the bottom nav.
5. The modal pops, auto-records 5s, shows the result with top-3 candidates.
6. Switch to **Found** to see the discovery you just unlocked.
7. **Stats** to see your completion %.
8. **Me → Sign out** when done.

### 2.6 Iterating

- **JS-only changes** (any `.tsx` / `.ts`): just save the file. Metro reloads.
- **app.json / native module changes**: re-run `pnpm build:dev:ios` and re-install on the phone.
- **Backend changes**: push to your repo. Coolify rebuilds & redeploys on push if you enable the GitHub webhook in the application settings.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `pnpm install` fails on `expo-modules-core` | Node version | Use Node 20 LTS (`nvm use 20`) |
| `eas build` says "no credentials" | Apple Developer team not linked | EAS prompt → log into Apple in the prompt |
| Sign-in works but `/auth/google` returns 401 | Google client ID mismatch | The `iosClientId` in `app.json` plugin must equal `GOOGLE_CLIENT_ID_IOS` env var on the API |
| `/recognize` returns 422 "clip too short" | iOS recorded < 2s | Make sure mic permission is granted, and the user holds the button > 2s (we auto-stop at 5s) |
| API takes 60+ sec to start | Model loading at startup | Normal. CPU loads CNN14 weights once; subsequent inferences are ~150ms |
| API container OOMs | CX21 RAM tight | Coolify → server → Rescale → CX22 (4GB) |
