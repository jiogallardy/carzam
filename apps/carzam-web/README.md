# carzam-web

Static marketing site for Carzam. No build step. Plain HTML + CSS + SVG.

```
apps/carzam-web/
├── index.html          ← landing page
├── privacy.html        ← privacy policy (linked from footer)
├── terms.html          ← terms of service (linked from footer)
├── assets/
│   ├── icon.svg        ← master icon (sphere + 3/4 car silhouette + sound wave)
│   ├── icon-1024.png   ← App Store master (1024×1024)
│   ├── icon-512.png
│   ├── icon-192.png    ← used on the site as the brand mark
│   ├── favicon.png
└── serve.sh            ← `./serve.sh` to run locally on :5173
```

## Local dev

```bash
cd apps/carzam-web
./serve.sh
```

Open http://localhost:5173.

## Re-render the icon

After editing `assets/icon.svg`:

```bash
cd assets
rsvg-convert -w 1024 -h 1024 icon.svg -o icon-1024.png
rsvg-convert -w 512  -h 512  icon.svg -o icon-512.png
rsvg-convert -w 192  -h 192  icon.svg -o icon-192.png
rsvg-convert -w 64   -h 64   icon.svg -o favicon.png
```

## Deploy

For now, deploy as a static container in Coolify next to the API:

1. Coolify Project Carzam → + New Resource → Static
2. Repo: `jiogallardy/carzam`, branch `main`, base directory `apps/carzam-web`
3. Build pack: **Static**
4. Publish directory: `.` (relative to base)
5. Domain: e.g. `carzam.app` or `www.carzam.app`

Or push to GitHub Pages, Cloudflare Pages, or Vercel — anywhere that serves static files.
