# Creator content: engine-sound waveforms from YouTube

How to grab a car's exhaust note straight from a YouTube video (or Short) and turn
it into a cool-looking waveform — either a **static PNG** or an **animated MP4**
(waveform that moves with the audio). This is the recipe used for the GT3 RS / GT4 RS
clips and is meant for making short-form creator content (Reels / TikTok / Shorts),
not for the training pipeline.

> First done 2026-05-21. Tools: `yt-dlp` (run via `uv`, no install needed) + `ffmpeg`.

---

## Prerequisites

- `ffmpeg` on PATH (`which ffmpeg`). Install with `brew install ffmpeg`.
- `yt-dlp` — **no install required**, run it through `uv` so you always get the latest
  (YouTube breaks old yt-dlp versions constantly):

  ```bash
  uv run --with yt-dlp yt-dlp --version
  ```

---

## Step 1 — Pull the audio from YouTube as WAV

**First N seconds only** (e.g. first 15s — `ffmpeg:-t 15`):

```bash
uv run --with yt-dlp yt-dlp \
  -x --audio-format wav \
  --postprocessor-args "ffmpeg:-t 15" \
  -o "$HOME/Downloads/clip_first15s.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Full video audio** (drop the `--postprocessor-args` line):

```bash
uv run --with yt-dlp yt-dlp \
  -x --audio-format wav \
  -o "$HOME/Downloads/clip_full.%(ext)s" \
  "https://www.youtube.com/watch?v=VIDEO_ID"
```

Sanity-check what you got:

```bash
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \
  "$HOME/Downloads/clip_full.wav"
```

### If a YouTube **Short** fails to download

Shorts (and sometimes regular videos) hit bot/SABR checks. The fix that worked was
adding `--remote-components ejs:github` so yt-dlp fetches the current JS player solver:

```bash
uv run --with yt-dlp yt-dlp --remote-components ejs:github \
  -x --audio-format wav \
  --postprocessor-args "ffmpeg:-t 15" \
  -o "$HOME/Downloads/short_first15s.%(ext)s" \
  "https://www.youtube.com/shorts/SHORT_ID"
```

Other things tried (less reliable, keep as fallbacks):
`--extractor-args "youtube:player_client=web,ios,mweb"` or `=tv,tv_embedded,android_vr`.

Quick way to confirm which URL is which (title/uploader without downloading):

```bash
uv run --with yt-dlp yt-dlp --remote-components ejs:github --skip-download \
  --print "%(id)s | %(title)s | %(uploader)s" "URL_1" "URL_2"
```

---

## Step 2 — Static PNG waveform (whole clip on one image)

`showwavespic` draws the entire waveform as a single image:

```bash
ffmpeg -y -i "$HOME/Downloads/clip_full.wav" \
  -filter_complex "showwavespic=s=1600x300:colors=white:split_channels=0" \
  -frames:v 1 \
  "$HOME/Downloads/clip_waveform.png"
```

- `s=1600x300` — output size (use `1200x240` for a smaller bar).
- `colors=white` — bar color (accepts CSS color names / hex; comma-separate for stereo).
- `split_channels=0` — collapse L/R into one waveform; `1` stacks them.

---

## Step 3 — Animated waveform MP4 (the one that actually moves)

This is what "I want the animated waveform, not a still PNG" means. `showwaves`
renders a video synced to the audio, and we mux the original audio back in:

```bash
ffmpeg -y -i "$HOME/Downloads/clip_full.wav" \
  -filter_complex "[0:a]showwaves=s=1280x300:mode=cline:colors=white:rate=30[v]" \
  -map "[v]" -map 0:a \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
  "$HOME/Downloads/clip_waveform.mp4"
```

- `mode=cline` — center line that pulses out (good clean look). Other modes:
  `point`, `line`, `p2p`, `cline`.
- `rate=30` — output FPS.
- `-map "[v]" -map 0:a` — keep the generated video **and** the original audio.
- `-pix_fmt yuv420p` — required for the MP4 to play everywhere (QuickTime / iOS / social).
- `-shortest` — stop when the audio ends.

### Look variations to try
- Filled bars instead of a line: swap `showwaves` for a `showcqt` or use
  `mode=line` with a thicker `colors`.
- Brand color: `colors=0xFF2D55` (hex, `0x` prefix) or `colors=red`.
- Vertical/story format: `s=720x1280`.
- Transparent background for compositing: render to `.mov` with
  `-c:v qtrle -pix_fmt argb` instead of libx264/yuv420p.

---

## Reference clips (what was made on 2026-05-21)

| Clip            | Source                                            | Output |
|-----------------|---------------------------------------------------|--------|
| GT4 RS (`gt4rs`) | `youtube.com/watch?v=nq1A2Gp2Qy4`                | `gt4rs_waveform.png` / `.mp4` |
| GT3 RS (`gt3rs`) | `youtube.com/shorts/9YcRJfv1DQM`                 | `gt3rs_waveform.png` / `.mp4` |

(The Short — `9YcRJfv1DQM` — is the one that needed `--remote-components ejs:github`.)

---

## One-shot helper

```bash
# usage: make_waveform.sh "<youtube-url>" <name> [seconds]
URL="$1"; NAME="$2"; SECS="${3:-}"
DL="$HOME/Downloads"
PP=""; [ -n "$SECS" ] && PP=(--postprocessor-args "ffmpeg:-t $SECS")
uv run --with yt-dlp yt-dlp --remote-components ejs:github \
  -x --audio-format wav "${PP[@]}" -o "$DL/${NAME}.%(ext)s" "$URL"
ffmpeg -y -i "$DL/${NAME}.wav" \
  -filter_complex "showwavespic=s=1600x300:colors=white" -frames:v 1 "$DL/${NAME}_wave.png"
ffmpeg -y -i "$DL/${NAME}.wav" \
  -filter_complex "[0:a]showwaves=s=1280x300:mode=cline:colors=white:rate=30[v]" \
  -map "[v]" -map 0:a -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$DL/${NAME}_wave.mp4"
echo "→ $DL/${NAME}_wave.png and $DL/${NAME}_wave.mp4"
```
