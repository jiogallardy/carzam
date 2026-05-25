#!/usr/bin/env bash
# Classify a YouTube clip with the local v11 model (or any other run dir).
#
# Usage:
#   scripts/classify-yt.sh <youtube-url> [--run-dir runs/<ts>] [--start 30 --duration 15]
#
# Examples:
#   scripts/classify-yt.sh "https://youtu.be/-vML9W60Bu0"
#   scripts/classify-yt.sh "https://youtu.be/XYZ" --start 60 --duration 10
#   scripts/classify-yt.sh "https://youtu.be/XYZ" --run-dir runs/20260507_171602
#
# Defaults to v11 (the contrastive 72-class model with embedding head).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <youtube-url> [--run-dir DIR] [--start SECONDS --duration SECONDS]" >&2
    exit 1
fi

URL="$1"; shift
RUN_DIR="runs/20260515_152336"   # v11 — change with --run-dir if you want v4 etc.
START=""
DURATION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --start)   START="$2";   shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
cd "$REPO_ROOT"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "run dir not found: $RUN_DIR" >&2
    exit 1
fi

WAV="$(mktemp -t carzam-yt-XXXX).wav"
trap 'rm -f "$WAV"' EXIT

echo "[1/2] downloading audio from $URL ..."
.venv/bin/yt-dlp \
    -f bestaudio \
    -x --audio-format wav \
    --postprocessor-args "-ac 1 -ar 16000" \
    --no-warnings --quiet \
    -o "$WAV" \
    "$URL"

# Optional time-slice with ffmpeg
if [[ -n "$START" || -n "$DURATION" ]]; then
    SLICE="${WAV%.wav}.slice.wav"
    FFMPEG_ARGS=()
    [[ -n "$START" ]]    && FFMPEG_ARGS+=(-ss "$START")
    [[ -n "$DURATION" ]] && FFMPEG_ARGS+=(-t "$DURATION")
    ffmpeg -y -loglevel error "${FFMPEG_ARGS[@]}" -i "$WAV" "$SLICE"
    mv "$SLICE" "$WAV"
fi

echo "[2/2] classifying with $RUN_DIR ..."
.venv/bin/carzam compare "$WAV" --run-dir "$RUN_DIR"
