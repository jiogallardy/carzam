#!/bin/bash
# Push an image to the e-paper from your laptop.
# Usage: ./push.sh <image-path> [--force] [--fit cover|letterbox] [--rotate 0|90|180|270]
set -euo pipefail

HOST="${EPD_HOST:-192.168.1.242:8765}"
IMG="${1:?usage: push.sh <image> [--force] [--fit ...] [--rotate ...]}"
shift || true

QS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --force) QS="${QS}&force=1"; shift;;
    --fit) QS="${QS}&fit=$2"; shift 2;;
    --rotate) QS="${QS}&rotate=$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

URL="http://${HOST}/image?$(echo "${QS}" | sed 's/^&//')"
echo "POST $URL"
curl -m 120 -X POST "$URL" -F "file=@${IMG}" -w "\n"
