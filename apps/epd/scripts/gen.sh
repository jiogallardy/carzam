#!/bin/bash
# Generate a cartoon image via Gemini and display on the e-paper.
# Usage:
#   ./gen.sh "<subject>" [<title>] [--force] [--setting "..."]
# Examples:
#   ./gen.sh "orange Porsche 911 GT3RS" "GT3RS LEGACY"
#   ./gen.sh "yellow Lamborghini Huracan" --force
set -euo pipefail

HOST="${EPD_HOST:-192.168.1.242:8765}"
SUBJECT="${1:?usage: gen.sh \"<subject>\" [<title>] [--force] [--setting \"...\"]}"
shift || true

TITLE=""
SETTING="european cobblestone"
FORCE="False"

while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE="True"; shift;;
    --setting) SETTING="$2"; shift 2;;
    *)
      if [ -z "$TITLE" ]; then
        TITLE="$1"; shift
      else
        echo "unknown arg: $1" >&2; exit 1
      fi
      ;;
  esac
done

body=$(python3 -c "
import json,sys
print(json.dumps({
  'subject': '''$SUBJECT''',
  'title': '''$TITLE''' or None,
  'setting': '''$SETTING''',
  'force': $FORCE,
}))
")

echo "POST http://${HOST}/prompt"
echo "  subject: $SUBJECT"
echo "  title:   ${TITLE:-<auto from subject>}"
echo "  setting: $SETTING"
echo "  (takes ~30s: 5-10s generate + 20s refresh)"
curl -m 180 -X POST "http://${HOST}/prompt" \
  -H 'Content-Type: application/json' \
  -d "$body" -w "\n"
