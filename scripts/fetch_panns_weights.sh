#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
URL="https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
DEST="weights/Cnn14_mAP=0.431.pth"
if [ -f "$DEST" ]; then
  echo "weights already present at $DEST"
  exit 0
fi
echo "downloading PANNs CNN14 weights (~80MB)..."
curl -L --fail -o "$DEST" "$URL"
echo "saved to $DEST"
