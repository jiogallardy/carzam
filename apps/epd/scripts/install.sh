#!/bin/bash
# Install epd on the Libre board. Run from inside ~/epd/.
set -euo pipefail

if [ ! -f .venv/bin/activate ]; then
  # nuke a half-built venv from a prior failed run
  rm -rf .venv
  # system-site-packages so we can use the apt-installed
  # python3-libgpiod, python3-spidev, python3-pil, python3-fastapi
  python3 -m venv --system-site-packages .venv
fi

. .venv/bin/activate
pip install --upgrade pip
pip install -e .

echo "epd installed. Run: epd show <path>  or  epd serve"
