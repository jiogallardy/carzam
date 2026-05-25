#!/bin/bash
# Push the epd project to the Libre board over SSH and install it.
# Usage: ./scripts/deploy.sh [HOST]   (default jiogallardy@192.168.50.239)
set -euo pipefail

HOST="${1:-jiogallardy@192.168.50.239}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> rsync project to $HOST:~/epd"
sshpass -p 'jiogallardy' rsync -avz --delete \
  --exclude '__pycache__' --exclude '.venv' --exclude '*.egg-info' \
  -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  "$PROJECT_DIR/" "$HOST:~/epd/"

echo "==> install on remote"
sshpass -p 'jiogallardy' ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$HOST" '
  cd ~/epd && bash scripts/install.sh
'
