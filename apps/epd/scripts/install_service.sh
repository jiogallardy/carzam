#!/bin/bash
# Install/refresh the epd systemd service. Run from ~/epd on the target board.
set -euo pipefail

SERVICE_SRC="$(cd "$(dirname "$0")" && pwd)/epd.service"
SERVICE_DST=/etc/systemd/system/epd.service

echo "==> writing $SERVICE_DST"
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo systemctl daemon-reload
sudo systemctl enable epd.service
sudo systemctl restart epd.service
sleep 2
sudo systemctl status epd.service --no-pager | head -20
