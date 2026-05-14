#!/usr/bin/env bash
# Local dev server for carzam-web. No build step; static files only.
set -euo pipefail
PORT=${PORT:-5173}
cd "$(dirname "$0")"
echo "→ Carzam web serving on http://localhost:${PORT}"
echo "→ Open this URL in your browser. Ctrl+C to stop."
exec python3 -m http.server "$PORT"
