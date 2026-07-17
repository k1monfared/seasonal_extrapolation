#!/usr/bin/env bash
# Interactive demo: serve the page on a free local port and open your browser.
set -eu

# Resolve this script's own directory and work from there.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Locate the directory that contains index.html (usually docs).
if [ -f "$SCRIPT_DIR/docs/index.html" ]; then
  DOCROOT="$SCRIPT_DIR/docs"
elif [ -f "$SCRIPT_DIR/index.html" ]; then
  DOCROOT="$SCRIPT_DIR"
else
  echo "No index.html found under $SCRIPT_DIR" >&2
  exit 1
fi

# Pick a free TCP port automatically so this never fails if a port is in use.
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
URL="http://127.0.0.1:${PORT}/index.html"

# Serve the docroot in the background.
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$DOCROOT" &
SERVER_PID=$!

# Stop the server when this script exits.
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Give the server a moment to come up.
sleep 1

echo "Serving $DOCROOT"
echo "Open: $URL"

# Open the browser, trying several openers, each guarded so a missing one is fine.
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "$URL" >/dev/null 2>&1 || true
else
  python3 -m webbrowser "$URL" >/dev/null 2>&1 || true
fi

echo "Press Ctrl-C to stop."
# Keep the server in the foreground until Ctrl-C.
wait "$SERVER_PID"
