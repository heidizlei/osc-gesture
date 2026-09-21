#!/usr/bin/env bash
# Launch the built app and check that it comes up and serves its UI.
#
#   packaging/smoke_test.sh ["dist/OSC Gesture.app"]
#
# Catches what only breaks once frozen -- a module PyInstaller didn't collect,
# a data file missing from the spec, something the hardened runtime blocks --
# before the build is notarized and shared. The camera switches on for a few
# seconds; OSC goes to a dead port so nothing reaches running music software.
set -euo pipefail

APP="${1:-dist/OSC Gesture.app}"
EXE="$APP/Contents/MacOS/$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$APP/Contents/Info.plist")"
# A free port, so the check can't be answered by a copy that's already running.
PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
LOG="$(mktemp)"

"$EXE" --no-browser --http-port "$PORT" --port 9 > "$LOG" 2>&1 &
PID=$!
# SIGKILL: if the app is showing an error alert, it won't act on anything gentler.
trap 'kill -9 "$PID" 2>/dev/null || true; rm -f "$LOG"' EXIT

for _ in $(seq 60); do
    STATE="$(curl -sf -m 1 "http://127.0.0.1:$PORT/state" || true)"
    if [[ "$STATE" == *'"presets"'* ]]; then
        kill -INT "$PID"
        wait "$PID" || true
        echo "Smoke test passed: the app started and served its UI."
        exit 0
    fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done

echo "Smoke test failed: the app didn't come up. Its output:" >&2
cat "$LOG" >&2
exit 1
