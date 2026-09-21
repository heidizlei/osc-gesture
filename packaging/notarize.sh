#!/usr/bin/env bash
# Notarize a signed build, staple the ticket, and zip it for sharing.
#
#   packaging/notarize.sh ["dist/OSC Gesture.app"]
#
# NOTARY_PROFILE names the notarytool keychain profile (default: osc-gesture).
# Create it once with:
#
#   xcrun notarytool store-credentials osc-gesture \
#       --apple-id <you@example.com> --team-id <TEAMID> --password <app-specific password>
set -euo pipefail

APP="${1:-dist/OSC Gesture.app}"
PROFILE="${NOTARY_PROFILE:-osc-gesture}"

SIGNATURE="$(codesign --display --verbose=2 "$APP" 2>&1 || true)"
if [[ "$SIGNATURE" != *"Authority=Developer ID Application"* ]]; then
    echo "$APP isn't signed with a Developer ID; run 'make sign' first." >&2
    exit 1
fi

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
NAME="$(basename "$APP" .app)"
ZIP="$(dirname "$APP")/${NAME// /-}-$VERSION-macOS-arm64.zip"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# notarytool takes a zip, and ditto (unlike zip) preserves the bundle's
# symlinks and extended attributes, which the signature covers.
ditto -c -k --keepParent "$APP" "$WORK/upload.zip"

echo "Submitting to Apple's notary service (usually a few minutes)..."
# Judged by the reported status rather than the exit code, so a rejection
# still gets as far as printing Apple's log below.
xcrun notarytool submit "$WORK/upload.zip" --keychain-profile "$PROFILE" \
    --wait --output-format plist > "$WORK/result.plist" || true
# Empty when absent. plutil reports errors on stdout, so go by its exit status.
field() {
    local value
    if value="$(plutil -extract "$1" raw -o - "$WORK/result.plist" 2>/dev/null)"; then
        echo "$value"
    fi
}
STATUS="$(field status)"
if [[ "$STATUS" != "Accepted" ]]; then
    echo "Notarization failed (status: ${STATUS:-none})." >&2
    ID="$(field id)"
    if [[ -n "$ID" ]]; then
        echo "Apple's log:" >&2
        xcrun notarytool log "$ID" --keychain-profile "$PROFILE" >&2
    else
        cat "$WORK/result.plist" >&2
    fi
    exit 1
fi

# Stapling attaches the ticket to the app, so Gatekeeper accepts it offline.
xcrun stapler staple "$APP"
spctl --assess --type execute --verbose=2 "$APP"

# The zip to share: made after stapling so the ticket travels with it.
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "Ready to share: $ZIP"
