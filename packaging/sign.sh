#!/usr/bin/env bash
# Re-sign the PyInstaller build with a Developer ID, ready for notarization.
#
#   packaging/sign.sh ["dist/OSC Gesture.app"]
#
# CODESIGN_IDENTITY picks the certificate. The default, "Developer ID
# Application", is matched by codesign as a substring, so it works as long as
# the keychain holds exactly one such certificate; otherwise pass the full
# name or its SHA-1 from `security find-identity -v -p codesigning`.
set -euo pipefail

APP="${1:-dist/OSC Gesture.app}"
IDENTITY="${CODESIGN_IDENTITY:-Developer ID Application}"
ENTITLEMENTS="$(dirname "$0")/entitlements.plist"

if [[ ! -d "$APP" ]]; then
    echo "No app at $APP; run 'make build' first." >&2
    exit 1
fi

# Notarization requires the hardened runtime and a secure timestamp on every
# executable in the bundle.
SIGN=(codesign --force --timestamp --options runtime --sign "$IDENTITY")

# Inside out: codesign won't seal a bundle around unsigned nested code, and
# Apple advises against --deep for signing. So every Mach-O file first, except
# the main executable, which is signed as part of the bundle below.
MAIN_EXE="$APP/Contents/MacOS/$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$APP/Contents/Info.plist")"
NESTED=()
while IFS= read -r -d '' f; do
    [[ "$f" == "$MAIN_EXE" ]] && continue
    [[ "$(file -b "$f")" == Mach-O* ]] && NESTED+=("$f")
done < <(find "$APP/Contents" -type f -print0)

echo "Signing ${#NESTED[@]} nested binaries as \"$IDENTITY\"..."
printf '%s\0' "${NESTED[@]}" | xargs -0 -n 50 "${SIGN[@]}"

echo "Signing the app bundle..."
"${SIGN[@]}" --entitlements "$ENTITLEMENTS" "$APP"

codesign --verify --deep --strict --verbose "$APP"
codesign --display --verbose=2 "$APP" 2>&1 | grep -E '^(Authority|TeamIdentifier|Timestamp|CodeDirectory)'
