# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the macOS app: `make build` -> dist/OSC Gesture.app
#
# The build is ad-hoc signed, which is enough to run it on this machine.
# `make sign` re-signs it with a Developer ID and `make notarize` gets Apple to
# notarize it for sharing -- see "Packaging" in README.md.
import importlib.util
import os

# classes/mac_app.py runs the .app under AppKit. Without PyObjC in the build
# environment the bundle would still build, then die on launch.
if importlib.util.find_spec("AppKit") is None:
    raise SystemExit("PyObjC is missing: pip install -r requirements-build.txt")

APP_NAME = "OSC Gesture"
BUNDLE_ID = "edu.mit.media.osc-gesture"
VERSION = "1.0.0"

datas = [
    ("hand_landmarker.task", "."),
    ("ui.html", "."),
]
# main.py falls back to an orchestra.json beside the app (sys._MEIPASS when
# frozen), so ship one if it's there.
if os.path.exists("orchestra.json"):
    datas.append(("orchestra.json", "."))

a = Analysis(
    ["main.py"],
    # The empty __init__.py at the project root makes PyInstaller search from
    # the parent directory, where `classes` isn't importable.
    pathex=[SPECPATH],
    datas=datas,
    # mediapipe declares these but nothing the app uses imports them at
    # runtime; left in they'd add ~330 MB (jaxlib alone is 240 MB).
    excludes=["jax", "jaxlib", "scipy", "sentencepiece", "ml_dtypes",
              "opt_einsum", "tkinter"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,
    # mediapipe ships universal2; thinning to arm64 drops the unused x86_64
    # slices. The pyenv/uv Pythons this is built with are arm64-only anyway.
    target_arch="arm64",
    # UPX and strip both invalidate code signatures on macOS.
    strip=False,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name=APP_NAME,
    strip=False,
    upx=False,
)
app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon="icon.icns" if os.path.exists("icon.icns") else None,
    bundle_identifier=BUNDLE_ID,
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": APP_NAME,
        "CFBundleVersion": VERSION,
        # mediapipe's native bindings are built for macOS 14.5; on anything
        # older, Finder refuses to open the app with a clear message rather
        # than letting it crash.
        "LSMinimumSystemVersion": "14.5",
        "LSApplicationCategoryType": "public.app-category.music",
        "NSHighResolutionCapable": True,
        # Without this the system kills the app the moment it opens the camera.
        "NSCameraUsageDescription":
            "OSC Gesture tracks your hands with the camera and turns their "
            "position and gestures into OSC messages.",
        # Shown when --host points at another machine, or the web UI is
        # served to a phone or tablet with --http-host 0.0.0.0.
        "NSLocalNetworkUsageDescription":
            "OSC Gesture sends OSC messages to music software on your local "
            "network, and can serve its controls to a phone or tablet.",
    },
)
