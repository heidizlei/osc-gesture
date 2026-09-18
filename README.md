# osc-gesture

Camera-based hand gesture controller that sends OSC messages to control tempo,
pitch range, and playback style (runs/chords).

## Setup

```bash
pip install -r requirements.txt
```

Requires `hand_landmarker.task` (MediaPipe hand landmark model) in the project
root — already included.

## Main controller (`main.py`)

```bash
python main.py --host <ip> --port <port>
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | OSC target host |
| `--port` | `9001` | OSC target port |
| `--ui` | `web` | `web` for the browser UI, `cv2` for the legacy OpenCV window |
| `--http-host` | `127.0.0.1` | Web UI bind address; `0.0.0.0` to reach it from a phone/tablet |
| `--http-port` | `8765` | Web UI port |
| `--no-browser` | off | Don't open a browser window on startup |
| `--orchestra-preset` | — | Orchestra preset JSON to read zone reset ranges from |

### Web UI (default)

On startup the app serves `ui.html` at `http://127.0.0.1:8765` and opens it in
your browser. The main row is three columns:

| Column | Contents |
|---|---|
| Left | Preset buttons (`1`–`5`), orchestra (`o`) and piano-only (`p`) toggles, landmark/debug toggles |
| Centre | Camera preview with the camera picker, draggable piano split and exclusion boundary |
| Right | Gesture readout and range bars on top, live OSC log below |

The debug-score panel sits full-width underneath. On a narrow screen the three
columns stack.

### Camera selection

A picker floats over the top-left of the preview. It lists the capture devices
found at startup — by name on macOS, as `Camera 0`, `Camera 1`… elsewhere —
and picking one switches tracking to it. The swap happens on the gesture
loop's next frame, and the new device is opened before the old one is let go,
so a camera that won't open (unplugged, or held by another app) leaves the
preview running and reports the failure next to the picker instead.

Finding cameras means opening every device in turn, so the list is built once
and cached. Press **⟳** to rescan after plugging one in. The picker is greyed
out when only one camera was found; rescanning re-enables it. The legacy
OpenCV window (`--ui cv2`) has no equivalent and always uses device 0.

### Range bars

Under the gesture readout, one bar per instrument shows where its pitch range
currently sits, on a fixed axis spanning the whole reachable range (ticks and
numbers are MIDI note numbers, one per octave). Each bar takes its
instrument's colour from the OSC log, so a bar and the messages that moved it
read as the same thing.

The **window** slider above the bars sets how wide each instrument's output
range is, from 1 semitone to 3 octaves (36), starting at 16 — the width the
app has always used. The window stays centred on the pitch the hand maps to;
an odd width puts the extra semitone above. Every range is clamped to valid
MIDI, so a wide window near either end of the mapping flattens against 0 or
127 rather than running past it. The OpenCV window has the same control as a
**Range window st** trackbar.

| Bar | Meaning |
|---|---|
| Solid | A hand is in that zone right now, driving the range |
| Thin dim line | No hand: the instrument is parked at the preset default it was reset to |
| Row dimmed, name struck through | Not in the active list — piano-only excludes it, so it can't sound |

The piano draws two bars when two hands are on it, one when a single hand is
(the second slot goes out as `-1 -1`, so there's nothing to draw). Outside
orchestra mode only the piano row exists. Hovering a row gives the exact
numbers, and the text line underneath keeps the piano's `L`/`R` readout.

The gesture readout above it is hidden in the `range` and `pedal-only`
presets, which don't act on gestures at all — the detector still runs, but
nothing downstream reads it, and showing its output would suggest otherwise.
The same applies to the OpenCV window's gesture HUD. The debug-score panel is
an explicit opt-in, so it still shows whatever you toggle it on for.

### OSC log

The bottom of the right column mirrors the `OSC →` lines the terminal prints,
with millisecond timestamps. Each message type gets its own colour, and
`/setOutputRange` is split further by which instrument it targets — piano,
brass and strings each read as a distinct colour, so a glance tells you which
zone is moving. A live update and that zone's reset share the zone's colour.
Failed sends override to red. It's fed by a wrapper around the OSC client
(`_OSCLog` in `classes/app.py`), so it captures messages from `GestureSender`
as well as the app's own — every send goes through one place. Failed sends are
logged in red with the error.

The log is capped at 40 messages, on both the server and the page, and scrolls
inside a fixed box sized to the camera preview rather than growing the page.
Each poll ships only the lines the browser hasn't seen yet. **clear** empties
it. The view auto-scrolls to the newest message unless you scroll up to read,
so scrolling back doesn't fight the feed.

The camera is still opened by Python (`cv2.VideoCapture`) and pushed to the
browser as an MJPEG stream; the page never uses `getUserMedia`. So camera
permission stays with the app (or the bundled `.app`), and there's no second
prompt from the browser.

To drive it from a phone or tablet on the same network:

```bash
python main.py --http-host 0.0.0.0
```

The startup banner prints the LAN URL to open. Note this exposes the controls
to anyone on that network — use it on a trusted one.

The browser keeps the same keyboard shortcuts as the OpenCV window: `1`–`5` for
presets, `l` for landmarks, `d` for the debug panel.

### Why the web UI is the default

Measured on an M3, per frame:

| Work | Cost |
|---|---|
| MediaPipe inference (CPU, 2 hands) | ~10.8 ms |
| `cv2.imshow` + `waitKey(1)` | ~14.7 ms |
| Overlay + HUD drawing | ~0.2 ms |
| Publishing a frame to the web UI | ~0.0002 ms |

Displaying the frame costs more than detecting hands in it, and because
capture, inference and display share one thread those costs add up: ~25.5 ms
per frame against a 30 fps camera's 33.3 ms budget. That leaves under 8 ms of
headroom, so any hitch — a `waitKey` that takes longer than usual, the debug
HUD, the periodic `gc.collect()` — overruns the budget. Frames then queue in
the capture buffer, and since the buffer never gets a chance to drain, the
latency stays. That accumulated delay is what "laggy" actually feels like; the
average frame rate can look fine while the preview trails your hands.

The web UI takes display off the gesture thread: the loop only swaps a frame
reference, and JPEG encoding runs on the server thread. Per-frame work drops to
~10.8 ms, raising headroom from ~8 ms to ~22 ms, which is what keeps the loop
from falling behind in the first place.

The drawing and buffer-copy savings elsewhere in this change are real but
minor by comparison (~0.2 ms/frame total) — removing `imshow` from the loop is
where the difference comes from.

The preview is capped at 15 fps (`PREVIEW_FPS` in `classes/web_ui.py`)
independently of the gesture loop, so a smooth-looking preview never costs
detection accuracy.

### Legacy OpenCV window

```bash
python main.py --ui cv2
```

Kept as a fallback. Control mode is selected live with number keys and shown at
the bottom of the HUD:

| Key | Preset | Behavior |
|---|---|---|
| `1` | `pedal-only` | Only hand-presence pause/resume (`/setManualPause`) |
| `2` | `range` | Pause + pitch range (`/setOutputRange`) from hand x-position — default on startup |
| `3` | `rc-slow` | Pause + range + runs/chords capped to moderate levels (baroque-style), no tempo adjust |
| `4` | `rc` | Pause + range + full-range runs/chords (`/playRuns`, `/playChords`), no tempo adjust |
| `5` | `tempo` | Everything, including gesture-driven tempo adjust (`/adjustTempo`) |

Other controls: `q` quit, `l` toggle landmark drawing, `d` toggle debug HUD,
`o` toggle orchestra mode, `p` toggle piano-only.

## Orchestra mode

Toggle it with the first checkbox under the presets, or the `o` key. It splits
the active region into three instrument zones:

```
┌───────────────────────────────────┐
│   BRASS ← left hand               │   ← top half, no divider: the zone
│           right hand → STRINGS    │      is the hand, not the side
├───────────────────────────────────┤   ← horizontal bar (draggable)
│              PIANO                │   ← bottom half
│           (4-arg form)            │
├───────────────────────────────────┤   ← exclusion boundary (draggable)
│▒▒▒▒▒▒▒ excluded region ▒▒▒▒▒▒▒▒▒▒▒│
└───────────────────────────────────┘
```

Above the piano split there's no left/right boundary: **the left hand plays
brass and the right hand plays strings**, wherever each one is. The hands can
cross over without swapping instruments, and either can reach the full pitch
range.

This uses MediaPipe's handedness classification. Its raw label is the
opposite of the hand you're actually using here — it assumes a mirrored
image and the tracker already flips the camera frame
([`tracker.py`](classes/tracker.py)), so the two mirrorings cancel — and
`_handedness` in `classes/app.py` swaps it back. That swap was settled
against the live camera, so if a future MediaPipe changes the convention,
that's the one line to flip. A hand it can't label — vanishingly rare —
falls back to the frame halves, brass left, strings right.

The amber piano split is draggable, as is the exclusion boundary. The split is
stored as a fraction of the active region, so it stays valid when you move the
exclusion boundary.

### Messages

| Zone | Message |
|---|---|
| Piano | `/setOutputRange lo hi lo2 hi2` — the original four-argument form |
| Strings | `/setOutputRange 48 lo hi -1 -1` |
| Brass | `/setOutputRange 61 lo hi -1 -1` |

The leading argument is the zone's **General MIDI program** — 48 string
ensemble, 61 brass section — read from the preset entry the zone maps to, so
pointing the app at a different preset changes the ids it sends.

Each zone's range is driven by the hands currently in it, independently of the
others. Two hands in the same zone collapse to one message at their mean x —
which above the split only happens when MediaPipe labels both the same. The
piano keeps its original behaviour: one hand drives both channels, two hands
drive left and right separately.

Hand position is the **centroid of the five finger-tip landmarks** (4, 8, 12,
16, 20). That's used for whether a hand is above the piano split, for the
pitch mapping, and for whether it's inside the active area at all — but no
longer for brass vs strings.

`x` maps across the whole frame, so brass and strings each reach the whole
pitch range rather than half of it.

### Forced instruments

Alongside the range messages, `/setForcedInstruments` carries the ids of the
**brass and strings** zones that currently hold a hand, so one of those
instruments is forced on only while a hand is in its zone:

| Occupied zones | Message |
|---|---|
| Strings | `/setForcedInstruments 48` |
| Strings + brass | `/setForcedInstruments 48 61` |
| Neither (whatever the piano zone holds) | `/setForcedInstruments` — no arguments, clearing the list |

The piano is never forced: it's the instrument playing underneath, so this
list is what a hand brings in over it. A hand entering or leaving the piano
zone therefore sends nothing.

It's edge-triggered on the same per-zone engagement the resets use: one
message when the set of occupied zones changes, none while it holds steady. A
hand crossing from brass to strings is a single message with the new list,
not a remove followed by an add. Ids come from the preset, same as the range
messages.

Orchestra mode only — outside it the brass and strings zones don't exist, so
leaving orchestra mode sends the empty list and re-entering it states the
current zones.

#### Piano-only

The **piano-only** checkbox (or the `p` key) narrows `/setActiveInstruments`
to the piano plus whichever of brass and strings a hand is in, so nothing
else can sound:

| State | `/setActiveInstruments` |
|---|---|
| Piano-only off | `1 48 61` — all three enabled |
| Piano-only, no hand in brass/strings | `1` |
| Piano-only, hand in strings | `1 48` |
| Piano-only, hands in strings + brass | `1 48 61` |

It leaves `/setForcedInstruments` alone — that stays exactly the occupied
brass/strings zones, in both modes. The active list is sent just before the
forced one, so an instrument is enabled before it's forced, and only when the
set changes.

Piano-only applies inside orchestra mode only (the checkbox is disabled
outside it), and toggling it re-sends immediately. Turning it off — or
leaving orchestra mode while it's on — re-enables all three, so the receiver
is never left narrowed. The app assumes it starts with all three enabled and
so sends nothing on that address until piano-only first narrows the set.

### Zone resets

When a zone's last hand leaves it, that zone is reset once to its instrument's
default range, so an instrument doesn't stay parked where a hand left it. The
reset is edge-triggered — it fires on the transition to empty, not on every
empty frame — and it clears that zone's send throttle so returning a hand to
the same spot re-sends immediately.

Each zone maps to a zero-based slot in an orchestra preset's
`outputInstruments` list, and both its default range and the instrument id it
sends come from that entry:

| Zone | Preset slot | Sent as | Default range |
|---|---|---|---|
| Piano | `outputInstruments[0]` — `id: 1` (grand piano) | — (4-arg form) | 26–89 |
| Strings | `outputInstruments[1]` — `id: 48` (string ensemble) | `48` | 33–94 |
| Brass | `outputInstruments[2]` — `id: 61` (brass section) | `61` | 36–92 |

(The ids and ranges shown are from the preset this was built against; the app
reads whatever your preset has, and falls back to these when an entry has no
usable `id`.) The piano's live updates use the four-argument form with no
instrument argument, so its id only ever appears in a reset — and there the
four-argument form is used too, so it never goes out at all.

Point the app at a preset with `--orchestra-preset path/to/orchestra.json`;
without it, the app looks for `orchestra.json` beside itself and otherwise
falls back to built-in values. The zone→preset-slot mapping lives in
`_REGION_INSTRUMENT` in `classes/app.py` — one dict to change if a zone should
drive a different preset instrument.

Outside orchestra mode only the piano zone exists, so only it resets; the
brass/strings resets never fire.

### Detection exclusion area

The red bottom region is masked before hand detection and excluded from
gesture, pitch-range, and hand-presence controls. A smaller percentage means a
larger excluded region (10–95%; starts at 75%).

- **Web UI:** drag the red region or its handle directly.
- **OpenCV window:** use the native **Active area %** trackbar, or drag in the
  red area if your OpenCV backend supports it.

Changes last for the current session. A packaged executable must be rebuilt to
pick up code changes.

## Recording gesture clips (`record.py`)

```bash
python record.py
```

Captures labelled hand-landmark clips into `recordings/*.npz` for later
analysis/tuning. See [recordings/README.md](recordings/README.md) for the data
format.

## Playback (`playback.py`)

```bash
python playback.py
```

Replays recorded `.npz` clips from `recordings/` through the gesture
detector/sender, useful for testing detection logic without a camera.

## Web controller (`web_controller.py`)

```bash
python web_controller.py --host <ip> --port <port> [--http-port 8080]
```

Serves `controller.html`, a browser-based manual OSC controller that forwards
button/slider input as OSC UDP messages (no camera/gesture detection). This is
separate from `main.py --ui web`, which does run the camera and gesture
detection.


## Packaging

Builds a self-contained macOS app, with Python, MediaPipe, OpenCV and the hand
model all inside, for Apple Silicon Macs on macOS 14.5 or later.

After changing the code, one command takes you to a new shareable zip:

```bash
make release
```

It runs these steps, each of which also works on its own:

| Step | What it does | Time |
|---|---|---|
| `make build` | `dist/OSC Gesture.app`, ad-hoc signed: runs on this Mac only | ~15 s |
| `make sign` | re-signs with your Developer ID, hardened runtime | ~15 s |
| `make check` | launches the signed app and checks it serves its UI (the camera blinks on) | ~5 s |
| `make notarize` | notarizes with Apple, staples the ticket, zips for sharing | a few min |

The result is `dist/OSC-Gesture-<version>-macOS-arm64.zip`, the file to share.
The ticket is stapled inside, so recipients unzip it and double-click, with no
Gatekeeper warnings, even offline. Bump `VERSION` in `osc-gesture.spec` when you
hand out a new build, so people can tell builds apart.

`make check` exists because some breakage only shows up in the frozen app: a
module PyInstaller didn't pick up, a data file the spec doesn't list, something
the hardened runtime blocks. It fails before anything is uploaded, printing the
app's output. PyInstaller follows imports on its own, so new code in `classes/`
needs nothing extra. Two kinds of change do need a step first:

- **A new pip dependency:** install it into the build environment as well as
  adding it to `requirements.txt`.
- **A new non-Python file** the app reads at runtime (like `ui.html`): add it
  to `datas` in `osc-gesture.spec`, and load it through `_resource_path()`.

While iterating, `python main.py` from source is quicker than rebuilding;
`make watch` rebuilds the app on every save if you need the bundled one.

### One-time setup

Install the build tools into the environment the app runs from, which is
`.venv` by default:

```bash
uv pip install -r requirements-build.txt   # or: pip install -r requirements-build.txt
```

Per-machine settings go in `local.mk` at the project root. It's gitignored, and
any of its settings can also be given on the command line for a single run
(`make notarize NOTARY_PROFILE=…`):

```make
# notarytool keychain profile
NOTARY_PROFILE = my-profile
# build from another environment
PYINSTALLER = /opt/miniconda3/envs/pipe/bin/pyinstaller
# only needed with several Developer ID certificates
CODESIGN_IDENTITY = Developer ID Application: … (TEAMID)
```

Signing needs a **Developer ID Application** certificate in your keychain
(`security find-identity -v -p codesigning` lists them).

Notarizing needs a notarytool keychain profile for that certificate's team.
An existing one works, even one made for another app; name it in `local.mk`.
Without a `local.mk` setting, `make notarize` looks for a profile called
`osc-gesture`. To create one, you need an Apple ID on the team and an
app-specific password for it (account.apple.com → Sign-In and Security →
App-Specific Passwords):

```bash
xcrun notarytool store-credentials osc-gesture \
    --apple-id <you@example.com> --team-id <TEAMID> --password <app-specific-password>
```

### Using the app

- Double-clicking it opens the controls in the default browser. The first
  launch asks for camera access, and the preview starts as soon as it's
  allowed. No restart is needed.
- It sits in the Dock: quit from there or with Cmd-Q, which stops the camera.
  Clicking the Dock icon, or opening the app again, re-opens the controls if the
  tab was closed.
- Flags work when you run the executable inside the bundle directly:

  ```bash
  "/Applications/OSC Gesture.app/Contents/MacOS/OSC Gesture" --host 192.168.1.20 --port 9001
  ```

- If camera access was denied, re-enable it in System Settings → Privacy &
  Security → Camera, or reset it with
  `tccutil reset Camera edu.mit.media.osc-gesture`.

### How the bundle is put together

- The spec bundles `hand_landmarker.task` and `ui.html` (plus `orchestra.json`,
  if one exists at the project root) as data files. They're resolved at runtime
  through `_resource_path()`.
- It excludes `jax`, `jaxlib` and `scipy`. MediaPipe declares them, but nothing
  the app uses imports them, and they'd add ~330 MB. If a change starts using a
  MediaPipe feature that needs them, take them out of `excludes`.
- In the frozen app, `main.py` runs the web UI under a native AppKit event loop
  (`classes/mac_app.py`) so the app can be quit and re-opened like any other.
  From source, nothing changes.
- `packaging/entitlements.plist` grants the camera. The hardened runtime that
  notarization requires blocks it otherwise.

## Project layout

- `classes/app.py` — main gesture-controller app (`OSCGestureApp`)
- `classes/web_ui.py` — local HTTP server: MJPEG preview + JSON control API
- `ui.html` — the web UI page (bundled into the app via the spec's `datas`)
- `classes/tracker.py` — MediaPipe hand tracking wrapper
- `classes/gesture_detector.py` — classifies hand motion into noop/runs/chords/faster/slower
- `classes/gesture_sender.py` — debounces/accumulates gesture detections into OSC messages
- `classes/recorder_app.py`, `classes/playback_app.py` — recording/playback tooling
- `classes/mac_app.py` — native macOS event loop for the packaged `.app`
- `osc-gesture.spec`, `packaging/` — PyInstaller spec, signing and notarization scripts
- `recordings/` — recorded `.npz` gesture clips
