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
| Left | Preset buttons (`1`–`5`), orchestra toggle (`o`), landmark/debug toggles |
| Centre | Camera preview with the draggable region dividers |
| Right | Gesture readout on top, live OSC log below |

The debug-score panel sits full-width underneath. On a narrow screen the three
columns stack.

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
`o` toggle orchestra mode.

## Orchestra mode

Toggle it with the checkbox under the presets, or the `o` key. It splits the
active region into three instrument zones:

```
┌─────────────────┬─────────────────┐
│     BRASS       │    STRINGS      │   ← top half, split by a vertical bar
│   (instr 2)     │   (instr 1)     │      (both bars draggable)
├─────────────────┴─────────────────┤   ← horizontal bar (draggable)
│              PIANO                │   ← bottom half
│           (4-arg form)            │
├───────────────────────────────────┤   ← exclusion boundary (draggable)
│▒▒▒▒▒▒▒ excluded region ▒▒▒▒▒▒▒▒▒▒▒│
└───────────────────────────────────┘
```

Both amber dividers are draggable, as is the exclusion boundary. The nearest
bar wins when two are close together. The piano split is stored as a fraction
of the active region, so it stays valid when you move the exclusion boundary.

### Messages

| Zone | Message |
|---|---|
| Piano | `/setOutputRange 1 lo hi lo2 hi2` — the four-argument form led by the piano's id |
| Strings | `/setOutputRange 48 lo hi -1 -1` |
| Brass | `/setOutputRange 61 lo hi -1 -1` |

The first argument is the instrument's `id` from the preset (the token instrument
JordanAI keys its output instruments by), not its position in the list: JordanAI
looks the number up as an id, so an index would have moved the piano (id 1) and
ignored the brass. Outside orchestra mode the piano keeps the plain four-argument
form, which JordanAI applies to its first active instrument whatever its id.

Each zone's range is driven by the hands currently inside it, independently of
the others. Two hands in the same zone collapse to one message at their mean x.
The piano keeps its original behaviour: one hand drives both channels, two
hands drive left and right separately.

Hand position is the **centroid of the five finger-tip landmarks** (4, 8, 12,
16, 20). That's used for every position decision — which zone a hand is in,
the pitch mapping, and whether a hand is inside the active area at all.

`x` maps across the whole frame rather than across each column, so each column
reaches only its own half of the pitch range: brass the lower half, strings the
upper.

### Zone resets

When a zone's last hand leaves it, that zone is reset once to its instrument's
default range, so an instrument doesn't stay parked where a hand left it. The
reset is edge-triggered — it fires on the transition to empty, not on every
empty frame — and it clears that zone's send throttle so returning a hand to
the same spot re-sends immediately.

Defaults come from an orchestra preset's `outputInstruments` list, indexed by
the zone's zero-based position (`_REGION_INSTRUMENT`); the entry's `id` is what
goes on the wire. The `id`s are General MIDI programs, which is what fixes the
order — 1 is grand piano, 48 string ensemble, 61 brass section:

| Zone | Index | Preset entry | Default range |
|---|---|---|---|
| Piano | 0 | `outputInstruments[0]`, `id: 1` (grand piano) | 26–89 |
| Strings | 1 | `outputInstruments[1]`, `id: 48` (string ensemble) | 33–94 |
| Brass | 2 | `outputInstruments[2]`, `id: 61` (brass section) | 36–92 |

(The ranges shown are from the preset this was built against; the app reads
whatever your preset has, ids included.)

Point the app at a preset with `--orchestra-preset path/to/orchestra.json`;
without it, the app looks for `orchestra.json` beside itself and otherwise
falls back to built-in values. The zone→position mapping lives in
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

```bash
make build     # or: pyinstaller --noconfirm osc-gesture.spec
```

Produces `dist/OSC Gesture.app`. The spec bundles `hand_landmarker.task` and
`ui.html` as data files, resolved at runtime through `_resource_path()`, so the
web UI works from the frozen app with no extra dependencies.

## Project layout

- `classes/app.py` — main gesture-controller app (`OSCGestureApp`)
- `classes/web_ui.py` — local HTTP server: MJPEG preview + JSON control API
- `ui.html` — the web UI page (bundled into the app via the spec's `datas`)
- `classes/tracker.py` — MediaPipe hand tracking wrapper
- `classes/gesture_detector.py` — classifies hand motion into noop/runs/chords/faster/slower
- `classes/gesture_sender.py` — debounces/accumulates gesture detections into OSC messages
- `classes/recorder_app.py`, `classes/playback_app.py` — recording/playback tooling
- `recordings/` — recorded `.npz` gesture clips
