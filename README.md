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
| `--host` | `100.101.30.29` | OSC target host |
| `--port` | `9001` | OSC target port |

Control mode is no longer passed via CLI args — it's selected live with number
keys while the camera window is focused, and shown at the bottom of the HUD:

| Key | Preset | Behavior |
|---|---|---|
| `1` | `pedal-only` | Only hand-presence pause/resume (`/setManualPause`) |
| `2` | `range` | Pause + pitch range (`/setOutputRange`) from hand x-position — default on startup |
| `3` | `rc-slow` | Pause + range + runs/chords capped to moderate levels (baroque-style), no tempo adjust |
| `4` | `rc` | Pause + range + full-range runs/chords (`/playRuns`, `/playChords`), no tempo adjust |
| `5` | `tempo` | Everything, including gesture-driven tempo adjust (`/adjustTempo`) |

Other controls: `q` quit, `l` toggle landmark drawing, `d` toggle debug HUD.

### Detection exclusion area

The red bottom region is masked before hand detection and excluded from gesture,
pitch-range, and hand-presence controls. Adjust its boundary using the native
**Active area %** slider in the camera window (10–95%; starts at 75%). A smaller
percentage means a larger excluded region. You can also drag in the red area;
use the slider if image dragging does not respond on your OpenCV backend.
Changes last for the current session. Restart the app after updating the code;
an existing packaged executable must be rebuilt to include these controls.


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
button/slider input as OSC UDP messages (no camera/gesture detection).

## Project layout

- `classes/app.py` — main gesture-controller app (`OSCGestureApp`)
- `classes/tracker.py` — MediaPipe hand tracking wrapper
- `classes/gesture_detector.py` — classifies hand motion into noop/runs/chords/faster/slower
- `classes/gesture_sender.py` — debounces/accumulates gesture detections into OSC messages
- `classes/recorder_app.py`, `classes/playback_app.py` — recording/playback tooling
- `recordings/` — recorded `.npz` gesture clips
