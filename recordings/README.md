# Recording Format

Each clip is saved as a compressed NumPy archive (`.npz`).

## Filename

```
<mode>_<YYYY-MM-DDTHH-MM-SS.mmm>.npz
```

Example: `runs_2026-05-15T14-32-05.123.npz`

## Loading

```python
import numpy as np

clip = np.load("runs_2026-05-15T14-32-05.123.npz", allow_pickle=True)
timestamps     = clip["timestamps"]      # (N,)          float64
landmarks      = clip["landmarks"]       # (N, 2, 21, 3) float32
world_landmarks = clip["world_landmarks"] # (N, 2, 21, 3) float32
mode           = str(clip["mode"][0])    # e.g. "runs"
```

## Arrays

### `timestamps` — `(N,)` float64
Seconds elapsed since the start of the clip. `timestamps[0]` is always `0.0`.

### `landmarks` — `(N, 2, 21, 3)` float32
Hand landmarks in **normalized image space** (x, y in [0, 1] relative to frame dimensions; z is depth relative to the wrist).

Axes: `[frame, hand_index, landmark_index, xyz]`

### `world_landmarks` — `(N, 2, 21, 3)` float32
Hand landmarks in **metric 3D space** (x, y, z in metres; origin at the hand's geometric centre). Captures hand shape and pose, independent of position in the frame.

Axes: `[frame, hand_index, landmark_index, xyz]`

### `mode` — `(1,)` str array
The label assigned when the clip was recorded. One of: `noop`, `runs`, `chords`, `faster`, `slower`.

## Notes

- `hand_index 0` = first detected hand, `hand_index 1` = second detected hand.
- Frames where fewer than 2 hands are detected have `NaN` in the missing hand's slice.
- Frames where no hands are detected have `NaN` in both hand slices.
- The MediaPipe hand landmarker returns up to 2 hands per frame.
- Each hand has 21 landmarks following the [MediaPipe hand topology](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

## Landmark Index Reference

| Index | Landmark        |
|-------|-----------------|
| 0     | WRIST           |
| 1–4   | THUMB (CMC→TIP) |
| 5–8   | INDEX (MCP→TIP) |
| 9–12  | MIDDLE (MCP→TIP)|
| 13–16 | RING (MCP→TIP)  |
| 17–20 | PINKY (MCP→TIP) |
