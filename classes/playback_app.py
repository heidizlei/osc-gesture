import os
import glob
import time

import cv2
import numpy as np
import mediapipe as mp

# Hand skeleton connections from MediaPipe
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

CANVAS_W = 1280
CANVAS_H = 720

# Two distinct hand colours (BGR)
HAND_COLOURS = [
    (0, 255, 100),   # hand 0 — green
    (0, 180, 255),   # hand 1 — orange
]
CONNECTION_COLOURS = [
    (0, 180, 60),
    (0, 120, 200),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _list_recordings(directory="recordings"):
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    return files


def _load_recording(path):
    data = np.load(path, allow_pickle=True)
    return {
        "timestamps": data["timestamps"],          # (N,)
        "landmarks": data["landmarks"],            # (N, 2, 21, 3)  screen-norm
        "world_landmarks": data["world_landmarks"],
        "mode": str(data["mode"][0]),
    }


class PlaybackApp:
    """
    Plays back .npz hand-landmark recordings on a synthetic canvas.

    Controls:
        SPACE     — play / pause
        LEFT/RIGHT — prev / next recording
        R         — restart current recording
        Q / ESC   — quit
    """

    def __init__(self, recordings_dir="recordings"):
        self.recordings_dir = recordings_dir
        self.files = _list_recordings(recordings_dir)
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in '{recordings_dir}'")

        self.file_idx = 0
        self.recording = None       # loaded dict
        self.frame_idx = 0
        self.playing = False
        self.playback_start_wall = None   # wall-clock time when play started
        self.playback_start_rec = None    # recording timestamp at that moment
        self.running = True

        self._load_current()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_current(self):
        path = self.files[self.file_idx]
        self.recording = _load_recording(path)
        self.frame_idx = 0
        self.playing = False
        self.playback_start_wall = None
        self.playback_start_rec = None
        print(f"Loaded [{self.recording['mode']}] {os.path.basename(path)}  "
              f"({len(self.recording['timestamps'])} frames, "
              f"{self.recording['timestamps'][-1]:.2f}s)")

    # ------------------------------------------------------------------
    # Frame lookup by wall-clock time
    # ------------------------------------------------------------------

    def _current_frame_idx(self):
        """Return the recording frame index that matches the current playback time."""
        if not self.playing or self.playback_start_wall is None:
            return self.frame_idx

        elapsed = time.time() - self.playback_start_wall
        rec_t = self.playback_start_rec + elapsed
        timestamps = self.recording["timestamps"]

        # Find the last frame whose timestamp <= rec_t
        idx = int(np.searchsorted(timestamps, rec_t, side="right")) - 1
        idx = max(0, min(idx, len(timestamps) - 1))
        return idx

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _make_canvas(self):
        return np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    def _draw_hand(self, canvas, landmarks_21x3, hand_idx):
        """Draw one hand (21 × 3 array, NaN = missing) onto canvas."""
        if np.all(np.isnan(landmarks_21x3)):
            return

        dot_colour = HAND_COLOURS[hand_idx % len(HAND_COLOURS)]
        line_colour = CONNECTION_COLOURS[hand_idx % len(CONNECTION_COLOURS)]

        h, w = canvas.shape[:2]

        def to_px(lm):
            return int(lm[0] * w), int(lm[1] * h)

        # Connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            s = landmarks_21x3[start_idx]
            e = landmarks_21x3[end_idx]
            if np.any(np.isnan(s)) or np.any(np.isnan(e)):
                continue
            cv2.line(canvas, to_px(s), to_px(e), line_colour, 2, cv2.LINE_AA)

        # Landmark dots
        for lm in landmarks_21x3:
            if np.any(np.isnan(lm)):
                continue
            cv2.circle(canvas, to_px(lm), 5, dot_colour, -1, cv2.LINE_AA)

        # Wrist label
        wrist = landmarks_21x3[0]
        if not np.any(np.isnan(wrist)):
            label = "L" if hand_idx == 0 else "R"
            cv2.putText(canvas, label, (to_px(wrist)[0] + 8, to_px(wrist)[1] - 8),
                        FONT, 0.6, dot_colour, 1, cv2.LINE_AA)

    def _draw_hud(self, canvas, frame_idx):
        rec = self.recording
        n = len(rec["timestamps"])
        t = rec["timestamps"][frame_idx]
        total_t = rec["timestamps"][-1]
        mode = rec["mode"]
        filename = os.path.basename(self.files[self.file_idx])
        h, w = canvas.shape[:2]

        # ---- Top bar ----
        cv2.rectangle(canvas, (0, 0), (w, 60), (20, 20, 20), -1)

        cv2.putText(canvas, mode.upper(), (12, 40),
                    FONT, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, filename, (140, 40),
                    FONT, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        # Play/pause indicator
        state_text = "PLAY" if self.playing else "PAUSED"
        state_colour = (0, 200, 80) if self.playing else (80, 80, 200)
        cv2.putText(canvas, state_text, (w - 120, 40),
                    FONT, 0.7, state_colour, 2, cv2.LINE_AA)

        # ---- Progress bar ----
        bar_x, bar_y, bar_w, bar_h = 0, h - 20, w, 20
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 40), -1)
        filled = int(bar_w * (frame_idx / max(n - 1, 1)))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      (0, 160, 255), -1)

        # Time counter over progress bar
        time_text = f"{t:.2f}s / {total_t:.2f}s   frame {frame_idx + 1}/{n}"
        cv2.putText(canvas, time_text, (8, h - 4),
                    FONT, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

        # ---- File list (bottom-left) ----
        list_y = h - 35
        for i, fp in enumerate(reversed(self.files[-5:])):  # show up to 5 recent
            actual_idx = len(self.files) - 1 - i
            label = ("▶ " if actual_idx == self.file_idx else "  ") + os.path.basename(fp)
            colour = (255, 220, 0) if actual_idx == self.file_idx else (100, 100, 100)
            cv2.putText(canvas, label, (8, list_y),
                        FONT, 0.42, colour, 1, cv2.LINE_AA)
            list_y -= 18
            if list_y < 70:
                break

        # ---- Controls hint ----
        hint = "SPACE: play/pause   LEFT/RIGHT: prev/next   R: restart   Q: quit"
        cv2.putText(canvas, hint, (8, h - 26),
                    FONT, 0.42, (70, 70, 70), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print("Playback controls: SPACE=play/pause  LEFT/RIGHT=prev/next  R=restart  Q=quit")
        cv2.namedWindow("Hand Landmark Playback", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Landmark Playback", CANVAS_W, CANVAS_H)

        while self.running:
            # ------ determine which frame to show ------
            if self.playing:
                new_idx = self._current_frame_idx()
                if new_idx >= len(self.recording["timestamps"]) - 1:
                    # Reached end — stop
                    self.frame_idx = len(self.recording["timestamps"]) - 1
                    self.playing = False
                else:
                    self.frame_idx = new_idx
            # (paused: frame_idx unchanged)

            # ------ render ------
            canvas = self._make_canvas()

            lm_frame = self.recording["landmarks"][self.frame_idx]   # (2, 21, 3)
            for hand_idx in range(2):
                self._draw_hand(canvas, lm_frame[hand_idx], hand_idx)

            self._draw_hud(canvas, self.frame_idx)

            cv2.imshow("Hand Landmark Playback", canvas)

            # ------ key handling ------
            # Use raw key (no & 0xFF) so arrow keys work on macOS.
            # macOS: left=2 (0xF702), right=3 (0xF703)
            # Linux: left=81, right=83
            raw_key = cv2.waitKey(16)
            key = raw_key & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                self.running = False

            elif key == ord(' '):
                if self.playing:
                    # Pause: freeze current frame index
                    self.playing = False
                else:
                    # Play (or resume)
                    self.playing = True
                    self.playback_start_wall = time.time()
                    self.playback_start_rec = self.recording["timestamps"][self.frame_idx]

            elif key == ord('r'):
                self.frame_idx = 0
                self.playing = False

            elif raw_key in (81, 65361, 2) or key == ord('h'):  # LEFT arrow
                self.file_idx = (self.file_idx - 1) % len(self.files)
                self._load_current()

            elif raw_key in (83, 65363, 3) or key == ord('l'):  # RIGHT arrow
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self._load_current()

        cv2.destroyAllWindows()
