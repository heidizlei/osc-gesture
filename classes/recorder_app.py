import time
import os
import gc
import cv2
import numpy as np
from datetime import datetime
from .tracker import HandTracker, HandLandmarkDrawer

MODES = {
    ord('1'): 'noop',
    ord('2'): 'runs',
    ord('3'): 'chords',
    ord('4'): 'faster',
    ord('5'): 'slower',
}

COUNTDOWN_SECS = 3.0
TRIM_SECS = 3.0


class RecorderApp:
    IDLE = 'IDLE'
    COUNTDOWN = 'COUNTDOWN'
    RECORDING = 'RECORDING'

    def __init__(self, model_path='hand_landmarker.task', camera_index=0):
        self.hand_tracker = HandTracker(
            model_path=model_path,
            camera_index=camera_index,
            use_gpu=False   # GPU = leaks on macOS; CPU recommended
        )

        self.state = self.IDLE
        self.current_mode: str = ''
        self.countdown_start: float = 0.0
        self.recording_start: float = 0.0

        # Each entry: {"t": float, "lm": (2,21,3) float32, "wlm": (2,21,3) float32}
        self.buffer = []

        self.draw_landmarks = False
        self.running = True


    # ----------------------------
    # Landmark helpers
    # ----------------------------

    def _pack_landmarks(self, hand_list, world_hand_list):
        """
        Convert results.hand_landmarks / results.hand_world_landmarks
        to two (2, 21, 3) float32 arrays, NaN-padded for missing hands.
        """
        lm = np.full((2, 21, 3), np.nan, dtype=np.float32)
        wlm = np.full((2, 21, 3), np.nan, dtype=np.float32)

        for i, hand in enumerate(hand_list[:2]):
            for j, pt in enumerate(hand):
                lm[i, j] = [pt.x, pt.y, pt.z]

        for i, hand in enumerate(world_hand_list[:2]):
            for j, pt in enumerate(hand):
                wlm[i, j] = [pt.x, pt.y, pt.z]

        return lm, wlm


    # ----------------------------
    # Save
    # ----------------------------

    def _save(self):
        if not self.buffer:
            print("Nothing recorded.")
            self.state = self.IDLE
            return

        stop_time = time.time()
        cutoff = stop_time - TRIM_SECS
        trimmed = [f for f in self.buffer if f['t'] < cutoff]

        if not trimmed:
            print("All frames fell inside the 3-second trim window; nothing saved.")
            self.buffer = []
            self.state = self.IDLE
            return

        # Relativize timestamps so t=0 is the first frame
        t0 = trimmed[0]['t']
        timestamps = np.array([f['t'] - t0 for f in trimmed], dtype=np.float64)
        landmarks = np.stack([f['lm'] for f in trimmed])           # (N, 2, 21, 3)
        world_landmarks = np.stack([f['wlm'] for f in trimmed])    # (N, 2, 21, 3)

        os.makedirs('recordings', exist_ok=True)
        ts_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')[:-3]
        filename = f"recordings/{self.current_mode}_{ts_str}.npz"

        np.savez_compressed(
            filename,
            timestamps=timestamps,
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            mode=np.array([self.current_mode]),
        )

        print(f"Saved {len(trimmed)} frames ({timestamps[-1]:.2f}s) → {filename}")
        self.buffer = []
        self.state = self.IDLE


    # ----------------------------
    # Overlay rendering
    # ----------------------------

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]

        if self.state == self.COUNTDOWN:
            elapsed = time.time() - self.countdown_start
            count = max(1, COUNTDOWN_SECS - int(elapsed))

            # Mode label — top-left
            cv2.putText(frame, self.current_mode.upper(), (20, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            # Large countdown number — centered
            text = str(int(count))
            font_scale, thickness = 6.0, 10
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cx, cy = (w - tw) // 2, (h + th) // 2
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 0), thickness, cv2.LINE_AA)

        elif self.state == self.RECORDING:
            duration = time.time() - self.recording_start

            # Mode label — top-left
            cv2.putText(frame, self.current_mode.upper(), (20, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            # REC status line
            rec_text = f"REC  {duration:.1f}s  [{len(self.buffer)} frames]"
            cv2.putText(frame, rec_text, (20, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 60, 255), 2, cv2.LINE_AA)

            # Red dot — top-right
            cv2.circle(frame, (w - 30, 30), 13, (0, 0, 220), -1)

        else:  # IDLE
            cv2.putText(frame, "1:noop  2:runs  3:chords  4:faster  5:slower", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "SPACE: stop + save   L: landmarks   Q: quit", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)


    # ----------------------------
    # Main loop
    # ----------------------------

    def run(self):
        print("Hand Trajectory Recorder")
        print("  1-5 : select mode and start 3-second countdown")
        print("  SPACE: stop recording, trim last 3s, save clip")
        print("  L    : toggle landmark drawing")
        print("  Q    : quit")

        frame_counter = 0

        while self.running:
            frame, results = self.hand_tracker.get_frame_and_landmarks()

            if frame is not None:
                now = time.time()

                # --- State transitions ---
                if self.state == self.COUNTDOWN:
                    if now - self.countdown_start >= COUNTDOWN_SECS:
                        self.state = self.RECORDING
                        self.recording_start = now
                        print(f"Recording [{self.current_mode}] — press SPACE to stop.")

                # --- Buffer frames during RECORDING ---
                # Always buffer (NaN when no hand) to preserve timeline continuity
                if self.state == self.RECORDING:
                    hand_list = results.hand_landmarks if (results and results.hand_landmarks) else []
                    world_list = results.hand_world_landmarks if (results and results.hand_world_landmarks) else []
                    lm, wlm = self._pack_landmarks(hand_list, world_list)
                    self.buffer.append({'t': now, 'lm': lm, 'wlm': wlm})

                # --- Overlay + optional landmarks ---
                if self.draw_landmarks:
                    frame = HandLandmarkDrawer.draw_landmarks(frame, results)

                self._draw_overlay(frame)

                cv2.imshow("Hand Recorder", frame)

                # --- Key handling ---
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.running = False

                elif key == ord(' '):
                    if self.state == self.RECORDING:
                        self._save()
                    elif self.state == self.COUNTDOWN:
                        print("Countdown canceled.")
                        self.state = self.IDLE
                        self.buffer = []

                elif key == ord('l'):
                    self.draw_landmarks = not self.draw_landmarks
                    print("Draw landmarks:", self.draw_landmarks)

                elif key in MODES:
                    if self.state == self.RECORDING and self.buffer:
                        print("Discarding unsaved recording.")
                    self.buffer = []
                    self.current_mode = MODES[key]
                    self.countdown_start = now
                    self.state = self.COUNTDOWN
                    print(f"Mode: {self.current_mode} — starting countdown...")

            if frame_counter % 300 == 0:
                gc.collect()
            frame_counter += 1

        cv2.destroyAllWindows()
        self.hand_tracker.close()
