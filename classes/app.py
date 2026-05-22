import time
import cv2
import numpy as np
from pythonosc import udp_client
from .tracker import HandTracker, HandLandmarkDrawer
from .gesture_detector import GestureDetector
from .gesture_sender import GestureSender
import gc

_MODE_COLORS = {
    'noop':   (160, 160, 160),
    'runs':   (80,  200, 80),
    'chords': (200, 200, 80),
    'faster': (80,  200, 200),
    'slower': (80,  80,  220),
}


class OSCGestureApp:
    def __init__(self,
                 model_path="hand_landmarker.task",
                 camera_index=0,
                 ip="0.0.0.0",
                 port=9001,
                 baroque=False,
                 mode="tempo"):
        self.interval = (-8, 8)

        # OSC client
        self.osc_client = udp_client.SimpleUDPClient(ip, port)

        # Hand tracking
        self.hand_tracker = HandTracker(
            model_path=model_path,
            camera_index=camera_index,
            use_gpu=False   # GPU = leaks on macOS; CPU recommended
        )

        self.draw_landmarks = False

        self.min_val = 10
        self.max_val = 117
        self.left_val = 63
        self.right_val = 63

        self.last_left = None
        self.last_right = None
        self.change_threshold = 4

        self.running = True

        self.osc_interval = 0.2
        self.inactivity_interval = 1.0
        self.last_osc_time = time.time()
        self.inactivity_message_sent = False

        self.mode = mode   # 'pause-only', 'range', 'tempo'

        # Gesture detection + OSC sending
        self.gesture_detector = GestureDetector()
        self.gesture_sender   = GestureSender(baroque=baroque)
        self.gesture_result   = ('noop', 0.0)
        self.debug_mode       = False  # toggle with 'd' key

        # Hand presence tracking
        self.hand_present       = False
        self.last_hand_present  = False
        self.hand_absent_since  = None   # timestamp when hand first disappeared
        self.PAUSE_ABSENT_S     = 0.5    # seconds of sustained absence before sending pause

        # Only top part of camera image is active for control
        self.active_area_ratio = 3 / 4


    # ----------------------------
    # Gesture detection
    # ----------------------------

    def _extract_world_landmarks(self, results):
        """Convert MediaPipe results to (2, 21, 3) float32 array, NaN for missing hands."""
        wl = np.full((2, 21, 3), np.nan, dtype=np.float32)
        if results and results.hand_world_landmarks:
            for i, hand in enumerate(results.hand_world_landmarks[:2]):
                for j, lm in enumerate(hand):
                    wl[i, j] = (lm.x, lm.y, lm.z)
        return wl

    def _extract_wrist_img(self, results):
        """Wrist y position per hand in normalised image coords [0,1], NaN if missing."""
        wy = np.full(2, np.nan, dtype=np.float32)
        if results and results.hand_landmarks:
            for i, hand in enumerate(results.hand_landmarks[:2]):
                wy[i] = hand[0].y   # landmark 0 = wrist, y = vertical screen position
        return wy

    def _draw_gesture_hud(self, frame, mode, intensity):
        color = _MODE_COLORS.get(mode, (160, 160, 160))
        label = f"{mode}  {intensity:.2f}"
        cv2.putText(frame, label, (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
        # intensity bar (200 px wide)
        bar_x, bar_y, bar_h = 20, 52, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 200, bar_y + bar_h),
                      (60, 60, 60), -1)
        filled = int(intensity * 200)
        if filled > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                          color, -1)

    def _draw_debug_hud(self, frame):
        """Overlay raw decision scores for each hand (toggle with 'd')."""
        det  = self.gesture_detector
        DIM  = (80,  80,  80)    # dimmed label colour
        OK   = (20,  130, 20)    # value meets threshold  (dark green)
        FAIL = (180, 30,  30)    # value misses threshold (dark red)
        INFO = (50,  50,  50)    # neutral info
        HDR  = (100, 80,  0)     # header (dark amber)
        BG   = (255, 255, 255)   # white background

        x, y, lh = 20, 82, 36
        pad = 6  # background padding around text

        def _bg_rect(text, font_scale, thickness=1):
            """Draw a white filled rect behind the upcoming text at (x, y)."""
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, thickness)
            cv2.rectangle(frame,
                          (x - pad, y - th - pad),
                          (x + tw + pad, y + bl + pad),
                          BG, -1)

        _bg_rect("[ DEBUG SCORES ]", 0.90)
        cv2.putText(frame, "[ DEBUG SCORES ]", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, HDR, 1, cv2.LINE_AA)
        y += lh

        def _put(text, col):
            nonlocal y
            _bg_rect(text, 0.80)
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 1, cv2.LINE_AA)
            y += lh

        for hi, sc in enumerate(det.debug_scores):
            _put(f"── Hand {hi} ──", HDR)

            if sc.get('tip_artic') is None:
                _put("  (no data)", DIM)
                continue

            ta  = sc['tip_artic']
            td  = sc['tip_disp']
            ext = sc['ext_change']
            av  = sc['ang_vel']

            # --- Runs / Chords ---
            ta_ok = ta >= det.T_RUN_TIP_ARTIC
            td_ok = td >= det.T_CHORD_TIP_DISP
            ec_ok = ext >= det.T_RUN_EXT
            _put(f"  tip_artic {ta:.3f}>={det.T_RUN_TIP_ARTIC}{'Y' if ta_ok else 'N'}"
                 f"  ext_chg {ext:.2f}>={det.T_RUN_EXT}{'Y' if ec_ok else 'N'}",
                 OK if (ta_ok and ec_ok) else FAIL)
            _put(f"  tip_disp  {td:.3f}>={det.T_CHORD_TIP_DISP}{'Y' if td_ok else 'N'}"
                 f"  → {'CHORD' if sc['is_chord'] else 'chord'}  {'RUN' if sc['is_run'] else 'run'}",
                 OK if (sc['is_chord'] or sc['is_run']) else DIM)

            # --- Faster ---
            idx_ext      = sc['index_ext']
            others_dist  = sc['others_dist_max']
            idx_hold     = sc['index_hold']
            idx_ext_ok   = idx_ext   >= det.T_INDEX_EXT
            others_ok    = others_dist < det.T_OTHERS_DIST
            hold_ok      = idx_hold  >= det.T_INDEX_HOLD
            av_ok        = av        >= det.T_ROTATE
            faster_armed = idx_ext_ok and others_ok and hold_ok
            _put(f"  idx_ext {idx_ext:.2f}>={det.T_INDEX_EXT}{'Y' if idx_ext_ok else 'N'}"
                 f"  others_d {others_dist:.3f}<{det.T_OTHERS_DIST}{'Y' if others_ok else 'N'}"
                 f"  hold {idx_hold:.2f}s>={det.T_INDEX_HOLD}{'Y' if hold_ok else 'N'}",
                 OK if faster_armed else (FAIL if not idx_ext_ok else INFO))
            _put(f"  ang_vel {av:.2f}>={det.T_ROTATE}{'Y' if av_ok else 'N'}"
                 f"  → {'FASTER' if (faster_armed and av_ok) else 'faster'}",
                 OK if (faster_armed and av_ok) else (INFO if faster_armed else FAIL))

            # --- Slower ---
            me = sc['mean_ext']
            fi = sc['fist_intensity']
            fp = sc['fist_pending']
            me_fist = me < det.T_FIST
            me_open = me > det.T_OPEN
            _put(f"  mean_ext {me:.2f}  {'<FIST' if me_fist else ('>OPEN' if me_open else 'mid')}"
                 f"  fist={'PENDING' if fp else f'int={fi:.2f}'}",
                 OK if fp else (INFO if me_fist else DIM))

            y += 6

    # ----------------------------
    # OSC sending
    # ----------------------------

    def send_osc_message(self, left_val=None, right_val=None):
        if self.mode == 'pause-only':
            return
        if left_val == -1 and right_val == -1:
            arg1, arg2, arg3, arg4 = 36, 120, -1, -1
        else:
            left_val = left_val if left_val is not None else self.left_val
            right_val = right_val if right_val is not None else self.right_val

            arg1 = left_val + self.interval[0]
            arg2 = left_val + self.interval[1]
            arg3 = right_val + self.interval[0]
            arg4 = right_val + self.interval[1]

        try:
            self.osc_client.send_message("/setOutputRange", [arg1, arg2, arg3, arg4])
            print(f"OSC → /setOutputRange {arg1} {arg2} {arg3} {arg4}")
        except Exception as e:
            print("OSC send error:", e)

    def send_manual_pause(self, pause_flag):
        try:
            self.osc_client.send_message("/setManualPause", pause_flag)
            print(f"OSC → /setManualPause {pause_flag}")
        except Exception as e:
            print("Pause OSC error:", e)


    # ----------------------------
    # Hand position mapping
    # ----------------------------

    def map_hand_x_to_val(self, x_norm):
        if x_norm is None:
            return None
        x_norm = max(0.0, min(1.0, x_norm))
        return int(self.min_val + x_norm * (self.max_val - self.min_val))


    # ----------------------------
    # Events
    # ----------------------------


    # ----------------------------
    # Inactivity
    # ----------------------------

    def handle_inactivity(self):
        now = time.time()
        if now - self.last_osc_time >= self.inactivity_interval and not self.inactivity_message_sent:
            self.send_osc_message(left_val=-1, right_val=-1)
            self.inactivity_message_sent = True


    # ----------------------------
    # Main Loop
    # ----------------------------

    def run(self):
        print("Running OSC Gesture App.")

        frame_counter = 0

        while self.running:
            # ------------------------
            # Camera + Hand Tracking
            # ------------------------
            frame, results = self.hand_tracker.get_frame_and_landmarks()

            if frame is not None:
                annotated = frame  # operate directly on original frame

                h, w, _ = annotated.shape
                inactive_y = int(h * self.active_area_ratio)

                # Transparent overlay for bottom 1/4
                overlay = annotated.copy()
                cv2.rectangle(
                    overlay,
                    (0, inactive_y),
                    (w, h),
                    (100, 100, 255),    # color
                    -1
                )
                alpha = 0.25
                cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

                self.hand_present = bool(results and results.hand_landmarks)

                # Gesture detection (runs every frame, reports every 500 ms)
                wl  = self._extract_world_landmarks(results)
                wy  = self._extract_wrist_img(results)
                prev = self.gesture_result
                self.gesture_result = self.gesture_detector.update(wl, time.time(), wrist_y=wy)
                # Only tick the sender when a new report has been emitted
                if self.gesture_result is not prev and self.mode == 'tempo':
                    mode, intensity = self.gesture_result
                    self.gesture_sender.tick(mode, intensity, self.osc_client)

                # Send pause only after hand has been absent for PAUSE_ABSENT_S,
                # to avoid spurious pause/resume on brief detection dropouts.
                now = time.time()
                if self.hand_present:
                    self.hand_absent_since = None
                    if not self.last_hand_present:
                        self.send_manual_pause(0)  # resume immediately on reappearance
                        self.last_hand_present = True
                else:
                    if self.hand_absent_since is None:
                        self.hand_absent_since = now
                    elif (self.last_hand_present and
                          now - self.hand_absent_since >= self.PAUSE_ABSENT_S):
                        self.send_manual_pause(1)  # pause after sustained absence
                        self.last_hand_present = False

                if self.hand_present:
                    positions = []
                    for hand in results.hand_landmarks:
                        xs = [lm.x for lm in hand]
                        ys = [lm.y for lm in hand]
                        positions.append((float(np.mean(xs)), float(np.mean(ys))))

                    # Single-hand → both channels
                    if len(positions) == 1:
                        x, y = positions[0]
                        if y <= self.active_area_ratio:
                            mapped = self.map_hand_x_to_val(x)
                            self.left_val = mapped
                            self.right_val = mapped

                    # Two hands → left/right
                    elif len(positions) >= 2:
                        (x1, y1), (x2, y2) = positions[:2]

                        if y1 <= self.active_area_ratio:
                            self.left_val = self.map_hand_x_to_val(x1)
                        if y2 <= self.active_area_ratio:
                            self.right_val = self.map_hand_x_to_val(x2)

                    # Throttle OSC sends; only fire when change exceeds threshold
                    now = time.time()
                    if now - self.last_osc_time >= self.osc_interval:
                        left_changed = self.last_left is None or abs(self.left_val - self.last_left) > self.change_threshold
                        right_changed = self.last_right is None or abs(self.right_val - self.last_right) > self.change_threshold

                        if left_changed or right_changed:
                            self.last_left = self.left_val
                            self.last_right = self.right_val
                            self.last_osc_time = now
                            self.inactivity_message_sent = False
                            self.send_osc_message()

                # ---- show camera ----
                if self.draw_landmarks:
                    annotated = HandLandmarkDrawer.draw_landmarks(annotated, results)
                self._draw_gesture_hud(annotated, *self.gesture_result)
                if self.debug_mode:
                    self._draw_debug_hud(annotated)
                cv2.imshow("Hand Camera", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('l'):
                    self.draw_landmarks = not self.draw_landmarks
                    print("Draw landmarks:", self.draw_landmarks)
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print("Debug mode:", self.debug_mode)

            if frame_counter % 300 == 0:
                gc.collect()
            frame_counter += 1

        cv2.destroyAllWindows()
        self.hand_tracker.close()
