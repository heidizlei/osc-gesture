import time
import cv2
import numpy as np
from pythonosc import udp_client
from .tracker import HandTracker, HandLandmarkDrawer
from .gesture_detector import GestureDetector
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
                 port=9001):
        self.interval = (-6, 6)

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

        # Gesture detection
        self.gesture_detector = GestureDetector()
        self.gesture_result   = ('noop', 0.0)
        self.debug_mode       = False  # toggle with 'd' key

        # Hand presence tracking
        self.hand_present = False
        self.last_hand_present = False

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
            _put(f"Hand {hi}:", INFO)

            if sc.get('tip_artic') is None:
                _put("  (no data)", DIM)
                continue

            ta  = sc['tip_artic']
            mcp = sc['mcp_speed']
            ext = sc['ext_change']
            av  = sc['ang_vel']

            # tip_artic — used by both run and chord rules
            ta_ok_run   = ta  >= det.T_RUN_TIP_ARTIC
            td  = sc.get('tip_disp', 0.0)
            ta_ok_chord = td  >= det.T_CHORD_TIP_DISP
            _put(f"  tip_artic  {ta:.3f}  run>={det.T_RUN_TIP_ARTIC}" +
                 (" Y" if ta_ok_run else " N") +
                 f"  tip_disp {td:.3f}  chord>={det.T_CHORD_TIP_DISP}" +
                 (" Y" if ta_ok_chord else " N"),
                 OK if ta_ok_chord else (OK if ta_ok_run else FAIL))

            # mcp_speed — chord rule only
            mcp_ok = mcp >= det.T_CHORD_MCP
            _put(f"  mcp_speed  {mcp:.3f}  chord>={det.T_CHORD_MCP}" +
                 (" Y" if mcp_ok else " N"),
                 OK if mcp_ok else FAIL)

            # ext_change — run rule only
            ext_ok = ext >= det.T_RUN_EXT
            _put(f"  ext_change {ext:.3f}  run>={det.T_RUN_EXT}" +
                 (" Y" if ext_ok else " N"),
                 OK if ext_ok else FAIL)

            # angular velocity — faster rule
            av_ok = av >= det.T_ROTATE
            _put(f"  ang_vel    {av:.3f}  faster>={det.T_ROTATE}" +
                 (" Y" if av_ok else " N"),
                 OK if av_ok else FAIL)

            # index hold / mean_ext / fist
            idx = sc['index_hold']
            me  = sc['mean_ext']
            fi  = sc['fist_intensity']
            fp  = sc['fist_pending']
            _put(f"  idx_hold={idx:.2f}s  me={me:.2f}  fist={'P' if fp else 'F'}({fi:.2f})",
                 INFO)

            # decision summary
            chord_s = "CHORD" if sc['is_chord'] else "chord"
            run_s   = "RUN"   if sc['is_run']   else "run"
            _put(f"  → {chord_s}  {run_s}",
                 OK if (sc['is_chord'] or sc['is_run']) else DIM)

            y += 4  # small gap between hands

    # ----------------------------
    # OSC sending
    # ----------------------------

    def send_osc_message(self, left_val=None, right_val=None):
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
                wl = self._extract_world_landmarks(results)
                self.gesture_result = self.gesture_detector.update(wl, time.time())

                # Send pause/unpause only when state changes
                if self.hand_present != self.last_hand_present:
                    if self.hand_present:
                        self.send_manual_pause(0)  # resume
                    else:
                        self.send_manual_pause(1)  # pause
                    self.last_hand_present = self.hand_present

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
