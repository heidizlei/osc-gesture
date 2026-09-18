import time
import os
import sys
import cv2
import numpy as np
from pythonosc import udp_client
from .tracker import HandTracker, HandLandmarkDrawer
from .gesture_detector import GestureDetector
from .gesture_sender import GestureSender
import gc


def _resource_path(relative_path):
    """Resolve a path to a bundled resource.

    Works both when running from source (resolves relative to the project
    root) and when frozen into a PyInstaller onefile binary (resolves
    relative to the temp extraction dir, sys._MEIPASS).
    """
    if os.path.isabs(relative_path):
        return relative_path
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


_MODE_COLORS = {
    'noop':   (160, 160, 160),
    'runs':   (80,  200, 80),
    'chords': (200, 200, 80),
    'faster': (80,  200, 200),
    'slower': (80,  80,  220),
}

_WINDOW_NAME = "Hand Camera"
_BOUNDARY_TRACKBAR = "Active area %"


class OSCGestureApp:
    # preset name -> (app_mode, baroque, tempo_enabled)
    _PRESETS = {
        'pedal-only': dict(app_mode='pause', baroque=False, tempo_enabled=False),
        'range':      dict(app_mode='range', baroque=False, tempo_enabled=False),
        'rc-slow':    dict(app_mode='tempo', baroque=True,  tempo_enabled=False),
        'rc':         dict(app_mode='tempo', baroque=False, tempo_enabled=False),
        'tempo':      dict(app_mode='tempo', baroque=False, tempo_enabled=True),
    }
    # keyboard key -> preset name (in the order the buttons were requested)
    _PRESET_KEYS = {
        ord('1'): 'pedal-only',
        ord('2'): 'range',
        ord('3'): 'rc-slow',
        ord('4'): 'rc',
        ord('5'): 'tempo',
    }

    def __init__(self,
                 model_path="hand_landmarker.task",
                 camera_index=0,
                 ip="0.0.0.0",
                 port=9001,
                 preset="range"):
        self.interval = (-8, 8)

        # OSC client
        self.osc_client = udp_client.SimpleUDPClient(ip, port)

        # Hand tracking
        self.hand_tracker = HandTracker(
            model_path=_resource_path(model_path),
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

        # Gesture detection + OSC sending
        self.gesture_detector = GestureDetector()
        self.gesture_sender   = GestureSender()
        self.gesture_result   = ('noop', 0.0)
        self.debug_mode       = False  # toggle with 'd' key

        # Preset: 'pedal-only' | 'range' | 'rc-slow' | 'rc' | 'tempo'
        # keys 1-5 switch presets live; sets self.mode/gesture_sender.baroque/enable_tempo
        self._apply_preset(preset)

        # Hand presence tracking
        self.hand_present       = False
        self.last_hand_present  = False
        self.hand_absent_since  = None   # timestamp when hand first disappeared
        self.PAUSE_ABSENT_S     = 0.5    # seconds of sustained absence before sending pause

        # Only top part of camera image is active for control
        self.active_area_ratio = 3 / 4
        self._boundary_dragging = False
        self._frame_height = None
        self._boundary_trackbar_ready = False


    # ----------------------------
    # Presets
    # ----------------------------

    def _apply_preset(self, name):
        cfg = self._PRESETS[name]
        self.preset = name
        self.mode = cfg['app_mode']
        self.gesture_sender.baroque      = bool(cfg['baroque'])
        self.gesture_sender.enable_tempo = bool(cfg['tempo_enabled'])
        print(f"Preset -> {name}  (mode={self.mode}, baroque={cfg['baroque']}, "
              f"tempo={cfg['tempo_enabled']})")


    # ----------------------------
    # Gesture detection
    # ----------------------------

    def _active_hand_indices(self, results):
        """Return hands whose landmark centroid is above the exclusion boundary."""
        if not results or not results.hand_landmarks:
            return []
        return [
            i for i, hand in enumerate(results.hand_landmarks[:2])
            if float(np.mean([lm.y for lm in hand])) <= self.active_area_ratio
        ]

    def _extract_world_landmarks(self, results, hand_indices):
        """Convert MediaPipe results to (2, 21, 3) float32 array, NaN for missing hands."""
        wl = np.full((2, 21, 3), np.nan, dtype=np.float32)
        if results and results.hand_world_landmarks:
            for i, hand_index in enumerate(hand_indices):
                hand = results.hand_world_landmarks[hand_index]
                for j, lm in enumerate(hand):
                    wl[i, j] = (lm.x, lm.y, lm.z)
        return wl

    def _extract_wrist_img(self, results, hand_indices):
        """Wrist y position per hand in normalised image coords [0,1], NaN if missing."""
        wy = np.full(2, np.nan, dtype=np.float32)
        if results and results.hand_landmarks:
            for i, hand_index in enumerate(hand_indices):
                hand = results.hand_landmarks[hand_index]
                wy[i] = hand[0].y
        return wy

    def _extract_image_landmarks(self, results, hand_indices):
        """Image-space landmarks (2, 21, 3) float32, NaN for missing hands."""
        lm = np.full((2, 21, 3), np.nan, dtype=np.float32)
        if results and results.hand_landmarks:
            for i, hand_index in enumerate(hand_indices):
                hand = results.hand_landmarks[hand_index]
                for j, p in enumerate(hand):
                    lm[i, j] = (p.x, p.y, p.z)
        return lm

    def _set_active_area_ratio(self, ratio):
        """Keep the exclusion mask, overlay, and native slider in sync."""
        percent = int(round(float(np.clip(ratio, 0.1, 0.95)) * 100))
        self.active_area_ratio = percent / 100.0
        if self._boundary_trackbar_ready:
            if cv2.getTrackbarPos(_BOUNDARY_TRACKBAR, _WINDOW_NAME) != percent:
                cv2.setTrackbarPos(_BOUNDARY_TRACKBAR, _WINDOW_NAME, percent)

    def _handle_boundary_slider(self, percent):
        self._set_active_area_ratio(percent / 100.0)

    def _create_camera_window(self):
        cv2.namedWindow(_WINDOW_NAME)
        cv2.setMouseCallback(_WINDOW_NAME, self._handle_mouse)
        cv2.createTrackbar(_BOUNDARY_TRACKBAR, _WINDOW_NAME,
                          round(self.active_area_ratio * 100), 95,
                          self._handle_boundary_slider)
        cv2.setTrackbarMin(_BOUNDARY_TRACKBAR, _WINDOW_NAME, 10)
        self._boundary_trackbar_ready = True

    def _handle_mouse(self, event, x, y, flags, param):
        """Drag anywhere in the red exclusion area to set its top boundary."""
        if self._frame_height is None:
            return
        boundary_y = int(self._frame_height * self.active_area_ratio)
        if event == cv2.EVENT_LBUTTONDOWN and y >= boundary_y - 12:
            self._boundary_dragging = True
            self._set_active_area_ratio(y / self._frame_height)
        elif event == cv2.EVENT_MOUSEMOVE and self._boundary_dragging:
            self._set_active_area_ratio(y / self._frame_height)
        elif event == cv2.EVENT_LBUTTONUP:
            self._boundary_dragging = False

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

        # Last sent mode — top right, white background
        sent_mode  = self.gesture_sender._last_sent_mode or 'none'
        sent_level = self.gesture_sender._last_sent_level
        sent_label = sent_mode if sent_level is None else f"{sent_mode} L{sent_level}"
        sent_color = _MODE_COLORS.get(sent_mode, (80, 80, 80))
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        (tw, th), bl = cv2.getTextSize(sent_label, font, scale, thick)
        h, w = frame.shape[:2]
        tx = w - tw - 16
        ty = th + 12
        cv2.rectangle(frame, (tx - 8, 4), (w - 4, ty + bl + 6), (255, 255, 255), -1)
        cv2.putText(frame, sent_label, (tx, ty), font, scale, sent_color, thick, cv2.LINE_AA)

    def _draw_preset_hud(self, frame):
        """Bottom-left legend showing the active preset and its key binding."""
        h = frame.shape[0]
        legend = "1 pedal-only  2 range  3 rc-slow  4 rc  5 tempo"
        current = f"current: {self.preset}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        (tw, th), bl = cv2.getTextSize(legend, font, scale, thick)
        cv2.rectangle(frame, (14, h - th - bl - 34), (14 + tw + 12, h - 4),
                      (255, 255, 255), -1)
        cv2.putText(frame, legend, (20, h - th - 14), font, scale, (60, 60, 60),
                    thick, cv2.LINE_AA)
        cv2.putText(frame, current, (20, h - 12), font, scale, (0, 100, 0),
                    thick, cv2.LINE_AA)

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
            others_ok    = others_dist < det.T_OTHERS_DIST_RATIO
            hold_ok      = idx_hold  >= det.T_INDEX_HOLD
            av_ok        = av        >= det.T_ROTATE
            faster_armed = idx_ext_ok and others_ok and hold_ok
            _put(f"  idx_ext {idx_ext:.2f}>={det.T_INDEX_EXT}{'Y' if idx_ext_ok else 'N'}"
                 f"  others_d {others_dist:.3f}<{det.T_OTHERS_DIST_RATIO}{'Y' if others_ok else 'N'}"
                 f"  hold {idx_hold:.2f}s>={det.T_INDEX_HOLD}{'Y' if hold_ok else 'N'}",
                 OK if faster_armed else (FAIL if not idx_ext_ok else INFO))
            _put(f"  ang_vel {av:.2f}>={det.T_ROTATE}{'Y' if av_ok else 'N'}"
                 f"  → {'FASTER' if (faster_armed and av_ok) else 'faster'}",
                 OK if (faster_armed and av_ok) else (INFO if faster_armed else FAIL))

            # --- Slower ---
            me = sc['mean_ext']
            fi = sc['fist_intensity']
            fp = sc['fist_pending']
            me_open = me > det.T_OPEN
            _put(f"  mean_ext {me:.2f}  {'>OPEN' if me_open else 'mid'}"
                 f"  fist={'PENDING' if fp else f'int={fi:.2f}'}",
                  OK if fp else DIM)

            y += 6

    # ----------------------------
    # OSC sending
    # ----------------------------

    def send_osc_message(self, left_val=None, right_val=None):
        if self.mode == 'pause':
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
        self._create_camera_window()

        while self.running:
            # ------------------------
            # Camera + Hand Tracking
            # ------------------------
            frame, results = self.hand_tracker.get_frame_and_landmarks(
                active_area_ratio=self.active_area_ratio)

            if frame is not None:
                annotated = frame  # operate directly on original frame

                h, w, _ = annotated.shape
                self._frame_height = h
                inactive_y = int(h * self.active_area_ratio)

                # Transparent overlay for the adjustable exclusion region
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
                cv2.line(annotated, (0, inactive_y), (w, inactive_y), (0, 0, 255), 2)
                cv2.putText(annotated,
                            f"Active {self.active_area_ratio:.0%} - use slider or drag red area",
                            (12, inactive_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                active_hand_indices = self._active_hand_indices(results)
                self.hand_present = bool(active_hand_indices)

                # Gesture detection (runs every frame, reports every 500 ms)
                # Hands below the red boundary are excluded from gesture, range,
                # and hand-presence processing.
                if not active_hand_indices:
                    self.gesture_detector.reset()
                wl  = self._extract_world_landmarks(results, active_hand_indices)
                wy  = self._extract_wrist_img(results, active_hand_indices)
                il  = self._extract_image_landmarks(results, active_hand_indices)
                prev = self.gesture_result
                self.gesture_result = self.gesture_detector.update(
                    wl, time.time(), wrist_y=wy, image_landmarks=il)
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
                    for hand_index in active_hand_indices:
                        hand = results.hand_landmarks[hand_index]
                        xs = [lm.x for lm in hand]
                        ys = [lm.y for lm in hand]
                        positions.append((float(np.mean(xs)), float(np.mean(ys))))

                    # Single-hand → both channels
                    if len(positions) == 1:
                        x, y = positions[0]
                        mapped = self.map_hand_x_to_val(x)
                        self.left_val = mapped
                        self.right_val = mapped

                    # Two hands → left/right
                    elif len(positions) >= 2:
                        (x1, y1), (x2, y2) = positions[:2]

                        self.left_val = self.map_hand_x_to_val(x1)
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
                    annotated = HandLandmarkDrawer.draw_landmarks(
                        annotated, results, active_hand_indices)
                self._draw_gesture_hud(annotated, *self.gesture_result)
                self._draw_preset_hud(annotated)
                if self.debug_mode:
                    self._draw_debug_hud(annotated)
                cv2.imshow(_WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('l'):
                    self.draw_landmarks = not self.draw_landmarks
                    print("Draw landmarks:", self.draw_landmarks)
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print("Debug mode:", self.debug_mode)
                elif key in self._PRESET_KEYS:
                    self._apply_preset(self._PRESET_KEYS[key])

            if frame_counter % 300 == 0:
                gc.collect()
            frame_counter += 1

        cv2.destroyAllWindows()
        self.hand_tracker.close()
