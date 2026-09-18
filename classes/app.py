import time
import os
import sys
import json
import threading
import collections
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


# MediaPipe hand landmark indices for the five finger tips. All hand-position
# decisions (active area, orchestra region, pitch mapping) use the centroid of
# these rather than of all 21 landmarks.
_TIP_LANDMARKS = (4, 8, 12, 16, 20)

# Region -> the zero-based index of the instrument it drives in an orchestra
# preset's "outputInstruments" list. The instrument argument sent with
# /setOutputRange is that entry's `id` (the token instrument JordanAI keys its
# output instruments by: 1 grand piano, 48 strings, 61 brass), never the index.
# Change this one mapping to re-assign which preset instrument a region drives.
_REGION_INSTRUMENT = {
    'piano':   0,
    'strings': 1,
    'brass':   2,
}

# Fallback instrument ranges, used when no orchestra preset file is found.
# Mirrors the "outputInstruments" defaults of the preset this was built
# against; the ids are General MIDI programs (1 grand piano, 48 string
# ensemble, 61 brass section), which is what fixes the order above.
_DEFAULT_INSTRUMENTS = [
    {'id': 1,  'low': 26, 'high': 89},   # piano
    {'id': 48, 'low': 33, 'high': 94},   # strings
    {'id': 61, 'low': 36, 'high': 92},   # brass
]

_ORCHESTRA_PRESET = "orchestra.json"


def _load_instrument_defaults(preset_path=None):
    """Map zero-based instrument index -> (id, low, high) from an orchestra preset.

    Reads "outputInstruments" out of a preset JSON file so a region's reset
    range and instrument id match whatever the receiving app is configured
    for. Falls back to _DEFAULT_INSTRUMENTS when there's no readable preset.
    """
    instruments = _DEFAULT_INSTRUMENTS
    path = preset_path or _resource_path(_ORCHESTRA_PRESET)
    try:
        with open(path) as fh:
            found = json.load(fh).get('outputInstruments')
        if isinstance(found, list) and found:
            instruments = found
            print(f"Orchestra instrument defaults loaded from {path}")
        else:
            print(f"No usable outputInstruments in {path}; using built-in defaults")
    except FileNotFoundError:
        if preset_path:      # only noisy when the user asked for a specific file
            print(f"Orchestra preset not found: {path}; using built-in defaults")
    except (OSError, ValueError) as e:
        print(f"Could not read orchestra preset {path}: {e}; using built-in defaults")

    ranges = {}
    for index, inst in enumerate(instruments):
        try:
            ranges[index] = (int(inst['id']), int(inst['low']), int(inst['high']))
        except (KeyError, TypeError, ValueError):
            continue
    return ranges

_MODE_COLORS = {
    'noop':   (160, 160, 160),
    'runs':   (80,  200, 80),
    'chords': (200, 200, 80),
    'faster': (80,  200, 200),
    'slower': (80,  80,  220),
}

def _jsonable(value):
    """Coerce numpy scalars and nested containers into JSON-serialisable values."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):       # before int: bool subclasses int
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    return value


class _OSCLog:
    """Forwarding wrapper that records every OSC message sent.

    Wrapping the client rather than hooking each call site means the log also
    picks up the sends made by GestureSender, which talks to the client
    directly. Exceptions still propagate, so the existing error printing at the
    call sites is unchanged.
    """

    def __init__(self, client, maxlen=40):
        self._client  = client
        self._lock    = threading.Lock()
        self._entries = collections.deque(maxlen=maxlen)
        self._seq     = 0

    def send_message(self, address, value):
        try:
            self._client.send_message(address, value)
        except Exception as e:
            self._record(address, value, error=str(e))
            raise
        self._record(address, value)

    def _record(self, address, value, error=None):
        if isinstance(value, (list, tuple)):
            args = [_jsonable(v) for v in value]
        elif value is None:
            args = []
        else:
            args = [_jsonable(value)]
        with self._lock:
            self._seq += 1
            self._entries.append({
                'seq':   self._seq,
                't':     time.time(),
                'addr':  address,
                'args':  args,
                'error': error,
            })

    def since(self, seq):
        """(entries newer than `seq`, current high-water mark).

        The browser passes back the last seq it saw, so each poll ships only
        new lines instead of the whole buffer.
        """
        with self._lock:
            if seq is None:
                entries = list(self._entries)
            else:
                entries = [e for e in self._entries if e['seq'] > seq]
            return entries, self._seq

    def clear(self):
        with self._lock:
            self._entries.clear()


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
                 preset="range",
                 orchestra_preset=None):
        self.interval = (-8, 8)

        # OSC client, wrapped so the web UI can show the same feed the
        # terminal prints. Both names point at the wrapper; osc_log stays
        # valid even if a caller swaps out osc_client.
        self.osc_log    = _OSCLog(udp_client.SimpleUDPClient(ip, port))
        self.osc_client = self.osc_log

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

        # Orchestra mode: the active region splits into a piano half (bottom)
        # and a brass/strings half (top), divided again into two columns.
        self.orchestra_mode = False
        # Fraction of the active region, so the divider stays valid when the
        # exclusion boundary moves. Above it = brass/strings, below = piano.
        self.orchestra_split_ratio = 0.5
        # Frame x dividing brass (left) from strings (right).
        self.orchestra_column_ratio = 0.5
        self._instr_state = {name: {'last_val': None, 'last_time': 0.0}
                             for name in ('brass', 'strings')}
        self._instr_defaults = _load_instrument_defaults(orchestra_preset)
        self.active_regions = []
        # Per-region "a hand was here last frame", so a reset fires on the
        # transition to empty rather than on every empty frame.
        self._engaged = {region: False for region in _REGION_INSTRUMENT}

        # Only top part of camera image is active for control
        self.active_area_ratio = 3 / 4
        self._boundary_dragging = False
        self._frame_height = None
        self._boundary_trackbar_ready = False

        # Newest frame published to the web UI. The gesture loop only swaps a
        # reference in here; JPEG encoding happens on the server thread so the
        # preview never gates detection.
        self._frame_lock   = threading.Lock()
        self._latest_frame = None
        self._latest_hands = []
        self._frame_seq    = 0

        self.fps = 0.0
        self._last_frame_time = None
        self._tint = None       # cached solid-colour buffer for the cv2 overlay


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

    @staticmethod
    def _hand_position(hand):
        """Hand position as the centroid of the five finger-tip landmarks."""
        return (float(np.mean([hand[i].x for i in _TIP_LANDMARKS])),
                float(np.mean([hand[i].y for i in _TIP_LANDMARKS])))

    def _active_hand_indices(self, results):
        """Return hands whose finger-tip centroid is above the exclusion boundary."""
        if not results or not results.hand_landmarks:
            return []
        return [
            i for i, hand in enumerate(results.hand_landmarks[:2])
            if self._hand_position(hand)[1] <= self.active_area_ratio
        ]

    def _orchestra_split_y(self):
        """Absolute frame y of the piano / brass-strings divider."""
        return self.active_area_ratio * self.orchestra_split_ratio

    def _hand_region(self, x, y):
        """'piano', 'brass' or 'strings' for a position inside the active area."""
        if not self.orchestra_mode:
            return 'piano'
        if y >= self._orchestra_split_y():
            return 'piano'
        return 'brass' if x < self.orchestra_column_ratio else 'strings'

    def _set_orchestra_split(self, frame_y):
        """Move the piano divider, given an absolute frame y."""
        if self.active_area_ratio <= 0:
            return
        frac = float(frame_y) / self.active_area_ratio
        self.orchestra_split_ratio = float(np.clip(frac, 0.1, 0.9))

    def _set_orchestra_column(self, frame_x):
        self.orchestra_column_ratio = float(np.clip(float(frame_x), 0.05, 0.95))

    def _set_orchestra_mode(self, on):
        if on == self.orchestra_mode:
            return
        self.orchestra_mode = on
        # The regions changed under the hands, so forget the throttle state.
        self._reset_range_throttle()
        print("Orchestra mode:", on)

    def _reset_range_throttle(self):
        """Forget the last sent values so a layout change sends promptly."""
        self.last_left = None
        self.last_right = None
        for st in self._instr_state.values():
            st['last_val'] = None
            st['last_time'] = 0.0

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

    def _instrument_id(self, region):
        """The preset id of the instrument a region drives, or None when the
        preset has no entry at that index."""
        default = self._instr_defaults.get(_REGION_INSTRUMENT[region])
        return default[0] if default else None

    def send_osc_message(self, left_val=None, right_val=None):
        """The piano's range, both hands. Outside orchestra mode the four-argument
        form: JordanAI applies it to its first active instrument, whatever id
        the preset gives it. In orchestra mode the piano's id leads, so the
        message can never land on another instrument."""
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

        args = [arg1, arg2, arg3, arg4]
        if self.orchestra_mode:
            piano_id = self._instrument_id('piano')
            if piano_id is None:
                return
            args = [piano_id] + args
        try:
            self.osc_client.send_message("/setOutputRange", args)
            print("OSC → /setOutputRange " + " ".join(str(a) for a in args))
        except Exception as e:
            print("OSC send error:", e)

    def _send_range(self, args, label):
        try:
            self.osc_client.send_message("/setOutputRange", args)
            print("OSC → /setOutputRange " +
                  " ".join(str(a) for a in args) + f"  ({label})")
        except Exception as e:
            print("OSC send error:", e)

    def send_instrument_range(self, instr, lo, hi):
        """Orchestra-mode range for one instrument: id, lo, hi, -1, -1."""
        if self.mode == 'pause':
            return
        instrument_id = self._instrument_id(instr)
        if instrument_id is None:
            return
        self._send_range([instrument_id, lo, hi, -1, -1], instr)

    def send_region_reset(self, region):
        """Reset one region's instrument to its preset default range.

        Outside orchestra mode the piano keeps the plain four-argument form;
        in orchestra mode every region carries its instrument id, matching
        how the live updates are sent.
        """
        if self.mode == 'pause':
            return
        default = self._instr_defaults.get(_REGION_INSTRUMENT[region])
        if default is None:
            return
        instrument_id, lo, hi = default
        if region == 'piano' and not self.orchestra_mode:
            args = [lo, hi, -1, -1]
        else:
            args = [instrument_id, lo, hi, -1, -1]
        self._send_range(args, f"{region} reset")

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
    # Frame processing (UI-independent)
    # ----------------------------

    def _tick_fps(self):
        now = time.perf_counter()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if not self.fps else self.fps * 0.9 + inst * 0.1
        self._last_frame_time = now

    def _step(self):
        """Run one capture -> track -> detect -> send cycle.

        Draws nothing, so both the OpenCV window and the web UI can drive it.
        Returns (frame, results, active_hand_indices); frame is None when the
        camera gave us nothing this round.
        """
        frame, results = self.hand_tracker.get_frame_and_landmarks(
            active_area_ratio=self.active_area_ratio)
        if frame is None:
            return None, None, []

        self._frame_height = frame.shape[0]
        self._tick_fps()

        active_hand_indices = self._active_hand_indices(results)
        self.hand_present = bool(active_hand_indices)

        # Gesture detection (runs every frame, reports every 500 ms).
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

        positions = [self._hand_position(results.hand_landmarks[i])
                     for i in active_hand_indices]
        self.active_regions = sorted({self._hand_region(x, y) for x, y in positions})

        self._update_hand_presence()
        self._update_region_engagement()
        if positions:
            self._update_output_range(positions)

        return frame, results, active_hand_indices

    def _update_region_engagement(self):
        """Reset a region's instrument once its last hand leaves it.

        Without this an instrument holds whatever range a hand last set — the
        piano would stay parked while both hands are up in the brass/strings
        half, and vice versa. Outside orchestra mode only 'piano' is ever
        engaged, so the other regions never fire.
        """
        live = set(self.active_regions)
        for region in self._engaged:
            if region in live:
                self._engaged[region] = True
            elif self._engaged[region]:
                self._engaged[region] = False
                self.send_region_reset(region)
                self._clear_region_throttle(region)

    def _clear_region_throttle(self, region):
        """Forget a region's last sent value so the next hand re-sends at once,
        rather than being compared against what it sent before leaving."""
        if region == 'piano':
            self.last_left = None
            self.last_right = None
        else:
            self._instr_state[region]['last_val'] = None
            self._instr_state[region]['last_time'] = 0.0

    def _update_hand_presence(self):
        """Send pause only after the hand has been absent for PAUSE_ABSENT_S,
        to avoid spurious pause/resume on brief detection dropouts."""
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

    def _update_output_range(self, positions):
        """Route hand positions to instrument channels and send their ranges.

        Outside orchestra mode every hand drives the piano channel, which keeps
        the original single-hand / two-hand behaviour.
        """
        if not self.orchestra_mode:
            self._update_piano(positions)
            return

        grouped = {'piano': [], 'brass': [], 'strings': []}
        for x, y in positions:
            grouped[self._hand_region(x, y)].append((x, y))

        if grouped['piano']:
            self._update_piano(grouped['piano'])
        for instr in ('brass', 'strings'):
            if grouped[instr]:
                self._update_instrument(instr, grouped[instr])

    def _update_instrument(self, instr, positions):
        """Send one instrument's range from the hands inside its column.

        x maps across the whole frame rather than across the column, so each
        column reaches its own half of the pitch range: brass the lower half,
        strings the upper.
        """
        x = float(np.mean([p[0] for p in positions]))
        val = self.map_hand_x_to_val(x)

        st = self._instr_state[instr]
        now = time.time()
        if now - st['last_time'] < self.osc_interval:
            return
        if (st['last_val'] is not None and
                abs(val - st['last_val']) <= self.change_threshold):
            return
        st['last_val']  = val
        st['last_time'] = now
        self.send_instrument_range(instr, val + self.interval[0],
                                   val + self.interval[1])

    def _update_piano(self, positions):
        # Single-hand -> both channels
        if len(positions) == 1:
            x, y = positions[0]
            mapped = self.map_hand_x_to_val(x)
            self.left_val = mapped
            self.right_val = mapped

        # Two hands -> left/right
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


    # ----------------------------
    # Web UI interface
    # ----------------------------

    def _publish_frame(self, frame, results, active_hand_indices):
        """Hand the newest frame (and landmark coords) to the web UI.

        Nothing is drawn here: the browser composites the boundary, HUD and
        landmarks, so the preview costs this loop one reference swap.
        """
        hands = []
        if self.draw_landmarks and results and results.hand_landmarks:
            for i in active_hand_indices:
                hands.append([[round(lm.x, 4), round(lm.y, 4)]
                              for lm in results.hand_landmarks[i]])
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_hands = hands
            self._frame_seq   += 1

    def latest_frame(self):
        """(frame, seq) for the newest published frame; seq lets the streamer
        skip re-encoding a frame it has already sent."""
        with self._frame_lock:
            return self._latest_frame, self._frame_seq

    def web_state(self, osc_since=None):
        """Snapshot of everything the browser renders as HUD.

        `osc_since` is the last OSC seq the browser has, so only newer log
        lines get shipped.
        """
        mode, intensity = self.gesture_result
        sender     = self.gesture_sender
        sent_mode  = sender._last_sent_mode or 'none'
        sent_level = sender._last_sent_level
        with self._frame_lock:
            hands = self._latest_hands
        osc_entries, osc_seq = self.osc_log.since(osc_since)
        state = {
            'preset':         self.preset,
            'presets':        list(self._PRESETS),
            'mode':           mode,
            'intensity':      round(float(intensity), 3),
            'last_sent':      sent_mode if sent_level is None else f"{sent_mode} L{sent_level}",
            'active_area':    round(self.active_area_ratio, 3),
            'hand_present':   self.hand_present,
            'draw_landmarks': self.draw_landmarks,
            'debug':          self.debug_mode,
            'fps':            round(self.fps, 1),
            'left_val':       self.left_val,
            'right_val':      self.right_val,
            'interval':       list(self.interval),
            'hands':          hands,
            'osc':            osc_entries,
            'osc_seq':        osc_seq,
            'orchestra':        self.orchestra_mode,
            'orchestra_split':  round(self._orchestra_split_y(), 4),
            'orchestra_column': round(self.orchestra_column_ratio, 4),
            'active_regions':   list(self.active_regions),
        }
        if self.debug_mode:
            det = self.gesture_detector
            state['debug_scores'] = _jsonable(det.debug_scores)
            state['thresholds'] = {
                'run_tip_artic':     det.T_RUN_TIP_ARTIC,
                'chord_tip_disp':    det.T_CHORD_TIP_DISP,
                'run_ext':           det.T_RUN_EXT,
                'index_ext':         det.T_INDEX_EXT,
                'others_dist_ratio': det.T_OTHERS_DIST_RATIO,
                'index_hold':        det.T_INDEX_HOLD,
                'rotate':            det.T_ROTATE,
                'open':              det.T_OPEN,
            }
        return state

    def apply_control(self, payload):
        """Apply a control message from the browser and return the new state.

        Raises ValueError on an unknown preset so the handler can answer 400.
        """
        if 'preset' in payload:
            name = payload['preset']
            if name not in self._PRESETS:
                raise ValueError(f"unknown preset: {name!r}")
            self._apply_preset(name)
        if 'active_area' in payload:
            self._set_active_area_ratio(float(payload['active_area']))
        if 'draw_landmarks' in payload:
            self.draw_landmarks = bool(payload['draw_landmarks'])
            print("Draw landmarks:", self.draw_landmarks)
        if 'debug' in payload:
            self.debug_mode = bool(payload['debug'])
            print("Debug mode:", self.debug_mode)
        if 'orchestra' in payload:
            self._set_orchestra_mode(bool(payload['orchestra']))
        if 'orchestra_split' in payload:
            self._set_orchestra_split(payload['orchestra_split'])
        if 'orchestra_column' in payload:
            self._set_orchestra_column(payload['orchestra_column'])
        if payload.get('clear_osc'):
            self.osc_log.clear()
        if payload.get('quit'):
            self.running = False
        # Deliberately omits OSC entries: the browser tracks those through its
        # own polling seq, and returning them here would double them up.
        return self.web_state(osc_since=self.osc_log.since(None)[1])


    # ----------------------------
    # OpenCV rendering
    # ----------------------------

    def _strip_tint(self, shape):
        """Cached solid-colour buffer for the exclusion-region blend."""
        if self._tint is None or self._tint.shape != shape:
            self._tint = np.empty(shape, dtype=np.uint8)
            self._tint[:] = (100, 100, 255)
        return self._tint

    def _render_cv2(self, frame, results, active_hand_indices):
        """Draw the overlay and HUDs onto `frame`, in place."""
        h, w = frame.shape[:2]
        inactive_y = int(h * self.active_area_ratio)

        # Tint only the excluded strip. Copying the whole frame and blending
        # all of it cost more than everything else drawn here combined.
        if inactive_y < h:
            strip = frame[inactive_y:, :]
            cv2.addWeighted(strip, 0.75, self._strip_tint(strip.shape), 0.25, 0,
                            dst=strip)
        cv2.line(frame, (0, inactive_y), (w, inactive_y), (0, 0, 255), 2)
        cv2.putText(frame,
                    f"Active {self.active_area_ratio:.0%} - use slider or drag red area",
                    (12, inactive_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if self.orchestra_mode:
            self._draw_orchestra_regions(frame, w, h, inactive_y)

        if self.draw_landmarks:
            HandLandmarkDrawer.draw_landmarks(
                frame, results, active_hand_indices, copy=False)
        self._draw_gesture_hud(frame, *self.gesture_result)
        self._draw_preset_hud(frame)
        if self.debug_mode:
            self._draw_debug_hud(frame)

    def _draw_orchestra_regions(self, frame, w, h, inactive_y):
        """Divider bars and region labels for orchestra mode."""
        BAR   = (0, 200, 255)     # amber
        LIVE  = (80, 255, 120)    # a region currently holding a hand
        split_y = int(h * self._orchestra_split_y())
        col_x   = int(w * self.orchestra_column_ratio)

        cv2.line(frame, (0, split_y), (w, split_y), BAR, 2)
        cv2.line(frame, (col_x, 0), (col_x, split_y), BAR, 2)

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        def label(text, x, y, region):
            color = LIVE if region in self.active_regions else BAR
            cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

        label("brass",   14,         split_y - 12, 'brass')
        label("strings", col_x + 14, split_y - 12, 'strings')
        label("piano",   14,         inactive_y - 28, 'piano')

    def _handle_key(self, key):
        if key == ord('q'):
            self.running = False
        elif key == ord('l'):
            self.draw_landmarks = not self.draw_landmarks
            print("Draw landmarks:", self.draw_landmarks)
        elif key == ord('d'):
            self.debug_mode = not self.debug_mode
            print("Debug mode:", self.debug_mode)
        elif key == ord('o'):
            self._set_orchestra_mode(not self.orchestra_mode)
        elif key in self._PRESET_KEYS:
            self._apply_preset(self._PRESET_KEYS[key])


    # ----------------------------
    # Main Loop
    # ----------------------------

    def run(self, ui="web", http_host="127.0.0.1", http_port=8765,
            open_browser=True):
        if ui == "web":
            return self._run_web(http_host, http_port, open_browser)
        return self._run_cv2()

    def _run_cv2(self):
        print("Running OSC Gesture App (OpenCV window).")

        frame_counter = 0
        self._create_camera_window()

        try:
            while self.running:
                frame, results, active_hand_indices = self._step()
                if frame is not None:
                    self._render_cv2(frame, results, active_hand_indices)
                    cv2.imshow(_WINDOW_NAME, frame)
                    self._handle_key(cv2.waitKey(1) & 0xFF)

                if frame_counter % 300 == 0:
                    gc.collect()
                frame_counter += 1
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            cv2.destroyAllWindows()
            self.hand_tracker.close()

    def _run_web(self, http_host, http_port, open_browser):
        from .web_ui import WebUI

        web = WebUI(self, host=http_host, port=http_port)
        web.start(open_browser=open_browser)

        frame_counter = 0
        try:
            while self.running:
                frame, results, active_hand_indices = self._step()
                if frame is not None:
                    self._publish_frame(frame, results, active_hand_indices)

                if frame_counter % 300 == 0:
                    gc.collect()
                frame_counter += 1
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            web.stop()
            self.hand_tracker.close()
