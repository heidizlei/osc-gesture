import numpy as np

TIPS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
MCPS  = [2, 5, 9, 13, 17]    # corresponding MCP/IP joints
MCPS4 = [5, 9, 13, 17]       # palm MCPs (no thumb)


class _HandState:
    """Per-hand stateful tracking for slower and faster gestures."""
    __slots__ = ('t_open', 'fist_start_t', 'fist_pending', 'fist_intensity',
                 'index_since', 'index_last_ok', 'mean_ext')

    def __init__(self):
        self.t_open         = None   # last t where mean_ext > T_OPEN
        self.fist_start_t   = None   # when current fist epoch began
        self.fist_pending   = False  # fist confirmed, not yet reported
        self.fist_intensity = 0.0
        self.index_since    = None   # when index-extended pose began (for hold timer)
        self.index_last_ok  = None   # last t where index_ok was True (for dropout tolerance)
        self.mean_ext       = 0.0    # latest mean finger extension (for debug display)


class GestureDetector:
    """
    Frame-rate-independent gesture detector for four modes:
      runs, chords, faster, slower

    Usage:
        det = GestureDetector()
        # each camera frame:
        mode, intensity = det.update(world_landmarks_2x21x3, timestamp_seconds)
    """

    REPORT_INTERVAL = 0.5   # s — how often mode/intensity refreshes
    MAX_WINDOW      = 2.0   # s — longest look-back (needed for slower)

    # Detection thresholds
    T_CHORD_MCP        = 0.25   # m/s — mcp_speed_wr (palm pivot) floor for chords
    T_CHORD_TIP_ARTIC  = 0.30   # m/s — tip_artic (finger flex) floor for chords
    T_RUN_TIP_ARTIC    = 0.15   # m/s — tip_artic floor for runs
    T_RUN_EXT          = 0.40   # /s  — ext_change_rate floor for runs
    T_ROTATE      = 1.0    # rad/s
    T_INDEX_EXT   = 0.80   # extension ratio — index counts as "extended"
    T_OTHERS_CURL = 0.55   # extension ratio — other fingers must be below this
    T_FIST        = 0.58   # mean extension — hand counts as fist
    T_OPEN        = 0.70   # mean extension — hand counts as open
    T_FIST_HOLD   = 0.10   # s — must hold fist this long to confirm
    T_INDEX_HOLD    = 0.333  # s — index must be extended this long to arm faster
    T_INDEX_DROPOUT = 0.30   # s — tolerate brief dropouts in index-extended pose
    T_INDEX_EXT     = 0.70   # lowered from 0.80 — recording shows idx_ext ~0.75-0.94

    # Intensity normalisation
    NORM_RUN    = 0.50   # m/s  — tip_artic at which run_intensity = 1.0
    NORM_CHORD  = 0.50   # m/s  — mcp_speed_wr at which chord_intensity = 1.0
    NORM_ROTATE = 4.0    # rad/s
    REF_CLENCH  = 1.0    # s    — clench at this duration → slower_intensity = 1.0

    def __init__(self):
        self._buf         = []                              # [(t, wl_2x21x3), ...]
        self._hand_states = [_HandState(), _HandState()]
        self._last_report = 0.0
        self._result      = ('noop', 0.0)
        self.debug_scores = [{}, {}]                        # populated each _detect() call

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, world_landmarks, t):
        """
        world_landmarks : np.ndarray (2, 21, 3), float32, NaN for missing hands
        t               : float, seconds since some fixed epoch (e.g. time.time())
        Returns (mode: str, intensity: float in [0, 1]).
        Mode is refreshed every REPORT_INTERVAL seconds.
        """
        self._buf.append((t, world_landmarks.copy()))
        cutoff = t - self.MAX_WINDOW
        while len(self._buf) > 1 and self._buf[0][0] < cutoff:
            self._buf.pop(0)

        for hi in range(2):
            h = world_landmarks[hi]
            if not np.isnan(h[0, 0]):
                self._tick_hand(hi, h, t)

        if t - self._last_report >= self.REPORT_INTERVAL:
            self._last_report = t
            self._result = self._detect(t)

        return self._result

    def reset(self):
        self._buf.clear()
        self._hand_states = [_HandState(), _HandState()]
        self._last_report = 0.0
        self._result = ('noop', 0.0)

    # ------------------------------------------------------------------
    # Per-frame state machine
    # ------------------------------------------------------------------

    def _extension(self, h):
        """Extension ratio for all 5 fingers. h: (21, 3)"""
        ext = np.empty(5)
        for i in range(5):
            tv = h[TIPS[i]] - h[MCPS[i]]
            mv = h[MCPS[i]] - h[0]
            ext[i] = np.linalg.norm(tv) / (np.linalg.norm(mv) + 1e-6)
        return ext

    def _tick_hand(self, hi, h, t):
        s   = self._hand_states[hi]
        ext = self._extension(h)
        me  = float(ext.mean())
        s.mean_ext = me

        index_ok = (ext[1] > self.T_INDEX_EXT and
                    ext[2] < self.T_OTHERS_CURL and
                    ext[3] < self.T_OTHERS_CURL and
                    ext[4] < self.T_OTHERS_CURL)

        # --- Slower state machine (hysteresis: only resets on full open) ---
        # Skip when index is extended — the two gestures are physically exclusive.
        if not index_ok:
            if me > self.T_OPEN:
                s.t_open       = t
                s.fist_start_t = None
                # leave fist_pending intact so _detect() can consume it this tick
            elif me < self.T_FIST:
                if s.fist_start_t is None:
                    s.fist_start_t = t
                elif not s.fist_pending and (t - s.fist_start_t >= self.T_FIST_HOLD):
                    if s.t_open is not None:
                        dur = max(s.fist_start_t - s.t_open, 0.05)
                        s.fist_intensity = float(np.clip(self.REF_CLENCH / dur, 0.0, 1.0))
                        s.fist_pending = True
            # zone [T_FIST, T_OPEN]: transitional — maintain fist_start_t

        # --- Faster: track index-extended hold with dropout tolerance ---
        if index_ok:
            s.index_last_ok = t
            if s.index_since is None:
                s.index_since = t
        elif s.index_last_ok is not None and (t - s.index_last_ok) > self.T_INDEX_DROPOUT:
            s.index_since   = None
            s.index_last_ok = None

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    def _hand_window(self, hi, window_s):
        """(times, lms) for hand hi within the last window_s seconds."""
        if not self._buf:
            return [], []
        t_now  = self._buf[-1][0]
        cutoff = t_now - window_s
        times, lms = [], []
        for t, wl in self._buf:
            if t < cutoff:
                continue
            h = wl[hi]
            if not np.isnan(h[0, 0]):
                times.append(t)
                lms.append(h)
        return times, lms

    def _motion_features(self, times, lms):
        """
        Returns dict(tip_artic, mcp_speed, ext_change) from a frame sequence,
        or None if fewer than 2 frames are available.

        tip_artic   — mean fingertip speed relative to its OWN MCP joint (m/s).
                       Pure finger articulation, invariant to palm/wrist motion.
        mcp_speed   — speed of palm centroid in wrist-relative frame (m/s).
                       Captures the palm pivot motion of chord gestures.
        ext_change  — rate of change of finger extension ratios (1/s).
                       Articulation magnitude — fingers flexing/extending.
        """
        if len(times) < 2:
            return None
        h  = np.stack(lms)
        ts = np.array(times)
        dt = np.diff(ts)

        # tip_artic — tips relative to their own MCPs (pure finger articulation)
        tip_rel_mcp = h[:, TIPS] - h[:, MCPS]   # (N, 5, 3)
        tip_artic = float(
            np.linalg.norm(
                np.diff(tip_rel_mcp, axis=0) / dt[:, None, None], axis=2
            ).mean()
        )

        # mcp_speed — palm centroid in wrist-relative frame (chord pivot signal)
        mcp_wr = h[:, MCPS4] - h[:, [0]]   # (N, 4, 3)
        mcp_speed = float(
            np.linalg.norm(
                np.diff(mcp_wr.mean(axis=1), axis=0) / dt[:, None], axis=1
            ).mean()
        )

        ext = np.zeros((len(h), 5))
        for i in range(5):
            tv = h[:, TIPS[i]] - h[:, MCPS[i]]
            mv = h[:, MCPS[i]] - h[:, 0]
            ext[:, i] = np.linalg.norm(tv, axis=1) / (np.linalg.norm(mv, axis=1) + 1e-6)
        ext_change = float((np.abs(np.diff(ext, axis=0)) / dt[:, None]).mean())

        return {'tip_artic': tip_artic, 'mcp_speed': mcp_speed, 'ext_change': ext_change}

    def _palm_angular_vel(self, times, lms):
        """Mean |angular velocity of palm normal around forearm axis| in rad/s."""
        if len(times) < 2:
            return 0.0
        h  = np.stack(lms)
        ts = np.array(times)
        dt = np.diff(ts)

        fa = (h[:, 5] + h[:, 17]) * 0.5 - h[:, 0]
        fa = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-6)
        la = h[:, 5] - h[:, 17]
        la = la / (np.linalg.norm(la, axis=1, keepdims=True) + 1e-6)
        pn = np.cross(fa, la)
        pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-6)

        omega = (np.cross(pn[:-1], pn[1:]) * fa[:-1]).sum(axis=1) / (dt + 1e-6)
        return float(np.abs(omega).mean())

    # ------------------------------------------------------------------
    # Detection (called every REPORT_INTERVAL)
    # ------------------------------------------------------------------

    def _detect(self, now):
        # Build per-hand debug snapshot before the main detection loop
        dbg = []
        for hi in range(2):
            s = self._hand_states[hi]
            times, lms = self._hand_window(hi, 0.5)
            feat = self._motion_features(times, lms)
            ang_vel = self._palm_angular_vel(times, lms) if len(times) >= 2 else 0.0
            index_hold = (now - s.index_since) if s.index_since is not None else 0.0
            dbg.append({
                'tip_artic':      feat['tip_artic']  if feat else None,
                'mcp_speed':      feat['mcp_speed']  if feat else None,
                'ext_change':     feat['ext_change'] if feat else None,
                'ang_vel':        ang_vel,
                'index_hold':     index_hold,
                'mean_ext':       s.mean_ext,
                'fist_pending':   s.fist_pending,
                'fist_intensity': s.fist_intensity,
                'is_chord': bool(feat and
                                 feat['mcp_speed']  >= self.T_CHORD_MCP and
                                 feat['tip_artic']  >= self.T_CHORD_TIP_ARTIC),
                'is_run':   bool(feat and
                                 feat['tip_artic']  >= self.T_RUN_TIP_ARTIC and
                                 feat['ext_change'] >= self.T_RUN_EXT),
            })
        self.debug_scores = dbg

        best_mode, best_intensity = 'noop', 0.0

        for hi in range(2):
            s = self._hand_states[hi]

            # 1. Slower — one-shot per clench; consume the pending flag
            if s.fist_pending:
                if s.fist_intensity > best_intensity:
                    best_mode, best_intensity = 'slower', s.fist_intensity
                s.fist_pending = False

            # 2. Faster — index held long enough + rotation rate
            if s.index_since is not None and now - s.index_since >= self.T_INDEX_HOLD:
                times, lms = self._hand_window(hi, 0.5)
                av = self._palm_angular_vel(times, lms)
                if av >= self.T_ROTATE:
                    intensity = float(np.clip(av / self.NORM_ROTATE, 0.0, 1.0))
                    if intensity > best_intensity:
                        best_mode, best_intensity = 'faster', intensity

            # 3. Chords / Runs — skip if index is in extended-hold mode; the wrist
            #    rotation during faster inherently elevates mcp_speed_wr.
            if s.index_since is not None:
                continue

            times, lms = self._hand_window(hi, 0.5)
            feat = self._motion_features(times, lms)
            if feat is None:
                continue

            # Chord: palm pivots (high mcp_speed) AND fingers flex with it (high tip_artic).
            # The conjunction is what separates chords from vigorous runs.
            is_chord = (feat['mcp_speed'] >= self.T_CHORD_MCP and
                        feat['tip_artic'] >= self.T_CHORD_TIP_ARTIC)
            if is_chord:
                intensity = float(np.clip(feat['mcp_speed'] / self.NORM_CHORD, 0.0, 1.0))
                if intensity > best_intensity:
                    best_mode, best_intensity = 'chords', intensity
            elif (feat['tip_artic'] >= self.T_RUN_TIP_ARTIC and
                  feat['ext_change'] >= self.T_RUN_EXT):
                intensity = float(np.clip(feat['tip_artic'] / self.NORM_RUN, 0.0, 1.0))
                if intensity > best_intensity:
                    best_mode, best_intensity = 'runs', intensity

        return best_mode, best_intensity


# ------------------------------------------------------------------
# Recording replay utility
# ------------------------------------------------------------------

def replay(path, verbose=True):
    """
    Run the detector against a saved .npz recording.
    Returns list of (timestamp, mode, intensity) at every REPORT_INTERVAL tick.

    Example:
        results = replay("recordings/runs_2026-05-15T16-46-03.024.npz")
    """
    clip   = np.load(path, allow_pickle=True)
    ts     = clip["timestamps"].astype(float)
    wl     = clip["world_landmarks"].astype(float)
    label  = str(clip["mode"][0])

    det     = GestureDetector()
    reports = []

    for i in range(len(ts)):
        mode, intensity = det.update(wl[i], ts[i])
        # collect only frames where a report was just emitted
        if abs(ts[i] - det._last_report) < 1e-6 or i == len(ts) - 1:
            reports.append((float(ts[i]), mode, intensity))

    if verbose:
        print(f"\n{'='*50}")
        print(f"Recording : {path.split('/')[-1]}")
        print(f"Label     : {label}")
        print(f"Duration  : {ts[-1]:.1f}s   frames: {len(ts)}")
        print(f"{'─'*50}")
        for t, mode, intensity in reports:
            marker = ' <--' if mode != 'noop' else ''
            print(f"  t={t:6.2f}   {mode:<8s}  {intensity:.3f}{marker}")

    return label, reports
