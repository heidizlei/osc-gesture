import numpy as np

TIPS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
MCPS  = [2, 5, 9, 13, 17]    # corresponding MCP/IP joints
MCPS4 = [5, 9, 13, 17]       # palm MCPs (no thumb)


class _HandState:
    """Per-hand stateful tracking for slower and faster gestures."""
    __slots__ = ('t_open', 'fist_start_t', 'fist_pending', 'fist_intensity',
                 'index_since', 'index_last_ok', 'mean_ext',
                 'index_ext_since')

    def __init__(self):
        self.t_open          = None   # last t where mean_ext > T_OPEN
        self.fist_start_t    = None   # when current fist epoch began
        self.fist_pending    = False  # fist confirmed, not yet reported
        self.fist_intensity  = 0.0
        self.index_since     = None   # when full faster pose began (index out + others curled)
        self.index_last_ok   = None   # last t where index_ok was True (for dropout tolerance)
        self.mean_ext        = 0.0    # latest mean finger extension (for debug display)
        self.index_ext_since = None   # when index alone became extended (runs suppression gate)


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
    T_CHORD_MCP        = 0.10   # m/s — mcp_speed_wr floor for chords; filters rigid wrist
                           #     translation (no palm pivot → near-zero mcp_speed)
    T_CHORD_TIP_DISP        = 0.90   # normalised (tip_disp/hand_size) floor for chords
    T_CHORD_TIP_DISP_IN_RUNS = 1.10  # raised threshold when already in runs mode
    T_RUN_OPEN              = 0.65   # mean extension — hand must be this open for runs to fire
    T_RUN_TIP_ARTIC         = 0.07   # m/s — tip_artic floor for runs (also when coming from chords)
    T_RUN_EXT          = 0.40   # /s  — ext_change_rate floor for runs
    T_ROTATE      = 1.0    # rad/s
    T_INDEX_EXT       = 0.60   # extension ratio — index counts as extended (lowered for new angle)
    T_OTHERS_DIST_RATIO = 1.0  # max (others_tip_dist / hand_size) — normalised, camera-angle independent
                               # faster (curled): p90=0.96; noop/runs (extended): p10=1.24
    T_FIST        = 0.58   # mean extension — hand counts as fist
    T_OPEN        = 0.70   # mean extension — hand counts as open
    T_FIST_HOLD   = 0.10   # s — must hold fist this long to confirm
    T_INDEX_HOLD      = 0.333  # s — full faster pose must be held to arm faster
    T_INDEX_DROPOUT   = 0.30   # s — tolerate brief dropouts in index-extended pose
    T_INDEX_SUPPRESS  = 0.10   # s — suppress runs after index has been extended this long

    # Intensity normalisation
    NORM_RUN         = 0.25   # m/s — palm-normal projected tip_artic at which run_intensity = 1.0
    NORM_CHORD_MCP   = 0.50   # m/s — mcp_speed at which chord_intensity = 1.0
    NORM_ROTATE = 8.0    # rad/s — scaled up for index-tip angular velocity range
    MIN_CLENCH  = 0.10   # s    — fastest clench → intensity = 1.0
    MAX_CLENCH  = 0.60   # s    — slowest clench → intensity = 0.0 (observed range 0.07–0.58s)

    # Vertical movement suppression (image-space wrist y, [0,1] normalised screen)
    T_VERTICAL_VEL = 0.40  # screen-heights/s — force noop above this wrist vertical speed

    # Hysteresis
    NOOP_DEAD_TIME = 0.30  # s — hold last active mode this long before dropping to noop
    SWITCH_HOLD    = 0.50  # s — new mode must be consistently detected this long before
                           #     switching away from an existing active mode

    def __init__(self):
        self._buf         = []                              # [(t, wl_2x21x3), ...]
        self._hand_states = [_HandState(), _HandState()]
        self._last_report = 0.0
        self._result      = ('noop', 0.0)
        self.debug_scores = [{}, {}]                        # populated each _detect() call
        # Hysteresis state
        self._active_mode    = 'noop'     # currently committed mode
        self._last_active_t  = 0.0        # last time a non-noop mode was returned
        self._last_active_result = ('noop', 0.0)
        self._pending_mode   = 'noop'     # candidate mode being evaluated for switch
        self._pending_since  = 0.0        # when pending mode first appeared

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, world_landmarks, t, wrist_y=None):
        """
        world_landmarks : np.ndarray (2, 21, 3), float32, NaN for missing hands
        t               : float, seconds since some fixed epoch (e.g. time.time())
        wrist_y         : np.ndarray (2,) normalised image-space wrist y [0,1], NaN if missing
        Returns (mode: str, intensity: float in [0, 1]).
        Mode is refreshed every REPORT_INTERVAL seconds.
        """
        self._buf.append((t, world_landmarks.copy(),
                          wrist_y.copy() if wrist_y is not None else np.full(2, np.nan)))
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
        self._hand_states       = [_HandState(), _HandState()]
        self._last_report       = 0.0
        self._result            = ('noop', 0.0)
        self._active_mode       = 'noop'
        self._last_active_t     = 0.0
        self._last_active_result = ('noop', 0.0)
        self._pending_mode      = 'noop'
        self._pending_since     = 0.0

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

        # Normalised tip-to-wrist for middle/ring/pinky: ratio vs hand size is
        # camera-angle independent (absolute distances vary with camera angle).
        hand_size = float(np.linalg.norm(h[MCPS4] - h[[0]], axis=1).mean())
        others_dist_ok = (np.linalg.norm(h[12] - h[0]) / (hand_size + 1e-6) < self.T_OTHERS_DIST_RATIO and
                          np.linalg.norm(h[16] - h[0]) / (hand_size + 1e-6) < self.T_OTHERS_DIST_RATIO and
                          np.linalg.norm(h[20] - h[0]) / (hand_size + 1e-6) < self.T_OTHERS_DIST_RATIO)
        index_ok = ext[1] > self.T_INDEX_EXT and others_dist_ok

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
                        # Slow clench (long duration) → high intensity; fast → low
                        s.fist_intensity = float(np.clip(
                            (dur - self.MIN_CLENCH) / (self.MAX_CLENCH - self.MIN_CLENCH),
                            0.0, 1.0))
                        s.fist_pending = True
            # zone [T_FIST, T_OPEN]: transitional — maintain fist_start_t

        # --- Index extension tracker (runs suppression) ---
        # Uses a higher threshold than the faster pose to avoid suppressing runs when
        # the index naturally extends through its strike arc.
        if ext[1] > 0.85:
            if s.index_ext_since is None:
                s.index_ext_since = t
        else:
            s.index_ext_since = None

        # --- Faster: track full pose (index out + others curled) with dropout tolerance ---
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
        for t, wl, *_ in self._buf:
            if t < cutoff:
                continue
            h = wl[hi]
            if not np.isnan(h[0, 0]):
                times.append(t)
                lms.append(h)
        return times, lms

    def _motion_features(self, times, lms):
        """
        Returns dict(tip_artic, tip_disp, mcp_speed, ext_change), or None if < 2 frames.

        tip_artic   — mean fingertip speed relative to its OWN MCP (m/s).
                       Pure finger articulation; used for run detection.
        tip_disp    — peak-to-peak tip displacement normalised by wrist-to-MCP hand size.
                       Dimensionless ratio, hand-size independent. Chord petting sweeps
                       tips through a large arc (>1× hand size); runs stay local (<0.9×).
        mcp_speed   — speed of palm centroid in wrist-relative frame (m/s).
                       Palm pivot signal; used as the primary chord threshold.
        ext_change  — mean rate of change of finger extension ratios (1/s).
                       Discriminates active runs from idle drift.
        """
        if len(times) < 2:
            return None
        h  = np.stack(lms)
        ts = np.array(times)
        dt = np.diff(ts)

        # tip_artic — velocity of tips relative to their own MCPs, projected onto the
        # palm normal. This isolates finger-strike motion (perpendicular to palm) and
        # rejects horizontal wrist translation, which has no palm-normal component.
        tip_rel_mcp = h[:, TIPS] - h[:, MCPS]               # (N, 5, 3)
        vel         = np.diff(tip_rel_mcp, axis=0) / dt[:, None, None]  # (N-1, 5, 3)
        fa  = (h[:, 5] + h[:, 17]) * 0.5 - h[:, 0]
        la  = h[:, 5] - h[:, 17]
        fa  = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-6)
        la  = la / (np.linalg.norm(la, axis=1, keepdims=True) + 1e-6)
        pn  = np.cross(fa, la)
        pn  = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-6)  # (N, 3)
        pn_mid = pn[:-1]                                                 # (N-1, 3)
        proj = (vel * pn_mid[:, None, :]).sum(axis=2)                   # (N-1, 5) signed
        tip_artic = float(np.abs(proj).mean())

        # tip_disp — peak-to-peak tip displacement normalised by wrist-to-MCP hand size.
        # Dividing by hand size makes the threshold hand-size independent.
        tip_wr   = h[:, TIPS] - h[:, [0]]                        # (N, 5, 3)
        pn_mean  = pn.mean(axis=0)
        pn_mean  = pn_mean / (np.linalg.norm(pn_mean) + 1e-6)
        proj_pos = (tip_wr * pn_mean[None, None, :]).sum(axis=2)  # (N, 5) scalar projection
        tip_disp_m   = float((proj_pos.max(axis=0) - proj_pos.min(axis=0)).mean())
        hand_size    = float(np.linalg.norm(h[:, MCPS4] - h[:, [0]], axis=2).mean())
        tip_disp     = tip_disp_m / (hand_size + 1e-6)           # dimensionless ratio

        # mcp_speed — palm centroid in wrist-relative frame (chord pivot signal)
        mcp_wr = h[:, MCPS4] - h[:, [0]]         # (N, 4, 3)
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

        return {'tip_artic': tip_artic, 'tip_disp': tip_disp,
                'mcp_speed': mcp_speed, 'ext_change': ext_change}

    def _rotation_angular_vel(self, times, lms):
        """
        Max of palm-normal and index-tip angular velocities around the forearm axis (rad/s).
        Index tip has ~2-3x larger lever arm than palm normal, giving a stronger signal.
        Taking the max means either signal alone is sufficient to detect the gesture.
        """
        if len(times) < 2:
            return 0.0
        h  = np.stack(lms)
        ts = np.array(times)
        dt = np.diff(ts)

        fa = (h[:, 5] + h[:, 17]) * 0.5 - h[:, 0]
        fa = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-6)

        # palm normal angular velocity
        la = h[:, 5] - h[:, 17]
        la = la / (np.linalg.norm(la, axis=1, keepdims=True) + 1e-6)
        pn = np.cross(fa, la)
        pn = pn / (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-6)
        palm_omega = np.abs(
            (np.cross(pn[:-1], pn[1:]) * fa[:-1]).sum(axis=1) / (dt + 1e-6)
        ).mean()

        # index tip angular velocity around forearm axis
        tip_rel = h[:, 8] - h[:, 5]                                   # index tip rel to MCP
        tip_perp = tip_rel - (tip_rel * fa).sum(axis=1, keepdims=True) * fa  # project out fa
        tip_perp = tip_perp / (np.linalg.norm(tip_perp, axis=1, keepdims=True) + 1e-6)
        index_omega = np.abs(
            (np.cross(tip_perp[:-1], tip_perp[1:]) * fa[:-1]).sum(axis=1) / (dt + 1e-6)
        ).mean()

        return float(max(palm_omega, index_omega))

    def _chord_frequency(self, times, lms):
        """
        Estimate petting frequency (Hz) from zero-crossings of MCP centroid
        velocity projected onto the palm normal. Uses a 1s look-back window.
        Returns 0.0 if fewer than 2 crossings are found.
        """
        if len(times) < 3:
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
        pn_mean = pn.mean(axis=0)
        pn_mean = pn_mean / (np.linalg.norm(pn_mean) + 1e-6)

        mcp_wr  = h[:, MCPS4] - h[:, [0]]
        mcp_vel = np.diff(mcp_wr.mean(axis=1), axis=0) / dt[:, None]  # (N-1, 3)
        signed  = (mcp_vel * pn_mean).sum(axis=1)                      # (N-1,) signed

        signs = np.sign(signed)
        signs[signs == 0] = 1
        crossings = np.where(np.diff(signs) != 0)[0]
        if len(crossings) < 2:
            return 0.0
        half_periods = np.diff(ts[1:][crossings])
        return float(1.0 / (2.0 * half_periods.mean() + 1e-6))

    # ------------------------------------------------------------------
    # Detection (called every REPORT_INTERVAL)
    # ------------------------------------------------------------------

    def _wrist_vertical_vel(self):
        """Max |dy/dt| of wrist in image-space across both hands (screen-heights/s)."""
        if len(self._buf) < 2:
            return 0.0
        # Use last 500ms of image-space wrist y values
        t_now  = self._buf[-1][0]
        cutoff = t_now - 0.5
        ts, ys = [], [[], []]
        for t, _, wy in self._buf:
            if t < cutoff:
                continue
            ts.append(t)
            for hi in range(2):
                ys[hi].append(float(wy[hi]))
        if len(ts) < 2:
            return 0.0
        dt  = np.diff(ts)
        max_vel = 0.0
        for hi in range(2):
            y_arr = np.array(ys[hi])
            valid = ~np.isnan(y_arr)
            if valid.sum() < 2:
                continue
            # only compute velocity for consecutive valid frames
            dy  = np.diff(y_arr)
            vel = np.abs(dy) / (dt + 1e-6)
            # mask pairs where either frame is nan
            pair_valid = valid[:-1] & valid[1:]
            if pair_valid.any():
                max_vel = max(max_vel, float(vel[pair_valid].mean()))
        return max_vel

    def _detect(self, now):
        # Vertical movement gate: force noop if wrist is translating vertically too fast
        if self._wrist_vertical_vel() > self.T_VERTICAL_VEL:
            self._active_mode = 'noop'
            self._pending_mode = 'noop'
            return ('noop', 0.0)

        # Build per-hand debug snapshot before the main detection loop
        dbg = []
        for hi in range(2):
            s = self._hand_states[hi]
            times, lms = self._hand_window(hi, 0.5)
            feat = self._motion_features(times, lms)
            ang_vel = self._rotation_angular_vel(times, lms) if len(times) >= 2 else 0.0
            index_hold = (now - s.index_since) if s.index_since is not None else 0.0
            ext_frame = self._extension(lms[-1]) if lms else None
            dbg.append({
                'tip_artic':       feat['tip_artic']  if feat else None,
                'tip_disp':        feat['tip_disp']   if feat else None,
                'ext_change':      feat['ext_change'] if feat else None,
                'ang_vel':         ang_vel,
                'index_hold':      index_hold,
                'index_ext_since': s.index_ext_since,
                'index_ext':       float(ext_frame[1]) if ext_frame is not None else 0.0,
                'others_dist_max': float(max(
                    np.linalg.norm(lms[-1][12] - lms[-1][0]),
                    np.linalg.norm(lms[-1][16] - lms[-1][0]),
                    np.linalg.norm(lms[-1][20] - lms[-1][0]),
                ) / (float(np.linalg.norm(lms[-1][MCPS4] - lms[-1][[0]], axis=1).mean()) + 1e-6)
                ) if lms else 0.0,
                'mean_ext':        s.mean_ext,
                'fist_pending':    s.fist_pending,
                'fist_intensity':  s.fist_intensity,
                'is_chord': bool(feat and
                                 feat['tip_disp']  >= self.T_CHORD_TIP_DISP and
                                 feat['mcp_speed'] >= self.T_CHORD_MCP),
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
                av = self._rotation_angular_vel(times, lms)
                if av >= self.T_ROTATE:
                    intensity = float(np.clip(av / self.NORM_ROTATE, 0.0, 1.0))
                    if intensity > best_intensity:
                        best_mode, best_intensity = 'faster', intensity

            # 3. Chords / Runs — skip if index has been extended long enough.
            #    The full faster pose (index + others curled) suppresses via index_since;
            #    index alone held > T_INDEX_SUPPRESS also suppresses runs specifically.
            if s.index_since is not None:
                continue
            index_suppressed = (s.index_ext_since is not None and
                                now - s.index_ext_since >= self.T_INDEX_SUPPRESS)

            times, lms = self._hand_window(hi, 0.5)
            feat = self._motion_features(times, lms)
            if feat is None:
                continue

            # Chord: palm pivots (mcp_speed) AND tips sweep a large arc (tip_disp).
            # Raise the tip_disp threshold when already in runs to reduce spurious switching.
            disp_thr = (self.T_CHORD_TIP_DISP_IN_RUNS
                        if self._active_mode == 'runs'
                        else self.T_CHORD_TIP_DISP)
            is_chord = (feat['tip_disp']  >= disp_thr and
                        feat['mcp_speed'] >= self.T_CHORD_MCP)
            if is_chord:
                intensity = float(np.clip(feat['mcp_speed'] / self.NORM_CHORD_MCP, 0.0, 1.0))
                if intensity > best_intensity:
                    best_mode, best_intensity = 'chords', intensity
            elif (not index_suppressed and
                  s.mean_ext >= self.T_RUN_OPEN and
                  feat['tip_artic'] >= self.T_RUN_TIP_ARTIC and
                  feat['ext_change'] >= self.T_RUN_EXT):
                intensity = float(np.clip(feat['tip_artic'] / self.NORM_RUN, 0.0, 1.0))
                if intensity > best_intensity:
                    best_mode, best_intensity = 'runs', intensity

        # --- Hysteresis ---
        now = self._buf[-1][0] if self._buf else 0.0

        # (1) Mode-switch hold: when already in an active mode, a different mode must
        #     be consistently detected for SWITCH_HOLD seconds before committing to it.
        #     Faster/slower are one-shot events and always pass through immediately.
        if best_mode in ('faster', 'slower'):
            self._active_mode   = best_mode
            self._pending_mode  = best_mode
            self._pending_since = now
        elif self._active_mode == 'noop':
            # No active mode — commit immediately (no hold needed for first detection)
            self._active_mode   = best_mode
            self._pending_mode  = best_mode
            self._pending_since = now
        elif best_mode == self._active_mode:
            # Continuing same mode — reset pending
            self._pending_mode  = best_mode
            self._pending_since = now
        else:
            # Candidate switch: track how long this new mode has been dominant
            if best_mode != self._pending_mode:
                self._pending_mode  = best_mode
                self._pending_since = now
            elif now - self._pending_since >= self.SWITCH_HOLD:
                self._active_mode = best_mode

        committed_mode      = self._active_mode
        committed_intensity = best_intensity if committed_mode == best_mode else self._last_active_result[1]

        # (3) Noop dead-time: hold the last active mode briefly after signal drops.
        if committed_mode == 'noop' and now - self._last_active_t < self.NOOP_DEAD_TIME:
            return self._last_active_result

        if committed_mode != 'noop':
            self._last_active_t      = now
            self._last_active_result = (committed_mode, committed_intensity)

        return committed_mode, committed_intensity


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
