import time

_RUNS_LEVELS   = [(0.85, 0), (0.60, 1), (0.40, 2), (-1, 3)]   # (threshold, level)
_CHORDS_LEVELS = [(0.70, 0), (0.30, 1), (-1, 2)]

def _runs_level(intensity):
    for thr, lvl in _RUNS_LEVELS:
        if intensity > thr:
            return lvl
    return 3

def _chords_level(intensity):
    for thr, lvl in _CHORDS_LEVELS:
        if intensity > thr:
            return lvl
    return 2


class GestureSender:
    """
    Translates gesture detector output into OSC messages.

    Call tick(mode, intensity, osc_client) every 500 ms (i.e. every
    time the detector emits a new result).

    OSC messages sent:
        /playRuns   [0-3]     runs level (0=VeryFast … 3=Slow)
        /playChords [0-2]     chords level (0=Fast … 2=Slow)
        /adjustTempo [ratio]  1.7 (faster) or 0.6 (slower)
        /resetControl []      before tempo; also after 2s noop while active
    """

    RUNS_CHORDS_REQUIRED  = 3      # consecutive ticks to trigger
    FASTER_REQUIRED       = 2      # consecutive ticks to trigger
    MODE_LOCK_S           = 4.0    # seconds before runs↔chords switch is allowed
    TEMPO_BLOCK_S         = 4.0    # seconds between tempo messages
    NOOP_RESET_S          = 2.0    # seconds of noop before sending resetControl

    FASTER_RATIO = 1.7
    SLOWER_RATIO = 0.6

    def __init__(self):
        # Runs/chords accumulator
        self._cons_mode       = None
        self._cons_count      = 0
        self._cons_intensities = []

        # Last sent state
        self._last_sent_mode  = None   # 'runs' or 'chords'
        self._last_sent_level = None
        self._last_sent_t     = 0.0

        # Faster accumulator
        self._faster_count    = 0

        # Tempo block
        self._tempo_blocked_until = 0.0

        # Noop tracking
        self._noop_since      = None

    def tick(self, mode, intensity, osc_client):
        now = time.time()

        # ----------------------------------------------------------------
        # Slower — one-shot, send immediately
        # ----------------------------------------------------------------
        if mode == 'slower':
            self._cons_mode   = None
            self._cons_count  = 0
            self._faster_count = 0
            self._noop_since  = None
            if now >= self._tempo_blocked_until:
                self._send_tempo(self.SLOWER_RATIO, osc_client)
                self._tempo_blocked_until = now + self.TEMPO_BLOCK_S
            return

        # ----------------------------------------------------------------
        # Faster — 2 consecutive ticks
        # ----------------------------------------------------------------
        if mode == 'faster':
            self._cons_mode  = None
            self._cons_count = 0
            self._noop_since = None
            self._faster_count += 1
            if self._faster_count >= self.FASTER_REQUIRED and now >= self._tempo_blocked_until:
                self._send_tempo(self.FASTER_RATIO, osc_client)
                self._tempo_blocked_until = now + self.TEMPO_BLOCK_S
                self._faster_count = 0
            return

        self._faster_count = 0

        # ----------------------------------------------------------------
        # Noop
        # ----------------------------------------------------------------
        if mode == 'noop':
            self._cons_count = 0
            self._cons_mode  = None
            if self._noop_since is None:
                self._noop_since = now
            elif (now - self._noop_since >= self.NOOP_RESET_S
                  and self._last_sent_mode in ('runs', 'chords')):
                self._send(osc_client, '/resetControl', [])
                self._last_sent_mode  = None
                self._last_sent_level = None
                self._noop_since      = None   # don't re-fire until next noop streak
            return

        self._noop_since = None

        # ----------------------------------------------------------------
        # Runs / chords — 3 consecutive, resend only on level change
        # ----------------------------------------------------------------
        if mode not in ('runs', 'chords'):
            return

        in_lock = (self._last_sent_mode is not None
                   and now - self._last_sent_t < self.MODE_LOCK_S)

        if mode != self._cons_mode:
            if in_lock and mode != self._last_sent_mode:
                return   # locked to sent mode, ignore different mode
            self._cons_mode       = mode
            self._cons_count      = 1
            self._cons_intensities = [intensity]
        else:
            self._cons_count += 1
            self._cons_intensities.append(intensity)

        if self._cons_count >= self.RUNS_CHORDS_REQUIRED:
            avg  = sum(self._cons_intensities) / len(self._cons_intensities)
            lvl  = _runs_level(avg) if mode == 'runs' else _chords_level(avg)
            # only send if level changed (or first send)
            if lvl != self._last_sent_level or mode != self._last_sent_mode:
                addr = '/playRuns' if mode == 'runs' else '/playChords'
                self._send(osc_client, addr, [lvl])
                self._last_sent_mode  = mode
                self._last_sent_level = lvl
                self._last_sent_t     = now
            self._cons_count       = 0
            self._cons_intensities = []

    # ------------------------------------------------------------------

    def _send_tempo(self, ratio, osc_client):
        if self._last_sent_mode in ('runs', 'chords'):
            self._send(osc_client, '/resetControl', [])
        self._send(osc_client, '/adjustTempo', [ratio])

    def _send(self, osc_client, address, args):
        try:
            osc_client.send_message(address, args)
            arg_str = '  ' + '  '.join(str(a) for a in args) if args else ''
            print(f"OSC → {address}{arg_str}")
        except Exception as e:
            print(f"OSC error {address}: {e}")
