"""Per-hand red-boundary hysteresis shared by camera and mock input."""


class BoundaryGate:
    MARGIN = 0.02             # fraction of frame height
    SETTLE_S = 0.25
    REARM_S = 0.6
    LOSS_S = 0.15

    def __init__(self):
        self.hands = {}
        self.ready = set()
        self.entered = set()

    def clear(self):
        self.hands.clear()
        self.ready.clear()
        self.entered.clear()

    def update(self, positions, boundary, now):
        accepted = []
        self.ready.clear()
        self.entered.clear()
        seen = set()
        for x, y, label in positions:
            label = label or ('Left' if x < 0.5 else 'Right')
            seen.add(label)
            s = self.hands.setdefault(label, dict(
                active=False, armed=True, confirmed=False, since=None,
                red_since=None, side=None, crossings=[], seen=now))
            if now - s['seen'] > self.LOSS_S:
                s.update(active=False, confirmed=False, since=None)
            if now - s['seen'] >= self.REARM_S:
                s.update(armed=True, side=None, crossings=[])
            s['seen'] = now
            inside = y <= boundary
            deep = y <= boundary - self.MARGIN
            s['crossings'] = [t for t in s['crossings'] if now - t < self.REARM_S]
            if s['side'] is not None and inside != s['side']:
                s['crossings'].append(now)
            s['side'] = inside

            if y > boundary + self.MARGIN:
                if s['red_since'] is None:
                    s['red_since'] = now
                s.update(active=False, confirmed=False, since=None)
                if now - s['red_since'] >= self.REARM_S:
                    s['armed'] = True
                    s['crossings'].clear()
            else:
                s['red_since'] = None

            # A bounce back over the line consumes the immediate-entry privilege.
            # Continuous, clearly active input can always recover after SETTLE_S.
            bouncing = len(s['crossings']) >= 2
            if bouncing:
                s.update(active=False, confirmed=False, armed=False)

            if deep:
                if s['since'] is None:
                    s['since'] = now
            else:
                s['since'] = None

            if inside and s['armed'] and not s['active']:
                s.update(active=True, armed=False, entered=now, entry_y=y)
                self.entered.add(label)
            if deep and now - s['since'] >= self.SETTLE_S:
                s.update(active=True, confirmed=True)
                s['crossings'].clear()
            elif (s['active'] and not s['confirmed']
                  and now - s['entered'] >= self.SETTLE_S):
                s['active'] = False

            if s['active']:
                # The first region gets its range and activation immediately.
                # Hold that region until settled, deferring subsequent changes.
                accepted.append((x, y if s['confirmed'] else s['entry_y'], label))
                if s['confirmed'] and deep:
                    self.ready.add(label)

        for label, s in self.hands.items():
            if label not in seen:
                s['since'] = None
                if now - s['seen'] > self.LOSS_S:
                    s.update(active=False, confirmed=False)
                if now - s['seen'] >= self.REARM_S:
                    s.update(armed=True, side=None, crossings=[])
        return accepted
