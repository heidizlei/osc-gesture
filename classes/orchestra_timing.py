"""Coalesce accompaniment changes briefly after a piano range update."""
import threading
import time


class OrchestraTiming:
    COOLDOWN = 0.5

    def __init__(self, send, piano_id):
        self.send = send
        self.piano_id = piano_id
        self.lock = threading.RLock()
        self.deadline = 0.0
        self.pending = {}
        self.timer = None
        self.closed = False
        self.occupancy_mask = 0
        self.last_occupancy = None

    def send_message(self, address, value):
        with self.lock:
            if self.closed:
                self.send(address, value)
                return
            is_range = address == '/setOutputRange'
            piano = is_range and (len(value) == 4 or value[0] == self.piano_id)
            occupancy = address == '/setOrchestraOccupancy'
            # Starting/stopping piano must never wait behind accompaniment.
            edge = occupancy and (not value[3] or not self.occupancy_mask)
            deferred = (is_range and not piano) or address in {
                '/setActiveInstruments', '/setForcedInstruments', '/setOrchestraOccupancy'}
            key = (address, value[0] if is_range else None)
            if address == '/setGestureControl' or (address == '/setModelConfig'
                    and value[0] == 'orchestraPianoPedalMode'):
                self.pending.clear()
                self.deadline = 0.0
            if deferred and not edge and time.monotonic() < self.deadline:
                self.pending[key] = (address, list(value))
                if occupancy and self.last_occupancy is not None:
                    # Renew receiver liveness without leaking the pending change.
                    self.send(address, self.last_occupancy)
                self._schedule()
                return
            self.pending.pop(key, None)
            self.send(address, value)
            if occupancy:
                self.occupancy_mask = value[3]
                self.last_occupancy = list(value)
            if piano:
                self.deadline = time.monotonic() + self.COOLDOWN

    def _schedule(self):
        if self.timer is None:
            self.timer = threading.Timer(max(0, self.deadline - time.monotonic()), self._flush)
            self.timer.daemon = True
            self.timer.start()

    def _flush(self):
        with self.lock:
            self.timer = None
            if self.closed or not self.pending:
                return
            if time.monotonic() < self.deadline:
                self._schedule()
                return
            messages = list(self.pending.values())
            self.pending.clear()
            for address, value in messages:
                self.send(address, value)
                if address == '/setOrchestraOccupancy':
                    self.occupancy_mask = value[3]
                    self.last_occupancy = list(value)

    def close(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
            if self.timer is not None:
                self.timer.cancel()
