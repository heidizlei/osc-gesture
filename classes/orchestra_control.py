"""Orchestra state transport and camera-side loss grace (no MIDI dependency)."""
import json
import socket
import threading
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


class HandGrace:
    def __init__(self):
        self.seen = {}

    def update(self, positions, observed_labels, region_for, now, grace):
        # A seen hand outside the active area or in a new region exits immediately.
        live = {label: region_for(x, y, label) for x, y, label in positions}
        for label in observed_labels:
            if label not in live:
                self.seen.pop(label, None)
        for label, region in live.items():
            self.seen[label] = (region, now)
        self.seen = {label: item for label, item in self.seen.items()
                     if label in live or now - item[1] < grace}
        return sorted({region for region, _ in self.seen.values()})


class OrchestraControl:
    def __init__(self, send, host):
        self.send = send
        self.lock = threading.RLock()
        self.requested = None
        self.status = {}
        self.received = 0.0
        self.simulated = False
        self.down = True
        self.browser_seen = 0.0
        self.occupancy = [1, 48, 61, 0]
        self.occupancy_seen = 0.0
        self.stop_event = threading.Event()
        dispatcher = Dispatcher()
        dispatcher.map('/orchestraState', self._receive)
        self.server = ThreadingOSCUDPServer(('0.0.0.0', 0), dispatcher)
        # Determine the interface used to reach the configured OSC receiver.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((host, 9))
            self.reply_host = sock.getsockname()[0]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def enabled(self):
        with self.lock:
            return bool(self.requested if self.requested is not None else self.status.get('mode', False))

    def _receive(self, address, payload):
        try:
            status = json.loads(payload)
            if not isinstance(status, dict) or 'mode' not in status:
                return
        except (ValueError, TypeError):
            return
        with self.lock:
            self.status = status
            self.received = time.monotonic()
            if self.requested is not None and bool(status['mode']) == self.requested:
                self.requested = None

    def set_mode(self, enabled):
        with self.lock:
            self.requested = bool(enabled)
            if not enabled:
                self.simulated = False
            self.send('/setModelConfig', ['orchestraPianoPedalMode', int(enabled)])

    def set_source(self, simulated):
        with self.lock:
            if simulated and not self.enabled:
                raise ValueError('Enable Pedal controls piano first')
            if simulated and not self.simulated:
                self.down = bool(self.status.get('pedalDown', True))
            self.simulated = bool(simulated)
            self.browser_seen = time.monotonic()
            self.send('/setOrchestraPedalSource', int(self.simulated))
            if self.simulated:
                self.send('/setOrchestraPianoPedal', int(self.down))

    def pedal(self, down):
        with self.lock:
            if not self.simulated or not self.enabled:
                raise ValueError('Select Simulated pedal first')
            self.down = bool(down)
            self.browser_seen = time.monotonic()
            self.send('/setOrchestraPianoPedal', int(self.down))

    def set_occupancy(self, piano, strings, brass, mask):
        with self.lock:
            values = [piano, strings, brass, mask]
            changed = values != self.occupancy
            self.occupancy = values
            self.occupancy_seen = time.monotonic()
            if changed and self.enabled:
                self.send('/setOrchestraOccupancy', values)

    def state(self):
        with self.lock:
            self.browser_seen = time.monotonic()
            return dict(requested=self.enabled, confirmed=time.monotonic() - self.received < 2,
                        receiver=dict(self.status), simulated=self.simulated, down=self.down)

    def _run(self):
        while not self.stop_event.wait(0.3):
            try:
                with self.lock:
                    if self.requested is not None:
                        self.send('/setModelConfig', ['orchestraPianoPedalMode', int(self.requested)])
                    if self.enabled:
                        values = list(self.occupancy)
                        if time.monotonic() - self.occupancy_seen > 1:
                            values[3] = 0
                        self.send('/setOrchestraOccupancy', values)
                        if self.simulated and time.monotonic() - self.browser_seen > 2:
                            self.simulated = False
                        self.send('/setOrchestraPedalSource', int(self.simulated))
                        if self.simulated:
                            self.send('/setOrchestraPianoPedal', int(self.down))
                    self.send('/getOrchestraState', [self.reply_host, self.server.server_address[1]])
            except OSError:
                pass  # UI shows stale/unconfirmed receiver status until it recovers.

    def close(self):
        self.stop_event.set()
        self.send('/setOrchestraPedalSource', 0)
        self.send('/setOrchestraOccupancy', self.occupancy[:3] + [0])
        self.server.shutdown()
        self.server.server_close()
