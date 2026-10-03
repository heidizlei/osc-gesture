"""Opt-in, complete MIDI State gesture snapshots over OSC/UDP."""

import math
import socket
import struct
import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


INSTRUMENTS = {'none': 0, 'brass': 1, 'strings': 2, 'piano': 3}
MAX_VALUE = 0xffffffff
CHANNEL = 15
PERIOD = 0.020
CAPTURE_TIMEOUT = 0.100
VIEW_CONTROLLER = 30
VIEW_ADDRESS = '/showGestureView'
# MIDI State's multicast address for Group n (SPECIFICATION.md, network push mode)
MULTICAST_ADDRESS = '239.253.254.{}'


def normalized(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Gesture coordinates must be finite')
    return min(1.0, max(0.0, value))


def encode_snapshot(hands, split, cutoff, sequence, now, group=1, view=False):
    """Encode channel 16 CC 20..30; coordinates use full-range uint32.

    CC 30 is the stage view: 0 for the score, MIDI 1.0 127 (upscaled to
    0xffffffff) for the gesture view. It rides in every snapshot as state, so a
    lost packet or a freshly loaded page picks it up from the next one.
    """
    if not 1 <= group <= 16:
        raise ValueError('Group must be 1..16')
    if not 0 <= split <= cutoff <= 1:
        raise ValueError('Dividers must satisfy 0 <= split <= cutoff <= 1')
    values = []
    for label in ('Left', 'Right'):
        hand = hands.get(label)
        if hand is None:
            values.extend((0, 0, 0, 0))
        else:
            x, y, instrument = hand
            values.extend((round(normalized(x) * MAX_VALUE),
                           round(normalized(y) * MAX_VALUE), MAX_VALUE,
                           INSTRUMENTS[instrument]))
    values.extend((round(split * MAX_VALUE), round(cutoff * MAX_VALUE),
                   MAX_VALUE if view else 0))
    header = struct.pack('>BBHHHQQQ', 1, 0, 1 << CHANNEL,
                         sequence & 0xffff, 0, now, now, now)
    words = [0x00400000, 0x003003c0]
    for controller, value in enumerate(values, 20):
        words.extend((0x00400000,
                      0x40b00000 | ((group - 1) << 24) | (CHANNEL << 16) | (controller << 8),
                      value))
    return header + struct.pack('>' + 'I' * len(words), *words)


class MidiStateSender:
    def __init__(self, host=None, port=4200, group=2, stream=1, view_port=None,
                 multicast=True, ttl=1, interface=None):
        # One Group or several (e.g. 2 for the stage and 3 for the visuals): the same snapshot goes to each
        groups = sorted(set([group] if isinstance(group, int) else group))
        if (not groups or not all(1 <= g <= 16 for g in groups)
                or not 0 <= stream <= 255 or not 1 <= port <= 65535):
            raise ValueError('Invalid MIDI State port, group, or stream')
        if view_port is not None and not 1 <= view_port <= 65535:
            raise ValueError('Invalid gesture view OSC port')
        if not multicast and not host:
            raise ValueError('MIDI State needs multicast or a host')
        # Each Group's snapshot goes to its multicast address, and to the unicast host if one is given
        unicast = SimpleUDPClient(host, port) if host else None
        self.targets = []
        for g in groups:
            clients = [unicast] if unicast else []
            if multicast:
                client = SimpleUDPClient(MULTICAST_ADDRESS.format(g), port)
                client._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
                client._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
                if interface:
                    client._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
                                            socket.inet_aton(interface))
                clients.append(client)
            self.targets.append((g, f'/midi-state/{g}/{stream}', clients))
        self.sequence = 0
        self.lock = threading.Lock()
        self.snapshot = ({}, .375, .75, float('-inf'))
        self.view = False
        self.stop_event = threading.Event()
        self.last_error = None
        self.thread = threading.Thread(target=self._run, name='midi-state', daemon=True)
        self.thread.start()
        self.view_server = None
        if view_port is not None:
            dispatcher = Dispatcher()
            dispatcher.map(VIEW_ADDRESS, self._receive_view)
            self.view_server = ThreadingOSCUDPServer(('0.0.0.0', view_port), dispatcher)
            threading.Thread(target=self.view_server.serve_forever, name='midi-state-view',
                             daemon=True).start()

    def set_view(self, gesture):
        """Show the gesture view on the stage (True) or the score (False)."""
        with self.lock:
            self.view = bool(gesture)

    def _receive_view(self, address, *args):
        # `/showGestureView 1` (or 127) shows gestures, `/showGestureView 0` the
        # score. Anything unreadable is ignored rather than guessed.
        if len(args) != 1 or isinstance(args[0], str):
            return
        try:
            self.set_view(float(args[0]) != 0)
        except (TypeError, ValueError):
            return

    def update(self, positions, split, cutoff, region_for):
        hands = {}
        ambiguous = set()
        for x, y, label in positions:
            if label not in ('Left', 'Right'):
                continue
            if label in hands:
                ambiguous.add(label)
            instrument = region_for(x, y, label) if y <= cutoff else 'none'
            hands[label] = (normalized(x), normalized(y), instrument)
        for label in ambiguous:
            hands.pop(label, None)
        with self.lock:
            self.snapshot = (hands, split, cutoff, time.monotonic())

    def _send(self, missing=False):
        with self.lock:
            hands, split, cutoff, captured = self.snapshot
            view = self.view and not missing
        if missing or time.monotonic() - captured > CAPTURE_TIMEOUT:
            hands = {}
        now = time.monotonic_ns()
        for group, address, clients in self.targets:
            clip = encode_snapshot(hands, split, cutoff, self.sequence, now, group, view)
            for client in clients:
                client.send_message(address, clip)
        self.sequence = (self.sequence + 1) & 0xffff

    def _run(self):
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            try:
                self._send()
                self.last_error = None
            except OSError as error:
                self.last_error = str(error)
            deadline += PERIOD
            remaining = deadline - time.monotonic()
            if remaining < 0:
                deadline = time.monotonic()
                remaining = 0
            self.stop_event.wait(remaining)

    def close(self):
        # The final snapshot also hands the stage back to the score.
        if self.view_server is not None:
            self.view_server.shutdown()
            self.view_server.server_close()
        self.stop_event.set()
        self.thread.join(timeout=1)
        if not self.thread.is_alive():
            try:
                self._send(missing=True)
            except OSError:
                pass