"""Opt-in, complete MIDI State gesture snapshots over OSC/UDP."""

import math
import struct
import threading
import time

from pythonosc.udp_client import SimpleUDPClient


INSTRUMENTS = {'none': 0, 'brass': 1, 'strings': 2, 'piano': 3}
MAX_VALUE = 0xffffffff
CHANNEL = 15
PERIOD = 0.020
CAPTURE_TIMEOUT = 0.100


def normalized(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Gesture coordinates must be finite')
    return min(1.0, max(0.0, value))


def encode_snapshot(hands, split, cutoff, sequence, now, group=1):
    """Encode channel 16 CC 20..29; coordinates use full-range uint32."""
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
    values.extend((round(split * MAX_VALUE), round(cutoff * MAX_VALUE)))
    header = struct.pack('>BBHHHQQQ', 1, 0, 1 << CHANNEL,
                         sequence & 0xffff, 0, now, now, now)
    words = [0x00400000, 0x003003c0]
    for controller, value in enumerate(values, 20):
        words.extend((0x00400000,
                      0x40b00000 | ((group - 1) << 24) | (CHANNEL << 16) | (controller << 8),
                      value))
    return header + struct.pack('>' + 'I' * len(words), *words)


class MidiStateSender:
    def __init__(self, host, port=4200, group=1, stream=0):
        if not 1 <= group <= 16 or not 0 <= stream <= 255 or not 1 <= port <= 65535:
            raise ValueError('Invalid MIDI State port, group, or stream')
        self.client = SimpleUDPClient(host, port)
        self.address = f'/midi-state/{group}/{stream}'
        self.group = group
        self.sequence = 0
        self.lock = threading.Lock()
        self.snapshot = ({}, .375, .75, float('-inf'))
        self.stop_event = threading.Event()
        self.last_error = None
        self.thread = threading.Thread(target=self._run, name='midi-state', daemon=True)
        self.thread.start()

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
        if missing or time.monotonic() - captured > CAPTURE_TIMEOUT:
            hands = {}
        clip = encode_snapshot(hands, split, cutoff, self.sequence,
                               time.monotonic_ns(), self.group)
        self.sequence = (self.sequence + 1) & 0xffff
        self.client.send_message(self.address, clip)

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
        self.stop_event.set()
        self.thread.join(timeout=1)
        if not self.thread.is_alive():
            try:
                self._send(missing=True)
            except OSError:
                pass