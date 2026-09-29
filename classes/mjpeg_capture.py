"""HTTP MJPEG input with one pending frame, bounded reads and reconnects."""
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np


class MJPEGCapture:
    STALE_SECONDS = 1.0
    MAX_BYTES = 8 * 1024 * 1024

    def __init__(self, url):
        self.url = url
        self._condition = threading.Condition()
        self._frame = None
        self._received = 0.0
        self.error = 'Connecting to camera stream'
        self._stop = threading.Event()
        self._thread = None
        self.open()

    @property
    def stale(self):
        return time.monotonic() - self._received > self.STALE_SECONDS

    def isOpened(self):
        return (self._thread is not None and self._thread.is_alive()
                and not self._stop.is_set())

    def open(self, *_):
        if self.isOpened():
            return True
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._receive, daemon=True)
        self._thread.start()
        return True

    def read(self):
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame is not None or self._stop.is_set(), timeout=0.1)
            frame, self._frame = self._frame, None
            if self.stale:
                self.error = 'Camera stream has no recent frames; reconnecting if needed'
                return False, None
            return frame is not None, frame

    def release(self):
        self._stop.set()
        with self._condition:
            self._frame = None
            self._received = 0.0
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _receive(self):
        stop = self._stop
        while not stop.is_set():
            try:
                with urlopen(self.url, timeout=2) as response:
                    buffer = bytearray()
                    last_jpeg = time.monotonic()
                    while not stop.is_set():
                        chunk = response.read1(65536)
                        if not chunk:
                            raise OSError('Camera stream closed')
                        buffer.extend(chunk)
                        if len(buffer) > self.MAX_BYTES:
                            raise OSError('Camera JPEG exceeds 8 MiB')
                        # JPEG markers also work with multipart servers that omit
                        # Content-Length. Retain partial markers between reads.
                        while True:
                            start = buffer.find(b'\xff\xd8')
                            if start < 0:
                                buffer[:] = buffer[-1:]
                                break
                            if start:
                                del buffer[:start]
                            end = buffer.find(b'\xff\xd9', 2)
                            if end < 0:
                                break
                            jpeg = bytes(buffer[:end + 2])
                            del buffer[:end + 2]
                            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                last_jpeg = time.monotonic()
                                with self._condition:
                                    if stop.is_set():
                                        return
                                    self._frame = frame
                                    self._received = last_jpeg
                                    self.error = None
                                    self._condition.notify_all()
                        if time.monotonic() - last_jpeg > 2:
                            raise OSError('Camera stream stopped producing JPEG frames')
            except Exception as exc:
                self.error = f'Camera stream: {exc}'
            stop.wait(0.5)
