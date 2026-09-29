"""Exercise actual HTTP/JPEG transport without opening a physical webcam."""
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import cv2
import numpy as np

from camera_server.server import Camera, Handler
from classes.mjpeg_capture import MJPEGCapture


class QuietHandler(Handler):
    def log_message(self, *_):
        pass


class MJPEGTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(None)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        self.server.camera = self.camera
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.capture = MJPEGCapture(f'http://127.0.0.1:{self.server.server_port}/stream')

    def tearDown(self):
        self.camera.stop.set()
        with self.camera.condition:
            self.camera.condition.notify_all()
        self.capture.release()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def publish(self, value):
        ok, jpeg = cv2.imencode('.jpg', np.full((24, 32, 3), value, np.uint8))
        self.assertTrue(ok)
        with self.camera.condition:
            self.camera.jpeg = jpeg.tobytes()
            self.camera.sequence += 1
            self.camera.condition.notify_all()

    def receive(self, value):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            ok, frame = self.capture.read()
            if ok and abs(float(frame.mean()) - value) < 2:
                return frame
        self.fail(f'Did not receive frame {value}: {self.capture.error}')

    def test_decode_and_no_duplicate_frames(self):
        self.publish(35)
        frame = self.receive(35)
        self.assertEqual(frame.shape, (24, 32, 3))
        self.assertEqual(self.capture.read(), (False, None))

    def test_latest_frame_replaces_unconsumed_frame(self):
        self.publish(20)
        self.receive(20)
        for value in (40, 80, 120):
            self.publish(value)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with self.capture._condition:
                    frame = self.capture._frame
                    if frame is not None and abs(float(frame.mean()) - value) < 2:
                        break
                time.sleep(0.01)
            else:
                self.fail('Receiver did not drain stream')
        ok, frame = self.capture.read()
        self.assertTrue(ok)
        self.assertAlmostEqual(float(frame.mean()), 120, delta=2)

    def test_stale_frame_is_rejected(self):
        self.publish(60)
        self.receive(60)
        with patch.object(self.capture, 'STALE_SECONDS', 0.01):
            time.sleep(0.02)
            self.assertEqual(self.capture.read(), (False, None))
            self.assertTrue(self.capture.stale)
            self.assertIsNotNone(self.capture.error)

    def test_release_and_resume(self):
        self.publish(50)
        self.receive(50)
        self.capture.release()
        self.assertFalse(self.capture.isOpened())
        self.assertTrue(self.capture.stale)
        self.assertTrue(self.capture.open())
        self.publish(90)
        self.receive(90)

    def test_reconnect_after_eof(self):
        self.publish(30)
        self.receive(30)
        self.camera.stop.set()
        with self.camera.condition:
            self.camera.condition.notify_all()
        deadline = time.monotonic() + 2
        while not self.capture.error and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(self.capture.error)
        self.camera.stop.clear()
        self.publish(160)
        self.receive(160)


if __name__ == '__main__':
    unittest.main()
