#!/usr/bin/env python3
"""Standalone webcam-to-MJPEG server for the stage Mac mini."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import sys

import cv2


class Camera:
    def __init__(self, args):
        self.args = args
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        args = self.args
        while not self.stop.is_set():
            cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                cap.set(cv2.CAP_PROP_FPS, args.fps)
                if not cap.isOpened():
                    print('Cannot open camera; retrying. Check camera permission.', flush=True)
                while not self.stop.is_set() and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    if ok:
                        with self.condition:
                            self.jpeg = jpeg.tobytes()
                            self.sequence += 1
                            self.condition.notify_all()
            finally:
                cap.release()
                with self.condition:
                    self.jpeg = None
            self.stop.wait(1)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/', '/stream'):
            self.send_error(404)
            return
        if self.path == '/':
            body = b'<html><title>Stage camera</title><img src="/stream"></html>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.connection.settimeout(2)
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        camera = self.server.camera
        sequence = -1
        try:
            while not camera.stop.is_set():
                with camera.condition:
                    camera.condition.wait_for(lambda: camera.stop.is_set() or
                        (camera.jpeg is not None and camera.sequence != sequence), timeout=1)
                    if camera.jpeg is None or camera.sequence == sequence:
                        continue
                    jpeg, sequence = camera.jpeg, camera.sequence
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                 str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--quality', type=int, default=72)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) < 1 or not 1 <= args.quality <= 100:
        parser.error('width, height and fps must be positive; quality must be 1–100')
    camera = Camera(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.camera = camera
    camera.thread.start()
    print(f'Stage camera: http://<mac-mini-ip>:{args.port}/stream', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop.set()
        with camera.condition:
            camera.condition.notify_all()
        server.server_close()
        camera.thread.join(timeout=3)


if __name__ == '__main__':
    main()
