"""Local web UI for the gesture controller.

Serves ui.html plus an MJPEG camera preview and a small JSON control API, so
the app gets real clickable buttons and draggable sliders without OpenCV's
HighGUI window. Uses only the standard library, which keeps the PyInstaller
spec unchanged.

Division of labour: the gesture loop publishes raw frames and a state dict; the
browser composites the exclusion region, HUD and landmarks. JPEG encoding runs
on a server thread, so the preview never gates detection.
"""

import json
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

from .app import _resource_path

PREVIEW_FPS    = 15     # preview cap; the gesture loop runs as fast as it can
PREVIEW_MAX_W  = 960    # downscale wider frames before encoding
JPEG_QUALITY   = 72

_UI_FILE = "ui.html"


def get_local_ip():
    """Best-effort LAN address, for the phone/tablet hint printed at startup."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class _Handler(BaseHTTPRequestHandler):
    app = None          # injected by WebUI.start()
    server_version = "OSCGesture/1.0"

    def log_message(self, fmt, *args):
        pass            # the gesture loop owns stdout

    # ---- helpers ----

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- routes ----

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                with open(_resource_path(_UI_FILE), "rb") as fh:
                    self._send_bytes(fh.read(), "text/html; charset=utf-8")
            except OSError as e:
                self._send_json({"error": f"cannot read {_UI_FILE}: {e}"}, 500)
        elif path == "/state":
            self._send_json(self.app.web_state(osc_since=self._osc_since()))
        elif path == "/cameras":
            # Slow (it opens every capture device), which is why it's its own
            # route rather than part of the 10 Hz /state poll.
            refresh = parse_qs(urlparse(self.path).query).get("refresh")
            self._send_json(self.app.camera_options(refresh=bool(refresh)))
        elif path == "/stream":
            self._stream_mjpeg()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/control", "/osc"):
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json({"error": f"bad request: {e}"}, 400)
            return
        try:
            if path == "/osc":
                # The manual tab's own address/args, sent verbatim. Separate
                # from /control because that one names app state, while this
                # is a passthrough the app doesn't interpret.
                self._send_json(self.app.send_manual_osc(
                    payload.get("address"), payload.get("args")))
            else:
                self._send_json(self.app.apply_control(payload))
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)

    def _osc_since(self):
        """Last OSC seq the browser reports having; None means send the buffer."""
        raw = parse_qs(urlparse(self.path).query).get("osc_since", [None])[0]
        try:
            return int(raw) if raw is not None else None
        except ValueError:
            return None

    # ---- MJPEG preview ----

    def _stream_mjpeg(self):
        boundary = "oscgestureframe"
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        interval = 1.0 / PREVIEW_FPS
        last_seq = -1
        try:
            while self.app.running:
                started = time.perf_counter()
                frame, seq = self.app.latest_frame()
                # Re-encoding a frame the loop has not replaced yet is pure waste.
                if frame is not None and seq != last_seq:
                    last_seq = seq
                    jpeg = self._encode(frame)
                    if jpeg is not None:
                        self.wfile.write(f"--{boundary}\r\n".encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                remaining = interval - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass        # browser navigated away or reloaded

    @staticmethod
    def _encode(frame):
        h, w = frame.shape[:2]
        if w > PREVIEW_MAX_W:
            scale = PREVIEW_MAX_W / w
            frame = cv2.resize(frame, (PREVIEW_MAX_W, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buf.tobytes() if ok else None


class WebUI:
    """Runs the HTTP server on a daemon thread alongside the gesture loop."""

    def __init__(self, app, host="127.0.0.1", port=8765):
        self.app = app
        self.host = host
        self.port = port
        self._server = None
        self._thread = None

    def start(self, open_browser=True):
        handler = type("_BoundHandler", (_Handler,), {"app": self.app})
        # Threaded: the MJPEG response holds its connection open for the whole
        # session, so /state and /control need their own threads to get served.
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()

        url = f"http://{self.host}:{self.port}"
        print(f"Running OSC Gesture App (web UI) -> {url}")
        if self.host in ("0.0.0.0", ""):
            print(f"  on this network: http://{get_local_ip()}:{self.port}")
        else:
            print("  pass --http-host 0.0.0.0 to reach it from a phone/tablet")
        if open_browser:
            webbrowser.open(url)

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
