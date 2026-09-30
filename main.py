import os

# PyInstaller's matplotlib runtime hook points MPLCONFIGDIR at a fresh temp dir on
# every launch (deleted at exit), forcing a font-cache rebuild each time. Override
# it with a stable, persistent cache dir before matplotlib gets imported (via
# mediapipe's drawing_utils) so the cache actually gets reused across runs.
os.environ["MPLCONFIGDIR"] = os.path.join(
    os.path.expanduser("~"), ".cache", "osc-gesture", "matplotlib"
)

import argparse
import sys
from classes.app import OSCGestureApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSC gesture controller")
    parser.add_argument("--host",    default="127.0.0.1", help="OSC target host")
    parser.add_argument("--port",    default=9001, type=int,  help="OSC target port")
    parser.add_argument("--camera-url", help="HTTP(S) MJPEG camera stream URL")
    parser.add_argument("--ui", choices=("web", "cv2"), default="web",
                        help="web: browser UI with buttons/sliders (default); "
                             "cv2: legacy OpenCV window")
    parser.add_argument("--http-host", default="127.0.0.1",
                        help="web UI bind address; use 0.0.0.0 to reach it "
                             "from a phone or tablet on the same network")
    parser.add_argument("--http-port", default=8765, type=int,
                        help="web UI port")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window on startup")
    parser.add_argument("--midi-state-host", help="enable stage snapshots to this proxy host")
    parser.add_argument("--midi-state-port", default=4200, type=int)
    parser.add_argument("--midi-state-group", default=2, type=int, choices=range(1, 17))
    parser.add_argument("--midi-state-stream", default=0, type=int, choices=range(256))
    parser.add_argument("--midi-state-view-port", type=int,
                        help="listen here for OSC /showGestureView 1|0 to switch the stage view")
    parser.add_argument("--orchestra-preset", default=None,
                        help="orchestra preset JSON to read instrument ids "
                             "from (its outputInstruments ids); defaults to "
                             "orchestra.json beside the app, then to built-in "
                             "values. Playing ranges are not read from it — "
                             "those are set in the UI")
    args = parser.parse_args()

    if not 1 <= args.midi_state_port <= 65535:
        parser.error("--midi-state-port must be 1..65535")
    if args.midi_state_view_port is not None and not 1 <= args.midi_state_view_port <= 65535:
        parser.error("--midi-state-view-port must be 1..65535")
    if args.midi_state_view_port is not None and not args.midi_state_host:
        parser.error("--midi-state-view-port requires --midi-state-host")

    if args.camera_url:
        from urllib.parse import urlparse
        parsed = urlparse(args.camera_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            parser.error("--camera-url must be an HTTP(S) stream URL")

    app = OSCGestureApp(ip=args.host, port=args.port,
                        camera_url=args.camera_url,
                        orchestra_preset=args.orchestra_preset)
    if args.midi_state_host:
        from classes.midi_state_sender import MidiStateSender
        app.midi_state_sender = MidiStateSender(args.midi_state_host, args.midi_state_port,
                                                args.midi_state_group, args.midi_state_stream,
                                                args.midi_state_view_port)
    if args.ui == "web" and getattr(sys, "frozen", False) and sys.platform == "darwin":
        # The .app needs a native event loop to be quittable from the Dock;
        # see classes/mac_app.py.
        from classes.mac_app import run_as_mac_app
        run_as_mac_app(app,
                       http_host=args.http_host,
                       http_port=args.http_port,
                       open_browser=not args.no_browser)
    else:
        app.run(ui=args.ui,
                http_host=args.http_host,
                http_port=args.http_port,
                open_browser=not args.no_browser)
