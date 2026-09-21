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
    parser.add_argument("--orchestra-preset", default=None,
                        help="orchestra preset JSON to read instrument reset "
                             "ranges from (its outputInstruments low/high); "
                             "defaults to orchestra.json beside the app, then "
                             "to built-in values")
    args = parser.parse_args()

    app = OSCGestureApp(ip=args.host, port=args.port,
                        orchestra_preset=args.orchestra_preset)
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
