import os

# PyInstaller's matplotlib runtime hook points MPLCONFIGDIR at a fresh temp dir on
# every launch (deleted at exit), forcing a font-cache rebuild each time. Override
# it with a stable, persistent cache dir before matplotlib gets imported (via
# mediapipe's drawing_utils) so the cache actually gets reused across runs.
os.environ["MPLCONFIGDIR"] = os.path.join(
    os.path.expanduser("~"), ".cache", "osc-gesture", "matplotlib"
)

import argparse
from classes.app import OSCGestureApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSC gesture controller")
    parser.add_argument("--host",    default="127.0.0.1", help="OSC target host")
    parser.add_argument("--port",    default=9001, type=int,  help="OSC target port")
    args = parser.parse_args()

    app = OSCGestureApp(ip=args.host, port=args.port)
    app.run()
