import argparse
from classes.app import OSCGestureApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSC gesture controller")
    parser.add_argument("--host",    default="100.101.30.29", help="OSC target host")
    parser.add_argument("--port",    default=9001, type=int,  help="OSC target port")
    parser.add_argument("--baroque", action="store_true",     help="Cap runs/chords at moderate level")
    args = parser.parse_args()

    app = OSCGestureApp(ip=args.host, port=args.port, baroque=args.baroque)
    app.run()
