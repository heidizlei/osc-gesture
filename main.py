import argparse
from classes.app import OSCGestureApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSC gesture controller")
    parser.add_argument("--ip",   default="100.101.30.29", help="OSC target IP")
    parser.add_argument("--port", default=9001, type=int,  help="OSC target port")
    args = parser.parse_args()

    app = OSCGestureApp(ip=args.ip, port=args.port)
    app.run()
