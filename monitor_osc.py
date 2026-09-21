#!/usr/bin/env python3
"""
Monitor OSC events on a specified port and print them in real-time.

Usage:
    conda run -n jambot python3 monitor_osc.py --port 9002
    conda run -n jambot python3 monitor_osc.py --port 9002 --filter noteon
"""
import argparse
import sys
from pythonosc import dispatcher, osc_server


def main():
    parser = argparse.ArgumentParser(description='Monitor OSC traffic')
    parser.add_argument('--port', type=int, default=9002, help='OSC port to listen on')
    parser.add_argument('--filter', help='Only show messages matching this address (e.g., noteon)')
    args = parser.parse_args()

    disp = dispatcher.Dispatcher()

    def generic_handler(addr, *osc_args):
        """Handle any OSC message and print it."""
        if args.filter and args.filter not in addr:
            return
        args_str = ', '.join(str(arg) for arg in osc_args)
        if args_str:
            print(f"[{addr}]  {args_str}")
        else:
            print(f"[{addr}]")

    # Catch-all handler for any address
    disp.set_default_handler(generic_handler)

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", args.port), disp)

    print(f"Monitoring OSC on 127.0.0.1:{args.port}")
    if args.filter:
        print(f"Filter: {args.filter}")
    print("Waiting for messages... (Press Ctrl+C to stop)\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
