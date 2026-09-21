#!/usr/bin/env python3
"""
Capture /noteon and /noteoff OSC events from the live JordanAI app and write to MIDI.

Usage:
    conda run -n jambot python3 capture_live_generation.py --port 9001 --output /tmp/capture.mid

Then in JordanAI app:
  - Settings → Output → OSC → localhost:9001
  - Play/generate → events stream to this script
  - Press Ctrl+C to stop and save MIDI
"""

import argparse
import signal
import sys
from pathlib import Path

import mido
from pythonosc import dispatcher, osc_server


class MIDICapture:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.mid = mido.MidiFile()
        self.track = mido.MidiTrack()
        self.mid.tracks.append(self.track)
        self.mid.ticks_per_beat = 480

        # Track active notes: (channel, note) → (onset_time, velocity)
        self.notes_on = {}
        self.current_time = 0
        self.note_count = 0

        print(f"Capturing to: {self.output_path}")
        print("Listening for /noteon and /noteoff events...")
        print("Press Ctrl+C to stop and save MIDI.")

    def handle_noteon(self, unused_addr, ch, note, velocity):
        """Handle /noteon instrument note velocity"""
        key = (ch, note)
        self.notes_on[key] = (self.current_time, velocity)
        self.note_count += 1
        print(f"  /noteon  ch={ch} note={note:3d} vel={velocity:3d}  (onset={self.current_time})")

    def handle_noteoff(self, unused_addr, ch, note):
        """Handle /noteoff instrument note"""
        key = (ch, note)
        if key in self.notes_on:
            onset, velocity = self.notes_on.pop(key)
            duration = self.current_time - onset
            # Convert to MIDI (time in ticks, assuming 480 ticks/beat)
            # For now, use a simple linear mapping: assume 480 ticks = 1 beat
            self.track.append(mido.Message('note_on', note=note, velocity=velocity, time=onset))
            self.track.append(mido.Message('note_off', note=note, time=duration))
            print(f"  /noteoff ch={ch} note={note:3d}              (duration={duration})")
        else:
            print(f"  [WARN] /noteoff ch={ch} note={note} without matching note_on")

    def handle_bar(self, unused_addr, bar_num):
        """Handle /bar bar_number (use to advance time)"""
        # Assume each bar is 480 ticks (1 beat)
        self.current_time = bar_num * 480
        print(f"  [bar {bar_num}]")

    def save(self):
        """Save captured MIDI to file"""
        self.mid.save(str(self.output_path))
        print(f"\n✓ Saved {self.note_count} notes to {self.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001, help="OSC listen port")
    parser.add_argument("--output", default="/tmp/live_capture.mid", help="Output MIDI file")
    args = parser.parse_args()

    capture = MIDICapture(args.output)

    # Setup OSC dispatcher
    disp = dispatcher.Dispatcher()
    disp.map("/noteon", capture.handle_noteon)
    disp.map("/noteoff", capture.handle_noteoff)
    disp.map("/bar", capture.handle_bar)

    # Setup server
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", args.port), disp)
    print(f"\n[OSC] Listening on 127.0.0.1:{args.port}\n")

    # Graceful shutdown
    def signal_handler(sig, frame):
        print("\n[OSC] Shutting down...")
        server.shutdown()
        capture.save()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
