#!/usr/bin/env python3
"""
Plot a captured live session's piano roll with OSC control overlay.
Uses the live-capture specific plotter (white background, rust/orange colors).

Usage:
    python3 plot_capture.py /path/to/capture_dir
    python3 plot_capture.py ~/haires/workspace/control_runs/E1_live/my_session_dir
"""
import sys
from pathlib import Path

# Import the live capture plotter
sys.path.insert(0, str(Path(__file__).parent))
from plot_live_capture import plot_capture

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_capture.py <run_dir>")
        print("  run_dir: directory containing output_000.mid and schedule.json")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    midi_file = run_dir / "output_000.mid"

    if not midi_file.exists():
        print(f"Error: {midi_file} not found")
        sys.exit(1)

    print(f"Plotting: {run_dir.name}")
    plot_capture(run_dir)
    print(f"✓ Saved pianoroll.png")

if __name__ == "__main__":
    main()
