#!/usr/bin/env python3
"""
Plot captured live session with clean visualization: white background,
control intervals shown as step-function pitch bands, no legend.

Usage:
    python3 plot_live_capture.py /path/to/capture_dir
    python3 plot_live_capture.py ~/haires/workspace/control_runs/E1_live/session_dir
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import mido

FONT_DIR = Path(__file__).parent / "fonts"
for _font_file in ("LibreFranklin-Regular.ttf", "LibreFranklin-Bold.ttf"):
    _font_path = FONT_DIR / _font_file
    if _font_path.exists():
        fm.fontManager.addfont(str(_font_path))
if (FONT_DIR / "LibreFranklin-Regular.ttf").exists():
    plt.rcParams["font.family"] = "Libre Franklin"

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Warm orange/rust color palette
NOTE_FILL_COLOR = "#e8703a"  # Rust orange
NOTE_EDGE_COLOR = "#8a3a0a"  # Dark orange border
CONTROL_BAND_COLOR = "#fbe3d4"  # Very light cream-orange for control regions
RUNS_BAND_COLOR = "#fff0a8"     # Light yellow for /playRuns intervals
CHORDS_BAND_COLOR = "#c9e8b8"   # Light green for /playChords intervals
RUNS_LABEL_COLOR = "#8a7000"    # Darkened version of RUNS_BAND_COLOR, for label text
CHORDS_LABEL_COLOR = "#2e6b22"  # Darkened version of CHORDS_BAND_COLOR, for label text
TEMPO_FASTER_BAND_COLOR = "#b8d8f5"  # Light blue for /adjustTempo ratio > 1
TEMPO_SLOWER_BAND_COLOR = "#dcc6f0"  # Light purple for /adjustTempo ratio < 1
TEMPO_FASTER_LABEL_COLOR = "#1f4e8a"  # Darkened version of TEMPO_FASTER_BAND_COLOR
TEMPO_SLOWER_LABEL_COLOR = "#5a2d82"  # Darkened version of TEMPO_SLOWER_BAND_COLOR

# Level names match the web controller's Runs/Chords button labels exactly
RUNS_LEVEL_NAMES = {0: "VeryFast", 1: "Fast", 2: "Moderate", 3: "Slow"}
CHORDS_LEVEL_NAMES = {0: "Fast", 1: "Moderate", 2: "Slow"}


def load_notes(midi_path: Path):
    """Load notes from MIDI file."""
    mid = mido.MidiFile(str(midi_path))
    tpq = mid.ticks_per_beat
    tempo_us = 500000
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            break
    cs_per_tick = (tempo_us / 1_000_000) * 100 / tpq
    notes = []
    for tr in mid.tracks:
        abs_tick = 0
        pending = {}
        for msg in tr:
            abs_tick += msg.time
            t = abs_tick * cs_per_tick
            key = (msg.channel, msg.note) if hasattr(msg, "note") else None
            if msg.type == "note_on" and msg.velocity > 0:
                pending[key] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on") and key in pending:
                on, vel = pending.pop(key)
                notes.append((on, t, msg.note, msg.channel, vel))
    return sorted(notes, key=lambda n: n[0])


def plot_capture(run_dir: Path) -> None:
    """Plot live capture with clean white-background style."""
    midi_path = run_dir / "output_000.mid"
    sched_path = run_dir / "schedule.json"

    if not midi_path.exists():
        print(f"  [plot] missing {midi_path}, skipping")
        return

    notes = load_notes(midi_path)
    if not notes:
        print(f"no notes in {midi_path}")
        return

    # Load schedule for control intervals
    sched = []
    if sched_path.exists():
        sched = json.loads(sched_path.read_text())

    # Setup figure with white background
    fig, ax = plt.subplots(figsize=(19.2, 6), facecolor='white')
    ax.set_facecolor('white')

    pitches = [n[2] for n in notes]
    pitch_lo = max(0, min(pitches) - 3)
    pitch_hi = min(127, max(pitches) + 3)
    total_s = max(n[1] for n in notes) / 100.0

    # Shade /playRuns and /playChords intervals (ended by the next mode event
    # or a /resetControl) as full-height bands behind the notes.
    mode_events = sorted(
        [e for e in sched if e.get("address") in ("/playRuns", "/playChords", "/resetControl")],
        key=lambda e: e["time_cs"],
    )
    for i, ev in enumerate(mode_events):
        addr = ev["address"]
        if addr not in ("/playRuns", "/playChords"):
            continue
        start_s = ev["time_cs"] / 100.0
        end_s = (mode_events[i + 1]["time_cs"] / 100.0
                 if i + 1 < len(mode_events) else total_s + 0.2)
        level = ev["args"][0] if ev.get("args") else None
        if addr == "/playRuns":
            band_color, label_color = RUNS_BAND_COLOR, RUNS_LABEL_COLOR
            level_name = RUNS_LEVEL_NAMES.get(level, f"L{level}")
            label = f"{level_name}\nRuns"
        else:
            band_color, label_color = CHORDS_BAND_COLOR, CHORDS_LABEL_COLOR
            level_name = CHORDS_LEVEL_NAMES.get(level, f"L{level}")
            label = f"{level_name}\nChords"
        ax.axvspan(start_s, end_s, color=band_color, alpha=0.55, zorder=-1, linewidth=0)
        # y in axes-fraction so the label always sits near the top of the band,
        # regardless of the pitch range plotted, and stays covered by the shading.
        ax.text((start_s + end_s) / 2, 0.97, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=18, color=label_color,
                fontweight='bold', linespacing=1.1, zorder=5)

    # Shade /adjustTempo intervals, ending each band when its linear tempo
    # ramp finishes (mirrors OSCController.cpp's rampCs formula on the
    # JordanAI side, used whenever the OSC message omits an explicit rampCs).
    tempo_events = sorted(
        [e for e in sched if e.get("address") == "/adjustTempo"],
        key=lambda e: e["time_cs"],
    )
    for ev in tempo_events:
        args = ev.get("args") or [1.0]
        ratio = args[0]
        if len(args) >= 2:
            ramp_cs = args[1]
        else:
            dev = abs(math.log2(ratio)) if ratio > 0 else 0.0
            ramp_cs = round(200.0 + 500.0 * dev)
            ramp_cs = max(200, min(700, ramp_cs))
        start_s = ev["time_cs"] / 100.0
        end_s = start_s + ramp_cs / 100.0
        if ratio > 1.0:
            band_color, label_color, label = TEMPO_FASTER_BAND_COLOR, TEMPO_FASTER_LABEL_COLOR, "Faster"
        else:
            band_color, label_color, label = TEMPO_SLOWER_BAND_COLOR, TEMPO_SLOWER_LABEL_COLOR, "Slower"
        ax.axvspan(start_s, end_s, color=band_color, alpha=0.55, zorder=-1, linewidth=0)
        ax.text((start_s + end_s) / 2, 0.97, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=18, color=label_color,
                fontweight='bold', zorder=5)

    # Plot control intervals (pitch ranges from /setOutputRange events)
    range_events = [e for e in sched if e.get("address") == "/setOutputRange"]
    if range_events:
        range_events = sorted(range_events, key=lambda e: e["time_cs"])
        for i, ev in enumerate(range_events):
            # Extract pitch range (args can be [low, high] or [id, low, high] or [low, high, low2, high2])
            args = ev["args"]
            if len(args) == 2:
                band_low, band_high = args[0], args[1]
            elif len(args) == 3:
                band_low, band_high = args[1], args[2]
            elif len(args) == 4:
                band_low, band_high = args[0], args[1]
            else:
                continue

            start_s = ev["time_cs"] / 100.0
            end_s = (range_events[i + 1]["time_cs"] / 100.0
                     if i + 1 < len(range_events) else total_s + 0.2)

            # Light cream control band
            ax.axhspan(band_low - 0.5, band_high + 0.5,
                       xmin=start_s / (total_s + 0.2),
                       xmax=end_s / (total_s + 0.2),
                       color=CONTROL_BAND_COLOR, alpha=0.6, zorder=0, linewidth=0)

    # Plot notes
    for on, off, pitch, ch, vel in notes:
        alpha = 0.7 + 0.25 * (vel / 127)  # velocity affects opacity
        face = mcolors.to_rgba(NOTE_FILL_COLOR, alpha=alpha)
        rect = mpatches.Rectangle(
            (on / 100.0, pitch - 0.4),
            max(off - on, 2) / 100.0, 0.8,
            facecolor=face, edgecolor=NOTE_EDGE_COLOR, linewidth=1.4,
        )
        ax.add_patch(rect)

    # Axes setup
    c_pitches = [p for p in range(pitch_lo, pitch_hi + 1) if p % 12 == 0]
    ax.set_yticks(c_pitches)
    ax.set_yticklabels([f"{NOTE_NAMES[p%12]}{p//12-1}" for p in c_pitches],
                       fontsize=14, color="#222")
    ax.set_ylim(pitch_lo - 0.5, pitch_hi + 0.5)
    ax.set_xlim(0, total_s + 0.2)

    ax.set_xlabel("Time (s)", fontsize=15, color="#222")
    ax.set_ylabel("Pitch", fontsize=15, color="#222")
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.3, color="#999")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#999")
    ax.spines['bottom'].set_color("#999")
    ax.tick_params(colors="#444", labelsize=13)

    plt.tight_layout()
    out = run_dir / "pianoroll.png"
    plt.savefig(out, dpi=140, facecolor='white', edgecolor='none')
    plt.close()
    print(f"  → {out}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_live_capture.py <run_dir>")
        print("  run_dir: directory containing output_000.mid and schedule.json")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    print(f"Plotting: {run_dir.name}")
    plot_capture(run_dir)


if __name__ == "__main__":
    main()
