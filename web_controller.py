#!/usr/bin/env python3
"""
web_controller.py — Browser-based OSC controller.

Serves the page in the HTML constant below and accepts POST requests from
the browser, forwarding them as OSC UDP packets via pythonosc. The
controller.html written beside this script is a copy for easy editing --
it is rewritten from HTML on every run, and never read back.

Usage:
    python web_controller.py
    python web_controller.py --host 192.168.1.20 --port 9001
    python web_controller.py --http-port 8080   # main.py's web UI also uses 8765
    python web_controller.py --capture-port 9002  # OSC port to capture MIDI on
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import mido
from pythonosc import dispatcher, osc_server, udp_client

# Import the live capture plotter
try:
    from plot_live_capture import plot_capture
except ImportError:
    plot_capture = None

# Setup file logging for debugging
LOG_FILE = Path("/tmp/web_controller_debug.log")
def log_msg(msg):
    """Log to both console and file."""
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ---------------------------------------------------------------------------
# Capture state management
# ---------------------------------------------------------------------------

MIDI_TICKS_PER_BEAT = 480
MIDI_TEMPO = mido.bpm2tempo(120)  # microseconds per beat; fixes the cs→ticks mapping below

MAX_NOTE_DURATION_S = 1.0
SOUNDFONT_PATH = "/Users/hlei/haires/jam_bot-misc/data/soundfonts/YDP-GrandPiano-20160804.sf2"
FLUIDSYNTH_GAIN = 0.4  # fluidsynth default is 0.2; doubled for louder renders


def clip_midi_note_durations(midi_path: Path, max_duration_s: float = MAX_NOTE_DURATION_S) -> int:
    """Clip any note longer than max_duration_s in-place. Returns the number of notes clipped."""
    mid = mido.MidiFile(str(midi_path))
    tpq = mid.ticks_per_beat
    tempo_us = MIDI_TEMPO
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            tempo_us = msg.tempo
            break
    max_ticks = round(mido.second2tick(max_duration_s, tpq, tempo_us))

    clipped_count = 0
    for tr in mid.tracks:
        abs_tick = 0   # real running position — must not be mutated when a note is clipped
        pending = {}   # key -> on_tick
        new_msgs = []  # [recorded_tick, msg]
        for msg in tr:
            abs_tick += msg.time
            recorded_tick = abs_tick
            key = (msg.channel, msg.note) if hasattr(msg, 'note') else None
            if msg.type == 'note_on' and msg.velocity > 0:
                pending[key] = abs_tick
            elif msg.type in ('note_off', 'note_on') and key in pending:
                on_tick = pending.pop(key)
                if abs_tick - on_tick > max_ticks:
                    recorded_tick = on_tick + max_ticks
                    clipped_count += 1
            new_msgs.append([recorded_tick, msg.copy()])

        new_msgs.sort(key=lambda x: x[0])
        tr.clear()
        prev = 0
        for recorded_tick, msg in new_msgs:
            msg.time = max(0, recorded_tick - prev)
            prev = recorded_tick
            tr.append(msg)

    mid.save(str(midi_path))
    return clipped_count


def synthesize_audio(midi_path: Path, wav_path: Path, soundfont: str = SOUNDFONT_PATH) -> bool:
    """Render midi_path to wav_path via fluidsynth fast-render (writes to file, never plays aloud)."""
    if not Path(soundfont).exists():
        print(f"[Capture] Warning: soundfont not found at {soundfont}, skipping synthesis")
        return False
    try:
        subprocess.run(
            ['fluidsynth', '-ni', '-g', str(FLUIDSYNTH_GAIN), '-F', str(wav_path),
             '-r', '44100', soundfont, str(midi_path)],
            check=True, capture_output=True, text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Capture] Warning: fluidsynth failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("[Capture] Warning: fluidsynth not found on PATH, skipping synthesis")
        return False


class CaptureState:
    """Thread-safe capture of MIDI + OSC control events."""
    def __init__(self):
        self.active = False
        self.start_time_s = 0.0
        self.midi_events = []  # [(time_cs, 'note_on'|'note_off', note, velocity), ...]
        self.osc_events = []   # [(time_cs, address, args), ...]
        self.output_dir = None
        self.lock = threading.Lock()

    def start(self, output_dir: Path):
        """Begin capture session."""
        with self.lock:
            self.active = True
            self.start_time_s = time.time()
            # Create timestamped subdirectory for this capture
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(output_dir) / timestamp
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.midi_events = []
            self.osc_events = []
            print(f"[Capture] Started → {self.output_dir}")

    def record_osc(self, address: str, args: list):
        """Record an OSC control event."""
        with self.lock:
            if not self.active:
                return
            elapsed_s = time.time() - self.start_time_s
            time_cs = int(elapsed_s * 100)
            self.osc_events.append((time_cs, address, args))
            print(f"[Capture] OSC @{time_cs}cs: {address} {args}")

    def record_noteon(self, ch: int, note: int, velocity_int: int):
        """Record a MIDI note-on event. velocity_int is already denormalized to 0-127."""
        with self.lock:
            if not self.active:
                log_msg(f"[OSC] /noteon ch={ch} note={note} vel={velocity_int}/127 (capture inactive)")
                return
            elapsed_s = time.time() - self.start_time_s
            time_cs = int(elapsed_s * 100)
            self.midi_events.append((time_cs, 'note_on', note, velocity_int))
            log_msg(f"[Capture] NOTE-ON @{time_cs}cs: ch={ch} note={note} vel={velocity_int}/127")

    def record_noteoff(self, ch: int, note: int):
        """Record a MIDI note-off event."""
        with self.lock:
            if not self.active:
                log_msg(f"[OSC] /noteoff ch={ch} note={note} (capture inactive)")
                return
            elapsed_s = time.time() - self.start_time_s
            time_cs = int(elapsed_s * 100)
            self.midi_events.append((time_cs, 'note_off', note, 0))
            log_msg(f"[Capture] NOTE-OFF @{time_cs}cs: ch={ch} note={note}")

    def stop(self):
        """End capture and save files."""
        with self.lock:
            if not self.active:
                return
            self.active = False

            midi_path = self.output_dir / "output_000.mid"
            sched_path = self.output_dir / "schedule.json"
            meta_path = self.output_dir / "metadata.json"

            # Build the MIDI track from absolute timestamps: events must be written
            # in time order with *delta* ticks between them (mido.Message.time is a
            # delta, not absolute time) — sort first, then convert cs → ticks.
            mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage('set_tempo', tempo=MIDI_TEMPO, time=0))

            prev_ticks = 0
            for time_cs, kind, note, velocity in sorted(self.midi_events, key=lambda e: e[0]):
                ticks = round(mido.second2tick(time_cs / 100.0, MIDI_TICKS_PER_BEAT, MIDI_TEMPO))
                delta = max(0, ticks - prev_ticks)
                prev_ticks = ticks
                track.append(mido.Message(kind, note=note, velocity=velocity, time=delta))

            mid.save(str(midi_path))
            print(f"[Capture] Saved MIDI: {midi_path}")

            # Clip overlong notes (e.g. unmatched note-ons left ringing) before
            # plotting/synthesis so both reflect the same durations.
            clipped_count = clip_midi_note_durations(midi_path)
            print(f"[Capture] Clipped {clipped_count} note(s) longer than {MAX_NOTE_DURATION_S}s")

            # Render to audio for quick listening (writes to file only, never plays aloud)
            wav_path = self.output_dir / "output_000.wav"
            if synthesize_audio(midi_path, wav_path):
                print(f"[Capture] Saved audio: {wav_path}")

            # Save schedule as OSC events
            sched_data = [
                {"time_cs": time_cs, "address": addr, "args": args}
                for time_cs, addr, args in self.osc_events
            ]
            sched_path.write_text(json.dumps(sched_data, indent=2))
            print(f"[Capture] Saved schedule: {sched_path}")

            # Save metadata for compatibility with piano roll plotter
            elapsed_s = time.time() - self.start_time_s
            meta = {
                "experiment_id": "E1_live",
                "cluster": "live_capture",
                "prompt_stem": self.output_dir.name,
                "seed": 0,
                "end_cs": int(elapsed_s * 100),
                "prompt_boundary_cs": 0,  # No prompt in live capture
                "label": "live_controls",
                "intervention": "E1_live_capture",
                "note_count": sum(1 for e in self.midi_events if e[1] == 'note_on'),
                "control_events": len(sched_data),
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"[Capture] Saved metadata: {meta_path}")

            # Auto-generate piano roll visualization
            if plot_capture:
                try:
                    plot_capture(self.output_dir)
                except Exception as e:
                    print(f"[Capture] Warning: failed to auto-generate pianoroll: {e}")

            return str(self.output_dir)


# Global capture state (thread-safe)
capture_state = CaptureState()


class ReceivedLog:
    """Thread-safe ring buffer of recently received OSC messages, for display in the UI."""
    def __init__(self, maxlen=200):
        self.lock = threading.Lock()
        self.entries = []
        self.maxlen = maxlen
        self.seq = 0

    def add(self, address, args):
        with self.lock:
            self.seq += 1
            self.entries.append({"seq": self.seq, "t": time.time(), "address": address, "args": args})
            if len(self.entries) > self.maxlen:
                self.entries = self.entries[-self.maxlen:]

    def since(self, last_seq: int):
        with self.lock:
            return [e for e in self.entries if e["seq"] > last_seq]


# Global received-message log (thread-safe), polled by the browser
received_log = ReceivedLog()


# ---------------------------------------------------------------------------
# Embedded HTML (also written as controller.html for easy editing)
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OSC Controller</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #282828;
    color: #e8e8e8;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    padding: 24px 20px 190px;
    max-width: 780px;
    margin: 0 auto;
  }

  h1 {
    text-align: center;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 28px;
    color: #ccc;
  }

  section {
    background: #333;
    border-radius: 10px;
    padding: 18px 20px 16px;
    margin-bottom: 18px;
  }

  section h2 {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 14px;
  }

  /* ---- Pitch Range ---- */
  #pitchCanvas {
    display: block;
    width: 100%;
    height: 60px;
    cursor: crosshair;
    border-radius: 6px;
  }
  .pitch-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 10px;
  }
  .pitch-controls label {
    font-size: 0.88rem;
    color: #aaa;
  }
  #semitoneInput {
    width: 62px;
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid #555;
    background: #282828;
    color: #e8e8e8;
    font-size: 0.9rem;
    text-align: center;
  }
  .pitch-info {
    display: flex;
    gap: 20px;
    margin-top: 8px;
    font-size: 0.82rem;
  }
  #hand1Info { color: #f0883e; }
  #hand2Info { color: #6ab3f7; }

  /* ---- Button grids ---- */
  .btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }

  button {
    cursor: pointer;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: filter 0.1s, transform 0.08s;
    user-select: none;
  }
  button:active { transform: scale(0.95); filter: brightness(0.85); }

  .btn-runs   { background: #2563a8; color: #d0e8ff; }
  .btn-chords { background: #1e7a42; color: #c6f0d6; }
  .btn-reset  { background: #7a3a1e; color: #ffd6c6; }
  .btn-tempo-up   { background: #5a4b8a; color: #ddd4ff; }
  .btn-tempo-down { background: #8a6a2a; color: #fff0cc; }

  button:hover { filter: brightness(1.18); }

  /* ---- Status bar ---- */
  #statusBar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #1a1a1a;
    border-top: 1px solid #555;
    padding: 8px 18px;
    font-size: 0.92rem;
    color: #666;
    font-family: monospace;
    letter-spacing: 0.02em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  #statusBar.ok  { color: #4dbb88; }
  #statusBar.err { color: #f66; }

  /* ---- Received OSC log (sits above the sent-controls status bar) ---- */
  #receivedLog {
    position: fixed;
    bottom: 37px; left: 0; right: 0;
    max-height: 130px;
    overflow-y: auto;
    background: #1a1a1a;
    border-top: 1px solid #333;
    padding: 6px 18px;
    font-size: 0.82rem;
    font-family: monospace;
    color: #6ab3f7;
  }
  #receivedLog .row { white-space: nowrap; padding: 1px 0; }
  #receivedLog .row.noteoff { color: #4dbb88; }

  /* ---- Tempo section layout ---- */
  .tempo-label {
    font-size: 0.8rem;
    color: #888;
    margin-bottom: 6px;
  }
  .tempo-group { margin-bottom: 10px; }
  .tempo-group:last-child { margin-bottom: 0; }
</style>
</head>
<body>

<h1>OSC Controller</h1>

<!-- ================================================================ -->
<!-- 1. Pitch Range                                                    -->
<!-- ================================================================ -->
<section>
  <h2>Pitch Range</h2>
  <canvas id="pitchCanvas"></canvas>
  <div class="pitch-controls">
    <label for="semitoneInput">±&thinsp;semitones</label>
    <input type="number" id="semitoneInput" min="1" max="36" value="12">
  </div>
  <div class="pitch-info">
    <span id="hand1Info">Hand 1: —</span>
    <span id="hand2Info">Hand 2: —</span>
  </div>
</section>

<!-- ================================================================ -->
<!-- 2. Runs + Reset + Chords                                          -->
<!-- ================================================================ -->
<section>
  <h2>Runs &amp; Chords</h2>
  <div class="btn-row" style="margin-bottom:12px;">
    <button class="btn-runs"   onclick="sendRuns(0)">VeryFast (L0)</button>
    <button class="btn-runs"   onclick="sendRuns(1)">Fast (L1)</button>
    <button class="btn-runs"   onclick="sendRuns(2)">Moderate (L2)</button>
    <button class="btn-runs"   onclick="sendRuns(3)">Slow (L3)</button>
    <button class="btn-reset"  onclick="sendReset()">Reset</button>
  </div>
  <div class="btn-row">
    <button class="btn-chords" onclick="sendChords(0)">Fast (L0)</button>
    <button class="btn-chords" onclick="sendChords(1)">Moderate (L1)</button>
    <button class="btn-chords" onclick="sendChords(2)">Slow (L2)</button>
  </div>
</section>

<!-- ================================================================ -->
<!-- 3. Adjust Tempo                                                   -->
<!-- ================================================================ -->
<section>
  <h2>Adjust Tempo</h2>
  <div class="tempo-group">
    <div class="tempo-label">Speed Up</div>
    <div class="btn-row">
      <button class="btn-tempo-up" onclick="sendTempo(1.2)">×1.2</button>
      <button class="btn-tempo-up" onclick="sendTempo(1.4)">×1.4</button>
      <button class="btn-tempo-up" onclick="sendTempo(1.7)">×1.7</button>
      <button class="btn-tempo-up" onclick="sendTempo(2.0)">×2.0</button>
    </div>
  </div>
  <div class="tempo-group">
    <div class="tempo-label">Slow Down</div>
    <div class="btn-row">
      <button class="btn-tempo-down" onclick="sendTempo(0.8)">×0.8</button>
      <button class="btn-tempo-down" onclick="sendTempo(0.7)">×0.7</button>
      <button class="btn-tempo-down" onclick="sendTempo(0.6)">×0.6</button>
      <button class="btn-tempo-down" onclick="sendTempo(0.5)">×0.5</button>
    </div>
  </div>
</section>

<!-- ================================================================ -->
<!-- Capture Controls                                                  -->
<!-- ================================================================ -->
<section style="background: #2a3a2a;">
  <h2>Capture Playback</h2>
  <p style="font-size: 0.9rem; color: #aaa; margin-bottom: 10px;">
    Records both MIDI output and OSC controls with synchronized timing.
    Saves to ~/haires/workspace/control_runs/E1_live/
  </p>
  <div class="btn-row">
    <button style="background: #2a5a2a;" onclick="startCapture()">⏺ Start Capture</button>
    <button style="background: #8a3a2a;" onclick="stopCapture()">⏹ Stop & Save</button>
  </div>
  <div id="captureStatus" style="font-size: 0.85rem; color: #8a8; margin-top: 8px;"></div>
</section>

<div id="receivedLog"></div>
<div id="statusBar">Ready</div>

<script>
// ------------------------------------------------------------------ //
// MIDI note name helper: C0 = 12, C4 = 60                            //
// ------------------------------------------------------------------ //
const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function midiName(n) {
  const octave = Math.floor(n / 12) - 1;
  const name   = NOTE_NAMES[n % 12];
  return `${name}${octave}`;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ------------------------------------------------------------------ //
// Canvas pitch slider                                                 //
// ------------------------------------------------------------------ //
const canvas = document.getElementById('pitchCanvas');
const ctx    = canvas.getContext('2d');

// MIDI range shown on track
const MIDI_LO = 21;   // A0
const MIDI_HI = 108;  // C8

// Handle state: center MIDI notes for each hand
const handles = [60, 79];   // hand1=C4, hand2=G5
const COLORS  = ['#f0883e', '#6ab3f7'];
let linkedMode = false;      // true = single merged handle
const BAND_ALPHA = 0.28;

const semitoneInput = document.getElementById('semitoneInput');
const hand1Info     = document.getElementById('hand1Info');
const hand2Info     = document.getElementById('hand2Info');

function getSemitones() {
  const v = parseInt(semitoneInput.value, 10);
  return isNaN(v) ? 12 : clamp(v, 1, 36);
}

// Map MIDI note → canvas x pixel
function noteToX(note, w) {
  return (note - MIDI_LO) / (MIDI_HI - MIDI_LO) * w;
}
// Map canvas x → MIDI note (float)
function xToNote(x, w) {
  return MIDI_LO + (x / w) * (MIDI_HI - MIDI_LO);
}

// Octave C notes visible in range (C2=36 … C8=108)
const OCTAVE_MARKERS = [];
for (let midi = 36; midi <= 108; midi += 12) OCTAVE_MARKERS.push(midi);

function drawSlider() {
  const rect = canvas.getBoundingClientRect();
  const W = rect.width;
  const H = rect.height;
  const sem = getSemitones();

  ctx.clearRect(0, 0, W, H);

  // Track background
  const trackY = 28, trackH = 10;
  ctx.fillStyle = '#1a1a1a';
  ctx.beginPath();
  ctx.roundRect(0, trackY, W, trackH, 4);
  ctx.fill();

  // Draw range bands and handles
  const handleR = 9;
  const handleY = trackY + trackH / 2;

  if (linkedMode) {
    const lo = clamp(handles[0] - sem, MIDI_LO, MIDI_HI);
    const hi = clamp(handles[0] + sem, MIDI_LO, MIDI_HI);
    ctx.globalAlpha = BAND_ALPHA;
    ctx.fillStyle = '#ccc';
    ctx.beginPath();
    ctx.roundRect(noteToX(lo, W), trackY, noteToX(hi, W) - noteToX(lo, W), trackH, 4);
    ctx.fill();
    ctx.globalAlpha = 1;
  } else {
    handles.forEach((center, i) => {
      const lo = clamp(center - sem, MIDI_LO, MIDI_HI);
      const hi = clamp(center + sem, MIDI_LO, MIDI_HI);
      ctx.globalAlpha = BAND_ALPHA;
      ctx.fillStyle = COLORS[i];
      ctx.beginPath();
      ctx.roundRect(noteToX(lo, W), trackY, noteToX(hi, W) - noteToX(lo, W), trackH, 4);
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  }

  // Octave markers
  ctx.fillStyle = '#555';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  OCTAVE_MARKERS.forEach(midi => {
    const x = noteToX(midi, W);
    ctx.fillStyle = '#555';
    ctx.fillRect(x - 0.5, trackY - 5, 1, 5);
    ctx.fillStyle = '#666';
    ctx.fillText(midiName(midi), x, trackY - 7);
  });

  // Handles
  if (linkedMode) {
    const x = noteToX(handles[0], W);
    ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.arc(x, handleY, handleR, 0, Math.PI * 2);
    ctx.fillStyle = '#ddd';
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#222';
    ctx.font = 'bold 8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('L', x, handleY + 3);
  } else {
    handles.forEach((center, i) => {
      const x = noteToX(center, W);
      ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.arc(x, handleY, handleR, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[i];
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#111';
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(i + 1, x, handleY + 3.5);
    });
  }

  // Info labels
  updateInfoLabels(sem);
}

function updateInfoLabels(sem) {
  sem = sem ?? getSemitones();
  if (linkedMode) {
    const lo = clamp(handles[0] - sem, 0, 127);
    const hi = clamp(handles[0] + sem, 0, 127);
    hand1Info.textContent = `Linked: ${midiName(handles[0])} (${handles[0]}) → [${lo}, ${hi}]`;
    hand2Info.textContent = '';
  } else {
    [handles[0], handles[1]].forEach((center, i) => {
      const lo = clamp(center - sem, 0, 127);
      const hi = clamp(center + sem, 0, 127);
      const label = `Hand ${i+1}: ${midiName(center)} (${center}) → [${lo}, ${hi}]`;
      if (i === 0) hand1Info.textContent = label;
      else         hand2Info.textContent = label;
    });
  }
}

// Throttle pitch sends to at most once per 500 ms
let pitchThrottleTimer = null;
let pitchPending = false;

function sendPitchRange() {
  if (pitchThrottleTimer !== null) {
    pitchPending = true;  // will fire when the current window expires
    return;
  }
  _flushPitchRange();
  pitchThrottleTimer = setTimeout(() => {
    pitchThrottleTimer = null;
    if (pitchPending) { pitchPending = false; _flushPitchRange(); }
  }, 500);
}

function getCurrentRangeArgs() {
  const sem = getSemitones();
  if (linkedMode) {
    const lo = clamp(handles[0] - sem, 0, 127);
    const hi = clamp(handles[0] + sem, 0, 127);
    return [lo, hi, lo, hi];
  }
  const lo1 = clamp(handles[0] - sem, 0, 127);
  const hi1 = clamp(handles[0] + sem, 0, 127);
  const lo2 = clamp(handles[1] - sem, 0, 127);
  const hi2 = clamp(handles[1] + sem, 0, 127);
  return [lo1, hi1, lo2, hi2];
}

function _flushPitchRange() {
  sendOSC('/setOutputRange', getCurrentRangeArgs());
}

// ---- Sizing: keep canvas pixel dims in sync with CSS layout ----
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr  = window.devicePixelRatio || 1;
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);  // reset + set scale each time
  drawSlider();
}

// ---- Drag handling ----
let dragging        = null;   // handle index being dragged
let dragStartX      = null;   // canvas-x where press began
let dragStartHandle = null;   // handle index chosen at press
const DRAG_THRESHOLD = 4;     // px before drag activates

function pickHandle(ex) {
  if (linkedMode) return 0;
  const W = canvas.getBoundingClientRect().width;
  const dists = handles.map(c => Math.abs(noteToX(c, W) - ex));
  return dists[0] <= dists[1] ? 0 : 1;
}

function pointerX(e) {
  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  return clientX - rect.left;
}

canvas.addEventListener('mousedown', e => {
  e.preventDefault();
  dragStartX      = pointerX(e);
  dragStartHandle = pickHandle(dragStartX);
  dragging        = null;
});
canvas.addEventListener('touchstart', e => {
  e.preventDefault();
  dragStartX      = pointerX(e);
  dragStartHandle = pickHandle(dragStartX);
  dragging        = null;
}, { passive: false });

function moveDrag(e) {
  const x    = pointerX(e);
  const note = Math.round(clamp(xToNote(x, canvas.getBoundingClientRect().width), MIDI_LO, MIDI_HI));
  handles[dragging] = note;
  if (linkedMode) handles[1] = handles[0];
  drawSlider();
  sendPitchRange();
}

window.addEventListener('mousemove', e => {
  if (dragStartHandle === null) return;
  if (dragging === null) {
    if (Math.abs(pointerX(e) - dragStartX) >= DRAG_THRESHOLD)
      dragging = dragStartHandle;
  }
  if (dragging !== null) moveDrag(e);
});
window.addEventListener('mouseup', () => {
  dragging = null; dragStartX = null; dragStartHandle = null;
});
window.addEventListener('touchmove', e => {
  if (dragStartHandle === null) return;
  if (dragging === null) {
    if (Math.abs(pointerX(e) - dragStartX) >= DRAG_THRESHOLD)
      dragging = dragStartHandle;
  }
  if (dragging !== null) { e.preventDefault(); moveDrag(e); }
}, { passive: false });
window.addEventListener('touchend', () => {
  dragging = null; dragStartX = null; dragStartHandle = null;
});

// ---- Double-click: toggle linked / two-handle mode ----
canvas.addEventListener('dblclick', e => {
  e.preventDefault();
  const x = pointerX(e);
  if (linkedMode) {
    // Expand: both handles start at current single position
    handles[1] = handles[0];
    linkedMode = false;
  } else {
    // Collapse: keep handle closest to click
    const closest = pickHandle(x);
    handles[0] = handles[closest];
    linkedMode = true;
  }
  drawSlider();
  sendPitchRange();
});

semitoneInput.addEventListener('input',  () => { drawSlider(); sendPitchRange(); });
semitoneInput.addEventListener('change', () => { drawSlider(); sendPitchRange(); });

// Initial render — defer until layout is complete so getBoundingClientRect is right
window.addEventListener('load', () => {
  resizeCanvas();
  sendPitchRange();
});
window.addEventListener('resize', resizeCanvas);

// ------------------------------------------------------------------ //
// Action senders                                                      //
// ------------------------------------------------------------------ //
let runsOrChordsActive = false;

function sendRuns(level)   { runsOrChordsActive = true;  sendOSC('/playRuns',   [level]); }
function sendChords(level) { runsOrChordsActive = true;  sendOSC('/playChords', [level]); }
function sendReset()       { runsOrChordsActive = false; sendOSC('/resetControl', []); }

async function sendTempo(ratio) {
  if (runsOrChordsActive) {
    await sendOSC('/resetControl', []);
    runsOrChordsActive = false;
  }
  sendOSC('/adjustTempo', [ratio]);
}

// ------------------------------------------------------------------ //
// Capture controls                                                    //
// ------------------------------------------------------------------ //
const captureStatus = document.getElementById('captureStatus');

async function startCapture() {
  try {
    const res = await fetch('/capture/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address: '/setOutputRange', args: getCurrentRangeArgs() }),
    });
    if (res.ok) {
      const data = await res.json();
      captureStatus.textContent = `⏹ Capturing to ${data.output_dir}`;
      captureStatus.style.color = '#8f8';
    } else {
      captureStatus.textContent = `Error: ${res.status}`;
      captureStatus.style.color = '#f88';
    }
  } catch (e) {
    captureStatus.textContent = `Error: ${e.message}`;
    captureStatus.style.color = '#f88';
  }
}

async function stopCapture() {
  try {
    const res = await fetch('/capture/stop', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      captureStatus.textContent = `✓ Saved to ${data.output_dir}`;
      captureStatus.style.color = '#8f8';
    } else {
      captureStatus.textContent = `Error: ${res.status}`;
      captureStatus.style.color = '#f88';
    }
  } catch (e) {
    captureStatus.textContent = `Error: ${e.message}`;
    captureStatus.style.color = '#f88';
  }
}

// ------------------------------------------------------------------ //
// POST to Python bridge                                               //
// ------------------------------------------------------------------ //
const statusBar = document.getElementById('statusBar');
let statusTimer = null;

function setStatus(msg, cls) {
  statusBar.textContent = msg;
  statusBar.className = cls || '';
  // Only auto-clear errors, not successful sends
  clearTimeout(statusTimer);
  if (cls === 'err') {
    statusTimer = setTimeout(() => {
      statusBar.textContent = 'Ready';
      statusBar.className = '';
    }, 4000);
  }
}

async function sendOSC(address, args) {
  const argsStr = args.map(a => (typeof a === 'number' && !Number.isInteger(a))
    ? a.toFixed(2) : String(a)).join(' ');
  const cmdStr = args.length ? `${address}  ${argsStr}` : address;
  const body = JSON.stringify({ address, args });
  try {
    const res = await fetch('/osc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    });
    if (res.ok) {
      setStatus(cmdStr, 'ok');
    } else {
      const txt = await res.text();
      setStatus(`Error ${res.status}: ${txt}`, 'err');
    }
  } catch (e) {
    setStatus(`Network error: ${e.message}`, 'err');
  }
}

// ------------------------------------------------------------------ //
// Received OSC log — polls /received for /noteon, /noteoff, etc.     //
// from JordanAI and prints them above the sent-controls status bar.  //
// ------------------------------------------------------------------ //
const receivedLog = document.getElementById('receivedLog');
let lastSeq = 0;

async function pollReceived() {
  try {
    const res = await fetch(`/received?since=${lastSeq}`);
    if (!res.ok) return;
    const data = await res.json();
    for (const e of data.entries) {
      lastSeq = Math.max(lastSeq, e.seq);
      const argsStr = e.args.map(a => (typeof a === 'number' && !Number.isInteger(a))
        ? a.toFixed(2) : String(a)).join(' ');
      const row = document.createElement('div');
      row.className = 'row' + (e.address === '/noteoff' ? ' noteoff' : '');
      row.textContent = `${e.address}  ${argsStr}`;
      receivedLog.appendChild(row);
      while (receivedLog.children.length > 50) receivedLog.removeChild(receivedLog.firstChild);
    }
    if (data.entries.length) receivedLog.scrollTop = receivedLog.scrollHeight;
  } catch (e) {
    // ignore network errors during polling
  }
}
setInterval(pollReceived, 300);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class OSCBridgeHandler(BaseHTTPRequestHandler):
    """Serves the HTML page and handles POST /osc requests."""

    # injected by server setup
    osc_client: udp_client.SimpleUDPClient = None

    def log_message(self, fmt, *args):
        # Suppress per-request access log; errors still printed via log_error
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html', '/controller.html'):
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif urlparse(self.path).path == '/received':
            self._handle_received()
        else:
            self.send_error(404, 'Not found')

    def _handle_received(self):
        """Return recently received OSC messages (e.g. /noteon, /noteoff) for UI polling."""
        query = parse_qs(urlparse(self.path).query)
        since = int(query.get('since', ['0'])[0])
        entries = received_log.since(since)
        body = json.dumps({"entries": entries}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == '/osc':
            self._handle_osc()
        elif self.path == '/capture/start':
            self._handle_capture_start()
        elif self.path == '/capture/stop':
            self._handle_capture_stop()
        else:
            self.send_error(404, 'Not found')

    def _handle_osc(self):
        """Forward OSC messages to the live app and record if capturing."""
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
            address = str(data['address'])
            args = data.get('args', [])
        except (json.JSONDecodeError, KeyError) as e:
            self.send_error(400, f'Bad JSON: {e}')
            return

        # Record to capture if active
        capture_state.record_osc(address, args)

        # Forward as OSC to live app
        try:
            if args:
                self.osc_client.send_message(address, args)
            else:
                self.osc_client.send_message(address, [])
            print(f'OSC  {address}  {args}')
        except Exception as e:
            print(f'OSC ERROR {e}')
            self.send_error(500, str(e))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'ok')

    def _handle_capture_start(self):
        """Start capturing MIDI + OSC events."""
        length = int(self.headers.get('Content-Length', 0))
        initial_range = None
        if length:
            try:
                data = json.loads(self.rfile.read(length))
                initial_range = (str(data['address']), data.get('args', []))
            except (json.JSONDecodeError, KeyError):
                initial_range = None

        output_dir = Path.home() / 'haires/workspace/control_runs/E1_live'
        capture_state.start(output_dir)
        if initial_range:
            # Record the pitch range that was already active so the schedule
            # has a time_cs=0 entry instead of waiting for the next slider move.
            capture_state.record_osc(*initial_range)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        body = json.dumps({"status": "started", "output_dir": str(output_dir)}).encode()
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_capture_stop(self):
        """Stop capturing and save files."""
        output_dir = capture_state.stop()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        body = json.dumps({"status": "stopped", "output_dir": output_dir or ""}).encode()
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


def setup_osc_listener(listen_port: int):
    """Setup OSC listener to capture /noteon and /noteoff from live app."""
    disp = dispatcher.Dispatcher()

    def handle_noteon(unused_addr, ch, note, velocity):
        # velocity arrives as a normalized float (0.0-1.0) from JUCE's MidiMessage;
        # denormalize by multiplying, do NOT int() it first (that truncates to 0).
        velocity_int = int(round(velocity * 127)) if isinstance(velocity, float) else int(velocity)
        log_msg(f"[DEBUG] /noteon raw args: ch={ch} (type={type(ch).__name__}), note={note} (type={type(note).__name__}), velocity={velocity} (type={type(velocity).__name__}) -> {velocity_int}/127")
        received_log.add("/noteon", [int(ch), int(note), velocity_int])
        capture_state.record_noteon(int(ch), int(note), velocity_int)

    def handle_noteoff(unused_addr, ch, note):
        received_log.add("/noteoff", [int(ch), int(note)])
        capture_state.record_noteoff(int(ch), int(note))

    disp.map("/noteon", handle_noteon)
    disp.map("/noteoff", handle_noteoff)

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", listen_port), disp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log_msg(f'OSC listener : 127.0.0.1:{listen_port} (for /noteon, /noteoff capture)')
    return server


def main():
    parser = argparse.ArgumentParser(description='OSC web controller bridge')
    parser.add_argument('--host',          default='127.0.0.1',     help='OSC target host')
    parser.add_argument('--port',          default=9001, type=int,  help='OSC target port (JordanAI)')
    parser.add_argument('--http-port',     default=8765, type=int,  help='HTTP server port')
    parser.add_argument('--http-host',     default='0.0.0.0',       help='HTTP bind address')
    parser.add_argument('--capture-port',  default=9002, type=int,  help='OSC listen port for MIDI capture')
    args = parser.parse_args()

    osc_client = udp_client.SimpleUDPClient(args.host, args.port)
    print(f'OSC target   : {args.host}:{args.port}')

    # Setup OSC listener for capturing /noteon and /noteoff
    setup_osc_listener(args.capture_port)

    # Inject osc_client into handler class
    OSCBridgeHandler.osc_client = osc_client

    # Also write controller.html next to this script for easy editing
    html_path = Path(__file__).parent / 'controller.html'
    html_path.write_text(HTML, encoding='utf-8')

    server = HTTPServer((args.http_host, args.http_port), OSCBridgeHandler)

    local_ip = get_local_ip()
    print(f'HTTP server  : http://{local_ip}:{args.http_port}/')
    print(f'              http://localhost:{args.http_port}/')
    print('\nWorkflow:')
    print('  1. Enable OSC output in JordanAI → localhost:' + str(args.capture_port))
    print('  2. Click "Start Capture" button')
    print('  3. Make OSC control changes via web UI')
    print('  4. Click "Stop & Save"')
    print('  5. Results saved with synchronized MIDI + schedule.json')
    print('\nPress Ctrl+C to stop.')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
