"""
Session control panel for the Thai P300 speller.

One window, three things:
  1. Live signal quality  -- scrolling 8-channel EEG trace + a scalp map
     whose dots go green/yellow/red so bad electrode contact is visible
     BEFORE a session starts, not discovered in analysis afterwards.
  2. Target entry         -- type the characters to spell/calibrate on,
     press Enter (or Start) to run one cued trial per character.
  3. Settings             -- repetitions per character, flash/ISI timing.

Pressing Start launches record_session.py, waits for it to latch onto
the EEG outlet, then launches checkerboard_stimulus.py with the typed
target string -- automating the ordered two-terminal startup that
README_ACQUISITION.md otherwise describes by hand.

This GUI is a pure LSL consumer, like record_session.py: it needs the
Unicorn LSL Interface already streaming (no BrainFlow, no direct hardware
access -- see README_ACQUISITION.md). Deployment runs it on the same
Windows machine as the Interface; an older split-machine setup still
works unchanged, since streams are resolved by type, not host.

Run with:
    conda activate p300speller
    python collection/session_gui.py
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from tkinter import ttk

from pylsl import StreamInlet, resolve_byprop

from record_session import EEG_CHANNELS, SANITY_PTP_MIN_UV, SANITY_PTP_MAX_UV

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Scalp positions, top-down, nose at +y, laid out to fill the brain
# silhouette rather than to sit on a unit-radius 10-20 circle (which only
# covers the middle of the drawing and left the dots huddled around the
# fissure).
#
# Real montage: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 -- the Unicorn Hybrid
# Black's factory-default cap (see pipeline/constants.EEG_CHANNELS for the
# verification sources; CLAUDE.md's hard constraints record the correction).
# Four rows, evenly spaced front-to-back (0.60 / 0.12 / -0.36 / -0.84):
# Fz alone; C3/Cz/C4 on one coronal line; Pz alone, more posterior; Oz at
# the occipital pole, the most posterior point, on the midline. PO7/PO8 sit
# between the Pz and Oz rows and more lateral than C3/C4 -- parieto-
# occipital sites are more lateral than central ones in the real 10-20
# layout, not a mirror of the central row. An electrode map that is
# anatomically wrong is worse than none, since it invites reading "the
# left-central electrode is bad" off a dot that isn't left-central.
# Kept in sync with gui_logic.ELECTRODE_POS (a test asserts they are equal).
ELECTRODE_POS = {
    "Fz": (0.00, 0.60),
    "C3": (-0.48, 0.12),
    "Cz": (0.00, 0.12),
    "C4": (0.48, 0.12),
    "Pz": (0.00, -0.36),
    "PO7": (-0.58, -0.68),
    "Oz": (0.00, -0.84),
    "PO8": (0.58, -0.68),
}

# Trace display order: front -> back, left -> midline -> right within a
# level. Happens to equal EEG_CHANNELS' order for this montage (the
# Unicorn's default cap is already wired front-to-back), but the two stay
# separate constants: EEG_CHANNELS is stream indexing and must not move if
# a future montage's wire order isn't front-to-back, while this one is
# hand-curated for how the trace stack reads.
DISPLAY_ORDER = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

COLOR_BG = "#fafafa"
COLOR_FG = "#1a1a1a"
COLOR_MUTED = "#6b7280"
COLOR_TRACE = "#374151"
COLOR_BRAIN = "#d6dade"
QUALITY_GREEN = "#22c55e"
QUALITY_YELLOW = "#eab308"
QUALITY_RED = "#ef4444"
QUALITY_GRAY = "#d1d5db"

DISPLAY_WINDOW_S = 4.0
QUALITY_WINDOW_S = 1.0
REFRESH_MS = 100


def classify_quality(ptp_uv: float) -> str:
    """
    Rolling peak-to-peak amplitude (uV) -> quality color. Reuses
    record_session.py's SANITY_PTP_MIN/MAX_UV as the green band so this
    display and the recorder's pre-session check cannot silently drift
    apart on what counts as plausible EEG.
    """
    if ptp_uv < SANITY_PTP_MIN_UV:
        return QUALITY_RED       # flat / disconnected
    if ptp_uv > SANITY_PTP_MAX_UV * 2:
        return QUALITY_RED       # railed / clipped
    if ptp_uv > SANITY_PTP_MAX_UV:
        return QUALITY_YELLOW    # noisy
    return QUALITY_GREEN


def load_matrix_characters() -> list[str]:
    """Ordered 72-character list -- the single source of truth for what can be typed."""
    with open(os.path.join(THIS_DIR, "characters_th.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["characters"] if isinstance(data, dict) else data


class EEGStreamReader(threading.Thread):
    """
    Background thread: resolve the EEG outlet (retrying, so the GUI is
    usable before the Unicorn LSL Interface is up) and buffer samples
    per channel for the redraw timer.
    """

    def __init__(self, status_queue: queue.Queue, sfreq: float = 250.0):
        super().__init__(daemon=True)
        self.status_queue = status_queue
        self.sfreq = sfreq
        self._stop_event = threading.Event()
        maxlen = int(DISPLAY_WINDOW_S * sfreq) + 50
        self.buffers = {ch: deque(maxlen=maxlen) for ch in EEG_CHANNELS}
        self._lock = threading.Lock()
        self.connected = False

    def stop(self):
        self._stop_event.set()

    def snapshot(self) -> dict:
        with self._lock:
            return {ch: np.array(buf) for ch, buf in self.buffers.items()}

    def run(self):
        inlet = None
        while not self._stop_event.is_set() and inlet is None:
            streams = resolve_byprop("type", "EEG", timeout=1.0)
            if streams:
                inlet = StreamInlet(streams[0])
                self.connected = True
                self.status_queue.put("connected")
            else:
                self.status_queue.put("waiting")
                time.sleep(1.0)

        while not self._stop_event.is_set():
            chunk, _ts = inlet.pull_chunk(timeout=0.2)
            if not chunk:
                continue
            with self._lock:
                for sample in chunk:
                    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
                        if ch_idx < len(sample):
                            self.buffers[ch_name].append(sample[ch_idx])


class SessionGUI(tk.Tk):
    def __init__(self, demo: bool = False):
        super().__init__()
        self.demo = demo
        self.title("Thai P300 Speller" + (" [DEMO]" if demo else ""))
        self.geometry("1080x680")
        self.configure(bg=COLOR_BG)

        self.characters = load_matrix_characters()
        self.status_queue: queue.Queue = queue.Queue()
        self.reader = EEGStreamReader(self.status_queue)
        self.reader.start()

        self._recorder_proc: subprocess.Popen | None = None
        self._stimulus_proc: subprocess.Popen | None = None

        self._build_layout()
        self.after(REFRESH_MS, self._on_tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        style = ttk.Style(self)
        style.configure("TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_FG)
        style.configure("Muted.TLabel", background=COLOR_BG, foreground=COLOR_MUTED)
        style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_FG,
                        font=("Helvetica", 15, "bold"))
        style.configure("Status.TLabel", background=COLOR_BG, font=("Helvetica", 11))

        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="Thai P300 Speller", style="Title.TLabel").pack(side=tk.LEFT)
        if self.demo:
            tk.Label(header, text="DEMO MODE \u2014 synthetic data",
                     bg="#fef3c7", fg="#92400e", font=("Helvetica", 10, "bold"),
                     padx=8, pady=2).pack(side=tk.LEFT, padx=(12, 0))
        self.status_var = tk.StringVar(value="● Searching for EEG stream")
        self.status_label = ttk.Label(header, textvariable=self.status_var,
                                       style="Status.TLabel", foreground=COLOR_MUTED)
        self.status_label.pack(side=tk.RIGHT)

        # --- plots ---
        self.fig = plt.Figure(figsize=(10, 4.2), facecolor=COLOR_BG)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.12)
        self.ax_trace = self.fig.add_subplot(gs[0, 0])
        self.ax_head = self.fig.add_subplot(gs[0, 1])
        self._init_trace_axes()
        self._init_head_axes()
        # subplots_adjust, not tight_layout: the head axes is aspect-locked,
        # which tight_layout cannot lay out correctly (it warns and guesses).
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- controls ---
        controls = ttk.Frame(root, padding=(0, 12, 0, 0))
        controls.pack(fill=tk.X)

        row1 = ttk.Frame(controls)
        row1.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row1, text="Target characters").pack(side=tk.LEFT)
        self.targets_var = tk.StringVar()
        self.targets_entry = tk.Entry(row1, textvariable=self.targets_var, width=28,
                                       font=("Helvetica", 16), relief=tk.SOLID, bd=1)
        self.targets_entry.pack(side=tk.LEFT, padx=(10, 10))
        self.targets_entry.bind("<Return>", lambda _e: self._on_start())
        self.targets_entry.focus_set()

        self.start_button = tk.Button(row1, text="Start", command=self._on_start,
                                       font=("Helvetica", 12, "bold"), width=10,
                                       relief=tk.FLAT, bg="#2563eb", fg="white",
                                       activebackground="#1d4ed8", activeforeground="white",
                                       highlightbackground=COLOR_BG)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(row1, text="Stop", command=self._on_stop,
                                      font=("Helvetica", 12), width=8, relief=tk.FLAT,
                                      state=tk.DISABLED, highlightbackground=COLOR_BG)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="Session ID", style="Muted.TLabel").pack(side=tk.LEFT)
        self.subject_var = tk.StringVar(value="subject_001")
        ttk.Entry(row2, textvariable=self.subject_var, width=14).pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row2, text="Repetitions", style="Muted.TLabel").pack(side=tk.LEFT)
        self.reps_var = tk.IntVar(value=5)
        ttk.Spinbox(row2, from_=1, to=50, textvariable=self.reps_var,
                    width=5).pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row2, text="Flash ms", style="Muted.TLabel").pack(side=tk.LEFT)
        self.flash_var = tk.IntVar(value=62)
        ttk.Spinbox(row2, from_=10, to=500, textvariable=self.flash_var,
                    width=5).pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row2, text="ISI ms", style="Muted.TLabel").pack(side=tk.LEFT)
        self.isi_var = tk.IntVar(value=62)
        ttk.Spinbox(row2, from_=10, to=500, textvariable=self.isi_var,
                    width=5).pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row2, text="Screen", style="Muted.TLabel").pack(side=tk.LEFT)
        self.screen_var = tk.IntVar(value=0)
        ttk.Spinbox(row2, from_=0, to=3, textvariable=self.screen_var,
                    width=4).pack(side=tk.LEFT, padx=(6, 18))

        self.windowed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Windowed", variable=self.windowed_var).pack(side=tk.LEFT)

        self.session_var = tk.StringVar(value="Idle")
        ttk.Label(controls, textvariable=self.session_var, style="Muted.TLabel").pack(
            anchor=tk.W, pady=(10, 0))

    def _init_trace_axes(self):
        ax = self.ax_trace
        ax.set_facecolor(COLOR_BG)
        ax.set_title("Live EEG", fontsize=10, color=COLOR_FG, loc="left")
        ax.set_xlabel("seconds", fontsize=8, color=COLOR_MUTED)
        ax.set_xlim(0, DISPLAY_WINDOW_S)
        ax.set_ylim(-len(DISPLAY_ORDER), 1)
        ax.set_yticks([-i for i in range(len(DISPLAY_ORDER))])
        ax.set_yticklabels(DISPLAY_ORDER, fontsize=8, color=COLOR_FG)
        ax.tick_params(axis="x", labelsize=8, colors=COLOR_MUTED)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5,
                color="#cbd5e1", alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(COLOR_MUTED)

        self.trace_lines = {}
        for i, ch in enumerate(DISPLAY_ORDER):
            (line,) = ax.plot([], [], linewidth=0.7, color=COLOR_TRACE)
            self.trace_lines[ch] = (line, -i)

    @staticmethod
    def _brain_hemisphere(side: int, gap: float = 0.05,
                           r_x: float = 0.95, r_y: float = 1.0) -> np.ndarray:
        """
        Polygon outline for one brain hemisphere, as a lobed half-ellipse:
        a half-ellipse whose radius is modulated by a small sinusoid to
        suggest gyri, closed along a straight vertical inner edge so the
        two hemispheres leave a center gap (the longitudinal fissure).

        side: +1 for the right hemisphere, -1 for the left (mirrored).
        """
        t = np.linspace(0, np.pi, 400)
        bump = 1 + 0.035 * np.sin(11 * t)
        x = side * (gap + r_x * np.sin(t) * bump)
        y = r_y * np.cos(t) * bump
        return np.column_stack([x, y])

    def _init_head_axes(self):
        ax = self.ax_head
        ax.set_facecolor(COLOR_BG)
        ax.set_title("Signal quality", fontsize=10, color=COLOR_FG, loc="left")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-2.05, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        for side in (-1, 1):
            ax.add_patch(plt.Polygon(self._brain_hemisphere(side), closed=True,
                                      facecolor=COLOR_BRAIN, edgecolor="none", zorder=1))

        self.head_dots = {}
        for ch, (x, y) in ELECTRODE_POS.items():
            dot = ax.scatter([x], [y], s=90, c=QUALITY_GRAY, edgecolors="none", zorder=3)
            ax.annotate(ch, (x, y + 0.13), ha="center", va="bottom",
                        fontsize=7, color=COLOR_FG, zorder=4)
            self.head_dots[ch] = dot

        # Legend, laid out below the brain so the colors are self-explanatory
        # without a caption elsewhere in the window.
        for i, (color, label) in enumerate([
            (QUALITY_GREEN, "Good signal"),
            (QUALITY_YELLOW, "Noisy"),
            (QUALITY_RED, "Bad / flat"),
        ]):
            y = -1.35 - i * 0.22
            ax.scatter([-0.75], [y], s=70, c=color, edgecolors="none", zorder=3)
            ax.text(-0.55, y, label, fontsize=8, color=COLOR_FG,
                    ha="left", va="center")

    # ------------------------------------------------------------------
    # Redraw loop
    # ------------------------------------------------------------------
    def _on_tick(self):
        self._drain_status_queue()
        if self.reader.connected:
            snapshot = self.reader.snapshot()
            self._update_trace(snapshot)
            self._update_head(snapshot)
            self.canvas.draw_idle()
        self._poll_subprocesses()
        self.after(REFRESH_MS, self._on_tick)

    def _drain_status_queue(self):
        try:
            while True:
                kind = self.status_queue.get_nowait()
                if kind == "waiting":
                    self.status_var.set("● No EEG stream — start Unicorn LSL Interface")
                    self.status_label.configure(foreground=QUALITY_RED)
                elif kind == "connected":
                    self.status_var.set("● EEG connected")
                    self.status_label.configure(foreground=QUALITY_GREEN)
        except queue.Empty:
            pass

    def _update_trace(self, snapshot: dict):
        n_max = int(DISPLAY_WINDOW_S * self.reader.sfreq)
        for ch in DISPLAY_ORDER:
            arr = snapshot.get(ch, np.array([]))
            line, offset = self.trace_lines[ch]
            if arr.size == 0:
                continue
            y = arr[-n_max:]
            # Per-channel normalization so all 8 traces stay legible on one
            # shared axis regardless of relative amplitude.
            centered = y - y.mean()
            spread = max(np.std(centered), 1e-6)
            line.set_data(np.linspace(0, y.size / self.reader.sfreq, y.size),
                          centered / (spread * 4.0) + offset)

    def _update_head(self, snapshot: dict):
        n_quality = int(QUALITY_WINDOW_S * self.reader.sfreq)
        for ch, dot in self.head_dots.items():
            arr = snapshot.get(ch, np.array([]))
            if arr.size < 5:
                dot.set_color(QUALITY_GRAY)
                continue
            window = arr[-n_quality:] if arr.size >= n_quality else arr
            dot.set_color(classify_quality(float(window.max() - window.min())))

    def _poll_subprocesses(self):
        if self._stimulus_proc is not None and self._stimulus_proc.poll() is not None:
            self._stimulus_proc = None
            self.session_var.set("Stimulus finished — stopping recorder.")
            self._on_stop()
        if self._recorder_proc is not None and self._recorder_proc.poll() is not None:
            self._recorder_proc = None
            self._set_running(False)

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------
    def _on_start(self):
        if self._recorder_proc is not None or self._stimulus_proc is not None:
            return

        targets = self.targets_var.get().strip()
        if not targets:
            self.session_var.set("Type the characters to spell first.")
            return
        unknown = [c for c in targets if c not in self.characters]
        if unknown:
            self.session_var.set(
                f"Not in the 72-character matrix: {' '.join(unknown)}")
            return
        if not self.reader.connected:
            self.session_var.set(
                "No EEG stream — start the Unicorn LSL Interface first.")
            return

        subject = self.subject_var.get().strip() or "session"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Demo recordings are prefixed so a synthetic CSV can never be
        # mistaken for a real session on disk -- the file is otherwise
        # byte-for-byte indistinguishable from one.
        prefix = "demo_" if self.demo else ""
        out_path = os.path.join(THIS_DIR, "recordings", f"{prefix}{subject}_{stamp}.csv")

        self._set_running(True)
        self.session_var.set(f"Starting — {len(targets)} trials, recording to {out_path}")

        def launch():
            self._recorder_proc = subprocess.Popen(
                [sys.executable, os.path.join(THIS_DIR, "record_session.py"),
                 "--out", out_path],
                cwd=THIS_DIR)
            # Let the recorder resolve the EEG outlet and enter its marker
            # poll loop before the stimulus opens 'P300Markers' -- same
            # ordering README_ACQUISITION.md specifies for the manual flow.
            time.sleep(3.0)
            cmd = [sys.executable, os.path.join(THIS_DIR, "checkerboard_stimulus.py"),
                   "--targets", targets,
                   "--n-repetitions", str(self.reps_var.get()),
                   "--flash-ms", str(self.flash_var.get()),
                   "--isi-ms", str(self.isi_var.get()),
                   "--screen", str(self.screen_var.get())]
            if self.windowed_var.get():
                cmd.append("--windowed")
            self._stimulus_proc = subprocess.Popen(cmd, cwd=THIS_DIR)
            self.session_var.set(f"Running — {len(targets)} trials → {out_path}")

        threading.Thread(target=launch, daemon=True).start()

    def _on_stop(self):
        for proc in (self._stimulus_proc, self._recorder_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
        self._stimulus_proc = None
        self._recorder_proc = None
        self._set_running(False)
        self.session_var.set("Stopped.")

    def _set_running(self, running: bool):
        self.start_button.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)

    def _on_close(self):
        self.reader.stop()
        self._on_stop()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Thai P300 speller session control panel.")
    parser.add_argument("--demo", action="store_true",
                         help="Demo mode: label the window and prefix recordings "
                              "with 'demo_'. Pair with collection/mock_headset.py "
                              "to run the whole flow with no headset.")
    args = parser.parse_args()
    SessionGUI(demo=args.demo).mainloop()


if __name__ == "__main__":
    main()
