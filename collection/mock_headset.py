"""
Synthetic Unicorn Hybrid Black -- publishes a fake EEG LSL outlet so the
whole session flow (session_gui.py -> record_session.py ->
checkerboard_stimulus.py) can be run and demonstrated with no headset,
no Windows VM, and no Unicorn LSL Interface.

Nothing downstream needs to change to use this: every consumer in this
repo resolves streams by TYPE ('EEG' / 'Markers'), never by stream name
(record_session.resolve_eeg_inlet, session_gui.EEGStreamReader.run), so
this outlet is picked up automatically.

WHAT IT GENERATES
  - Background: one-pole-filtered white noise per channel ("pink-ish",
    1/f-leaning), scaled into the plausible-EEG band that
    record_session.py's sanity check and session_gui.py's scalp map both
    expect (1-200 uV peak-to-peak).
  - P300 responses: subscribes to the 'P300Markers' stream published by
    checkerboard_stimulus.py and superimposes a synthetic P300 deflection
    after every flash whose marker says is_target=true, weighted across
    channels by a centro-parietal topography. This is what makes a demo
    recording usable for developing epoching/ERP/calibration code --
    the CSV contains real structure, not just noise.

UNITS: microvolts. record_session.run_signal_sanity_check() and
session_gui.classify_quality() both treat raw sample values as uV
(SANITY_PTP_MIN_UV=1.0, SANITY_PTP_MAX_UV=200.0). Emitting volts here
would make every channel read as flat/red. Note config.py's
PTP_REJECT_THRESHOLD_V is VOLTS and scoped to the MOABB datasets --
deliberately not used here.

THE DATA IS FAKE. The outlet is named 'MockUnicornEEG' (not
'UnicornEEG') so it is unmistakable in logs, LabRecorder, and stream
listings that no brain produced it.

Usage:
    python mock_headset.py                          # all channels good
    python mock_headset.py --flat P3 --noisy C4     # simulate bad contacts
    python mock_headset.py --no-p300                # background noise only
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, local_clock

from record_session import EEG_CHANNELS, SANITY_PTP_MIN_UV, SANITY_PTP_MAX_UV

STREAM_NAME = "MockUnicornEEG"
SFREQ = 250.0
CHUNK_SAMPLES = 5            # 5 samples @ 250 Hz = one chunk per 20 ms
P300_PEAK_LATENCY_S = 0.35   # inside config.P300_SEARCH_WINDOW (0.25, 0.55)
P300_WIDTH_S = 0.06          # Gaussian sigma
P300_AMPLITUDE_UV = 6.0
P300_DURATION_S = 0.8        # after this, an event contributes ~nothing

# Relative P300 amplitude per channel: maximal centro-parietal (Pz/Cz),
# tapering frontally and occipitally. Keeps ERP/topography output
# anatomically plausible instead of uniform across the montage.
CHANNEL_GAIN = {
    "Fz": 0.45, "F3": 0.35,
    "Cz": 0.90, "C4": 0.70,
    "Pz": 1.00, "P3": 0.80,
    "O1": 0.40, "O2": 0.40,
}

# Converts the unit-variance filtered noise into microvolts. Tuned so a
# good channel reads roughly 40 uV peak-to-peak -- comfortably inside the
# 1-200 uV plausible band, and in the range real scalp EEG actually
# occupies rather than hugging the flat end of it.
BACKGROUND_SCALE_UV = 45.0

# A flat/disconnected electrode is simulated by scaling the background
# down. 0.005, not 0.02: at 0.02 it lands near 1 uV peak-to-peak,
# straddling SANITY_PTP_MIN_UV and intermittently rendering GREEN.
FLAT_SCALE = 0.005

# Noisy and railed contacts are simulated by ADDING a fixed-amplitude
# oscillation, not by scaling the noise up. Scaling cannot hit the yellow
# band reliably: the peak-to-peak of a random process over a 1-2 s window
# itself varies ~40% channel to channel, so a multiplier tuned to land at
# 300 uV drifts across the 400 uV red threshold on some windows and
# renders red when it means yellow. A deterministic additive component
# has a stable peak-to-peak, and better matches what these faults look
# like in practice (line-noise/EMG contamination, or a large artifact).
#   noisy  -> ~280 uV ptp + background  -> inside 200-400  -> yellow
#   railed -> ~700 uV ptp               -> above 400       -> red
NOISY_INTERFERENCE_UV = 140.0
RAILED_INTERFERENCE_UV = 350.0
# 47 Hz, not 50: at 250 Hz sampling, 50 Hz gives exactly 5 samples/cycle,
# so the sampled peak never reaches the true peak and the peak-to-peak
# depends on phase. 47 Hz is not a divisor of the rate, so the window
# covers the full swing.
INTERFERENCE_FREQ_HZ = 47.0


# ----------------------------------------------------------------------
# Pure signal functions -- no LSL, no threads, no clock. Unit-tested in
# tests/test_mock_headset.py (same split as flash_sequence.py vs
# checkerboard_stimulus.py: the logic that would silently corrupt a
# recording is kept independently testable).
# ----------------------------------------------------------------------

def p300_waveform(t_rel: np.ndarray, amplitude: float = P300_AMPLITUDE_UV,
                  peak_latency: float = P300_PEAK_LATENCY_S,
                  width: float = P300_WIDTH_S) -> np.ndarray:
    """
    Synthetic P300: a Gaussian bump peaking `peak_latency` seconds after
    stimulus onset. Zero for t_rel < 0 (a response cannot precede its
    stimulus).

    Args:
        t_rel: seconds relative to stimulus onset, shape (N,).
        amplitude: peak amplitude in microvolts.
        peak_latency: seconds from onset to peak.
        width: Gaussian sigma in seconds.

    Returns:
        Microvolt contribution at each t_rel, shape (N,).
    """
    t_rel = np.asarray(t_rel, dtype=float)
    y = amplitude * np.exp(-0.5 * ((t_rel - peak_latency) / width) ** 2)
    return np.where(t_rel < 0, 0.0, y)


def background_noise_step(state: np.ndarray, rng: np.random.Generator,
                          n_samples: int, alpha: float = 0.92) -> tuple:
    """
    Generate `n_samples` of one-pole-filtered white noise per channel.

    The filter state is carried in and out so consecutive chunks join
    seamlessly -- regenerating independent noise per chunk would put a
    discontinuity at every chunk boundary (an artifact at a fixed 20 ms
    period, which is exactly the kind of thing that would show up as a
    spurious oscillation in a PSD).

    Args:
        state: last filter value per channel, shape (n_channels,).
        rng: random generator.
        n_samples: samples to produce.
        alpha: pole location; higher = more low-frequency-weighted.

    Returns:
        (samples, new_state) where samples has shape
        (n_samples, n_channels), in arbitrary units (scaled by caller).
    """
    n_channels = state.shape[0]
    out = np.empty((n_samples, n_channels), dtype=float)
    current = state.copy()
    for i in range(n_samples):
        current = alpha * current + (1 - alpha) * rng.normal(0, 1, n_channels)
        out[i] = current
    return out, current


def channel_scale(channel: str, flat: set, base_amplitude: float) -> float:
    """Background amplitude multiplier: near-zero for a flat channel."""
    return base_amplitude * FLAT_SCALE if channel in flat else base_amplitude


def interference_uv(t: np.ndarray, channel: str, flat: set, noisy: set,
                    railed: set) -> np.ndarray:
    """
    Additive contamination (microvolts) for a simulated bad contact.

    Precedence is flat > railed > noisy: a disconnected electrode picks up
    nothing, so `flat` suppresses interference rather than adding to it,
    and a channel named in several sets resolves to the worst case (a typo
    cannot silently produce a better-looking channel than intended).
    """
    t = np.asarray(t, dtype=float)
    if channel in flat:
        return np.zeros_like(t)
    if channel in railed:
        amplitude = RAILED_INTERFERENCE_UV
    elif channel in noisy:
        amplitude = NOISY_INTERFERENCE_UV
    else:
        return np.zeros_like(t)
    return amplitude * np.sin(2 * np.pi * INTERFERENCE_FREQ_HZ * t)


# ----------------------------------------------------------------------
# Marker listening (I/O)
# ----------------------------------------------------------------------

class MarkerListener(threading.Thread):
    """
    Watches the 'P300Markers' stream and records the onset time of every
    target flash, so the generator can superimpose a P300 response.

    Re-resolves the stream whenever it goes away: checkerboard_stimulus.py
    creates a fresh outlet on every run and session_gui.py restarts it per
    session, so resolving once at startup would work for exactly one
    session and then silently stop injecting responses.
    """

    def __init__(self, event_sink: list, lock: threading.Lock):
        super().__init__(daemon=True)
        self.event_sink = event_sink
        self.lock = lock
        self._stop_event = threading.Event()
        self.n_targets_seen = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        inlet = None
        while not self._stop_event.is_set():
            if inlet is None:
                streams = resolve_byprop("type", "Markers", timeout=1.0)
                if not streams:
                    continue
                inlet = StreamInlet(streams[0])
                print("[mock_headset] Marker stream found -- P300 injection active.")

            try:
                samples, timestamps = inlet.pull_chunk(timeout=0.2)
            except Exception:
                # Stimulus exited; drop the inlet and go back to resolving.
                print("[mock_headset] Marker stream lost -- waiting for next session.")
                inlet = None
                continue

            if not samples:
                continue

            for sample, ts in zip(samples, timestamps):
                try:
                    event = json.loads(sample[0])
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue
                if event.get("event") == "flash" and event.get("is_target"):
                    with self.lock:
                        self.event_sink.append(ts)
                    self.n_targets_seen += 1


# ----------------------------------------------------------------------
# Generator loop (I/O)
# ----------------------------------------------------------------------

def run_mock_stream(flat: set, noisy: set, railed: set, amplitude: float,
                    inject_p300: bool, sfreq: float, seed: int):
    """
    Publish the synthetic EEG outlet and stream until interrupted.

    Streams continuously and unconditionally: record_session.record()
    skips its whole loop iteration on an empty EEG chunk, dropping any
    markers pulled alongside it, so a stalling producer would silently
    lose marker rows from the recording.
    """
    rng = np.random.default_rng(seed)
    n_channels = len(EEG_CHANNELS)

    info = StreamInfo(STREAM_NAME, "EEG", n_channels, sfreq, "float32", "mock_unicorn_1")
    channels_desc = info.desc().append_child("channels")
    for ch in EEG_CHANNELS:
        channels_desc.append_child("channel").append_child_value("label", ch)
    outlet = StreamOutlet(info)

    scales = np.array([channel_scale(ch, flat, amplitude) for ch in EEG_CHANNELS])
    gains = np.array([CHANNEL_GAIN[ch] for ch in EEG_CHANNELS])

    target_onsets: list = []
    lock = threading.Lock()
    listener = None
    if inject_p300:
        listener = MarkerListener(target_onsets, lock)
        listener.start()

    print(f"[mock_headset] Streaming '{STREAM_NAME}' "
          f"({n_channels}ch @ {sfreq:g} Hz, microvolts). SYNTHETIC DATA.")
    if flat or noisy or railed:
        print(f"[mock_headset] Simulated bad contacts -- "
              f"flat: {sorted(flat) or '-'}, noisy: {sorted(noisy) or '-'}, "
              f"railed: {sorted(railed) or '-'}")
    print("[mock_headset] Ctrl+C to stop.")

    state = np.zeros(n_channels)
    chunk_period = CHUNK_SAMPLES / sfreq
    next_emit = time.perf_counter()

    try:
        while True:
            noise, state = background_noise_step(state, rng, CHUNK_SAMPLES)
            # One-pole filtering shrinks variance a lot; renormalize so the
            # visible peak-to-peak lands mid-band rather than near-flat.
            chunk = noise * BACKGROUND_SCALE_UV * scales

            now = local_clock()
            sample_times = now + np.arange(CHUNK_SAMPLES) / sfreq

            if flat or noisy or railed:
                for ch_idx, ch in enumerate(EEG_CHANNELS):
                    chunk[:, ch_idx] += interference_uv(
                        sample_times, ch, flat, noisy, railed)

            if inject_p300:
                with lock:
                    # Drop events too old to still contribute, so this list
                    # cannot grow without bound over a long session.
                    target_onsets[:] = [
                        t for t in target_onsets
                        if now - t < P300_DURATION_S
                    ]
                    onsets = list(target_onsets)
                for onset in onsets:
                    # Per-event jitter: real P300 latency/amplitude varies
                    # trial to trial, and identical responses would make the
                    # synthetic ERP unrealistically clean.
                    latency = P300_PEAK_LATENCY_S + rng.normal(0, 0.02)
                    amp = P300_AMPLITUDE_UV * (1 + rng.normal(0, 0.15))
                    response = p300_waveform(sample_times - onset, amplitude=amp,
                                              peak_latency=latency)
                    # Additive superposition matters: at 62 ms flash + 62 ms
                    # ISI the SOA is ~124 ms, so several responses overlap
                    # inside one 250-550 ms P300 window.
                    chunk += response[:, None] * gains[None, :]

            outlet.push_chunk(chunk.astype(np.float32).tolist(), timestamp=sample_times[-1])

            next_emit += chunk_period
            sleep_s = next_emit - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Fell behind (system hiccup); resync rather than
                # accumulating an ever-growing backlog of missed chunks.
                next_emit = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[mock_headset] Stopped.")
    finally:
        if listener is not None:
            listener.stop()
            print(f"[mock_headset] Injected P300 responses for "
                  f"{listener.n_targets_seen} target flashes.")


def _parse_channels(raw: list, label: str) -> set:
    """Validate --flat/--noisy/--railed values against the real montage."""
    channels = set()
    for item in raw or []:
        for name in item.split(","):
            name = name.strip()
            if not name:
                continue
            if name not in EEG_CHANNELS:
                raise SystemExit(
                    f"error: --{label} got unknown channel {name!r}. "
                    f"Valid channels: {', '.join(EEG_CHANNELS)}")
            channels.add(name)
    return channels


def main():
    parser = argparse.ArgumentParser(
        description="Publish a synthetic Unicorn-like EEG LSL outlet for "
                    "hardware-free demos of the P300 speller session flow.")
    parser.add_argument("--flat", action="append", default=[],
                         help="Channel(s) to simulate as flat/disconnected "
                              "(renders red). Repeatable or comma-separated.")
    parser.add_argument("--noisy", action="append", default=[],
                         help="Channel(s) to simulate as noisy (renders yellow).")
    parser.add_argument("--railed", action="append", default=[],
                         help="Channel(s) to simulate as railed/clipped (renders red).")
    parser.add_argument("--amplitude", type=float, default=1.0,
                         help="Global amplitude multiplier for the background EEG "
                              "(default 1.0 = roughly 30-60 uV peak-to-peak).")
    parser.add_argument("--no-p300", action="store_true",
                         help="Do not inject P300 responses on target flashes "
                              "(background noise only).")
    parser.add_argument("--sfreq", type=float, default=SFREQ,
                         help="Sampling rate in Hz (default 250, matching the "
                              "Unicorn Hybrid Black).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    run_mock_stream(
        flat=_parse_channels(args.flat, "flat"),
        noisy=_parse_channels(args.noisy, "noisy"),
        railed=_parse_channels(args.railed, "railed"),
        amplitude=args.amplitude,
        inject_p300=not args.no_p300,
        sfreq=args.sfreq,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
