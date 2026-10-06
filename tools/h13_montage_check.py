"""
TESTING.md H13 (is the channel order Fz C3 Cz C4 Pz PO7 Oz PO8 physically right?) and H3 (are the
values microvolts in a plausible range?), from one 60 s recording made wearing the headset: blink
hard for the first 20 s, eyes closed for the next 20 s, clench the jaw for the last 20 s.

    $PY tools/h13_montage_check.py collection/recordings/bci2000/S01001/S01S001R01.dat
    $PY tools/h13_montage_check.py /tmp/h13.csv                  # a recorder CSV works too

bci2000/run_session.sh check records exactly this, with on-screen cues. The recording is split into
thirds by time. Expected: the biggest blink at Fz, the biggest eyes-closed alpha gain at Oz (PO7/PO8
also fine), the biggest jaw-clench gain at C3 or C4. Units: each channel's typical peak-to-peak over
2 s windows of the eyes-closed third, rated with pipeline/quality.py's thresholds (1-200 uV green).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ  # noqa: E402
from pipeline.quality import classify_ptp  # noqa: E402

EXPECTED = {"blink": ("Fz",), "alpha": ("Oz", "PO7", "PO8"), "jaw": ("C3", "C4")}


def load(path) -> tuple[np.ndarray, float, list, int, str | None]:
    """(samples x channels in uV, sampling rate, channel names, samples interpolated over, why the data
    is synthetic or None). A BCI2000 .dat's lost or counter-break samples are filled by linear
    interpolation."""
    path = Path(path)
    if path.suffix.lower() == ".dat":
        from pipeline.bci2000_dat import read_dat
        from pipeline.bci2000_recording import invalid_samples
        from pipeline.fit_from_recording import dat_synthetic_reason
        dat = read_dat(path)
        x = dat.signal_uv.astype(float).copy()
        bad = invalid_samples(dat)
        if bad.any() and (~bad).sum() >= 2:
            good = np.flatnonzero(~bad)
            for c in range(x.shape[1]):
                x[bad, c] = np.interp(np.flatnonzero(bad), good, x[good, c])
        return (x, dat.param_number("SamplingRate", "Hz"), dat.param_list("ChannelNames"), int(bad.sum()),
                dat_synthetic_reason(dat))
    import pandas as pd
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df[EEG_CHANNELS[0]], errors="coerce").notna()]
    synthetic = "demo_ recording" if path.name.startswith("demo_") else None
    return df[list(EEG_CHANNELS)].to_numpy(float), EEG_SFREQ, list(EEG_CHANNELS), 0, synthetic


def _band_range(seg, fs, lo, hi):
    y = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), seg, axis=0)
    return np.percentile(y, 99, axis=0) - np.percentile(y, 1, axis=0)


def _band_power(seg, fs, lo, hi):
    f, p = welch(seg, fs=fs, nperseg=int(fs * 2), axis=0)
    return p[(f >= lo) & (f <= hi)].mean(axis=0)


def montage_report(x: np.ndarray, fs: float, channels: list) -> dict:
    """Per channel: blink size (1-10 Hz, uV), alpha gain eyes-closed vs blink third (8-12 Hz), jaw gain vs
    eyes-closed third (20-100 Hz); and which channel is largest for each, with a pass flag."""
    n = len(x) // 3
    if n < int(fs * 4):
        raise ValueError(f"recording too short: {len(x) / fs:.1f} s; record 60 s (20 s per task)")
    blink, closed, jaw = x[:n], x[n:2 * n], x[2 * n:3 * n]
    nyq_hi = min(100.0, fs / 2 - 1)
    measures = {"blink": _band_range(blink, fs, 1, 10),
                "alpha": _band_power(closed, fs, 8, 12) / _band_power(blink, fs, 8, 12),
                "jaw": _band_power(jaw, fs, 20, nyq_hi) / _band_power(closed, fs, 20, nyq_hi)}
    largest = {k: channels[int(np.argmax(v))] for k, v in measures.items()}
    passed = {k: largest[k] in EXPECTED[k] for k in measures}
    return {"channels": list(channels), "measures": {k: v.tolist() for k, v in measures.items()},
            "largest": largest, "passed": passed}


def units_report(x: np.ndarray, fs: float, channels: list, window_s: float = 2.0) -> dict:
    """Median peak-to-peak over 2 s windows per channel, rated by pipeline/quality.py (H3)."""
    w = int(window_s * fs)
    k = len(x) // w
    if k < 1:
        raise ValueError("recording shorter than one 2 s window")
    ptp = np.median(np.ptp(x[:k * w].reshape(k, w, x.shape[1]), axis=1), axis=0)
    return {"channels": list(channels), "median_ptp_uv": ptp.tolist(),
            "levels": [classify_ptp(float(v)) for v in ptp]}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    x, fs, channels, filled, synthetic = load(argv[0])
    if synthetic:
        print(f"*** SYNTHETIC DATA ({synthetic}): these verdicts say nothing about a headset. ***")
    if channels != list(EEG_CHANNELS):
        print(f"warning: channel names {channels} differ from the montage {list(EEG_CHANNELS)}")
    third = len(x) // 3
    # Units on the eyes-closed third only: blinks and jaw clenches rightly push channels past 200 uV.
    m, u = montage_report(x, fs, channels), units_report(x[third:2 * third], fs, channels)
    print(f"{len(x) / fs:.1f} s at {fs:g} Hz" + (f"; {filled} lost sample(s) interpolated" if filled else ""))
    print(f"\n{'':6}{'blink uV':>10}{'alpha x':>10}{'jaw x':>10}{'p-p uV':>10}  level")
    for i, c in enumerate(channels):
        print(f"{c:6}{m['measures']['blink'][i]:10.1f}{m['measures']['alpha'][i]:10.2f}"
              f"{m['measures']['jaw'][i]:10.2f}{u['median_ptp_uv'][i]:10.1f}  {u['levels'][i]}")
    print("\nH13 montage (expected: blink Fz | alpha Oz, PO7 or PO8 | jaw C3 or C4)")
    for k in ("blink", "alpha", "jaw"):
        print(f"  {k:6} largest at {m['largest'][k]:4}  {'PASS' if m['passed'][k] else 'CHECK'}")
    bad = [c for c, lv in zip(channels, u["levels"]) if lv != "green"]
    print(f"\nH3 units: {'PASS, every channel green (1-200 uV)' if not bad else 'CHECK ' + ', '.join(bad)}")
    return 0 if all(m["passed"].values()) and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
