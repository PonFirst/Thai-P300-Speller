"""
Epoch cutting and preprocessing, shared by the offline fit script and the
live inference service.

One implementation for both paths is the point. A classifier fitted on epochs
cut one way and served epochs cut another -- a different window, filter, or
unit -- does not raise an error; it just scores worse, silently.

Each epoch is taken from a padded window around the flash, band-pass filtered
with zero phase on that window alone, then cropped. Because filtering never
looks outside the padded window, an epoch is identical whether it comes from
a whole recording or from the few seconds a live buffer holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

from pipeline.constants import EEG_CHANNELS, EEG_SFREQ

NOT_YET_AVAILABLE = "not_yet_available"
BEFORE_BUFFER = "before_buffer"
GAP_IN_WINDOW = "gap_in_window"
NON_MONOTONIC = "non_monotonic"
NON_FINITE = "non_finite"
MAX_SAMPLE_GAP_FACTOR = 1.5


@lru_cache(maxsize=16)
def _bandpass_sos(order: int, low: float, high: float, sfreq: float) -> np.ndarray:
    return butter(order, (low, high), btype="bandpass", fs=sfreq, output="sos")


def _sosfiltfilt_padlen(sos: np.ndarray) -> int:
    ntaps = 2 * len(sos) + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * ntaps


@dataclass(frozen=True)
class EpochSpec:
    """
    Stored inside every classifier checkpoint; the service refuses to serve a
    checkpoint whose spec doesn't match the live stream.

    band defaults to 1-24 Hz, the band of MOABB's P300 paradigm that the
    classifier pipelines were validated with on BNCI2014-008/009.
    """

    tmin: float = 0.0
    tmax: float = 0.8
    sfreq: float = EEG_SFREQ
    channels: tuple = tuple(EEG_CHANNELS)
    band: tuple | None = (1.0, 24.0)
    filter_order: int = 4
    pad_s: float = 0.5
    baseline: tuple | None = None
    units: str = "uV"

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        if self.band is not None:
            object.__setattr__(self, "band", (float(self.band[0]), float(self.band[1])))
        if self.baseline is not None:
            object.__setattr__(self, "baseline", (float(self.baseline[0]), float(self.baseline[1])))

        if self.units != "uV":
            raise ValueError(
                f"units must be 'uV', got {self.units!r}: the LSL stream, recorder CSV, and "
                "this module all work in microvolts (CLAUDE.md gotcha 1)")
        if not self.sfreq > 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")
        if not self.tmax > self.tmin:
            raise ValueError(f"tmax ({self.tmax}) must be greater than tmin ({self.tmin})")
        if self.n_times < 2:
            raise ValueError(f"epoch window yields {self.n_times} samples; need at least 2")
        if self.pad_s < 0:
            raise ValueError(f"pad_s must be >= 0, got {self.pad_s}")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channels must be non-empty and unique, got {self.channels}")
        if self.filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {self.filter_order}")
        if self.band is not None:
            low, high = self.band
            if not 0 < low < high < self.sfreq / 2:
                raise ValueError(f"band must satisfy 0 < low < high < sfreq/2, got {self.band}")
            padlen = _sosfiltfilt_padlen(_bandpass_sos(self.filter_order, low, high, self.sfreq))
            if self.n_padded <= padlen:
                raise ValueError(
                    f"padded window ({self.n_padded} samples) is too short for the band-pass "
                    f"filter (needs > {padlen}); increase pad_s or the epoch window")
        if self.baseline is not None:
            b0, b1 = self.baseline
            if not self.tmin <= b0 < b1 <= self.tmax:
                raise ValueError(f"baseline {self.baseline} must lie within [tmin, tmax]")
            i0, i1 = self._baseline_indices()
            if i1 <= i0:
                raise ValueError(f"baseline {self.baseline} covers no samples at {self.sfreq} Hz")

    @property
    def n_times(self) -> int:
        return int(round((self.tmax - self.tmin) * self.sfreq))

    @property
    def n_pad(self) -> int:
        return int(round(self.pad_s * self.sfreq))

    @property
    def n_padded(self) -> int:
        return self.n_times + 2 * self.n_pad

    def _baseline_indices(self) -> tuple[int, int]:
        b0, b1 = self.baseline
        return (int(round((b0 - self.tmin) * self.sfreq)),
                int(round((b1 - self.tmin) * self.sfreq)))

    def to_dict(self) -> dict:
        return {
            "tmin": self.tmin, "tmax": self.tmax, "sfreq": self.sfreq,
            "channels": list(self.channels),
            "band": list(self.band) if self.band is not None else None,
            "filter_order": self.filter_order, "pad_s": self.pad_s,
            "baseline": list(self.baseline) if self.baseline is not None else None,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpochSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown EpochSpec field(s) {sorted(unknown)}")
        return cls(**data)


def cut_padded_segment(timestamps, data, onset: float,
                       spec: EpochSpec) -> tuple[np.ndarray | None, str | None]:
    """
    Select the raw padded window for a flash at `onset` (LSL clock seconds).

    Returns (segment shaped (n_padded, n_channels), None) or (None, reason).
    Samples are chosen by index from the first sample at or after onset+tmin,
    so the epoch grid follows real sample times with no interpolation.
    """
    ts = np.asarray(timestamps, dtype=float)
    x = np.asarray(data)
    if x.ndim != 2 or x.shape[0] != ts.shape[0]:
        raise ValueError(f"data must be (n_samples, n_channels) aligned with timestamps; "
                         f"got data {x.shape}, timestamps {ts.shape}")
    if x.shape[1] != len(spec.channels):
        raise ValueError(f"data has {x.shape[1]} channels; EpochSpec expects "
                         f"{len(spec.channels)} {spec.channels}")
    if ts.size == 0:
        return None, NOT_YET_AVAILABLE

    period = 1.0 / spec.sfreq
    epoch_start = onset + spec.tmin
    # By time, not index: a short live buffer must not turn a future flash into before_buffer.
    if epoch_start - spec.n_pad * period < ts[0] - MAX_SAMPLE_GAP_FACTOR * period:
        return None, BEFORE_BUFFER

    i0 = int(np.searchsorted(ts, epoch_start, side="left"))
    start = i0 - spec.n_pad
    stop = i0 + spec.n_times + spec.n_pad
    if stop > ts.size:
        return None, NOT_YET_AVAILABLE
    if start < 0:
        return None, GAP_IN_WINDOW

    steps = np.diff(ts[start:stop])
    if np.any(steps <= 0):
        return None, NON_MONOTONIC
    if np.any(steps > MAX_SAMPLE_GAP_FACTOR * period):
        return None, GAP_IN_WINDOW
    if ts[i0] - epoch_start > MAX_SAMPLE_GAP_FACTOR * period:
        return None, GAP_IN_WINDOW

    segment = x[start:stop]
    if not np.all(np.isfinite(segment)):
        return None, NON_FINITE
    return segment, None


def preprocess_segment(segment, spec: EpochSpec) -> np.ndarray:
    """Band-pass (zero phase) the padded window, crop, baseline-correct. Returns (n_channels, n_times)."""
    seg = np.asarray(segment, dtype=np.float64)
    expected = (spec.n_padded, len(spec.channels))
    if seg.shape != expected:
        raise ValueError(f"segment must have shape {expected}, got {seg.shape}")
    x = seg.T.copy()
    if spec.band is not None:
        sos = _bandpass_sos(spec.filter_order, spec.band[0], spec.band[1], spec.sfreq)
        x = sosfiltfilt(sos, x, axis=-1)
    x = x[:, spec.n_pad:spec.n_pad + spec.n_times]
    if spec.baseline is not None:
        i0, i1 = spec._baseline_indices()
        x = x - x[:, i0:i1].mean(axis=1, keepdims=True)
    return np.ascontiguousarray(x)


def extract_epoch(timestamps, data, onset: float,
                  spec: EpochSpec) -> tuple[np.ndarray | None, str | None]:
    segment, reason = cut_padded_segment(timestamps, data, onset, spec)
    if segment is None:
        return None, reason
    return preprocess_segment(segment, spec), None


def extract_epochs(timestamps, data, onsets,
                   spec: EpochSpec) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Batch form for the offline fit script.

    Returns (X shaped (n_kept, n_channels, n_times), kept mask over onsets,
    per-onset reason or None).
    """
    epochs, kept, reasons = [], [], []
    for onset in onsets:
        epoch, reason = extract_epoch(timestamps, data, float(onset), spec)
        kept.append(epoch is not None)
        reasons.append(reason)
        if epoch is not None:
            epochs.append(epoch)
    shape = (0, len(spec.channels), spec.n_times)
    X = np.stack(epochs) if epochs else np.empty(shape)
    return X, np.asarray(kept, dtype=bool), reasons
