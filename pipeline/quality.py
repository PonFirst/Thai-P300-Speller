"""
Signal-quality levels, shared by the GUI scalp map and the inference
service's quality gate, so the dot the operator sees and the decision the
service makes can never disagree about which channel is bad.

Thresholds (peak-to-peak, microvolts), unchanged from the Tkinter panel:

    ptp < 1            red      flat / disconnected
    1 <= ptp <= 200    green
    200 < ptp <= 400   yellow   noisy
    ptp > 400          red      railed / clipped
    non-finite         unknown  gray on the map, rejected by the gate

A channel is also red when any 100 ms segment of the window, counted back from
the newest sample, is flat (< 1 uV peak-to-peak). Peak-to-peak over the whole
window cannot see a contact that fails mid-window: the older healthy samples
keep it green until they scroll out, about a second later. With the segment
check a flat onset shows after 100 ms of flatness, and railed data shows as
soon as it arrives.

Behavior change vs. the Tkinter panel: its classify_quality() returned green
for a NaN peak-to-peak, because every comparison with NaN is False and the
value fell through to the last branch. Non-finite input is unknown here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.constants import EEG_SFREQ, RAILED_PTP_UV, SANITY_PTP_MAX_UV, SANITY_PTP_MIN_UV

GREEN = "green"
YELLOW = "yellow"
RED = "red"
UNKNOWN = "unknown"

MIN_SAMPLES_FOR_QUALITY = 5
FLAT_SEGMENT_S = 0.1

GATE_SKIP = "skip"
GATE_WARN = "warn"
GATE_POLICIES = (GATE_SKIP, GATE_WARN)
REJECT_REASON_CHANNEL_QUALITY = "channel_quality"


def classify_ptp(ptp_uv: float) -> str:
    """Map one channel's peak-to-peak amplitude (uV) to a quality level."""
    if ptp_uv is None or not np.isfinite(ptp_uv):
        return UNKNOWN
    if ptp_uv < SANITY_PTP_MIN_UV:
        return RED
    if ptp_uv > RAILED_PTP_UV:
        return RED
    if ptp_uv > SANITY_PTP_MAX_UV:
        return YELLOW
    return GREEN


def _has_flat_segment(column: np.ndarray, segment: int) -> bool:
    """Any full segment, counted back from the newest sample, flatter than the flat threshold."""
    if column.size <= segment:
        return float(column.max() - column.min()) < SANITY_PTP_MIN_UV
    n_full = column.size // segment
    segments = column[column.size - n_full * segment:].reshape(n_full, segment)
    return bool(np.any(segments.max(axis=1) - segments.min(axis=1) < SANITY_PTP_MIN_UV))


def channel_levels(window_uv: np.ndarray, sfreq: float = EEG_SFREQ) -> list[str]:
    """
    Quality level per channel for a window shaped (n_samples, n_channels).

    A channel is unknown if the window is too short to judge or contains any
    non-finite sample (a stream gap), rather than being scored on partial data.
    """
    arr = np.asarray(window_uv, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"window must be 2-D (n_samples, n_channels), got shape {arr.shape}")
    segment = max(MIN_SAMPLES_FOR_QUALITY, int(round(FLAT_SEGMENT_S * sfreq)))
    levels = []
    for column in arr.T:
        if column.size < MIN_SAMPLES_FOR_QUALITY or not np.all(np.isfinite(column)):
            levels.append(UNKNOWN)
            continue
        level = classify_ptp(float(column.max() - column.min()))
        if level != RED and _has_flat_segment(column, segment):
            level = RED
        levels.append(level)
    return levels


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    levels: dict
    bad_channels: tuple
    noisy_channels: tuple
    reason: str | None


def gate_epoch(epoch_uv: np.ndarray, channels: list[str], policy: str = GATE_SKIP,
               sfreq: float = EEG_SFREQ) -> GateDecision:
    """
    Decide whether one flash epoch, shaped (n_channels, n_times), may be fed
    to the classifier.

    GATE_SKIP (default) rejects the flash if ANY channel is red or unknown over
    the epoch window. xDAWN estimates spatial filters and covariances across
    all channels jointly, so one railed electrode contaminates every
    component, not just its own, and would push a large, confident, wrong
    score into the posterior. Down-weighting instead would need a calibrated
    model of score reliability under contamination, which doesn't exist. A
    rejected flash is never shown to the controller, so it counts toward
    neither t_min nor t_max.

    GATE_WARN accepts every flash but still reports bad channels, for
    comparing the two policies offline.

    Yellow never causes rejection under either policy: blinks alone routinely
    push frontal channels past 200 uV.
    """
    if policy not in GATE_POLICIES:
        raise ValueError(f"unknown quality-gate policy {policy!r}; expected one of {GATE_POLICIES}")
    epoch = np.asarray(epoch_uv, dtype=float)
    if epoch.ndim != 2 or epoch.shape[0] != len(channels):
        raise ValueError(
            f"epoch must be (n_channels={len(channels)}, n_times), got shape {epoch.shape}")

    levels = dict(zip(channels, channel_levels(epoch.T, sfreq=sfreq)))
    bad = tuple(ch for ch in channels if levels[ch] in (RED, UNKNOWN))
    noisy = tuple(ch for ch in channels if levels[ch] == YELLOW)
    reason = REJECT_REASON_CHANNEL_QUALITY if bad else None
    accept = not bad if policy == GATE_SKIP else True
    return GateDecision(accept=accept, levels=levels, bad_channels=bad,
                        noisy_channels=noisy, reason=reason)
