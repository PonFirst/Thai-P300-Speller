"""
Tests for the synthetic-EEG generator's pure signal functions.

Scope note: these cover the math only -- no LSL outlet, no threads, no
marker stream. The point is that the parts capable of silently producing
an implausible/unusable demo recording (wrong P300 latency, chunk-boundary
noise artifacts, bad channels that don't actually land in the intended
quality bucket) are checkable without hardware or a running session.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

from mock_headset import (  # noqa: E402
    BACKGROUND_SCALE_UV,
    CHANNEL_GAIN,
    background_noise_step,
    channel_scale,
    interference_uv,
    p300_waveform,
)
from record_session import (  # noqa: E402
    EEG_CHANNELS,
    SANITY_PTP_MIN_UV,
    SANITY_PTP_MAX_UV,
)
from config import P300_SEARCH_WINDOW  # noqa: E402


def test_p300_peak_falls_inside_expected_search_window():
    """
    The injected peak must land inside config.P300_SEARCH_WINDOW -- that
    is the window every downstream analysis searches for a P300 in, so a
    peak outside it would make the demo data fail its own pipeline.
    """
    t = np.linspace(-0.2, 0.8, 1001)
    y = p300_waveform(t)
    peak_time = t[np.argmax(y)]
    lo, hi = P300_SEARCH_WINDOW
    assert lo <= peak_time <= hi


def test_p300_is_zero_before_stimulus_onset():
    """A response cannot precede its stimulus."""
    t = np.linspace(-0.5, -0.01, 100)
    assert np.all(p300_waveform(t) == 0.0)


def test_p300_superposition_is_additive():
    """
    Overlapping responses must sum. At 62ms flash + 62ms ISI the SOA is
    ~124ms, so 3-4 target responses can overlap within one P300 window;
    if the generator replaced rather than added, the synthetic ERP would
    understate the real overlap.
    """
    t = np.linspace(0, 1.0, 500)
    a = p300_waveform(t)
    b = p300_waveform(t - 0.124)
    combined = p300_waveform(t) + p300_waveform(t - 0.124)
    assert np.allclose(combined, a + b)
    # And the overlap actually produces a larger peak than either alone.
    assert combined.max() > a.max()


def test_p300_amplitude_scales_linearly():
    t = np.linspace(0, 1.0, 500)
    assert np.allclose(p300_waveform(t, amplitude=12.0),
                       2.0 * p300_waveform(t, amplitude=6.0))


def test_channel_gain_is_centro_parietal():
    """Pz/Cz should dominate; frontal and occipital sites weaker."""
    assert CHANNEL_GAIN["Pz"] == max(CHANNEL_GAIN.values())
    assert CHANNEL_GAIN["Cz"] > CHANNEL_GAIN["Fz"]
    assert CHANNEL_GAIN["Pz"] > CHANNEL_GAIN["Oz"]
    assert set(CHANNEL_GAIN) == set(EEG_CHANNELS)


def test_background_noise_is_continuous_across_chunks():
    """
    Chunk boundaries must not introduce a discontinuity: the filter state
    carries across calls, so a step at every 20ms chunk edge would show
    up as a spurious ~50Hz-family artifact in any PSD of the demo data.
    """
    rng = np.random.default_rng(0)
    state = np.zeros(len(EEG_CHANNELS))
    first, state = background_noise_step(state, rng, 5)
    second, _ = background_noise_step(state, rng, 5)

    within_chunk = np.abs(np.diff(first, axis=0)).max()
    at_boundary = np.abs(second[0] - first[-1]).max()
    assert at_boundary <= within_chunk * 5


def test_background_noise_lands_in_plausible_eeg_band():
    """
    Two seconds of good-channel background must sit inside the 1-200 uV
    band that record_session.run_signal_sanity_check() and
    session_gui.classify_quality() both treat as plausible EEG -- i.e.
    the demo must not trip its own sanity check.
    """
    rng = np.random.default_rng(0)
    n_channels = len(EEG_CHANNELS)
    state = np.zeros(n_channels)
    samples, _ = background_noise_step(state, rng, 500)  # 2 s at 250 Hz
    signal_uv = samples * BACKGROUND_SCALE_UV  # same scaling the generator applies

    ptp = signal_uv.max(axis=0) - signal_uv.min(axis=0)
    assert np.all(ptp > SANITY_PTP_MIN_UV)
    assert np.all(ptp < SANITY_PTP_MAX_UV)


def _simulate_channel(channel, flat, noisy, railed, seed, n=500, sfreq=250.0):
    """
    Reproduce what the generator emits for one channel over `n` samples,
    then return its peak-to-peak in uV -- background scaled by
    channel_scale, plus interference_uv, exactly as run_mock_stream does.
    """
    rng = np.random.default_rng(seed)
    samples, _ = background_noise_step(np.zeros(len(EEG_CHANNELS)), rng, n)
    ch_idx = EEG_CHANNELS.index(channel)
    scale = channel_scale(channel, flat, base_amplitude=1.0)
    t = np.arange(n) / sfreq
    trace = samples[:, ch_idx] * BACKGROUND_SCALE_UV * scale
    trace = trace + interference_uv(t, channel, flat, noisy, railed)
    return float(trace.max() - trace.min())


@pytest.mark.parametrize("bucket", ["flat", "noisy", "railed"])
def test_bad_channel_simulation_lands_in_intended_quality_bucket(bucket):
    """
    Each simulated fault must actually render as the colour it claims,
    for EVERY channel across several seeds, with margin.

    Two real bugs were caught here and must stay caught:
      - a "flat" channel emitting 1.04 uV (above the 1.0 uV threshold),
        i.e. rendering GREEN while claiming to be a dead electrode;
      - a "noisy" channel reaching 402 uV, crossing the 400 uV red
        threshold and rendering RED while claiming to be merely noisy.
    Both passed an earlier, weaker version of this test that checked one
    channel at one seed.
    """
    sets = {"flat": set(), "noisy": set(), "railed": set()}

    for seed in range(5):
        for channel in EEG_CHANNELS:
            sets[bucket] = {channel}
            ptp = _simulate_channel(channel, sets["flat"], sets["noisy"],
                                     sets["railed"], seed)
            msg = f"{bucket} {channel} seed {seed}: {ptp:.2f} uV"
            if bucket == "flat":
                assert ptp < SANITY_PTP_MIN_UV * 0.5, msg
            elif bucket == "railed":
                assert ptp > SANITY_PTP_MAX_UV * 2 * 1.1, msg
            else:
                assert SANITY_PTP_MAX_UV * 1.1 < ptp < SANITY_PTP_MAX_UV * 2 * 0.9, msg


def test_good_channels_stay_green_when_another_channel_is_faulty():
    """Simulating a fault on one electrode must not disturb the others."""
    for seed in range(5):
        ptp = _simulate_channel("Pz", flat={"PO7"}, noisy={"C4"}, railed=set(),
                                 seed=seed)
        assert SANITY_PTP_MIN_UV < ptp < SANITY_PTP_MAX_UV, f"seed {seed}: {ptp}"


def test_flat_suppresses_interference():
    """A disconnected electrode picks up nothing, including line noise."""
    t = np.arange(500) / 250.0
    both = interference_uv(t, "Pz", flat={"Pz"}, noisy={"Pz"}, railed={"Pz"})
    assert np.all(both == 0.0)


def test_railed_beats_noisy_when_channel_listed_in_both():
    t = np.arange(500) / 250.0
    railed_only = interference_uv(t, "Pz", set(), set(), {"Pz"})
    both = interference_uv(t, "Pz", set(), {"Pz"}, {"Pz"})
    assert np.allclose(both, railed_only)


def test_interference_is_zero_for_healthy_channel():
    t = np.arange(500) / 250.0
    assert np.all(interference_uv(t, "Pz", set(), {"C4"}, {"PO7"}) == 0.0)


def test_good_channel_scale_is_unmodified():
    assert channel_scale("Pz", flat=set(), base_amplitude=1.0) == 1.0
