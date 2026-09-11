import numpy as np
import pytest

from pipeline.constants import EEG_CHANNELS, RAILED_PTP_UV, SANITY_PTP_MAX_UV, SANITY_PTP_MIN_UV
from pipeline.quality import (
    GATE_SKIP,
    GATE_WARN,
    GREEN,
    RED,
    REJECT_REASON_CHANNEL_QUALITY,
    UNKNOWN,
    YELLOW,
    channel_levels,
    classify_ptp,
    gate_epoch,
)


def _tk_panel_classify(ptp_uv):
    """The Tkinter panel's classify_quality() logic, verbatim, mapped to levels."""
    if ptp_uv < SANITY_PTP_MIN_UV:
        return RED
    if ptp_uv > SANITY_PTP_MAX_UV * 2:
        return RED
    if ptp_uv > SANITY_PTP_MAX_UV:
        return YELLOW
    return GREEN


def _window(ptps, n_samples=250, seed=0):
    """(n_samples, n_channels) with each channel's peak-to-peak exactly ptps[i]."""
    rng = np.random.default_rng(seed)
    columns = []
    for ptp in ptps:
        raw = rng.normal(size=n_samples)
        unit = (raw - raw.min()) / (raw.max() - raw.min())
        columns.append(unit * ptp)
    return np.column_stack(columns)


@pytest.mark.parametrize("ptp,expected", [
    (0.0, RED), (0.999, RED), (1.0, GREEN), (50.0, GREEN), (200.0, GREEN),
    (200.001, YELLOW), (400.0, YELLOW), (400.001, RED), (1e6, RED),
])
def test_threshold_boundaries(ptp, expected):
    assert classify_ptp(ptp) == expected


def test_matches_tkinter_panel_for_every_finite_value():
    sweep = np.concatenate([np.linspace(0, 1000, 20001),
                            [SANITY_PTP_MIN_UV, SANITY_PTP_MAX_UV, RAILED_PTP_UV]])
    for ptp in sweep:
        assert classify_ptp(float(ptp)) == _tk_panel_classify(float(ptp)), ptp


def test_tkinter_panel_showed_nan_as_green():
    assert _tk_panel_classify(float("nan")) == GREEN


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
def test_non_finite_ptp_is_unknown(bad):
    assert classify_ptp(bad) == UNKNOWN


def test_channel_levels_per_channel():
    levels = channel_levels(_window([50, 0.2, 300, 600, 50, 50, 50, 50]))
    assert levels == [GREEN, RED, YELLOW, RED, GREEN, GREEN, GREEN, GREEN]


def test_nan_in_one_channel_only_marks_that_channel_unknown():
    window = _window([50] * 8)
    window[100, 3] = np.nan
    levels = channel_levels(window)
    assert levels[3] == UNKNOWN
    assert levels[:3] + levels[4:] == [GREEN] * 7


def test_too_few_samples_is_unknown():
    assert channel_levels(_window([50] * 8, n_samples=4)) == [UNKNOWN] * 8


def test_channel_levels_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        channel_levels(np.zeros(10))


def _epoch(ptps, n_times=200):
    return _window(ptps, n_samples=n_times).T


def test_all_green_epoch_accepted():
    decision = gate_epoch(_epoch([50] * 8), EEG_CHANNELS)
    assert decision.accept
    assert decision.reason is None
    assert decision.bad_channels == ()


@pytest.mark.parametrize("ptp", [0.1, 900.0])
def test_single_red_channel_rejects_whole_flash(ptp):
    ptps = [50.0] * 8
    ptps[EEG_CHANNELS.index("Pz")] = ptp
    decision = gate_epoch(_epoch(ptps), EEG_CHANNELS, policy=GATE_SKIP)
    assert not decision.accept
    assert decision.bad_channels == ("Pz",)
    assert decision.reason == REJECT_REASON_CHANNEL_QUALITY


def test_yellow_channel_never_rejects():
    ptps = [50.0] * 8
    ptps[EEG_CHANNELS.index("Fz")] = 300.0
    decision = gate_epoch(_epoch(ptps), EEG_CHANNELS)
    assert decision.accept
    assert decision.noisy_channels == ("Fz",)
    assert decision.reason is None


def test_unknown_quality_rejects():
    epoch = _epoch([50.0] * 8)
    epoch[EEG_CHANNELS.index("O1"), 10] = np.nan
    decision = gate_epoch(epoch, EEG_CHANNELS)
    assert not decision.accept
    assert decision.bad_channels == ("O1",)


def test_warn_policy_accepts_but_reports():
    ptps = [50.0] * 8
    ptps[EEG_CHANNELS.index("Pz")] = 0.1
    decision = gate_epoch(_epoch(ptps), EEG_CHANNELS, policy=GATE_WARN)
    assert decision.accept
    assert decision.bad_channels == ("Pz",)
    assert decision.reason == REJECT_REASON_CHANNEL_QUALITY


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="policy"):
        gate_epoch(_epoch([50.0] * 8), EEG_CHANNELS, policy="downweight")


def test_channel_count_mismatch_raises():
    with pytest.raises(ValueError, match="n_channels"):
        gate_epoch(_epoch([50.0] * 7), EEG_CHANNELS)


def _ar1_noise(n, seed, scale_uv=45.0, alpha=0.92):
    """EEG-like background: one-pole-filtered white noise, a few tens of uV peak-to-peak."""
    rng = np.random.default_rng(seed)
    out, state = np.empty(n), 0.0
    for i in range(n):
        state = alpha * state + (1 - alpha) * rng.normal()
        out[i] = state
    return out * scale_uv


def _noise_window(n=250, seed=0):
    return np.column_stack([_ar1_noise(n, seed=seed * 10 + c) for c in range(8)])


def test_flat_onset_mid_window_is_red_after_100_ms_of_flatness():
    window = _noise_window()
    window[-25:, 5] = window[-26, 5]
    assert classify_ptp(float(np.ptp(window[:, 5]))) == GREEN
    assert channel_levels(window)[5] == RED


def test_flatness_shorter_than_one_segment_is_not_yet_red():
    window = _noise_window(seed=2)
    window[-15:, 5] = window[-16, 5]
    assert channel_levels(window)[5] == GREEN


def test_railed_onset_is_red_as_soon_as_it_arrives():
    window = _noise_window(seed=3)
    window[-5:, 2] = 900.0
    assert channel_levels(window)[2] == RED


def test_realistic_noise_never_reads_flat():
    for seed in range(40):
        assert RED not in channel_levels(_noise_window(seed=seed)), seed


def test_gate_rejects_epoch_whose_channel_goes_flat_near_its_end():
    epoch = _noise_window(n=200, seed=4).T
    cz = EEG_CHANNELS.index("Cz")
    epoch[cz, -38:] = epoch[cz, -39]
    decision = gate_epoch(epoch, EEG_CHANNELS)
    assert not decision.accept
    assert decision.bad_channels == ("Cz",)


def test_flat_segment_length_follows_sample_rate():
    window = _noise_window(n=500, seed=5)
    window[-30:, 0] = window[-31, 0]
    assert channel_levels(window, sfreq=500.0)[0] == GREEN
    assert channel_levels(window, sfreq=250.0)[0] == RED
