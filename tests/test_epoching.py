import json

import numpy as np
import pytest

from pipeline import epoching as ep
from pipeline.constants import EEG_CHANNELS
from pipeline.epoching import EpochSpec

SFREQ = 250.0


def _stream(n=3000, seed=0, t0=1000.0):
    rng = np.random.default_rng(seed)
    ts = t0 + np.arange(n) / SFREQ
    data = rng.normal(0, 10, size=(n, len(EEG_CHANNELS)))
    return ts, data


def test_default_spec_shape():
    spec = EpochSpec()
    assert (spec.n_times, spec.n_pad, spec.n_padded) == (200, 125, 450)
    assert spec.channels == tuple(EEG_CHANNELS)


def test_spec_dict_roundtrip_through_json():
    spec = EpochSpec(tmin=-0.2, baseline=(-0.2, 0.0))
    assert EpochSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


@pytest.mark.parametrize("kwargs,match", [
    ({"units": "V"}, "microvolts"),
    ({"tmin": 0.5, "tmax": 0.5}, "tmax"),
    ({"band": (30.0, 200.0)}, "band"),
    ({"pad_s": 0.0, "tmax": 0.05}, "too short"),
    ({"channels": ("Fz", "Fz")}, "unique"),
    ({"baseline": (-0.5, 0.0)}, "baseline"),
])
def test_invalid_spec_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EpochSpec(**kwargs)


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown"):
        EpochSpec.from_dict({**EpochSpec().to_dict(), "resample": 2})


def test_extracts_epoch_of_expected_shape():
    ts, data = _stream()
    epoch, reason = ep.extract_epoch(ts, data, ts[1000], EpochSpec())
    assert reason is None
    assert epoch.shape == (8, 200)


def test_live_buffer_and_full_recording_give_identical_epochs():
    """Train/serve parity: filtering sees only the padded window, so buffer extent cannot change an epoch."""
    ts, data = _stream()
    spec = EpochSpec()
    onset = ts[1000]
    full, _ = ep.extract_epoch(ts, data, onset, spec)
    start, stop = 1000 - spec.n_pad, 1000 + spec.n_times + spec.n_pad
    minimal, _ = ep.extract_epoch(ts[start:stop], data[start:stop], onset, spec)
    wider, _ = ep.extract_epoch(ts[start - 300:stop + 300], data[start - 300:stop + 300], onset, spec)
    np.testing.assert_array_equal(full, minimal)
    np.testing.assert_array_equal(full, wider)


def test_onset_between_samples_uses_next_sample():
    ts, data = _stream()
    spec = EpochSpec()
    on_sample, _ = ep.extract_epoch(ts, data, ts[1000], spec)
    between, _ = ep.extract_epoch(ts, data, ts[999] + 0.001, spec)
    np.testing.assert_array_equal(on_sample, between)


@pytest.mark.parametrize("index,expected", [(50, ep.BEFORE_BUFFER), (2900, ep.NOT_YET_AVAILABLE)])
def test_window_outside_buffer(index, expected):
    ts, data = _stream()
    epoch, reason = ep.extract_epoch(ts, data, ts[index], EpochSpec())
    assert epoch is None
    assert reason == expected


def test_onset_long_before_buffer():
    ts, data = _stream()
    assert ep.extract_epoch(ts, data, ts[0] - 10.0, EpochSpec())[1] == ep.BEFORE_BUFFER


def test_empty_buffer_is_not_yet_available():
    assert ep.extract_epoch(np.empty(0), np.empty((0, 8)), 5.0, EpochSpec())[1] == ep.NOT_YET_AVAILABLE


def test_future_flash_in_short_live_buffer_is_not_yet_available():
    ts, data = _stream(n=40)
    assert ep.extract_epoch(ts, data, ts[-1] + 1.0, EpochSpec())[1] == ep.NOT_YET_AVAILABLE


def test_window_covered_in_time_but_missing_samples_is_a_gap():
    ts = np.concatenate([1000 + np.arange(10) / SFREQ, 1000.8 + np.arange(500) / SFREQ])
    data = np.random.default_rng(0).normal(size=(ts.size, 8))
    assert ep.extract_epoch(ts, data, 1001.0, EpochSpec())[1] == ep.GAP_IN_WINDOW


def test_gap_inside_window_rejected():
    ts, data = _stream()
    ts[1050:] += 0.2
    assert ep.extract_epoch(ts, data, ts[1000], EpochSpec())[1] == ep.GAP_IN_WINDOW


def test_non_monotonic_timestamps_rejected():
    ts, data = _stream()
    ts[1100] = ts[1099]
    assert ep.extract_epoch(ts, data, ts[1000], EpochSpec())[1] == ep.NON_MONOTONIC


def test_nan_inside_window_rejected():
    ts, data = _stream()
    data[1100, 4] = np.nan
    assert ep.extract_epoch(ts, data, ts[1000], EpochSpec())[1] == ep.NON_FINITE


def test_channel_count_mismatch_raises():
    ts, data = _stream()
    with pytest.raises(ValueError, match="channels"):
        ep.extract_epoch(ts, data[:, :7], ts[1000], EpochSpec())


def test_in_band_signal_kept_and_dc_removed():
    ts, _ = _stream()
    t = ts - ts[0]
    data = np.zeros((ts.size, 8))
    data[:, 0] = 10.0 * np.sin(2 * np.pi * 5.0 * t) + 500.0
    epoch, _ = ep.extract_epoch(ts, data, ts[1000], EpochSpec())
    assert abs(np.ptp(epoch[0]) - 20.0) < 2.0
    assert abs(epoch[0].mean()) < 1.5


def test_baseline_correction_zeroes_baseline_window():
    ts, data = _stream()
    spec = EpochSpec(tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0))
    epoch, _ = ep.extract_epoch(ts, data, ts[1000], spec)
    assert epoch.shape == (8, 250)
    np.testing.assert_allclose(epoch[:, :50].mean(axis=1), 0.0, atol=1e-9)


def test_batch_extraction_reports_per_onset_reasons():
    ts, data = _stream()
    X, kept, reasons = ep.extract_epochs(ts, data, [ts[1000], ts[50], ts[2990]], EpochSpec())
    assert X.shape == (1, 8, 200)
    assert kept.tolist() == [True, False, False]
    assert reasons == [None, ep.BEFORE_BUFFER, ep.NOT_YET_AVAILABLE]
