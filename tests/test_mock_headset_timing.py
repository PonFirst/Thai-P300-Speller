import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

from mock_headset import CHUNK_SAMPLES, draw_p300_params, next_chunk, sample_timestamps  # noqa: E402

SFREQ = 250.0
MAX_BACKLOG = 250


def _simulate(poll_times, t0=100.0):
    n_emitted, stamps = 0, []
    for now in poll_times:
        first, n = next_chunk(t0, n_emitted, now, SFREQ, MAX_BACKLOG)
        if n:
            stamps.append(sample_timestamps(t0, first, n, SFREQ))
            n_emitted = first + n
    return np.concatenate(stamps)


def test_jittery_polling_gives_contiguous_strictly_increasing_timestamps():
    rng = np.random.default_rng(0)
    polls = 100.0 + np.cumsum(np.abs(rng.normal(0.01, 0.006, size=2000)))
    steps = np.diff(_simulate(polls))
    assert np.all(steps > 0)
    np.testing.assert_allclose(steps, 1 / SFREQ, rtol=1e-6)


def test_old_stamp_at_now_scheme_overlapped_under_jitter():
    """The bug fixed here: stamping each chunk at wall-clock now overlaps whenever a poll runs early."""
    rng = np.random.default_rng(0)
    polls = 100.0 + np.cumsum(np.abs(rng.normal(0.02, 0.004, size=500)))
    old = np.concatenate([now + np.arange(CHUNK_SAMPLES) / SFREQ for now in polls])
    assert np.any(np.diff(old) <= 0)


def test_emission_keeps_pace_with_the_wall_clock():
    polls = 100.0 + np.arange(1, 501) * 0.01
    assert abs(_simulate(polls)[-1] - polls[-1]) <= 1 / SFREQ + 1e-9


def test_stall_skips_ahead_leaving_one_gap_not_a_stale_burst():
    polls = np.concatenate([100.0 + np.arange(1, 101) * 0.01, [104.0],
                            104.0 + np.arange(1, 50) * 0.01])
    ts = _simulate(polls)
    steps = np.diff(ts)
    assert np.all(steps > 0)
    gaps = steps[steps > 1.5 / SFREQ]
    assert gaps.size == 1 and 2.5 < gaps[0] < 3.5
    assert abs(ts[-1] - polls[-1]) <= 1 / SFREQ + 1e-9


def test_nothing_owed_between_grid_points():
    assert next_chunk(100.0, 1, 100.0 + 0.5 / SFREQ, SFREQ, MAX_BACKLOG) == (1, 0)


def test_p300_parameters_are_plausible_per_event_draws():
    rng = np.random.default_rng(0)
    draws = np.array([draw_p300_params(rng) for _ in range(1000)])
    latency, amplitude = draws[:, 0], draws[:, 1]
    assert 0.30 < latency.mean() < 0.40 and latency.std() < 0.05
    assert np.all(amplitude > 0)
