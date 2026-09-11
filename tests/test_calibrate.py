import os

import numpy as np

from stopping.calibrate import (
    PlattCalibrator,
    expected_calibration_error,
    plot_reliability_diagram,
)


def _make_synthetic_separable(n=500, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    # Positive class scores centered higher than negative class.
    scores = np.where(
        labels == 1,
        rng.normal(loc=2.0, scale=1.0, size=n),
        rng.normal(loc=-2.0, scale=1.0, size=n),
    )
    return scores, labels


def test_platt_fit_recovers_positive_slope_on_separable_data():
    scores, labels = _make_synthetic_separable()
    calibrator = PlattCalibrator(calibration_path="unused.csv")
    a, b = calibrator.fit(scores, labels)
    # Higher score -> higher P(target), so slope should be positive.
    assert a > 0


def test_ece_runs_without_error_on_random_data():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, size=200)
    labels = rng.integers(0, 2, size=200)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_reliability_diagram_saves_file(tmp_path):
    rng = np.random.default_rng(2)
    probs = rng.uniform(0, 1, size=200)
    labels = rng.integers(0, 2, size=200)
    out_path = str(tmp_path / "reliability_diagram.png")
    result_path = plot_reliability_diagram(probs, labels, n_bins=10, out_path=out_path)
    assert os.path.exists(result_path)
