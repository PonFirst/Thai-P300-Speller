"""
The BCI2000 classifier arithmetic (bci2000/CheckerboardSignalProcessing/RiemannP300Math.h) against
the fitted sklearn/pyriemann pipeline it replaces online, and against the numpy specification in
pipeline/bci2000_export.py.

A model is fitted on synthetic epochs with the real build_pipeline() and EpochSpec, exported with
the real exporter, and the same raw padded windows are scored three ways:
  pipeline.decision_function(preprocess_segment(window))   what fit_from_recording trains and scores
  bci2000_export.reference_score(model, window)            the step-by-step specification
  RiemannP300Math Score() via a compiled driver            what runs inside BCI2000
All three must agree to ~1e-9 relative. Compiles the driver with BCI2000's bundled Eigen; skips
without a compiler or without a BCI2000 source tree.
"""

import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from pipeline import bci2000_export as bx
from pipeline.classifier_io import load_build_pipeline, save_checkpoint
from pipeline.epoching import EpochSpec, preprocess_segment
from pipeline.manifest import CALIBRATION_FORMAT, REPO_ROOT, SCORE_METHOD, sha256_file

BCI2000_ROOT = Path(os.environ.get("BCI2000_ROOT", Path.home() / "BCI2000" / "trunk"))
EIGEN = BCI2000_ROOT / "src" / "extlib" / "eigen" / "eigen-3.3.7"
SRC = REPO_ROOT / "bci2000" / "CheckerboardSignalProcessing"


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None or not (EIGEN / "Eigen").is_dir():
        pytest.skip("needs a C++ compiler and BCI2000's bundled Eigen")
    exe = tmp_path_factory.mktemp("rcli") / "riemann_cli"
    subprocess.run([cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-isystem", str(EIGEN),
                    str(SRC / "tests" / "riemann_cli.cpp"), "-o", str(exe)], check=True, capture_output=True)
    return exe


def fit_model(tmp_path, spec: EpochSpec, seed: int, n: int = 180, amplitude: float = 6.0):
    """A real pipeline fitted on synthetic padded windows with a target bump; returns (pipe, windows, model)."""
    rng = np.random.default_rng(seed)
    windows = rng.normal(scale=12.0, size=(n, spec.n_padded, len(spec.channels)))
    windows += rng.normal(scale=40.0, size=(n, 1, len(spec.channels)))          # per-epoch DC offsets
    y = (np.arange(n) % 6 == 0).astype(int)
    t0 = spec.n_pad + int(0.3 * spec.sfreq)
    windows[y == 1, t0:t0 + int(0.15 * spec.sfreq), :] += amplitude
    epochs = np.stack([preprocess_segment(w, spec) for w in windows])
    pipe = load_build_pipeline()()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(epochs, y)
    ckpt = tmp_path / f"m{seed}.joblib"
    save_checkpoint(ckpt, pipe, spec, {"synthetic": True})
    cal = tmp_path / f"p{seed}.json"
    cal.write_text(json.dumps({
        "format": CALIBRATION_FORMAT, "a": 0.25, "b": -1.5, "score_method": SCORE_METHOD,
        "classifier_sha256": sha256_file(ckpt), "ece": 0.0, "n_scores": n, "n_targets": int(y.sum()),
        "created_utc": "2026-10-02T00:00:00+00:00", "source_recording_sha256": "0" * 64}))
    return pipe, windows, epochs, bx.extract_model(ckpt, cal)


def cli_input(m: bx.ExportedModel, windows) -> str:
    vals = [m.xdawn_filters.shape[1], m.n_times, m.n_pad, m.padlen, m.sos.shape[0], m.xdawn_filters.shape[0],
            m.prototypes.shape[0], int(m.baseline is not None), *(m.baseline or (0, 0))]
    parts = [" ".join(str(v) for v in vals)]
    for a in (m.sos, m.zi, m.xdawn_filters, m.prototypes, m.ref_invsqrt, m.lda_weights):
        parts.append(" ".join(repr(float(v)) for v in np.asarray(a).ravel()))
    parts.append(f"{m.lda_intercept!r} {int(m.gate_policy == 'skip')} {m.gate_flat_uv!r} {m.gate_railed_uv!r} "
                 f"{m.gate_segment}")
    parts.append(str(len(windows)))
    for w in windows:
        parts.append(" ".join("nan" if not np.isfinite(v) else repr(float(v)) for v in np.asarray(w).ravel()))
    return "\n".join(parts) + "\n"


def run_cli_both(exe, m, windows):
    """[(Score() or None, Decide() score or reason)] per window."""
    out = subprocess.run([str(exe)], input=cli_input(m, windows), capture_output=True, text=True, check=True)
    rows = []
    for line in out.stdout.splitlines():
        a, b = line.split()
        try:
            decided = float(b)
        except ValueError:
            decided = b
        rows.append((None if a == "reject" else float(a), decided))
    return rows


def run_cli(exe, m, windows):
    return [score for score, _ in run_cli_both(exe, m, windows)]


@pytest.mark.parametrize("seed", range(4))
def test_cpp_equals_the_fitted_pipeline_and_the_numpy_reference(cli, tmp_path, seed):
    spec = EpochSpec()
    pipe, windows, epochs, m = fit_model(tmp_path, spec, seed)
    sk = pipe.decision_function(epochs)
    ref = np.array([bx.reference_score(m, w) for w in windows])
    cpp = np.array(run_cli(cli, m, windows), dtype=float)
    scale = max(1.0, np.abs(sk).max())
    assert np.abs(ref - sk).max() <= 1e-9 * scale
    assert np.abs(cpp - sk).max() <= 1e-9 * scale
    assert np.abs(cpp - ref).max() <= 1e-9 * scale
    assert sk[::6].mean() > sk.mean()       # the model learned something, so the scores are not degenerate


def test_with_a_baseline_window(cli, tmp_path):
    spec = EpochSpec(baseline=(0.0, 0.1))
    pipe, windows, epochs, m = fit_model(tmp_path, spec, 11)
    assert m.baseline == (0, 25)
    cpp = np.array(run_cli(cli, m, windows), dtype=float)
    np.testing.assert_allclose(cpp, pipe.decision_function(epochs), rtol=1e-9, atol=1e-9)


def test_without_a_band_pass(cli, tmp_path):
    spec = EpochSpec(band=None, pad_s=0.0)
    pipe, windows, epochs, m = fit_model(tmp_path, spec, 12)
    assert m.sos.shape == (0, 6) and m.padlen == 0
    cpp = np.array(run_cli(cli, m, windows), dtype=float)
    np.testing.assert_allclose(cpp, pipe.decision_function(epochs), rtol=1e-9, atol=1e-9)


def test_unusual_but_valid_windows_still_agree(cli, tmp_path):
    spec = EpochSpec()
    pipe, windows, _, m = fit_model(tmp_path, spec, 13)
    rng = np.random.default_rng(5)
    odd = np.stack([
        windows[0] * 50.0,                                                    # large amplitude
        windows[1] + 1e4,                                                     # big DC offset
        windows[2] + np.linspace(-300, 300, spec.n_padded)[:, None],          # strong drift
        rng.standard_t(2, size=windows[3].shape) * 20,                        # heavy-tailed noise
    ])
    sk = pipe.decision_function(np.stack([preprocess_segment(w, spec) for w in odd]))
    cpp = np.array(run_cli(cli, m, odd), dtype=float)
    np.testing.assert_allclose(cpp, sk, rtol=1e-9, atol=1e-9 * max(1.0, np.abs(sk).max()))


def test_non_finite_and_degenerate_windows_are_rejected(cli, tmp_path):
    spec = EpochSpec()
    _, windows, _, m = fit_model(tmp_path, spec, 14)
    bad = windows[:3].copy()
    bad[0, 10, 2] = np.nan
    bad[1, 300, 0] = np.inf
    bad[2] = 0.0                                                              # flat: covariance not positive definite
    assert run_cli(cli, m, bad) == [None, None, None]


def test_exporter_refuses_what_the_filter_cannot_reproduce(tmp_path):
    spec = EpochSpec()
    pipe, windows, epochs, m = fit_model(tmp_path, spec, 21)
    ckpt, cal = tmp_path / "m21.joblib", tmp_path / "p21.json"
    # a calibration fitted for another checkpoint
    other = json.loads(cal.read_text())
    other["classifier_sha256"] = "f" * 64
    wrong_cal = tmp_path / "wrong.json"
    wrong_cal.write_text(json.dumps(other))
    with pytest.raises(bx.ExportError, match="another checkpoint"):
        bx.extract_model(ckpt, wrong_cal)
    # estimator options that change the covariance
    pipe.named_steps["cov"].kwds = {"assume_centered": True}
    save_checkpoint(tmp_path / "k.joblib", pipe, spec, {})
    other["classifier_sha256"] = sha256_file(tmp_path / "k.joblib")
    (tmp_path / "k.json").write_text(json.dumps(other))
    with pytest.raises(bx.ExportError, match="estimator options"):
        bx.extract_model(tmp_path / "k.joblib", tmp_path / "k.json")
    # an epoch that does not start at the onset
    spec_t = EpochSpec(tmin=-0.2, tmax=0.6)
    pipe2 = load_build_pipeline()()
    rng = np.random.default_rng(3)
    y = (np.arange(120) % 6 == 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe2.fit(rng.normal(size=(120, 8, spec_t.n_times)) + y[:, None, None], y)
    save_checkpoint(tmp_path / "t.joblib", pipe2, spec_t, {})
    other["classifier_sha256"] = sha256_file(tmp_path / "t.joblib")
    (tmp_path / "t.json").write_text(json.dumps(other))
    with pytest.raises(bx.ExportError, match="tmin"):
        bx.extract_model(tmp_path / "t.joblib", tmp_path / "t.json")


def test_reference_decision_gates_then_checks_rank_then_scores(tmp_path):
    spec = EpochSpec()
    pipe, windows, epochs, m = fit_model(tmp_path, spec, 23)
    score, reason = bx.reference_decision(m, windows[0])
    assert reason is None and score == pytest.approx(pipe.decision_function(epochs[:1])[0], rel=1e-9)
    railed = windows[1].copy()
    railed[:, 4] += np.where(np.arange(spec.n_padded) % 20 < 10, 600.0, -600.0)   # > 400 uV peak-to-peak
    assert bx.reference_decision(m, railed) == (None, "channel_quality")
    bridged = windows[2].copy()
    bridged[:, 3] = bridged[:, 2]                                                    # green, but dependent
    assert bx.reference_decision(m, bridged) == (None, "rank_deficient")
    nan = windows[3].copy()
    nan[5, 1] = np.nan
    assert bx.reference_decision(m, nan) == (None, "non_finite")


@pytest.mark.parametrize("seed", [31, 32])
def test_cpp_decisions_equal_reference_decision(cli, tmp_path, seed):
    # Normal windows and every rejection path, each decided by the C++ and by the Python specification
    # (whose quality gate is pipeline.quality.gate_epoch itself).
    spec = EpochSpec()
    _, windows, _, m = fit_model(tmp_path, spec, seed)
    rng = np.random.default_rng(seed)
    cases = [w.copy() for w in windows[:8]]
    railed = windows[8].copy()
    railed[:, 4] += np.where(np.arange(spec.n_padded) % 20 < 10, 600.0, -600.0)
    flat = windows[9].copy()
    flat[:, 2] = 3.0                                            # flat: < 1 uV peak-to-peak after the band-pass
    flat_segment = windows[10].copy()                          # alive, then flat for the epoch's second half:
    t = np.arange(spec.n_padded)                                # only the flat-segment rule can catch it
    flat_segment[:, 6] = np.where(t < spec.n_pad + 100, 3.0 * np.sin(2 * np.pi * 10 * t / spec.sfreq), 0.0)
    bridged = windows[11].copy()
    bridged[:, 3] = bridged[:, 2]
    near_bridged = windows[12].copy()
    near_bridged[:, 3] = near_bridged[:, 2] + rng.normal(0, 1e-3, spec.n_padded)   # nearly dependent: scored
    nan = windows[13].copy()
    nan[17, 5] = np.nan
    noisy = windows[14].copy()
    noisy[:, 0] += 120 * np.sin(np.arange(spec.n_padded) / 3.0)   # yellow, never rejected
    cases += [railed, flat, flat_segment, bridged, near_bridged, nan, noisy]
    cpp = run_cli_both(cli, m, cases)
    for i, (case, (_, decided)) in enumerate(zip(cases, cpp)):
        score, reason = bx.reference_decision(m, case)
        if reason is None:
            assert isinstance(decided, float), (i, decided)
            # A nearly singular covariance (case 12: two channels equal to within 1e-3 uV) makes the
            # tangent-space log ill-conditioned, so two correct eigensolvers agree to ~1e-5 there instead
            # of ~1e-12: conditioning, not a porting error, and far below anything that moves a decision.
            rel = 1e-4 if case is near_bridged else 1e-9
            assert decided == pytest.approx(score, rel=rel, abs=1e-9), i
        else:
            assert decided == reason, (i, decided, reason)
    reasons = [d for _, d in cpp]
    assert reasons[8:12] == ["channel_quality", "channel_quality", "channel_quality", "rank_deficient"]
    x = bx.preprocessed_epoch(m, flat_segment)[6]
    assert 1.0 <= x.max() - x.min() <= 400.0                    # the whole channel is green ...
    assert np.ptp(x[-25:]) < 1.0                                # ... but its last segment is flat
    assert reasons[13] == "non_finite" and isinstance(reasons[14], float)
