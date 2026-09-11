"""
Fit-script tests. No test may write into the repo: every call that reaches the
writing stage passes models_dir and manifest_path under tmp_path, the REPO_ROOT
defaults are checked only through the pure output_paths(), and
test_fit_tests_never_write_into_the_repo re-runs this file and verifies the
repo's models/ and root *session_manifest.json files are unchanged.
"""

import hashlib
import json
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pipeline import fit_from_recording as ffr
from pipeline import manifest as mf
from pipeline.classifier_io import load_checkpoint
from pipeline.constants import EEG_CHANNELS
from pipeline.epoching import EpochSpec

sys.path.insert(0, str(mf.REPO_ROOT / "collection"))
from flash_sequence import CheckerboardConfig, build_flash_sequence  # noqa: E402

SFREQ = 250.0
SOA = 0.124
T0 = 5000.0
GAINS = np.array([0.45, 0.35, 0.70, 0.90, 1.00, 0.80, 0.40, 0.40])  # Fz F3 C4 Cz Pz P3 O2 O1


def make_recording(path, *, n_trials=5, n_reps=2, seed=0, p300_uv=8.0, noise_uv=5.0,
                   gap_trial=None, scale=1.0):
    """Recorder-format CSV: noise, P300-like bumps after target flashes, real checkerboard groups."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    markers = []
    cursor = T0 + 1.0
    gap = None
    for trial in range(n_trials):
        target = int(rng.integers(72))
        markers.append((cursor, {"event": "trial_start", "trial_id": trial, "target_char_idx": target,
                                 "target_char": f"c{target}"}))
        groups = build_flash_sequence(CheckerboardConfig(n_chars=72, n_repetitions=n_reps))
        onset0 = cursor + 1.0
        for i, group in enumerate(groups):
            markers.append((onset0 + i * SOA, {"event": "flash", "trial_id": trial, "flash_idx": i,
                                               "group": list(group), "is_target": target in group}))
        if trial == gap_trial:
            gap = (onset0 + 10 * SOA + 0.2, onset0 + 10 * SOA + 0.3)
        cursor = onset0 + len(groups) * SOA
        markers.append((cursor, {"event": "trial_end", "trial_id": trial}))
        cursor += 1.5

    n = int((cursor + 1.0 - T0) * SFREQ)
    ts = T0 + np.arange(n) / SFREQ
    data = rng.normal(0, noise_uv, size=(n, 8))
    bump_t = np.arange(int(0.8 * SFREQ)) / SFREQ
    bump = p300_uv * np.exp(-0.5 * ((bump_t - 0.35) / 0.06) ** 2)
    for m_ts, event in markers:
        if event["event"] == "flash" and event["is_target"]:
            i0 = int(np.searchsorted(ts, m_ts))
            seg = data[i0:i0 + len(bump)]
            seg += bump[:len(seg), None] * GAINS
    data *= scale
    if gap is not None:
        # gap rows: all-NaN channels, bracketed by stream_gap markers (hardened recorder format)
        in_gap = (ts >= gap[0]) & (ts < gap[1])
        data[in_gap] = np.nan
        idx = np.flatnonzero(in_gap)
        markers += [(ts[idx[0]], {"event": "stream_gap_start"}), (ts[idx[-1]], {"event": "stream_gap_end"})]

    columns = {"timestamp": ts}
    for c, ch in enumerate(EEG_CHANNELS):
        columns[ch] = data[:, c]
    text = {col: [""] * n for col in ("marker_json_or_empty", "trial_id", "flash_idx", "group_json",
                                      "is_target", "target_char_idx")}
    for m_ts, event in markers:
        row = int(round((m_ts - T0) * SFREQ))
        text["marker_json_or_empty"][row] = json.dumps(event)
        text["trial_id"][row] = str(event.get("trial_id", ""))
        text["flash_idx"][row] = str(event.get("flash_idx", ""))
        text["group_json"][row] = json.dumps(event["group"]) if "group" in event else ""
        text["is_target"][row] = str(event.get("is_target", ""))
        text["target_char_idx"][row] = str(event.get("target_char_idx", ""))
    pd.DataFrame({**columns, **text}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    return make_recording(tmp_path_factory.mktemp("rec") / "subject_001.csv", gap_trial=2)


@pytest.fixture(scope="module")
def fitted(recording, tmp_path_factory):
    out = tmp_path_factory.mktemp("fit")
    corpus = out / "corpus.txt"
    chars = mf.load_characters_file(ffr.DEFAULT_CHARACTERS_PATH)
    corpus.write_text("".join(chars[:6]) * 40, encoding="utf-8")
    lines = []
    result = ffr.fit_from_recording(recording, "subject_001", models_dir=out / "models",
                                    manifest_path=out / "session_manifest.json", lm_corpus=corpus,
                                    log=lines.append)
    return result, out, lines


def test_end_to_end_writes_loadable_artifacts(fitted, recording):
    result, out, _ = fitted
    manifest = mf.load_manifest(out / "session_manifest.json")
    assert manifest.fingerprint == result.fingerprint
    assert manifest.classifier_path == (out / "models" / "subject_001_classifier.joblib").resolve()
    assert manifest.calibration_path == (out / "models" / "subject_001_platt.json").resolve()
    assert manifest.lm_config["type"] == "ngram"

    calibration = mf.load_calibration_file(manifest.calibration_path)
    assert calibration["classifier_sha256"] == mf.sha256_file(manifest.classifier_path)
    assert calibration["source_recording_sha256"] == mf.sha256_file(recording)
    assert (calibration["a"], calibration["b"]) == (result.a, result.b)
    assert calibration["n_scores"] == result.n_epochs

    ckpt = load_checkpoint(manifest.classifier_path)
    assert ckpt.epoch_spec == EpochSpec()
    assert ckpt.trained_on == {"recording_sha256": mf.sha256_file(recording), "n_epochs": result.n_epochs,
                               "n_targets": result.n_targets, "n_trials": 5}


def test_out_of_fold_auc_clearly_above_chance(fitted):
    result, _, _ = fitted
    assert result.auc > 0.8
    assert result.a > 0
    assert 0 <= result.ece < 0.2


def test_gap_rows_are_not_eeg_and_rejections_are_counted(fitted):
    result, _, lines = fitted
    assert result.n_flashes == 5 * 48
    assert result.rejected.get("gap_in_window", 0) >= 5
    assert result.n_epochs + sum(result.rejected.values()) == result.n_flashes
    assert any("non-EEG rows" in line for line in lines)


def test_out_of_fold_scores_never_score_a_trial_the_model_saw():
    groups = np.repeat(np.arange(6), 10)
    X = groups[:, None, None] * np.ones((60, 2, 3))
    y = np.tile([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 6)

    class Spy:
        def fit(self, X, y):
            self.seen = set(X[:, 0, 0].astype(int))
            return self

        def decision_function(self, X):
            assert not set(X[:, 0, 0].astype(int)) & self.seen
            return X[:, 0, 0]

    np.testing.assert_array_equal(ffr.out_of_fold_scores(X, y, groups, 3, Spy), groups)


def test_default_output_paths_resolve_against_repo_root(tmp_path, monkeypatch):
    # pure path computation: nothing is written
    monkeypatch.chdir(tmp_path)
    demo, ckpt, cal, man = ffr.output_paths(tmp_path / "rec.csv", "s1", "models")
    assert not demo
    assert ckpt == mf.REPO_ROOT / "models" / "s1_classifier.joblib"
    assert cal == mf.REPO_ROOT / "models" / "s1_platt.json"
    assert man == mf.REPO_ROOT / "session_manifest.json"
    _, ckpt, _, man = ffr.output_paths(tmp_path / "demo_rec.csv", "s1", "models", allow_demo=True)
    assert ckpt == mf.REPO_ROOT / "models" / "demo_s1_classifier.joblib"
    assert man == mf.REPO_ROOT / "demo_session_manifest.json"
    _, ckpt, _, man = ffr.output_paths(tmp_path / "rec.csv", "s1", tmp_path / "m", manifest_path="x.json")
    assert ckpt == tmp_path / "m" / "s1_classifier.joblib" and str(man) == "x.json"


def _isolate(tmp_path, monkeypatch):
    """
    Explicit output paths under tmp_path, run from another directory, and REPO_ROOT pointed at an
    empty directory as a second line of defense. Returns (argv output flags, models dir, manifest, cwd, fake root).
    """
    fake_root = tmp_path / "fake_repo_root"
    fake_root.mkdir()
    monkeypatch.setattr(ffr, "REPO_ROOT", fake_root)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    models, manifest = tmp_path / "out" / "models", tmp_path / "out" / "demo_session_manifest.json"
    return ["--models-dir", str(models), "--manifest", str(manifest)], models, manifest, cwd, fake_root


def test_demo_recording_refused_without_flag(tmp_path, monkeypatch, capsys):
    demo = make_recording(tmp_path / "demo_subject_001.csv", n_trials=1, n_reps=1)
    outputs, models, manifest, cwd, fake_root = _isolate(tmp_path, monkeypatch)
    assert ffr.main(["--recording", str(demo), "--id", "s1", *outputs]) == 2
    err = capsys.readouterr().err
    assert "--allow-demo" in err and err.count("\n") == 1
    assert not models.exists() and not manifest.exists()
    assert list(cwd.iterdir()) == [] and list(fake_root.iterdir()) == []


def test_demo_with_flag_prefixes_everything_and_refuses_overwrite(recording, tmp_path, monkeypatch, capsys):
    demo = tmp_path / "demo_subject_001.csv"
    demo.write_bytes(recording.read_bytes())
    outputs, models, man, cwd, fake_root = _isolate(tmp_path, monkeypatch)

    oof_seen, platt_seen = [], []
    real_oof = ffr.out_of_fold_scores

    class SpyPlatt(ffr.PlattCalibrator):
        def fit(self, scores=None, labels=None):
            platt_seen.append(np.array(scores))
            return super().fit(scores, labels)

    monkeypatch.setattr(ffr, "out_of_fold_scores", lambda *a, **k: oof_seen.append(real_oof(*a, **k)) or oof_seen[-1])
    monkeypatch.setattr(ffr, "PlattCalibrator", SpyPlatt)

    argv = ["--recording", str(demo), "--id", "s1", "--allow-demo", *outputs]
    assert ffr.main([*argv, "--n-splits", "3"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "out-of-fold AUC" in out
    # Platt is fitted on exactly the out-of-fold scores, never on in-sample ones
    np.testing.assert_array_equal(platt_seen[0], oof_seen[0])

    ckpt, cal = models / "demo_s1_classifier.joblib", models / "demo_s1_platt.json"
    assert ckpt.is_file() and cal.is_file() and man.is_file()
    manifest = mf.load_manifest(man)
    assert manifest.lm_config == {"type": "uniform"}
    assert manifest.fingerprint in out
    assert sorted(p.name for p in models.iterdir()) == ["demo_s1_classifier.joblib", "demo_s1_platt.json"]
    assert list(cwd.iterdir()) == [] and list(fake_root.iterdir()) == []

    before = {p: p.stat().st_mtime_ns for p in (ckpt, cal, man)}
    assert ffr.main([*argv, "--n-splits", "3"]) == 2
    err = capsys.readouterr().err
    assert "refusing to overwrite" in err and "--force" in err
    assert {p: p.stat().st_mtime_ns for p in (ckpt, cal, man)} == before
    # --force passes the overwrite check (and here fails later, on the split count)
    assert ffr.main([*argv, "--n-splits", "1", "--force"]) == 2
    assert "--n-splits" in capsys.readouterr().err
    assert list(fake_root.iterdir()) == []


def test_too_little_data_refused(tmp_path):
    rec = make_recording(tmp_path / "short.csv", n_trials=2, n_reps=1)
    with pytest.raises(ffr.FitError, match="too little usable data: 2 trials"):
        ffr.fit_from_recording(rec, "s1", models_dir=tmp_path / "models",
                               manifest_path=tmp_path / "m.json", log=lambda s: None)
    assert not (tmp_path / "models").exists()


def test_volts_recording_refused(tmp_path):
    rec = make_recording(tmp_path / "volts.csv", n_trials=3, n_reps=1, scale=1e-6)
    with pytest.raises(ffr.FitError, match="volts"):
        ffr.fit_from_recording(rec, "s1", models_dir=tmp_path / "models",
                               manifest_path=tmp_path / "m.json", log=lambda s: None)


def test_bad_inputs(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,Fz\n1.0,2.0\n")
    with pytest.raises(ffr.FitError, match="missing column"):
        ffr.load_recording(bad)
    with pytest.raises(ffr.FitError, match="not found"):
        ffr.load_recording(tmp_path / "nope.csv")
    with pytest.raises(ffr.FitError, match="safe filename"):
        ffr.output_paths(bad, "../evil", tmp_path)


def _repo_artifact_snapshot() -> dict:
    root = mf.REPO_ROOT
    paths = list(root.glob("*session_manifest.json"))
    if (root / "models").is_dir():
        paths += [p for p in (root / "models").rglob("*") if p.is_file()]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def test_fit_tests_never_write_into_the_repo():
    before = _repo_artifact_snapshot()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider",
         "-k", "not test_fit_tests_never_write_into_the_repo"],
        cwd=mf.REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert _repo_artifact_snapshot() == before
