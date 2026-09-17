"""
Tests for stopping/evaluate_recording.py: the real-recording replay/ITR
evaluator. No test writes into the repo -- everything lands under tmp_path.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

from pipeline import fit_from_recording as ffr  # noqa: E402
from pipeline import manifest as mf  # noqa: E402
from pipeline.constants import EEG_CHANNELS  # noqa: E402
from test_fit_from_recording import make_recording  # noqa: E402

from stopping import evaluate_recording as evalr  # noqa: E402

SFREQ = 250.0
SOA = 0.124
T0 = 5000.0


# ----------------------------------------------------------------------
# Pure logic
# ----------------------------------------------------------------------

def _flash(trial_id, flash_idx, onset, is_target, group):
    return evalr.ReplayFlash(trial_id, flash_idx, onset, is_target, tuple(group))


def test_build_group_table_dedups_by_membership_not_list_order():
    flashes = [_flash(0, 0, 1.0, True, [3, 1, 2]), _flash(0, 1, 1.1, False, [2, 3, 1])]
    groups, group_ids = evalr.build_group_table(flashes)
    assert group_ids == [0, 0]
    assert len(groups) == 1


def test_build_group_table_assigns_in_first_appearance_order():
    flashes = [_flash(0, 0, 1.0, True, [1, 2]), _flash(0, 1, 1.1, False, [3, 4]),
              _flash(0, 2, 1.2, False, [1, 2])]
    groups, group_ids = evalr.build_group_table(flashes)
    assert groups == [[1, 2], [3, 4]]
    assert group_ids == [0, 1, 0]


def test_empirical_seconds_per_flash_is_median_within_trial_gap():
    by_trial = {
        0: [_flash(0, 0, 0.000, True, [1]), _flash(0, 1, 0.124, False, [2]), _flash(0, 2, 0.248, False, [3])],
        1: [_flash(1, 0, 10.0, True, [1]), _flash(1, 1, 10.130, False, [2])],
    }
    # gaps: 0.124, 0.124 (trial 0), 0.130 (trial 1) -> median 0.124
    assert evalr.empirical_seconds_per_flash(by_trial) == pytest.approx(0.124)


def test_empirical_seconds_per_flash_raises_without_two_flash_trials():
    by_trial = {0: [_flash(0, 0, 0.0, True, [1])], 1: [_flash(1, 0, 5.0, True, [2])]}
    with pytest.raises(evalr.EvaluationError, match="seconds-per-flash"):
        evalr.empirical_seconds_per_flash(by_trial)


# ----------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------

def _write_minimal_csv(path, rows):
    """rows: list of (timestamp, {8 channel values}, marker_json or None)."""
    n = len(rows)
    columns = {"timestamp": [r[0] for r in rows]}
    for i, ch in enumerate(EEG_CHANNELS):
        columns[ch] = [r[1][i] for r in rows]
    columns["marker_json_or_empty"] = [json.dumps(r[2]) if r[2] is not None else "" for r in rows]
    for extra in ("trial_id", "flash_idx", "group_json", "is_target", "target_char_idx"):
        columns[extra] = [""] * n
    pd.DataFrame(columns).to_csv(path, index=False)


def test_load_recording_for_replay_extracts_group_and_target(tmp_path):
    clean = [1.0] * 8
    rows = [
        (0.000, clean, {"event": "trial_start", "trial_id": 0, "target_char_idx": 5, "target_char": "c5"}),
        (0.010, clean, {"event": "flash", "trial_id": 0, "flash_idx": 0, "group": [5, 9, 2], "is_target": True}),
        (0.020, clean, {"event": "flash", "trial_id": 0, "flash_idx": 1, "group": [1, 3], "is_target": False}),
        (0.030, clean, None),
    ]
    path = tmp_path / "rec.csv"
    _write_minimal_csv(path, rows)

    rec = evalr.load_recording_for_replay(path, n_chars=72)
    assert rec.targets == {0: 5}
    assert len(rec.flashes) == 2
    assert rec.flashes[0] == evalr.ReplayFlash(0, 0, 0.010, True, (5, 9, 2))
    assert rec.flashes[1].group == (1, 3)


def test_load_recording_for_replay_counts_malformed_and_duplicate(tmp_path):
    clean = [1.0] * 8
    rows = [
        (0.000, clean, {"event": "trial_start", "trial_id": 0, "target_char_idx": 5}),
        (0.010, clean, {"event": "flash", "trial_id": 0, "flash_idx": 0, "group": [5, 9], "is_target": True}),
        (0.020, clean, {"event": "flash", "trial_id": 0, "flash_idx": 0, "group": [5, 9], "is_target": True}),  # dup
        (0.030, clean, {"event": "flash", "trial_id": 0, "flash_idx": 1, "group": [], "is_target": False}),  # bad group
        (0.040, clean, {"event": "flash", "trial_id": 0}),  # missing fields
    ]
    path = tmp_path / "rec.csv"
    _write_minimal_csv(path, rows)

    rec = evalr.load_recording_for_replay(path, n_chars=72)
    assert len(rec.flashes) == 1
    assert rec.n_duplicate_flashes == 1
    assert rec.n_malformed_markers == 2


def test_missing_recording_raises():
    with pytest.raises(evalr.EvaluationError, match="not found"):
        evalr.load_recording_for_replay("/no/such/file.csv", n_chars=72)


# ----------------------------------------------------------------------
# End to end: fit a real checkpoint, then evaluate against a recording
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_manifest(tmp_path_factory):
    rec_dir = tmp_path_factory.mktemp("rec")
    recording = make_recording(rec_dir / "subject_001.csv", n_trials=8, n_reps=2, seed=1)
    out = tmp_path_factory.mktemp("fit")
    result = ffr.fit_from_recording(recording, "subject_001", models_dir=out / "models",
                                    manifest_path=out / "session_manifest.json")
    return out / "session_manifest.json", recording


def test_run_evaluation_end_to_end_produces_a_sane_table(fitted_manifest):
    manifest_path, recording = fitted_manifest
    df = evalr.run_evaluation(manifest_path, recording, theta=[0.6, 0.9], t_min=[1], log=lambda *_: None)

    expected_cols = {"theta", "t_min", "t_max", "lam", "llr_clip", "sigmoid_a", "sigmoid_b",
                     "mean_accuracy", "mean_flashes", "itr_bits_per_min"}
    assert expected_cols <= set(df.columns)
    assert len(df) == 2  # one row per swept theta
    assert set(df["theta"]) == {0.6, 0.9}
    assert (df["mean_accuracy"].between(0, 1)).all()
    assert (df["mean_flashes"] > 0).all()
    assert np.all(np.isfinite(df["itr_bits_per_min"]))
    # sorted by ITR descending
    assert list(df["itr_bits_per_min"]) == sorted(df["itr_bits_per_min"], reverse=True)


def test_run_evaluation_uses_manifest_defaults_when_grid_omitted(fitted_manifest):
    manifest_path, recording = fitted_manifest
    df = evalr.run_evaluation(manifest_path, recording, log=lambda *_: None)
    manifest = mf.load_manifest(manifest_path)
    assert len(df) == 1
    assert df.iloc[0]["theta"] == manifest.stopping["theta"]
    assert df.iloc[0]["t_min"] == manifest.stopping["t_min"]


def test_demo_recording_refused_without_allow_demo(fitted_manifest, tmp_path):
    manifest_path, recording = fitted_manifest
    demo_copy = tmp_path / "demo_subject_001.csv"
    demo_copy.write_bytes(Path(recording).read_bytes())

    with pytest.raises(evalr.EvaluationError, match="demo_"):
        evalr.run_evaluation(manifest_path, demo_copy, log=lambda *_: None)

    # explicit opt-in proceeds (and produces the same shape of result)
    df = evalr.run_evaluation(manifest_path, demo_copy, allow_demo=True, log=lambda *_: None)
    assert len(df) == 1


def test_cli_writes_csv(fitted_manifest, tmp_path, capsys):
    manifest_path, recording = fitted_manifest
    out_path = tmp_path / "results.csv"
    code = evalr.main(["--manifest", str(manifest_path), "--recording", str(recording),
                       "--out", str(out_path)])
    assert code == 0
    assert out_path.is_file()
    saved = pd.read_csv(out_path)
    assert "itr_bits_per_min" in saved.columns


def test_cli_bad_manifest_exits_2(tmp_path, capsys):
    code = evalr.main(["--manifest", str(tmp_path / "missing.json"), "--recording", str(tmp_path / "x.csv")])
    assert code == 2
    assert "evaluate_recording: error:" in capsys.readouterr().err
