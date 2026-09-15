import joblib
import numpy as np
import pytest

from pipeline import classifier_io as cio
from pipeline.classifier_io import CheckpointError, CheckpointMismatchError
from pipeline.constants import EEG_CHANNELS
from pipeline.epoching import EpochSpec

SPEC = EpochSpec()


def _toy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 5, size=(n, len(EEG_CHANNELS), SPEC.n_times))
    y = np.array([1 if i % 4 == 0 else 0 for i in range(n)])
    bump = np.exp(-0.5 * ((np.arange(SPEC.n_times) / SPEC.sfreq - 0.35) / 0.06) ** 2)
    X[y == 1] += 6.0 * bump
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = _toy_data()
    pipe = cio.load_build_pipeline()()
    pipe.fit(X, y)
    return pipe, X


def _save(tmp_path, pipe):
    return cio.save_checkpoint(tmp_path / "models" / "s1_classifier.joblib", pipe, SPEC,
                               {"recording_sha256": "0" * 64, "n_epochs": 60, "n_targets": 15, "n_trials": 3})


def test_roundtrip_reproduces_scores_and_metadata(tmp_path, fitted):
    pipe, X = fitted
    path = _save(tmp_path, pipe)
    ckpt = cio.load_checkpoint(path)
    assert ckpt.epoch_spec == SPEC
    assert ckpt.trained_on["n_trials"] == 3
    assert ckpt.labels == {"target": 1, "non_target": 0}
    assert set(ckpt.versions) == {"numpy", "scipy", "sklearn", "pyriemann"}
    assert cio.version_warnings(ckpt) == []
    np.testing.assert_array_equal(ckpt.pipeline.decision_function(X), pipe.decision_function(X))
    scores = [cio.score_epoch(ckpt, x) for x in X[:5]]
    # one epoch vs a batch of five can differ in the last bits (BLAS), not in value
    np.testing.assert_allclose(scores, pipe.decision_function(X[:5]), rtol=1e-12)
    assert all(isinstance(s, float) for s in scores)
    assert not list(path.parent.glob(".*.tmp"))


def test_raw_dict_matches_documented_format(tmp_path, fitted):
    data = joblib.load(_save(tmp_path, fitted[0]))
    assert set(data) == {"format", "pipeline", "epoch_spec", "labels", "score_method", "trained_on", "versions"}
    assert data["format"] == "p300-classifier-v1"
    assert data["score_method"] == "decision_function"
    assert data["epoch_spec"] == SPEC.to_dict()


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(format="p300-classifier-v0"), "format"),
    (lambda d: d.update(score_method="predict_proba"), "score_method"),
    (lambda d: d.pop("trained_on"), "missing"),
    (lambda d: d.update(extra=1), "unexpected"),
    (lambda d: d.update(labels={"target": 0, "non_target": 1}), "labels"),
    (lambda d: d["epoch_spec"].update(units="V"), "epoch_spec"),
    (lambda d: d["epoch_spec"].update(resample=2), "epoch_spec"),
    (lambda d: d.update(pipeline=object()), "decision_function"),
])
def test_invalid_checkpoint_rejected(tmp_path, fitted, mutate, match):
    path = _save(tmp_path, fitted[0])
    data = joblib.load(path)
    mutate(data)
    joblib.dump(data, path)
    with pytest.raises(CheckpointError, match=match):
        cio.load_checkpoint(path)


def test_non_dict_and_missing_file(tmp_path):
    path = tmp_path / "x.joblib"
    joblib.dump([1, 2, 3], path)
    with pytest.raises(CheckpointError, match="must contain a dict"):
        cio.load_checkpoint(path)
    with pytest.raises(CheckpointError, match="not found"):
        cio.load_checkpoint(tmp_path / "missing.joblib")
    (tmp_path / "junk.joblib").write_bytes(b"not a pickle")
    with pytest.raises(CheckpointError, match="cannot unpickle"):
        cio.load_checkpoint(tmp_path / "junk.joblib")


def test_unfitted_pipeline_refused_at_save(tmp_path):
    with pytest.raises(CheckpointError, match="not fitted"):
        cio.save_checkpoint(tmp_path / "c.joblib", cio.load_build_pipeline()(), SPEC, {})


def test_compatible_stream_accepted(tmp_path, fitted):
    ckpt = cio.load_checkpoint(_save(tmp_path, fitted[0]))
    cio.check_stream_compatibility(ckpt, list(EEG_CHANNELS), 250.0)
    cio.check_stream_compatibility(ckpt, 8, 250.0)


@pytest.mark.parametrize("channels,sfreq,match", [
    (7, 250.0, "channel count"),
    (EEG_CHANNELS + ["X"], 250.0, "channel count"),
    (list(reversed(EEG_CHANNELS)), 250.0, "^channels"),
    (8, 500.0, "sfreq"),
    (list(EEG_CHANNELS), 0.0, "sfreq"),
])
def test_incompatible_stream_names_the_field(tmp_path, fitted, channels, sfreq, match):
    ckpt = cio.load_checkpoint(_save(tmp_path, fitted[0]))
    with pytest.raises(CheckpointMismatchError, match=match):
        cio.check_stream_compatibility(ckpt, channels, sfreq)


def test_channel_label_policy(tmp_path, fitted):
    ckpt = cio.load_checkpoint(_save(tmp_path, fitted[0]))
    policy = cio.channels_for_compatibility_check

    # no usable labels: count only, silently
    assert policy(ckpt, None, 8) == (8, None)
    assert policy(ckpt, [], 8) == (8, None)
    assert policy(ckpt, EEG_CHANNELS[:7] + [""], 8) == (8, None)
    # matching electrode labels, ignoring case and whitespace
    assert policy(ckpt, list(EEG_CHANNELS), 8) == (8, None)
    assert policy(ckpt, [f" {ch.upper()} " for ch in EEG_CHANNELS], 8) == (8, None)

    # generic labels: count and sfreq only, with a warning naming the labels
    for generic in ([f"EEG {i + 1}" for i in range(8)], [f"Ch{i + 1}" for i in range(8)]):
        channels, warning = policy(ckpt, generic, 8)
        assert channels == 8
        assert generic[0] in warning and "count and sfreq only" in warning
        cio.check_stream_compatibility(ckpt, channels, 250.0)
    channels, warning = policy(ckpt, [f"EEG {i + 1}" for i in range(7)], 7)
    assert warning is not None
    with pytest.raises(CheckpointMismatchError, match="channel count"):
        cio.check_stream_compatibility(ckpt, channels, 250.0)

    # electrode names that differ are a real montage mismatch. The second
    # list is the reference paper's own montage (CLAUDE.md hard constraints)
    # -- a real, different set of electrode names, used here only as an
    # example of a mismatch, not as this project's channel identity.
    reference_paper_montage = ["Fz", "F3", "C4", "Cz", "Pz", "P3", "O2", "O1"]
    for labels in (list(reversed(EEG_CHANNELS)), reference_paper_montage):
        channels, warning = policy(ckpt, labels, 8)
        assert warning is None and channels == tuple(labels)
        with pytest.raises(CheckpointMismatchError, match="^channels"):
            cio.check_stream_compatibility(ckpt, channels, 250.0)


def test_score_epoch_checks_shape(tmp_path, fitted):
    ckpt = cio.load_checkpoint(_save(tmp_path, fitted[0]))
    with pytest.raises(ValueError, match=r"\(8, 200\)"):
        cio.score_epoch(ckpt, np.zeros((8, 199)))
    with pytest.raises(ValueError, match="shape"):
        cio.score_epoch(ckpt, np.zeros((200, 8)))
