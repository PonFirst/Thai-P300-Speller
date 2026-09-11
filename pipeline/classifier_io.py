"""
Classifier checkpoint I/O: the joblib dict written by
pipeline/fit_from_recording.py and served by pipeline/inference_service.py.
The format is specified in docs/INTERFACES.md ("Classifier checkpoint").

TRUST BOUNDARY: joblib is pickle, so load_checkpoint() executes code from the
file. Only load checkpoints you created. The session manifest's sha256 check
stops a checkpoint being swapped behind an unchanged path; it does not make an
untrusted file safe to open.

The pipeline itself comes from classifiers/xdawn_cov_ts_regLDA.build_pipeline()
and is stored unchanged. classifiers/ is not a package, so build_pipeline() is
loaded by file path (load_build_pipeline). Unpickling a fitted pipeline needs
only sklearn/pyriemann, not that module.
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

from pipeline.epoching import EpochSpec
from pipeline.manifest import REPO_ROOT, SCORE_METHOD

CHECKPOINT_FORMAT = "p300-classifier-v1"
LABELS = {"target": 1, "non_target": 0}
REQUIRED_KEYS = frozenset({"format", "pipeline", "epoch_spec", "labels", "score_method",
                           "trained_on", "versions"})
CLASSIFIER_MODULE_PATH = REPO_ROOT / "classifiers" / "xdawn_cov_ts_regLDA.py"
SFREQ_TOLERANCE_HZ = 1e-3


class CheckpointError(Exception):
    """A classifier checkpoint is missing, malformed, or unusable."""


class CheckpointMismatchError(CheckpointError):
    """The checkpoint was trained on data that differs from the live stream."""


@dataclass(frozen=True)
class Checkpoint:
    pipeline: Any
    epoch_spec: EpochSpec
    trained_on: dict
    versions: dict
    labels: dict = field(default_factory=lambda: dict(LABELS))
    path: Path | None = None


def library_versions() -> dict:
    import pyriemann
    import scipy
    import sklearn
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "pyriemann": pyriemann.__version__}


def load_build_pipeline() -> Callable:
    """classifiers/xdawn_cov_ts_regLDA.build_pipeline, imported by path, unchanged."""
    spec = importlib.util.spec_from_file_location("xdawn_cov_ts_regLDA", CLASSIFIER_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise CheckpointError(f"cannot import classifier module {CLASSIFIER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_pipeline


def _check_pipeline(pipeline, where: str) -> None:
    if not callable(getattr(pipeline, "decision_function", None)):
        raise CheckpointError(f"{where}: pipeline has no decision_function()")
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        raise CheckpointError(f"{where}: pipeline is not fitted (no classes_)")
    if [int(c) for c in classes] != [LABELS["non_target"], LABELS["target"]]:
        raise CheckpointError(
            f"{where}: pipeline classes_ must be [0, 1] so a positive decision_function means "
            f"target, got {list(classes)}")


def save_checkpoint(path, pipeline, epoch_spec: EpochSpec, trained_on: dict) -> Path:
    """Write the documented checkpoint dict atomically. Returns the resolved path."""
    if not isinstance(epoch_spec, EpochSpec):
        raise TypeError(f"epoch_spec must be an EpochSpec, got {type(epoch_spec).__name__}")
    if not isinstance(trained_on, dict):
        raise TypeError(f"trained_on must be a dict, got {type(trained_on).__name__}")
    _check_pipeline(pipeline, "save_checkpoint")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "pipeline": pipeline,
        "epoch_spec": epoch_spec.to_dict(),
        "labels": dict(LABELS),
        "score_method": SCORE_METHOD,
        "trained_on": dict(trained_on),
        "versions": library_versions(),
    }
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return path


def load_checkpoint(path) -> Checkpoint:
    """Load and validate a checkpoint. Executes pickle: see the module docstring."""
    path = Path(path).expanduser().resolve()
    where = f"checkpoint {path}"
    if not path.is_file():
        raise CheckpointError(f"{where}: file not found")
    try:
        data = joblib.load(path)
    except Exception as exc:  # corrupt/truncated file or incompatible library versions
        raise CheckpointError(f"{where}: cannot unpickle ({type(exc).__name__}: {exc})") from exc

    if not isinstance(data, dict):
        raise CheckpointError(f"{where}: must contain a dict, got {type(data).__name__}")
    missing = REQUIRED_KEYS - data.keys()
    extra = data.keys() - REQUIRED_KEYS
    if missing:
        raise CheckpointError(f"{where}: missing key(s) {sorted(missing)}")
    if extra:
        raise CheckpointError(f"{where}: unexpected key(s) {sorted(extra)}")
    if data["format"] != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{where}: format must be {CHECKPOINT_FORMAT!r}, got {data['format']!r}")
    if data["score_method"] != SCORE_METHOD:
        raise CheckpointError(
            f"{where}: score_method must be {SCORE_METHOD!r}, got {data['score_method']!r}")
    if data["labels"] != LABELS:
        raise CheckpointError(f"{where}: labels must be {LABELS}, got {data['labels']!r}")
    if not isinstance(data["epoch_spec"], dict):
        raise CheckpointError(f"{where}: epoch_spec must be a dict")
    try:
        spec = EpochSpec.from_dict(data["epoch_spec"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"{where}: invalid epoch_spec ({exc})") from exc
    for key in ("trained_on", "versions"):
        if not isinstance(data[key], dict):
            raise CheckpointError(f"{where}: {key} must be a dict")
    _check_pipeline(data["pipeline"], where)
    return Checkpoint(pipeline=data["pipeline"], epoch_spec=spec, trained_on=dict(data["trained_on"]),
                      versions=dict(data["versions"]), labels=dict(data["labels"]), path=path)


def version_warnings(checkpoint: Checkpoint) -> list[str]:
    """Library versions that differ from the ones the checkpoint was saved with."""
    current = library_versions()
    return [f"{lib} {checkpoint.versions.get(lib)!s} at training, {version} now"
            for lib, version in current.items() if checkpoint.versions.get(lib) != version]


def _label_key(label) -> str:
    return str(label).strip().casefold()


def channels_for_compatibility_check(checkpoint: Checkpoint, labels, channel_count: int):
    """
    Decide what the stream's channel labels let check_stream_compatibility()
    verify. Returns (channels argument, warning or None).

    The real Unicorn LSL Interface's labels are unverified, so:
      - no labels (None, empty, or any blank)         -> count only, no warning
      - labels equal the checkpoint's channels
        (ignoring case and surrounding whitespace)    -> count (labels already match)
      - no label names any checkpoint electrode
        ("EEG 1", "Ch1")                              -> count only, with a warning
      - electrode names that differ from the
        checkpoint (a real montage mismatch)          -> the labels, so the check raises
    """
    if not labels or not all(isinstance(label, str) and label.strip() for label in labels):
        return int(channel_count), None
    spec_keys = [_label_key(ch) for ch in checkpoint.epoch_spec.channels]
    label_keys = [_label_key(label) for label in labels]
    if label_keys == spec_keys:
        return len(labels), None
    if not set(label_keys) & set(spec_keys):
        return len(labels), (
            f"channel labels {list(labels)} name none of the checkpoint's electrodes "
            f"{list(checkpoint.epoch_spec.channels)}; checking channel count and sfreq only. "
            "Verify the montage order against the device.")
    return tuple(labels), None


def check_stream_compatibility(checkpoint: Checkpoint, channels, sfreq: float) -> None:
    """
    Raise CheckpointMismatchError unless the live stream matches the checkpoint.

    channels: the stream's channel labels (a sequence of str) or, when the
    stream publishes no labels, its channel count (int).
    """
    spec = checkpoint.epoch_spec
    if isinstance(channels, (int, np.integer)) and not isinstance(channels, bool):
        count, labels = int(channels), None
    else:
        labels = tuple(channels)
        count = len(labels)
    if count != len(spec.channels):
        raise CheckpointMismatchError(
            f"channel count: checkpoint expects {len(spec.channels)} {list(spec.channels)}, "
            f"stream has {count}")
    if labels is not None and labels != spec.channels:
        raise CheckpointMismatchError(
            f"channels: checkpoint expects {list(spec.channels)}, stream has {list(labels)}")
    if sfreq is None or not math.isfinite(float(sfreq)) or abs(float(sfreq) - spec.sfreq) > SFREQ_TOLERANCE_HZ:
        raise CheckpointMismatchError(
            f"sfreq: checkpoint expects {spec.sfreq:g} Hz, stream has {sfreq} Hz")


def score_epoch(checkpoint: Checkpoint, epoch) -> float:
    """decision_function score for one epoch shaped (n_channels, n_times); positive = target."""
    x = np.asarray(epoch, dtype=np.float64)
    spec = checkpoint.epoch_spec
    expected = (len(spec.channels), spec.n_times)
    if x.shape != expected:
        raise ValueError(f"epoch must have shape {expected} per the checkpoint's EpochSpec, got {x.shape}")
    scores = np.asarray(checkpoint.pipeline.decision_function(x[np.newaxis]))
    if scores.shape != (1,):
        raise CheckpointError(f"decision_function returned shape {scores.shape}, expected (1,)")
    return float(scores[0])
