"""
Experiment logs: the per-session and per-fit records that make a run auditable.

Every file is keyed off the recording CSV path, the sidecar convention
collection/record_session.py already uses:

  <csv>                        EEG + markers                                   (recorder)
  <csv>.log.jsonl              recorder process log: connection, gaps, warnings (recorder)
  <csv>.integrity.json         recorder exit summary                            (recorder)
  <csv>.session.json           what was spelled, settings, timeline, file links (GUI)
  <csv>.inference.log.jsonl    live inference: the LM prior at each character's
                               start, every flash update, every commit          (service)
  models/<id>_fit_report.json  classifier training log                          (fit script)

This module is the single definition of those names and of the session record
and fit report schemas. The GUI, the inference service launcher, the fit script
and the tests import from here; none of them spell the suffixes themselves.
Pure functions plus one atomic JSON writer: no LSL, no Qt.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SESSION_FORMAT = "p300-session-v1"
FIT_REPORT_FORMAT = "p300-fit-report-v1"

SESSION_SUFFIX = ".session.json"
INFERENCE_LOG_SUFFIX = ".inference.log.jsonl"
RECORDER_LOG_SUFFIX = ".log.jsonl"
INTEGRITY_SUFFIX = ".integrity.json"
FIT_REPORT_SUFFIX = "_fit_report.json"
_CLASSIFIER_STEM_SUFFIX = "_classifier"

KIND_CALIBRATION = "calibration"
KIND_INFERENCE = "inference"


def _sidecar(recording, suffix: str) -> Path:
    return Path(str(recording) + suffix)


def session_record_path(recording) -> Path:
    return _sidecar(recording, SESSION_SUFFIX)


def inference_log_path(recording) -> Path:
    return _sidecar(recording, INFERENCE_LOG_SUFFIX)


def recorder_log_path(recording) -> Path:
    return _sidecar(recording, RECORDER_LOG_SUFFIX)


def integrity_path(recording) -> Path:
    return _sidecar(recording, INTEGRITY_SUFFIX)


def fit_report_path(checkpoint_path) -> Path:
    """models/<id>_classifier.joblib -> models/<id>_fit_report.json (demo_ prefix carries over)."""
    checkpoint_path = Path(checkpoint_path)
    stem = checkpoint_path.stem
    if stem.endswith(_CLASSIFIER_STEM_SUFFIX):
        stem = stem[: -len(_CLASSIFIER_STEM_SUFFIX)]
    return checkpoint_path.with_name(stem + FIT_REPORT_SUFFIX)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sig_floats(values, digits: int = 6) -> list[float]:
    """
    Floats rounded to `digits` significant digits, for logging probability vectors.
    Significant digits, not decimal places, so a 1e-8 probability keeps its magnitude
    instead of collapsing to 0.
    """
    return [float(f"{float(v):.{digits}g}") for v in np.asarray(values, dtype=float).reshape(-1)]


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path, set, frozenset)):
        return str(obj) if isinstance(obj, Path) else sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def write_json_atomic(data: dict, path) -> Path:
    """Atomic write, key order preserved (these files are read by people)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return path


def read_json_if_exists(path) -> dict | None:
    """The parsed JSON object, or None if the file is missing, unreadable, or not an object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def build_session_record(*, session_id: str, demo: bool, targets: str, characters: list[str],
                         settings: dict, recording, manifest: dict | None,
                         started_utc: str) -> dict:
    """
    The record written when a session starts.

    `targets_text` is exactly what was typed into the panel: for a calibration
    session it is the calibration sentence. `target_sequence` lists it one
    character per trial, in order, with each character's index in the matrix.
    `kind` is derived: a session that ran with live inference (a manifest) is
    "inference", one that only recorded is "calibration".
    """
    index_of = {c: i for i, c in enumerate(characters)}
    live = manifest is not None
    return {
        "format": SESSION_FORMAT,
        "session_id": session_id,
        "demo": bool(demo),
        "kind": KIND_INFERENCE if live else KIND_CALIBRATION,
        "live_inference": live,
        "targets_text": targets,
        "target_sequence": [{"order": i + 1, "char": c, "char_idx": index_of.get(c)}
                            for i, c in enumerate(targets)],
        "settings": dict(settings),
        "manifest": manifest,
        "files": {
            "recording": str(recording),
            "recorder_log": str(recorder_log_path(recording)),
            "integrity": str(integrity_path(recording)),
            "inference_log": str(inference_log_path(recording)) if live else None,
        },
        "started_utc": started_utc,
        "ended_utc": None,
        "end": None,
    }


def finalize_session_record(record: dict, *, ended_utc: str, message: str, level: str,
                            phase_at_end: str) -> dict:
    """
    Adds the end-of-session fields. `phase_at_end` says how far the session got: a stop
    while still waiting for the service or recorder means no flashes were ever shown.
    """
    done = dict(record)
    done["ended_utc"] = ended_utc
    done["end"] = {"message": message, "level": level, "phase_at_end": phase_at_end}
    return done
