"""
Fit the classifier and its Platt calibration from a recorded calibration
session, and write the session manifest that every process loads.

    python -m pipeline.fit_from_recording --recording collection/recordings/subject_001.csv \\
        --id subject_001 [--models-dir models] [--manifest session_manifest.json] \\
        [--lm-corpus corpus.txt | --lm-transformer-checkpoint checkpoint.pt] \\
        [--allow-demo] [--force] [--n-splits 5] [--characters PATH]

Several BCI2000 .dat runs of one calibration may be given (--recording R01.dat R02.dat): each is
epoched on its own, and each run's selections stay separate cross-validation groups.

The recording is either a recorder CSV (the Python/LSL stack) or a BCI2000 .dat written by the
CheckerboardSpeller (pipeline/bci2000_recording.py: onsets from StimulusCode, labels from
StimulusType, selections as trials, lost Bluetooth samples rejected). A .dat fit also writes
models/<id>_bci2000.prm, the model for BCI2000's CheckerboardSignalProcessing
(pipeline/bci2000_export.py). Synthetic .dat data (the fake headset's gyroscope pattern) counts
as demo data.

Steps:
  1. Load the recorder CSV. Rows whose 8 channels are all finite form the
     continuous EEG; anything else (gap rows, marker rows inside a gap) is
     not EEG. Flash onsets and is_target labels come from flash marker rows;
     a flash's onset is its row's timestamp.
  2. Cut epochs with pipeline.epoching and the default EpochSpec, the same
     code the live service uses, and report rejections per reason.
  3. Out-of-fold decision_function scores via GroupKFold grouped by
     trial_id, a fresh build_pipeline() per fold. Platt (a, b) and ECE are
     fitted on those scores only: in-sample scores are overconfident and
     would inflate every LLR (docs/INTERFACES.md). The ECE is measured on the
     same out-of-fold scores the two Platt parameters were fitted to.
  4. Fit on all epochs, save the checkpoint, write the calibration JSON with
     the checkpoint's sha256, then build and write the manifest.
  5. Write models/<id>_fit_report.json (pipeline/experiment_log.py): the training log, with
     per-fold AUC, the recording's calibration sentence from <recording>.session.json, and
     every line printed above. The artifacts exist by then, so a report that can't be
     written is a warning, not a failure.

Safety:
  - A recording named demo_* is refused unless --allow-demo; with it, the
    checkpoint, calibration, and (when not given explicitly) the manifest are
    all prefixed demo_, so a model fitted on synthetic data cannot pass for a
    real one.
  - Existing artifacts are never overwritten without --force. The check runs
    before any fitting.

Exit codes: 0 success, 2 refused or failed (one line on stderr).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from collection.unicorn_protocol import SYNTHETIC_GYRO_COUNTS  # the fake headset's gyroscope marker
from pipeline import bci2000_export, bci2000_lm_export
from pipeline import experiment_log as xlog
from pipeline.bci2000_dat import DatError, read_dat
from pipeline.bci2000_recording import RecordingError, counter_breaks, eeg_microvolts, flash_events
from pipeline.classifier_io import CheckpointError, library_versions, load_build_pipeline, save_checkpoint
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ
from pipeline.epoching import RANK_DEFICIENT, EpochSpec, extract_epochs, is_rank_deficient
from pipeline.language_model import LanguageModelError
from pipeline.manifest import (CALIBRATION_FORMAT, DEFAULT_STOPPING, REPO_ROOT, SCORE_METHOD, ManifestError,
                               build_manifest_dict, load_characters_file, sha256_file, write_manifest)
from stopping.calibrate import PlattCalibrator, expected_calibration_error

DEMO_PREFIX = "demo_"
DEFAULT_CHARACTERS_PATH = REPO_ROOT / "collection" / "characters_th.json"
DEFAULT_MANIFEST_NAME = "session_manifest.json"
MIN_TRIALS = 3
MIN_TARGET_EPOCHS = 10
MIN_NONTARGET_EPOCHS = 10
# Median |sample| below this is volts, not microvolts (CLAUDE.md gotcha 1).
MIN_PLAUSIBLE_MEDIAN_ABS_UV = 1e-3
REQUIRED_COLUMNS = ["timestamp", *EEG_CHANNELS, "marker_json_or_empty"]
BCI2000_SUFFIX = ".dat"
_SUBJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class FitError(Exception):
    """The recording or the requested output cannot produce a valid model."""


@dataclass(frozen=True)
class FlashMarker:
    trial_id: int
    flash_idx: int
    onset: float
    is_target: bool


@dataclass
class Recording:
    path: Path
    sha256: str
    timestamps: np.ndarray
    data: np.ndarray
    flashes: list
    n_rows: int
    n_non_eeg_rows: int
    n_malformed_markers: int
    n_duplicate_flashes: int
    kind: str = "csv"                      # "csv" | "bci2000"
    bci2000: dict | None = None            # what the .dat says about its session, for the fit report


@dataclass
class FitResult:
    demo: bool
    n_flashes: int
    n_epochs: int
    n_targets: int
    n_trials: int
    rejected: dict
    n_splits: int
    auc: float
    a: float
    b: float
    ece: float
    checkpoint_path: Path
    calibration_path: Path
    manifest_path: Path
    fingerprint: str
    warnings: list = field(default_factory=list)
    report_path: Path | None = None
    bci2000_prm_path: Path | None = None


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_bci2000(path) -> bool:
    return Path(path).suffix.lower() == BCI2000_SUFFIX


REAL_SOURCE_FILTER = "UnicornSerialADC"


def dat_synthetic_reason(dat) -> str | None:
    """
    None only when the recording proves it came from the real headset: the source module was
    UnicornSerialSource reading the Bluetooth port (no TcpAddress), and the gyroscope states don't
    carry the fake headset's marker (0x5A5A on all three axes). Anything else -- the fake headset, a
    simulated BCI2000 source, missing evidence -- is synthetic: CLAUDE.md, synthetic data must
    identify itself, and absence of evidence doesn't count as real.
    """
    chain = dat.params.get("SignalSourceFilterChain")
    filters = [row[0] for row in chain.value] if chain is not None and chain.type == "matrix" else []
    if REAL_SOURCE_FILTER not in filters:
        return f"its source was {', '.join(f for f in filters if f != 'DataIOFilter') or 'not recorded'}, not the headset"
    if "TcpAddress" in dat.params and dat.param_str("TcpAddress").strip():
        return f"the source read TcpAddress {dat.param_str('TcpAddress')} (the fake headset), not the Bluetooth port"
    if not all(f"Gyroscope{a}" in dat.states for a in "XYZ"):
        return "it has no gyroscope states"
    if dat.n_samples and all(bool(np.all(dat.state(f"Gyroscope{a}") == g))
                             for a, g in zip("XYZ", SYNTHETIC_GYRO_COUNTS)):
        return "its gyroscope states carry the fake headset's marker"
    return None


def dat_is_synthetic(dat) -> bool:
    return dat_synthetic_reason(dat) is not None


def load_bci2000_recording(path, dat=None) -> Recording:
    path = Path(path)
    try:
        dat = dat if dat is not None else read_dat(path)
        flashes_dat = flash_events(dat)
    except (DatError, RecordingError) as exc:
        raise FitError(str(exc)) from exc
    data = eeg_microvolts(dat)
    lost = int((~np.all(np.isfinite(data), axis=1)).sum())
    flashes = [FlashMarker(trial_id=f.selection, flash_idx=f.code, onset=f.onset / EEG_SFREQ, is_target=f.is_target)
               for f in flashes_dat]
    breaks = counter_breaks(dat)
    meta = {
        "text_to_spell": dat.param_str("TextToSpell") if "TextToSpell" in dat.params else None,
        "synthetic_reason": dat_synthetic_reason(dat),
        "counter_breaks": int(len(breaks)),
        "truncated_tail_bytes": int(dat.truncated_tail_bytes),
        "n_selections": len({f.selection for f in flashes_dat}),
        "n_samples": dat.n_samples, "lost_samples": lost,
        "synthetic": dat_is_synthetic(dat),
        "subject": dat.param_str("SubjectName") if "SubjectName" in dat.params else None,
        "session": dat.param_str("SubjectSession") if "SubjectSession" in dat.params else None,
        "run": dat.param_str("SubjectRun") if "SubjectRun" in dat.params else None,
        "number_of_sequences": dat.param_str("NumberOfSequences") if "NumberOfSequences" in dat.params else None,
        "stimulus_duration": dat.param_str("StimulusDuration") if "StimulusDuration" in dat.params else None,
        "isi": dat.param_str("ISIMinDuration") if "ISIMinDuration" in dat.params else None,
    }
    return Recording(path=path, sha256=sha256_file(path), timestamps=np.arange(dat.n_samples) / EEG_SFREQ,
                     data=data, flashes=flashes, n_rows=dat.n_samples, n_non_eeg_rows=lost,
                     n_malformed_markers=0, n_duplicate_flashes=0, kind="bci2000", bci2000=meta)


def load_recording(path) -> Recording:
    path = Path(path)
    if not path.is_file():
        raise FitError(f"recording not found: {path}")
    if is_bci2000(path):
        return load_bci2000_recording(path)
    try:
        df = pd.read_csv(path, dtype={"marker_json_or_empty": str}, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise FitError(f"recording {path}: cannot parse CSV ({exc})") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FitError(f"recording {path}: missing column(s) {missing}")

    ts_all = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    data_all = np.column_stack([pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
                                for ch in EEG_CHANNELS])
    is_eeg = np.isfinite(ts_all) & np.all(np.isfinite(data_all), axis=1)

    flashes, seen = [], set()
    n_malformed = n_duplicate = 0
    markers = df["marker_json_or_empty"].fillna("")
    for row in np.flatnonzero(markers.str.strip().to_numpy() != ""):
        try:
            event = json.loads(markers.iat[row])
        except json.JSONDecodeError:
            n_malformed += 1
            continue
        if not isinstance(event, dict) or event.get("event") != "flash":
            continue
        trial_id, flash_idx, is_target = event.get("trial_id"), event.get("flash_idx"), event.get("is_target")
        onset = ts_all[row]
        if not (_is_int(trial_id) and _is_int(flash_idx) and isinstance(is_target, bool)
                and np.isfinite(onset)):
            n_malformed += 1
            continue
        if (trial_id, flash_idx) in seen:
            n_duplicate += 1
            continue
        seen.add((trial_id, flash_idx))
        flashes.append(FlashMarker(trial_id, flash_idx, float(onset), is_target))

    return Recording(path=path, sha256=sha256_file(path), timestamps=ts_all[is_eeg],
                     data=data_all[is_eeg], flashes=flashes, n_rows=len(df),
                     n_non_eeg_rows=int((~is_eeg).sum()), n_malformed_markers=n_malformed,
                     n_duplicate_flashes=n_duplicate)


def output_paths(recording_path, subject_id: str, models_dir, manifest_path=None,
                 allow_demo: bool = False, synthetic: str | None = None) -> tuple[bool, Path, Path, Path]:
    """(is_demo, checkpoint, calibration, manifest). Raises FitError for a refused demo recording.
    synthetic: why the recording itself says it is synthetic (a .dat not from the real headset), or None."""
    if not _SUBJECT_ID_RE.match(subject_id or ""):
        raise FitError(f"--id {subject_id!r} must be letters, digits, '_', '.', '-' (a safe filename)")
    demo = Path(recording_path).name.startswith(DEMO_PREFIX) or bool(synthetic)
    if demo and not allow_demo:
        why = synthetic if synthetic else "demo_ prefix"
        raise FitError(
            f"recording {Path(recording_path).name} is synthetic ({why}); refusing to fit a "
            "model on it. Pass --allow-demo to fit anyway; every artifact will be prefixed demo_.")
    prefix = DEMO_PREFIX if demo else ""
    stem = subject_id if subject_id.startswith(prefix) else prefix + subject_id
    # Relative --models-dir and the default manifest resolve against REPO_ROOT, not the cwd: the GUI
    # loads REPO_ROOT/session_manifest.json (demo_session_manifest.json in demo mode).
    models_dir = Path(models_dir)
    if not models_dir.is_absolute():
        models_dir = REPO_ROOT / models_dir
    manifest = Path(manifest_path) if manifest_path is not None else REPO_ROOT / (prefix + DEFAULT_MANIFEST_NAME)
    return (demo, models_dir / f"{stem}_classifier.joblib", models_dir / f"{stem}_platt.json", manifest)


def bci2000_prm_path(checkpoint_path) -> Path:
    """models/<stem>_classifier.joblib -> models/<stem>_bci2000.prm"""
    p = Path(checkpoint_path)
    stem = p.name[:-len("_classifier.joblib")] if p.name.endswith("_classifier.joblib") else p.stem
    return p.with_name(f"{stem}_bci2000.prm")


def _write_json_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def out_of_fold_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int,
                       build_pipeline: Callable, fold_log: list | None = None) -> np.ndarray:
    """
    decision_function scores where every epoch is scored by a model that never saw its trial.
    If fold_log is a list, one summary dict per fold is appended to it (for the fit report).
    """
    oof = np.full(len(y), np.nan)
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=groups)):
        if len(np.unique(y[train])) < 2:
            raise FitError(f"fold {fold}: training split has only one class; record more trials")
        pipe = build_pipeline()
        pipe.fit(X[train], y[train])
        oof[test] = pipe.decision_function(X[test])
        if fold_log is not None:
            both_classes = len(np.unique(y[test])) == 2
            fold_log.append({
                "fold": fold, "n_train_epochs": int(len(train)), "n_test_epochs": int(len(test)),
                "n_test_trials": int(len(np.unique(groups[test]))),
                "auc": float(roc_auc_score(y[test], oof[test])) if both_classes else None})
    if not np.all(np.isfinite(oof)):
        raise FitError("out-of-fold decision scores contain non-finite values")
    return oof


# A calibration may span several BCI2000 runs (rests between them). These must agree across the runs;
# a different session only warns (same person, but the cap was put on again).
SAME_ACROSS_RUNS = ("SubjectName", "SamplingRate", "ChannelNames", "TargetDefinitions", "SourceChGain",
                    "SourceChOffset")
GROUP_STRIDE = 1_000_000   # cross-validation groups: run index * stride + selection, unique across runs


def recording_list(recording) -> list[Path]:
    """One recording, or several BCI2000 .dat runs of one calibration."""
    paths = [Path(r) for r in recording] if isinstance(recording, (list, tuple)) else [Path(recording)]
    if not paths:
        raise FitError("no recording given")
    if len(paths) > 1:
        if not all(is_bci2000(p) for p in paths):
            raise FitError("several recordings are supported for BCI2000 .dat runs only; give one CSV")
        if len({p.resolve() for p in paths}) != len(paths):
            raise FitError("the same recording is given twice")
    return paths


def check_runs_match(paths: list[Path], dats: list) -> list[str]:
    """Refuses runs that can't share one model; returns warnings for ones that merely differ in session."""
    def value(dat, name):
        return dat.params[name].value if name in dat.params else None
    for name in SAME_ACROSS_RUNS:
        first = value(dats[0], name)
        for path, dat in zip(paths[1:], dats[1:]):
            if value(dat, name) != first:
                raise FitError(f"{path.name} and {paths[0].name} differ in {name}: one model can't be fit "
                               "across them")
    sessions = {str(value(d, "SubjectSession")) for d in dats}
    return [f"the runs come from different sessions ({', '.join(sorted(sessions))}): the cap was put on "
            "again in between"] if len(sessions) > 1 else []


def fit_from_recording(recording, subject_id: str, *, models_dir="models", manifest_path=None,
                       lm_corpus=None, lm_transformer_checkpoint=None, allow_demo: bool = False,
                       force: bool = False, n_splits: int = 5, characters_path=DEFAULT_CHARACTERS_PATH,
                       log: Callable[[str], None] = print) -> FitResult:
    paths = recording_list(recording)
    recording = paths[0]
    dats, run_warnings = [], []
    if is_bci2000(recording):
        for path in paths:
            if not path.is_file():
                raise FitError(f"recording not found: {path}")
            try:
                dats.append(read_dat(path))
            except DatError as exc:
                raise FitError(str(exc)) from exc
        run_warnings = check_runs_match(paths, dats)
    synthetic = next((f"{p.name}: {why}" if len(paths) > 1 else why
                      for p, why in ((p, dat_synthetic_reason(d)) for p, d in zip(paths, dats)) if why), None)
    demo, ckpt_path, cal_path, man_path = output_paths(recording, subject_id, models_dir, manifest_path, allow_demo,
                                                       synthetic=synthetic)
    # Every fit owns <stem>_bci2000.prm: a .dat fit writes it; a CSV fit removes an old one (under --force),
    # which would otherwise sit beside the new checkpoint and load the previous model into BCI2000.
    prm_path = bci2000_prm_path(ckpt_path)
    writes_prm = bool(dats)
    # The speller's own copy of the Transformer prior (bci2000_lm_export); the same ownership rule.
    lm_bin_path = prm_path.with_name(prm_path.name.replace("_bci2000.prm", "_lm.lmbin"))
    writes_lm_bin = writes_prm and lm_transformer_checkpoint is not None
    warnings = list(run_warnings)
    console: list[str] = []  # everything logged below, kept for the fit report
    t_start = time.monotonic()

    def emit(line: str) -> None:
        console.append(line)
        log(line)

    report_path = xlog.fit_report_path(ckpt_path)
    if demo and manifest_path is not None and not man_path.name.startswith(DEMO_PREFIX):
        warnings.append(f"demo recording, but --manifest {man_path} is not demo_-prefixed")
    existing = [str(p) for p in (ckpt_path, cal_path, man_path, report_path, prm_path, lm_bin_path) if p.exists()]
    if existing and not force:
        raise FitError(f"refusing to overwrite existing artifact(s) {existing}; pass --force to replace them")
    if n_splits < 2:
        raise FitError(f"--n-splits must be >= 2, got {n_splits}")
    load_characters_file(characters_path)
    if lm_corpus is not None and lm_transformer_checkpoint is not None:
        raise FitError("--lm-corpus and --lm-transformer-checkpoint are mutually exclusive; pick one LM")
    if dats and lm_corpus is not None:
        raise FitError("BCI2000's speller has the Transformer and the uniform prior; the n-gram (--lm-corpus) is "
                       "not ported to it. Use --lm-transformer-checkpoint, or neither for the uniform prior")
    if lm_corpus is not None and not Path(lm_corpus).is_file():
        raise FitError(f"--lm-corpus not found: {lm_corpus}")
    if lm_transformer_checkpoint is not None and not Path(lm_transformer_checkpoint).is_file():
        raise FitError(f"--lm-transformer-checkpoint not found: {lm_transformer_checkpoint}")

    recs = ([load_bci2000_recording(p, d) for p, d in zip(paths, dats)] if dats else [load_recording(recording)])
    spec = EpochSpec()
    Xs, flashes, group_ids, reasons_all = [], [], [], []
    for run, rec in enumerate(recs):
        emit(f"loaded {rec.path.name}: {rec.n_rows} rows, {len(rec.timestamps)} EEG samples, "
             f"{rec.n_non_eeg_rows} non-EEG rows, {len(rec.flashes)} flash markers"
             + (f", {rec.n_malformed_markers} malformed markers" if rec.n_malformed_markers else "")
             + (f", {rec.n_duplicate_flashes} duplicate flashes ignored" if rec.n_duplicate_flashes else ""))
        if len(rec.timestamps) == 0 or not rec.flashes:
            raise FitError(f"recording {rec.path}: no EEG samples or no flash markers")
        finite_rows = np.all(np.isfinite(rec.data), axis=1)
        if not finite_rows.any():
            raise FitError(f"recording {rec.path}: no valid EEG samples")
        median_abs = float(np.median(np.abs(rec.data[finite_rows])))
        if median_abs < MIN_PLAUSIBLE_MEDIAN_ABS_UV:
            raise FitError(f"recording {rec.path}: median |sample| is {median_abs:.3g}; that is volts, not "
                           "microvolts (CLAUDE.md gotcha 1)")

        # Each run is epoched on its own: no window reaches across the end of one file into the next.
        X, kept, reasons = extract_epochs(rec.timestamps, rec.data, [f.onset for f in rec.flashes], spec)
        # Linearly dependent channels make the covariance singular: no score exists, so neither fit nor serve.
        usable = np.array([not is_rank_deficient(e) for e in X], dtype=bool)
        if not usable.all():
            idx = np.flatnonzero(kept)
            for j in np.flatnonzero(~usable):
                kept[idx[j]] = False
                reasons[idx[j]] = RANK_DEFICIENT
            X = X[usable]
        # In a .dat the only non-finite samples are the ones the source filled in for lost packets.
        rename = {"non_finite": "lost_samples"} if rec.kind == "bci2000" else {}
        reasons_all += [rename.get(r, r) for r in reasons]
        run_flashes = [f for f, k in zip(rec.flashes, kept) if k]
        Xs.append(X)
        flashes += run_flashes
        group_ids += [run * GROUP_STRIDE + f.trial_id for f in run_flashes]
    X = np.concatenate(Xs) if len(Xs) > 1 else Xs[0]
    n_flashes = sum(len(r.flashes) for r in recs)
    rejected = dict(Counter(r for r in reasons_all if r is not None))
    y = np.array([int(f.is_target) for f in flashes], dtype=int)
    groups = np.array(group_ids, dtype=int)
    n_trials, n_targets = len(np.unique(groups)), int(y.sum())
    emit(f"epochs: {len(flashes)} used of {n_flashes} flashes"
         + (f" from {len(recs)} runs" if len(recs) > 1 else "") + f"; rejected {rejected or 'none'}")
    # Provenance: one recording's sha256, or for several runs the sha256 of their sha256s in order.
    source_sha = (recs[0].sha256 if len(recs) == 1
                  else hashlib.sha256("\n".join(r.sha256 for r in recs).encode()).hexdigest())
    source_name = ", ".join(r.path.name for r in recs)
    if n_trials < MIN_TRIALS or n_targets < MIN_TARGET_EPOCHS or len(y) - n_targets < MIN_NONTARGET_EPOCHS:
        raise FitError(
            f"too little usable data: {n_trials} trials, {n_targets} target and {len(y) - n_targets} "
            f"non-target epochs (need >= {MIN_TRIALS} trials, >= {MIN_TARGET_EPOCHS} target and "
            f">= {MIN_NONTARGET_EPOCHS} non-target epochs); rejected {rejected or 'none'}")

    build_pipeline = load_build_pipeline()
    k = min(n_splits, n_trials)
    fold_log: list = []
    oof = out_of_fold_scores(X, y, groups, k, build_pipeline, fold_log)
    auc = float(roc_auc_score(y, oof))
    calibrator = PlattCalibrator("<out-of-fold decision scores>")
    a, b = calibrator.fit(oof, y)
    ece = expected_calibration_error(calibrator.predict_proba(oof), y)
    emit(f"out-of-fold AUC {auc:.3f} (GroupKFold by trial_id, {k} folds); Platt a={a:.4f} b={b:.4f}; ECE {ece:.4f}")

    final = build_pipeline()
    final.fit(X, y)
    save_checkpoint(ckpt_path, final, spec, {"recording_sha256": source_sha, "n_epochs": int(len(y)),
                                             "n_targets": n_targets, "n_trials": int(n_trials)})
    _write_json_atomic({
        "format": CALIBRATION_FORMAT, "a": float(a), "b": float(b), "score_method": SCORE_METHOD,
        "classifier_sha256": sha256_file(ckpt_path), "ece": float(ece), "n_scores": int(len(oof)),
        "n_targets": n_targets, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_recording_sha256": source_sha,
    }, cal_path)
    if writes_prm:
        # Before the manifest: a failed export must not leave a new manifest beside an old model file.
        try:
            if writes_lm_bin and Path(lm_transformer_checkpoint).suffix.lower() == ".lmbin":
                # Already exported (the Windows package ships one): check it fits this matrix, then copy
                from stopping.thai_gpt_numpy import LmbinError, ThaiGPTNumpy
                try:
                    ThaiGPTNumpy(lm_transformer_checkpoint, load_characters_file(characters_path))
                except LmbinError as exc:
                    raise bci2000_lm_export.LMExportError(str(exc)) from exc
                shutil.copyfile(lm_transformer_checkpoint, lm_bin_path)
                emit(f"BCI2000 language model: {lm_bin_path} (copied from {Path(lm_transformer_checkpoint).name})")
            elif writes_lm_bin:
                bci2000_lm_export.export_lm(lm_transformer_checkpoint, lm_bin_path, characters_path)
                emit(f"BCI2000 language model: {lm_bin_path}")
            elif lm_bin_path.exists():
                lm_bin_path.unlink()      # --force refit with the uniform prior: the old LM is not this model's
            bci2000_export.export_prm(ckpt_path, cal_path, prm_path, {
                "recording": source_name, "recording_sha256": source_sha, "subject_id": subject_id,
                "demo": demo, "n_epochs": int(len(y)), "n_targets": n_targets, "out_of_fold_auc": round(auc, 4)},
                extra_lines=bci2000_export.stopping_lines(DEFAULT_STOPPING, lm_bin_path if writes_lm_bin else None))
        except (bci2000_export.ExportError, bci2000_lm_export.LMExportError) as exc:
            raise FitError(f"BCI2000 export: {exc}") from exc
        emit(f"BCI2000 classifier parameters: {prm_path}")
    elif prm_path.exists() or lm_bin_path.exists():
        for stale in (prm_path, lm_bin_path):   # --force refit from a CSV: the old BCI2000 model is no longer this subject's
            if stale.exists():
                stale.unlink()
        emit(f"removed the previous BCI2000 model {prm_path} (this fit is from a CSV recording)")
    if lm_corpus is not None:
        lm = {"type": "ngram", "corpus_path": Path(lm_corpus)}
    elif lm_transformer_checkpoint is not None:
        lm = {"type": "transformer", "checkpoint_path": Path(lm_transformer_checkpoint)}
    else:
        lm = None
    man_abs = man_path.resolve()
    manifest = write_manifest(build_manifest_dict(
        base_dir=man_abs.parent, characters_path=Path(characters_path), classifier_path=ckpt_path,
        calibration_path=cal_path, language_model=lm), man_abs)

    recording_report = [{
        "path": r.path.as_posix(), "sha256": r.sha256, "n_rows": r.n_rows,
        "n_eeg_samples": int(len(r.timestamps)), "n_non_eeg_rows": r.n_non_eeg_rows,
        "n_flash_markers": len(r.flashes), "n_malformed_markers": r.n_malformed_markers,
        "n_duplicate_flashes": r.n_duplicate_flashes,
        "integrity": xlog.read_json_if_exists(xlog.integrity_path(r.path)),
        "kind": r.kind,
        "bci2000": r.bci2000,
    } for r in recs]
    # The artifacts are written by now, so a fit report that can't be written is a warning, not a failure.
    try:
        xlog.write_json_atomic({
            "format": xlog.FIT_REPORT_FORMAT,
            "created_utc": xlog.utc_now_iso(),
            "duration_s": round(time.monotonic() - t_start, 2),
            "subject_id": subject_id,
            "demo": demo,
            # The first (or only) recording; "recordings" lists every run of a multi-run calibration.
            "recording": recording_report[0],
            "recordings": recording_report,
            # <recording>.session.json: the calibration sentence and settings; None for a recording
            # made before session records existed, or moved away from its sidecars.
            "calibration_session": xlog.read_json_if_exists(xlog.session_record_path(recording)),
            "epochs": {
                "epoch_spec": spec.to_dict(), "n_used": int(len(y)), "n_targets": n_targets,
                "n_non_targets": int(len(y)) - n_targets, "n_trials": int(n_trials), "rejected": rejected,
            },
            "cross_validation": {"scheme": "GroupKFold grouped by trial_id", "n_splits": k,
                                 "folds": fold_log, "out_of_fold_auc": auc},
            "platt": {"a": float(a), "b": float(b), "ece": float(ece), "n_scores": int(len(oof)),
                      "fitted_on": "out-of-fold decision scores"},
            "artifacts": {
                "classifier": {"path": Path(ckpt_path).as_posix(), "sha256": sha256_file(ckpt_path)},
                "calibration": {"path": Path(cal_path).as_posix(), "sha256": sha256_file(cal_path)},
                "manifest": {"path": manifest.path.as_posix(), "fingerprint": manifest.fingerprint},
                "bci2000_parameters": ({"path": Path(prm_path).as_posix(), "sha256": sha256_file(prm_path)}
                                       if writes_prm else None),
            },
            "language_model": manifest.lm_config,
            "versions": library_versions(),
            "warnings": list(warnings),
            "console": console,
        }, report_path)
    except OSError as exc:
        warnings.append(f"fit report not written: {exc}")
        report_path = None

    return FitResult(demo=demo, n_flashes=n_flashes, n_epochs=int(len(y)), n_targets=n_targets,
                     n_trials=int(n_trials), rejected=rejected, n_splits=k, auc=auc, a=float(a), b=float(b),
                     ece=float(ece), checkpoint_path=ckpt_path, calibration_path=cal_path,
                     manifest_path=manifest.path, fingerprint=manifest.fingerprint, warnings=warnings,
                     report_path=report_path, bci2000_prm_path=prm_path if writes_prm else None)


def format_summary(result: FitResult) -> str:
    lines = [
        "SYNTHETIC (demo) MODEL -- not for real sessions" if result.demo else "fit complete",
        f"  epochs used      {result.n_epochs} of {result.n_flashes} flashes "
        f"({result.n_targets} targets, {result.n_trials} trials)",
        f"  epochs rejected  {result.rejected or 'none'}",
        f"  out-of-fold AUC  {result.auc:.3f} ({result.n_splits} folds grouped by trial_id)",
        f"  Platt            a={result.a:.4f}  b={result.b:.4f}  ECE={result.ece:.4f}",
        f"  checkpoint       {result.checkpoint_path}",
        f"  calibration      {result.calibration_path}",
        f"  manifest         {result.manifest_path}",
        f"  fingerprint      {result.fingerprint}",
        f"  fit report       {result.report_path}",
    ]
    if result.bci2000_prm_path is not None:
        lines.append(f"  BCI2000 model    {result.bci2000_prm_path}  (load in Config with CheckerboardSignalProcessing)")
    lines += [f"  warning          {w}" for w in result.warnings]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit classifier + Platt calibration from a recording "
                                                 "and write the session manifest.")
    parser.add_argument("--recording", required=True, nargs="+",
                        help="recorder CSV, or one or more BCI2000 .dat runs from CheckerboardSpeller (one "
                             "calibration split into runs: same subject, montage and matrix)")
    parser.add_argument("--id", required=True, dest="subject_id", help="subject id used in artifact names")
    parser.add_argument("--models-dir", default="models",
                        help="checkpoint and calibration directory; a relative path resolves against the repo root")
    parser.add_argument("--manifest", default=None,
                        help=f"manifest to write (default <repo root>/{DEFAULT_MANIFEST_NAME}, demo_-prefixed for "
                             "demo data, which is where the GUI looks)")
    parser.add_argument("--lm-corpus", default=None, help="UTF-8 Thai corpus; selects the ngram LM (default uniform)")
    parser.add_argument("--lm-transformer-checkpoint", default=None,
                        help="gpt_dev.ipynb checkpoint (.pt), or its BCI2000 export (.lmbin, needs no torch); "
                             "selects the transformer LM (default uniform); mutually exclusive with --lm-corpus")
    parser.add_argument("--allow-demo", action="store_true", help="allow a demo_ recording; prefixes every artifact")
    parser.add_argument("--force", action="store_true", help="overwrite existing artifacts")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--characters", default=str(DEFAULT_CHARACTERS_PATH), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        result = fit_from_recording(args.recording, args.subject_id, models_dir=args.models_dir,
                                    manifest_path=args.manifest, lm_corpus=args.lm_corpus,
                                    lm_transformer_checkpoint=args.lm_transformer_checkpoint,
                                    allow_demo=args.allow_demo, force=args.force, n_splits=args.n_splits,
                                    characters_path=args.characters)
    except (FitError, ManifestError, CheckpointError, LanguageModelError, bci2000_export.ExportError) as exc:
        print(f"fit_from_recording: error: {exc}", file=sys.stderr)
        return 2
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
