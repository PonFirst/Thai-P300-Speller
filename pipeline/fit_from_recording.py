"""
Fit the classifier and its Platt calibration from a recorded calibration
session, and write the session manifest that every process loads.

    python -m pipeline.fit_from_recording --recording collection/recordings/subject_001.csv \\
        --id subject_001 [--models-dir models] [--manifest session_manifest.json] \\
        [--lm-corpus corpus.txt | --lm-transformer-checkpoint checkpoint.pt] \\
        [--allow-demo] [--force] [--n-splits 5] [--characters PATH]

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
import json
import os
import re
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

from pipeline import experiment_log as xlog
from pipeline.classifier_io import CheckpointError, library_versions, load_build_pipeline, save_checkpoint
from pipeline.constants import EEG_CHANNELS
from pipeline.epoching import EpochSpec, extract_epochs
from pipeline.language_model import LanguageModelError
from pipeline.manifest import (CALIBRATION_FORMAT, REPO_ROOT, SCORE_METHOD, ManifestError,
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


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_recording(path) -> Recording:
    path = Path(path)
    if not path.is_file():
        raise FitError(f"recording not found: {path}")
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
                 allow_demo: bool = False) -> tuple[bool, Path, Path, Path]:
    """(is_demo, checkpoint, calibration, manifest). Raises FitError for a refused demo recording."""
    if not _SUBJECT_ID_RE.match(subject_id or ""):
        raise FitError(f"--id {subject_id!r} must be letters, digits, '_', '.', '-' (a safe filename)")
    demo = Path(recording_path).name.startswith(DEMO_PREFIX)
    if demo and not allow_demo:
        raise FitError(
            f"recording {Path(recording_path).name} is synthetic (demo_ prefix); refusing to fit a "
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


def fit_from_recording(recording, subject_id: str, *, models_dir="models", manifest_path=None,
                       lm_corpus=None, lm_transformer_checkpoint=None, allow_demo: bool = False,
                       force: bool = False, n_splits: int = 5, characters_path=DEFAULT_CHARACTERS_PATH,
                       log: Callable[[str], None] = print) -> FitResult:
    recording = Path(recording)
    demo, ckpt_path, cal_path, man_path = output_paths(recording, subject_id, models_dir,
                                                       manifest_path, allow_demo)
    warnings = []
    console: list[str] = []  # everything logged below, kept for the fit report
    t_start = time.monotonic()

    def emit(line: str) -> None:
        console.append(line)
        log(line)

    report_path = xlog.fit_report_path(ckpt_path)
    if demo and manifest_path is not None and not man_path.name.startswith(DEMO_PREFIX):
        warnings.append(f"demo recording, but --manifest {man_path} is not demo_-prefixed")
    existing = [str(p) for p in (ckpt_path, cal_path, man_path, report_path) if p.exists()]
    if existing and not force:
        raise FitError(f"refusing to overwrite existing artifact(s) {existing}; pass --force to replace them")
    if n_splits < 2:
        raise FitError(f"--n-splits must be >= 2, got {n_splits}")
    load_characters_file(characters_path)
    if lm_corpus is not None and lm_transformer_checkpoint is not None:
        raise FitError("--lm-corpus and --lm-transformer-checkpoint are mutually exclusive; pick one LM")
    if lm_corpus is not None and not Path(lm_corpus).is_file():
        raise FitError(f"--lm-corpus not found: {lm_corpus}")
    if lm_transformer_checkpoint is not None and not Path(lm_transformer_checkpoint).is_file():
        raise FitError(f"--lm-transformer-checkpoint not found: {lm_transformer_checkpoint}")

    rec = load_recording(recording)
    emit(f"loaded {rec.path.name}: {rec.n_rows} rows, {len(rec.timestamps)} EEG samples, "
         f"{rec.n_non_eeg_rows} non-EEG rows, {len(rec.flashes)} flash markers"
         + (f", {rec.n_malformed_markers} malformed markers" if rec.n_malformed_markers else "")
         + (f", {rec.n_duplicate_flashes} duplicate flashes ignored" if rec.n_duplicate_flashes else ""))
    if len(rec.timestamps) == 0 or not rec.flashes:
        raise FitError(f"recording {rec.path}: no EEG samples or no flash markers")
    median_abs = float(np.median(np.abs(rec.data)))
    if median_abs < MIN_PLAUSIBLE_MEDIAN_ABS_UV:
        raise FitError(f"recording {rec.path}: median |sample| is {median_abs:.3g}; that is volts, not "
                       "microvolts (CLAUDE.md gotcha 1)")

    spec = EpochSpec()
    X, kept, reasons = extract_epochs(rec.timestamps, rec.data, [f.onset for f in rec.flashes], spec)
    rejected = dict(Counter(r for r in reasons if r is not None))
    flashes = [f for f, k in zip(rec.flashes, kept) if k]
    y = np.array([int(f.is_target) for f in flashes], dtype=int)
    groups = np.array([f.trial_id for f in flashes], dtype=int)
    n_trials, n_targets = len(np.unique(groups)), int(y.sum())
    emit(f"epochs: {len(flashes)} used of {len(rec.flashes)} flashes; rejected {rejected or 'none'}")
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
    save_checkpoint(ckpt_path, final, spec, {"recording_sha256": rec.sha256, "n_epochs": int(len(y)),
                                             "n_targets": n_targets, "n_trials": int(n_trials)})
    _write_json_atomic({
        "format": CALIBRATION_FORMAT, "a": float(a), "b": float(b), "score_method": SCORE_METHOD,
        "classifier_sha256": sha256_file(ckpt_path), "ece": float(ece), "n_scores": int(len(oof)),
        "n_targets": n_targets, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_recording_sha256": rec.sha256,
    }, cal_path)
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

    # The artifacts are written by now, so a fit report that can't be written is a warning, not a failure.
    try:
        xlog.write_json_atomic({
            "format": xlog.FIT_REPORT_FORMAT,
            "created_utc": xlog.utc_now_iso(),
            "duration_s": round(time.monotonic() - t_start, 2),
            "subject_id": subject_id,
            "demo": demo,
            "recording": {
                "path": str(rec.path), "sha256": rec.sha256, "n_rows": rec.n_rows,
                "n_eeg_samples": int(len(rec.timestamps)), "n_non_eeg_rows": rec.n_non_eeg_rows,
                "n_flash_markers": len(rec.flashes), "n_malformed_markers": rec.n_malformed_markers,
                "n_duplicate_flashes": rec.n_duplicate_flashes,
                "integrity": xlog.read_json_if_exists(xlog.integrity_path(recording)),
            },
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
                "classifier": {"path": str(ckpt_path), "sha256": sha256_file(ckpt_path)},
                "calibration": {"path": str(cal_path), "sha256": sha256_file(cal_path)},
                "manifest": {"path": str(manifest.path), "fingerprint": manifest.fingerprint},
            },
            "language_model": manifest.lm_config,
            "versions": library_versions(),
            "warnings": list(warnings),
            "console": console,
        }, report_path)
    except OSError as exc:
        warnings.append(f"fit report not written: {exc}")
        report_path = None

    return FitResult(demo=demo, n_flashes=len(rec.flashes), n_epochs=int(len(y)), n_targets=n_targets,
                     n_trials=int(n_trials), rejected=rejected, n_splits=k, auc=auc, a=float(a), b=float(b),
                     ece=float(ece), checkpoint_path=ckpt_path, calibration_path=cal_path,
                     manifest_path=manifest.path, fingerprint=manifest.fingerprint, warnings=warnings,
                     report_path=report_path)


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
    lines += [f"  warning          {w}" for w in result.warnings]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit classifier + Platt calibration from a recording "
                                                 "and write the session manifest.")
    parser.add_argument("--recording", required=True, help="recorder CSV")
    parser.add_argument("--id", required=True, dest="subject_id", help="subject id used in artifact names")
    parser.add_argument("--models-dir", default="models",
                        help="checkpoint and calibration directory; a relative path resolves against the repo root")
    parser.add_argument("--manifest", default=None,
                        help=f"manifest to write (default <repo root>/{DEFAULT_MANIFEST_NAME}, demo_-prefixed for "
                             "demo data, which is where the GUI looks)")
    parser.add_argument("--lm-corpus", default=None, help="UTF-8 Thai corpus; selects the ngram LM (default uniform)")
    parser.add_argument("--lm-transformer-checkpoint", default=None,
                        help="gpt_dev.ipynb checkpoint (.pt); selects the transformer LM (default uniform); "
                             "mutually exclusive with --lm-corpus")
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
    except (FitError, ManifestError, CheckpointError, LanguageModelError) as exc:
        print(f"fit_from_recording: error: {exc}", file=sys.stderr)
        return 2
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
