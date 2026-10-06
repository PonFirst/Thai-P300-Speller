"""
Evaluate the Bayesian stopping controller against a REAL recorded session:
real classifier scores (from a fitted checkpoint), real per-flash groups,
real target characters -- producing accuracy / mean-flashes / Wolpaw ITR
across a swept hyperparameter grid. This is the real-data counterpart to
stopping/validate_offline.py's fabricated-grid sanity check.

    python -m stopping.evaluate_recording --manifest session_manifest.json \\
        --recording collection/recordings/<held-out-session>.csv \\
        [--theta 0.7 0.8 0.9 0.95] [--t-min 1 3 6] [--t-max 120] [--lam 0.7] \\
        [--out results/replay/<name>.csv] [--allow-demo]

The recording may also be a BCI2000 .dat from CheckerboardSpeller (load_bci2000_for_replay):
flashes, groups and onsets come from its states (pipeline/bci2000_recording.py), each selection's
target from its cue (AttendedChar), and samples lost to the Bluetooth link are NaN, so their epochs
are rejected exactly as the BCI2000 classifier rejected them online. Data from anything but the
real headset counts as demo data (pipeline/fit_from_recording.dat_synthetic_reason).

Recommended use: replay a session that was NOT used to fit the checkpoint
(a held-out test/copy-spelling session), so the reported numbers describe
generalization, not the fit. Nothing here enforces that -- the recording
and the manifest's classifier are independent inputs -- so a self-check
against the calibration recording itself is possible but will read
optimistic; label it as such if you report it.

Pipeline:
  1. Load the manifest's components (checkpoint, calibration, language
     model, gate policy) via pipeline.inference_service's own loader, so
     replay is built from exactly the same pieces live inference would use.
  2. Parse the recording CSV directly (not via pipeline.fit_from_recording,
     which discards each flash's group and each trial's target -- exactly
     the two fields replay needs).
  3. Cut every flash's epoch with pipeline.epoching (the same EpochSpec the
     checkpoint was fit with -- train/serve/replay parity), quality-gate it
     exactly as the live service would, and score accepted epochs with the
     checkpoint's own pipeline.decision_function.
  4. Group flashes into per-trial (score, group_id) sequences in their
     original within-session order, resolving each trial's true target from
     its trial_start marker.
  5. Assign each trial's language-model prior from the TRUE preceding
     characters (ground-truth copy-spelling context), not from what the
     replayed controller predicted -- this isolates the stopping
     controller's own behaviour from classifier-error propagation through
     the LM. Note this choice when citing the numbers; replaying against
     the controller's own running predictions instead would be a different,
     also-valid experiment (error propagation across a session) that this
     script does not run.
  6. Sweep the requested hyperparameter grid through
     stopping.replay.run_sweep() (real BayesianStoppingController calls,
     not a mock) and report accuracy / mean flashes-to-stop / Wolpaw ITR
     per configuration.

seconds_per_flash for the ITR conversion is measured empirically from the
recording's own flash timestamps (median within-trial inter-flash gap), not
assumed from the paradigm's nominal 124 ms -- so it reflects what actually
happened in this session. Override with --seconds-per-flash to force a
value (e.g. to compare against a different nominal schedule).

Exit codes: 0 success, 2 refused or failed (one line on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.classifier_io import CheckpointMismatchError, check_stream_compatibility, score_epoch
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ
from pipeline.epoching import extract_epochs, is_rank_deficient
from pipeline.inference_service import load_service_components
from pipeline.manifest import ManifestError, sha256_file
from pipeline.quality import gate_epoch
from stopping.bayesian_stopping import BayesianStoppingController
from stopping.replay import run_sweep

DEMO_PREFIX = "demo_"
REQUIRED_COLUMNS = ["timestamp", *EEG_CHANNELS, "marker_json_or_empty"]


class EvaluationError(Exception):
    """The recording or the requested evaluation cannot be run."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ReplayFlash:
    trial_id: int
    flash_idx: int
    onset: float
    is_target: bool
    group: tuple


@dataclass
class RecordingForReplay:
    path: Path
    sha256: str
    timestamps: np.ndarray
    data: np.ndarray
    flashes: list          # ReplayFlash, in file order
    targets: dict          # trial_id -> target_char_idx
    n_rows: int
    n_malformed_markers: int = 0
    n_duplicate_flashes: int = 0


def load_bci2000_for_replay(path, n_chars: int) -> RecordingForReplay:
    """A CheckerboardSpeller .dat as replay input: trials are selections, targets their cues."""
    from pipeline.bci2000_dat import DatError, read_dat
    from pipeline.bci2000_recording import RecordingError, eeg_microvolts, flash_events

    path = Path(path)
    try:
        dat = read_dat(path)
        flashes = flash_events(dat)
    except (DatError, RecordingError) as exc:
        raise EvaluationError(str(exc)) from exc
    if any(c >= n_chars for f in flashes for c in (*f.group, f.attended)):
        raise EvaluationError(f"recording {path.name}: character indices beyond the manifest's {n_chars} characters")
    replay = [ReplayFlash(trial_id=f.selection, flash_idx=f.code, onset=f.onset / EEG_SFREQ,
                          is_target=f.is_target, group=tuple(f.group)) for f in flashes]
    targets = {f.selection: f.attended for f in flashes}
    return RecordingForReplay(path=path, sha256=sha256_file(path), timestamps=np.arange(dat.n_samples) / EEG_SFREQ,
                              data=eeg_microvolts(dat), flashes=replay, targets=targets, n_rows=dat.n_samples)


def load_recording_for_replay(path, n_chars: int) -> RecordingForReplay:
    """
    Parse a recorder CSV for replay: unlike pipeline.fit_from_recording's
    loader, this keeps each flash's group and each trial's true target --
    the two fields replay needs and training doesn't. A BCI2000 .dat goes to
    load_bci2000_for_replay.
    """
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"recording not found: {path}")
    if path.suffix.lower() == ".dat":
        return load_bci2000_for_replay(path, n_chars)
    try:
        df = pd.read_csv(path, dtype={"marker_json_or_empty": str}, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise EvaluationError(f"recording {path}: cannot parse CSV ({exc})") from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise EvaluationError(f"recording {path}: missing column(s) {missing}")

    ts_all = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    data_all = np.column_stack([pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=float)
                                for ch in EEG_CHANNELS])
    is_eeg = np.isfinite(ts_all) & np.all(np.isfinite(data_all), axis=1)

    flashes, targets, seen = [], {}, set()
    n_malformed = n_duplicate = 0
    markers = df["marker_json_or_empty"].fillna("")
    for row in np.flatnonzero(markers.str.strip().to_numpy() != ""):
        try:
            event = json.loads(markers.iat[row])
        except json.JSONDecodeError:
            n_malformed += 1
            continue
        if not isinstance(event, dict):
            n_malformed += 1
            continue
        kind = event.get("event")
        onset = ts_all[row]

        if kind == "trial_start":
            trial_id, target_char_idx = event.get("trial_id"), event.get("target_char_idx")
            if _is_int(trial_id) and _is_int(target_char_idx) and 0 <= target_char_idx < n_chars:
                targets[trial_id] = target_char_idx
            else:
                n_malformed += 1
            continue

        if kind != "flash":
            continue
        trial_id, flash_idx, is_target = event.get("trial_id"), event.get("flash_idx"), event.get("is_target")
        group = event.get("group")
        valid_group = (isinstance(group, list) and len(group) > 0 and len(set(group)) == len(group)
                       and all(_is_int(g) and 0 <= g < n_chars for g in group))
        if not (_is_int(trial_id) and _is_int(flash_idx) and isinstance(is_target, bool)
                and valid_group and np.isfinite(onset)):
            n_malformed += 1
            continue
        if (trial_id, flash_idx) in seen:
            n_duplicate += 1
            continue
        seen.add((trial_id, flash_idx))
        flashes.append(ReplayFlash(trial_id, flash_idx, float(onset), is_target, tuple(group)))

    return RecordingForReplay(path=path, sha256=sha256_file(path), timestamps=ts_all[is_eeg],
                              data=data_all[is_eeg], flashes=flashes, targets=targets,
                              n_rows=len(df), n_malformed_markers=n_malformed,
                              n_duplicate_flashes=n_duplicate)


def build_group_table(flashes: list) -> tuple[list, list]:
    """
    Assign each distinct flash-group composition a stable index, in order
    of first appearance. Two flashes with the same characters (regardless
    of list order) are the same group -- order within a group never
    matters to the Bayesian update, only membership.

    Returns (groups: list[list[int]], group_ids: list[int] aligned with
    `flashes`).
    """
    groups, index_of, group_ids = [], {}, []
    for f in flashes:
        key = tuple(sorted(f.group))
        idx = index_of.get(key)
        if idx is None:
            idx = len(groups)
            index_of[key] = idx
            groups.append(list(f.group))
        group_ids.append(idx)
    return groups, group_ids


def empirical_seconds_per_flash(flashes_by_trial: dict) -> float:
    """
    Median within-trial gap between consecutive flash onsets, across every
    trial that has at least two flashes -- the real observed flash cadence
    of this session, not the paradigm's nominal timing.
    """
    gaps = []
    for trial_flashes in flashes_by_trial.values():
        onsets = sorted(f.onset for f in trial_flashes)
        gaps.extend(b - a for a, b in zip(onsets, onsets[1:]) if b > a)
    if not gaps:
        raise EvaluationError("no within-trial flash-to-flash gaps found; pass --seconds-per-flash explicitly")
    return float(np.median(gaps))


@dataclass
class AssembledTrials:
    trials: list                      # dicts ready for stopping.replay.run_sweep
    groups: list
    n_trials_used: int
    n_trials_excluded_no_target: int
    n_trials_excluded_no_accepted_flashes: int
    n_flashes_total: int
    n_flashes_epoch_rejected: int
    n_flashes_quality_rejected: int
    n_flashes_scored: int
    seconds_per_flash: float
    n_flashes_rank_deficient: int = 0     # linearly dependent channels: no score exists


def assemble_trials(rec: RecordingForReplay, *, epoch_spec, gate_policy: str, characters: list,
                    language_model, checkpoint, seconds_per_flash: float | None = None,
                    log=print) -> AssembledTrials:
    groups, group_ids = build_group_table(rec.flashes)

    onsets = [f.onset for f in rec.flashes]
    X, kept, reasons = extract_epochs(rec.timestamps, rec.data, onsets, epoch_spec)
    n_epoch_rejected = int((~kept).sum())

    scored = {}   # flash list-index -> score, for kept flashes that also pass the quality gate
    n_quality_rejected = 0
    n_rank_deficient = 0
    epoch_i = 0
    for i, keep in enumerate(kept):
        if not keep:
            continue
        epoch = X[epoch_i]
        epoch_i += 1
        decision = gate_epoch(epoch, epoch_spec.channels, gate_policy)
        if not decision.accept:
            n_quality_rejected += 1
            continue
        if is_rank_deficient(epoch):        # as the service: a singular covariance has no score
            n_rank_deficient += 1
            continue
        scored[i] = float(score_epoch(checkpoint, epoch))

    # (flash, its position in rec.flashes/group_ids) per trial, so lookups below are O(1),
    # not a linear rec.flashes.index(f) per flash.
    flashes_by_trial = {}
    for i, f in enumerate(rec.flashes):
        flashes_by_trial.setdefault(f.trial_id, []).append((f, i))

    if seconds_per_flash is None:
        seconds_per_flash = empirical_seconds_per_flash(
            {tid: [f for f, _ in items] for tid, items in flashes_by_trial.items()})

    trials, no_target, no_flashes = [], 0, 0
    committed_text = ""
    for trial_id in sorted(flashes_by_trial):
        if trial_id not in rec.targets:
            no_target += 1
            log(f"trial {trial_id}: no trial_start / target_char_idx recorded; excluded")
            continue
        trial_items = sorted(flashes_by_trial[trial_id], key=lambda item: item[0].flash_idx)
        scores, g_ids = [], []
        for _f, i in trial_items:
            if i in scored:
                scores.append(scored[i])
                g_ids.append(group_ids[i])
        if not scores:
            no_flashes += 1
            log(f"trial {trial_id}: every flash rejected (epoch or quality gate); excluded")
            continue
        target_char_idx = rec.targets[trial_id]
        prior = language_model.get_prior(committed_text)
        trials.append({"scores": scores, "group_ids": g_ids, "true_char_idx": target_char_idx,
                       "lm_prior": prior})
        committed_text += characters[target_char_idx]

    return AssembledTrials(
        trials=trials, groups=groups, n_trials_used=len(trials),
        n_trials_excluded_no_target=no_target, n_trials_excluded_no_accepted_flashes=no_flashes,
        n_flashes_total=len(rec.flashes), n_flashes_epoch_rejected=n_epoch_rejected,
        n_flashes_quality_rejected=n_quality_rejected, n_flashes_scored=len(scored),
        n_flashes_rank_deficient=n_rank_deficient,
        seconds_per_flash=seconds_per_flash)


def run_evaluation(manifest_path, recording_path, *, theta=None, t_min=None, t_max=None, lam=None,
                   llr_clip=None, gate_policy=None, seconds_per_flash=None, allow_demo=False,
                   log=print) -> pd.DataFrame:
    why = "demo_ prefix" if Path(recording_path).name.startswith(DEMO_PREFIX) else None
    if why is None and Path(recording_path).suffix.lower() == ".dat" and Path(recording_path).is_file():
        from pipeline.bci2000_dat import DatError, read_dat
        from pipeline.fit_from_recording import dat_synthetic_reason
        try:
            why = dat_synthetic_reason(read_dat(recording_path))
        except DatError as exc:
            raise EvaluationError(str(exc)) from exc
    demo = why is not None
    if demo and not allow_demo:
        raise EvaluationError(
            f"recording {Path(recording_path).name} is synthetic ({why}); refusing to evaluate "
            "against it. Pass --allow-demo to proceed anyway; results are not a real measurement.")
    if demo:
        log("*** DEMO recording: results below describe synthetic data, not real performance. ***")

    comps = load_service_components(manifest_path)
    epoch_spec = comps.epoch_spec
    st = comps.manifest.stopping
    theta = list(theta) if theta is not None else [st["theta"]]
    t_min = list(t_min) if t_min is not None else [st["t_min"]]
    t_max = list(t_max) if t_max is not None else [st["t_max"]]
    lam = list(lam) if lam is not None else [st["lam"]]
    llr_clip = list(llr_clip) if llr_clip is not None else [st["llr_clip"]]

    rec = load_recording_for_replay(recording_path, len(comps.characters))
    log(f"loaded {rec.path.name}: {rec.n_rows} rows, {len(rec.timestamps)} EEG samples, "
        f"{len(rec.flashes)} flash markers, {len(rec.targets)} trial targets"
        + (f", {rec.n_malformed_markers} malformed markers" if rec.n_malformed_markers else "")
        + (f", {rec.n_duplicate_flashes} duplicate flashes ignored" if rec.n_duplicate_flashes else ""))
    if not rec.flashes or not rec.targets:
        raise EvaluationError(f"recording {rec.path}: no usable flash markers or trial targets")

    try:
        check_stream_compatibility(comps.checkpoint, list(EEG_CHANNELS), EEG_SFREQ)
    except CheckpointMismatchError as exc:
        raise EvaluationError(f"checkpoint does not match this recording's montage: {exc}") from exc

    policy = gate_policy or comps.gate_policy
    assembled = assemble_trials(rec, epoch_spec=epoch_spec, gate_policy=policy,
                                characters=comps.characters, language_model=comps.language_model,
                                checkpoint=comps.checkpoint, seconds_per_flash=seconds_per_flash, log=log)
    log(f"trials: {assembled.n_trials_used} used, {assembled.n_trials_excluded_no_target} excluded "
        f"(no target), {assembled.n_trials_excluded_no_accepted_flashes} excluded (no accepted flashes)")
    log(f"flashes: {assembled.n_flashes_total} total, {assembled.n_flashes_epoch_rejected} epoch-rejected, "
        f"{assembled.n_flashes_quality_rejected} quality-gate-rejected, "
        f"{assembled.n_flashes_rank_deficient} rank-deficient, {assembled.n_flashes_scored} scored")
    log(f"seconds/flash: {assembled.seconds_per_flash:.4f} (empirical, gate policy {policy!r})")
    if assembled.n_trials_used < 10:
        log(f"*** only {assembled.n_trials_used} usable trials; accuracy/ITR estimates will be noisy. ***")

    param_grid = {
        "theta": theta, "t_min": t_min, "t_max": t_max, "lam": lam, "llr_clip": llr_clip,
        "sigmoid_a": [comps.calibration["a"]], "sigmoid_b": [comps.calibration["b"]],
    }
    df = run_sweep(assembled.trials, assembled.groups, param_grid,
                   seconds_per_flash=assembled.seconds_per_flash, controller_cls=BayesianStoppingController)
    return df.sort_values("itr_bits_per_min", ascending=False).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamic stopping against a real recorded session: accuracy, mean "
                    "flashes-to-stop, and Wolpaw ITR across a swept hyperparameter grid.")
    parser.add_argument("--manifest", required=True, help="session_manifest.json (checkpoint + calibration + LM)")
    parser.add_argument("--recording", required=True, help="recorder CSV to replay -- ideally held out from fitting")
    parser.add_argument("--theta", type=float, nargs="+", default=None,
                        help="posterior-confidence threshold(s) to sweep (default: the manifest's own theta)")
    parser.add_argument("--t-min", type=int, nargs="+", default=None,
                        help="minimum-flashes-before-stopping value(s) to sweep (default: the manifest's own t_min)")
    parser.add_argument("--t-max", type=int, nargs="+", default=None, help="override t_max (default: manifest's)")
    parser.add_argument("--lam", type=float, nargs="+", default=None, help="override lam (default: manifest's)")
    parser.add_argument("--llr-clip", type=float, nargs="+", default=None, help="override llr_clip (default: manifest's)")
    parser.add_argument("--gate-policy", choices=["skip", "warn"], default=None,
                        help="override the manifest's quality-gate policy for this replay")
    parser.add_argument("--seconds-per-flash", type=float, default=None,
                        help="override the empirically-measured flash cadence")
    parser.add_argument("--allow-demo", action="store_true", help="allow a demo_ recording (results are not real)")
    parser.add_argument("--out", default=None, help="CSV path to save the results table to")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        df = run_evaluation(
            args.manifest, args.recording, theta=args.theta, t_min=args.t_min, t_max=args.t_max,
            lam=args.lam, llr_clip=args.llr_clip, gate_policy=args.gate_policy,
            seconds_per_flash=args.seconds_per_flash, allow_demo=args.allow_demo)
    except (EvaluationError, ManifestError) as exc:
        print(f"evaluate_recording: error: {exc}", file=sys.stderr)
        return 2

    with pd.option_context("display.width", 120):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
