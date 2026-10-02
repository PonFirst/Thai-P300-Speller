"""
Live P300 inference service: flash epochs -> classifier score -> Bayesian
stopping controller -> committed characters, published on the P300Inference
LSL stream (event contract in pipeline/events.py, docs/INTERFACES.md).

Transport decision: the service consumes the EEG and marker LSL streams
directly, not the recorder's CSV. That is lower latency, and there is no
file-tailing race against a writer that flushes in chunks.

Layers:
  InferenceEngine   pure logic: no LSL, no threads, no clock of its own. Fed
                    EEG chunks, parsed markers, and the current LSL time, it
                    emits event dicts through a callback.
  EegSilenceMonitor pure wall-clock silence detector for the runner.
  ServiceRunner     LSL wiring: loads the manifest, publishes the outlet,
                    resolves EEG and markers, runs the loop.
  main()            CLI.

    python -m pipeline.inference_service --manifest PATH [--eeg-timeout 30]
        [--log-file PATH] [--eeg-type EEG] [--marker-type Markers]
        [--inference-type P300Inference]

Exit codes:
  0  stopped by SIGTERM/SIGINT (service_status "stopped")
  1  unexpected crash (traceback in the log)
  2  configuration: bad arguments; manifest, characters, checkpoint,
     calibration, LM, or controller failed to load; or the checkpoint doesn't
     match the EEG stream (channel count, sfreq, or electrode labels)
  3  no EEG stream found within --eeg-timeout
  4  a BayesianStoppingController method raised NotImplementedError
Every non-zero exit prints one line to stderr.

Status stream: LSL does not replay samples to late subscribers. After
creating the outlet the runner waits up to consumer_wait_s (2 s) for a
consumer before emitting "starting", and once running it re-emits "ready"
every READY_HEARTBEAT_S, paused while the EEG stream is silent. The outlet's
desc() always carries the manifest fingerprint.

EEG silence: no chunk for EEG_SILENCE_WARN_S of wall-clock time emits one
warning; the first chunk afterwards is logged and "ready" is re-emitted. A
queued flash whose epoch should have completed more than STALE_EPOCH_S ago is
rejected as epoch_incomplete, so a trial_end can never wait forever.

Channel labels: generic labels that name no checkpoint electrode ("EEG 1",
"Ch1") produce a warning and a count/sfreq-only check; electrode labels that
differ from the checkpoint are a montage mismatch, exit 2.

Controller boundary: the controller is a black box. Every call goes through
InferenceEngine._call_controller, which turns NotImplementedError into
ControllerNotImplementedError naming the method, trial, and flash. The
service checks what comes out (the posterior must be finite and sum to 1, and
a stop must satisfy one of the two stopping conditions) but never recomputes
it.

Experiment log: besides the operational messages, the --log-file carries one structured
record per event (service_config, then per character trial_start with the LM prior,
flash_update per flash, flash_rejected, char_committed). Schema in docs/INTERFACES.md and
pipeline/experiment_log.py. The cued target, when the stimulus announces one, is read only to
score commits in the log; decoding never uses it.

Timing: flashes are processed in timestamp order, each once its epoch window
(tmax + padding after onset) is in the buffer. A trial_end waits behind its
trial's queued flashes, because it arrives before the last ~1.3 s of epochs
are complete.
"""

from __future__ import annotations

import argparse
import bisect
import functools
import json
import logging
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pylsl
from pylsl.util import LostError

from pipeline import events as ev
from pipeline.classifier_io import (CheckpointMismatchError, channels_for_compatibility_check,
                                    check_stream_compatibility, load_checkpoint, score_epoch,
                                    version_warnings)
from pipeline.constants import (EEG_INLET_DEJITTER, EEG_STREAM_TYPE, INFERENCE_FINGERPRINT_KEY,
                                INFERENCE_STREAM_NAME, INFERENCE_STREAM_TYPE, MARKER_STREAM_TYPE)
from pipeline.epoching import NOT_YET_AVAILABLE, EpochSpec, extract_epoch, is_rank_deficient
from pipeline.experiment_log import sig_floats
from pipeline.language_model import build_language_model, validate_prior
from pipeline.manifest import (load_calibration_file, load_characters_file, load_manifest,
                               short_fingerprint)
from pipeline.quality import GATE_POLICIES, gate_epoch

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_CONFIG = 2
EXIT_NO_EEG = 3
EXIT_CONTROLLER_NOT_IMPLEMENTED = 4



def eeg_inlet_processing_flags(dejitter: bool) -> int:
    """proc_clocksync, plus proc_dejitter only when dejitter is True."""
    return pylsl.proc_clocksync | (pylsl.proc_dejitter if dejitter else 0)


# EEG inlet post-processing follows pipeline.constants.EEG_INLET_DEJITTER (False: dejitter
# can hide non-monotonic steps, and the fit script trains on the recorder's as-delivered
# timestamps, so the service must match the recorder). Markers are clock-synced only.
EEG_INLET_PROCESSING_FLAGS = eeg_inlet_processing_flags(EEG_INLET_DEJITTER)
MARKER_INLET_PROCESSING_FLAGS = pylsl.proc_clocksync

POSTERIOR_SUM_TOLERANCE = 1e-6
DEFAULT_BUFFER_S = 10.0
MARKER_CLOCK_AHEAD_WARN_S = 5.0
READY_HEARTBEAT_S = 2.0
EEG_SILENCE_WARN_S = 2.0
STALE_EPOCH_S = 5.0
ENGINE_CLOCK_INTERVAL_S = 0.1
LOGGER_NAME = "p300.inference"


class ControllerNotImplementedError(RuntimeError):
    """A stopping-controller method is still a NotImplementedError stub."""

    def __init__(self, method: str, trial_id: int, flash_idx: int | None):
        self.method, self.trial_id, self.flash_idx = method, trial_id, flash_idx
        where = (f"trial {trial_id}, flash {flash_idx}" if flash_idx is not None
                 else f"trial {trial_id}, before its first flash")
        super().__init__(
            f"BayesianStoppingController.{method}() raised NotImplementedError at {where}. "
            f"Implement {method}() in stopping/bayesian_stopping.py; the service cannot decode without it.")


# ----------------------------------------------------------------------------
# Pure engine
# ----------------------------------------------------------------------------

@dataclass
class _Marker:
    kind: str
    generation: int
    trial_id: int
    timestamp: float
    flash_idx: int | None = None
    group: tuple = ()
    admitted: bool = False
    # trial_start only. Log-only: the stimulus announces the cued target so the experiment log
    # can score each commit. Decoding never reads it.
    target_char_idx: int | None = None

    @property
    def trial_key(self):
        return (self.generation, self.trial_id)


@dataclass
class _Trial:
    key: tuple
    controller: Any
    flash_idxs: set = field(default_factory=set)
    committed: bool = False
    failed: bool = False
    warned_bad_channels: bool = False
    target_char_idx: int | None = None  # log-only, see _Marker

    @property
    def trial_id(self) -> int:
        return self.key[1]


def _is_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class InferenceEngine:
    """
    Pure decoding state machine; see the module docstring.

    controller_factory: () -> controller; a fresh controller per trial.
    language_model: get_prior(committed_text) -> (n_chars,) prior.
    scorer: epoch (n_channels, n_times) -> float decision score.
    emit: called with each event dict, already validated against events.py.
    """

    def __init__(self, *, controller_factory: Callable, language_model, scorer: Callable,
                 epoch_spec: EpochSpec, gate_policy: str, characters: list[str],
                 emit: Callable[[dict], None], fingerprint: str | None = None,
                 buffer_s: float = DEFAULT_BUFFER_S, stale_epoch_s: float = STALE_EPOCH_S,
                 logger: logging.Logger | None = None):
        if gate_policy not in GATE_POLICIES:
            raise ValueError(f"unknown quality-gate policy {gate_policy!r}; expected one of {GATE_POLICIES}")
        self.controller_factory = controller_factory
        self.language_model = language_model
        self.scorer = scorer
        self.spec = epoch_spec
        self.gate_policy = gate_policy
        self.characters = list(characters)
        self.n_chars = len(self.characters)
        self._emit_cb = emit
        self.fingerprint = fingerprint
        self.stale_epoch_s = stale_epoch_s
        self.log = logger or logging.getLogger(LOGGER_NAME + ".engine")

        window_s = (epoch_spec.tmax - epoch_spec.tmin) + 2 * epoch_spec.pad_s
        self._capacity = int(math.ceil(max(buffer_s, 2 * window_s) * epoch_spec.sfreq))
        self._channels = list(epoch_spec.channels)
        self._ts = np.empty(0)
        self._data = np.empty((0, len(self._channels)))
        self._queue: list = []
        self._seq = 0
        self._generation = 0
        self._now: float | None = None
        self._trial: _Trial | None = None
        self._closed: set = set()
        self._failure: ControllerNotImplementedError | None = None
        self._warned_clock = False
        self.committed_text = ""
        self.committed_indices: list[int] = []
        self.stats = {"flashes_accepted": 0, "flashes_rejected": 0, "characters_committed": 0}

    # -- inputs ---------------------------------------------------------------

    def on_eeg_chunk(self, samples, timestamps) -> None:
        self._check_alive()
        t = np.asarray(timestamps, dtype=float).reshape(-1)
        if t.size == 0:
            return
        x = np.asarray(samples, dtype=float)
        if x.shape != (t.size, len(self._channels)):
            raise ValueError(f"EEG chunk must be ({t.size}, {len(self._channels)}), got {x.shape}")
        self._ts = np.concatenate([self._ts, t])[-self._capacity:]
        self._data = np.concatenate([self._data, x])[-self._capacity:]
        self._process()

    def on_marker_text(self, text: str, timestamp: float) -> None:
        self._check_alive()
        try:
            event = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            self._warn(f"unparseable marker ignored: {str(text)[:120]!r}")
            return
        self.on_marker(event, timestamp)

    def on_marker(self, event: dict, timestamp: float) -> None:
        self._check_alive()
        marker = self._parse_marker(event, timestamp)
        if marker is None:
            return
        if (not self._warned_clock and self._ts.size
                and marker.timestamp > self._ts[-1] + MARKER_CLOCK_AHEAD_WARN_S):
            self._warned_clock = True
            self._warn(f"marker timestamp is {marker.timestamp - self._ts[-1]:.1f} s ahead of the newest EEG "
                       "sample; the EEG and marker clocks may not be synchronized")
        self._seq += 1
        bisect.insort(self._queue, (marker.timestamp, self._seq, marker))
        self._process()

    def on_clock(self, lsl_now: float) -> None:
        """
        The current LSL time (pylsl.local_clock(), the marker clock). Flashes
        whose epoch should have completed more than stale_epoch_s ago are
        rejected, so an EEG outage can't block a trial forever.
        """
        self._check_alive()
        self._now = float(lsl_now)
        self._process()

    def on_marker_stream_changed(self) -> None:
        """A new marker outlet: the stimulus restarted, so trial ids may repeat."""
        self._check_alive()
        self._generation += 1

    @property
    def pending_markers(self) -> int:
        return len(self._queue)

    # -- processing -----------------------------------------------------------

    def _check_alive(self):
        if self._failure is not None:
            raise self._failure

    def _parse_marker(self, event, timestamp) -> _Marker | None:
        if not isinstance(event, dict):
            self._warn(f"marker is not a JSON object, ignored: {str(event)[:120]!r}")
            return None
        kind = event.get("event")
        if kind not in ("trial_start", "flash", "trial_end"):
            self.log.debug("ignoring marker event %r", kind)
            return None
        ts = float(timestamp) if isinstance(timestamp, (int, float)) else math.nan
        trial_id = event.get("trial_id")
        if not _is_nonneg_int(trial_id) or not math.isfinite(ts):
            self._warn(f"malformed {kind} marker ignored (trial_id {trial_id!r}, timestamp {timestamp!r})")
            return None
        if kind != "flash":
            target = event.get("target_char_idx") if kind == "trial_start" else None
            if not (_is_nonneg_int(target) and target < self.n_chars):
                target = None  # optional and log-only: absent or malformed just means "unscored"
            return _Marker(kind, self._generation, trial_id, ts, target_char_idx=target)
        flash_idx, group = event.get("flash_idx"), event.get("group")
        valid_group = (isinstance(group, list) and len(group) > 0 and len(set(group)) == len(group)
                       and all(_is_nonneg_int(g) and g < self.n_chars for g in group))
        if not _is_nonneg_int(flash_idx) or not valid_group:
            self._warn(f"malformed flash marker ignored (trial {trial_id}, flash_idx {flash_idx!r}, "
                       f"group {str(group)[:80]})")
            return None
        return _Marker(kind, self._generation, trial_id, ts, flash_idx, tuple(group))

    def _is_stale(self, marker: _Marker) -> bool:
        if self._now is None:
            return False
        complete_at = marker.timestamp + self.spec.tmax + self.spec.pad_s
        return self._now > complete_at + self.stale_epoch_s

    def _process(self) -> None:
        while self._queue:
            marker = self._queue[0][2]
            if marker.kind == "flash":
                if not marker.admitted:
                    if not self._admit_flash(marker):
                        self._queue.pop(0)
                        continue
                    marker.admitted = True
                # Wait until the buffer reaches the epoch start. Before that, epoching clamps the
                # onset index to the buffer end and, while the buffer holds fewer than n_pad
                # samples (the first 0.5 s), reports a future flash as before_buffer.
                epoch, reason = None, NOT_YET_AVAILABLE
                if self._ts.size and self._ts[-1] >= marker.timestamp + self.spec.tmin:
                    epoch, reason = extract_epoch(self._ts, self._data, marker.timestamp, self.spec)
                if reason == NOT_YET_AVAILABLE:
                    if not self._is_stale(marker):
                        return
                    reason = "stale"
                self._queue.pop(0)
                self._resolve_flash(marker, epoch, reason)
            elif marker.kind == "trial_start":
                self._queue.pop(0)
                self._start_trial(marker.trial_key, flash_idx=None, target_char_idx=marker.target_char_idx)
            else:
                self._queue.pop(0)
                self._end_trial(marker.trial_key)

    def _admit_flash(self, marker: _Marker) -> bool:
        trial = self._trial
        if trial is None or trial.key != marker.trial_key:
            if marker.trial_key in self._closed:
                self._warn(f"flash {marker.flash_idx} of trial {marker.trial_id} arrived after the trial "
                           "ended; ignored")
                return False
            self._warn(f"flash {marker.flash_idx} of trial {marker.trial_id} arrived without a trial_start "
                       "(lost to stream-resolution latency?); starting the trial implicitly")
            trial = self._start_trial(marker.trial_key, flash_idx=marker.flash_idx)
            if trial is None:
                return False
        if marker.flash_idx in trial.flash_idxs:
            self._warn(f"duplicate flash {marker.flash_idx} of trial {marker.trial_id} ignored")
            return False
        trial.flash_idxs.add(marker.flash_idx)
        return not (trial.committed or trial.failed)

    def _start_trial(self, key: tuple, flash_idx: int | None,
                     target_char_idx: int | None = None) -> _Trial | None:
        current = self._trial
        if current is not None:
            if current.key == key:
                self._warn(f"duplicate trial_start for trial {key[1]} ignored")
                return current
            self._close_without_end(current, f"trial {key[1]} started")
        if key in self._closed:
            self._warn(f"trial_start for trial {key[1]}, which already ended, ignored")
            return None
        prior = validate_prior(self.language_model.get_prior(self.committed_text), self.n_chars,
                               source="language model")
        trial = _Trial(key=key, controller=self.controller_factory(), target_char_idx=target_char_idx)
        self._trial = trial
        self._call_controller(trial, "reset", prior, flash_idx=flash_idx)
        ok, initial_posterior = self._posterior_ok(trial, flash_idx, after="reset")
        if not ok:
            trial.failed = True
        self.log.info("trial %s started (context %r)", key[1], self.committed_text)
        if ok:
            self._log_trial_start(trial, prior, initial_posterior)
        return trial

    def _target_fields(self, trial: _Trial) -> dict:
        target = trial.target_char_idx
        return {"target_char_idx": target,
                "target_char": self.characters[target] if target is not None else None}

    def _log_trial_start(self, trial: _Trial, prior, initial_posterior) -> None:
        """
        The experiment log's per-character opening record: the LM's distribution as the model
        produced it (`lm_prior`, aligned to the characters in the service_config record), and
        the posterior the controller actually starts from (`initial_posterior`, the prior blended
        with the uniform floor), so the effect of the blend is visible in the log.
        """
        prior = np.asarray(prior, dtype=float)
        top = np.argsort(prior)[::-1][:5]
        fields = {
            "event": "trial_start", "trial_id": trial.trial_id, "context": self.committed_text,
            "lm_prior": sig_floats(prior), "initial_posterior": sig_floats(initial_posterior),
            "lm_prior_top5": [[self.characters[int(i)], float(f"{prior[i]:.6g}")] for i in top],
            **self._target_fields(trial),
        }
        target = trial.target_char_idx
        if target is not None:
            fields["target_prior_prob"] = float(f"{prior[target]:.6g}")
            fields["target_prior_rank"] = int(np.sum(prior > prior[target])) + 1
        self.log.info("trial_start", extra={"fields": fields})

    def _close_without_end(self, trial: _Trial, why: str) -> None:
        self._trial = None
        self._closed.add(trial.key)
        if not (trial.committed or trial.failed):
            self._warn(f"trial {trial.trial_id} closed without a trial_end marker ({why}; aborted or "
                       "marker lost); nothing committed")

    def _end_trial(self, key: tuple) -> None:
        trial = self._trial
        if trial is None or trial.key != key:
            self._warn(f"trial_end for trial {key[1]}, which is not the open trial, ignored")
            return
        self._trial = None
        self._closed.add(key)
        if trial.committed or trial.failed:
            return
        if trial.controller.t >= 1:
            char_idx = int(np.argmax(np.asarray(trial.controller.log_p, dtype=float)))
            self._commit(trial, char_idx, ev.REASON_TRIAL_ENDED)
        else:
            self._warn(f"trial {trial.trial_id} ended before any flash reached the controller; "
                       "nothing committed")

    def _resolve_flash(self, marker: _Marker, epoch, reason) -> None:
        trial = self._trial
        if trial is None or trial.key != marker.trial_key or trial.committed or trial.failed:
            return
        tid, idx = marker.trial_id, marker.flash_idx
        if reason is not None:
            self.log.info("flash %s of trial %s rejected: epoch %s", idx, tid, reason)
            self._reject(tid, idx, ev.REJECT_EPOCH_INCOMPLETE, [])
            return
        decision = gate_epoch(epoch, self._channels, self.gate_policy)
        if not decision.accept:
            self._reject(tid, idx, ev.REJECT_QUALITY_GATE, list(decision.bad_channels))
            return
        if decision.bad_channels:
            self.log.warning("flash %s of trial %s accepted under the warn policy with bad channels %s",
                             idx, tid, list(decision.bad_channels))
            if not trial.warned_bad_channels:
                trial.warned_bad_channels = True
                self._warn(f"trial {tid}: bad channels {list(decision.bad_channels)} accepted under the "
                           "'warn' quality-gate policy")
        if is_rank_deficient(epoch):
            # Linearly dependent channels (e.g. bridged electrodes the gate can't see): the covariance is
            # singular, and the classifier would raise or return an arbitrary score.
            self.log.warning("flash %s of trial %s: channels are linearly dependent; not scored", idx, tid)
            self._reject(tid, idx, ev.REJECT_INVALID_SCORE, [])
            return
        score = float(self.scorer(epoch))
        if not math.isfinite(score):
            self.log.warning("flash %s of trial %s: classifier score %r", idx, tid, score)
            self._reject(tid, idx, ev.REJECT_INVALID_SCORE, [])
            return

        controller = trial.controller
        stop, char_idx = self._call_controller(trial, "update", list(marker.group), score,
                                               len(marker.group) / self.n_chars, flash_idx=idx)
        ok, posterior = self._posterior_ok(trial, idx, after="update")
        if not ok:
            trial.failed = True
            return
        self.stats["flashes_accepted"] += 1
        max_prob = min(1.0, float(posterior.max()))
        self._log_flash_update(trial, idx, marker.group, score, posterior, max_prob)
        self._emit(ev.posterior_update(tid, idx, max_prob, int(posterior.argmax()), controller.theta,
                                       characters=self.characters))
        if not stop:
            return
        valid_idx = (_is_nonneg_int(char_idx) or isinstance(char_idx, np.integer)) and 0 <= char_idx < self.n_chars
        if not valid_idx:
            self._error(f"trial {tid}, flash {idx}: controller stopped with invalid char_idx {char_idx!r}; "
                        "stopped processing this trial, nothing committed")
            trial.failed = True
            return
        t = controller.t
        if max_prob >= controller.theta and t >= controller.t_min:
            reason = ev.REASON_THRESHOLD_CROSSED
        elif t >= controller.t_max:
            reason = ev.REASON_MAX_FLASHES_REACHED
        else:
            self._error(f"trial {tid}, flash {idx}: controller returned stop=True but neither stopping "
                        f"condition holds (max_prob {max_prob:.4f}, theta {controller.theta}, t {t}, "
                        f"t_min {controller.t_min}, t_max {controller.t_max}); stopped processing this "
                        "trial, nothing committed")
            trial.failed = True
            return
        self._commit(trial, int(char_idx), reason)

    def _call_controller(self, trial: _Trial, method: str, *args, flash_idx: int | None):
        """The only place the controller is called."""
        try:
            return getattr(trial.controller, method)(*args)
        except NotImplementedError as exc:
            self._failure = ControllerNotImplementedError(method, trial.trial_id, flash_idx)
            raise self._failure from exc

    def _posterior_ok(self, trial: _Trial, flash_idx, *, after: str) -> tuple[bool, np.ndarray | None]:
        """Returns (ok, posterior). posterior is the already-computed exp(log_p), reused by the
        caller so a valid posterior is never exponentiated twice in the same flash."""
        controller = trial.controller
        problem = None
        p = None
        try:
            log_p = np.asarray(controller.log_p, dtype=float)
        except (TypeError, ValueError):
            log_p, problem = None, f"log_p is not numeric ({type(controller.log_p).__name__})"
        t = getattr(controller, "t", None)
        if problem is None and log_p.shape != (self.n_chars,):
            problem = f"log_p has shape {log_p.shape}, expected ({self.n_chars},)"
        if problem is None:
            with np.errstate(over="ignore", invalid="ignore"):
                p = np.exp(log_p)
            if np.any(np.isnan(log_p)) or not np.all(np.isfinite(p)):
                problem = "posterior has non-finite entries"
            elif abs(float(p.sum()) - 1.0) > POSTERIOR_SUM_TOLERANCE:
                problem = f"posterior sums to {float(p.sum())!r}"
        if problem is None and not (_is_nonneg_int(t) and (after != "update" or t >= 1)):
            problem = f"flash count t is {t!r}"
        if problem is None:
            return True, p
        where = f"flash {flash_idx}" if flash_idx is not None else "trial_start"
        self._error(f"trial {trial.trial_id}, {where}: posterior after {after}() is invalid ({problem}); "
                    "stopped processing this trial, nothing committed")
        return False, None

    def _log_flash_update(self, trial: _Trial, flash_idx: int, group, score: float, posterior,
                          max_prob: float) -> None:
        """One record per accepted flash: what the classifier said and where the posterior went."""
        lead = int(posterior.argmax())
        fields = {
            "event": "flash_update", "trial_id": trial.trial_id, "flash_idx": flash_idx,
            "t": trial.controller.t, "group": list(group), "score": score,
            "pi": len(group) / self.n_chars, "max_prob": max_prob,
            "leading_char_idx": lead, "leading_char": self.characters[lead],
            "target_char_idx": trial.target_char_idx,
            "target_prob": (float(f"{posterior[trial.target_char_idx]:.6g}")
                            if trial.target_char_idx is not None else None),
        }
        self.log.info("flash_update", extra={"fields": fields})

    def _commit(self, trial: _Trial, char_idx: int, reason: str) -> None:
        char = self.characters[char_idx]
        self._emit(ev.character_committed(trial.trial_id, char_idx, char, trial.controller.t, reason,
                                          characters=self.characters))
        trial.committed = True
        self.committed_text += char
        self.committed_indices.append(char_idx)
        self.stats["characters_committed"] += 1
        self._log_commit(trial, char_idx, char, reason)

    def _log_commit(self, trial: _Trial, char_idx: int, char: str, reason: str) -> None:
        """The per-character closing record: the decision, why, and the final posterior."""
        posterior = np.exp(np.asarray(trial.controller.log_p, dtype=float))
        target = trial.target_char_idx
        fields = {
            "event": "char_committed", "trial_id": trial.trial_id, "char_idx": char_idx, "char": char,
            "reason": reason, "n_flashes_used": trial.controller.t,
            "max_prob": min(1.0, float(posterior.max())), "final_posterior": sig_floats(posterior),
            "committed_text": self.committed_text, **self._target_fields(trial),
            "correct": (char_idx == target) if target is not None else None,
        }
        self.log.info("char_committed", extra={"fields": fields})

    # -- output ---------------------------------------------------------------

    def _emit(self, event: dict) -> None:
        self._emit_cb(event)

    def _reject(self, trial_id, flash_idx, reason, channels) -> None:
        self.stats["flashes_rejected"] += 1
        self.log.info("flash_rejected", extra={"fields": {
            "event": "flash_rejected", "trial_id": trial_id, "flash_idx": flash_idx,
            "reason": reason, "channels": list(channels)}})
        self._emit(ev.flash_rejected(trial_id, flash_idx, reason, channels))

    def _warn(self, detail: str) -> None:
        self.log.warning(detail)
        self._emit(ev.service_status(ev.STATE_WARNING, detail, self.fingerprint))

    def _error(self, detail: str) -> None:
        self.log.error(detail)
        self._emit(ev.service_status(ev.STATE_ERROR, detail, self.fingerprint))


class EegSilenceMonitor:
    """
    Wall-clock EEG silence detector. The caller supplies monotonic time.
    check() returns the silence length once per episode, when it first exceeds
    the threshold; on_chunk() returns the gap length when chunks resume.
    """

    def __init__(self, now: float, threshold_s: float = EEG_SILENCE_WARN_S):
        self.threshold_s = threshold_s
        self.last_chunk = now
        self.silent = False

    def on_chunk(self, now: float) -> float | None:
        gap = now - self.last_chunk
        self.last_chunk = now
        if self.silent:
            self.silent = False
            return gap
        return None

    def check(self, now: float) -> float | None:
        elapsed = now - self.last_chunk
        if not self.silent and elapsed > self.threshold_s:
            self.silent = True
            return elapsed
        return None


# ----------------------------------------------------------------------------
# Construction from a manifest
# ----------------------------------------------------------------------------

@dataclass
class ServiceComponents:
    manifest: Any
    characters: list
    checkpoint: Any
    calibration: dict
    language_model: Any
    controller_factory: Callable
    scorer: Callable

    @property
    def epoch_spec(self) -> EpochSpec:
        return self.checkpoint.epoch_spec

    @property
    def gate_policy(self) -> str:
        return self.manifest.gate_policy


def service_config_fields(comps: ServiceComponents, *, fingerprint: str | None, eeg_name: str,
                          channel_labels, channel_count: int, srate: float) -> dict:
    """
    The experiment log's opening record. Every 72-long array in the later records is aligned to
    `characters` here, so a log file is readable on its own. The manifest embeds each artifact's
    sha256 (classifier, calibration, LM), so this pins exactly which models produced the log.
    """
    return {
        "event": "service_config",
        "manifest_path": str(comps.manifest.path),
        "manifest_fingerprint": fingerprint,
        "manifest": comps.manifest.data,
        "characters": list(comps.characters),
        "platt": {"a": comps.calibration["a"], "b": comps.calibration["b"]},
        "epoch_spec": comps.epoch_spec.to_dict(),
        "eeg_stream": {"name": eeg_name, "channel_count": channel_count,
                       "channel_labels": list(channel_labels) if channel_labels else None,
                       "srate": srate},
    }


def load_service_components(manifest_path) -> ServiceComponents:
    """Everything the engine needs, built only from the session manifest."""
    from stopping.bayesian_stopping import BayesianStoppingController

    manifest = load_manifest(manifest_path)
    characters = load_characters_file(manifest.characters_path)
    checkpoint = load_checkpoint(manifest.classifier_path)
    calibration = load_calibration_file(manifest.calibration_path)
    language_model = build_language_model(manifest, characters)
    st = manifest.stopping
    factory = functools.partial(
        BayesianStoppingController, n_chars=len(characters), lam=st["lam"], theta=st["theta"],
        t_min=st["t_min"], t_max=st["t_max"], sigmoid_a=calibration["a"], sigmoid_b=calibration["b"],
        llr_clip=st["llr_clip"])
    factory()  # constructor arguments are checked at load time, not at the first trial
    return ServiceComponents(manifest=manifest, characters=characters, checkpoint=checkpoint,
                             calibration=calibration, language_model=language_model,
                             controller_factory=factory, scorer=functools.partial(score_epoch, checkpoint))


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def default_log_path() -> Path:
    return Path("logs") / f"inference_{datetime.now():%Y%m%d_%H%M%S}.log.jsonl"


def setup_logging(log_file) -> logging.Logger:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonLinesFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    for old in list(logger.handlers):
        logger.removeHandler(old)
        old.close()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# LSL wiring
# ----------------------------------------------------------------------------

class _MarkerResolver(threading.Thread):
    """
    Probes for the marker stream in the background and opens an inlet the
    moment one is found: an inlet buffers nothing until it is opened, so
    markers pushed between construction and the first pull would be lost.
    Switches to a newer outlet when the stream is replaced (different uid).
    Inlets use recover=False, so a vanished outlet surfaces as LostError
    instead of silently re-attaching to a restarted stimulus.
    """

    def __init__(self, stream_type: str, stop_event: threading.Event, logger: logging.Logger,
                 probe_timeout: float = 0.3, interval: float = 0.5):
        super().__init__(daemon=True, name="marker-resolver")
        self.stream_type = stream_type
        self.stop_event = stop_event
        self.log = logger
        self.probe_timeout = probe_timeout
        self.interval = interval
        self._lock = threading.Lock()
        self._current_uid = None
        self._pending = None

    def take_new_inlet(self):
        with self._lock:
            pending, self._pending = self._pending, None
        return pending

    def mark_lost(self, uid) -> None:
        with self._lock:
            if self._current_uid == uid:
                self._current_uid = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                streams = pylsl.resolve_byprop("type", self.stream_type, minimum=1, timeout=self.probe_timeout)
                if streams:
                    newest = max(streams, key=lambda s: s.created_at())
                    with self._lock:
                        is_new = newest.uid() != self._current_uid
                    if is_new:
                        inlet = pylsl.StreamInlet(newest, max_buflen=360, recover=False,
                                                  processing_flags=MARKER_INLET_PROCESSING_FLAGS)
                        inlet.open_stream(timeout=2.0)
                        with self._lock:
                            if self._pending is not None:
                                self._pending[0].close_stream()
                            self._pending = (inlet, newest.uid(), newest.name())
                            self._current_uid = newest.uid()
            except Exception:
                self.log.exception("marker stream probe failed; retrying")
            self.stop_event.wait(self.interval)


class ServiceRunner:
    """Runs the service against live LSL streams. run() returns the exit code."""

    def __init__(self, manifest_path, *, eeg_type: str = EEG_STREAM_TYPE,
                 marker_type: str = MARKER_STREAM_TYPE, inference_type: str = INFERENCE_STREAM_TYPE,
                 eeg_timeout: float = 30.0, logger: logging.Logger | None = None,
                 controller_factory: Callable | None = None, consumer_wait_s: float = 2.0,
                 marker_probe_interval_s: float = 0.5, heartbeat_s: float = READY_HEARTBEAT_S,
                 eeg_silence_s: float = EEG_SILENCE_WARN_S):
        self.manifest_path = manifest_path
        self.eeg_type, self.marker_type, self.inference_type = eeg_type, marker_type, inference_type
        self.eeg_timeout = eeg_timeout
        self.log = logger or logging.getLogger(LOGGER_NAME)
        self.controller_factory_override = controller_factory
        self.consumer_wait_s = consumer_wait_s
        self.marker_probe_interval_s = marker_probe_interval_s
        self.heartbeat_s = heartbeat_s
        self.eeg_silence_s = eeg_silence_s
        self.stop_event = threading.Event()
        self.failure_message: str | None = None
        self.engine: InferenceEngine | None = None
        self._outlet = None
        self._characters = None
        self._fingerprint = None
        self._resolver: _MarkerResolver | None = None

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> int:
        try:
            return self._run()
        except ControllerNotImplementedError as exc:
            self.log.error(str(exc), exc_info=True)
            self._status(ev.STATE_ERROR, str(exc))
            return self._fail(EXIT_CONTROLLER_NOT_IMPLEMENTED, str(exc))
        except Exception as exc:
            self.log.exception("inference service crashed")
            detail = f"crashed: {type(exc).__name__}: {exc}"
            try:
                self._status(ev.STATE_ERROR, detail)
            except Exception:
                self.log.exception("could not publish the crash status")
            return self._fail(EXIT_CRASH, detail + " (traceback in the log)")
        finally:
            self.stop_event.set()
            if self._resolver is not None:
                self._resolver.join(timeout=2.0)

    def _fail(self, code: int, message: str) -> int:
        self.failure_message = message
        self.log.error("exiting with code %d: %s", code, message)
        return code

    def _emit(self, event: dict) -> None:
        text = ev.encode(event, characters=self._characters)
        self._outlet.push_sample([text])
        self.log.info("emitted %s", event["event"], extra={"fields": {"emitted": event}})

    def _status(self, state: str, detail: str) -> None:
        if self._outlet is None:
            self.log.info("status %s (no outlet yet): %s", state, detail)
            return
        self._emit(ev.service_status(state, detail, self._fingerprint))

    def _run(self) -> int:
        try:
            comps = load_service_components(self.manifest_path)
        except Exception as exc:
            self.log.exception("failed to load the session")
            return self._fail(EXIT_CONFIG, f"cannot load session from {self.manifest_path}: {exc}")
        if self.controller_factory_override is not None:
            comps.controller_factory = self.controller_factory_override
        self._characters = comps.characters
        self._fingerprint = comps.manifest.fingerprint
        short = short_fingerprint(self._fingerprint)

        info = pylsl.StreamInfo(INFERENCE_STREAM_NAME, self.inference_type, 1, 0.0, "string",
                                f"p300_inference_{short}")
        info.desc().append_child_value(INFERENCE_FINGERPRINT_KEY, self._fingerprint)
        self._outlet = pylsl.StreamOutlet(info)
        deadline = time.monotonic() + self.consumer_wait_s
        while time.monotonic() < deadline and not self._outlet.have_consumers() and not self.stop_event.is_set():
            time.sleep(0.05)
        self._status(ev.STATE_STARTING, f"manifest {short} loaded; waiting for EEG stream type {self.eeg_type!r}")

        eeg_inlet = self._resolve_eeg()
        if eeg_inlet is None:
            if self.stop_event.is_set():
                self._status(ev.STATE_STOPPED, "stopped before the EEG stream was found")
                return EXIT_OK
            message = f"no EEG stream of type {self.eeg_type!r} found within {self.eeg_timeout:g} s"
            self._status(ev.STATE_ERROR, message)
            return self._fail(EXIT_NO_EEG, message)

        eeg_info = eeg_inlet.info(timeout=5.0)
        channels, label_warning = channels_for_compatibility_check(
            comps.checkpoint, eeg_info.get_channel_labels(), eeg_info.channel_count())
        try:
            check_stream_compatibility(comps.checkpoint, channels, eeg_info.nominal_srate())
        except CheckpointMismatchError as exc:
            message = f"checkpoint does not match EEG stream {eeg_info.name()!r}: {exc}"
            self._status(ev.STATE_ERROR, message)
            return self._fail(EXIT_CONFIG, message)
        if label_warning:
            self._status(ev.STATE_WARNING, f"EEG stream {eeg_info.name()!r}: {label_warning}")
        for warning in version_warnings(comps.checkpoint):
            self._status(ev.STATE_WARNING, f"library version differs from training: {warning}")

        self.engine = InferenceEngine(
            controller_factory=comps.controller_factory, language_model=comps.language_model,
            scorer=comps.scorer, epoch_spec=comps.epoch_spec, gate_policy=comps.gate_policy,
            characters=comps.characters, emit=self._emit, fingerprint=self._fingerprint,
            logger=self.log.getChild("engine"))
        self.log.info("service_config", extra={"fields": service_config_fields(
            comps, fingerprint=self._fingerprint, eeg_name=eeg_info.name(),
            channel_labels=eeg_info.get_channel_labels(), channel_count=eeg_info.channel_count(),
            srate=eeg_info.nominal_srate())})
        eeg_inlet.open_stream(timeout=5.0)
        ready_detail = (f"EEG {eeg_info.name()!r} connected ({eeg_info.channel_count()} ch @ "
                        f"{eeg_info.nominal_srate():g} Hz); gate {comps.gate_policy}; "
                        f"LM {comps.manifest.lm_config['type']}; manifest {short}")
        self._status(ev.STATE_READY, ready_detail)

        self._resolver = _MarkerResolver(self.marker_type, self.stop_event, self.log.getChild("markers"),
                                         interval=self.marker_probe_interval_s)
        self._resolver.start()
        now = time.monotonic()
        silence = EegSilenceMonitor(now, self.eeg_silence_s)
        last_ready = last_clock = now
        marker_inlet, marker_uid = None, None
        while not self.stop_event.is_set():
            new = self._resolver.take_new_inlet()
            if new is not None:
                if marker_inlet is not None:
                    marker_inlet.close_stream()
                marker_inlet, marker_uid, name = new
                self.engine.on_marker_stream_changed()
                self.log.info("marker stream %r connected (uid %s)", name, marker_uid)

            samples, stamps = eeg_inlet.pull_chunk(timeout=0.02, max_samples=1024)
            now = time.monotonic()
            if stamps:
                gap = silence.on_chunk(now)
                if gap is not None:
                    self.log.info("EEG samples resumed after %.1f s of silence", gap)
                    self._status(ev.STATE_READY, f"EEG samples resumed after {gap:.1f} s; {ready_detail}")
                    last_ready = now
                self.engine.on_eeg_chunk(samples, stamps)
            else:
                silent_for = silence.check(now)
                if silent_for is not None:
                    self._status(ev.STATE_WARNING,
                                 f"no EEG samples from {eeg_info.name()!r} for {silent_for:.1f} s; flashes whose "
                                 f"epochs cannot complete are rejected after {STALE_EPOCH_S:g} s")

            if marker_inlet is not None:
                try:
                    m_samples, m_stamps = marker_inlet.pull_chunk(timeout=0.0)
                except LostError:
                    self.log.warning("marker stream lost (uid %s); waiting for a new one", marker_uid)
                    self._resolver.mark_lost(marker_uid)
                    marker_inlet = None
                    m_samples, m_stamps = [], []
                for sample, stamp in zip(m_samples, m_stamps):
                    self.engine.on_marker_text(sample[0], stamp)

            if now - last_clock >= ENGINE_CLOCK_INTERVAL_S:
                self.engine.on_clock(pylsl.local_clock())
                last_clock = now
            if not silence.silent and now - last_ready >= self.heartbeat_s:
                self._status(ev.STATE_READY, ready_detail)
                last_ready = now

        self.log.info("engine stats", extra={"fields": {"stats": self.engine.stats}})
        self._status(ev.STATE_STOPPED, "stop requested")
        return EXIT_OK

    def _resolve_eeg(self):
        deadline = time.monotonic() + self.eeg_timeout
        delay = 0.25
        while not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            streams = pylsl.resolve_byprop("type", self.eeg_type, minimum=1, timeout=min(1.0, remaining))
            if streams:
                if len(streams) > 1:
                    self.log.warning("%d EEG streams of type %r; using %r", len(streams), self.eeg_type,
                                     streams[0].name())
                return pylsl.StreamInlet(streams[0], max_buflen=30, recover=True,
                                         processing_flags=EEG_INLET_PROCESSING_FLAGS)
            self.log.info("no EEG stream of type %r yet; %.0f s left", self.eeg_type, remaining)
            self.stop_event.wait(min(delay, max(0.0, deadline - time.monotonic())))
            delay = min(delay * 2, 5.0)
        return None


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

class _OneLineArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"inference_service: exit {EXIT_CONFIG}: {message}", file=sys.stderr)
        raise SystemExit(EXIT_CONFIG)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _OneLineArgumentParser(prog="python -m pipeline.inference_service",
                                    description="Live P300 inference service (LSL in, P300Inference out).")
    parser.add_argument("--manifest", required=True, help="session_manifest.json")
    parser.add_argument("--eeg-timeout", type=float, default=30.0, help="seconds to wait for the EEG stream")
    parser.add_argument("--log-file", default=None,
                        help="JSON-lines log (default logs/inference_<YYYYmmdd_HHMMSS>.log.jsonl)")
    parser.add_argument("--eeg-type", default=EEG_STREAM_TYPE)
    parser.add_argument("--marker-type", default=MARKER_STREAM_TYPE)
    parser.add_argument("--inference-type", default=INFERENCE_STREAM_TYPE)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    log_file = Path(args.log_file) if args.log_file else default_log_path()
    try:
        logger = setup_logging(log_file)
    except OSError as exc:
        print(f"inference_service: exit {EXIT_CONFIG}: cannot open log file {log_file}: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    runner = ServiceRunner(args.manifest, eeg_type=args.eeg_type, marker_type=args.marker_type,
                           inference_type=args.inference_type, eeg_timeout=args.eeg_timeout, logger=logger)
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, lambda signum, frame: runner.request_stop())
    logger.info("starting", extra={"fields": {"argv": sys.argv, "args": vars(args)}})
    try:
        code = runner.run()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for handler in logger.handlers:
            handler.flush()
    if code != EXIT_OK:
        print(f"inference_service: exit {code}: {runner.failure_message} (log: {log_file})", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
