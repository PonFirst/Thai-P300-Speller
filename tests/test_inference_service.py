"""
InferenceEngine and ServiceRunner tests.

The controller doubles return hand-written literal posteriors. They perform no
Bayes arithmetic: the real update is the project owner's to write, in
stopping/bayesian_stopping.py. For N=4 with prior [0.4, 0.3, 0.2, 0.1] the
scripted posteriors are the worked examples
    group {0,1}, LLR log 3  -> [0.5, 0.375, 1/12, 1/24]
    then group {0}, LLR log 4 -> [0.8, 0.15, 1/30, 1/60]
    then group {0}, LLR log 4 -> [16/17, 3/68, 1/102, 1/204]
"""

import json
import logging
import math
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pytest
from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_byprop

from pipeline import events as ev
from pipeline import inference_service as svc
from pipeline import manifest as mf
from pipeline.classifier_io import load_build_pipeline, save_checkpoint
from pipeline.constants import EEG_CHANNELS, INFERENCE_FINGERPRINT_KEY
from pipeline.epoching import EpochSpec
from pipeline.inference_service import ControllerNotImplementedError, InferenceEngine
from pipeline.language_model import LanguageModelError
from stopping.bayesian_stopping import BayesianStoppingController

CHARS = ["ก", "ข", "ค", "ง"]
SPEC = EpochSpec()
SFREQ = 250.0
SOA = 0.124
T0 = 100.0
FP = "f" * 64

PRIOR = [0.4, 0.3, 0.2, 0.1]
P1 = [0.5, 0.375, 1 / 12, 1 / 24]
P2 = [0.8, 0.15, 1 / 30, 1 / 60]
P3 = [16 / 17, 3 / 68, 1 / 102, 1 / 204]


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------

class ScriptedController:
    """Scripted literal outputs; no Bayes update is computed here."""

    def __init__(self, steps, reset_posterior=PRIOR, theta=0.9, t_min=2, t_max=10):
        self.steps = list(steps)
        self.reset_posterior = reset_posterior
        self.theta, self.t_min, self.t_max = theta, t_min, t_max
        self.calls = []
        self.log_p = None
        self.t = 0

    def reset(self, lm_prior):
        self.calls.append(("reset", np.array(lm_prior)))
        self.log_p = np.log(np.array(self.reset_posterior, dtype=float))
        self.t = 0

    def update(self, group_idx, s_t, pi):
        self.calls.append(("update", list(group_idx), s_t, pi))
        posterior, stop, char_idx = self.steps.pop(0)
        self.log_p = np.log(np.array(posterior, dtype=float))
        self.t += 1
        return stop, char_idx

    @property
    def updates(self):
        return [c for c in self.calls if c[0] == "update"]


class UpdateNotImplementedController(ScriptedController):
    def update(self, group_idx, s_t, pi):
        self.calls.append(("update", list(group_idx), s_t, pi))
        raise NotImplementedError("update(): stub")


class Factory:
    def __init__(self, controllers):
        self.controllers = list(controllers)
        self.made = []

    def __call__(self):
        controller = self.controllers.pop(0)
        self.made.append(controller)
        return controller


class FixedLM:
    def __init__(self, prior):
        self.prior = prior
        self.contexts = []

    def get_prior(self, context):
        self.contexts.append(context)
        return np.array(self.prior, dtype=float)


class ScriptedScorer:
    def __init__(self, scores):
        self.scores = list(scores)
        self.shapes = []

    def __call__(self, epoch):
        self.shapes.append(epoch.shape)
        return self.scores.pop(0)


class Stream:
    def __init__(self, duration_s=20.0, seed=0, flat_channel=None, drop=None, t0=T0):
        n = int(duration_s * SFREQ)
        self.ts = t0 + np.arange(n) / SFREQ
        self.data = np.random.default_rng(seed).normal(0, 10, size=(n, 8))
        if flat_channel:
            self.data[:, EEG_CHANNELS.index(flat_channel)] = 0.0
        if drop:
            keep = ~((self.ts >= drop[0]) & (self.ts < drop[1]))
            self.ts, self.data = self.ts[keep], self.data[keep]
        self.pos = 0

    def feed_until(self, engine, t, chunk=25):
        stop = int(np.searchsorted(self.ts, t))
        while self.pos < stop:
            end = min(self.pos + chunk, stop)
            engine.on_eeg_chunk(self.data[self.pos:end], self.ts[self.pos:end])
            self.pos = end


def make_engine(controllers, scores, prior=PRIOR, policy="skip"):
    events = []
    factory, lm, scorer = Factory(controllers), FixedLM(prior), ScriptedScorer(scores)
    engine = InferenceEngine(controller_factory=factory, language_model=lm, scorer=scorer, epoch_spec=SPEC,
                             gate_policy=policy, characters=CHARS, emit=events.append, fingerprint=FP)
    return engine, events, lm, scorer


def start(engine, tid, ts):
    engine.on_marker({"event": "trial_start", "trial_id": tid, "target_char_idx": 0, "target_char": "ก"}, ts)


def flash(engine, tid, idx, group, ts):
    engine.on_marker({"event": "flash", "trial_id": tid, "flash_idx": idx, "group": group, "is_target": False}, ts)


def end(engine, tid, ts):
    engine.on_marker({"event": "trial_end", "trial_id": tid}, ts)


def PU(tid, idx, p, lead, theta=0.9):
    return ev.posterior_update(tid, idx, p, lead, theta, characters=CHARS)


def CC(tid, cidx, n, reason):
    return ev.character_committed(tid, cidx, CHARS[cidx], n, reason, characters=CHARS)


def REJ(tid, idx, reason, channels=()):
    return ev.flash_rejected(tid, idx, reason, list(channels))


def STATUS(state, text):
    return ("status", state, text)


def assert_events(actual, expected):
    for event in actual:
        ev.validate_event(event, characters=CHARS)
    assert len(actual) == len(expected), f"got {[(e['event'], e.get('detail', e.get('flash_idx'))) for e in actual]}"
    for got, want in zip(actual, expected):
        if isinstance(want, tuple):
            assert got["event"] == ev.SERVICE_STATUS and got["state"] == want[1], got
            assert want[2] in got["detail"], got
            assert got["manifest_fingerprint"] == FP
            continue
        assert got.keys() == want.keys(), (got, want)
        for key, value in want.items():
            if isinstance(value, float):
                assert math.isclose(got[key], value, rel_tol=1e-12), (key, got, want)
            else:
                assert got[key] == value, (key, got, want)


# ---------------------------------------------------------------------------
# engine: trial lifecycle
# ---------------------------------------------------------------------------

def test_full_session_event_order_and_commit_reasons():
    c0 = ScriptedController([(P1, False, None), (P2, False, None), (P3, True, 0)])
    c1 = ScriptedController([(P1, False, None), ([0.6, 0.2, 0.1, 0.1], True, 0)], t_max=2)
    c2 = ScriptedController([(P1, False, None), ([0.25, 0.6, 0.1, 0.05], False, None)])
    c3 = ScriptedController([])
    engine, events, lm, scorer = make_engine([c0, c1, c2, c3], [1.5, 2.0, 2.5, 0.3, 0.4, 0.5, 0.6])

    base = [T0 + 2.0 + 3.0 * k for k in range(4)]
    groups0 = [[0, 1], [0], [0], [2, 3], [1]]
    start(engine, 0, base[0] - 0.5)
    for i, g in enumerate(groups0):
        flash(engine, 0, i, g, base[0] + i * SOA)
    end(engine, 0, base[0] + 5 * SOA)
    start(engine, 1, base[1] - 0.5)
    for i in range(3):
        flash(engine, 1, i, [0, 1], base[1] + i * SOA)
    end(engine, 1, base[1] + 3 * SOA)
    start(engine, 2, base[2] - 0.5)
    for i in range(2):
        flash(engine, 2, i, [1], base[2] + i * SOA)
    end(engine, 2, base[2] + 2 * SOA)
    start(engine, 3, base[3] - 0.5)
    end(engine, 3, base[3])
    assert events == []

    Stream().feed_until(engine, T0 + 15.0)
    assert_events(events, [
        PU(0, 0, 0.5, 0), PU(0, 1, 0.8, 0), PU(0, 2, 16 / 17, 0),
        CC(0, 0, 3, ev.REASON_THRESHOLD_CROSSED),
        PU(1, 0, 0.5, 0), PU(1, 1, 0.6, 0),
        CC(1, 0, 2, ev.REASON_MAX_FLASHES_REACHED),
        PU(2, 0, 0.5, 0), PU(2, 1, 0.6, 1),
        CC(2, 1, 2, ev.REASON_TRIAL_ENDED),
        STATUS(ev.STATE_WARNING, "trial 3 ended before any flash"),
    ])
    reasons = [e["reason"] for e in events if e["event"] == ev.CHARACTER_COMMITTED]
    assert [ev.is_fallback(r) for r in reasons] == [False, True, True]

    # nothing reaches the controller or the classifier after a commit
    assert [(u[1], u[2], u[3]) for u in c0.updates] == [([0, 1], 1.5, 0.5), ([0], 2.0, 0.25), ([0], 2.5, 0.25)]
    assert len(c1.updates) == 2 and len(c2.updates) == 2 and c3.calls[0][0] == "reset"
    assert scorer.scores == [] and scorer.shapes == [(8, 200)] * 7
    for c in (c0, c1, c2, c3):
        assert c.calls[0][0] == "reset"
        np.testing.assert_array_equal(c.calls[0][1], PRIOR)
    assert lm.contexts == ["", "ก", "กก", "กกข"]
    assert engine.committed_text == "กกข" and engine.pending_markers == 0


@pytest.mark.parametrize("posterior,t_min", [
    ([0.6, 0.2, 0.1, 0.1], 1),   # max_prob < theta and t < t_max
    (P3, 2),                     # max_prob >= theta but t < t_min, and t < t_max
])
def test_stop_without_a_stopping_condition_commits_nothing(posterior, t_min):
    c = ScriptedController([(posterior, True, 0), (P1, False, None)], t_min=t_min, t_max=10)
    engine, events, _, _ = make_engine([c], [1.0, 1.0])
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    flash(engine, 0, 1, [0], T0 + 2.0 + SOA)
    end(engine, 0, T0 + 2.0 + 2 * SOA)
    Stream().feed_until(engine, T0 + 5.0)
    assert_events(events, [PU(0, 0, max(posterior), 0),
                           STATUS(ev.STATE_ERROR, "trial 0, flash 0: controller returned stop=True but "
                                                  "neither stopping condition holds")])
    assert len(c.updates) == 1 and engine.committed_text == "" and engine.pending_markers == 0


def test_trial_end_waits_for_queued_flash_epochs():
    c = ScriptedController([(P1, False, None), (P2, False, None), (P3, False, None)])
    engine, events, _, _ = make_engine([c], [1.0, 1.0, 1.0])
    onsets = [T0 + 2.0 + i * SOA for i in range(3)]
    start(engine, 0, T0 + 1.5)
    for i, t in enumerate(onsets):
        flash(engine, 0, i, [0], t)
    end(engine, 0, onsets[-1] + SOA)

    stream = Stream()
    stream.feed_until(engine, onsets[1] + 1.35)  # flash 1's window is complete, flash 2's is not
    assert_events(events, [PU(0, 0, 0.5, 0), PU(0, 1, 0.8, 0)])
    assert engine.pending_markers == 2 and engine.committed_text == ""

    stream.feed_until(engine, onsets[2] + 2.0)
    assert_events(events, [PU(0, 0, 0.5, 0), PU(0, 1, 0.8, 0), PU(0, 2, 16 / 17, 0),
                           CC(0, 0, 3, ev.REASON_TRIAL_ENDED)])


def test_stale_flash_is_rejected_so_trial_end_is_not_blocked():
    c = ScriptedController([(P1, False, None)])
    engine, events, _, _ = make_engine([c], [1.0])
    onsets = [T0 + 2.0, T0 + 2.0 + SOA]
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], onsets[0])
    flash(engine, 0, 1, [0], onsets[1])
    end(engine, 0, onsets[1] + SOA)
    Stream().feed_until(engine, onsets[0] + 1.35)  # then the EEG stops
    assert_events(events, [PU(0, 0, 0.5, 0)])

    complete_at = onsets[1] + SPEC.tmax + SPEC.pad_s
    engine.on_clock(complete_at + 4.9)
    assert engine.pending_markers == 2 and len(events) == 1
    engine.on_clock(complete_at + 5.1)
    assert_events(events, [PU(0, 0, 0.5, 0), REJ(0, 1, ev.REJECT_EPOCH_INCOMPLETE),
                           CC(0, 0, 1, ev.REASON_TRIAL_ENDED)])
    assert engine.pending_markers == 0


def test_stale_flash_with_no_eeg_at_all():
    c = ScriptedController([])
    engine, events, _, _ = make_engine([c], [])
    start(engine, 0, T0)
    flash(engine, 0, 0, [0], T0 + 1.0)
    end(engine, 0, T0 + 1.2)
    engine.on_clock(T0 + 1.0)
    assert events == [] and engine.pending_markers == 2
    engine.on_clock(T0 + 1.0 + SPEC.tmax + SPEC.pad_s + 5.5)
    assert_events(events, [REJ(0, 0, ev.REJECT_EPOCH_INCOMPLETE),
                           STATUS(ev.STATE_WARNING, "trial 0 ended before any flash")])
    assert c.updates == []


def test_eeg_silence_monitor_warns_once_and_reports_recovery():
    monitor = svc.EegSilenceMonitor(now=0.0, threshold_s=2.0)
    assert monitor.check(1.9) is None and monitor.check(2.0) is None
    assert monitor.check(2.5) == 2.5 and monitor.silent
    assert monitor.check(5.0) is None
    assert monitor.on_chunk(6.0) == 6.0 and not monitor.silent
    assert monitor.on_chunk(6.1) is None and monitor.check(7.0) is None
    assert monitor.check(8.2) == pytest.approx(2.1)


def test_incomplete_epochs_are_rejected_not_scored():
    c = ScriptedController([(P1, False, None)])
    engine, events, _, scorer = make_engine([c], [1.0])
    start(engine, 0, T0)
    flash(engine, 0, 0, [0], T0 + 0.1)       # window starts before the first sample
    flash(engine, 0, 1, [0], T0 + 3.0)       # window spans a dropout
    flash(engine, 0, 2, [0], T0 + 6.0)
    Stream(drop=(T0 + 3.3, T0 + 3.35)).feed_until(engine, T0 + 9.0)
    assert_events(events, [REJ(0, 0, ev.REJECT_EPOCH_INCOMPLETE), REJ(0, 1, ev.REJECT_EPOCH_INCOMPLETE),
                           PU(0, 2, 0.5, 0)])
    assert len(c.updates) == 1 and len(scorer.shapes) == 1


def test_implicit_trial_start_and_duplicate_flash():
    c = ScriptedController([(P1, False, None), (P2, False, None)])
    engine, events, lm, _ = make_engine([c], [1.0, 1.0])
    flash(engine, 5, 0, [0, 1], T0 + 2.0)
    flash(engine, 5, 0, [0, 1], T0 + 2.0)
    flash(engine, 5, 1, [0], T0 + 2.0 + SOA)
    assert c.calls[0][0] == "reset" and lm.contexts == [""]
    Stream().feed_until(engine, T0 + 5.0)
    assert_events(events, [STATUS(ev.STATE_WARNING, "flash 0 of trial 5 arrived without a trial_start"),
                           PU(5, 0, 0.5, 0),
                           STATUS(ev.STATE_WARNING, "duplicate flash 0 of trial 5"),
                           PU(5, 1, 0.8, 0)])
    assert len(c.updates) == 2


@pytest.mark.parametrize("bad", [[0.5, 0.5, 0.5, 0.5], [math.nan, 0.5, 0.25, 0.25]])
def test_invalid_posterior_stops_the_trial_without_commit(bad):
    c = ScriptedController([(bad, False, None), (P1, False, None)])
    engine, events, _, _ = make_engine([c], [1.0, 1.0])
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    flash(engine, 0, 1, [0], T0 + 2.0 + SOA)
    end(engine, 0, T0 + 2.0 + 2 * SOA)
    with np.errstate(invalid="ignore"):
        Stream().feed_until(engine, T0 + 5.0)
    assert_events(events, [STATUS(ev.STATE_ERROR, "trial 0, flash 0: posterior after update() is invalid")])
    assert len(c.updates) == 1 and engine.committed_text == ""


def test_invalid_language_model_prior_fails_loudly():
    engine, events, _, _ = make_engine([ScriptedController([])], [], prior=[0.5, 0.5, 0.5, 0.5])
    with pytest.raises(LanguageModelError, match="sum to 1"):
        start(engine, 0, T0)


# ---------------------------------------------------------------------------
# engine: scores and quality gate
# ---------------------------------------------------------------------------

def test_non_finite_scores_never_reach_update():
    c = ScriptedController([(P1, False, None)])
    engine, events, _, _ = make_engine([c], [math.nan, math.inf, -math.inf, 1e300])
    onsets = [T0 + 2.0 + i * SOA for i in range(4)]
    start(engine, 0, T0 + 1.5)
    for i, t in enumerate(onsets):
        flash(engine, 0, i, [0, 1], t)
    stream = Stream()
    stream.feed_until(engine, onsets[2] + 1.35)
    assert_events(events, [REJ(0, i, ev.REJECT_INVALID_SCORE) for i in range(3)])
    assert c.updates == []
    np.testing.assert_array_equal(c.log_p, np.log(PRIOR))

    stream.feed_until(engine, onsets[3] + 2.0)
    assert_events(events, [REJ(0, i, ev.REJECT_INVALID_SCORE) for i in range(3)] + [PU(0, 3, 0.5, 0)])
    assert c.updates == [("update", [0, 1], 1e300, 0.5)]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_quality_gate_flat_channel(policy, seed):
    c = ScriptedController([(P1, False, None)])
    engine, events, _, scorer = make_engine([c], [1.0], policy=policy)
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    Stream(seed=seed, flat_channel="Pz").feed_until(engine, T0 + 4.0)
    if policy == "skip":
        assert_events(events, [REJ(0, 0, ev.REJECT_QUALITY_GATE, ["Pz"])])
        assert c.updates == [] and scorer.shapes == []
    else:
        # warn lets the flash past the gate, but a flat channel makes the channels linearly dependent:
        # the real classifier's covariance is singular, so the flash is not scored either way.
        assert_events(events, [STATUS(ev.STATE_WARNING, "bad channels ['Pz']"),
                               REJ(0, 0, ev.REJECT_INVALID_SCORE, [])])
        assert c.updates == [] and scorer.shapes == []


def test_clean_epochs_pass_the_gate_across_seeds():
    for seed in range(5):
        c = ScriptedController([(P1, False, None)])
        engine, events, _, _ = make_engine([c], [1.0])
        start(engine, 0, T0 + 1.5)
        flash(engine, 0, 0, [0], T0 + 2.0)
        Stream(seed=seed).feed_until(engine, T0 + 4.0)
        assert_events(events, [PU(0, 0, 0.5, 0)])


# ---------------------------------------------------------------------------
# engine: unimplemented controller
# ---------------------------------------------------------------------------

def test_update_not_implemented_fails_loudly_at_that_call():
    c = UpdateNotImplementedController([])
    engine, events, _, _ = make_engine([c], [math.nan, 1.0])
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    flash(engine, 0, 1, [2, 3], T0 + 2.0 + SOA)
    end(engine, 0, T0 + 2.5)
    stream = Stream()
    with pytest.raises(ControllerNotImplementedError) as info:
        stream.feed_until(engine, T0 + 5.0)
    err = info.value
    assert (err.method, err.trial_id, err.flash_idx) == ("update", 0, 1)
    assert "update()" in str(err) and "trial 0, flash 1" in str(err)
    assert "stopping/bayesian_stopping.py" in str(err)
    assert isinstance(err.__cause__, NotImplementedError)
    # the reset before it really happened, with the LM prior
    assert [call[0] for call in c.calls] == ["reset", "update"]
    np.testing.assert_array_equal(c.calls[0][1], PRIOR)

    assert_events(events, [REJ(0, 0, ev.REJECT_INVALID_SCORE)])
    with pytest.raises(ControllerNotImplementedError):
        stream.feed_until(engine, T0 + 8.0)
    with pytest.raises(ControllerNotImplementedError):
        start(engine, 1, T0 + 9.0)
    with pytest.raises(ControllerNotImplementedError):
        engine.on_clock(T0 + 30.0)
    assert len(events) == 1 and len(c.calls) == 2


def _real_reset_is_stub():
    try:
        BayesianStoppingController(n_chars=4).reset(np.full(4, 0.25))
    except NotImplementedError:
        return True
    return False


@pytest.mark.skipif(not _real_reset_is_stub(), reason="BayesianStoppingController.reset() is implemented now")
def test_real_controller_stub_names_reset():
    events = []
    engine = InferenceEngine(controller_factory=lambda: BayesianStoppingController(n_chars=4),
                             language_model=FixedLM(PRIOR), scorer=ScriptedScorer([]), epoch_spec=SPEC,
                             gate_policy="skip", characters=CHARS, emit=events.append, fingerprint=FP)
    with pytest.raises(ControllerNotImplementedError, match=r"reset\(\).*trial 0, before its first flash"):
        start(engine, 0, T0)
    assert events == []

    engine2 = InferenceEngine(controller_factory=lambda: BayesianStoppingController(n_chars=4),
                              language_model=FixedLM(PRIOR), scorer=ScriptedScorer([]), epoch_spec=SPEC,
                              gate_policy="skip", characters=CHARS, emit=events.append, fingerprint=FP)
    with pytest.raises(ControllerNotImplementedError, match=r"reset\(\).*trial 7, flash 3") as info:
        flash(engine2, 7, 3, [0], T0 + 2.0)
    assert info.value.method == "reset"


# ---------------------------------------------------------------------------
# experiment log: the structured records the engine writes to its logger
# ---------------------------------------------------------------------------

class _FieldsCapture(logging.Handler):
    """Collects the structured `fields` of every record, i.e. what JsonLinesFormatter writes."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            self.records.append(dict(fields))

    def of(self, event):
        return [r for r in self.records if r.get("event") == event]


def make_logged_engine(controllers, scores, prior):
    capture = _FieldsCapture()
    logger = logging.getLogger(f"test.engine.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(capture)
    events = []
    engine = InferenceEngine(controller_factory=Factory(controllers), language_model=FixedLM(prior),
                             scorer=ScriptedScorer(scores), epoch_spec=SPEC, gate_policy="skip",
                             characters=CHARS, emit=events.append, fingerprint=FP, logger=logger)
    return engine, capture


LM_PRIOR = [0.7, 0.1, 0.1, 0.1]     # what the language model says
BLENDED = [0.4, 0.3, 0.2, 0.1]      # what the (scripted) controller starts its posterior from


def test_log_has_the_lm_prior_at_trial_start_and_every_flash_after():
    c = ScriptedController([(P1, False, None), (P3, True, 0)], reset_posterior=BLENDED, t_min=2)
    engine, cap = make_logged_engine([c], [1.5, 2.0], LM_PRIOR)
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0, 1], T0 + 2.0)
    flash(engine, 0, 1, [0], T0 + 2.0 + SOA)
    end(engine, 0, T0 + 2.0 + 2 * SOA)
    Stream().feed_until(engine, T0 + 5.0)

    (opening,) = cap.of("trial_start")
    assert opening["trial_id"] == 0 and opening["context"] == ""
    assert opening["lm_prior"] == LM_PRIOR            # the LM's own distribution ...
    assert opening["initial_posterior"] == BLENDED    # ... and where the controller actually starts
    assert opening["lm_prior_top5"][0] == ["ก", 0.7]
    assert (opening["target_char_idx"], opening["target_char"]) == (0, "ก")
    assert opening["target_prior_prob"] == 0.7 and opening["target_prior_rank"] == 1

    first, second = cap.of("flash_update")
    assert (first["flash_idx"], first["t"], first["group"], first["score"], first["pi"]) == (0, 1, [0, 1], 1.5, 0.5)
    assert first["leading_char"] == "ก"
    assert first["max_prob"] == pytest.approx(0.5) and first["target_prob"] == pytest.approx(0.5)
    assert (second["t"], second["group"], second["pi"]) == (2, [0], 0.25)
    assert second["target_prob"] == pytest.approx(16 / 17, rel=1e-5)

    (commit,) = cap.of("char_committed")
    assert (commit["char"], commit["char_idx"], commit["reason"]) == ("ก", 0, ev.REASON_THRESHOLD_CROSSED)
    assert commit["n_flashes_used"] == 2 and commit["committed_text"] == "ก"
    assert commit["correct"] is True
    assert commit["final_posterior"] == pytest.approx(P3, rel=1e-5)
    # opening -> flashes -> commit, in that order
    assert [r["event"] for r in cap.records] == ["trial_start", "flash_update", "flash_update", "char_committed"]


def test_log_scores_a_wrong_commit_and_ranks_the_target_in_the_prior():
    c = ScriptedController([(P3, True, 0)], reset_posterior=BLENDED, t_min=1)
    engine, cap = make_logged_engine([c], [1.0], LM_PRIOR)
    engine.on_marker({"event": "trial_start", "trial_id": 0, "target_char_idx": 1, "target_char": "ข"}, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    Stream().feed_until(engine, T0 + 4.0)
    (opening,) = cap.of("trial_start")
    assert opening["target_char"] == "ข" and opening["target_prior_rank"] == 2   # only ก outranks it
    (commit,) = cap.of("char_committed")
    assert commit["char"] == "ก" and commit["target_char"] == "ข" and commit["correct"] is False


@pytest.mark.parametrize("marker_target", [None, 99, -1, "ก", True, 1.5])
def test_log_without_a_usable_target_is_unscored_and_decoding_is_unaffected(marker_target):
    c = ScriptedController([(P3, True, 0)], reset_posterior=BLENDED, t_min=1)
    engine, cap = make_logged_engine([c], [1.0], LM_PRIOR)
    marker = {"event": "trial_start", "trial_id": 0}
    if marker_target is not None:
        marker["target_char_idx"] = marker_target
    engine.on_marker(marker, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    Stream().feed_until(engine, T0 + 4.0)
    (opening,) = cap.of("trial_start")
    assert opening["target_char_idx"] is None and "target_prior_prob" not in opening
    (update,) = cap.of("flash_update")
    assert update["target_prob"] is None
    (commit,) = cap.of("char_committed")
    assert commit["correct"] is None and commit["char"] == "ก"
    assert engine.committed_text == "ก"


def test_log_records_rejected_flashes_and_a_trial_end_commit():
    c = ScriptedController([(P1, False, None)], reset_posterior=BLENDED, t_min=2)
    engine, cap = make_logged_engine([c], [1.0], LM_PRIOR)
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    Stream(flat_channel="Pz").feed_until(engine, T0 + 4.0)
    (rejected,) = cap.of("flash_rejected")
    assert (rejected["flash_idx"], rejected["reason"], rejected["channels"]) == (0, ev.REJECT_QUALITY_GATE, ["Pz"])

    c2 = ScriptedController([(P1, False, None)], reset_posterior=BLENDED, t_min=2)
    engine, cap = make_logged_engine([c2], [1.0], LM_PRIOR)
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    end(engine, 0, T0 + 2.0 + SOA)
    Stream().feed_until(engine, T0 + 5.0)
    (commit,) = cap.of("char_committed")
    assert commit["reason"] == ev.REASON_TRIAL_ENDED and commit["n_flashes_used"] == 1


def test_log_records_survive_the_json_lines_formatter():
    """What the service writes to disk: one valid JSON object per record, Thai text unescaped."""
    c = ScriptedController([(P3, True, 0)], reset_posterior=BLENDED, t_min=1)
    engine, cap = make_logged_engine([c], [1.0], LM_PRIOR)
    start(engine, 0, T0 + 1.5)
    flash(engine, 0, 0, [0], T0 + 2.0)
    Stream().feed_until(engine, T0 + 4.0)
    formatter = svc.JsonLinesFormatter()
    lines = []
    for fields in cap.records:
        record = logging.LogRecord("p300.inference.engine", logging.INFO, __file__, 0, fields["event"], (), None)
        record.fields = fields
        lines.append(formatter.format(record))
    assert [json.loads(line)["event"] for line in lines] == ["trial_start", "flash_update", "char_committed"]
    assert "ก" in lines[-1]    # Thai text is written as text, not \u-escapes


# ---------------------------------------------------------------------------
# construction from a manifest
# ---------------------------------------------------------------------------

REPO_CHARACTERS = mf.REPO_ROOT / "collection" / "characters_th.json"
UNIFORM72 = [1 / 72] * 72


def _literal72(top):
    return [top] + [(1 - top) / 71] * 71


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    root = tmp_path_factory.mktemp("session")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 5, size=(60, 8, SPEC.n_times))
    y = np.array([1 if i % 4 == 0 else 0 for i in range(60)])
    X[y == 1, :, 80:95] += 5.0
    pipe = load_build_pipeline()()
    pipe.fit(X, y)
    ckpt = save_checkpoint(root / "models" / "t_classifier.joblib", pipe, SPEC,
                           {"recording_sha256": "0" * 64, "n_epochs": 60, "n_targets": 15, "n_trials": 3})
    cal = root / "models" / "t_platt.json"
    cal.write_text(json.dumps({"format": mf.CALIBRATION_FORMAT, "a": 1.5, "b": -0.5,
                               "score_method": mf.SCORE_METHOD, "classifier_sha256": mf.sha256_file(ckpt)}))
    data = mf.build_manifest_dict(base_dir=root, characters_path=REPO_CHARACTERS, classifier_path=ckpt,
                                  calibration_path=cal, stopping={"theta": 0.85, "t_min": 3},
                                  quality_gate={"policy": "warn"})
    return mf.write_manifest(data, root / "session_manifest.json")


def test_components_built_from_manifest(session):
    comps = svc.load_service_components(session.path)
    controller = comps.controller_factory()
    assert isinstance(controller, BayesianStoppingController)
    assert (controller.n_chars, controller.theta, controller.t_min, controller.t_max) == (72, 0.85, 3, 120)
    assert (controller.sigmoid_a, controller.sigmoid_b, controller.lam, controller.llr_clip) == (1.5, -0.5, 0.7, 8.0)
    assert comps.gate_policy == "warn" and comps.epoch_spec == SPEC
    assert np.all(comps.language_model.get_prior("") == 1 / 72)
    assert isinstance(comps.scorer(np.random.default_rng(1).normal(0, 5, (8, 200))), float)
    InferenceEngine(controller_factory=comps.controller_factory, language_model=comps.language_model,
                    scorer=comps.scorer, epoch_spec=comps.epoch_spec, gate_policy=comps.gate_policy,
                    characters=comps.characters, emit=lambda e: None)


def test_service_config_record_pins_the_models_and_aligns_the_arrays(session):
    comps = svc.load_service_components(session.path)
    fields = svc.service_config_fields(comps, fingerprint=session.fingerprint, eeg_name="UnicornEEG",
                                       channel_labels=list(EEG_CHANNELS), channel_count=8, srate=250.0)
    assert fields["event"] == "service_config"
    assert fields["manifest_fingerprint"] == session.fingerprint == mf.fingerprint_of(fields["manifest"])
    assert fields["characters"] == comps.characters and len(fields["characters"]) == 72
    assert fields["platt"] == {"a": 1.5, "b": -0.5}
    assert fields["manifest"]["language_model"] == {"type": "uniform"}
    assert fields["manifest"]["classifier"]["sha256"] == mf.sha256_file(comps.manifest.classifier_path)
    assert fields["epoch_spec"] == SPEC.to_dict()
    assert fields["eeg_stream"] == {"name": "UnicornEEG", "channel_count": 8,
                                    "channel_labels": list(EEG_CHANNELS), "srate": 250.0}
    json.dumps(fields, ensure_ascii=False)   # serializable as written to the log
    unlabeled = svc.service_config_fields(comps, fingerprint=None, eeg_name="x", channel_labels=None,
                                          channel_count=8, srate=250.0)
    assert unlabeled["eeg_stream"]["channel_labels"] is None


def test_eeg_inlet_flags_follow_the_dejitter_constant(session, monkeypatch):
    import pylsl
    from pipeline import constants

    assert svc.eeg_inlet_processing_flags(False) == pylsl.proc_clocksync
    assert svc.eeg_inlet_processing_flags(True) == pylsl.proc_clocksync | pylsl.proc_dejitter
    assert svc.EEG_INLET_PROCESSING_FLAGS == svc.eeg_inlet_processing_flags(constants.EEG_INLET_DEJITTER)
    assert svc.MARKER_INLET_PROCESSING_FLAGS == pylsl.proc_clocksync

    # the runner opens the EEG inlet with exactly these flags
    created = []
    monkeypatch.setattr(svc.pylsl, "resolve_byprop", lambda *a, **k: ["fake-stream-info"])
    monkeypatch.setattr(svc.pylsl, "StreamInlet", lambda info, **kwargs: created.append(kwargs) or "inlet")
    monkeypatch.setattr(svc, "EEG_INLET_PROCESSING_FLAGS", svc.eeg_inlet_processing_flags(True))
    runner = svc.ServiceRunner(session.path, eeg_type=f"EEGTEST_{uuid.uuid4().hex}", eeg_timeout=1.0)
    assert runner._resolve_eeg() == "inlet"
    assert created[0]["processing_flags"] == pylsl.proc_clocksync | pylsl.proc_dejitter
    assert created[0]["recover"] is True


def test_cli_bad_manifest_exits_2_with_one_stderr_line(tmp_path, capsys):
    log = tmp_path / "svc.log.jsonl"
    code = svc.main(["--manifest", str(tmp_path / "missing.json"), "--log-file", str(log)])
    assert code == svc.EXIT_CONFIG
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "exit 2" in err and "session manifest not found" in err
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert any("traceback" in r for r in records)
    assert signal.getsignal(signal.SIGINT) == signal.default_int_handler


# ---------------------------------------------------------------------------
# real LSL (unique stream types only)
# ---------------------------------------------------------------------------

def _types():
    uid = uuid.uuid4().hex
    return f"EEGTEST_{uid}", f"MARKERSTEST_{uid}", f"P300INFTEST_{uid}"


class EEGOutlet(threading.Thread):
    """Regular 250 Hz grid on the LSL clock, so timestamps have no jitter gaps. pause() drops samples."""

    def __init__(self, stream_type, n_channels=8, labels=None):
        super().__init__(daemon=True)
        info = StreamInfo("EEGTest", stream_type, n_channels, SFREQ, "float32", f"eegtest_{uuid.uuid4().hex}")
        info.set_channel_labels(list(EEG_CHANNELS[:n_channels]) if labels is None else list(labels))
        self.outlet = StreamOutlet(info)
        self.n_channels = n_channels
        self.stop_event = threading.Event()
        self.paused = threading.Event()

    def run(self):
        rng = np.random.default_rng(0)
        t_start, sent = local_clock(), 0
        while not self.stop_event.is_set():
            due = int((local_clock() - t_start) * SFREQ)
            if due > sent:
                if not self.paused.is_set():
                    chunk = rng.normal(0, 10, size=(due - sent, self.n_channels)).astype(np.float32)
                    self.outlet.push_chunk(chunk.tolist(), timestamp=t_start + (due - 1) / SFREQ)
                sent = due
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2)


def _pull_until(inlet, received, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        samples, _ = inlet.pull_chunk(timeout=0.1)
        received.extend(ev.decode(s[0]) for s in samples)
        if predicate(received):
            return True
    return False


def _has(kind, state=None, text=None, since=0):
    def check(events):
        return any(e["event"] == kind and (state is None or e.get("state") == state)
                   and (text is None or text in e.get("detail", "")) for e in events[since:])
    return check


def _is_status(event, state):
    return event["event"] == ev.SERVICE_STATUS and event["state"] == state


def _start_runner(runner):
    codes = []
    thread = threading.Thread(target=lambda: codes.append(runner.run()), daemon=True)
    thread.start()
    return thread, codes


def _connect(inference_type, timeout=15):
    found = resolve_byprop("type", inference_type, timeout=timeout)
    assert found, "inference outlet never appeared"
    inlet = StreamInlet(found[0])
    inlet.open_stream(timeout=5)
    return inlet


def test_runner_over_real_lsl(session):
    eeg_type, marker_type, inference_type = _types()
    eeg = EEGOutlet(eeg_type)
    eeg.start()
    markers = StreamOutlet(StreamInfo("MarkersTest", marker_type, 1, 0, "string", f"mk_{uuid.uuid4().hex}"))
    steps = [(_literal72(0.3), False, None), (_literal72(0.6), False, None), (_literal72(0.95), True, 0)]
    factory = Factory([ScriptedController(steps, reset_posterior=UNIFORM72)])
    logger = logging.getLogger(f"test.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    capture = _FieldsCapture()
    logger.addHandler(capture)
    runner = svc.ServiceRunner(session.path, eeg_type=eeg_type, marker_type=marker_type,
                               inference_type=inference_type, eeg_timeout=10,
                               logger=logger, controller_factory=factory,
                               consumer_wait_s=15, marker_probe_interval_s=0.2)
    thread, codes = _start_runner(runner)
    received = []
    try:
        inlet = _connect(inference_type)
        assert inlet.info(timeout=5).desc().child_value(INFERENCE_FINGERPRINT_KEY) == session.fingerprint
        assert _pull_until(inlet, received, _has(ev.SERVICE_STATUS, ev.STATE_READY), 15), received
        assert markers.wait_for_consumers(10)

        onset0 = local_clock() + 0.8
        markers.push_sample([json.dumps({"event": "trial_start", "trial_id": 0, "target_char_idx": 0,
                                         "target_char": "x"})], onset0 - 0.5)
        for i in range(4):
            markers.push_sample([json.dumps({"event": "flash", "trial_id": 0, "flash_idx": i,
                                             "group": [0, 1, 2, 3, 4, 5], "is_target": True})], onset0 + i * SOA)
        markers.push_sample([json.dumps({"event": "trial_end", "trial_id": 0})], onset0 + 4 * SOA)
        assert _pull_until(inlet, received, _has(ev.CHARACTER_COMMITTED), 15), received

        runner.request_stop()
        thread.join(timeout=10)
        assert codes == [svc.EXIT_OK]
        _pull_until(inlet, received, _has(ev.SERVICE_STATUS, ev.STATE_STOPPED), 5)
    finally:
        runner.request_stop()
        thread.join(timeout=10)
        eeg.stop()

    first_ready = next(i for i, e in enumerate(received) if _is_status(e, ev.STATE_READY))
    events = [e for i, e in enumerate(received) if i == first_ready or not _is_status(e, ev.STATE_READY)]
    summary = [(e["event"], e.get("state"), e.get("flash_idx"), e.get("reason")) for e in events]
    assert summary == [
        (ev.SERVICE_STATUS, ev.STATE_STARTING, None, None),
        (ev.SERVICE_STATUS, ev.STATE_READY, None, None),
        (ev.POSTERIOR_UPDATE, None, 0, None),
        (ev.POSTERIOR_UPDATE, None, 1, None),
        (ev.POSTERIOR_UPDATE, None, 2, None),
        (ev.CHARACTER_COMMITTED, None, None, ev.REASON_THRESHOLD_CROSSED),
        (ev.SERVICE_STATUS, ev.STATE_STOPPED, None, None),
    ]
    committed = events[5]
    assert (committed["char_idx"], committed["n_flashes_used"]) == (0, 3)
    assert all(e["manifest_fingerprint"] == session.fingerprint for e in events if e["event"] == ev.SERVICE_STATUS)
    assert [round(e["max_prob"], 6) for e in events[2:5]] == [0.3, 0.6, 0.95]

    # the experiment log, written by the real runner: config first, then the trial's records in order
    (config,) = capture.of("service_config")
    assert config["manifest_fingerprint"] == session.fingerprint
    assert config["eeg_stream"]["channel_count"] == 8 and config["eeg_stream"]["srate"] == SFREQ
    assert len(config["characters"]) == 72
    (opening,) = capture.of("trial_start")
    assert opening["lm_prior"] == [float(f"{1 / 72:.6g}")] * 72          # this manifest's LM is uniform
    assert opening["initial_posterior"] == pytest.approx(UNIFORM72, rel=1e-5)
    assert (opening["target_char_idx"], opening["target_prior_rank"]) == (0, 1)
    experiment_events = {"service_config", "trial_start", "flash_update", "flash_rejected", "char_committed"}
    kinds = [r["event"] for r in capture.records if r.get("event") in experiment_events]
    assert kinds == ["service_config", "trial_start", "flash_update", "flash_update", "flash_update",
                     "char_committed"]
    assert all(isinstance(r["score"], float) and math.isfinite(r["score"]) for r in capture.of("flash_update"))
    (committed_record,) = capture.of("char_committed")
    assert committed_record["correct"] is True and committed_record["reason"] == ev.REASON_THRESHOLD_CROSSED


def test_runner_heartbeat_generic_labels_and_eeg_silence(session):
    eeg_type, marker_type, inference_type = _types()
    generic = [f"EEG {i + 1}" for i in range(8)]
    eeg = EEGOutlet(eeg_type, labels=generic)
    eeg.start()
    runner = svc.ServiceRunner(session.path, eeg_type=eeg_type, marker_type=marker_type,
                               inference_type=inference_type, eeg_timeout=10, consumer_wait_s=15,
                               logger=logging.getLogger(f"test.{uuid.uuid4().hex}"))
    thread, codes = _start_runner(runner)
    early, late = [], []
    try:
        inlet = _connect(inference_type)
        assert _pull_until(inlet, early, _has(ev.SERVICE_STATUS, ev.STATE_READY), 15), early
        # generic labels: a warning naming them, then ready instead of exit 2
        label_warnings = [e for e in early if _is_status(e, ev.STATE_WARNING)]
        assert len(label_warnings) == 1
        assert "EEG 1" in label_warnings[0]["detail"] and "count and sfreq only" in label_warnings[0]["detail"]

        # a subscriber that arrives after the one-shot ready still gets one
        late_inlet = _connect(inference_type)
        assert _pull_until(late_inlet, late, _has(ev.SERVICE_STATUS, ev.STATE_READY), svc.READY_HEARTBEAT_S + 2)

        # EEG silence: one warning, no heartbeat while silent, ready again on recovery
        mark = len(early)
        eeg.paused.set()
        assert _pull_until(inlet, early, _has(ev.SERVICE_STATUS, ev.STATE_WARNING, "no EEG samples", mark), 6), early
        warned_at = len(early)
        _pull_until(inlet, early, lambda events: False, svc.READY_HEARTBEAT_S + 0.5)
        assert not any(_is_status(e, ev.STATE_READY) for e in early[warned_at:])
        eeg.paused.clear()
        assert _pull_until(inlet, early, _has(ev.SERVICE_STATUS, ev.STATE_READY, "resumed", mark), 5), early[mark:]
        assert sum("no EEG samples" in e.get("detail", "") for e in early) == 1

        runner.request_stop()
        thread.join(timeout=10)
        assert codes == [svc.EXIT_OK]
    finally:
        runner.request_stop()
        thread.join(timeout=10)
        eeg.stop()


@pytest.mark.skipif(not _real_reset_is_stub(), reason="BayesianStoppingController.reset() is implemented now")
def test_runner_exits_4_when_reset_is_a_stub(session):
    eeg_type, marker_type, inference_type = _types()
    eeg = EEGOutlet(eeg_type)
    eeg.start()
    markers = StreamOutlet(StreamInfo("MarkersTest", marker_type, 1, 0, "string", f"mk_{uuid.uuid4().hex}"))
    runner = svc.ServiceRunner(session.path, eeg_type=eeg_type, marker_type=marker_type,
                               inference_type=inference_type, eeg_timeout=10,
                               logger=logging.getLogger(f"test.{uuid.uuid4().hex}"),
                               consumer_wait_s=15, marker_probe_interval_s=0.2)
    thread, codes = _start_runner(runner)
    received = []
    try:
        inlet = _connect(inference_type)
        assert _pull_until(inlet, received, _has(ev.SERVICE_STATUS, ev.STATE_READY), 15)
        assert markers.wait_for_consumers(10)
        markers.push_sample([json.dumps({"event": "trial_start", "trial_id": 0, "target_char_idx": 0,
                                         "target_char": "x"})], local_clock())
        thread.join(timeout=10)
        assert codes == [svc.EXIT_CONTROLLER_NOT_IMPLEMENTED]
        assert "reset()" in runner.failure_message
        assert _pull_until(inlet, received, _has(ev.SERVICE_STATUS, ev.STATE_ERROR), 5)
        error = next(e for e in received if _is_status(e, ev.STATE_ERROR))
        assert "reset()" in error["detail"]
    finally:
        runner.request_stop()
        thread.join(timeout=10)
        eeg.stop()


def test_runner_exits_3_without_eeg(session):
    eeg_type, marker_type, inference_type = _types()
    runner = svc.ServiceRunner(session.path, eeg_type=eeg_type, marker_type=marker_type,
                               inference_type=inference_type, eeg_timeout=1.0, consumer_wait_s=0,
                               logger=logging.getLogger(f"test.{uuid.uuid4().hex}"))
    assert runner.run() == svc.EXIT_NO_EEG
    assert "no EEG stream" in runner.failure_message


@pytest.mark.parametrize("n_channels,labels,match", [
    (7, EEG_CHANNELS[:7], "channel count"),
    (8, list(reversed(EEG_CHANNELS)), "channels: checkpoint expects"),
    # The reference paper's own montage (CLAUDE.md hard constraints): a real,
    # different set of electrode names, used only as a mismatch example, not
    # this project's channel identity. This literal used to equal
    # EEG_CHANNELS's OLD value -- once EEG_CHANNELS was corrected to the real
    # montage this case silently stopped being a mismatch at all, so
    # check_stream_compatibility no longer raised and runner.run() hung
    # forever in its main loop instead of failing fast. Always pick this
    # montage independently of whatever EEG_CHANNELS currently is.
    (8, ["Fz", "F3", "C4", "Cz", "Pz", "P3", "O2", "O1"], "channels: checkpoint expects"),
])
def test_runner_exits_2_on_montage_mismatch(session, n_channels, labels, match):
    eeg_type, marker_type, inference_type = _types()
    eeg = EEGOutlet(eeg_type, n_channels=n_channels, labels=labels)
    eeg.start()
    try:
        runner = svc.ServiceRunner(session.path, eeg_type=eeg_type, marker_type=marker_type,
                                   inference_type=inference_type, eeg_timeout=10, consumer_wait_s=0,
                                   logger=logging.getLogger(f"test.{uuid.uuid4().hex}"))
        assert runner.run() == svc.EXIT_CONFIG
        assert match in runner.failure_message
    finally:
        eeg.stop()


def test_cli_subprocess_sigterm_exits_0(session, tmp_path):
    eeg_type, marker_type, inference_type = _types()
    eeg = EEGOutlet(eeg_type)
    eeg.start()
    log = tmp_path / "svc.log.jsonl"
    proc = subprocess.Popen(
        [sys.executable, "-m", "pipeline.inference_service", "--manifest", str(session.path),
         "--eeg-type", eeg_type, "--marker-type", marker_type, "--inference-type", inference_type,
         "--eeg-timeout", "20", "--log-file", str(log)],
        cwd=mf.REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    received = []
    try:
        inlet = _connect(inference_type, timeout=20)
        assert _pull_until(inlet, received, _has(ev.SERVICE_STATUS, ev.STATE_READY), 20), received
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 0
        _pull_until(inlet, received, _has(ev.SERVICE_STATUS, ev.STATE_STOPPED), 5)
        assert received[-1]["state"] == ev.STATE_STOPPED
        # liblsl writes its own connection logs to stderr; the service itself prints only on failure
        assert b"inference_service:" not in proc.stderr.read()
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert all(records)
        # the real CLI writes the opening config record, pinning which models produced this log
        (config,) = [r for r in records if r.get("event") == "service_config"]
        assert config["manifest_fingerprint"] == session.fingerprint
        assert config["eeg_stream"]["channel_count"] == 8 and len(config["characters"]) == 72
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
        eeg.stop()
