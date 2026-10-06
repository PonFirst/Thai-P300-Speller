"""
What a BCI2000 CheckerboardSpeller session did, read back from its recording: every character's cue,
language-model prior, classifier results, stopping decision and selection -- in the record format
the Python inference service logs (pipeline/experiment_log.py), so stopping/summarize_experiment.py
summarizes a BCI2000 session with the same code as a Python-stack one.

Where each piece comes from:
  .dat states    StimulusCodeRes / ClassifierScore / ClassifierRejectedCode (CheckerboardSignalProcessing),
                 FlashChar1-6, AttendedChar, SelectedChar, PosteriorMax, StoppingFlashes, StopReason,
                 SpellerRunID (CheckerboardSpeller); the flash onsets and selections as in
                 pipeline/bci2000_recording.py
  .dat params    TargetDefinitions, TextResult, StoppingRule and the stopping values, LanguageModel
  .applog        each character's full prior ("PRIOR <selection> <72 values>"), after the "RUN <id>" line
                 whose id is this recording's SpellerRunID (BCI2000 keeps one .applog per session folder,
                 shared by all its runs)

The controller's input order is the order the speller received the results, so a selection's first
StoppingFlashes results are the ones it used; results that arrived after it stopped are not records.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pipeline import events as ev
from pipeline.bci2000_dat import DatError, DatFile
from pipeline.bci2000_recording import flash_events

STOP_REASONS = {1: ev.REASON_THRESHOLD_CROSSED, 2: ev.REASON_MAX_FLASHES_REACHED, 3: ev.REASON_TRIAL_ENDED}
MEAN_RULE_REASON = "mean_score_all_flashes"   # StoppingRule 0: no stopping, the best mean score after all flashes
REJECTED_REASON = "classifier_rejected"       # the filter's ClassifierRejectedCode: lost samples, gate, rank


class SessionError(ValueError):
    """The recording is not a CheckerboardSpeller session this module can read."""


def characters(dat: DatFile) -> list[str]:
    return [row[1] for row in dat.param("TargetDefinitions").value]


def selection_of_samples(dat: DatFile) -> np.ndarray:
    """0-based selection index of every sample (-1 before the first); selections start when
    PhaseInSequence enters the pre-sequence phase, as pipeline/bci2000_recording.flash_events."""
    phase = dat.state("PhaseInSequence")
    starts = (phase == 1) & (np.r_[0, phase[:-1]] != 1)
    return np.cumsum(starts) - 1


def result_events(dat: DatFile) -> tuple[list, list]:
    """(results, rejections) in arrival order. results: (sample, code, score); rejections: (sample, code).
    The classifier sets its states once per sample block, so each block holds at most one of each."""
    block = int(dat.param_number("SampleBlockSize"))
    starts = np.arange(0, dat.n_samples, block)
    code = dat.state("StimulusCodeRes")[starts].astype(int)
    score = dat.state_float("ClassifierScore")[starts].astype(float)
    rej = dat.state("ClassifierRejectedCode")[starts].astype(int) if "ClassifierRejectedCode" in dat.states \
        else np.zeros(len(starts), dtype=int)
    results = [(int(starts[i]), int(code[i]), float(score[i])) for i in np.flatnonzero(code > 0)]
    rejections = [(int(starts[i]), int(rej[i])) for i in np.flatnonzero(rej > 0)]
    return results, rejections


def run_id(dat: DatFile) -> int | None:
    if "SpellerRunID" not in dat.states:
        return None
    ids = np.unique(dat.state("SpellerRunID"))
    ids = ids[ids > 0]
    return int(ids[0]) if len(ids) == 1 else None


def find_applog(dat_path) -> Path | None:
    logs = sorted(Path(dat_path).parent.glob("*.applog"))
    return logs[0] if len(logs) == 1 else None


def logged_priors(dat: DatFile, applog_text: str | None) -> dict:
    """{0-based selection: prior array} from this run's section of the speller's .applog."""
    rid = run_id(dat)
    if rid is None or not applog_text:
        return {}
    priors, inside = {}, False
    for line in applog_text.splitlines():
        m = re.match(r"\s*RUN (\d+)\s*$", line)
        if m:
            inside = int(m.group(1)) == rid
            continue
        m = re.match(r"\s*PRIOR (\d+) (.*)$", line)
        if inside and m:
            priors[int(m.group(1)) - 1] = np.array(m.group(2).split(), dtype=float)
    return priors


def seconds_per_flash(dat: DatFile) -> float | None:
    """Median gap between consecutive flash onsets within a selection, in seconds."""
    fs = dat.param_number("SamplingRate", "Hz")
    by_sel = {}
    for f in flash_events(dat):
        by_sel.setdefault(f.selection, []).append(f.onset)
    gaps = [b - a for onsets in by_sel.values() for a, b in zip(onsets, onsets[1:])]
    return float(np.median(gaps)) / fs if gaps else None


def session_records(dat: DatFile, applog_text: str | None = None) -> list[dict]:
    """
    The session as inference-log records: one service_config, then per selection a trial_start,
    its flash_update / flash_rejected records and a char_committed (absent if nothing was selected).
    """
    try:
        chars = characters(dat)
        flashes = flash_events(dat)
    except DatError as exc:
        raise SessionError(str(exc)) from exc
    n = len(chars)
    p = dat.param_str
    bayesian = "StoppingRule" in dat.params and p("StoppingRule") == "1"
    lm_type = "transformer" if "LanguageModel" in dat.params and p("LanguageModel") == "1" else "uniform"
    stopping = ({"lam": float(p("StoppingLambda")), "theta": float(p("StoppingTheta")),
                 "t_min": int(p("StoppingTMin")), "t_max": int(p("StoppingTMax")),
                 "llr_clip": float(p("StoppingLLRClip"))} if bayesian else None)
    records = [{
        "event": "service_config", "source": "bci2000", "recording": dat.path.name,
        "manifest_fingerprint": p("RiemannModelID") if "RiemannModelID" in dat.params else None,
        "manifest": {"language_model": {"type": lm_type}, "stopping": stopping},
        "stopping_rule": "bayesian" if bayesian else "mean_score", "characters": chars,
    }]
    priors = logged_priors(dat, applog_text)
    if bayesian and lm_type == "uniform":
        priors = {s: priors.get(s, np.full(n, 1 / n)) for s in set(priors) | {f.selection for f in flashes}}

    sel = selection_of_samples(dat)
    groups = {(f.selection, f.code): f.group for f in flashes}
    presented = {}
    for f in flashes:
        presented[f.selection] = presented.get(f.selection, 0) + 1
    results, rejections = result_events(dat)
    attended = dat.state("AttendedChar").astype(int)
    selected = dat.state("SelectedChar").astype(int)
    used_state = dat.state("StoppingFlashes").astype(int) if "StoppingFlashes" in dat.states else None
    reason_state = dat.state("StopReason").astype(int) if "StopReason" in dat.states else None
    pmax = dat.state_float("PosteriorMax") if "PosteriorMax" in dat.states else None

    context = p("TextResult") if "TextResult" in dat.params else ""
    for s in range(int(sel.max()) + 1 if dat.n_samples else 0):
        mask = sel == s
        if not mask.any():
            continue
        target = int(attended[mask].max()) - 1
        target = target if target >= 0 else None
        prior = priors.get(s)
        start = {"event": "trial_start", "trial_id": s, "context": context,
                 "target_char_idx": target, "target_char": chars[target] if target is not None else None}
        if prior is not None and len(prior) == n:
            start["lm_prior"] = [float(v) for v in prior]
            if target is not None:
                start["target_prior_prob"] = float(prior[target])
                start["target_prior_rank"] = int(1 + np.sum(prior > prior[target]))
        records.append(start)

        sel_results = [(smp, code, score) for smp, code, score in results if sel[smp] == s]
        n_used = int(used_state[mask].max()) if bayesian and used_state is not None else len(sel_results)
        for t, (_, code, score) in enumerate(sel_results[:n_used], start=1):
            group = groups.get((s, code), ())
            records.append({"event": "flash_update", "trial_id": s, "flash_idx": code, "t": t,
                            "group": list(group), "score": score, "pi": len(group) / n})
        for _, code in (r for r in rejections if sel[r[0]] == s):
            records.append({"event": "flash_rejected", "trial_id": s, "flash_idx": code, "reason": REJECTED_REASON})

        choice = int(selected[mask].max()) - 1
        if choice >= 0:
            reason = (STOP_REASONS.get(int(reason_state[mask].max())) if bayesian and reason_state is not None
                      else MEAN_RULE_REASON)
            records.append({
                "event": "char_committed", "trial_id": s, "char_idx": choice, "char": chars[choice],
                "reason": reason, "n_flashes_used": n_used, "n_flashes_presented": presented.get(s, 0),
                "max_prob": float(pmax[mask][-1]) if bayesian and pmax is not None else None,
                "target_char_idx": target, "target_char": chars[target] if target is not None else None,
                "correct": (choice == target) if target is not None else None,
            })
            context += chars[choice]
    return records
