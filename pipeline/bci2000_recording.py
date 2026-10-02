"""
A CheckerboardSpeller calibration recording (BCI2000 .dat) as training data: continuous EEG in
microvolts, and every flash with its onset sample, selection and target label.

The flash convention is the one the online filter uses (bci2000/CheckerboardSignalProcessing/
RiemannP300Classifier.h), so training and serving cut identical epochs:
  - a flash's onset is the first sample whose StimulusCode is nonzero and differs from the
    previous sample's (StimulusCode is 0 between flashes);
  - its label is StimulusType at that sample (1: the flash contained the cued character);
  - its selection (the CV group) counts pre-sequence phases (PhaseInSequence entering 1).
Samples the source filled in for lost Bluetooth packets (Validation = 0) become NaN, and so does
the first sample after any break in the device's sample counter (a counter reset, or blocks
BCI2000 dropped): any epoch whose padded window touches one is rejected, exactly as the online
filter rejects it.

check_recording() refuses what training on would be wrong rather than merely noisy: a montage or
sampling rate other than pipeline.constants', a source-side filter that the online path would
not see the same way, no cued characters (free spelling has no labels), or labels that disagree
with the recorded flash groups (a sign that states and samples are misaligned).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.bci2000_dat import DatError, DatFile
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ

FLASH_STATES = tuple(f"FlashChar{i}" for i in range(1, 7))
REQUIRED_STATES = ("StimulusCode", "StimulusType", "PhaseInSequence", "AttendedChar", *FLASH_STATES)
# Signal-changing source-side settings: the model is fitted on the plain signal, and the online
# filter refuses the same settings. (SourceFilter runs before the file writer, so it would change the
# recording too; AlignChannels changes only what signal processing sees.)
SOURCE_FILTER_PARAMS = ("NotchFilter", "HighPassFilter", "LowPassFilter", "AlignChannels")


class RecordingError(ValueError):
    """The .dat file cannot be used as calibration data."""


@dataclass(frozen=True)
class DatFlash:
    selection: int          # 0-based, the CV group
    code: int               # StimulusCode: the flash's number within its selection
    onset: int              # sample index within the recording
    is_target: bool
    group: tuple            # 0-based character indices lit by the flash
    attended: int           # 0-based cued character index


def check_recording(dat: DatFile) -> None:
    name = dat.path.name
    missing = [s for s in REQUIRED_STATES if s not in dat.states]
    if missing:
        raise RecordingError(f"{name}: no state(s) {missing}; record calibration with CheckerboardSpeller")
    try:
        channels = dat.param_list("ChannelNames")
        fs = dat.param_number("SamplingRate", "Hz")
    except DatError as exc:
        raise RecordingError(str(exc)) from exc
    if channels != list(EEG_CHANNELS):
        raise RecordingError(f"{name}: ChannelNames are {channels}; the classifier needs {list(EEG_CHANNELS)} in "
                             "that order (CLAUDE.md: the verified montage)")
    if fs != EEG_SFREQ:
        raise RecordingError(f"{name}: SamplingRate {fs:g} Hz, expected {EEG_SFREQ:g} Hz")
    for p in SOURCE_FILTER_PARAMS:
        if p in dat.params:
            value = dat.param_str(p)
            try:
                off = value == "" or float(value) == 0
            except ValueError:
                off = False
            if not off:
                raise RecordingError(f"{name}: {p} = {value}; record with {p} = 0")


def flash_events(dat: DatFile) -> list[DatFlash]:
    """Every flash in recording order. Raises RecordingError if the labels contradict the groups."""
    check_recording(dat)
    code = dat.state("StimulusCode").astype(np.int64)
    stype = dat.state("StimulusType")
    phase = dat.state("PhaseInSequence")
    attended = dat.state("AttendedChar").astype(np.int64)
    lit = np.stack([dat.state(s).astype(np.int64) for s in FLASH_STATES], axis=1)

    prev = np.r_[0, code[:-1]]
    onsets = np.flatnonzero((code > 0) & (code != prev))
    pre_start = (phase == 1) & (np.r_[0, phase[:-1]] != 1)
    selection_of = np.cumsum(pre_start) - 1

    flashes = []
    for i in onsets:
        sel = int(selection_of[i])
        if sel < 0:
            raise RecordingError(f"{dat.path.name}: a flash at sample {i} comes before any selection started "
                                 "(no PhaseInSequence pre-sequence phase was recorded: PreSequenceDuration must be "
                                 "at least one block, for the cue)")
        group = tuple(int(c) - 1 for c in lit[i] if c > 0)
        att = int(attended[i]) - 1
        if att < 0:
            raise RecordingError(f"{dat.path.name}: no cued character at sample {i}; calibration needs copy mode "
                                 "(InterpretMode 2) with TextToSpell")
        if not group:
            raise RecordingError(f"{dat.path.name}: flash at sample {i} has no FlashChar states set")
        is_target = bool(stype[i])
        if is_target != (att in group):
            raise RecordingError(f"{dat.path.name}: at sample {i} StimulusType is {int(stype[i])} but the cued "
                                 f"character {'is' if att in group else 'is not'} in the flash group; states "
                                 "and samples disagree, refusing to train on it")
        flashes.append(DatFlash(selection=sel, code=int(code[i]), onset=int(i), is_target=is_target,
                                group=group, attended=att))
    return flashes


def counter_breaks(dat: DatFile) -> np.ndarray:
    """Indices of samples whose device Counter does not follow the previous sample's by exactly 1."""
    if "Counter" not in dat.states or dat.n_samples < 2:
        return np.zeros(0, dtype=np.int64)
    c = dat.state("Counter").astype(np.int64)
    return np.flatnonzero(np.diff(c) != 1) + 1


def invalid_samples(dat: DatFile) -> np.ndarray:
    """Boolean mask: samples filled in for lost packets (Validation = 0) or right after a counter break.
    The online filter (RiemannP300Classifier) rejects a window containing any of them."""
    bad = np.zeros(dat.n_samples, dtype=bool)
    if "Validation" in dat.states:
        bad |= dat.state("Validation") == 0
    bad[counter_breaks(dat)] = True
    return bad


def eeg_microvolts(dat: DatFile) -> np.ndarray:
    """(n_samples, 8) float64; NaN on invalid_samples()."""
    x = np.array(dat.signal_uv, dtype=np.float64, copy=True)
    x[invalid_samples(dat)] = np.nan
    return x
