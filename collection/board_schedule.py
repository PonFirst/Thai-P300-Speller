"""
Timeline for the in-window stimulus board: the same trial structure and marker
JSON as checkerboard_stimulus.py's run_session()/run_trial(), as plain data, so
the Qt board only has to show each step and push its marker.

Per trial, relative to its cue onset:
    0                          cue on (target highlighted)      -> trial_start marker
    cue_ms                     cue off
    cue_ms + k*(flash+isi)     group k lit                      -> flash marker
      + flash_ms               group k off
    after the last ISI         trial_end marker
    + inter_selection_ms       next trial's cue, or the end of the session
"""

from __future__ import annotations

from dataclasses import dataclass

from flash_sequence import CheckerboardConfig, build_flash_sequence

CUE_ON = "cue_on"
CUE_OFF = "cue_off"
FLASH_ON = "flash_on"
FLASH_OFF = "flash_off"
TRIAL_END = "trial_end"
DONE = "done"

DEFAULT_CUE_MS = 2000.0
DEFAULT_INTER_SELECTION_MS = 3500.0


@dataclass(frozen=True)
class BoardStep:
    at_s: float
    kind: str
    trial_id: int | None = None
    target_idx: int | None = None
    group: tuple = ()
    marker: dict | None = None


def trial_start_marker(trial_id: int, target_idx: int, target_char: str) -> dict:
    return {"event": "trial_start", "trial_id": trial_id, "target_char_idx": target_idx,
            "target_char": target_char}


def flash_marker(trial_id: int, flash_idx: int, group, target_idx: int) -> dict:
    return {"event": "flash", "trial_id": trial_id, "flash_idx": flash_idx, "group": list(group),
            "is_target": target_idx in group}


def trial_end_marker(trial_id: int) -> dict:
    return {"event": "trial_end", "trial_id": trial_id}


def build_session_schedule(characters, targets: str, *, n_repetitions: int, flash_ms: float, isi_ms: float,
                           cue_ms: float = DEFAULT_CUE_MS,
                           inter_selection_ms: float = DEFAULT_INTER_SELECTION_MS,
                           flash_groups=build_flash_sequence) -> list[BoardStep]:
    """Every step of a session in time order, ending with a DONE step."""
    characters = list(characters)
    if not targets:
        raise ValueError("targets is empty")
    unknown = [c for c in targets if c not in characters]
    if unknown:
        raise ValueError(f"target string contains characters not in the matrix: {unknown}")
    if n_repetitions < 1:
        raise ValueError(f"n_repetitions must be >= 1, got {n_repetitions}")
    if flash_ms <= 0:
        raise ValueError(f"flash_ms must be > 0, got {flash_ms}")
    for name, value in (("isi_ms", isi_ms), ("cue_ms", cue_ms), ("inter_selection_ms", inter_selection_ms)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    steps = []
    t = 0.0
    for trial_id, char in enumerate(targets):
        target_idx = characters.index(char)
        steps.append(BoardStep(t, CUE_ON, trial_id, target_idx,
                               marker=trial_start_marker(trial_id, target_idx, char)))
        t += cue_ms / 1000.0
        steps.append(BoardStep(t, CUE_OFF, trial_id, target_idx))
        groups = flash_groups(CheckerboardConfig(n_chars=len(characters), n_repetitions=n_repetitions))
        for flash_idx, group in enumerate(groups):
            group = tuple(group)
            steps.append(BoardStep(t, FLASH_ON, trial_id, target_idx, group,
                                   marker=flash_marker(trial_id, flash_idx, group, target_idx)))
            steps.append(BoardStep(t + flash_ms / 1000.0, FLASH_OFF, trial_id, target_idx, group))
            t += (flash_ms + isi_ms) / 1000.0
        steps.append(BoardStep(t, TRIAL_END, trial_id, target_idx, marker=trial_end_marker(trial_id)))
        t += inter_selection_ms / 1000.0
    steps.append(BoardStep(t, DONE))
    return steps
