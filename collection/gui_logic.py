"""
Pure logic for the Qt session control panel (collection/session_gui_qt.py).

Nothing here imports Qt, pylsl, or starts a process, so all of it is
unit-tested in tests/test_gui_logic.py without a display, a headset, or LSL:

  - settings validation and recording-path generation
  - the EEG connection state machine (searching / connected / lost)
  - trace normalization, the scalp-map geometry, quality colours
  - the P300Inference display model (composed text, gauge, counters,
    manifest-mismatch and malformed-event handling)
  - human-readable exit codes and the session start/stop sequence, driven
    by an injected process launcher and clock
"""

from __future__ import annotations

import os
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECTION_DIR = REPO_ROOT / "collection"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import events as ev  # noqa: E402
from pipeline import experiment_log as xlog  # noqa: E402
from pipeline import quality  # noqa: E402
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ  # noqa: E402
from pipeline.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH, ManifestError, ManifestMismatchError, assert_same_fingerprint, load_manifest,
    short_fingerprint,
)

CHARACTERS_PATH = COLLECTION_DIR / "characters_th.json"
RECORDINGS_DIR = COLLECTION_DIR / "recordings"

# ----------------------------------------------------------------------
# Display constants (ported unchanged from the Tkinter panel)
# ----------------------------------------------------------------------

# Scalp positions, top-down, nose at +y, laid out to fill brain_hemisphere()'s
# silhouette rather than to sit on a unit-radius 10-20 circle (which only
# covers the middle of the drawing and left the dots huddled around the
# fissure).
#
# Real montage: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 -- the Unicorn Hybrid
# Black's factory-default cap (see pipeline/constants.EEG_CHANNELS for the
# verification sources; CLAUDE.md's hard constraints record the correction).
# This positions the dots to match real relative 10-20 geometry, not just to
# match the names:
#   - Four rows, evenly spaced front-to-back (0.60 / 0.12 / -0.36 / -0.84):
#     Fz alone; C3/Cz/C4 on one coronal line; Pz alone, more posterior;
#     Oz at the occipital pole, the most posterior point, on the midline.
#   - PO7/PO8 sit between the Pz and Oz rows (y=-0.68), and more lateral
#     than C3/C4 (|x|=0.58 vs 0.48) -- parieto-occipital sites are more
#     lateral than central ones in the real 10-20 layout, not a mirror of
#     the central row.
#   - All eight checked inside brain_hemisphere()'s outline with
#     matplotlib.path.Path against the actual rendered polygon, not
#     eyeballed.
ELECTRODE_POS = {
    "Fz": (0.00, 0.60),
    "C3": (-0.48, 0.12),
    "Cz": (0.00, 0.12),
    "C4": (0.48, 0.12),
    "Pz": (0.00, -0.36),
    "PO7": (-0.58, -0.68),
    "Oz": (0.00, -0.84),
    "PO8": (0.58, -0.68),
}

DISPLAY_ORDER = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

DISPLAY_WINDOW_S = 4.0
QUALITY_WINDOW_S = 1.0
REFRESH_MS = 100
STREAM_LOST_AFTER_S = 1.0

COLOR_BG = "#fafafa"
COLOR_FG = "#1a1a1a"
COLOR_MUTED = "#6b7280"
COLOR_TRACE = "#374151"
COLOR_BRAIN = "#d6dade"
QUALITY_GREEN = "#22c55e"
QUALITY_YELLOW = "#eab308"
QUALITY_RED = "#ef4444"
# Darker than the Tk panel's #d1d5db, which was nearly invisible on the
# #d6dade brain fill. Gray now carries meaning (unknown), so it must show.
QUALITY_GRAY = "#9ca3af"
COLOR_FALLBACK = "#c2410c"
COLOR_ERROR = "#b91c1c"

QUALITY_COLORS = {
    quality.GREEN: QUALITY_GREEN,
    quality.YELLOW: QUALITY_YELLOW,
    quality.RED: QUALITY_RED,
    quality.UNKNOWN: QUALITY_GRAY,
}

QUALITY_LEGEND = [
    (QUALITY_GREEN, "Good signal"),
    (QUALITY_YELLOW, "Noisy"),
    (QUALITY_RED, "Bad / flat"),
    (QUALITY_GRAY, "No data"),
]

THAI_FONT_PREFERENCE = ("Sarabun", "Thonburi")


def quality_color(level: str) -> str:
    """Quality level from pipeline.quality -> dot colour. Anything unrecognised is gray."""
    return QUALITY_COLORS.get(level, QUALITY_GRAY)


def pick_thai_family(available: Iterable[str],
                     preference: Iterable[str] = THAI_FONT_PREFERENCE) -> str | None:
    """First preferred family that is installed, or None (use the system default)."""
    installed = set(available)
    for family in preference:
        if family in installed:
            return family
    return None


# ----------------------------------------------------------------------
# Traces and scalp map
# ----------------------------------------------------------------------

def normalize_trace(y, offset: float) -> np.ndarray:
    """
    Per-channel normalization so all 8 traces stay visible on one shared
    axis: (y - mean) / (std * 4) + offset, std floored at 1e-6 so a flat
    channel draws as a straight line instead of dividing by zero.
    """
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return arr
    centered = arr - arr.mean()
    spread = max(float(np.std(centered)), 1e-6)
    return centered / (spread * 4.0) + offset


def brain_hemisphere(side: int, gap: float = 0.05, r_x: float = 0.95,
                     r_y: float = 1.0, n_points: int = 400) -> np.ndarray:
    """
    Outline of one hemisphere as a lobed half-ellipse (exact port of the Tk
    panel's _brain_hemisphere): the radius is modulated by 1 + 0.035 sin(11 t)
    to suggest gyri, and the inner edge is offset by `gap` so the two
    hemispheres leave a longitudinal fissure. side: +1 right, -1 left.
    Returns (n_points, 2) world coordinates, +y toward the nose.
    """
    t = np.linspace(0, np.pi, n_points)
    bump = 1 + 0.035 * np.sin(11 * t)
    x = side * (gap + r_x * np.sin(t) * bump)
    y = r_y * np.cos(t) * bump
    return np.column_stack([x, y])


class SampleBuffer:
    """
    Rolling (n_samples, n_channels) buffer of the most recent EEG, in
    EEG_CHANNELS column order. Only touched from the GUI thread.
    """

    def __init__(self, n_channels: int = len(EEG_CHANNELS), sfreq: float = EEG_SFREQ,
                 seconds: float = DISPLAY_WINDOW_S, extra: int = 50):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.capacity = int(seconds * sfreq) + extra
        self._data = np.empty((0, n_channels))

    def __len__(self):
        return self._data.shape[0]

    def clear(self):
        self._data = np.empty((0, self.n_channels))

    def append(self, chunk) -> int:
        """Append a chunk; short rows are NaN-padded, extra columns dropped. Returns rows added."""
        arr = np.asarray(chunk, dtype=float)
        if arr.size == 0:
            return 0
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < self.n_channels:
            pad = np.full((arr.shape[0], self.n_channels - arr.shape[1]), np.nan)
            arr = np.hstack([arr, pad])
        elif arr.shape[1] > self.n_channels:
            arr = arr[:, :self.n_channels]
        self._data = np.vstack([self._data, arr])[-self.capacity:]
        return arr.shape[0]

    def latest(self, n: int | None = None) -> np.ndarray:
        return self._data if n is None else self._data[-n:]


def display_traces(buffer: SampleBuffer, window_s: float = DISPLAY_WINDOW_S) -> dict:
    """channel -> (x seconds, normalized y) for each channel in DISPLAY_ORDER, row i at offset -i."""
    data = buffer.latest(int(window_s * buffer.sfreq))
    out = {}
    for i, ch in enumerate(DISPLAY_ORDER):
        column = data[:, EEG_CHANNELS.index(ch)] if data.size else np.empty(0)
        x = np.linspace(0, column.size / buffer.sfreq, column.size)
        out[ch] = (x, normalize_trace(column, -i))
    return out


def display_isolated(char: str) -> str:
    """A character shown on its own: combining marks get a dotted-circle carrier (◌่) so they stay visible."""
    import unicodedata
    if char and unicodedata.category(char[0]) in ("Mn", "Mc", "Me"):
        return "◌" + char
    return char


def scalp_levels(buffer: SampleBuffer, connected: bool,
                 window_s: float = QUALITY_WINDOW_S) -> dict:
    """
    channel -> quality level over the last `window_s`, via
    pipeline.quality.channel_levels (the same function the service's quality
    gate uses). A stream that is not currently delivering reports unknown for
    every channel: stale data must not keep a dot green.
    """
    if not connected or len(buffer) == 0:
        return {ch: quality.UNKNOWN for ch in EEG_CHANNELS}
    levels = quality.channel_levels(buffer.latest(int(window_s * buffer.sfreq)))
    return dict(zip(EEG_CHANNELS, levels))


# ----------------------------------------------------------------------
# Settings and recording paths
# ----------------------------------------------------------------------

DEMO_PREFIX = "demo_"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SESSION_ID_MAX_LEN = 64

DEFAULT_SETTINGS = {
    "session_id": "subject_001",
    "repetitions": 5,
    "flash_ms": 62,
    "isi_ms": 62,
    "screen": 0,
    "windowed": False,
}

BOUNDS = {
    "repetitions": (1, 50),
    "flash_ms": (10, 500),
    "isi_ms": (10, 500),
    "screen": (0, 3),
}

_LABELS = {"repetitions": "Repetitions", "flash_ms": "Flash ms", "isi_ms": "ISI ms", "screen": "Screen"}


@dataclass(frozen=True)
class SessionSettings:
    session_id: str
    targets: str
    repetitions: int
    flash_ms: int
    isi_ms: int
    screen: int
    windowed: bool


def validate_session_id(session_id) -> str | None:
    """Error message, or None if the session ID is usable."""
    if not isinstance(session_id, str) or not session_id.strip():
        return "Session ID is empty."
    sid = session_id.strip()
    if len(sid) > SESSION_ID_MAX_LEN:
        return f"Session ID is longer than {SESSION_ID_MAX_LEN} characters."
    if not SESSION_ID_RE.match(sid):
        return "Session ID may only contain letters, digits, '_' and '-'."
    # "demo" alone is reserved too: recording_path appends "_<timestamp>", so
    # it would produce demo_<timestamp>.csv for a real session.
    if (sid + "_").lower().startswith(DEMO_PREFIX):
        return (f"Session ID must not be 'demo' or start with '{DEMO_PREFIX}': that prefix is "
                "reserved for synthetic demo recordings.")
    return None


def _parse_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def validate_targets(targets, characters: list[str]) -> str | None:
    if not isinstance(targets, str) or not targets.strip():
        return "Type the characters to spell first."
    allowed = set(characters)
    unknown = []
    for c in targets.strip():
        if c not in allowed and c not in unknown:
            unknown.append(c)
    if unknown:
        shown = " ".join("(space)" if c.isspace() else c for c in unknown)
        return f"Not in the {len(characters)}-character matrix: {shown}"
    return None


def validate_settings(*, session_id, targets, repetitions, flash_ms, isi_ms, screen,
                      windowed, characters: list[str]) -> tuple[SessionSettings | None, list[str]]:
    """Returns (settings, []) when valid, else (None, [error, ...])."""
    errors = []
    for message in (validate_targets(targets, characters), validate_session_id(session_id)):
        if message:
            errors.append(message)
    numbers = {}
    for key, raw in (("repetitions", repetitions), ("flash_ms", flash_ms),
                     ("isi_ms", isi_ms), ("screen", screen)):
        lo, hi = BOUNDS[key]
        value = _parse_int(raw)
        if value is None or not lo <= value <= hi:
            errors.append(f"{_LABELS[key]} must be a whole number from {lo} to {hi}, got {raw!r}.")
        numbers[key] = value
    if errors:
        return None, errors
    return SessionSettings(session_id=session_id.strip(), targets=targets.strip(),
                           windowed=bool(windowed), **numbers), []


def recording_path(session_id: str, *, demo: bool, now: datetime,
                   recordings_dir=RECORDINGS_DIR,
                   exists: Callable[[Path], bool] = os.path.exists) -> Path:
    """
    <recordings>/[demo_]<session_id>_<YYYYmmdd_HHMMSS>.csv, never an existing
    path: a collision gets _1, _2, ... appended. The recorder refuses to
    overwrite (exit 2), so this keeps two Starts within one second apart.
    """
    prefix = DEMO_PREFIX if demo else ""
    stem = f"{prefix}{session_id}_{now.strftime('%Y%m%d_%H%M%S')}"
    directory = Path(recordings_dir)
    candidate = directory / f"{stem}.csv"
    n = 1
    while exists(candidate):
        candidate = directory / f"{stem}_{n}.csv"
        n += 1
    return candidate


# ----------------------------------------------------------------------
# EEG connection state machine
# ----------------------------------------------------------------------

SEARCHING = "searching"
CONNECTED = "connected"
LOST = "lost"

STATUS_TEXT = {
    SEARCHING: "● No EEG stream — start the Unicorn LSL Interface (Windows) or unicorn_relay.py (Mac)",
    CONNECTED: "● EEG connected",
    LOST: "● EEG stream lost",
}
STATUS_COLOR = {SEARCHING: QUALITY_RED, CONNECTED: QUALITY_GREEN, LOST: QUALITY_RED}


class EegConnection:
    """
    searching --stream resolved--> connected --no samples for > lost_after_s--> lost
    lost --samples arrive--> connected

    A stream counts as delivering from the moment it is resolved, so a
    resolved stream that never sends anything turns lost after lost_after_s.
    Once a stream has been seen, dropping the inlet keeps the state at lost
    (not searching): "EEG stream lost" is the more useful message. Start is
    allowed only while connected. Timestamps are injected (seconds, monotonic).
    """

    def __init__(self, lost_after_s: float = STREAM_LOST_AFTER_S):
        self.lost_after_s = lost_after_s
        self.state = SEARCHING
        self.last_activity: float | None = None
        self.stream_name: str | None = None

    def on_stream_found(self, now: float, name: str | None = None) -> str:
        self.stream_name = name
        self.last_activity = now
        self.state = CONNECTED
        return self.state

    def on_samples(self, now: float, n_samples: int = 1) -> str:
        if n_samples > 0:
            self.last_activity = now
            self.state = CONNECTED
        return self.state

    def on_stream_dropped(self, now: float) -> str:
        if self.state != SEARCHING:
            self.state = LOST
        return self.state

    def tick(self, now: float) -> str:
        if (self.state == CONNECTED and self.last_activity is not None
                and now - self.last_activity > self.lost_after_s):
            self.state = LOST
        return self.state

    @property
    def start_allowed(self) -> bool:
        return self.state == CONNECTED

    @property
    def status_text(self) -> str:
        return STATUS_TEXT[self.state]

    @property
    def status_color(self) -> str:
        return STATUS_COLOR[self.state]


# ----------------------------------------------------------------------
# Manifest availability
# ----------------------------------------------------------------------

MANIFEST_MISSING_MESSAGE = (
    "Inference unavailable: session manifest not found — record a calibration "
    "session and run pipeline.fit_from_recording")


DEMO_MANIFEST_MISSING_MESSAGE = (
    "Inference unavailable: demo session manifest not found — record a demo session and run "
    "pipeline.fit_from_recording --allow-demo")


def default_manifest_path(demo: bool, repo_root=REPO_ROOT) -> Path:
    """
    session_manifest.json at the repo root, or demo_session_manifest.json in
    demo mode (the fit script prefixes every demo artifact with demo_).
    Independent of the working directory. A demo panel never defaults to a
    real model, and a real panel never defaults to a synthetic one.
    """
    name = DEFAULT_MANIFEST_PATH.name
    return Path(repo_root) / (f"{DEMO_PREFIX}{name}" if demo else name)


def resolve_manifest_arg(arg, demo: bool) -> Path:
    """--manifest as given (relative to the caller's cwd), or the mode's default when omitted."""
    return default_manifest_path(demo) if arg is None else Path(arg).expanduser()


def load_manifest_for_gui(path, loader, demo: bool = False) -> tuple[object | None, str | None]:
    """
    (manifest, None) or (None, reason shown next to the disabled Live
    inference toggle). Never raises: calibration sessions must work without
    a manifest.
    """
    if path is None or not Path(path).expanduser().is_file():
        return None, DEMO_MANIFEST_MISSING_MESSAGE if demo else MANIFEST_MISSING_MESSAGE
    try:
        return loader(path), None
    except ManifestError as exc:
        return None, f"Inference unavailable: {exc}"
    except Exception as exc:  # a bug in a loader must not stop recording
        return None, f"Inference unavailable: could not load {path} ({type(exc).__name__}: {exc})"


# ----------------------------------------------------------------------
# P300Inference display model
# ----------------------------------------------------------------------

FALLBACK_EXPLANATION = {
    ev.REASON_MAX_FLASHES_REACHED: "forced at the flash limit (t_max) before reaching the threshold",
    ev.REASON_TRIAL_ENDED: "the stimulus ended the trial before the threshold was reached",
    ev.REASON_THRESHOLD_CROSSED: "posterior crossed the threshold",
}


@dataclass(frozen=True)
class CommittedChar:
    trial_id: int
    char_idx: int
    char: str
    n_flashes_used: int
    reason: str

    @property
    def is_fallback(self) -> bool:
        return ev.is_fallback(self.reason)

    def describe(self) -> str:
        kind = "FALLBACK" if self.is_fallback else "confident"
        return (f"{self.char}  trial {self.trial_id}: {self.reason} after {self.n_flashes_used} "
                f"flashes ({kind}: {FALLBACK_EXPLANATION[self.reason]})")


@dataclass(frozen=True)
class GaugeState:
    trial_id: int
    flash_idx: int
    max_prob: float
    theta: float
    leading_char_idx: int
    leading_char: str


class InferenceDisplayModel:
    """
    Applies P300Inference events to what the panel shows. Pure state: the Qt
    side feeds it payload strings and reads the attributes back.

    Manifest check: the stream's desc() fingerprint is authoritative and must
    equal this panel's own (pipeline.manifest.assert_same_fingerprint; an
    absent desc fingerprint is also a mismatch). A service_status may carry
    null; one carrying a different fingerprint contradicts the desc and is
    also treated as a mismatch. Once a stream mismatches, nothing more from
    it is displayed until a new stream connects.

    Readiness: LSL does not replay samples to late subscribers, so the
    service repeats `ready` as a heartbeat and any `ready` counts -- seeing
    `starting` is not required. It must arrive on a stream connected after
    the session started (reset_session), so a `ready` still buffered from the
    previous session's service cannot start the next one.
    """

    def __init__(self, expected_fingerprint: str | None, characters: list[str] | None = None):
        self.expected_fingerprint = expected_fingerprint
        self.characters = list(characters) if characters is not None else None
        self.malformed_count = 0
        self.last_malformed_error: str | None = None
        self.stream_connected = False
        self.stream_fingerprint_ok = False
        self.stream_serial = 0
        self.mismatch_message: str | None = None
        self.ignored_after_mismatch = 0
        self.reset_session()

    # -- lifecycle -------------------------------------------------------
    def reset_session(self):
        """Clear what one session displays (called on Start). Keeps the stream connection facts."""
        self.committed: list[CommittedChar] = []
        self._committed_trials: set[int] = set()
        self.gauge: GaugeState | None = None
        self.rejected_by_reason = {reason: 0 for reason in ev.REJECT_REASONS}
        self.last_rejected_channels: list[str] = []
        self.service_state: str | None = None
        self.service_detail = ""
        self.stale_updates_ignored = 0
        self.duplicate_commits_ignored = 0
        self._ready_serial: int | None = None
        self._session_serial = self.stream_serial

    def on_stream_connected(self, desc_fingerprint: str | None):
        self.stream_serial += 1
        self._ready_serial = None
        self.stream_connected = True
        self.mismatch_message = None
        self.ignored_after_mismatch = 0
        self._committed_trials = set()
        self.gauge = None
        self.stream_fingerprint_ok = self._check_fingerprint(desc_fingerprint, "P300Inference stream desc")

    def on_stream_lost(self):
        self.stream_connected = False
        self.stream_fingerprint_ok = False

    # -- derived -----------------------------------------------------------
    @property
    def mismatch(self) -> bool:
        return self.mismatch_message is not None

    @property
    def composed_text(self) -> str:
        return "".join(c.char for c in self.committed)

    @property
    def rejected_quality(self) -> int:
        return self.rejected_by_reason[ev.REJECT_QUALITY_GATE]

    @property
    def rejected_other(self) -> int:
        return sum(n for r, n in self.rejected_by_reason.items() if r != ev.REJECT_QUALITY_GATE)

    @property
    def service_ready(self) -> bool:
        """
        A `ready` (first or heartbeat) on the current stream, which connected
        after the session started and whose desc fingerprint matched.
        """
        return (self.stream_connected and self.stream_fingerprint_ok and not self.mismatch
                and self.stream_serial > self._session_serial
                and self._ready_serial == self.stream_serial)

    def char_for(self, idx: int) -> str:
        if self.characters is not None and 0 <= idx < len(self.characters):
            return self.characters[idx]
        return f"#{idx}"

    # -- event application ----------------------------------------------
    def _check_fingerprint(self, observed, context) -> bool:
        if self.expected_fingerprint is None:
            self.mismatch_message = (
                "Inference output hidden: this panel has no valid session manifest loaded, "
                "so the stream's models cannot be verified.")
            return False
        try:
            assert_same_fingerprint(self.expected_fingerprint, observed or None, context=context)
        except ManifestMismatchError as exc:
            self.mismatch_message = f"Inference output hidden: {exc}"
            return False
        return True

    def apply_payload(self, text) -> str | None:
        """Decode and apply one LSL string sample. Returns the event type applied, or None."""
        try:
            event = ev.decode(text, characters=self.characters) if self.characters is not None \
                else ev.decode(text)
        except ev.EventSchemaError as exc:
            self.malformed_count += 1
            self.last_malformed_error = str(exc)
            return None
        return self.apply_event(event)

    def apply_event(self, event: dict) -> str | None:
        """Apply an already-validated event."""
        if self.mismatch:
            self.ignored_after_mismatch += 1
            return None
        kind = event["event"]
        if kind == ev.SERVICE_STATUS:
            observed = event["manifest_fingerprint"]
            if observed is not None and observed != self.expected_fingerprint:
                self.mismatch_message = (
                    "Inference output hidden: session manifest mismatch -- a service_status reports "
                    f"{short_fingerprint(observed)}, this panel loaded "
                    f"{short_fingerprint(self.expected_fingerprint)}.")
                return None
            self.service_state = event["state"]
            self.service_detail = event["detail"]
            if self.service_state == ev.STATE_READY:
                self._ready_serial = self.stream_serial
            elif self.service_state in (ev.STATE_STARTING, ev.STATE_ERROR, ev.STATE_STOPPED):
                self._ready_serial = None
        elif kind == ev.POSTERIOR_UPDATE:
            if event["trial_id"] in self._committed_trials:
                self.stale_updates_ignored += 1
                return None
            self.gauge = GaugeState(
                trial_id=event["trial_id"], flash_idx=event["flash_idx"],
                max_prob=event["max_prob"], theta=event["theta_current"],
                leading_char_idx=event["leading_char_idx"],
                leading_char=self.char_for(event["leading_char_idx"]))
        elif kind == ev.CHARACTER_COMMITTED:
            if event["trial_id"] in self._committed_trials:
                self.duplicate_commits_ignored += 1
                return None
            self._committed_trials.add(event["trial_id"])
            self.committed.append(CommittedChar(
                trial_id=event["trial_id"], char_idx=event["char_idx"], char=event["char"],
                n_flashes_used=event["n_flashes_used"], reason=event["reason"]))
            self.gauge = None
        elif kind == ev.FLASH_REJECTED:
            self.rejected_by_reason[event["reason"]] += 1
            self.last_rejected_channels = list(event["channels"])
        return kind

    # -- text for labels ----------------------------------------------------
    def service_text(self) -> str:
        if not self.stream_connected:
            return "Inference service: no P300Inference stream"
        if self.service_state is None:
            return "Inference service: stream connected, no status yet"
        detail = f" — {self.service_detail}" if self.service_detail else ""
        return f"Inference service: {self.service_state}{detail}"

    def rejected_text(self) -> str:
        channels = f" (last: {', '.join(self.last_rejected_channels)})" \
            if self.rejected_quality and self.last_rejected_channels else ""
        return (f"Rejected flashes: {self.rejected_quality} quality gate{channels} · "
                f"{self.rejected_other} other")

    def malformed_text(self) -> str:
        if not self.malformed_count:
            return ""
        noun = "event" if self.malformed_count == 1 else "events"
        return f"{self.malformed_count} malformed inference {noun} (last: {self.last_malformed_error})"


# ----------------------------------------------------------------------
# Exit codes
# ----------------------------------------------------------------------

STOPPING_METHODS = ("reset", "llr", "update")


EXIT_LINE_RE = re.compile(r"^[\w.-]+: exit -?\d+:\s*(.*)$")
LIBLSL_LOG_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \(")


def exit_reason(lines: Iterable[str]) -> str | None:
    """
    A process's one-line exit reason from its stderr tail. The service's own
    `inference_service: exit N: <reason>` line wins (prefix stripped); without
    one, the last line that is not liblsl's own log output.
    """
    cleaned = [line.strip() for line in lines if line and line.strip()]
    for line in reversed(cleaned):
        match = EXIT_LINE_RE.match(line)
        if match:
            return match.group(1) or line
    for line in reversed(cleaned):
        if not LIBLSL_LOG_RE.match(line):
            return line
    return None


def not_implemented_method(lines: Iterable[str]) -> str:
    """Which BayesianStoppingController method the service's stderr names, or a generic label."""
    text = "\n".join(lines)
    match = re.search(r"BayesianStoppingController\.(\w+)", text)
    if match:
        return match.group(1)
    match = re.search(r"\b(reset|llr|update)\(\)", text)
    if match:
        return match.group(1)
    return "reset/llr/update"


def describe_recorder_exit(code: int, stderr_lines: Iterable[str] = ()) -> str:
    reason = exit_reason(stderr_lines)
    suffix = f": {reason}" if reason else ""
    if code == 0:
        return "Recorder finished normally."
    if code == 1:
        return f"Recorder crashed (exit 1){suffix}"
    if code == 2:
        return f"Recorder refused to start: bad arguments or the output file already exists{suffix}"
    if code == 3:
        return f"Recorder could not find the EEG stream — is the Unicorn LSL Interface (Windows) or unicorn_relay.py (Mac) streaming?{suffix}"
    if code < 0:
        return f"Recorder was killed by signal {-code}."
    return f"Recorder exited with unexpected code {code}{suffix}"


def describe_service_exit(code: int, stderr_lines: Iterable[str] = ()) -> str:
    lines = list(stderr_lines)
    reason = exit_reason(lines)
    suffix = f": {reason}" if reason else ""
    if code == 0:
        return "Live inference service finished normally."
    if code == 1:
        return f"Live inference crashed (exit 1){suffix}"
    if code == 2:
        return f"Live inference refused to start: configuration, manifest or checkpoint error{suffix}"
    if code == 3:
        return f"Live inference could not find the EEG stream{suffix}"
    if code == 4:
        return (f"Live inference stopped: BayesianStoppingController."
                f"{not_implemented_method(lines)}() is not implemented yet")
    if code < 0:
        return f"Live inference service was killed by signal {-code}."
    return f"Live inference exited with unexpected code {code}{suffix}"


def describe_stimulus_exit(code: int) -> str:
    if code == 0:
        return "Stimulus finished — stopping recorder."
    return f"Stimulus exited with code {code} — stopping recorder."


# ----------------------------------------------------------------------
# Session orchestration
# ----------------------------------------------------------------------

def service_command(python: str, manifest_path, extra_args: Iterable[str] = ()) -> list[str]:
    return [python, "-m", "pipeline.inference_service", "--manifest", str(manifest_path), *extra_args]


def recorder_command(python: str, out_path) -> list[str]:
    return [python, str(COLLECTION_DIR / "record_session.py"), "--out", str(out_path)]


def stimulus_command(python: str, settings: SessionSettings) -> list[str]:
    cmd = [python, str(COLLECTION_DIR / "checkerboard_stimulus.py"),
           "--targets", settings.targets,
           "--n-repetitions", str(settings.repetitions),
           "--flash-ms", str(settings.flash_ms),
           "--isi-ms", str(settings.isi_ms),
           "--screen", str(settings.screen)]
    if settings.windowed:
        cmd.append("--windowed")
    return cmd


IDLE = "idle"
WAITING_SERVICE = "waiting_service"
WAITING_RECORDER = "waiting_recorder"
RUNNING = "running"
STOPPING = "stopping"

INFO = "info"
ERROR = "error"

SERVICE_READY_TIMEOUT_S = 20.0
RECORDER_READY_TIMEOUT_S = 35.0
STOP_KILL_AFTER_S = 3.0
RECORDER_STARTED_EVENT = "recording_started"


def recorder_log_path(out_path) -> Path:
    """The recorder's structured log, which it writes next to its CSV by default."""
    return xlog.recorder_log_path(out_path)


def recorder_started(out_path) -> bool:
    """True once the recorder's structured log records that EEG recording has begun."""
    import json
    try:
        with open(recorder_log_path(out_path), encoding="utf-8") as log:
            for line in log:
                try:
                    if json.loads(line).get("event") == RECORDER_STARTED_EVENT:
                        return True
                except (ValueError, AttributeError):
                    continue
    except OSError:
        return False
    return False


class SessionLogger:
    """
    Writes <recording>.session.json for one session: what was spelled (for a calibration
    session, the calibration sentence), the settings, and the start/end timeline. Schema and
    file names live in pipeline/experiment_log.py. begin() at Start, end() at stop.
    """

    def __init__(self, *, demo: bool, characters: list[str]):
        self.demo = demo
        self.characters = list(characters)
        self.record: dict | None = None
        self.path: Path | None = None

    def begin(self, settings: SessionSettings, out_path, manifest_path) -> Path:
        manifest = None
        if manifest_path is not None:
            manifest = {"path": Path(manifest_path).as_posix(), "fingerprint": None, "language_model": None}
            try:
                loaded = load_manifest(manifest_path, verify_hashes=False)
                manifest["fingerprint"] = loaded.fingerprint
                manifest["language_model"] = loaded.lm_config
            except ManifestError as exc:
                manifest["error"] = str(exc)
        options = {k: v for k, v in asdict(settings).items() if k not in ("session_id", "targets")}
        record = xlog.build_session_record(
            session_id=settings.session_id, demo=self.demo, targets=settings.targets,
            characters=self.characters, settings=options, recording=out_path, manifest=manifest,
            started_utc=xlog.utc_now_iso())
        path = xlog.session_record_path(out_path)
        xlog.write_json_atomic(record, path)
        self.record, self.path = record, path
        return path

    def end(self, message: str, level: str, phase_at_end: str) -> Path | None:
        """Finalizes the record written by begin(); a no-op if no session is open."""
        if self.record is None:
            return None
        record, self.record = self.record, None
        done = xlog.finalize_session_record(record, ended_utc=xlog.utc_now_iso(), message=message,
                                            level=level, phase_at_end=phase_at_end)
        return xlog.write_json_atomic(done, self.path)


class SessionOrchestrator:
    """
    Start order: [inference service -> wait for a verified `ready`] ->
    recorder -> wait until its log reports recording_started -> stimulus.
    A fixed delay instead could start flashing before the recorder had found
    the EEG stream (its own EEG timeout is 30 s). When the stimulus exits, the
    recorder and service are stopped. Stop is terminate, then kill after
    `kill_after_s`.

    Non-blocking: the GUI calls tick() from its refresh timer. Processes are
    started through `launcher(name, cmd, cwd) -> handle`, where a handle has
    poll(), terminate(), kill(), and stderr_lines(); `recorder_ready(out_path)`
    defaults to reading the recorder's log. Tests inject fakes for both.
    `notify(message, level)` receives user-facing messages.
    """

    NAMES = ("stimulus", "recorder", "service")

    def __init__(self, launcher, clock, notify, *, python: str = sys.executable,
                 repo_root=REPO_ROOT, collection_dir=COLLECTION_DIR,
                 service_ready_timeout_s: float = SERVICE_READY_TIMEOUT_S,
                 recorder_ready_timeout_s: float = RECORDER_READY_TIMEOUT_S,
                 kill_after_s: float = STOP_KILL_AFTER_S,
                 service_extra_args: Iterable[str] = (),
                 recorder_ready: Callable = recorder_started,
                 session_log: SessionLogger | None = None):
        self.service_extra_args = list(service_extra_args)
        self.session_log = session_log
        self.launcher = launcher
        self.clock = clock
        self.notify = notify
        self.python = python
        self.repo_root = Path(repo_root)
        self.collection_dir = Path(collection_dir)
        self.service_ready_timeout_s = service_ready_timeout_s
        self.recorder_ready_timeout_s = recorder_ready_timeout_s
        self.recorder_ready = recorder_ready
        self.kill_after_s = kill_after_s
        self.phase = IDLE
        self.procs: dict = {}
        self._stopping: dict = {}   # name -> (handle, kill deadline)
        self._phase_started = 0.0
        self.settings: SessionSettings | None = None
        self.out_path: Path | None = None
        self.stimulus_runner = None

    @property
    def active(self) -> bool:
        """True while any part of a session is starting, running, or still shutting down."""
        return self.phase != IDLE or bool(self._stopping)

    def start(self, settings: SessionSettings, out_path, manifest_path=None, stimulus_runner=None) -> bool:
        """stimulus_runner(settings) -> handle runs the board in-process instead of launching the stimulus."""
        if self.active:
            return False
        self.settings = settings
        self.out_path = Path(out_path)
        self.stimulus_runner = stimulus_runner
        self._begin_session_log(manifest_path)
        if manifest_path is not None:
            service_args = list(self.service_extra_args)
            if self.session_log is not None:
                # Keep this session's inference log next to its recording.
                service_args += ["--log-file", str(xlog.inference_log_path(self.out_path))]
            self._launch("service", service_command(self.python, manifest_path, service_args),
                         self.repo_root)
            self._enter(WAITING_SERVICE)
            self.notify("Starting live inference service — waiting for it to report ready…", INFO)
        else:
            self._start_recorder()
        return True

    def tick(self, service_ready: bool = False):
        now = self.clock()
        self._reap_stopping(now)

        if self.phase == WAITING_SERVICE:
            code = self._exit_code("service")
            if code is not None:
                lines = self._take("service").stderr_lines()
                self._abort(describe_service_exit(code, lines) +
                            " Session not started. Untick Live inference to record without it.")
            elif service_ready:
                self.notify("Live inference ready.", INFO)
                self._start_recorder()
            elif now - self._phase_started > self.service_ready_timeout_s:
                self._abort(f"Live inference did not report ready (with a matching manifest "
                            f"fingerprint) within {self.service_ready_timeout_s:.0f} s. Session not started.")
            return

        if self.phase in (WAITING_RECORDER, RUNNING):
            self._check_service_during_session()

        if self.phase == WAITING_RECORDER:
            code = self._exit_code("recorder")
            if code is not None:
                lines = self._take("recorder").stderr_lines()
                self._abort(describe_recorder_exit(code, lines) + " Session not started.")
            elif self.recorder_ready(self.out_path):
                if self.stimulus_runner is not None:
                    self.procs["stimulus"] = self.stimulus_runner(self.settings)
                else:
                    self._launch("stimulus", stimulus_command(self.python, self.settings), self.collection_dir)
                self._enter(RUNNING)
                n = len(self.settings.targets)
                self.notify(f"Running — {n} trial{'s' if n != 1 else ''} → {self.out_path}", INFO)
            elif now - self._phase_started > self.recorder_ready_timeout_s:
                self._abort(f"Recorder did not report that recording started within "
                            f"{self.recorder_ready_timeout_s:.0f} s (see {recorder_log_path(self.out_path)}). "
                            "Session not started.")
            return

        if self.phase == RUNNING:
            code = self._exit_code("stimulus")
            if code is not None:
                self._take("stimulus")
                self.stop(describe_stimulus_exit(code), level=INFO if code == 0 else ERROR)
                return
            code = self._exit_code("recorder")
            if code is not None:
                lines = self._take("recorder").stderr_lines()
                self.stop(describe_recorder_exit(code, lines) + " Stopping the session.", level=ERROR)

    def _check_service_during_session(self):
        code = self._exit_code("service")
        if code is not None:
            lines = self._take("service").stderr_lines()
            self.notify(describe_service_exit(code, lines) + " Recording continues.",
                        INFO if code == 0 else ERROR)

    def stop(self, message: str = "Stopped.", level: str = INFO):
        """Terminate every running process (stimulus, recorder, service); kill stragglers later in tick()."""
        now = self.clock()
        phase_at_end = self.phase
        for name in self.NAMES:
            handle = self.procs.pop(name, None)
            if handle is None:
                continue
            if handle.poll() is None:
                try:
                    handle.terminate()
                except OSError:
                    pass
                self._stopping[name] = (handle, now + self.kill_after_s)
        self.phase = IDLE
        self._end_session_log(message, level, phase_at_end)
        self.notify(message, level)

    def shutdown(self, wait_s: float = STOP_KILL_AFTER_S, sleep=None):
        """Blocking stop for window close: terminate, wait up to wait_s, then kill."""
        import time as _time
        sleep = sleep or _time.sleep
        if self.phase != IDLE or self.procs:
            self.stop("Closing — stopping all processes.")
        deadline = self.clock() + wait_s
        while self._stopping and self.clock() < deadline:
            self._reap_stopping(self.clock(), kill=False)
            if self._stopping:
                sleep(0.05)
        for name, (handle, _) in list(self._stopping.items()):
            if handle.poll() is None:
                try:
                    handle.kill()
                except OSError:
                    pass
        self._stopping.clear()

    # -- helpers ----------------------------------------------------------
    def _begin_session_log(self, manifest_path):
        """A session log that can't be written is reported, never fatal: the recording is what matters."""
        if self.session_log is None:
            return
        try:
            self.session_log.begin(self.settings, self.out_path, manifest_path)
        except OSError as exc:
            self.notify(f"Could not write the session log ({exc}); the recording itself is unaffected.", ERROR)

    def _end_session_log(self, message, level, phase_at_end):
        if self.session_log is None:
            return
        try:
            self.session_log.end(message, level, phase_at_end)
        except OSError as exc:
            self.notify(f"Could not finish the session log ({exc}); the recording itself is unaffected.", ERROR)

    def _start_recorder(self):
        self._launch("recorder", recorder_command(self.python, self.out_path), self.collection_dir)
        self._enter(WAITING_RECORDER)
        self.notify(f"Starting recorder — {len(self.settings.targets)} trials, recording to "
                    f"{self.out_path}", INFO)

    def _launch(self, name, cmd, cwd):
        self.procs[name] = self.launcher(name, cmd, Path(cwd))

    def _enter(self, phase):
        self.phase = phase
        self._phase_started = self.clock()

    def _exit_code(self, name):
        handle = self.procs.get(name)
        return None if handle is None else handle.poll()

    def _take(self, name):
        return self.procs.pop(name)

    def _abort(self, message):
        self.stop(message, level=ERROR)

    def _reap_stopping(self, now, kill: bool = True):
        for name, (handle, deadline) in list(self._stopping.items()):
            if handle.poll() is not None:
                del self._stopping[name]
            elif kill and now >= deadline:
                try:
                    handle.kill()
                except OSError:
                    pass
                del self._stopping[name]


class StderrTail:
    """Thread-safe bounded tail of a process's stderr lines (appended by a reader thread)."""

    def __init__(self, maxlen: int = 50):
        self._lines: deque = deque(maxlen=maxlen)

    def append(self, line: str):
        self._lines.append(line.rstrip("\n"))

    def lines(self) -> list[str]:
        return list(self._lines)
