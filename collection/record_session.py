"""
LSL-consuming EEG recorder for the Thai P300 speller.

Does NOT talk to the Unicorn Hybrid Black directly -- no BrainFlow, no
Unicorn SDK calls. BrainFlow's UnicornBoard class is compiled out on
macOS entirely (unicorn_board.h is wrapped in
`#if defined __linux__ || defined _WIN32`), and g.tec's own Unicorn
Suite / Unicorn LSL Interface is Windows-only, so on this project's
Apple Silicon development machine the only viable path is: run g.tec's
official Unicorn LSL Interface on a Windows-on-ARM VM (Parallels, with
the Unicorn's USB dongle passed through) or on Linux, which publishes an
LSL outlet, and have this script pull from that outlet over the network/
loopback. This script is a pure LSL consumer -- see README_ACQUISITION.md
for the full machine topology and startup order.

Consumes two LSL streams per the shared interface contract (resolved by
TYPE, configurable with --eeg-type / --marker-type):
  EEG stream:    name='UnicornEEG', type='EEG', channel_count=8,
                 sample_rate=250, dtype=float32, microvolts
  Marker stream: name='P300Markers', type='Markers', channel_count=1,
                 sample_rate=0 (irregular), dtype=string, one
                 JSON-encoded event per sample (see
                 collection/checkerboard_stimulus.py for the producer).

Writes one CSV per session with columns:
  timestamp, Fz, F3, C4, Cz, Pz, P3, O2, O1, marker_json_or_empty,
  trial_id, flash_idx, group_json, is_target, target_char_idx

Besides one row per EEG sample, three kinds of rows have all 8 channels
NaN and are identified by `marker_json_or_empty`:
  {"event": "stream_gap_start", ...}   EEG went silent / timestamps jumped
  {"event": "stream_gap_end", "duration_s": ..., "n_missing_est": ...}
  a stimulus marker that could not be attached to its nearest EEG sample
      (nearest sample already written, marker inside an EEG gap, or that
      sample already carries a marker); timestamp = the marker's own LSL
      timestamp, parsed marker columns filled.
EEG consumers must take samples from rows with finite channels only.

Side files (next to the CSV):
  <csv>.log.jsonl          structured log, one JSON object per line
                           (override with --log-file)
  <csv>.integrity.json     summary written at exit: counts of gaps,
                           non-monotonic timestamps, rate drift,
                           duplicate / orphan markers, rows written,
                           shutdown reason

Startup: EEG is resolved with exponential backoff (--eeg-timeout). EEG
recording starts as soon as EEG is found; the marker stream is resolved
from inside the recording loop (--marker-timeout only warns), and its
inlet is opened immediately so no marker pushed after connection is lost.
The 2 s signal sanity check runs on the first samples as they are
recorded, not as a blocking pre-pull.

SIGTERM is handled exactly like Ctrl+C: held rows are flushed, the
integrity summary is written, exit code 0.

EXIT CODES
  0  normal stop (Ctrl+C, SIGTERM, --duration elapsed)
  1  unexpected error
  2  bad arguments or output path problem (including: --out already exists)
  3  EEG stream not found within --eeg-timeout

CSV (not Parquet): chosen so the file can be opened/inspected by hand or
in Excel mid-collection without a Python session, and so a crash mid-write
loses at most the last buffered rows rather than corrupting a whole
row-group -- append-friendly plain text beats Parquet's write-once-per-
row-group model for a script that streams rows continuously for an
unknown session length. Switch to Parquet later for analysis-time
compaction if file size becomes a problem; it is not a problem at
8ch x 250Hz for a single session.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
import os
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-exported: session_gui.py, mock_headset.py and tests import these from here.
from pipeline.constants import EEG_CHANNELS, SANITY_PTP_MAX_UV, SANITY_PTP_MIN_UV  # noqa: E402
from pipeline.constants import EEG_INLET_DEJITTER, EEG_SFREQ, EEG_STREAM_TYPE, MARKER_STREAM_TYPE  # noqa: E402
# Same step threshold the epoch cutter uses to call a window gapped.
from pipeline.epoching import MAX_SAMPLE_GAP_FACTOR  # noqa: E402

CSV_COLUMNS = (
    ["timestamp"] + EEG_CHANNELS +
    ["marker_json_or_empty", "trial_id", "flash_idx", "group_json",
     "is_target", "target_char_idx"]
)

SANITY_CHECK_DURATION_S = 2.0

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BAD_ARGS = 2
EXIT_EEG_NOT_FOUND = 3

LOGGER_NAME = "record_session"
logger = logging.getLogger(LOGGER_NAME)

GAP_START_EVENT = "stream_gap_start"
GAP_END_EVENT = "stream_gap_end"


# ----------------------------------------------------------------------
# Row construction (output format -- unchanged)
# ----------------------------------------------------------------------

def parse_marker_json(marker_str: str) -> dict:
    """
    Parse one marker sample's JSON string into the shared-contract event
    dict. Returns {} on empty/unparseable input rather than raising --
    a single malformed marker sample should not crash a multi-hour
    recording session; it just becomes a row with all marker fields empty.
    Valid JSON that is not an object (e.g. a bare number) counts as
    unparseable for the same reason.
    """
    if not marker_str:
        return {}
    try:
        event = json.loads(marker_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("*** WARNING: unparseable marker JSON, skipping: %r ***", marker_str,
                       extra={"event": "marker_unparseable", "fields": {"raw": marker_str}})
        return {}
    if not isinstance(event, dict):
        logger.warning("*** WARNING: marker JSON is not an object, skipping: %r ***", marker_str,
                       extra={"event": "marker_unparseable", "fields": {"raw": marker_str}})
        return {}
    return event


def event_to_row_fields(event: dict) -> dict:
    """
    Map one parsed marker event dict to this CSV's parsed marker columns.
    Fields not present in a given event type (e.g. group_json for a
    trial_start/trial_end event) are left empty.
    """
    return {
        "trial_id": event.get("trial_id", ""),
        "flash_idx": event.get("flash_idx", ""),
        "group_json": json.dumps(event["group"]) if "group" in event else "",
        "is_target": event.get("is_target", ""),
        "target_char_idx": event.get("target_char_idx", ""),
    }


def merge_by_nearest_timestamp(eeg_samples: list, eeg_timestamps: list,
                                marker_samples: list, marker_timestamps: list) -> list:
    """
    Merge EEG samples and marker events into output rows by nearest
    timestamp: every EEG sample becomes one row, and any marker whose
    timestamp is closer to that EEG sample than to its neighbors is
    attached to it. Markers arrive far less often than EEG samples (one
    per flash vs. 250/sec), so this is effectively "attach each marker
    to the EEG sample closest in time to it," with all other rows
    carrying an empty marker.

    The live recorder applies the same rule across chunk boundaries (see
    Recorder._place_marker); this function remains the reference for it.

    Returns:
        list of dicts, each with keys timestamp, <EEG_CHANNELS...>,
        marker_json_or_empty, trial_id, flash_idx, group_json, is_target,
        target_char_idx -- one dict per EEG sample, in timestamp order.
    """
    rows = []
    for ts, sample in zip(eeg_timestamps, eeg_samples):
        row = {"timestamp": ts}
        for ch_idx, ch_name in enumerate(EEG_CHANNELS):
            row[ch_name] = sample[ch_idx] if ch_idx < len(sample) else ""
        row["marker_json_or_empty"] = ""
        row.update({"trial_id": "", "flash_idx": "", "group_json": "",
                    "is_target": "", "target_char_idx": ""})
        rows.append(row)

    if not rows or not marker_timestamps:
        return rows

    eeg_ts_arr = np.asarray(eeg_timestamps)
    for m_ts, m_sample in zip(marker_timestamps, marker_samples):
        nearest_idx = int(np.argmin(np.abs(eeg_ts_arr - m_ts)))
        event = parse_marker_json(m_sample[0] if m_sample else "")
        rows[nearest_idx]["marker_json_or_empty"] = m_sample[0] if m_sample else ""
        rows[nearest_idx].update(event_to_row_fields(event))

    return rows


def nan_row(timestamp: float, marker_json: str = "", event: dict | None = None) -> dict:
    """A CSV row with every EEG channel NaN: gap markers and standalone marker rows."""
    row = {"timestamp": timestamp}
    for ch_name in EEG_CHANNELS:
        row[ch_name] = float("nan")
    row["marker_json_or_empty"] = marker_json
    row.update(event_to_row_fields(event or {}))
    return row


def evaluate_sanity(samples) -> tuple[bool, list[dict]]:
    """
    Per-channel peak-to-peak plausibility check on (n_samples, n_channels)
    microvolt data. Pure; the recorder logs the result.

    Returns (ok, [{"channel", "ptp_uv", "status"}]) with status one of
    "ok", "flat", "railed" (clipped/railed/wrong units), "non_finite",
    "missing" (stream has fewer channels than EEG_CHANNELS).
    """
    arr = np.asarray(samples, dtype=float)
    results = []
    if arr.ndim != 2 or arr.shape[0] == 0:
        return False, [{"channel": ch, "ptp_uv": None, "status": "missing"} for ch in EEG_CHANNELS]
    ok = True
    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        if ch_idx >= arr.shape[1]:
            results.append({"channel": ch_name, "ptp_uv": None, "status": "missing"})
            ok = False
            continue
        col = arr[:, ch_idx]
        if not np.all(np.isfinite(col)):
            results.append({"channel": ch_name, "ptp_uv": None, "status": "non_finite"})
            ok = False
            continue
        ptp_uv = float(col.max() - col.min())
        if ptp_uv < SANITY_PTP_MIN_UV:
            status = "flat"
        elif ptp_uv > SANITY_PTP_MAX_UV:
            status = "railed"
        else:
            status = "ok"
        ok = ok and status == "ok"
        results.append({"channel": ch_name, "ptp_uv": ptp_uv, "status": status})
    return ok, results


# ----------------------------------------------------------------------
# Structured logging
# ----------------------------------------------------------------------

def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per record: time, level, event, msg, plus the record's fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": getattr(record, "event", "message"),
            "msg": record.getMessage(),
        }
        for key, value in (getattr(record, "fields", None) or {}).items():
            payload[key if key not in payload else f"field_{key}"] = _json_safe(value)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _ConsoleFilter(logging.Filter):
    def __init__(self, min_level: int, max_level: int):
        super().__init__()
        self.min_level, self.max_level = min_level, max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return (getattr(record, "console", True)
                and self.min_level <= record.levelno < self.max_level)


def setup_logging(log_path: str, console: bool = True) -> logging.Logger:
    """
    Attach a JSON-lines file handler (appending, so a retried run keeps the
    earlier attempt's log) and concise console handlers: INFO to stdout,
    WARNING and above to stderr. Undo with teardown_logging().
    """
    teardown_logging()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(JsonLinesFormatter())
    logger.addHandler(file_handler)
    if console:
        fmt = logging.Formatter("[record_session] %(message)s")
        out = logging.StreamHandler(sys.stdout)
        out.addFilter(_ConsoleFilter(logging.DEBUG, logging.WARNING))
        out.setFormatter(fmt)
        err = logging.StreamHandler(sys.stderr)
        err.addFilter(_ConsoleFilter(logging.WARNING, logging.CRITICAL + 1))
        err.setFormatter(fmt)
        logger.addHandler(out)
        logger.addHandler(err)
    return logger


def teardown_logging() -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()
        except Exception:  # pragma: no cover - best effort on shutdown
            pass


# ----------------------------------------------------------------------
# Injected dependencies: LSL access and time. Unit tests substitute fakes.
# ----------------------------------------------------------------------

class SystemClock:
    """Wall-clock source for silence/timeout logic (never used for sample timestamps)."""

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


def inlet_processing_names(kind: str, clocksync: bool = True) -> list[str]:
    """
    pylsl processing flags for an inlet, by pylsl attribute name.

    proc_clocksync maps sample timestamps into this machine's local_clock()
    domain: without it EEG timestamps from the Windows VM / Linux box stay in
    that machine's clock and nearest-sample merging against locally produced
    markers is meaningless (for a local stream the correction is ~0).
    --no-clocksync removes it (debugging only).

    proc_dejitter is decided once, for the recorder and the inference service
    together, by pipeline.constants.EEG_INLET_DEJITTER (EEG only). Off, the
    integrity checks see timestamps as the device delivers them; measured on an
    outlet that stamps chunks on arrival, dejitter silently removed
    non-monotonic steps by moving timestamps up to ~19 ms.
    """
    names = ["proc_clocksync"] if clocksync else []
    if kind == "eeg" and EEG_INLET_DEJITTER:
        names.append("proc_dejitter")
    return names


class LslBackend:
    """
    Real pylsl access. Inlets are created with recover=True (measured:
    recover=False raises LostError on the next pull and drops samples that
    had already arrived, e.g. the stimulus's final trial_end).
    """

    def __init__(self, clocksync: bool = True):
        self.clocksync = clocksync

    def resolve(self, stream_type: str, timeout: float) -> list:
        from pylsl import resolve_byprop
        return list(resolve_byprop("type", stream_type, timeout=max(0.0, timeout)))

    def processing_flag_names(self, kind: str) -> list[str]:
        return inlet_processing_names(kind, self.clocksync)

    def processing_flags(self, kind: str) -> int:
        import pylsl
        flags = 0
        for name in self.processing_flag_names(kind):
            flags |= getattr(pylsl, name)
        return flags

    def make_inlet(self, info, kind: str):
        """kind is "eeg" or "marker"."""
        from pylsl import StreamInlet
        return StreamInlet(info, max_buflen=360, recover=True,
                           processing_flags=self.processing_flags(kind))


def describe_stream(info) -> dict:
    """Identity fields of a pylsl StreamInfo (or a fake with the same methods)."""
    out = {}
    for key in ("name", "type", "source_id", "uid", "hostname", "nominal_srate",
                "channel_count", "created_at"):
        try:
            out[key] = getattr(info, key)()
        except Exception:
            out[key] = None
    return out


@dataclass
class RecorderConfig:
    out_path: str
    log_path: str | None = None
    integrity_path: str | None = None
    eeg_type: str = EEG_STREAM_TYPE
    marker_type: str = MARKER_STREAM_TYPE
    eeg_timeout_s: float = 30.0
    marker_timeout_s: float = 120.0
    marker_timeout_repeat_s: float = 30.0
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 5.0
    eeg_probe_timeout_s: float = 1.0
    marker_probe_timeout_s: float = 0.1
    marker_probe_every_s: float = 0.25      # while no marker inlet
    marker_recheck_every_s: float = 5.0     # while connected: replaced outlet?
    marker_idle_recheck_s: float = 1.0      # faster rechecks once markers go quiet
    open_timeout_s: float = 5.0
    pull_timeout_s: float = 0.1
    holdback_s: float = 1.0
    silence_s: float = 0.5
    gap_jump_s: float = 0.1
    restart_gap_s: float = 1.0
    session_duration_s: float | None = None
    sanity_duration_s: float = SANITY_CHECK_DURATION_S
    skip_sanity_check: bool = False
    rate_window_s: float = 10.0
    rate_tolerance: float = 0.02
    rate_warn_every_s: float = 30.0
    no_samples_warn_s: float = 5.0
    gap_ongoing_warn_s: float = 5.0

    def __post_init__(self):
        if self.log_path is None:
            self.log_path = self.out_path + ".log.jsonl"
        if self.integrity_path is None:
            self.integrity_path = self.out_path + ".integrity.json"


@dataclass
class _Row:
    ts: float
    kind: str        # "eeg" | "gap" | "marker"
    fields: dict


@dataclass
class _Marker:
    ts: float
    raw: str
    event: dict
    arrival: float   # wall clock (monotonic) when pulled
    outlet_key: str


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------

class Recorder:
    """
    The recording loop, separated from LSL by `backend` (resolve / make_inlet)
    and from time by `clock` (monotonic / sleep), so it runs on fakes in tests.
    """

    def __init__(self, config: RecorderConfig, backend=None, clock=None, log: logging.Logger | None = None):
        self.cfg = config
        self.backend = backend if backend is not None else LslBackend()
        self.clock = clock if clock is not None else SystemClock()
        self.log = log if log is not None else logger
        self._stop_reason: str | None = None

        self._csv_file = None
        self._writer = None
        self._started_wall_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # EEG
        self._eeg_inlet = None
        self._eeg_desc: dict | None = None
        self._srate = EEG_SFREQ
        self._period = 1.0 / EEG_SFREQ
        self._eeg_connected_at: float | None = None
        self._last_eeg_ts: float | None = None
        self._max_eeg_ts: float | None = None
        self._last_sample_wall: float | None = None
        self._last_written_eeg_ts: float | None = None
        # Timestamps of recently written EEG rows, so a very late marker's
        # standalone row can say whether its nearest sample was already on disk.
        self._written_eeg_ts: deque = deque(maxlen=int(60 * EEG_SFREQ))
        self._gap_open = False
        self._gap_start_ts: float | None = None
        self._gap_opened_wall: float | None = None
        self._last_gap_ongoing_warn = 0.0
        self._last_no_samples_warn: float | None = None
        self._n_eeg_samples = 0

        # Rows held back so late markers can still reach their nearest sample.
        self._held: list[_Row] = []
        self._pending: list[_Marker] = []

        # Sanity check on the first samples.
        self._sanity_needed = 0
        self._sanity_buf: list = []
        self._sanity_result: dict | None = None

        # Rate drift.
        self._rate_pts: deque = deque()
        self._rate_count = 0
        self._last_rate_warn: float | None = None
        self._rate_min: float | None = None
        self._rate_max: float | None = None
        self._rate_last: float | None = None

        # Markers
        self._marker_inlet = None
        self._marker_desc: dict | None = None
        self._marker_key: str | None = None
        self._marker_run = 0
        self._marker_search_start: float | None = None
        self._next_marker_probe = 0.0
        self._next_marker_recheck = 0.0
        self._last_marker_arrival: float | None = None
        self._last_marker_timeout_warn: float | None = None
        self._marker_outlet_invisible_logged = False
        self._multiple_marker_streams_logged = False
        self._marker_connections: list[dict] = []
        self._recent_marker_ids: deque = deque(maxlen=4096)
        self._recent_marker_set: set = set()
        self._flash_seen: dict = {}        # (key, trial_id, flash_idx) -> ts
        self._trial_starts: dict = {}      # (key, trial_id) -> ts
        self._orphan_trials_logged: set = set()

        # Counters for the integrity summary.
        self.counts = {
            "eeg_resolve_attempts": 0,
            "eeg_ts_non_monotonic": 0,
            "eeg_ts_duplicate": 0,
            "eeg_ts_backwards": 0,
            "eeg_ts_irregular_steps": 0,
            "eeg_pull_errors": 0,
            "stream_gaps": 0,
            "rate_drift_warnings": 0,
            "markers_received": 0,
            "markers_attached": 0,
            "marker_standalone_rows": 0,
            "duplicate_flash_markers": 0,
            "redelivered_markers": 0,
            "flash_without_trial_start": 0,
            "unparseable_markers": 0,
            "marker_reconnections": 0,
            "marker_run_restarts_suspected": 0,
            "marker_pull_errors": 0,
            "marker_timeout_warnings": 0,
        }
        self.standalone_by_reason: dict = {}
        self.orphan_trials: list = []
        self.gaps: list[dict] = []
        self.rows_written = {"total": 0, "eeg": 0, "gap": 0, "marker_standalone": 0}
        self._max_attach_distance_s = 0.0
        self._detail_logged: dict = {}
        self._console_last: dict = {}

    # -- small helpers ---------------------------------------------------

    def _log(self, level: int, event: str, msg: str, console: bool = True, **fields) -> None:
        self.log.log(level, msg, extra={"event": event, "fields": fields, "console": console})

    def _detail_ok(self, kind: str) -> bool:
        """File-log the first 50 occurrences of a repeated integrity event, then every 500th."""
        n = self._detail_logged.get(kind, 0) + 1
        self._detail_logged[kind] = n
        return n <= 50 or n % 500 == 0

    def _console_ok(self, kind: str, now: float, every_s: float = 10.0) -> bool:
        last = self._console_last.get(kind)
        if last is None or now - last >= every_s:
            self._console_last[kind] = now
            return True
        return False

    def _processing_names(self, kind: str):
        """Effective inlet processing flags, when the backend can report them (fakes may not)."""
        names = getattr(self.backend, "processing_flag_names", None)
        return list(names(kind)) if callable(names) else None

    def request_stop(self, reason: str) -> None:
        """Safe to call from a signal handler: only sets a flag the loop polls."""
        if self._stop_reason is None:
            self._stop_reason = reason

    def _sleep_interruptible(self, seconds: float) -> None:
        end = self.clock.monotonic() + seconds
        while self._stop_reason is None:
            remaining = end - self.clock.monotonic()
            if remaining <= 0:
                return
            self.clock.sleep(min(0.1, remaining))

    # -- top level -------------------------------------------------------

    def run(self) -> int:
        """Record until stopped. Returns an exit code (see module docstring)."""
        code = EXIT_OK
        cfg = self.cfg
        self._log(logging.INFO, "recorder_start",
                  f"Starting recorder -> {cfg.out_path} (log: {cfg.log_path})",
                  config=asdict(cfg), pid=os.getpid())
        try:
            if not self._connect_eeg():
                if self._stop_reason is not None:
                    self._log(logging.INFO, "shutdown",
                              f"Stopped ({self._stop_reason}) before the EEG stream was found; "
                              "no CSV written.", reason=self._stop_reason, rows_written=0)
                    return EXIT_OK
                self._log(logging.ERROR, "eeg_not_found",
                          f"ERROR: no EEG stream (type={cfg.eeg_type!r}) found within "
                          f"{cfg.eeg_timeout_s:g}s -- is the Unicorn LSL Interface (or "
                          f"mock_headset.py) streaming? See README_ACQUISITION.md.",
                          stream_type=cfg.eeg_type, timeout_s=cfg.eeg_timeout_s,
                          attempts=self.counts["eeg_resolve_attempts"])
                return EXIT_EEG_NOT_FOUND
            try:
                self._open_csv()
            except OSError as exc:
                self._log(logging.ERROR, "output_open_failed",
                          f"ERROR: cannot create {cfg.out_path}: {exc}", error=str(exc))
                self._close_inlets()
                return EXIT_BAD_ARGS
            self._loop()
        except Exception as exc:
            code = EXIT_ERROR
            self.request_stop(f"error: {type(exc).__name__}: {exc}")
            self.log.error("Unexpected error: %s", exc, exc_info=True,
                           extra={"event": "unexpected_error", "fields": {"error": repr(exc)}})
        finally:
            if self._csv_file is not None:
                try:
                    self._shutdown(code)
                except Exception as exc:
                    code = EXIT_ERROR
                    self.log.error("Error during shutdown: %s", exc, exc_info=True,
                                   extra={"event": "shutdown_error", "fields": {"error": repr(exc)}})
        return code

    def _open_csv(self) -> None:
        # "x": never overwrite, even if the path appeared after argument checks.
        self._csv_file = open(self.cfg.out_path, "x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._csv_file.flush()

    # -- EEG resolution ----------------------------------------------------

    def _connect_eeg(self) -> bool:
        """Resolve + open the EEG inlet with exponential backoff. False on timeout or stop."""
        cfg = self.cfg
        start = self.clock.monotonic()
        delay = cfg.backoff_initial_s
        attempt = 0
        while self._stop_reason is None:
            attempt += 1
            self.counts["eeg_resolve_attempts"] = attempt
            elapsed = self.clock.monotonic() - start
            probe = max(0.0, min(cfg.eeg_probe_timeout_s, cfg.eeg_timeout_s - elapsed))
            infos = self.backend.resolve(cfg.eeg_type, probe)
            elapsed = self.clock.monotonic() - start
            if infos:
                if len(infos) > 1:
                    self._log(logging.WARNING, "eeg_multiple_streams",
                              f"*** WARNING: {len(infos)} streams of type {cfg.eeg_type!r}; "
                              f"using the first ({describe_stream(infos[0]).get('name')}). ***",
                              streams=[describe_stream(i) for i in infos])
                if self._open_eeg_inlet(infos[0], attempt, elapsed):
                    return True
            remaining = cfg.eeg_timeout_s - elapsed
            if remaining <= 0:
                return False
            wait = min(delay, remaining)
            self._log(logging.INFO, "eeg_resolve_attempt",
                      f"Waiting for EEG stream (type={cfg.eeg_type!r}): attempt {attempt}, "
                      f"{elapsed:.1f}s / {cfg.eeg_timeout_s:g}s elapsed, retrying in {wait:.1f}s",
                      attempt=attempt, elapsed_s=round(elapsed, 3), timeout_s=cfg.eeg_timeout_s,
                      next_retry_s=round(wait, 3), found=bool(infos))
            self._sleep_interruptible(wait)
            delay = min(delay * 2.0, cfg.backoff_max_s)
        return False

    def _open_eeg_inlet(self, info, attempt: int, elapsed: float) -> bool:
        desc = describe_stream(info)
        try:
            inlet = self.backend.make_inlet(info, "eeg")
            inlet.open_stream(timeout=self.cfg.open_timeout_s)
        except Exception as exc:
            self._log(logging.WARNING, "eeg_open_failed",
                      f"*** WARNING: found EEG stream {desc.get('name')!r} but could not open it: {exc} ***",
                      stream=desc, error=repr(exc), attempt=attempt)
            return False
        self._eeg_inlet = inlet
        self._eeg_desc = desc
        srate = desc.get("nominal_srate") or 0.0
        if srate and srate > 0:
            self._srate = float(srate)
            self._period = 1.0 / self._srate
        if srate != EEG_SFREQ or desc.get("channel_count") != len(EEG_CHANNELS):
            self._log(logging.WARNING, "eeg_contract_mismatch",
                      f"*** WARNING: EEG stream reports {desc.get('channel_count')} ch @ {srate} Hz; "
                      f"the contract is {len(EEG_CHANNELS)} ch @ {EEG_SFREQ:g} Hz. ***",
                      stream=desc, expected_srate=EEG_SFREQ, expected_channels=len(EEG_CHANNELS))
        now = self.clock.monotonic()
        self._eeg_connected_at = now
        self._marker_search_start = now
        self._next_marker_probe = now
        if not self.cfg.skip_sanity_check:
            self._sanity_needed = max(1, int(round(self.cfg.sanity_duration_s * self._srate)))
        self._log(logging.INFO, "eeg_connected",
                  f"Found EEG stream {desc.get('name')!r} ({desc.get('channel_count')} ch @ {srate} Hz, "
                  f"source_id={desc.get('source_id')!r}) after {attempt} attempt(s), {elapsed:.1f}s. "
                  f"Recording to {self.cfg.out_path} -- Ctrl+C to stop.",
                  stream=desc, attempt=attempt, elapsed_s=round(elapsed, 3),
                  out_path=self.cfg.out_path, processing_flags=self._processing_names("eeg"))
        if self.cfg.skip_sanity_check:
            self._log(logging.INFO, "sanity_check_skipped",
                      "Skipping signal sanity check (--skip-sanity-check).")
        return True

    # -- main loop -------------------------------------------------------------

    def _loop(self) -> None:
        cfg = self.cfg
        rec_start = self.clock.monotonic()
        self._log(logging.INFO, "recording_started", "Recording EEG; searching for the marker stream.",
                  marker_type=cfg.marker_type, marker_timeout_s=cfg.marker_timeout_s)
        while True:
            now = self.clock.monotonic()
            if self._stop_reason is not None:
                break
            if cfg.session_duration_s is not None and now - rec_start >= cfg.session_duration_s:
                self.request_stop("duration_elapsed")
                break
            self._service_marker_connection(now)

            chunk, stamps = self._pull_eeg()
            now = self.clock.monotonic()
            if chunk:
                self._ingest_eeg(chunk, stamps, now)
            else:
                self._check_silence(now)

            self._pull_markers(now)
            self._place_pending(now, force=False)
            self._flush(all_rows=self._gap_open)

    def _pull_eeg(self):
        try:
            chunk, stamps = self._eeg_inlet.pull_chunk(timeout=self.cfg.pull_timeout_s)
        except Exception as exc:
            # recover=True inlets do not raise LostError; anything here is
            # unexpected but must not end the session.
            self.counts["eeg_pull_errors"] += 1
            now = self.clock.monotonic()
            if self._detail_ok("eeg_pull_error"):
                self._log(logging.WARNING, "eeg_pull_error", f"*** WARNING: EEG pull failed: {exc} ***",
                          console=self._console_ok("eeg_pull_error", now), error=repr(exc),
                          count=self.counts["eeg_pull_errors"])
            self.clock.sleep(self.cfg.pull_timeout_s)
            return [], []
        return chunk, stamps

    # -- marker connection ------------------------------------------------------

    def _service_marker_connection(self, now: float) -> None:
        cfg = self.cfg
        if self._marker_inlet is None:
            if now < self._next_marker_probe:
                return
            self._next_marker_probe = now + cfg.marker_probe_every_s
            infos = self.backend.resolve(cfg.marker_type, cfg.marker_probe_timeout_s)
            if infos:
                self._connect_marker(infos[0], reason="found" if not self._marker_connections else "reappeared")
                return
            waited = now - (self._marker_search_start or now)
            if not self._marker_connections and waited >= cfg.marker_timeout_s:
                last = self._last_marker_timeout_warn
                if last is None or now - last >= cfg.marker_timeout_repeat_s:
                    self._last_marker_timeout_warn = now
                    self.counts["marker_timeout_warnings"] += 1
                    self._log(logging.WARNING, "marker_timeout",
                              f"*** WARNING: no marker stream (type={cfg.marker_type!r}) after "
                              f"{waited:.0f}s. Recording EEG WITHOUT stimulus events -- start "
                              f"checkerboard_stimulus.py. Still searching. ***",
                              waited_s=round(waited, 1), timeout_s=cfg.marker_timeout_s,
                              warnings=self.counts["marker_timeout_warnings"])
            return

        idle = self._last_marker_arrival is None or now - self._last_marker_arrival >= cfg.marker_idle_recheck_s
        due = now >= self._next_marker_recheck
        if not due and not (idle and now >= self._next_marker_probe):
            return
        self._next_marker_recheck = now + cfg.marker_recheck_every_s
        self._next_marker_probe = now + cfg.marker_idle_recheck_s
        infos = self.backend.resolve(cfg.marker_type, cfg.marker_probe_timeout_s)
        uids = [describe_stream(i).get("uid") for i in infos]
        current = self._marker_desc.get("uid")
        if current in uids:
            self._marker_outlet_invisible_logged = False
            if len(set(uids)) > 1 and not self._multiple_marker_streams_logged:
                self._multiple_marker_streams_logged = True
                self._log(logging.WARNING, "marker_multiple_streams",
                          f"*** WARNING: {len(set(uids))} marker streams of type {cfg.marker_type!r} "
                          f"visible; staying on uid {current}. ***",
                          streams=[describe_stream(i) for i in infos])
            return
        if not infos:
            if not self._marker_outlet_invisible_logged:
                self._marker_outlet_invisible_logged = True
                self._log(logging.INFO, "marker_outlet_not_visible",
                          "Marker outlet no longer visible (stimulus exited?); keeping the inlet "
                          "for recovery and watching for a new outlet.", uid=current)
            return
        new_desc = describe_stream(infos[0])
        self._log(logging.WARNING, "marker_outlet_replaced",
                  f"Marker outlet replaced (uid {current} -> {new_desc.get('uid')}, "
                  f"source_id {self._marker_desc.get('source_id')!r} -> {new_desc.get('source_id')!r}); "
                  "reconnecting.", old=self._marker_desc, new=new_desc)
        old_key = self._marker_key
        drained = self._disconnect_marker()
        created_at = new_desc.get("created_at")
        # With the same source_id the old inlet may already have auto-recovered
        # onto the new outlet; credit what it delivered after the new outlet's
        # creation to the new outlet.
        before = [d for d in drained if not created_at or d[1] < created_at]
        after = [d for d in drained if created_at and d[1] >= created_at]
        self._ingest_markers(before, old_key, now)
        self._connect_marker(infos[0], reason="replaced", carried=after)

    def _connect_marker(self, info, reason: str, carried=None) -> None:
        desc = describe_stream(info)
        try:
            inlet = self.backend.make_inlet(info, "marker")
            # Open at once: an unopened inlet buffers nothing (measured).
            inlet.open_stream(timeout=self.cfg.open_timeout_s)
        except Exception as exc:
            self.counts["marker_pull_errors"] += 1
            self._log(logging.WARNING, "marker_open_failed",
                      f"*** WARNING: could not open marker stream: {exc} ***",
                      stream=desc, error=repr(exc))
            return
        now = self.clock.monotonic()
        self._marker_inlet = inlet
        self._marker_desc = desc
        self._marker_run += 1
        self._marker_key = f"{desc.get('uid')}#{self._marker_run}"
        self._last_marker_arrival = None
        self._marker_outlet_invisible_logged = False
        self._next_marker_recheck = now + self.cfg.marker_recheck_every_s
        self._next_marker_probe = now + self.cfg.marker_idle_recheck_s
        if self._marker_connections:
            self.counts["marker_reconnections"] += 1
        waited = now - (self._marker_search_start or now)
        self._marker_connections.append({"reason": reason, "stream": desc,
                                         "connected_after_s": round(waited, 3)})
        self._log(logging.INFO, "marker_connected",
                  f"Marker stream {desc.get('name')!r} connected ({reason}, uid={desc.get('uid')}, "
                  f"{waited:.1f}s after EEG).", stream=desc, reason=reason,
                  waited_s=round(waited, 3), outlet_key=self._marker_key,
                  processing_flags=self._processing_names("marker"))
        # Prime the inlet while the outlet is alive: measured with pylsl 1.18.2,
        # a string inlet whose FIRST pull happens after its outlet died blocks
        # indefinitely. Keep whatever this returns.
        if carried:
            self._ingest_markers(carried, self._marker_key, now)
        self._pull_markers(now)

    def _disconnect_marker(self) -> list:
        """Drain the old inlet without ingesting, close it, and return [(raw, ts)]."""
        inlet = self._marker_inlet
        if inlet is None:
            return []
        drained = self._pull_marker_samples(inlet) or []
        try:
            inlet.close_stream()
        except Exception:
            pass
        self._marker_inlet = None
        return drained

    def _close_inlets(self) -> None:
        for inlet in (self._marker_inlet, self._eeg_inlet):
            if inlet is None:
                continue
            try:
                inlet.close_stream()
            except Exception:
                pass

    # -- marker ingestion and integrity -------------------------------------

    def _pull_marker_samples(self, inlet):
        """[(raw, ts)] from one zero-timeout pull, or None if the pull failed."""
        try:
            chunk, stamps = inlet.pull_chunk(timeout=0.0)
        except Exception as exc:
            self.counts["marker_pull_errors"] += 1
            self._log(logging.WARNING, "marker_pull_error",
                      f"*** WARNING: marker pull failed ({exc}); re-resolving the marker stream. ***",
                      error=repr(exc), stream=self._marker_desc)
            return None
        return [((s[0] if s else ""), float(t)) for s, t in zip(chunk, stamps)]

    def _pull_markers(self, now: float) -> None:
        inlet = self._marker_inlet
        if inlet is None:
            return
        items = self._pull_marker_samples(inlet)
        if items is None:
            try:
                inlet.close_stream()
            except Exception:
                pass
            self._marker_inlet = None
            self._next_marker_probe = now
            return
        self._ingest_markers(items, self._marker_key, now)

    def _ingest_markers(self, items, key: str, now: float) -> None:
        if not items:
            return
        self._last_marker_arrival = now
        for raw, ts in items:
            key = self._ingest_marker(raw, ts, key, now)

    def _bump_run(self, why: str, **ctx) -> str:
        self._marker_run += 1
        uid = (self._marker_desc or {}).get("uid")
        self._marker_key = f"{uid}#{self._marker_run}"
        self.counts["marker_run_restarts_suspected"] += 1
        self._log(logging.WARNING, "marker_run_restart_suspected",
                  f"Marker stream restarted without a new outlet uid ({why}); "
                  "resetting per-outlet duplicate / trial_start tracking.",
                  reason=why, outlet_key=self._marker_key, **ctx)
        return self._marker_key

    def _ingest_marker(self, raw: str, ts: float, key: str, now: float) -> str:
        self.counts["markers_received"] += 1
        ident = (raw, ts)
        if ident in self._recent_marker_set:
            # Identical payload AND timestamp: the same sample delivered twice
            # (e.g. by an auto-recovered old inlet and its replacement).
            self.counts["redelivered_markers"] += 1
            self._log(logging.INFO, "marker_redelivered",
                      "Dropped a marker delivered twice (same payload and timestamp).",
                      console=False, ts=ts, raw=raw)
            return key
        if len(self._recent_marker_ids) == self._recent_marker_ids.maxlen:
            self._recent_marker_set.discard(self._recent_marker_ids[0])
        self._recent_marker_ids.append(ident)
        self._recent_marker_set.add(ident)

        event = parse_marker_json(raw)
        if raw and not event:
            self.counts["unparseable_markers"] += 1
        kind = event.get("event")
        trial_id = _hashable(event.get("trial_id"))
        restart_gap = self.cfg.restart_gap_s

        if kind == "trial_start" and trial_id is not None:
            prev = self._trial_starts.get((key, trial_id))
            if prev is not None and ts - prev > restart_gap:
                key = self._bump_run("trial_start repeated for an already-started trial_id",
                                     trial_id=trial_id, previous_ts=prev, ts=ts)
            self._trial_starts[(key, trial_id)] = ts
        elif kind == "flash":
            flash_idx = _hashable(event.get("flash_idx"))
            prev = self._flash_seen.get((key, trial_id, flash_idx))
            if prev is not None:
                if abs(ts - prev) <= restart_gap:
                    self.counts["duplicate_flash_markers"] += 1
                    if self._detail_ok("duplicate_flash"):
                        self._log(logging.WARNING, "duplicate_flash_marker",
                                  f"*** WARNING: duplicate flash marker (trial {trial_id}, flash "
                                  f"{flash_idx}); keeping only the first in the CSV. ***",
                                  console=self._console_ok("duplicate_flash", now, 2.0),
                                  outlet_key=key, trial_id=trial_id, flash_idx=flash_idx,
                                  first_ts=prev, ts=ts, raw=raw,
                                  count=self.counts["duplicate_flash_markers"])
                    return key
                # Same (trial, flash) seconds later can only be a restarted stimulus
                # (it renumbers from 0): keep it, start a new run.
                key = self._bump_run("flash (trial_id, flash_idx) repeated after "
                                     f"{ts - prev:.1f}s", trial_id=trial_id,
                                     flash_idx=flash_idx, previous_ts=prev, ts=ts)
            self._flash_seen[(key, trial_id, flash_idx)] = ts
            if (key, trial_id) not in self._trial_starts:
                self.counts["flash_without_trial_start"] += 1
                if (key, trial_id) not in self._orphan_trials_logged:
                    self._orphan_trials_logged.add((key, trial_id))
                    self.orphan_trials.append({"outlet_key": key, "trial_id": trial_id,
                                               "first_flash_idx": flash_idx, "first_flash_ts": ts})
                    self._log(logging.WARNING, "flash_without_trial_start",
                              f"*** WARNING: flash for trial {trial_id} with no preceding trial_start "
                              "from this outlet -- that trial has no target label in this CSV. For "
                              "the first trial of a run this is the stimulus pushing trial_start "
                              "before the recorder connected. ***",
                              outlet_key=key, trial_id=trial_id, flash_idx=flash_idx, ts=ts,
                              marker_connections=len(self._marker_connections))

        marker = _Marker(ts=ts, raw=raw, event=event, arrival=now, outlet_key=key)
        bisect.insort(self._pending, marker, key=lambda m: m.ts)
        return key

    # -- marker placement ------------------------------------------------------

    def _place_pending(self, now: float, force: bool) -> None:
        if not self._pending:
            return
        keep = [m for m in self._pending if not self._place_marker(m, now, force)]
        self._pending = keep

    def _place_marker(self, m: _Marker, now: float, force: bool) -> bool:
        """
        Attach to the nearest EEG sample (np.argmin semantics: earliest on ties)
        once that sample is decidable, or write a standalone NaN row. Never drops.
        Returns False only while waiting for EEG that may still contain the nearest sample.
        """
        tol = MAX_SAMPLE_GAP_FACTOR * self._period
        if not force and not self._gap_open:
            if self._max_eeg_ts is None or m.ts > self._max_eeg_ts:
                if now - m.arrival < self.cfg.holdback_s:
                    return False
        best, best_d = None, math.inf
        for row in reversed(self._held):
            if row.kind != "eeg":
                continue
            d = abs(row.ts - m.ts)
            if d <= best_d:
                best, best_d = row, d
        d_written = (abs(self._last_written_eeg_ts - m.ts)
                     if self._last_written_eeg_ts is not None else math.inf)
        if d_written > tol and self._written_eeg_ts and m.ts < self._last_written_eeg_ts:
            # Rare (a marker older than the newest written row): scan recent history.
            d_written = float(np.min(np.abs(np.fromiter(self._written_eeg_ts, dtype=float) - m.ts)))
        if min(best_d, d_written) > tol:
            reason = "no_eeg_sample_within_1.5_periods"
        elif d_written <= best_d:
            reason = "nearest_sample_already_written"
        elif best.fields["marker_json_or_empty"]:
            reason = "nearest_sample_already_has_marker"
        else:
            best.fields["marker_json_or_empty"] = m.raw
            best.fields.update(event_to_row_fields(m.event))
            self.counts["markers_attached"] += 1
            self._max_attach_distance_s = max(self._max_attach_distance_s, best_d)
            return True
        self._standalone_marker(m, reason, min(best_d, d_written), now)
        return True

    def _standalone_marker(self, m: _Marker, reason: str, distance: float, now: float) -> None:
        row = nan_row(m.ts, m.raw, m.event)
        pos = bisect.bisect_right([r.ts for r in self._held], m.ts)
        self._held.insert(pos, _Row(m.ts, "marker", row))
        self.counts["marker_standalone_rows"] += 1
        self.standalone_by_reason[reason] = self.standalone_by_reason.get(reason, 0) + 1
        if self._detail_ok("standalone_marker"):
            self._log(logging.WARNING, "marker_standalone_row",
                      f"*** WARNING: marker {m.event.get('event')!r} (trial {m.event.get('trial_id')}, "
                      f"flash {m.event.get('flash_idx')}) written as its own NaN-channel row: {reason}. ***",
                      console=self._console_ok("standalone_marker", now, 2.0),
                      reason=reason, ts=m.ts, raw=m.raw, nearest_eeg_distance_s=_finite_or_none(distance),
                      in_gap=self._gap_open, outlet_key=m.outlet_key)

    # -- EEG ingestion, gaps, integrity ----------------------------------------

    def _ingest_eeg(self, chunk, stamps, now: float) -> None:
        rows = merge_by_nearest_timestamp(chunk, stamps, [], [])
        jitter_limit = MAX_SAMPLE_GAP_FACTOR * self._period
        for i, (row, ts) in enumerate(zip(rows, stamps)):
            ts = float(ts)
            prev = self._last_eeg_ts
            if prev is None:
                self._log(logging.INFO, "eeg_first_sample", "First EEG sample received.",
                          ts=ts, wall_s_after_connect=round(now - (self._eeg_connected_at or now), 3))
            else:
                step = ts - prev
                if self._gap_open:
                    self._close_gap(prev, ts, now)
                    if step <= 0:
                        self._flag_non_monotonic(prev, ts, i, now)
                elif step <= 0:
                    self._flag_non_monotonic(prev, ts, i, now)
                elif step > self.cfg.gap_jump_s:
                    self._timestamp_jump_gap(prev, ts, i, now)
                elif step > jitter_limit:
                    self.counts["eeg_ts_irregular_steps"] += 1
                    self._max_irregular_step = max(getattr(self, "_max_irregular_step", 0.0), step)
                    if self._detail_ok("irregular_step"):
                        self._log(logging.INFO, "eeg_timestamp_irregular_step",
                                  "EEG timestamp step above 1.5 sample periods (below the gap threshold).",
                                  console=False, prev_ts=prev, ts=ts, step_s=step,
                                  within_chunk=i > 0, count=self.counts["eeg_ts_irregular_steps"])
            self._held.append(_Row(ts, "eeg", row))
            self._last_eeg_ts = ts
            if self._max_eeg_ts is None or ts > self._max_eeg_ts:
                self._max_eeg_ts = ts
            if self._sanity_needed and self._sanity_result is None and len(self._sanity_buf) < self._sanity_needed:
                self._sanity_buf.append(chunk[i])
        self._n_eeg_samples += len(rows)
        self._last_sample_wall = now
        if rows:
            self._update_rate(float(stamps[-1]), len(rows), now)
        if self._sanity_needed and self._sanity_result is None and len(self._sanity_buf) >= self._sanity_needed:
            self._run_sanity(partial=False)

    def _flag_non_monotonic(self, prev: float, ts: float, index_in_chunk: int, now: float) -> None:
        step = ts - prev
        self.counts["eeg_ts_non_monotonic"] += 1
        self.counts["eeg_ts_duplicate" if step == 0 else "eeg_ts_backwards"] += 1
        if self._detail_ok("non_monotonic"):
            self._log(logging.WARNING, "eeg_timestamp_non_monotonic",
                      f"*** WARNING: non-monotonic EEG timestamp (step {step * 1000:.3f} ms, "
                      f"{'within' if index_in_chunk > 0 else 'across'} chunk); kept as received. ***",
                      console=self._console_ok("non_monotonic", now),
                      prev_ts=prev, ts=ts, step_s=step, within_chunk=index_in_chunk > 0,
                      sample_index=self._n_eeg_samples + index_in_chunk,
                      count=self.counts["eeg_ts_non_monotonic"])

    def _check_silence(self, now: float) -> None:
        cfg = self.cfg
        if self._last_sample_wall is None:
            since = now - (self._eeg_connected_at or now)
            last = self._last_no_samples_warn
            if since >= cfg.no_samples_warn_s and (last is None or now - last >= cfg.no_samples_warn_s):
                self._last_no_samples_warn = now
                self._log(logging.WARNING, "eeg_no_samples",
                          f"*** WARNING: EEG stream connected {since:.0f}s ago but no samples received. "
                          "Is the Unicorn LSL Interface actually streaming? ***", waited_s=round(since, 1))
            return
        silence = now - self._last_sample_wall
        if not self._gap_open:
            if silence > cfg.silence_s:
                self._open_gap(now, silence)
        elif now - self._last_gap_ongoing_warn >= cfg.gap_ongoing_warn_s:
            self._last_gap_ongoing_warn = now
            self._log(logging.WARNING, "stream_gap_ongoing",
                      f"*** WARNING: still no EEG samples ({silence:.1f}s); recording continues and "
                      "will resume automatically. ***", silence_s=round(silence, 3))

    def _open_gap(self, now: float, silence: float) -> None:
        prev = self._last_eeg_ts
        # Markers near the last sample can still attach before rows are flushed.
        self._place_pending(now, force=True)
        self._flush(all_rows=True)
        ts_row = prev + self._period / 2.0
        payload = {"event": GAP_START_EVENT, "detected_by": "wall_clock_silence",
                   "last_sample_ts": prev, "silence_s": round(silence, 3)}
        self._held.append(_Row(ts_row, "gap", nan_row(ts_row, json.dumps(payload))))
        self._flush(all_rows=True)   # on disk now: a crash mid-outage still shows the gap
        self._gap_open = True
        self._gap_start_ts = ts_row
        self._gap_opened_wall = self._last_sample_wall
        self._last_gap_ongoing_warn = now
        self.counts["stream_gaps"] += 1
        self._reset_rate()
        self._log(logging.WARNING, "stream_gap_start",
                  f"*** WARNING: EEG stream silent for {silence:.2f}s -- gap start written to CSV. "
                  "Keeping the inlet open; it recovers automatically. ***",
                  last_sample_ts=prev, silence_s=round(silence, 3), gap_index=self.counts["stream_gaps"])

    def _gap_numbers(self, prev: float, ts: float) -> tuple[float, int]:
        span = ts - prev
        return max(0.0, span - self._period), max(0, int(round(span * self._srate)) - 1)

    def _close_gap(self, prev: float, ts: float, now: float) -> None:
        duration, n_missing = self._gap_numbers(prev, ts)
        wall = now - (self._gap_opened_wall or now)
        ts_row = max(ts - self._period / 2.0, self._gap_start_ts)
        payload = {"event": GAP_END_EVENT, "duration_s": round(duration, 4), "n_missing_est": n_missing,
                   "detected_by": "wall_clock_silence", "wall_silence_s": round(wall, 3),
                   "last_sample_ts": prev, "first_sample_ts": ts}
        self._held.append(_Row(ts_row, "gap", nan_row(ts_row, json.dumps(payload))))
        self._gap_open = False
        self.gaps.append({k: payload[k] for k in payload if k != "event"})
        self._log(logging.WARNING, "stream_gap_end",
                  f"EEG resumed after a {duration:.2f}s gap (~{n_missing} samples missing, "
                  f"{wall:.2f}s without data).", **{k: v for k, v in payload.items() if k != "event"})

    def _timestamp_jump_gap(self, prev: float, ts: float, index_in_chunk: int, now: float) -> None:
        duration, n_missing = self._gap_numbers(prev, ts)
        start_ts = prev + self._period / 2.0
        end_ts = max(ts - self._period / 2.0, start_ts)
        start = {"event": GAP_START_EVENT, "detected_by": "timestamp_jump", "last_sample_ts": prev}
        end = {"event": GAP_END_EVENT, "duration_s": round(duration, 4), "n_missing_est": n_missing,
               "detected_by": "timestamp_jump", "last_sample_ts": prev, "first_sample_ts": ts}
        self._held.append(_Row(start_ts, "gap", nan_row(start_ts, json.dumps(start))))
        self._held.append(_Row(end_ts, "gap", nan_row(end_ts, json.dumps(end))))
        self.counts["stream_gaps"] += 1
        self._reset_rate()
        self.gaps.append({k: end[k] for k in end if k != "event"})
        self._log(logging.WARNING, "stream_gap_timestamp_jump",
                  f"*** WARNING: EEG timestamps jumped {ts - prev:.3f}s without a wall-clock stall "
                  f"(~{n_missing} samples missing); gap rows written. ***",
                  console=self._console_ok("ts_jump", now, 2.0), within_chunk=index_in_chunk > 0,
                  **{k: v for k, v in end.items() if k != "event"})

    # -- sample rate and sanity -------------------------------------------------

    def _reset_rate(self) -> None:
        self._rate_pts.clear()
        self._rate_count = 0

    def _update_rate(self, last_ts: float, n: int, now: float) -> None:
        cfg = self.cfg
        self._rate_count += n
        pts = self._rate_pts
        pts.append((last_ts, self._rate_count))
        while len(pts) > 2 and pts[1][0] <= last_ts - cfg.rate_window_s:
            pts.popleft()
        t0, c0 = pts[0]
        span = last_ts - t0
        if span <= 0 or span < 0.9 * cfg.rate_window_s:
            return
        rate = (self._rate_count - c0) / span
        self._rate_last = rate
        self._rate_min = rate if self._rate_min is None else min(self._rate_min, rate)
        self._rate_max = rate if self._rate_max is None else max(self._rate_max, rate)
        deviation = rate / self._srate - 1.0
        if abs(deviation) > cfg.rate_tolerance:
            last = self._last_rate_warn
            if last is None or now - last >= cfg.rate_warn_every_s:
                self._last_rate_warn = now
                self.counts["rate_drift_warnings"] += 1
                self._log(logging.WARNING, "eeg_rate_drift",
                          f"*** WARNING: effective EEG rate {rate:.2f} Hz over the last {span:.1f}s "
                          f"({deviation * 100:+.1f}% vs {self._srate:g} Hz nominal). ***",
                          rate_hz=rate, window_s=span, deviation=deviation, nominal_hz=self._srate)

    def _run_sanity(self, partial: bool) -> None:
        samples = self._sanity_buf
        ok, results = evaluate_sanity(samples)
        self._sanity_result = {"ok": ok, "n_samples": len(samples), "partial": partial, "channels": results}
        self._sanity_buf = []
        lo, hi = SANITY_PTP_MIN_UV, SANITY_PTP_MAX_UV
        for r in results:
            name, ptp, status = r["channel"], r["ptp_uv"], r["status"]
            if status == "ok":
                self._log(logging.INFO, "sanity_channel", f"  {name}: {ptp:.2f} uV p-p (OK)",
                          channel=name, ptp_uv=ptp, status=status)
            elif status == "flat":
                self._log(logging.WARNING, "sanity_channel",
                          f"*** WARNING: channel {name} peak-to-peak = {ptp:.4f} uV -- looks FLAT/"
                          f"disconnected (expected {lo}-{hi} uV). ***", channel=name, ptp_uv=ptp, status=status)
            elif status == "railed":
                self._log(logging.WARNING, "sanity_channel",
                          f"*** WARNING: channel {name} peak-to-peak = {ptp:.4f} uV -- looks CLIPPED/"
                          f"RAILED or in the wrong units (expected {lo}-{hi} uV). ***",
                          channel=name, ptp_uv=ptp, status=status)
            else:
                self._log(logging.WARNING, "sanity_channel",
                          f"*** WARNING: channel {name}: {status.replace('_', '-')} data. ***",
                          channel=name, ptp_uv=ptp, status=status)
        if ok:
            self._log(logging.INFO, "sanity_check",
                      f"Signal sanity check passed ({len(samples)} samples).", **self._sanity_result)
        else:
            self._log(logging.WARNING, "sanity_check",
                      "*** Signal sanity check FAILED for one or more channels -- fix electrode "
                      "contact / stream units before collecting real data. Recording continues "
                      "(not aborted automatically), but do not trust this session's data without "
                      "investigating the warning(s) above. ***", **self._sanity_result)

    # -- writing and shutdown ----------------------------------------------------

    def _flush(self, all_rows: bool = False) -> None:
        held = self._held
        if not held:
            return
        if all_rows:
            k = len(held)
        else:
            horizon = None if self._max_eeg_ts is None else self._max_eeg_ts - self.cfg.holdback_s
            max_held = int(4 * self.cfg.holdback_s * self._srate) + 256
            k = 0
            while k < len(held) and ((horizon is not None and held[k].ts < horizon)
                                     or len(held) - k > max_held):
                k += 1
        if k == 0:
            return
        for row in held[:k]:
            self._writer.writerow(row.fields)
            self.rows_written["total"] += 1
            if row.kind == "eeg":
                self.rows_written["eeg"] += 1
                self._last_written_eeg_ts = row.ts
                self._written_eeg_ts.append(row.ts)
            elif row.kind == "gap":
                self.rows_written["gap"] += 1
            else:
                self.rows_written["marker_standalone"] += 1
        del held[:k]
        self._csv_file.flush()

    def _shutdown(self, code: int) -> None:
        reason = self._stop_reason or "unknown"
        now = self.clock.monotonic()
        if code == EXIT_OK:
            self._pull_markers(now)   # trailing markers (e.g. the last trial_end)
        self._place_pending(now, force=True)
        self._flush(all_rows=True)
        if self._gap_open:
            self.gaps.append({"last_sample_ts": self._last_eeg_ts, "open_at_shutdown": True})
            self._log(logging.WARNING, "stream_gap_open_at_shutdown",
                      "*** WARNING: recording stopped during an EEG gap (no stream_gap_end row). ***",
                      last_sample_ts=self._last_eeg_ts)
        if self._sanity_needed and self._sanity_result is None:
            if self._sanity_buf:
                self._run_sanity(partial=True)
            else:
                self._sanity_result = {"ok": False, "n_samples": 0, "partial": True, "channels": []}
                self._log(logging.WARNING, "sanity_check",
                          "*** WARNING: no EEG samples received, so the signal sanity check never ran. "
                          "Is the Unicorn LSL Interface actually streaming? ***", **self._sanity_result)
        self._csv_file.close()
        self._close_inlets()
        summary = self.integrity_summary(code, reason)
        tmp = self.cfg.integrity_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.cfg.integrity_path)
        self._log(logging.INFO, "integrity_summary",
                  "Integrity: " + (", ".join(summary["issues"]) if summary["issues"] else "no issues found"),
                  counts=self.counts, issues=summary["issues"], path=self.cfg.integrity_path)
        rw = self.rows_written
        self._log(logging.INFO, "shutdown",
                  f"Stopped ({reason}). Wrote {rw['total']} rows ({rw['eeg']} EEG, {rw['gap']} gap, "
                  f"{rw['marker_standalone']} standalone marker) to {self.cfg.out_path}.",
                  reason=reason, exit_code=code, rows_written=rw, csv_path=self.cfg.out_path,
                  integrity_path=self.cfg.integrity_path)

    def integrity_summary(self, code: int, reason: str) -> dict:
        c = self.counts
        issues = []
        for key, label in (("stream_gaps", "EEG gaps"), ("eeg_ts_non_monotonic", "non-monotonic EEG timestamps"),
                           ("rate_drift_warnings", "sample-rate drift warnings"),
                           ("duplicate_flash_markers", "duplicate flash markers"),
                           ("flash_without_trial_start", "flashes without trial_start"),
                           ("marker_standalone_rows", "standalone marker rows"),
                           ("unparseable_markers", "unparseable markers"),
                           ("marker_run_restarts_suspected", "suspected stimulus restarts"),
                           ("marker_timeout_warnings", "marker-stream timeout warnings")):
            if c[key]:
                issues.append(f"{c[key]} {label}")
        if not self._marker_connections:
            issues.append("marker stream never connected")
        if self._sanity_result is not None and not self._sanity_result.get("ok"):
            issues.append("signal sanity check failed")
        return {
            "format": "p300-recorder-integrity-v1",
            "csv_path": self.cfg.out_path,
            "log_path": self.cfg.log_path,
            "started_at": self._started_wall_iso,
            "stopped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "shutdown_reason": reason,
            "exit_code": code,
            "issues": issues,
            "rows_written": self.rows_written,
            "eeg_samples_received": self._n_eeg_samples,
            "eeg_stream": self._eeg_desc,
            "nominal_srate_used_hz": self._srate,
            "inlet_processing_flags": {"eeg": self._processing_names("eeg"),
                                       "marker": self._processing_names("marker")},
            "marker_connections": self._marker_connections,
            "sanity_check": self._sanity_result,
            "counts": c,
            "standalone_marker_rows_by_reason": self.standalone_by_reason,
            "trials_without_trial_start": self.orphan_trials,
            "gaps": self.gaps[:1000],
            "gaps_truncated": len(self.gaps) > 1000,
            "effective_rate_hz": {"min": self._rate_min, "max": self._rate_max, "last": self._rate_last,
                                  "window_s": self.cfg.rate_window_s, "tolerance": self.cfg.rate_tolerance},
            "max_marker_attach_distance_s": self._max_attach_distance_s,
            "max_irregular_step_s": getattr(self, "_max_irregular_step", None),
            "session_ended_in_gap": self._gap_open,
        }


def _hashable(value):
    try:
        hash(value)
        return value
    except TypeError:
        return json.dumps(value, sort_keys=True, default=str)


def _finite_or_none(value: float):
    return value if value is not None and math.isfinite(value) else None


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a Thai P300 speller session from existing "
                    "LSL EEG + marker outlets (does not talk to hardware directly).",
        epilog="Exit codes: 0 normal stop (Ctrl+C / SIGTERM / --duration); 1 unexpected error; "
               "2 bad arguments or output path problem (including an existing --out); "
               "3 EEG stream not found within --eeg-timeout.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path; must not exist (default: "
                             "collection/recordings/session_<timestamp>.csv).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds of recording, counted from when the "
                             "EEG stream is found (default: run until Ctrl+C / SIGTERM).")
    parser.add_argument("--eeg-type", type=str, default=EEG_STREAM_TYPE,
                        help=f"LSL type of the EEG stream (default {EEG_STREAM_TYPE!r}).")
    parser.add_argument("--marker-type", type=str, default=MARKER_STREAM_TYPE,
                        help=f"LSL type of the marker stream (default {MARKER_STREAM_TYPE!r}).")
    parser.add_argument("--eeg-timeout", type=float, default=None,
                        help="Seconds to keep resolving the EEG stream before exiting with code 3 "
                             "(default 30).")
    parser.add_argument("--marker-timeout", type=float, default=120.0,
                        help="Seconds without a marker stream before a loud, repeated warning. "
                             "Recording continues either way (default 120).")
    parser.add_argument("--log-file", type=str, default=None,
                        help="JSON-lines log path (default: <out>.log.jsonl).")
    parser.add_argument("--holdback", type=float, default=1.0,
                        help="Seconds of EEG rows held in memory so late markers reach their "
                             "nearest sample (default 1.0).")
    parser.add_argument("--silence-threshold", type=float, default=0.5,
                        help="Wall-clock seconds without EEG samples that open a stream gap "
                             "(default 0.5).")
    parser.add_argument("--gap-jump", type=float, default=0.1,
                        help="EEG timestamp step (s) that counts as a gap even without a wall-clock "
                             "stall; smaller steps above 1.5 periods are only counted as irregular "
                             "(default 0.1).")
    parser.add_argument("--no-clocksync", action="store_true",
                        help="Do not map stream timestamps into this machine's clock (debugging only).")
    parser.add_argument("--skip-sanity-check", action="store_true",
                        help="Skip the signal amplitude sanity check on the first 2 s "
                             "(not recommended -- for debugging only).")
    parser.add_argument("--poll-interval", type=float, default=None,
                        help="Legacy: now the cap on the EEG resolution backoff interval (default 5).")
    parser.add_argument("--max-wait", type=float, default=None,
                        help="Legacy: used as --eeg-timeout when that is not given.")
    return parser


def _default_out_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings", f"session_{stamp}.csv")


def run_cli(argv=None, backend=None, clock=None, install_signal_handlers: bool = True,
            console: bool = True) -> int:
    """Parse arguments, run one recording, return the exit code. Tests inject backend/clock."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: 2 on bad arguments, 0 on --help
        return exc.code if isinstance(exc.code, int) else EXIT_BAD_ARGS

    def bad(msg: str) -> int:
        print(f"[record_session] ERROR: {msg}", file=sys.stderr)
        return EXIT_BAD_ARGS

    eeg_timeout = args.eeg_timeout if args.eeg_timeout is not None else (
        args.max_wait if args.max_wait is not None else 30.0)
    backoff_max = args.poll_interval if args.poll_interval is not None else 5.0
    for flag, value in (("--eeg-timeout", eeg_timeout), ("--marker-timeout", args.marker_timeout),
                        ("--holdback", args.holdback), ("--silence-threshold", args.silence_threshold),
                        ("--gap-jump", args.gap_jump), ("--poll-interval", backoff_max)):
        if not (math.isfinite(value) and value > 0):
            return bad(f"{flag} must be a positive number, got {value}")
    if args.duration is not None and not (math.isfinite(args.duration) and args.duration > 0):
        return bad(f"--duration must be a positive number, got {args.duration}")
    if not args.eeg_type.strip() or not args.marker_type.strip():
        return bad("--eeg-type and --marker-type must be non-empty")
    if args.eeg_type == args.marker_type:
        return bad("--eeg-type and --marker-type must differ")

    out_path = os.path.abspath(args.out if args.out else _default_out_path())
    if os.path.lexists(out_path):
        return bad(f"output file already exists, refusing to overwrite: {out_path}")
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    except OSError as exc:
        return bad(f"cannot create output directory for {out_path}: {exc}")
    log_path = os.path.abspath(args.log_file) if args.log_file else out_path + ".log.jsonl"
    if log_path == out_path:
        return bad("--log-file must differ from --out")
    try:
        setup_logging(log_path, console=console)
    except OSError as exc:
        return bad(f"cannot open log file {log_path}: {exc}")

    previous_handlers = {}
    try:
        cfg = RecorderConfig(
            out_path=out_path, log_path=log_path,
            eeg_type=args.eeg_type, marker_type=args.marker_type,
            eeg_timeout_s=eeg_timeout, marker_timeout_s=args.marker_timeout,
            backoff_max_s=backoff_max, holdback_s=args.holdback,
            silence_s=args.silence_threshold, gap_jump_s=args.gap_jump,
            session_duration_s=args.duration, skip_sanity_check=args.skip_sanity_check)
        if args.max_wait is not None or args.poll_interval is not None:
            logger.info("Legacy flags mapped: --max-wait -> EEG timeout %gs, --poll-interval -> backoff cap %gs.",
                        eeg_timeout, backoff_max,
                        extra={"event": "legacy_flags", "fields": {"max_wait": args.max_wait,
                                                                    "poll_interval": args.poll_interval}})
        recorder = Recorder(cfg, backend=backend if backend is not None else
                            LslBackend(clocksync=not args.no_clocksync), clock=clock)
        if install_signal_handlers:
            def _handler(signum, _frame):
                recorder.request_stop(signal.Signals(signum).name)
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, _handler)
        return recorder.run()
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True,
                     extra={"event": "unexpected_error", "fields": {"error": repr(exc)}})
        return EXIT_ERROR
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        teardown_logging()


def main():
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
