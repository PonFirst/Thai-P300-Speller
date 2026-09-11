"""
Tests for collection/record_session.py.

Most tests drive the real recording loop on fake LSL inlets and a fake
clock (Recorder takes `backend` and `clock`), so they are deterministic and
take milliseconds. One integration test uses real LSL with a randomized
stream type -- never the production types EEG / Markers / P300Inference,
which other processes on this machine may be publishing.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

import record_session as rs  # noqa: E402

SFREQ = 250.0
PERIOD = 1.0 / SFREQ


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeClock:
    """Shared time for wall clock and LSL timestamps; only advances when asked."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0
        self.sleeps: list[float] = []
        self.hooks: list = []   # callables(now) run after every advance

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        if dt > 0:
            self.t += dt
        for hook in list(self.hooks):
            hook(self.t)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class FakeInfo:
    _n = 0

    def __init__(self, name, stype, channel_count, srate, source_id="src", created_at=0.0):
        FakeInfo._n += 1
        self._name, self._type = name, stype
        self._cc, self._srate, self._source_id = channel_count, srate, source_id
        self._uid = f"uid-{FakeInfo._n}"
        self._created_at = created_at

    def name(self): return self._name
    def type(self): return self._type
    def source_id(self): return self._source_id
    def uid(self): return self._uid
    def hostname(self): return "fakehost"
    def nominal_srate(self): return self._srate
    def channel_count(self): return self._cc
    def created_at(self): return self._created_at


class FakeEEGOutlet:
    """
    A 250 Hz source whose samples become available at their own timestamp
    (+ `latency`). Samples with timestamp inside an outage are never delivered;
    `deliver_after` postpones delivery of everything until that time (a stall).
    """

    def __init__(self, clock: FakeClock, start: float, value_fn=None, latency: float = 0.0):
        self.clock = clock
        self.start = start
        self.next_idx = 0
        self.outages: list[tuple[float, float]] = []
        self.latency = latency
        self.value_fn = value_fn or (lambda i, t: [10.0 * math.sin(0.3 * i + c) for c in range(8)])
        self.ts_override: dict[int, float] = {}
        self.deliver_after: float | None = None

    def ts_of(self, idx: int) -> float:
        return self.ts_override.get(idx, self.start + idx * PERIOD)

    def available(self, now: float):
        chunk, stamps = [], []
        if self.deliver_after is not None and now < self.deliver_after:
            return chunk, stamps
        while self.start + self.next_idx * PERIOD + self.latency <= now:
            idx = self.next_idx
            self.next_idx += 1
            nominal = self.start + idx * PERIOD
            if any(a <= nominal < b for a, b in self.outages):
                continue
            chunk.append(self.value_fn(idx, nominal))
            stamps.append(self.ts_of(idx))
        return chunk, stamps


class FakeEEGInlet:
    def __init__(self, outlet: FakeEEGOutlet, log: list):
        self.outlet, self.log = outlet, log

    def open_stream(self, timeout=None):
        self.log.append(("open_stream", "eeg"))

    def pull_chunk(self, timeout=0.0, max_samples=1024):
        self.log.append(("pull", "eeg"))
        self.outlet.clock.advance(timeout)
        return self.outlet.available(self.outlet.clock.monotonic())

    def close_stream(self):
        self.log.append(("close_stream", "eeg"))


class FakeMarkerOutlet:
    """Markers are (lsl_ts, raw, arrival_time); pulled once arrival_time <= now."""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.queue: list[tuple[float, str, float]] = []

    def push(self, ts: float, event: dict | str, delay: float = 0.0):
        raw = event if isinstance(event, str) else json.dumps(event)
        self.queue.append((ts, raw, ts + delay))


class FakeMarkerInlet:
    def __init__(self, outlet: FakeMarkerOutlet, log: list, name="marker"):
        self.outlet, self.log, self.name = outlet, log, name
        self.opened = False
        self.delivered_upto = 0

    def open_stream(self, timeout=None):
        self.opened = True
        self.log.append(("open_stream", self.name))

    def pull_chunk(self, timeout=0.0, max_samples=1024):
        self.log.append(("pull", self.name))
        now = self.outlet.clock.monotonic()
        samples, stamps, keep = [], [], []
        for item in self.outlet.queue:
            ts, raw, arrival = item
            if arrival <= now:
                samples.append([raw])
                stamps.append(ts)
            else:
                keep.append(item)
        self.outlet.queue = keep
        return samples, stamps

    def close_stream(self):
        self.log.append(("close_stream", self.name))


class FakeBackend:
    """
    `eeg_available_at` / `marker_available_at`: wall time the stream becomes
    resolvable (None = never). resolve() advances the clock by its timeout when
    nothing is found, as the real call would block.
    """

    def __init__(self, clock: FakeClock, eeg_outlet=None, marker_outlet=None,
                 eeg_available_at=0.0, marker_available_at=0.0,
                 eeg_type="EEG", marker_type="Markers"):
        self.clock = clock
        self.eeg_outlet, self.marker_outlet = eeg_outlet, marker_outlet
        self.eeg_available_at, self.marker_available_at = eeg_available_at, marker_available_at
        self.eeg_type, self.marker_type = eeg_type, marker_type
        self.eeg_info = FakeInfo("MockUnicornEEG", eeg_type, 8, SFREQ, "eeg_src")
        self.marker_info = FakeInfo("P300Markers", marker_type, 1, 0.0, "thai_speller_1")
        self.log: list = []
        self.resolve_calls: list = []
        self.marker_inlets: list[FakeMarkerInlet] = []

    def resolve(self, stream_type, timeout):
        now = self.clock.monotonic()
        self.resolve_calls.append((stream_type, timeout, now))
        if stream_type == self.eeg_type:
            ok = self.eeg_available_at is not None and now >= self.eeg_available_at
            info = self.eeg_info
        elif stream_type == self.marker_type:
            ok = self.marker_available_at is not None and now >= self.marker_available_at
            info = self.marker_info
        else:
            ok, info = False, None
        if ok:
            return [info]
        self.clock.advance(timeout)
        return []

    def make_inlet(self, info, kind):
        self.log.append(("make_inlet", kind))
        if info.type() == self.eeg_type:
            return FakeEEGInlet(self.eeg_outlet, self.log)
        inlet = FakeMarkerInlet(self.marker_outlet, self.log, name=f"marker:{info.uid()}")
        self.marker_inlets.append(inlet)
        return inlet


class ScriptedEEGOutlet:
    """Returns one predefined list of timestamps per pull (then nothing)."""

    def __init__(self, clock: FakeClock, chunks: list[list[float]]):
        self.clock = clock
        self.chunks = [list(c) for c in chunks]
        self.next_idx = sum(len(c) for c in chunks)

    def available(self, now: float):
        if not self.chunks:
            return [], []
        stamps = self.chunks.pop(0)
        return [[10.0 * math.sin(40.0 * t + c) for c in range(8)] for t in stamps], stamps


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _detach_log_handlers():
    yield
    rs.teardown_logging()


def make_world(**backend_kw):
    clock = FakeClock(1000.0)
    eeg = FakeEEGOutlet(clock, start=1000.0)
    markers = FakeMarkerOutlet(clock)
    backend = FakeBackend(clock, eeg, markers, **backend_kw)
    return clock, eeg, markers, backend


def run_recorder(tmp_path, clock, backend, duration=5.0, **cfg_kw):
    out = tmp_path / "rec.csv"
    config = rs.RecorderConfig(out_path=str(out), session_duration_s=duration, **cfg_kw)
    rs.setup_logging(config.log_path, console=False)
    recorder = rs.Recorder(config, backend=backend, clock=clock)
    code = recorder.run()
    rs.teardown_logging()
    return code, recorder, out


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def read_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def events(log, name):
    return [r for r in log if r["event"] == name]


def read_summary(out):
    with open(str(out) + ".integrity.json", encoding="utf-8") as f:
        return json.load(f)


def is_nan_row(row):
    return all(row[ch] == "nan" for ch in rs.EEG_CHANNELS)


def nominal(idx):
    return 1000.0 + idx * PERIOD


def marker_event(row):
    return json.loads(row["marker_json_or_empty"]) if row["marker_json_or_empty"] else None


def trial_start(trial_id):
    return {"event": "trial_start", "trial_id": trial_id, "target_char_idx": 15, "target_char": "ก"}


def flash(trial_id, flash_idx, group=(3, 11, 19)):
    return {"event": "flash", "trial_id": trial_id, "flash_idx": flash_idx,
            "group": list(group), "is_target": False}


# ----------------------------------------------------------------------
# (a) normal operation
# ----------------------------------------------------------------------

def test_normal_operation_attaches_markers_to_nearest_sample_including_late_one(tmp_path):
    clock, eeg, markers, backend = make_world()
    pushed = [
        (nominal(250) + 0.0012, trial_start(1), 0.0),     # nearest: 250
        (nominal(375) + 0.0021, flash(1, 0), 0.0),        # nearest: 376
        (nominal(550) + 0.0010, flash(1, 1), 0.6),        # arrives 0.6 s late -> still 550
        (nominal(1000) - 0.0015, {"event": "trial_end", "trial_id": 1}, 0.0),  # nearest: 1000
    ]
    for ts, event, delay in pushed:
        markers.push(ts, event, delay=delay)

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=5.0)

    assert code == rs.EXIT_OK
    header, rows = read_csv(out)
    assert header == rs.CSV_COLUMNS
    assert not any(is_nan_row(r) for r in rows)
    assert len(rows) == eeg.next_idx > 1200          # every delivered sample, held rows flushed
    stamps = [float(r["timestamp"]) for r in rows]
    assert stamps[0] == nominal(0)
    assert all(b > a for a, b in zip(stamps, stamps[1:]))

    with_marker = [r for r in rows if r["marker_json_or_empty"]]
    assert [float(r["timestamp"]) for r in with_marker] == [nominal(250), nominal(376), nominal(550), nominal(1000)]
    assert [marker_event(r)["event"] for r in with_marker] == ["trial_start", "flash", "flash", "trial_end"]
    assert with_marker[0]["target_char_idx"] == "15"
    assert with_marker[1]["flash_idx"] == "0" and with_marker[1]["group_json"] == "[3, 11, 19]"
    assert with_marker[2]["flash_idx"] == "1" and with_marker[2]["is_target"] == "False"
    assert with_marker[3]["trial_id"] == "1" and with_marker[3]["group_json"] == ""

    # Identical to the reference whole-recording merge.
    samples = [[float(r[ch]) for ch in rs.EEG_CHANNELS] for r in rows]
    reference = rs.merge_by_nearest_timestamp(
        samples, stamps, [[json.dumps(e)] for _, e, _ in pushed], [ts for ts, _, _ in pushed])
    assert [r["marker_json_or_empty"] for r in reference] == [r["marker_json_or_empty"] for r in rows]

    # open_stream() immediately after the marker inlet is created, then a priming pull.
    i = backend.log.index(("make_inlet", "marker"))
    name = backend.marker_inlets[0].name
    assert backend.log[i + 1] == ("open_stream", name)
    assert backend.log[i + 2] == ("pull", name)

    summary = read_summary(out)
    assert summary["exit_code"] == 0 and summary["shutdown_reason"] == "duration_elapsed"
    assert summary["counts"]["markers_attached"] == 4
    assert summary["counts"]["marker_standalone_rows"] == 0
    assert summary["issues"] == []
    assert summary["rows_written"]["eeg"] == len(rows)


def test_eeg_recorded_before_marker_stream_appears(tmp_path):
    clock, eeg, markers, backend = make_world(marker_available_at=1002.0)
    markers.push(1002.3, trial_start(0))

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=4.0)

    assert code == 0
    _, rows = read_csv(out)
    assert float(rows[0]["timestamp"]) == nominal(0)
    connected = events(read_log(rec.cfg.log_path), "marker_connected")
    assert len(connected) == 1 and 2.0 <= connected[0]["waited_s"] < 2.5
    probes = [c for c in backend.resolve_calls if c[0] == "Markers"]
    assert probes and all(timeout <= 0.1 for _, timeout, _ in probes)
    assert sum(1 for r in rows if r["marker_json_or_empty"]) == 1


def test_sanity_check_runs_on_recorded_samples_and_flags_each_channel():
    rng_seeds = [0, 1, 2]
    for seed in rng_seeds:
        rng = __import__("numpy").random.default_rng(seed)
        for ch_idx, ch_name in enumerate(rs.EEG_CHANNELS):
            for fault, expected in (("flat", "flat"), ("railed", "railed")):
                data = rng.normal(0.0, 10.0, size=(500, 8))
                if fault == "flat":
                    data[:, ch_idx] = 3.0 + rng.normal(0.0, 0.01, size=500)
                else:
                    data[:, ch_idx] = 300.0 * __import__("numpy").sin(__import__("numpy").arange(500) * 0.37)
                ok, results = rs.evaluate_sanity(data)
                assert not ok
                statuses = {r["channel"]: r["status"] for r in results}
                assert statuses[ch_name] == expected, (seed, ch_name, fault)
                assert all(s == "ok" for c, s in statuses.items() if c != ch_name), (seed, ch_name, fault)
    ok, results = rs.evaluate_sanity([[1.0] * 6] * 10)
    assert not ok and [r["status"] for r in results[-2:]] == ["missing", "missing"]


def test_sanity_check_is_not_a_blocking_pre_pull(tmp_path):
    clock, eeg, markers, backend = make_world()
    p3 = rs.EEG_CHANNELS.index("P3")
    eeg.value_fn = lambda i, t: [0.001 if c == p3 else 10.0 * math.sin(0.3 * i + c) for c in range(8)]

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=3.0)

    assert code == 0
    _, rows = read_csv(out)
    assert float(rows[0]["timestamp"]) == nominal(0)      # the first 2 s are in the CSV
    sanity = events(read_log(rec.cfg.log_path), "sanity_check")
    assert len(sanity) == 1 and sanity[0]["ok"] is False and sanity[0]["n_samples"] == 500
    assert {c["channel"]: c["status"] for c in sanity[0]["channels"]}["P3"] == "flat"
    assert "signal sanity check failed" in read_summary(out)["issues"]


def test_inlet_processing_flags_follow_shared_dejitter_switch(tmp_path, monkeypatch):
    assert rs.inlet_processing_names("marker") == ["proc_clocksync"]
    assert rs.inlet_processing_names("eeg", clocksync=False) == (
        ["proc_dejitter"] if rs.EEG_INLET_DEJITTER else [])
    monkeypatch.setattr(rs, "EEG_INLET_DEJITTER", False)
    assert rs.inlet_processing_names("eeg") == ["proc_clocksync"]
    monkeypatch.setattr(rs, "EEG_INLET_DEJITTER", True)
    assert rs.inlet_processing_names("eeg") == ["proc_clocksync", "proc_dejitter"]
    assert rs.inlet_processing_names("marker") == ["proc_clocksync"]
    monkeypatch.setattr(rs, "EEG_INLET_DEJITTER", False)

    clock, eeg, markers, backend = make_world()
    backend.processing_flag_names = lambda kind: rs.inlet_processing_names(kind)
    code, rec, out = run_recorder(tmp_path, clock, backend, duration=1.0)
    assert code == 0
    assert read_summary(out)["inlet_processing_flags"] == {"eeg": ["proc_clocksync"],
                                                           "marker": ["proc_clocksync"]}
    log = read_log(rec.cfg.log_path)
    assert events(log, "eeg_connected")[0]["processing_flags"] == ["proc_clocksync"]
    assert events(log, "marker_connected")[0]["processing_flags"] == ["proc_clocksync"]


# ----------------------------------------------------------------------
# (b) EEG never appears
# ----------------------------------------------------------------------

def test_eeg_never_found_exits_3_within_timeout_with_growing_backoff(tmp_path, capsys):
    clock, eeg, markers, backend = make_world(eeg_available_at=None)
    out = tmp_path / "never.csv"
    wall0 = time.monotonic()

    code = rs.run_cli(["--out", str(out), "--eeg-timeout", "30"], backend=backend, clock=clock,
                      install_signal_handlers=False)

    assert code == rs.EXIT_EEG_NOT_FOUND == 3
    assert 30.0 - 1e-6 <= clock.t - 1000.0 <= 31.0
    assert time.monotonic() - wall0 < 5.0                  # fake clock: no real waiting
    assert not out.exists()
    attempts = events(read_log(str(out) + ".log.jsonl"), "eeg_resolve_attempt")
    retries = [a["next_retry_s"] for a in attempts]
    assert retries[:4] == [0.5, 1.0, 2.0, 4.0]
    assert max(retries) <= 5.0
    assert all(b >= a for a, b in zip(retries[:-2], retries[1:-1]))
    assert all(a["timeout_s"] == 30.0 for a in attempts)
    elapsed = [a["elapsed_s"] for a in attempts]
    assert elapsed == sorted(elapsed)
    err_lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(err_lines) == 1 and "no EEG stream" in err_lines[0]


# ----------------------------------------------------------------------
# (c) dropout and recovery
# ----------------------------------------------------------------------

def test_silence_dropout_writes_gap_rows_and_recording_continues(tmp_path):
    clock, eeg, markers, backend = make_world()
    eeg.outages.append((1002.0, 1004.0))
    markers.push(1001.0, trial_start(1))
    markers.push(1003.0, flash(1, 0))                      # inside the gap
    out = tmp_path / "rec.csv"
    seen = {}

    def gap_on_disk(now):
        if now >= 1003.5 and "checked" not in seen:
            seen["checked"] = True
            seen["gap_start_on_disk"] = "stream_gap_start" in out.read_text(encoding="utf-8")
    clock.hooks.append(gap_on_disk)

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=7.0)

    assert code == 0
    assert seen["gap_start_on_disk"] is True                # visible even if we crashed mid-outage
    _, rows = read_csv(out)
    kinds = [(marker_event(r) or {}).get("event") for r in rows]
    gs, ge = kinds.index("stream_gap_start"), kinds.index("stream_gap_end")
    assert gs < ge
    for r in (rows[gs], rows[ge]):
        assert is_nan_row(r)
        assert all(r[c] == "" for c in ("trial_id", "flash_idx", "group_json", "is_target", "target_char_idx"))
    end = marker_event(rows[ge])
    assert end["duration_s"] == pytest.approx(2.0, abs=0.01)
    assert end["n_missing_est"] == 500
    assert end["last_sample_ts"] == pytest.approx(nominal(499))
    assert end["first_sample_ts"] == pytest.approx(nominal(1000))

    eeg_ts = [float(r["timestamp"]) for r in rows if not is_nan_row(r)]
    assert not any(1002.0 <= t < 1004.0 for t in eeg_ts)
    assert all(float(r["timestamp"]) < 1002.0 for r in rows[:gs])
    assert sum(1 for t in eeg_ts if t >= 1004.0) >= 2.5 * SFREQ      # post-gap data written

    in_gap = [r for r in rows[gs:ge] if r["flash_idx"] == "0"]
    assert len(in_gap) == 1 and is_nan_row(in_gap[0]) and float(in_gap[0]["timestamp"]) == 1003.0
    assert in_gap[0]["trial_id"] == "1" and in_gap[0]["group_json"] == "[3, 11, 19]"

    log = read_log(rec.cfg.log_path)
    assert len(events(log, "stream_gap_start")) == 1
    assert events(log, "stream_gap_end")[0]["duration_s"] == pytest.approx(2.0, abs=0.01)
    summary = read_summary(out)
    assert summary["counts"]["stream_gaps"] == 1
    assert summary["gaps"][0]["n_missing_est"] == 500
    assert summary["standalone_marker_rows_by_reason"] == {"no_eeg_sample_within_1.5_periods": 1}


def test_timestamp_jump_without_stall_is_a_gap(tmp_path):
    clock, eeg, markers, backend = make_world()
    eeg.outages.append((1002.0, 1002.2))                   # 0.2 s: below the 0.5 s silence threshold

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=4.0)

    assert code == 0
    _, rows = read_csv(out)
    gap_events = [marker_event(r) for r in rows if is_nan_row(r)]
    assert [g["event"] for g in gap_events] == ["stream_gap_start", "stream_gap_end"]
    assert all(g["detected_by"] == "timestamp_jump" for g in gap_events)
    assert gap_events[1]["n_missing_est"] == 50
    assert gap_events[1]["duration_s"] == pytest.approx(0.2, abs=0.005)


# ----------------------------------------------------------------------
# (d) integrity checks
# ----------------------------------------------------------------------

def test_non_monotonic_timestamps_flagged_not_reordered_or_dropped(tmp_path):
    clock = FakeClock(1000.0)
    chunks, t = [], 1000.0
    for _ in range(60):
        chunks.append([t + k * PERIOD for k in range(25)])
        t += 25 * PERIOD
    chunks[2][0] = chunks[1][-1]                  # duplicate timestamp across chunks
    chunks[4][10] = chunks[4][9] - 0.001          # backwards step within a chunk
    eeg = ScriptedEEGOutlet(clock, chunks)
    backend = FakeBackend(clock, eeg, FakeMarkerOutlet(clock))

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=5.5)

    assert code == 0
    _, rows = read_csv(out)
    expected = [ts for c in chunks for ts in c][:len(rows)]
    assert len(rows) == 55 * 25
    assert [float(r["timestamp"]) for r in rows] == expected        # as received
    log = read_log(rec.cfg.log_path)
    flagged = events(log, "eeg_timestamp_non_monotonic")
    assert len(flagged) == 2
    assert {f["within_chunk"] for f in flagged} == {True, False}
    counts = read_summary(out)["counts"]
    assert counts["eeg_ts_non_monotonic"] == 2
    assert counts["eeg_ts_duplicate"] == 1 and counts["eeg_ts_backwards"] == 1
    assert "2 non-monotonic EEG timestamps" in read_summary(out)["issues"]


def test_duplicate_redelivered_and_orphan_flash_markers(tmp_path):
    clock, eeg, markers, backend = make_world()
    markers.push(1001.0, trial_start(0))
    markers.push(1001.5, flash(0, 3))
    markers.push(1001.6, flash(0, 3))             # duplicate key 0.1 s later -> dropped
    markers.push(1001.7, flash(0, 4))
    markers.push(1001.7, flash(0, 4), delay=0.05) # same payload + timestamp -> redelivery
    markers.push(1002.0, flash(5, 0))             # trial 5 never started
    markers.push(1004.0, flash(0, 3))             # same key 2.5 s later: restarted stimulus, kept

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=6.0)

    assert code == 0
    _, rows = read_csv(out)
    f03 = [float(r["timestamp"]) for r in rows if r["trial_id"] == "0" and r["flash_idx"] == "3"]
    assert f03 == [pytest.approx(nominal(375)), pytest.approx(1004.0)]
    assert sum(1 for r in rows if r["trial_id"] == "0" and r["flash_idx"] == "4") == 1
    log = read_log(rec.cfg.log_path)
    dup = events(log, "duplicate_flash_marker")
    assert len(dup) == 1 and dup[0]["first_ts"] == 1001.5 and dup[0]["ts"] == 1001.6
    orphans = events(log, "flash_without_trial_start")
    assert [o["trial_id"] for o in orphans] == [5, 0]
    assert len(events(log, "marker_run_restart_suspected")) == 1
    summary = read_summary(out)
    c = summary["counts"]
    assert c["duplicate_flash_markers"] == 1
    assert c["redelivered_markers"] == 1
    assert c["flash_without_trial_start"] == 2
    assert [t["trial_id"] for t in summary["trials_without_trial_start"]] == [5, 0]


def test_late_and_colliding_markers_written_as_own_rows(tmp_path):
    clock, eeg, markers, backend = make_world()
    markers.push(1001.0, trial_start(0))
    markers.push(nominal(400) + 0.0005, flash(0, 1))      # same nearest sample as flash 2 ...
    markers.push(nominal(400) - 0.0005, flash(0, 2))      # ... which is earlier, so it gets the sample
    markers.push(1001.9, flash(0, 3), delay=2.0)          # arrives after its sample was written

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=5.0)

    assert code == 0
    _, rows = read_csv(out)
    by_flash = {r["flash_idx"]: r for r in rows if r["flash_idx"] != ""}
    assert not is_nan_row(by_flash["2"]) and float(by_flash["2"]["timestamp"]) == nominal(400)
    assert is_nan_row(by_flash["1"]) and float(by_flash["1"]["timestamp"]) == nominal(400) + 0.0005
    assert is_nan_row(by_flash["3"]) and float(by_flash["3"]["timestamp"]) == 1001.9
    reasons = read_summary(out)["standalone_marker_rows_by_reason"]
    assert reasons == {"nearest_sample_already_has_marker": 1, "nearest_sample_already_written": 1}
    assert len(events(read_log(rec.cfg.log_path), "marker_standalone_row")) == 2


def test_marker_outlet_replaced_reconnects_and_resets_tracking(tmp_path):
    clock, eeg, markers, backend = make_world()
    markers.push(1000.5, trial_start(0))
    markers.push(1000.6, flash(0, 0))
    old_uid = backend.marker_info.uid()

    def restart_stimulus(now):
        if now >= 1002.0 and backend.marker_info.uid() == old_uid:
            backend.marker_info = FakeInfo("P300Markers", "Markers", 1, 0.0, "other_src", created_at=1002.0)
    clock.hooks.append(restart_stimulus)
    markers.push(1004.0, trial_start(0))                   # new run renumbers from 0
    markers.push(1004.1, flash(0, 0))

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=6.0)

    assert code == 0
    log = read_log(rec.cfg.log_path)
    assert len(events(log, "marker_outlet_replaced")) == 1
    new_uid = backend.marker_info.uid()
    assert ("open_stream", f"marker:{new_uid}") in backend.log
    assert ("close_stream", f"marker:{old_uid}") in backend.log
    c = read_summary(out)["counts"]
    assert c["marker_reconnections"] == 1
    assert c["duplicate_flash_markers"] == 0 and c["flash_without_trial_start"] == 0
    _, rows = read_csv(out)
    assert sum(1 for r in rows if r["flash_idx"] == "0") == 2


# ----------------------------------------------------------------------
# Shutdown, output path, marker timeout
# ----------------------------------------------------------------------

def test_sigterm_flushes_held_rows_and_writes_summary(tmp_path):
    clock, eeg, markers, backend = make_world()
    markers.push(1001.0, trial_start(0))
    markers.push(1002.95, flash(0, 0))                     # still held when SIGTERM arrives
    out = tmp_path / "term.csv"
    fired = {}

    def send_sigterm(now):
        if now >= 1003.0 and not fired:
            fired["at"] = now
            os.kill(os.getpid(), signal.SIGTERM)
    clock.hooks.append(send_sigterm)
    previous = signal.getsignal(signal.SIGTERM)

    code = rs.run_cli(["--out", str(out)], backend=backend, clock=clock, console=False)

    assert code == 0
    assert signal.getsignal(signal.SIGTERM) is previous
    _, rows = read_csv(out)
    assert len(rows) == eeg.next_idx                        # nothing held back was lost
    assert float(rows[-1]["timestamp"]) >= 1002.9
    assert any(r["flash_idx"] == "0" and not is_nan_row(r) for r in rows)
    summary = read_summary(out)
    assert summary["shutdown_reason"] == "SIGTERM" and summary["exit_code"] == 0
    assert summary["rows_written"]["eeg"] == eeg.next_idx
    assert events(read_log(str(out) + ".log.jsonl"), "shutdown")[0]["reason"] == "SIGTERM"


def test_existing_output_is_refused_with_exit_2(tmp_path, capsys):
    clock, eeg, markers, backend = make_world()
    out = tmp_path / "exists.csv"
    out.write_text("precious data\n", encoding="utf-8")

    code = rs.run_cli(["--out", str(out)], backend=backend, clock=clock, install_signal_handlers=False)

    assert code == rs.EXIT_BAD_ARGS == 2
    assert out.read_text(encoding="utf-8") == "precious data\n"
    assert "refusing to overwrite" in capsys.readouterr().err
    assert backend.resolve_calls == []
    assert rs.run_cli(["--out", str(tmp_path / "x.csv"), "--eeg-timeout", "-1"], backend=backend,
                      clock=clock, install_signal_handlers=False) == 2


def test_marker_timeout_warns_repeatedly_but_recording_continues(tmp_path):
    clock, eeg, markers, backend = make_world(marker_available_at=None)

    code, rec, out = run_recorder(tmp_path, clock, backend, duration=5.0,
                                  marker_timeout_s=2.0, marker_timeout_repeat_s=1.0)

    assert code == 0
    warnings = events(read_log(rec.cfg.log_path), "marker_timeout")
    assert len(warnings) >= 2 and all(w["level"] == "WARNING" for w in warnings)
    _, rows = read_csv(out)
    assert len(rows) == eeg.next_idx > 1200
    summary = read_summary(out)
    assert summary["counts"]["marker_timeout_warnings"] == len(warnings)
    assert "marker stream never connected" in summary["issues"]
    assert any(c[0] == "Markers" and c[2] > 1004.0 for c in backend.resolve_calls)  # still searching


# ----------------------------------------------------------------------
# Real LSL: outlet killed and restarted with the same source_id
# ----------------------------------------------------------------------

_OUTLET_SCRIPT = textwrap.dedent("""
    import sys, time
    from pylsl import StreamInfo, StreamOutlet, local_clock
    stype, source_id = sys.argv[1], sys.argv[2]
    info = StreamInfo("RecorderTestEEG", stype, 8, 250.0, "float32", source_id)
    outlet = StreamOutlet(info)
    period, n = 1.0 / 250.0, 5
    next_t = local_clock()
    i = 0
    while True:
        if local_clock() < next_t + (n - 1) * period:
            time.sleep(0.004)
            continue
        chunk = [[float((i + k) % 40)] * 8 for k in range(n)]
        outlet.push_chunk(chunk, timestamp=next_t + (n - 1) * period)
        next_t += n * period
        i += n
""")


def _wait_for_event(log_path: Path, name: str, deadline_s: float) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if log_path.exists():
            try:
                if events(read_log(log_path), name):
                    return True
            except json.JSONDecodeError:
                pass
        time.sleep(0.1)
    return False


def test_real_lsl_outlet_restart_records_gap_and_resumes(tmp_path):
    token = uuid.uuid4().hex
    eeg_type, marker_type = f"EEGTEST_{token}", f"MKTEST_{token}"
    source_id = f"recorder_test_{token}"
    script = tmp_path / "outlet.py"
    script.write_text(_OUTLET_SCRIPT, encoding="utf-8")
    out = tmp_path / "real.csv"
    log_path = Path(str(out) + ".log.jsonl")
    procs = []

    def spawn(args, **kw):
        proc = subprocess.Popen(args, **kw)
        procs.append(proc)
        return proc

    try:
        outlet_a = spawn([sys.executable, str(script), eeg_type, source_id],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        recorder = spawn([sys.executable, str(REPO_ROOT / "collection" / "record_session.py"),
                          "--out", str(out), "--eeg-type", eeg_type, "--marker-type", marker_type,
                          "--eeg-timeout", "10", "--marker-timeout", "60"],
                         cwd=str(REPO_ROOT), stdout=open(tmp_path / "rec.out", "w"),
                         stderr=open(tmp_path / "rec.err", "w"))
        assert _wait_for_event(log_path, "eeg_first_sample", 10.0), (tmp_path / "rec.err").read_text()
        time.sleep(1.0)
        outlet_a.kill()
        outlet_a.wait()
        time.sleep(1.5)
        spawn([sys.executable, str(script), eeg_type, source_id],
              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert _wait_for_event(log_path, "stream_gap_end", 10.0), (tmp_path / "rec.err").read_text()
        time.sleep(1.0)
        recorder.send_signal(signal.SIGTERM)
        assert recorder.wait(timeout=10) == 0, (tmp_path / "rec.err").read_text()
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    _, rows = read_csv(out)
    gap_rows = [(i, marker_event(r)) for i, r in enumerate(rows) if is_nan_row(r)]
    names = [e["event"] for _, e in gap_rows]
    assert "stream_gap_start" in names and "stream_gap_end" in names
    end_idx, end = next((i, e) for i, e in gap_rows if e["event"] == "stream_gap_end")
    assert 1.0 <= end["duration_s"] <= 8.0
    assert end["n_missing_est"] > 200
    resumed = [r for r in rows[end_idx + 1:] if not is_nan_row(r)]
    assert len(resumed) >= 100
    assert float(resumed[0]["timestamp"]) >= end["first_sample_ts"] - 1e-9
    summary = read_summary(out)
    assert summary["shutdown_reason"] == "SIGTERM" and summary["exit_code"] == 0
    assert summary["counts"]["stream_gaps"] >= 1
