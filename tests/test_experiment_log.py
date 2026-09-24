"""
Experiment-log tests: the shared file naming and schemas (pipeline/experiment_log.py) and the
session record the panel writes (collection/gui_logic.SessionLogger + SessionOrchestrator).
Every file is written under tmp_path.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline import experiment_log as xlog
from pipeline import manifest as mf

sys.path.insert(0, str(mf.REPO_ROOT / "collection"))
import gui_logic as gl  # noqa: E402

REPO_CHARACTERS = mf.REPO_ROOT / "collection" / "characters_th.json"
SETTINGS = gl.SessionSettings("s1", "กข", 3, 80, 90, 1, True)


@pytest.fixture(scope="module")
def characters():
    return mf.load_characters_file(REPO_CHARACTERS)


# ---------------------------------------------------------------------------
# naming and helpers
# ---------------------------------------------------------------------------

def test_sidecar_names_hang_off_the_recording_path():
    rec = Path("/r/s1_20260923_101500.csv")
    assert xlog.session_record_path(rec) == Path("/r/s1_20260923_101500.csv.session.json")
    assert xlog.inference_log_path(rec) == Path("/r/s1_20260923_101500.csv.inference.log.jsonl")
    assert xlog.recorder_log_path(rec) == Path("/r/s1_20260923_101500.csv.log.jsonl")
    assert xlog.integrity_path(rec) == Path("/r/s1_20260923_101500.csv.integrity.json")
    # the panel's existing helper and the shared one can never disagree
    assert gl.recorder_log_path(rec) == xlog.recorder_log_path(rec)


@pytest.mark.parametrize("checkpoint,expected", [
    ("models/subject_001_classifier.joblib", "models/subject_001_fit_report.json"),
    ("models/demo_s1_classifier.joblib", "models/demo_s1_fit_report.json"),
    ("models/odd.joblib", "models/odd_fit_report.json"),
])
def test_fit_report_path_follows_the_checkpoint_name(checkpoint, expected):
    assert xlog.fit_report_path(checkpoint) == Path(expected)


def test_sig_floats_keeps_magnitude_and_length():
    values = [0.123456789, 1e-8, 0.9999996, 0.0, 1.0]
    out = xlog.sig_floats(values)
    assert out == [0.123457, 1e-08, 1.0, 0.0, 1.0]
    assert len(xlog.sig_floats(np.full(72, 1 / 72))) == 72
    assert xlog.sig_floats(np.array([[0.5, 0.25]])) == [0.5, 0.25]   # any shape is flattened


def test_write_json_atomic_roundtrip_types_and_no_temp_files(tmp_path):
    path = tmp_path / "deep" / "dir" / "x.json"
    data = {"z_first": 1, "a_second": "ก", "np": np.float32(0.5), "n": np.int64(3), "arr": np.arange(3),
            "p": Path("/a/b"), "s": {"y", "x"}}
    xlog.write_json_atomic(data, path)
    text = path.read_text(encoding="utf-8")
    assert "ก" in text                                   # not \u-escaped
    loaded = json.loads(text)
    assert list(loaded)[:2] == ["z_first", "a_second"]   # insertion order, read by people
    assert loaded["np"] == 0.5 and loaded["n"] == 3 and loaded["arr"] == [0, 1, 2]
    assert loaded["p"] == "/a/b" and loaded["s"] == ["x", "y"]
    assert [p.name for p in path.parent.iterdir()] == ["x.json"]


def test_write_json_atomic_failure_leaves_no_partial_file(tmp_path):
    path = tmp_path / "x.json"
    xlog.write_json_atomic({"ok": 1}, path)
    with pytest.raises(TypeError):
        xlog.write_json_atomic({"bad": object()}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": 1}      # the old file survives
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_read_json_if_exists(tmp_path):
    assert xlog.read_json_if_exists(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{oops", encoding="utf-8")
    assert xlog.read_json_if_exists(tmp_path / "bad.json") is None
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert xlog.read_json_if_exists(tmp_path / "list.json") is None
    (tmp_path / "ok.json").write_text('{"a": 1}', encoding="utf-8")
    assert xlog.read_json_if_exists(tmp_path / "ok.json") == {"a": 1}


# ---------------------------------------------------------------------------
# session record
# ---------------------------------------------------------------------------

def test_session_record_lists_the_calibration_sentence_trial_by_trial(characters):
    rec = Path("/r/s1.csv")
    record = xlog.build_session_record(
        session_id="s1", demo=False, targets="กขก", characters=characters,
        settings={"repetitions": 3}, recording=rec, manifest=None, started_utc="2026-09-23T10:15:00+00:00")
    assert record["format"] == xlog.SESSION_FORMAT
    assert record["kind"] == xlog.KIND_CALIBRATION and record["live_inference"] is False
    assert record["targets_text"] == "กขก"
    assert record["target_sequence"] == [
        {"order": 1, "char": "ก", "char_idx": characters.index("ก")},
        {"order": 2, "char": "ข", "char_idx": characters.index("ข")},
        {"order": 3, "char": "ก", "char_idx": characters.index("ก")}]
    assert record["files"] == {"recording": "/r/s1.csv", "recorder_log": "/r/s1.csv.log.jsonl",
                               "integrity": "/r/s1.csv.integrity.json", "inference_log": None}
    assert record["ended_utc"] is None and record["end"] is None and record["manifest"] is None


def test_session_record_with_a_manifest_is_an_inference_session(characters):
    record = xlog.build_session_record(
        session_id="s1", demo=True, targets="ก", characters=characters, settings={},
        recording=Path("/r/demo_s1.csv"), manifest={"path": "/m.json", "fingerprint": "f" * 64},
        started_utc="t")
    assert record["kind"] == xlog.KIND_INFERENCE and record["live_inference"] is True and record["demo"] is True
    assert record["files"]["inference_log"] == "/r/demo_s1.csv.inference.log.jsonl"


def test_finalize_does_not_mutate_the_open_record(characters):
    record = xlog.build_session_record(session_id="s", demo=False, targets="ก", characters=characters,
                                       settings={}, recording=Path("/r/s.csv"), manifest=None, started_utc="t0")
    done = xlog.finalize_session_record(record, ended_utc="t1", message="Stopped.", level="info",
                                        phase_at_end="running")
    assert record["ended_utc"] is None and record["end"] is None
    assert done["started_utc"] == "t0" and done["ended_utc"] == "t1"
    assert done["end"] == {"message": "Stopped.", "level": "info", "phase_at_end": "running"}


# ---------------------------------------------------------------------------
# SessionLogger
# ---------------------------------------------------------------------------

def _manifest(root: Path):
    models = root / "models"
    models.mkdir(exist_ok=True)
    clf = models / "clf.joblib"
    clf.write_bytes(b"fake-classifier")
    cal = models / "platt.json"
    cal.write_text(json.dumps({"format": mf.CALIBRATION_FORMAT, "a": 1.0, "b": 0.0,
                               "score_method": mf.SCORE_METHOD, "classifier_sha256": mf.sha256_file(clf)}))
    data = mf.build_manifest_dict(base_dir=root, characters_path=REPO_CHARACTERS, classifier_path=clf,
                                  calibration_path=cal)
    return mf.write_manifest(data, root / "session_manifest.json")


def test_session_logger_writes_at_start_and_finalizes_at_stop(tmp_path, characters):
    out = tmp_path / "recordings" / "s1_20260923_101500.csv"    # the directory does not exist yet
    logger = gl.SessionLogger(demo=False, characters=characters)
    path = logger.begin(SETTINGS, out, manifest_path=None)
    assert path == xlog.session_record_path(out) and path.is_file()

    opened = json.loads(path.read_text(encoding="utf-8"))
    assert opened["session_id"] == "s1" and opened["kind"] == xlog.KIND_CALIBRATION
    assert opened["targets_text"] == "กข"
    assert [t["char"] for t in opened["target_sequence"]] == ["ก", "ข"]
    assert opened["settings"] == {"repetitions": 3, "flash_ms": 80, "isi_ms": 90, "screen": 1, "windowed": True}
    assert opened["manifest"] is None and opened["ended_utc"] is None and opened["started_utc"]

    assert logger.end("Stimulus finished — stopping recorder.", gl.INFO, gl.RUNNING) == path
    closed = json.loads(path.read_text(encoding="utf-8"))
    assert closed["started_utc"] == opened["started_utc"] and closed["ended_utc"]
    assert closed["end"] == {"message": "Stimulus finished — stopping recorder.", "level": "info",
                             "phase_at_end": "running"}
    assert logger.end("again", gl.INFO, gl.IDLE) is None     # nothing open: a no-op, file untouched
    assert json.loads(path.read_text(encoding="utf-8")) == closed


def test_session_logger_records_the_manifest_it_runs_against(tmp_path, characters):
    manifest = _manifest(tmp_path)
    out = tmp_path / "s1.csv"
    logger = gl.SessionLogger(demo=False, characters=characters)
    logger.begin(SETTINGS, out, manifest_path=manifest.path)
    record = json.loads(xlog.session_record_path(out).read_text(encoding="utf-8"))
    assert record["kind"] == xlog.KIND_INFERENCE
    assert record["manifest"] == {"path": str(manifest.path), "fingerprint": manifest.fingerprint,
                                  "language_model": {"type": "uniform"}}
    assert record["files"]["inference_log"] == str(xlog.inference_log_path(out))


def test_session_logger_survives_an_unreadable_manifest(tmp_path, characters):
    out = tmp_path / "s1.csv"
    logger = gl.SessionLogger(demo=False, characters=characters)
    logger.begin(SETTINGS, out, manifest_path=tmp_path / "missing.json")
    record = json.loads(xlog.session_record_path(out).read_text(encoding="utf-8"))
    assert record["manifest"]["fingerprint"] is None and "not found" in record["manifest"]["error"]


# ---------------------------------------------------------------------------
# orchestrator wiring
# ---------------------------------------------------------------------------

class FakeProc:
    def __init__(self, name, cmd, cwd):
        self.name, self.cmd, self.cwd, self.returncode = name, cmd, cwd, None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def stderr_lines(self):
        return []


class Harness:
    def __init__(self, session_log):
        self.launched, self.messages, self.now = {}, [], 100.0
        self.orch = gl.SessionOrchestrator(
            self.launch, lambda: self.now, lambda m, level: self.messages.append((level, m)), python="PY",
            recorder_ready=lambda out: True, session_log=session_log)

    def launch(self, name, cmd, cwd):
        self.launched[name] = FakeProc(name, cmd, cwd)
        return self.launched[name]


def test_orchestrator_gives_the_service_a_log_file_next_to_the_recording(tmp_path, characters):
    out = tmp_path / "s1.csv"
    h = Harness(gl.SessionLogger(demo=False, characters=characters))
    h.orch.start(SETTINGS, out, manifest_path="/m.json")
    assert h.launched["service"].cmd == ["PY", "-m", "pipeline.inference_service", "--manifest", "/m.json",
                                         "--log-file", str(xlog.inference_log_path(out))]


def test_orchestrator_without_a_session_log_changes_nothing(tmp_path):
    h = Harness(None)
    h.orch.start(SETTINGS, tmp_path / "s1.csv", manifest_path="/m.json")
    assert h.launched["service"].cmd == ["PY", "-m", "pipeline.inference_service", "--manifest", "/m.json"]
    assert list(tmp_path.iterdir()) == []


def test_session_record_ends_with_how_far_the_session_got(tmp_path, characters):
    out = tmp_path / "s1.csv"
    h = Harness(gl.SessionLogger(demo=False, characters=characters))
    h.orch.start(SETTINGS, out, manifest_path="/m.json")
    h.orch.stop("Stopped.")                                   # while still waiting for the service
    record = json.loads(xlog.session_record_path(out).read_text(encoding="utf-8"))
    assert record["end"] == {"message": "Stopped.", "level": "info", "phase_at_end": gl.WAITING_SERVICE}

    out2 = tmp_path / "s2.csv"
    h = Harness(gl.SessionLogger(demo=False, characters=characters))
    h.orch.start(SETTINGS, out2)                              # calibration: recorder only
    h.orch.tick()
    assert h.orch.phase == gl.RUNNING
    h.launched["stimulus"].returncode = 0
    h.orch.tick()
    record = json.loads(xlog.session_record_path(out2).read_text(encoding="utf-8"))
    assert record["kind"] == xlog.KIND_CALIBRATION
    assert record["end"]["phase_at_end"] == gl.RUNNING and "Stimulus finished" in record["end"]["message"]


def test_a_session_log_that_cannot_be_written_never_blocks_the_session(tmp_path):
    class Broken:
        def begin(self, *args):
            raise OSError("disk full")

        def end(self, *args):
            raise OSError("disk full")

    h = Harness(Broken())
    assert h.orch.start(SETTINGS, tmp_path / "s1.csv")
    assert h.orch.phase == gl.WAITING_RECORDER and "recorder" in h.launched
    assert any(level == gl.ERROR and "Could not write the session log (disk full)" in m
               for level, m in h.messages)
    h.orch.stop("Stopped.")
    assert any(level == gl.ERROR and "Could not finish the session log" in m for level, m in h.messages)
    assert h.orch.phase == gl.IDLE
