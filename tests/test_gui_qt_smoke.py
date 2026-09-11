"""
Offscreen smoke tests for collection/session_gui_qt.py.

No LSL and no real processes: the window is built with start_readers=False,
a fake launcher, and a fake clock. Synthetic samples and events go through
the same slots the LSL threads' signals are connected to, and the Qt event
loop is driven by hand (QApplication.processEvents / QTimer), since
pytest-qt is not installed.
"""

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # before any PySide6 import

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import threading  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import gui_logic as gl  # noqa: E402
import session_gui_qt as sgq  # noqa: E402
from pipeline import events as ev  # noqa: E402
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ  # noqa: E402
from pipeline.manifest import ManifestError, load_characters_file  # noqa: E402

CHARACTERS = load_characters_file(gl.CHARACTERS_PATH)
FP = "c" * 64
NOW = datetime(2026, 9, 11, 9, 30, 0)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pump(ms: int = 0):
    """Run the event loop for `ms` milliseconds (at least one pass)."""
    QApplication.processEvents()
    if ms > 0:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()
    QApplication.processEvents()


class FakeClock:
    def __init__(self):
        self.t = 500.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeProc:
    def __init__(self, name, cmd, cwd):
        self.name, self.cmd, self.cwd = name, cmd, cwd
        self.returncode = None
        self.terminated = self.killed = False
        self.lines = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def stderr_lines(self):
        return list(self.lines)


class FakeMarkerOutlet:
    def __init__(self):
        self.samples = []

    def have_consumers(self):
        return True

    def push_sample(self, sample, timestamp):
        self.samples.append((json.loads(sample[0]), timestamp))


class Rig:
    def __init__(self, tmp_path, *, manifest=True, demo=False):
        self.clock = FakeClock()
        self.launched = {}
        self.order = []
        manifest_file = tmp_path / "session_manifest.json"
        if manifest:
            manifest_file.write_text("{}", encoding="utf-8")
        fake_manifest = SimpleNamespace(fingerprint=FP, path=manifest_file,
                                        characters_path=gl.CHARACTERS_PATH)
        self.board_outlet = FakeMarkerOutlet()
        self.window = sgq.SessionWindow(
            demo=demo, manifest_path=manifest_file, manifest_loader=lambda _p: fake_manifest,
            launcher=self.launch, clock=self.clock, now=lambda: NOW,
            recordings_dir=tmp_path / "recordings", start_readers=False, auto_refresh=False,
            board_outlet_factory=lambda: self.board_outlet, board_run_options={"consumer_settle_s": 0.0},
            board_schedule_options={"cue_ms": 20, "inter_selection_ms": 20})
        self.window.resize(1200, 900)
        self.window.show()
        pump()

    def launch(self, name, cmd, cwd):
        proc = FakeProc(name, cmd, cwd)
        self.launched[name] = proc
        self.order.append(name)
        return proc

    def feed_eeg(self, seconds=1.0, flat=(), seed=0):
        """Feed 5-sample chunks (like the Unicorn/mock) through the reader slot, advancing the clock."""
        rng = np.random.default_rng(seed)
        n_chunks = int(seconds * EEG_SFREQ / 5)
        for k in range(n_chunks):
            t = (k * 5 + np.arange(5)) / EEG_SFREQ
            chunk = 20 * np.sin(2 * np.pi * 10 * t)[:, None] + rng.normal(0, 3, (5, 8))
            for ch in flat:
                chunk[:, EEG_CHANNELS.index(ch)] = rng.normal(0, 0.01, 5)
            self.clock.advance(0.02)
            self.window.on_eeg_samples(chunk)

    def connect_eeg(self, **kwargs):
        self.window.on_eeg_stream_found("MockUnicornEEG")
        self.feed_eeg(**kwargs)
        self.window.refresh()
        pump()

    def close(self):
        self.window.close()
        self.window.deleteLater()
        pump()


@pytest.fixture
def rig(app, tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def _rgb(widget):
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = image.width(), image.height(), image.bytesPerLine()
    raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint8)[:bpl * h].reshape(h, bpl)
    return raw[:, :w * 3].reshape(h, w, 3), image.devicePixelRatio()


def _near(rgb, hex_color, tol=14):
    c = QColor(hex_color)
    target = np.array([c.red(), c.green(), c.blue()])
    return np.all(np.abs(rgb.astype(int) - target) <= tol, axis=-1)


def _payload(event):
    return ev.encode(event, characters=CHARACTERS)


def _commit(trial, char, reason=ev.REASON_THRESHOLD_CROSSED, n=12):
    return _payload(ev.character_committed(trial, CHARACTERS.index(char), char, n, reason, characters=CHARACTERS))


# ----------------------------------------------------------------------
# EEG status gating Start
# ----------------------------------------------------------------------

def test_start_disabled_while_searching_or_lost(rig):
    w = rig.window
    w.targets_edit.setText("กข")
    assert w.status_label.text() == "● No EEG stream — start Unicorn LSL Interface"
    assert not w.start_button.isEnabled() and not w.stop_button.isEnabled()
    QTest.keyClick(w.targets_edit, Qt.Key.Key_Return)
    assert rig.launched == {} and "No EEG stream" in w.session_label.text()

    rig.connect_eeg()
    assert w.status_label.text() == "● EEG connected" and w.start_button.isEnabled()

    rig.clock.advance(1.5)
    w.refresh()
    pump()
    assert w.status_label.text() == "● EEG stream lost"
    assert not w.start_button.isEnabled()
    assert all(w.scalp_map.level(ch) == gl.quality.UNKNOWN for ch in EEG_CHANNELS)
    w._on_start()
    assert rig.launched == {} and "EEG stream lost" in w.session_label.text()

    rig.feed_eeg(seconds=0.2)
    w.refresh()
    assert w.start_button.isEnabled()


def test_samples_emitted_from_a_worker_thread_reach_the_gui_thread(rig):
    reader = sgq.LslEegReader("unused-never-started")
    reader.stream_found.connect(rig.window.on_eeg_stream_found, sgq.QUEUED)
    reader.samples.connect(rig.window.on_eeg_samples, sgq.QUEUED)

    def worker():
        reader.stream_found.emit("ThreadedFake")
        for _ in range(50):
            reader.samples.emit(np.ones((5, 8)))
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert len(rig.window.buffer) == 0          # nothing delivered until the GUI loop runs
    pump(20)
    assert len(rig.window.buffer) == 250 and rig.window.connection.stream_name == "ThreadedFake"


# ----------------------------------------------------------------------
# Scalp map
# ----------------------------------------------------------------------

def test_flat_channel_dot_turns_red_after_one_refresh(rig):
    w = rig.window
    w.on_eeg_stream_found("MockUnicornEEG")
    rig.feed_eeg(seconds=1.0, flat=("P3",))
    assert w.scalp_map.level("P3") == gl.quality.UNKNOWN     # not yet refreshed
    w.refresh()
    pump()
    assert w.scalp_map.dot_color("P3") == QColor(gl.QUALITY_RED)
    for ch in EEG_CHANNELS:
        if ch != "P3":
            assert w.scalp_map.dot_color(ch) == QColor(gl.QUALITY_GREEN), ch

    rgb, dpr = _rgb(w.scalp_map)
    for ch, color in (("P3", gl.QUALITY_RED), ("Pz", gl.QUALITY_GREEN)):
        center = w.scalp_map.dot_center(ch)
        assert _near(rgb[int(center.y() * dpr), int(center.x() * dpr)], color), ch
    assert set(w.trace_view._traces) == set(gl.DISPLAY_ORDER)


# ----------------------------------------------------------------------
# Inference display
# ----------------------------------------------------------------------

def test_composed_text_gauge_and_fallback_style(rig):
    w = rig.window
    w.on_inference_stream_found(FP, "P300Inference")
    w.on_inference_payload(_payload(ev.service_status(ev.STATE_READY, "loaded", FP)))
    w.on_inference_payload(_payload(ev.posterior_update(0, 3, 0.42, CHARACTERS.index("ก"), 0.9)))
    pump()
    assert w.gauge.state().max_prob == 0.42 and w.gauge.state().leading_char == "ก"
    assert w.service_label.text() == "Inference service: ready — loaded"

    w.on_inference_payload(_commit(0, "ก"))
    assert w.gauge.state() is None
    w.on_inference_payload(_commit(1, "่"))
    w.on_inference_payload(_payload(ev.flash_rejected(2, 4, ev.REJECT_QUALITY_GATE, ["Pz"])))
    w.on_inference_payload(_commit(2, "อ", reason=ev.REASON_TRIAL_ENDED, n=60))
    w.refresh()
    pump()

    assert w.composed_view.text() == "ก่อ"
    styles = [(s, e, fb) for s, e, fb, _ in w.composed_view.cluster_styles()]
    assert styles == [(0, 2, False), (2, 3, True)]   # ก่ stacks into one cluster
    assert "Rejected flashes: 1 quality gate" in w.rejected_label.text()

    rects = w.composed_view.cluster_rects()
    confident_rect, fallback_rect = rects[0][0], rects[1][0]
    rgb, dpr = _rgb(w.composed_view)
    orange = _near(rgb, gl.COLOR_FALLBACK, tol=24)
    in_fallback = orange[:, int(fallback_rect.left() * dpr):int(fallback_rect.right() * dpr) + 1]
    in_confident = orange[:, :max(0, int(confident_rect.right() * dpr) - 1)]
    assert in_fallback.sum() > 30, "fallback cluster is not drawn in the fallback colour/underline"
    assert in_confident.sum() == 0, "a confident commit picked up the fallback style"

    items = [w.history_list.item(i) for i in range(w.history_list.count())]
    assert [i.text().split()[0] for i in items] == ["✓", "✓", "▲"]
    assert items[2].text().startswith("▲ FALLBACK")
    assert items[2].foreground().color() == QColor(gl.COLOR_FALLBACK) and items[2].font().underline()
    assert "trial_ended after 60 flashes" in items[2].toolTip()
    assert not items[0].font().underline()


def test_malformed_events_are_counted_and_shown(rig):
    w = rig.window
    w.on_inference_stream_found(FP, "P300Inference")
    assert not w.malformed_label.isVisible()
    for bad in ("{oops", '{"event": "posterior_update", "trial_id": 1}', None):
        w.on_inference_payload(bad)
    pump()
    assert w.malformed_label.isVisible()
    assert w.malformed_label.text().startswith("3 malformed inference events")
    w.on_inference_payload(_commit(0, "ก"))
    assert w.composed_view.text() == "ก"


def test_manifest_mismatch_shows_error_and_hides_output(rig):
    w = rig.window
    w.on_inference_stream_found("0000", "P300Inference")
    w.on_inference_payload(_payload(ev.service_status(ev.STATE_READY, "", "0000")))
    w.on_inference_payload(_commit(0, "ก"))
    pump()
    assert w.mismatch_banner.isVisible() and "mismatch" in w.mismatch_banner.text()
    assert w.composed_view.text() == "" and w.history_list.count() == 0

    w.on_inference_stream_dropped("LostError")
    w.on_inference_stream_found(FP, "P300Inference")
    w.on_inference_payload(_commit(0, "ก"))
    pump()
    assert not w.mismatch_banner.isVisible() and w.composed_view.text() == "ก"

    w.on_inference_payload(_payload(ev.service_status(ev.STATE_WARNING, "x", "d" * 64)))
    pump()
    assert w.mismatch_banner.isVisible() and w.composed_view.text() == ""


# ----------------------------------------------------------------------
# Session control
# ----------------------------------------------------------------------

def test_enter_launches_inference_then_recorder_then_stimulus_and_stop_terminates(rig):
    w = rig.window
    rig.connect_eeg()
    assert w.live_check.isEnabled()
    w.live_check.setChecked(True)
    w.board_combo.setCurrentIndex(w.board_combo.findData(sgq.BOARD_SEPARATE))
    w.windowed_check.setChecked(True)
    w.reps_spin.setValue(3)
    w.targets_edit.setText("กข")
    QTest.keyClick(w.targets_edit, Qt.Key.Key_Return)
    assert rig.order == ["service"]
    assert rig.launched["service"].cmd[1:] == ["-m", "pipeline.inference_service", "--manifest",
                                                str(w.manifest.path)]
    assert not w.start_button.isEnabled() and w.stop_button.isEnabled()

    w.on_inference_stream_found(FP, "P300Inference")
    w.on_inference_payload(_payload(ev.service_status(ev.STATE_READY, "", FP)))
    rig.feed_eeg(0.1)
    w.refresh()
    assert rig.order == ["service", "recorder"]
    out = Path(rig.launched["recorder"].cmd[-1])
    assert out.name == "subject_001_20260911_093000.csv"

    rig.feed_eeg(3.1)
    w.refresh()
    assert "stimulus" not in rig.launched     # the recorder has not reported recording_started yet
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".log.jsonl").write_text('{"event": "recording_started"}\n', encoding="utf-8")
    w.refresh()
    stim = rig.launched["stimulus"]
    assert stim.cmd[2:] == ["--targets", "กข", "--n-repetitions", "3", "--flash-ms", "62", "--isi-ms", "62",
                            "--screen", "0", "--windowed"]
    assert all(p.cmd[0] == sys.executable for p in rig.launched.values())
    QTest.mouseClick(w.stop_button, Qt.MouseButton.LeftButton)
    assert all(rig.launched[n].terminated for n in ("service", "recorder", "stimulus"))
    assert w.session_label.text() == "Stopped."
    w.refresh()
    assert w.start_button.isEnabled() and not w.stop_button.isEnabled()


def test_service_not_implemented_exit_is_explained(rig):
    w = rig.window
    rig.connect_eeg()
    w.live_check.setChecked(True)
    w.targets_edit.setText("ก")
    w._on_start()
    service = rig.launched["service"]
    w.on_inference_stream_found(FP, "P300Inference")
    w.on_inference_payload(_payload(ev.service_status(ev.STATE_READY, "", FP)))
    w.refresh()
    assert rig.order == ["service", "recorder"]
    service.returncode = 4
    service.lines = ["inference_service: exit 4: BayesianStoppingController.update() raised "
                     "NotImplementedError at trial 0, flash 3. (log: x.jsonl)",
                     "2026-09-11 16:09:38.241 (  17.442s) [R_P300Inferenc  ] data_receiver.cpp:344 ERR| broke off"]
    w.refresh()
    assert w.session_label.text().startswith(
        "Live inference stopped: BayesianStoppingController.update() is not implemented yet")
    assert "Recording continues" in w.session_label.text() and "ERR|" not in w.session_label.text()
    assert not rig.launched["recorder"].terminated and w.stop_button.isEnabled()


def test_ready_from_a_stream_that_predates_start_does_not_launch_the_recorder(rig):
    w = rig.window
    rig.connect_eeg()
    w.on_inference_stream_found(FP, "P300Inference")                 # already running before Start
    w.on_inference_payload(_payload(ev.service_status(ev.STATE_READY, "old", FP)))
    w.live_check.setChecked(True)
    w.targets_edit.setText("ก")
    w._on_start()
    w.on_inference_payload(_payload(ev.service_status(ev.STATE_READY, "old heartbeat", FP)))
    rig.feed_eeg(0.2)
    w.refresh()
    assert rig.order == ["service"]
    w.on_inference_stream_dropped("LostError")
    w.on_inference_stream_found(FP, "P300Inference")                 # the service this Start launched
    w.on_inference_payload(_payload(ev.service_status(ev.STATE_READY, "heartbeat", FP)))
    rig.feed_eeg(0.1)
    w.refresh()
    assert rig.order == ["service", "recorder"]


@pytest.mark.parametrize("demo,name", [(True, "demo_session_manifest.json"), (False, "session_manifest.json")])
def test_manifest_defaults_to_the_mode_file_at_repo_root(app, tmp_path, monkeypatch, demo, name):
    monkeypatch.chdir(tmp_path)
    seen = []

    def loader(path):
        seen.append(Path(path))
        raise ManifestError("stub loader")
    window = sgq.SessionWindow(demo=demo, manifest_loader=loader, start_readers=False, auto_refresh=False)
    try:
        assert window.manifest_path == REPO_ROOT / name
        assert all(path == REPO_ROOT / name for path in seen)
        assert window.manifest is None and not window.live_check.isEnabled()
        assert sgq.build_arg_parser().parse_args(["--demo"] if demo else []).manifest is None
    finally:
        window.close()
        window.deleteLater()
        pump()


def test_invalid_settings_block_start(rig):
    w = rig.window
    rig.connect_eeg()
    w.session_edit.setText("demo_subject")
    w.targets_edit.setText("กZ")
    w._on_start()
    assert rig.launched == {}
    assert "reserved" in w.session_label.text() and "Z" in w.session_label.text()


def test_closing_the_window_stops_every_process(app, tmp_path):
    r = Rig(tmp_path)
    r.connect_eeg()
    r.window.board_combo.setCurrentIndex(r.window.board_combo.findData(sgq.BOARD_SEPARATE))
    r.window.targets_edit.setText("ก")
    r.window._on_start()
    out = Path(r.launched["recorder"].cmd[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".log.jsonl").write_text('{"event": "recording_started"}\n', encoding="utf-8")
    r.feed_eeg(0.2)
    r.window.refresh()
    assert set(r.launched) == {"recorder", "stimulus"}
    r.close()
    assert r.launched["recorder"].terminated and r.launched["stimulus"].terminated


def test_in_window_board_replaces_the_traces_and_publishes_markers(app, tmp_path):
    r = Rig(tmp_path)
    try:
        w = r.window
        assert w.board_mode() == sgq.BOARD_IN_PANEL
        assert not w.screen_spin.isEnabled() and not w.windowed_check.isEnabled()
        r.connect_eeg()
        w.reps_spin.setValue(1)
        w.flash_spin.setValue(10)
        w.isi_spin.setValue(10)
        w.targets_edit.setText("ก")
        w._on_start()
        out = Path(r.launched["recorder"].cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + ".log.jsonl").write_text('{"event": "recording_started"}\n', encoding="utf-8")
        w.refresh()
        assert "stimulus" not in r.launched
        assert w.board_view.isVisible() and not w.trace_view.isVisible()
        run = w.board_run
        assert run is not None and run.poll() is None
        deadline = time.monotonic() + 15
        while run.poll() is None and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.002)
        assert run.poll() == 0
        events = [m["event"] for m, _ in r.board_outlet.samples]
        assert events[0] == "trial_start" and events[-1] == "trial_end" and events.count("flash") == 24
        w.refresh()
        assert not w.board_view.isVisible() and w.trace_view.isVisible()
        assert r.launched["recorder"].terminated
    finally:
        r.close()


def test_calibration_without_manifest_and_demo_mode(app, tmp_path):
    r = Rig(tmp_path, manifest=False, demo=True)
    try:
        w = r.window
        assert not w.live_check.isEnabled()
        assert w.inference_note.text() == gl.DEMO_MANIFEST_MISSING_MESSAGE
        assert w.demo_banner is not None and w.demo_banner.text() == "DEMO MODE — synthetic data"
        assert w.windowTitle() == "Thai P300 Speller [DEMO]"
        r.connect_eeg()
        w.targets_edit.setText("ก")
        w._on_start()
        assert r.order == ["recorder"]
        assert Path(r.launched["recorder"].cmd[-1]).name.startswith("demo_subject_001_")
    finally:
        r.close()


def test_default_settings_match_tk_panel(rig):
    w = rig.window
    assert (w.session_edit.text(), w.reps_spin.value(), w.flash_spin.value(), w.isi_spin.value(),
            w.screen_spin.value(), w.windowed_check.isChecked()) == ("subject_001", 5, 62, 62, 0, False)
    assert w.demo_banner is None and w.windowTitle() == "Thai P300 Speller"


def test_default_stream_types_give_the_plain_service_command(rig):
    assert rig.window.orchestrator.service_extra_args == []


def test_non_default_stream_types_are_forwarded_to_the_service(app, tmp_path):
    window = sgq.SessionWindow(manifest_path=tmp_path / "none.json", start_readers=False, auto_refresh=False,
                               eeg_type="EEG_t1", inference_type="P300Inference_t1")
    try:
        assert window.orchestrator.service_extra_args == ["--eeg-type", "EEG_t1",
                                                          "--inference-type", "P300Inference_t1"]
    finally:
        window.close()
        window.deleteLater()
        pump()


# ----------------------------------------------------------------------
# Thai rendering
# ----------------------------------------------------------------------

def _ink(image: QImage) -> np.ndarray:
    w, h, bpl = image.width(), image.height(), image.bytesPerLine()
    raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint8)[:bpl * h].reshape(h, bpl)
    return raw[:, :w] < 128


def test_thai_tone_mark_stacks_above_its_base(app):
    family = sgq.thai_font_family()
    if family is None:
        pytest.skip("no verified Thai font installed (neither Sarabun nor Thonburi); "
                    "tone-mark stacking cannot be checked")
    font = sgq.thai_font(72)
    base = _ink(sgq.render_text_image("กอ", font))
    full = _ink(sgq.render_text_image("ก่อ", font))
    assert base.shape == full.shape, (
        f"{family}: adding the tone mark changed the text width, so it was laid out as a separate "
        "(detached) glyph instead of stacking on ก")

    mark = full & ~base
    ys, xs = np.nonzero(mark)
    base_ys, base_xs = np.nonzero(base)
    assert xs.size > 20, f"{family}: no tone-mark ink found"
    assert ys.mean() < base_ys.min(), f"{family}: tone mark is not above the base glyphs"
    assert ys.max() <= base_ys.min() + 3, f"{family}: tone mark overlaps the base glyphs vertically"
    assert base_xs.min() <= xs.min() and xs.max() <= base_xs.max(), (
        f"{family}: tone mark ink falls outside the base glyphs' horizontal span (detached)")
