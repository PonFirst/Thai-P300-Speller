import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

import board_qt as bq  # noqa: E402
import board_schedule as bs  # noqa: E402
from pipeline.manifest import load_characters_file  # noqa: E402

CHARS = load_characters_file(REPO_ROOT / "collection" / "characters_th.json")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeOutlet:
    def __init__(self, consumers=True):
        self.consumers = consumers
        self.samples = []

    def have_consumers(self):
        return self.consumers

    def push_sample(self, sample, timestamp):
        self.samples.append((json.loads(sample[0]), timestamp))


def _run_until(predicate, timeout_s):
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.001)
    return predicate()


def _board():
    board = bq.BoardView(CHARS)
    board.resize(900, 640)
    board.show()
    QApplication.processEvents()
    return board


def _tiny_schedule(targets="ก", reps=1, flash=12, isi=12, cue=30, inter=30):
    return bs.build_session_schedule(CHARS, targets, n_repetitions=reps, flash_ms=flash, isi_ms=isi,
                                     cue_ms=cue, inter_selection_ms=inter)


def _pixel(board, idx):
    image = board.grab().toImage()
    rect = board.cell_rect(idx)
    ratio = image.devicePixelRatio()
    return image.pixelColor(int((rect.left() + rect.width() * 0.12) * ratio),
                            int((rect.top() + rect.height() * 0.12) * ratio)).name()


def test_flashing_cells_are_white_boxes_and_the_cue_is_lime(app):
    board = _board()
    try:
        board.set_state(flashing=[0, 10], cue_idx=5)
        board.repaint()
        assert _pixel(board, 0) == bq.COLOR_FLASH_BG and _pixel(board, 10) == bq.COLOR_FLASH_BG
        assert _pixel(board, 1) == bq.COLOR_BOARD_BG
        assert board.cell_colors(5) == (bq.COLOR_BOARD_BG, bq.COLOR_CUE)
        board.clear()
        assert board.cell_colors(0) == (bq.COLOR_BOARD_BG, bq.COLOR_IDLE)
    finally:
        board.close()


def test_board_rejects_wrong_character_count(app):
    with pytest.raises(ValueError, match="characters"):
        bq.BoardView(CHARS[:71])


def test_session_pushes_every_marker_in_order_on_schedule_then_finishes(app):
    board = _board()
    schedule = _tiny_schedule("กข")
    outlet = FakeOutlet()
    try:
        run = bq.EmbeddedBoardRun(board, schedule, outlet, consumer_settle_s=0.0).start()
        assert run.poll() is None
        assert _run_until(lambda: run.poll() is not None, 15)
        assert run.poll() == bq.EXIT_FINISHED
        expected = [s.marker for s in schedule if s.marker is not None]
        assert [m for m, _ in outlet.samples] == expected
        stamps = np.array([t for _, t in outlet.samples])
        assert np.all(np.diff(stamps) >= 0)
        planned = np.array([s.at_s for s in schedule if s.marker is not None])
        assert np.max(np.abs((stamps - stamps[0]) - (planned - planned[0]))) < 0.05
        assert board.flashing == frozenset() and board.cue_idx is None
    finally:
        board.close()


def test_terminate_stops_markers_and_clears_the_board(app):
    board = _board()
    outlet = FakeOutlet()
    try:
        run = bq.EmbeddedBoardRun(board, _tiny_schedule(reps=5, flash=20, isi=20, cue=10, inter=10), outlet,
                                  consumer_settle_s=0.0).start()
        assert _run_until(lambda: len(outlet.samples) >= 5, 15)
        run.terminate()
        pushed = len(outlet.samples)
        _run_until(lambda: False, 0.3)
        assert run.poll() == bq.EXIT_TERMINATED
        assert len(outlet.samples) == pushed
        assert board.flashing == frozenset() and board.cue_idx is None
    finally:
        board.close()


def test_first_marker_waits_for_a_consumer(app):
    board = _board()
    outlet = FakeOutlet(consumers=False)
    try:
        run = bq.EmbeddedBoardRun(board, _tiny_schedule(), outlet, consumer_wait_s=10.0,
                                  consumer_settle_s=0.05).start()
        _run_until(lambda: False, 0.3)
        assert outlet.samples == [] and run.poll() is None
        outlet.consumers = True
        assert _run_until(lambda: run.poll() == bq.EXIT_FINISHED, 15)
        assert outlet.samples[0][0]["event"] == "trial_start"
    finally:
        board.close()


def test_no_consumer_gives_a_notice_and_starts_anyway(app):
    board = _board()
    outlet = FakeOutlet(consumers=False)
    notices = []
    try:
        schedule = _tiny_schedule()
        run = bq.EmbeddedBoardRun(board, schedule, outlet, consumer_wait_s=0.1, consumer_settle_s=0.0)
        run.notice.connect(notices.append)
        run.start()
        assert _run_until(lambda: run.poll() == bq.EXIT_FINISHED, 15)
        assert any("No marker consumer" in n for n in notices)
        assert len(outlet.samples) == len([s for s in schedule if s.marker is not None])
    finally:
        board.close()


def test_real_lsl_consumer_receives_the_board_markers(app):
    from pylsl import StreamInlet, resolve_byprop

    stream_type = f"MKBOARD_{uuid.uuid4().hex}"
    outlet = bq.lsl_marker_outlet(stream_type)
    board = _board()
    try:
        streams = resolve_byprop("type", stream_type, timeout=10)
        assert streams, "board marker stream not found"
        inlet = StreamInlet(streams[0])
        inlet.open_stream(timeout=5)
        schedule = _tiny_schedule(flash=15, isi=15, cue=20, inter=50)
        run = bq.EmbeddedBoardRun(board, schedule, outlet, consumer_settle_s=0.1).start()
        assert _run_until(lambda: run.poll() == bq.EXIT_FINISHED, 20)
        expected = [s.marker for s in schedule if s.marker is not None]
        received = []
        deadline = time.monotonic() + 5
        while len(received) < len(expected) and time.monotonic() < deadline:
            chunk, _ = inlet.pull_chunk(timeout=0.2)
            received += [json.loads(sample[0]) for sample in chunk]
        assert received == expected
    finally:
        board.close()
