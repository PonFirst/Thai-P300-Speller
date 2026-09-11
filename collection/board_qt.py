"""
In-window stimulus board for the session panel: draws the 72-character
checkerboard inside the Qt window and publishes the same P300Markers JSON as
checkerboard_stimulus.py (board_schedule.py builds the timeline).

TIMING -- read before recording real data with it. PsychoPy stamps each flash
with the GPU flip. Qt widgets have no equivalent: measured on this Mac (120 Hz
display), neither QOpenGLWidget.frameSwapped nor timer-driven repaints are
locked to the display refresh (median intervals 4-4.5 ms). Each marker is
stamped with pylsl.local_clock() right after the repaint call returns, but
macOS usually defers the actual paint (median repaint call 0.02 ms), so the
flash can reach the screen up to about one to two refreshes later (~8-17 ms at
120 Hz). Steps run on absolute times, so errors do not accumulate: on screen
here, flashes came 123.97 ms apart (median; plan 124, sd 1.1 ms) and every
marker stamp stayed within 7 ms of plan. Check with a photodiode (TESTING.md
H7) before relying on in-window sessions for real data; the separate PsychoPy
window stays the precise option.
"""

from __future__ import annotations

import json
import time

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

import board_schedule as bs
import gui_logic as gl
from pipeline.constants import MARKER_STREAM_NAME, MARKER_STREAM_TYPE

BOARD_ROWS, BOARD_COLS = 8, 9
COLOR_BOARD_BG = "#000000"
COLOR_IDLE = "#dcdcdc"          # gainsboro, as in the PsychoPy stimulus
COLOR_FLASH_BG = "#ffffff"
COLOR_FLASH_TEXT = "#000000"
COLOR_CUE = "#00ff00"           # lime

CONSUMER_WAIT_S = 15.0
CONSUMER_SETTLE_S = 2.0
CONSUMER_POLL_S = 0.05
EMBEDDED_SOURCE_ID = "thai_speller_embedded"

EXIT_FINISHED = 0
EXIT_FAILED = 1
EXIT_TERMINATED = -15


def lsl_marker_outlet(stream_type: str = MARKER_STREAM_TYPE):
    from pylsl import StreamInfo, StreamOutlet
    return StreamOutlet(StreamInfo(MARKER_STREAM_NAME, stream_type, 1, 0, "string", EMBEDDED_SOURCE_ID))


class BoardView(QWidget):
    """The character grid: black cells, light text; a flashing cell is a white box with black text; the cue is lime."""

    def __init__(self, characters, n_rows: int = BOARD_ROWS, n_cols: int = BOARD_COLS, parent=None):
        super().__init__(parent)
        characters = list(characters)
        if len(characters) != n_rows * n_cols:
            raise ValueError(f"{len(characters)} characters for a {n_rows}x{n_cols} board")
        self.characters = characters
        self.n_rows, self.n_cols = n_rows, n_cols
        self.flashing: frozenset = frozenset()
        self.cue_idx: int | None = None
        self._family = gl.pick_thai_family(QFontDatabase.families())
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def set_state(self, flashing=(), cue_idx=None):
        self.flashing = frozenset(flashing)
        self.cue_idx = cue_idx

    def clear(self):
        self.set_state()
        self.update()

    def cell_rect(self, idx: int) -> QRectF:
        row, col = divmod(idx, self.n_cols)
        width, height = self.width() / self.n_cols, self.height() / self.n_rows
        return QRectF(col * width, row * height, width, height)

    def cell_colors(self, idx: int) -> tuple[str, str]:
        """(background, text) for one cell."""
        if idx in self.flashing:
            return COLOR_FLASH_BG, COLOR_FLASH_TEXT
        return COLOR_BOARD_BG, COLOR_CUE if idx == self.cue_idx else COLOR_IDLE

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(COLOR_BOARD_BG))
        font = QFont(self._family) if self._family else QFont()
        font.setPixelSize(max(10, int(self.height() / self.n_rows * 0.55)))
        p.setFont(font)
        for idx, char in enumerate(self.characters):
            rect = self.cell_rect(idx)
            background, text = self.cell_colors(idx)
            if background != COLOR_BOARD_BG:
                inset_x, inset_y = rect.width() * 0.05, rect.height() * 0.05
                p.fillRect(rect.adjusted(inset_x, inset_y, -inset_x, -inset_y), QColor(background))
            p.setPen(QColor(text))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, gl.display_isolated(char))
        p.end()


class EmbeddedBoardRun(QObject):
    """
    Plays one session's schedule on a BoardView and pushes each step's marker.
    Exposes the handle interface SessionOrchestrator uses for a stimulus
    process: poll() / terminate() / kill() / stderr_lines().

    Before the first trial it waits for a marker consumer, then consumer_settle_s
    more, for the same reason as checkerboard_stimulus.wait_for_marker_consumers():
    LSL never delivers samples pushed before a consumer subscribed.
    """

    notice = Signal(str)

    def __init__(self, board: BoardView, schedule, outlet, *, clock=time.monotonic, local_clock=None,
                 consumer_wait_s: float = CONSUMER_WAIT_S, consumer_settle_s: float = CONSUMER_SETTLE_S,
                 parent=None):
        super().__init__(parent)
        if local_clock is None:
            from pylsl import local_clock
        self.board = board
        self.schedule = list(schedule)
        self.outlet = outlet
        self.clock = clock
        self.local_clock = local_clock
        self.consumer_wait_s = consumer_wait_s
        self.consumer_settle_s = consumer_settle_s
        self.returncode: int | None = None
        self.errors: list[str] = []
        self._phase = "waiting_consumers"
        self._next = 0
        self._t0 = 0.0
        self._wait_started = 0.0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_timer)

    def start(self) -> "EmbeddedBoardRun":
        self._wait_started = self.clock()
        self._schedule_in(0.0)
        return self

    def poll(self):
        return self.returncode

    def terminate(self):
        self._finish(EXIT_TERMINATED)

    def kill(self):
        self._finish(EXIT_TERMINATED)

    def stderr_lines(self) -> list[str]:
        return list(self.errors)

    def _schedule_in(self, seconds: float):
        self._timer.start(max(0, int(round(seconds * 1000))))

    def _on_timer(self):
        if self.returncode is not None:
            return
        try:
            if self._phase == "waiting_consumers":
                self._wait_for_consumers()
            elif self._phase == "settling":
                self._phase = "running"
                self._t0 = self.clock()
                self._run_due_steps()
            else:
                self._run_due_steps()
        except Exception as exc:  # a board failure must end the session visibly, not hang it
            self.errors.append(f"board: {type(exc).__name__}: {exc}")
            self._finish(EXIT_FAILED)

    def _wait_for_consumers(self):
        if self.outlet.have_consumers():
            self._phase = "settling"
            self._schedule_in(self.consumer_settle_s)
        elif self.clock() - self._wait_started >= self.consumer_wait_s:
            self.notice.emit(f"No marker consumer connected within {self.consumer_wait_s:g} s; starting the "
                             "board anyway. Markers pushed before a recorder subscribes are lost.")
            self._phase = "settling"
            self._schedule_in(0.0)
        else:
            self._schedule_in(CONSUMER_POLL_S)

    def _run_due_steps(self):
        elapsed = self.clock() - self._t0
        due = []
        while self._next < len(self.schedule) and self.schedule[self._next].at_s <= elapsed + 0.0005:
            due.append(self.schedule[self._next])
            self._next += 1
        if due:
            for step in due:
                self._apply(step)
            self.board.repaint()
            stamp = self.local_clock()
            for step in due:
                if step.marker is not None:
                    self.outlet.push_sample([json.dumps(step.marker)], stamp)
            if due[-1].kind == bs.DONE:
                self._finish(EXIT_FINISHED)
                return
        if self._next < len(self.schedule):
            self._schedule_in(self.schedule[self._next].at_s - (self.clock() - self._t0))

    def _apply(self, step):
        board = self.board
        if step.kind == bs.CUE_ON:
            board.set_state(board.flashing, step.target_idx)
        elif step.kind == bs.CUE_OFF:
            board.set_state(board.flashing, None)
        elif step.kind == bs.FLASH_ON:
            board.set_state(step.group, board.cue_idx)
        elif step.kind == bs.FLASH_OFF:
            board.set_state((), board.cue_idx)

    def _finish(self, code: int):
        if self.returncode is not None:
            return
        self._timer.stop()
        self.returncode = code
        self.board.clear()
