"""
Session control panel for the Thai P300 speller (PySide6).

Replaces the Tkinter panel (collection/session_gui.py, kept as a fallback)
with the same behaviour, plus a live-inference display:

  1. Live signal quality: 8 scrolling traces, front to back (Fz C3 Cz C4 Pz
     PO7 Oz PO8), and the two-hemisphere scalp map. Dot colours come from
     pipeline.quality.channel_levels over the last 1 s; gray means unknown
     (no data, a gap, or a stream that stopped delivering).
  2. Target entry and settings. Enter or Start launches
     [pipeline.inference_service ->] record_session.py -> the stimulus board, drawn
     in this window (board_qt.py) or in a separate PsychoPy window (checkerboard_stimulus.py).
  3. Live inference (P300Inference stream): composed Thai text, a
     per-character confidence gauge, fallback commits marked, rejected-flash
     counters, service state, manifest-mismatch and malformed-event reports.

Everything that doesn't need Qt lives in gui_logic.py and is unit-tested.
LSL is read on background threads that hand data to the GUI thread through
queued Qt signals; the event loop never blocks on LSL.

This panel is a pure LSL consumer: the Unicorn headset is reachable only
through the vendor LSL Interface (see README_ACQUISITION.md).

Run with:
    conda activate p300speller
    python collection/session_gui_qt.py [--demo] [--manifest PATH]
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for _path in (str(REPO_ROOT), str(THIS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTextBoundaryFinder, QTimer, Signal  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPolygonF, QTextLayout, QTextOption,
)
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QSizePolicy, QSpinBox, QStackedWidget, QToolTip, QVBoxLayout,
    QWidget,
)

import board_qt as bq  # noqa: E402
import board_schedule as bs  # noqa: E402
import gui_logic as gl  # noqa: E402
from pipeline.constants import (  # noqa: E402
    EEG_STREAM_TYPE, INFERENCE_FINGERPRINT_KEY, INFERENCE_STREAM_TYPE, MARKER_STREAM_TYPE,
)
from pipeline.manifest import (  # noqa: E402
    ManifestError, load_characters_file, load_manifest, short_fingerprint,
)

QUEUED = Qt.ConnectionType.QueuedConnection
BOARD_IN_PANEL = "panel"
BOARD_SEPARATE = "window"


# ----------------------------------------------------------------------
# Fonts and Thai text layout
# ----------------------------------------------------------------------

def ui_font(pixel_size: int, bold: bool = False) -> QFont:
    font = QFont(QApplication.font())
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    return font


def thai_font_family() -> str | None:
    """Sarabun, else Thonburi (both verified to stack tone marks under Qt on macOS), else None."""
    return gl.pick_thai_family(QFontDatabase.families())


def thai_font(pixel_size: int, bold: bool = False) -> QFont:
    family = thai_font_family()
    font = QFont(family) if family else QFont(QApplication.font())
    font.setPixelSize(pixel_size)
    font.setBold(bold)
    return font


def single_line_layout(text: str, font: QFont):
    """
    One QTextLayout for the whole string, so HarfBuzz shapes base consonants
    and their combining marks together. Returns (layout, line or None); keep
    the layout alive while using the line.
    """
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.NoWrap)
    layout.setTextOption(option)
    layout.beginLayout()
    line = layout.createLine()
    if line.isValid():
        line.setLineWidth(100000.0)
        line.setPosition(QPointF(0.0, 0.0))
    else:
        line = None
    layout.endLayout()
    return layout, line


def cursor_x(line, position: int) -> float:
    result = line.cursorToX(position)
    return float(result[0] if isinstance(result, (tuple, list)) else result)


def grapheme_clusters(text: str) -> list[tuple[int, int]]:
    """[start, end) grapheme clusters: a Thai consonant and the marks stacked on it form one."""
    if not text:
        return []
    finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Grapheme, text)
    bounds = [0]
    while True:
        position = finder.toNextBoundary()
        if position == -1:
            break
        if position > bounds[-1]:
            bounds.append(position)
    if bounds[-1] != len(text):
        bounds.append(len(text))
    return list(zip(bounds, bounds[1:]))


def render_text_image(text: str, font: QFont, padding: int = 16) -> QImage:
    """Black text on white via the same layout path as ComposedTextView (used by the rendering test)."""
    layout, line = single_line_layout(text, font)
    width = int(np.ceil(line.naturalTextWidth())) + 2 * padding if line else 2 * padding
    height = int(np.ceil(line.height())) + 2 * padding if line else 2 * padding
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor("black"))
    layout.draw(painter, QPointF(padding, padding))
    painter.end()
    return image.convertToFormat(QImage.Format.Format_Grayscale8)


# ----------------------------------------------------------------------
# Painted widgets
# ----------------------------------------------------------------------

class TraceView(QWidget):
    """8 stacked, per-channel-normalized traces over the last 4 s (QPainter)."""

    TITLE_H, LEFT, RIGHT, BOTTOM = 22, 40, 10, 26

    def __init__(self, parent=None):
        super().__init__(parent)
        self._traces: dict = {}
        self.setMinimumSize(420, 250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_traces(self, traces: dict):
        self._traces = traces
        self.update()

    def plot_rect(self) -> QRectF:
        return QRectF(self.LEFT, self.TITLE_H, max(10.0, self.width() - self.LEFT - self.RIGHT),
                      max(10.0, self.height() - self.TITLE_H - self.BOTTOM))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(gl.COLOR_BG))
        p.setPen(QColor(gl.COLOR_FG))
        p.setFont(ui_font(13))
        p.drawText(QPointF(self.LEFT, 15), "Live EEG")

        r = self.plot_rect()
        n = len(gl.DISPLAY_ORDER)
        y_top, y_bottom = 1.0, -float(n)

        def px(x):
            return r.left() + np.asarray(x) / gl.DISPLAY_WINDOW_S * r.width()

        def py(y):
            return r.top() + (y_top - np.asarray(y)) / (y_top - y_bottom) * r.height()

        p.setPen(QPen(QColor("#cbd5e1"), 1, Qt.PenStyle.DotLine))
        for second in range(int(gl.DISPLAY_WINDOW_S) + 1):
            x = float(px(second))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        for i in range(n):
            y = float(py(-i))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        p.setPen(QPen(QColor(gl.COLOR_MUTED), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        p.setFont(ui_font(11))
        for second in range(int(gl.DISPLAY_WINDOW_S) + 1):
            x = float(px(second))
            p.drawText(QRectF(x - 15, r.bottom() + 2, 30, 14), Qt.AlignmentFlag.AlignCenter, str(second))
        p.drawText(QRectF(r.right() - 60, r.bottom() + 12, 60, 14),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "seconds")
        p.setPen(QColor(gl.COLOR_FG))
        for i, ch in enumerate(gl.DISPLAY_ORDER):
            y = float(py(-i))
            p.drawText(QRectF(0, y - 8, self.LEFT - 6, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, ch)

        p.setClipRect(r)
        p.setPen(QPen(QColor(gl.COLOR_TRACE), 1.0))
        for ch in gl.DISPLAY_ORDER:
            x, y = self._traces.get(ch, (np.empty(0), np.empty(0)))
            if len(x) < 2:
                continue
            xs, ys = px(x).tolist(), py(y).tolist()
            p.drawPolyline(QPolygonF([QPointF(a, b) for a, b in zip(xs, ys)]))
        p.end()


class ScalpMap(QWidget):
    """Two-hemisphere brain silhouette with one quality-coloured dot per electrode, plus a legend."""

    X_RANGE = (-1.35, 1.35)
    Y_RANGE = (-2.05, 1.2)
    TITLE_H = 22
    DOT_RADIUS = 0.075

    def __init__(self, parent=None):
        super().__init__(parent)
        self._levels = {ch: gl.quality.UNKNOWN for ch in gl.ELECTRODE_POS}
        self.setMinimumSize(220, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_levels(self, levels: dict):
        levels = {ch: levels.get(ch, gl.quality.UNKNOWN) for ch in gl.ELECTRODE_POS}
        if levels != self._levels:
            self._levels = levels
            self.update()

    def level(self, ch: str) -> str:
        return self._levels[ch]

    def dot_color(self, ch: str) -> QColor:
        return QColor(gl.quality_color(self._levels[ch]))

    def _scale_origin(self):
        width_w = self.X_RANGE[1] - self.X_RANGE[0]
        height_w = self.Y_RANGE[1] - self.Y_RANGE[0]
        avail_h = max(10.0, self.height() - self.TITLE_H)
        scale = min(self.width() / width_w, avail_h / height_w)
        left = (self.width() - width_w * scale) / 2
        top = self.TITLE_H + (avail_h - height_w * scale) / 2
        return scale, left, top

    def to_px(self, x: float, y: float) -> QPointF:
        scale, left, top = self._scale_origin()
        return QPointF(left + (x - self.X_RANGE[0]) * scale, top + (self.Y_RANGE[1] - y) * scale)

    def dot_center(self, ch: str) -> QPointF:
        return self.to_px(*gl.ELECTRODE_POS[ch])

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(gl.COLOR_BG))
        p.setPen(QColor(gl.COLOR_FG))
        p.setFont(ui_font(13))
        p.drawText(QPointF(4, 15), "Signal quality")

        scale, _, _ = self._scale_origin()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(gl.COLOR_BRAIN))
        for side in (-1, 1):
            outline = gl.brain_hemisphere(side)
            p.drawPolygon(QPolygonF([self.to_px(float(x), float(y)) for x, y in outline]))

        radius = max(4.0, self.DOT_RADIUS * scale)
        label_font = ui_font(max(9, int(radius * 1.3)))
        p.setFont(label_font)
        # Size the label box from the font's own metrics. The font scales with
        # the dot radius, so the old fixed 16 px box shaved the tops off the
        # letters on any map big enough to push the font past 16 px.
        label_h = p.fontMetrics().height()
        label_w = max(60.0, label_h * 3.0)
        for ch, (x, y) in gl.ELECTRODE_POS.items():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self.dot_color(ch))
            p.drawEllipse(self.to_px(x, y), radius, radius)
            anchor = self.to_px(x, y + 0.13)
            p.setPen(QColor(gl.COLOR_FG))
            p.drawText(QRectF(anchor.x() - label_w / 2, anchor.y() - label_h, label_w, label_h),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, ch)

        legend_font = ui_font(max(10, int(radius * 1.4)))
        for i, (color, label) in enumerate(gl.QUALITY_LEGEND):
            y = -1.35 - i * 0.22
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color))
            p.drawEllipse(self.to_px(-0.75, y), radius * 0.9, radius * 0.9)
            anchor = self.to_px(-0.55, y)
            p.setPen(QColor(gl.COLOR_FG))
            p.setFont(legend_font)
            p.drawText(QRectF(anchor.x(), anchor.y() - 10, 200, 20),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        p.end()


class ConfidenceGauge(QWidget):
    """max_prob climbing toward a marked theta line, with the leading character. Empty after a commit."""

    COLOR_TRACK = "#e5e7eb"
    COLOR_FILL = "#2563eb"
    COLOR_CROSSED = "#16a34a"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state: gl.GaugeState | None = None
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_state(self, state):
        if state != self._state:
            self._state = state
            self.update()

    def state(self):
        return self._state

    def bar_rect(self) -> QRectF:
        return QRectF(84, 34, max(40.0, self.width() - 84 - 60), 18)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("white"))
        s = self._state
        box = QRectF(8, 8, 64, 64)
        p.setPen(QPen(QColor("#d1d5db"), 1))
        p.setBrush(QColor(gl.COLOR_BG))
        p.drawRoundedRect(box, 6, 6)
        bar = self.bar_rect()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.COLOR_TRACK))
        p.drawRoundedRect(bar, 4, 4)
        if s is None:
            p.setPen(QColor(gl.COLOR_MUTED))
            p.setFont(ui_font(12))
            p.drawText(QRectF(bar.left(), 6, bar.width(), 24), Qt.AlignmentFlag.AlignVCenter,
                       "Confidence — waiting for posterior updates")
            p.end()
            return

        p.setPen(QColor(gl.COLOR_FG))
        p.setFont(thai_font(40))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, gl.display_isolated(s.leading_char))

        crossed = s.max_prob >= s.theta
        fill = QRectF(bar.left(), bar.top(), bar.width() * s.max_prob, bar.height())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self.COLOR_CROSSED if crossed else self.COLOR_FILL))
        p.drawRoundedRect(fill, 4, 4)

        theta_x = bar.left() + bar.width() * s.theta
        p.setPen(QPen(QColor(gl.COLOR_FG), 2))
        p.drawLine(QPointF(theta_x, bar.top() - 7), QPointF(theta_x, bar.bottom() + 7))
        p.setFont(ui_font(11))
        p.drawText(QRectF(theta_x - 40, 8, 80, 18), Qt.AlignmentFlag.AlignCenter, f"θ = {s.theta:.2f}")
        p.drawText(QRectF(bar.right() + 6, bar.top() - 2, 54, 22), Qt.AlignmentFlag.AlignVCenter,
                   f"{s.max_prob:.2f}")
        p.setPen(QColor(gl.COLOR_MUTED))
        p.drawText(QRectF(bar.left(), bar.bottom() + 6, bar.width(), 18), Qt.AlignmentFlag.AlignVCenter,
                   f"Leading character · p(max) = {s.max_prob:.2f} · flash {s.flash_idx} · trial {s.trial_id}")
        p.end()


class ComposedTextView(QWidget):
    """
    The committed characters as ONE text run (combining marks stack on
    their base). Fallback commits are recoloured by redrawing the same
    layout clipped to their grapheme cluster, and additionally get a thick
    underline and a triangle marker, so the distinction does not rely on
    colour. Hovering a cluster shows reason and n_flashes_used.
    """

    FONT_PX = 40
    PAD = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list = []
        self._font = thai_font(self.FONT_PX)
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def set_entries(self, entries):
        entries = list(entries)
        if entries != self._entries:
            self._entries = entries
            self.update()

    def text(self) -> str:
        return "".join(entry.char for entry in self._entries)

    def cluster_styles(self) -> list[tuple[int, int, bool, list]]:
        """(start, end, is_fallback, entries) per grapheme cluster; fallback if any commit in it was."""
        styles = []
        for start, end in grapheme_clusters(self.text()):
            entries = self._entries[start:end]
            styles.append((start, end, any(e.is_fallback for e in entries), entries))
        return styles

    def _geometry(self):
        layout, line = single_line_layout(self.text(), self._font)
        if line is None:
            return layout, line, 0.0, 0.0
        avail = self.width() - 2 * self.PAD
        origin_x = self.PAD + min(0.0, avail - line.naturalTextWidth())   # keep the newest text visible
        origin_y = max(4.0, (self.height() - 18 - line.height()) / 2)   # room below for underline + marker
        return layout, line, origin_x, origin_y

    def cluster_rects(self) -> list[tuple[QRectF, bool, list]]:
        layout, line, ox, oy = self._geometry()
        if line is None:
            return []
        rects = []
        for start, end, fallback, entries in self.cluster_styles():
            x0, x1 = cursor_x(line, start), cursor_x(line, end)
            rects.append((QRectF(ox + min(x0, x1), 0, abs(x1 - x0), self.height()), fallback, entries))
        return rects

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("white"))
        layout, line, ox, oy = self._geometry()
        if line is None or not self._entries:
            p.setPen(QColor(gl.COLOR_MUTED))
            p.setFont(ui_font(13))
            p.drawText(self.rect().adjusted(self.PAD, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter,
                       "Committed characters appear here")
            p.end()
            return

        origin = QPointF(ox, oy)
        p.setPen(QColor(gl.COLOR_FG))
        layout.draw(p, origin)

        underline_y = oy + line.height() + 3
        for rect, fallback, _entries in self.cluster_rects():
            if not fallback:
                continue
            p.save()
            p.setClipRect(rect)
            p.setPen(QColor(gl.COLOR_FALLBACK))
            layout.draw(p, origin)
            p.restore()
            p.setPen(QPen(QColor(gl.COLOR_FALLBACK), 3))
            p.drawLine(QPointF(rect.left() + 1, underline_y), QPointF(rect.right() - 1, underline_y))
            cx = rect.center().x()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(gl.COLOR_FALLBACK))
            p.drawPolygon(QPolygonF([QPointF(cx, underline_y + 3), QPointF(cx - 5, underline_y + 11),
                                     QPointF(cx + 5, underline_y + 11)]))
        p.end()

    def event(self, event):
        if event.type() == QEvent.Type.ToolTip:
            pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
            for rect, _fallback, entries in self.cluster_rects():
                if rect.left() <= pos.x() <= rect.right():
                    QToolTip.showText(event.globalPos(), "\n".join(e.describe() for e in entries), self)
                    return True
            QToolTip.hideText()
            event.ignore()
            return True
        return super().event(event)


# ----------------------------------------------------------------------
# LSL readers (background threads -> queued signals)
# ----------------------------------------------------------------------

class _LslReader(QObject):
    def __init__(self, stream_type: str, parent=None):
        super().__init__(parent)
        self.stream_type = stream_type
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_safely, daemon=True,
                                        name=f"lsl-{stream_type}")

    def start(self):
        self._thread.start()

    def stop(self, join_s: float = 3.0):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(join_s)

    def _run_safely(self):
        try:
            self._run()
        except Exception as exc:   # never let a reader thread die silently
            print(f"[session_gui_qt] {self.stream_type} reader stopped: {type(exc).__name__}: {exc}",
                  file=sys.stderr)


class LslEegReader(_LslReader):
    stream_found = Signal(str)
    samples = Signal(object)
    stream_dropped = Signal(str)

    def _run(self):
        from pylsl import StreamInlet, resolve_byprop
        while not self._stop.is_set():
            streams = resolve_byprop("type", self.stream_type, timeout=0.5)
            if not streams:
                continue
            inlet = None
            try:
                # recover=False: a vanished outlet raises LostError, so the next
                # stream is resolved afresh instead of silently reattaching.
                inlet = StreamInlet(streams[0], max_buflen=10, recover=False)
                inlet.open_stream(timeout=2.0)
                self.stream_found.emit(streams[0].name())
                while not self._stop.is_set():
                    chunk, _timestamps = inlet.pull_chunk(timeout=0.1)
                    if chunk:
                        self.samples.emit(np.asarray(chunk, dtype=float))
            except Exception as exc:
                if not self._stop.is_set():
                    self.stream_dropped.emit(f"{type(exc).__name__}: {exc}")
            finally:
                if inlet is not None:
                    try:
                        inlet.close_stream()
                    except Exception:
                        pass


class LslInferenceReader(_LslReader):
    stream_found = Signal(object, str)   # desc fingerprint (str or None), stream name
    payload = Signal(object)
    stream_dropped = Signal(str)

    def _run(self):
        from pylsl import StreamInlet, resolve_byprop
        while not self._stop.is_set():
            streams = resolve_byprop("type", self.stream_type, timeout=0.5)
            if not streams:
                continue
            newest = max(streams, key=lambda s: s.created_at())
            inlet = None
            try:
                inlet = StreamInlet(newest, max_buflen=60, recover=False)
                inlet.open_stream(timeout=2.0)   # start buffering before anything else
                info = inlet.info(timeout=2.0)
                fingerprint = info.desc().child_value(INFERENCE_FINGERPRINT_KEY) or None
                self.stream_found.emit(fingerprint, newest.name())
                while not self._stop.is_set():
                    chunk, _timestamps = inlet.pull_chunk(timeout=0.2)
                    for sample in chunk or ():
                        self.payload.emit(sample[0] if sample else None)
            except Exception as exc:
                if not self._stop.is_set():
                    self.stream_dropped.emit(f"{type(exc).__name__}: {exc}")
            finally:
                if inlet is not None:
                    try:
                        inlet.close_stream()
                    except Exception:
                        pass


# ----------------------------------------------------------------------
# Subprocesses
# ----------------------------------------------------------------------

class PopenHandle:
    """subprocess.Popen plus a reader thread that tails stderr (echoed to the terminal with a prefix)."""

    def __init__(self, name: str, cmd: list[str], cwd: Path, capture_stderr: bool):
        self.name = name
        self.tail = gl.StderrTail()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stderr=subprocess.PIPE if capture_stderr else None,
            text=True, encoding="utf-8", errors="replace")
        self._reader = None
        if capture_stderr:
            self._reader = threading.Thread(target=self._pump, daemon=True, name=f"stderr-{name}")
            self._reader.start()

    def _pump(self):
        for line in self.proc.stderr:
            sys.stderr.write(f"[{self.name}] {line}")
            self.tail.append(line)

    def poll(self):
        return self.proc.poll()

    def terminate(self):
        self.proc.terminate()

    def kill(self):
        self.proc.kill()

    def stderr_lines(self) -> list[str]:
        if self._reader is not None:
            self._reader.join(1.0)
        return self.tail.lines()


def popen_launcher(name: str, cmd: list[str], cwd: Path) -> PopenHandle:
    print(f"[session_gui_qt] launching {name}: {' '.join(cmd)}  (cwd {cwd})")
    return PopenHandle(name, cmd, cwd, capture_stderr=name in ("service", "recorder"))


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------

class SessionWindow(QMainWindow):
    """
    Everything that touches LSL or processes is injectable:
      start_readers=False  no LSL threads (tests call the on_* slots directly)
      launcher             SessionOrchestrator launcher (tests pass fakes)
      clock / now          monotonic seconds / wall-clock datetime
      auto_refresh=False   no QTimer (tests call refresh())
      board_outlet_factory / board_run_options / board_schedule_options
                           the in-window board's marker outlet, consumer wait and timeline
    manifest_path=None uses the mode's default at the repo root
    (demo_session_manifest.json in demo mode, session_manifest.json otherwise).
    Child processes always run with sys.executable, never a bare `python`.
    """

    def __init__(self, *, demo: bool = False, manifest_path=None,
                 manifest_loader=load_manifest, characters=None, launcher=None,
                 clock=time.monotonic, now=datetime.now, recordings_dir=gl.RECORDINGS_DIR,
                 start_readers: bool = True, auto_refresh: bool = True,
                 eeg_type: str = EEG_STREAM_TYPE, inference_type: str = INFERENCE_STREAM_TYPE,
                 orchestrator_options: dict | None = None, marker_type: str = MARKER_STREAM_TYPE,
                 board_outlet_factory=None, board_run_options: dict | None = None,
                 board_schedule_options: dict | None = None):
        super().__init__()
        self.demo = demo
        self.board_outlet_factory = board_outlet_factory or (lambda: bq.lsl_marker_outlet(marker_type))
        self.board_run_options = dict(board_run_options or {})
        self.board_schedule_options = dict(board_schedule_options or {})
        self.board_run = None
        self._clock = clock
        self._now = now
        self.recordings_dir = recordings_dir
        self._closed = False
        self.characters = characters if characters is not None else load_characters_file(gl.CHARACTERS_PATH)

        self.manifest_path = gl.resolve_manifest_arg(manifest_path, demo)
        self.manifest, self.manifest_problem = gl.load_manifest_for_gui(self.manifest_path, manifest_loader,
                                                                        demo=demo)
        self.inference_model = gl.InferenceDisplayModel(
            getattr(self.manifest, "fingerprint", None), self._inference_characters())
        self.connection = gl.EegConnection()
        self.buffer = gl.SampleBuffer()
        # Non-default stream types (testing only) are forwarded to the service so it
        # publishes and listens where this panel does. Defaults keep the plain command.
        service_args = []
        if eeg_type != EEG_STREAM_TYPE:
            service_args += ["--eeg-type", eeg_type]
        if inference_type != INFERENCE_STREAM_TYPE:
            service_args += ["--inference-type", inference_type]
        options = {"service_extra_args": service_args,
                   "session_log": gl.SessionLogger(demo=demo, characters=self.characters),
                   **(orchestrator_options or {})}
        self.orchestrator = gl.SessionOrchestrator(
            launcher or popen_launcher, clock, self._on_session_message, **options)

        self.setWindowTitle("Thai P300 Speller" + (" [DEMO]" if demo else ""))
        self.resize(1200, 880)
        self._build_ui()

        self.eeg_reader = self.inference_reader = None
        if start_readers:
            self.eeg_reader = LslEegReader(eeg_type, parent=self)
            self.eeg_reader.stream_found.connect(self.on_eeg_stream_found, QUEUED)
            self.eeg_reader.samples.connect(self.on_eeg_samples, QUEUED)
            self.eeg_reader.stream_dropped.connect(self.on_eeg_stream_dropped, QUEUED)
            self.inference_reader = LslInferenceReader(inference_type, parent=self)
            self.inference_reader.stream_found.connect(self.on_inference_stream_found, QUEUED)
            self.inference_reader.payload.connect(self.on_inference_payload, QUEUED)
            self.inference_reader.stream_dropped.connect(self.on_inference_stream_dropped, QUEUED)
            self.eeg_reader.start()
            self.inference_reader.start()

        self.timer = QTimer(self)
        self.timer.setInterval(gl.REFRESH_MS)
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self._render_inference()
        if auto_refresh:
            self.timer.start()

    def _inference_characters(self):
        path = getattr(self.manifest, "characters_path", None)
        if path is not None:
            try:
                return load_characters_file(path)
            except (ManifestError, OSError):
                pass
        return self.characters

    # -- layout --------------------------------------------------------------
    def _build_ui(self):
        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)

        setup_page = QWidget()
        setup_page.setObjectName("central")
        setup_page.setStyleSheet(f"#central {{ background: {gl.COLOR_BG}; }} QLabel {{ color: {gl.COLOR_FG}; }}")
        root = QVBoxLayout(setup_page)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Thai P300 Speller")
        title.setFont(ui_font(19, bold=True))
        header.addWidget(title)
        self.demo_banner = None
        if self.demo:
            self.demo_banner = QLabel("DEMO MODE — synthetic data")
            self.demo_banner.setStyleSheet("background:#fef3c7; color:#92400e; font-weight:bold; "
                                           "padding:2px 8px; border-radius:3px;")
            header.addWidget(self.demo_banner)
        header.addStretch(1)
        self.status_label = QLabel()
        self.status_label.setFont(ui_font(14))
        header.addWidget(self.status_label)
        root.addLayout(header)

        self.mismatch_banner = QLabel()
        self.mismatch_banner.setWordWrap(True)
        self.mismatch_banner.setStyleSheet("background:#fee2e2; color:#991b1b; border:2px solid #ef4444; "
                                           "font-weight:bold; padding:8px; border-radius:4px;")
        self.mismatch_banner.hide()
        root.addWidget(self.mismatch_banner)

        plots = QHBoxLayout()
        self.trace_view = TraceView()
        self.scalp_map = ScalpMap()
        plots.addWidget(self.trace_view, 21)
        plots.addWidget(self.scalp_map, 10)
        root.addLayout(plots, 5)

        panel = QFrame()
        panel.setObjectName("inferencePanel")
        panel.setStyleSheet("#inferencePanel { background: white; border: 1px solid #e5e7eb; border-radius: 6px; }")
        grid = QGridLayout(panel)
        grid.setContentsMargins(10, 8, 10, 8)
        heading = QLabel("Live inference")
        heading.setFont(ui_font(13, bold=True))
        grid.addWidget(heading, 0, 0)
        self.composed_view = ComposedTextView()
        grid.addWidget(self.composed_view, 1, 0)
        self.fallback_legend = QLabel(
            f'<span style="color:{gl.COLOR_FALLBACK}; text-decoration:underline; font-weight:bold;">'
            f'▲ underlined</span> = fallback commit (max_flashes_reached or trial_ended: a limit forced it, '
            f'not confidence). Hover a character, or see the commit log.')
        self.fallback_legend.setWordWrap(True)
        self.fallback_legend.setStyleSheet(f"color:{gl.COLOR_MUTED}; font-size:11px;")
        grid.addWidget(self.fallback_legend, 2, 0)
        self.gauge = ConfidenceGauge()
        grid.addWidget(self.gauge, 3, 0)

        side = QVBoxLayout()
        self.service_label = QLabel()
        self.rejected_label = QLabel()
        self.malformed_label = QLabel()
        self.malformed_label.setWordWrap(True)
        self.malformed_label.setStyleSheet(f"color:{gl.COLOR_FALLBACK}; font-weight:bold;")
        for label in (self.service_label, self.rejected_label, self.malformed_label):
            side.addWidget(label)
        log_title = QLabel("Commit log")
        log_title.setStyleSheet(f"color:{gl.COLOR_MUTED};")
        side.addWidget(log_title)
        self.history_list = QListWidget()
        self.history_list.setFont(thai_font(16))
        self.history_list.setMinimumWidth(340)
        side.addWidget(self.history_list, 1)
        grid.addLayout(side, 0, 1, 4, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        root.addWidget(panel, 3)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Target characters"))
        self.targets_edit = QLineEdit()
        self.targets_edit.setFont(thai_font(22))
        self.targets_edit.setMinimumWidth(320)
        self.targets_edit.returnPressed.connect(self._on_start)
        row1.addWidget(self.targets_edit)
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet(
            "QPushButton { background:#2563eb; color:white; font-weight:bold; padding:6px 22px; border-radius:4px; }"
            "QPushButton:disabled { background:#bfdbfe; color:#eff6ff; }")
        self.start_button.clicked.connect(self._on_start)
        row1.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("QPushButton { padding:6px 16px; }")
        self.stop_button.clicked.connect(self._on_stop)
        row1.addWidget(self.stop_button)
        row1.addStretch(1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        defaults = gl.DEFAULT_SETTINGS

        def muted(text):
            label = QLabel(text)
            label.setStyleSheet(f"color:{gl.COLOR_MUTED};")
            return label

        def spin(key):
            box = QSpinBox()
            box.setRange(*gl.BOUNDS[key])
            box.setValue(defaults[key])
            return box

        row2.addWidget(muted("Session ID"))
        self.session_edit = QLineEdit(defaults["session_id"])
        self.session_edit.setMaximumWidth(150)
        row2.addWidget(self.session_edit)
        row2.addWidget(muted("Repetitions"))
        self.reps_spin = spin("repetitions")
        row2.addWidget(self.reps_spin)
        row2.addWidget(muted("Flash ms"))
        self.flash_spin = spin("flash_ms")
        row2.addWidget(self.flash_spin)
        row2.addWidget(muted("ISI ms"))
        self.isi_spin = spin("isi_ms")
        row2.addWidget(self.isi_spin)
        row2.addWidget(muted("Board"))
        self.board_combo = QComboBox()
        self.board_combo.addItem("In this window", BOARD_IN_PANEL)
        self.board_combo.addItem("Separate window (precise timing)", BOARD_SEPARATE)
        self.board_combo.setToolTip(
            "In this window: flash times are estimated (about one screen refresh). Separate window: PsychoPy "
            "stamps each flash at the real screen flip; use it for real data until the in-window board has "
            "been checked with a photodiode.")
        row2.addWidget(self.board_combo)
        row2.addWidget(muted("Screen"))
        self.screen_spin = spin("screen")
        row2.addWidget(self.screen_spin)
        self.windowed_check = QCheckBox("Windowed")
        self.windowed_check.setChecked(defaults["windowed"])
        row2.addWidget(self.windowed_check)
        self.board_combo.currentIndexChanged.connect(self._on_board_mode_changed)
        self._on_board_mode_changed()
        self.live_check = QCheckBox("Live inference")
        self.live_check.setChecked(False)
        self.live_check.setEnabled(self.manifest is not None)
        row2.addWidget(self.live_check)
        row2.addStretch(1)
        root.addLayout(row2)

        self.inference_note = QLabel()
        self.inference_note.setWordWrap(True)
        if self.manifest is None:
            self.inference_note.setText(self.manifest_problem)
            self.inference_note.setStyleSheet("color:#b45309;")
            self.live_check.setToolTip(self.manifest_problem)
        else:
            fp = short_fingerprint(self.manifest.fingerprint)
            self.inference_note.setText(f"Session manifest {getattr(self.manifest, 'path', '')} "
                                        f"(fingerprint {fp})")
            self.inference_note.setStyleSheet(f"color:{gl.COLOR_MUTED};")
        root.addWidget(self.inference_note)

        self.session_label = QLabel("Idle")
        self.session_label.setWordWrap(True)
        self.session_label.setStyleSheet(f"color:{gl.COLOR_MUTED};")
        root.addWidget(self.session_label)
        self.targets_edit.setFocus()

        self.pages.addWidget(setup_page)
        self.pages.addWidget(self._build_run_page())
        self.pages.setCurrentWidget(setup_page)

    def _build_run_page(self) -> QWidget:
        """
        Distraction-free calibration screen: just the board (no traces, scalp
        map, live-inference panel, or settings), matching the reference
        vendor speller UI. Shown full-window while an in-window board session
        runs; a minimal overlay bar keeps only Stop and status visible.
        """
        page = QWidget()
        page.setStyleSheet("background:#000000;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        overlay = QHBoxLayout()
        overlay.setContentsMargins(10, 6, 10, 6)
        self.run_demo_tag = QLabel("DEMO")
        self.run_demo_tag.setStyleSheet("color:#fbbf24; font-weight:bold; font-size:11px;")
        self.run_demo_tag.setVisible(self.demo)
        overlay.addWidget(self.run_demo_tag)
        self.run_status_label = QLabel("Idle")
        self.run_status_label.setStyleSheet(f"color:{gl.COLOR_MUTED}; font-size:12px;")
        self.run_status_label.setWordWrap(True)
        overlay.addWidget(self.run_status_label, 1)
        run_stop_button = QPushButton("Stop")
        run_stop_button.setStyleSheet(
            "QPushButton { background:#1f2937; color:#e5e7eb; padding:4px 14px; border-radius:4px; "
            "border:1px solid #374151; } QPushButton:hover { background:#374151; }")
        run_stop_button.clicked.connect(self._on_stop)
        overlay.addWidget(run_stop_button)
        layout.addLayout(overlay)

        self.board_view = bq.BoardView(self.characters)
        layout.addWidget(self.board_view, 1)
        return page

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.pages.currentWidget() is self.run_page:
            self._on_stop()
            return
        super().keyPressEvent(event)

    @property
    def run_page(self) -> QWidget:
        return self.pages.widget(1)

    # -- slots fed by the LSL threads (and by tests) ---------------------------
    def on_eeg_stream_found(self, name: str = ""):
        self.buffer.clear()
        self.connection.on_stream_found(self._clock(), name)
        self._render_status()

    def on_eeg_samples(self, chunk):
        added = self.buffer.append(chunk)
        self.connection.on_samples(self._clock(), added)

    def on_eeg_stream_dropped(self, reason: str = ""):
        self.connection.on_stream_dropped(self._clock())
        self._render_status()

    def on_inference_stream_found(self, fingerprint, name: str = ""):
        self.inference_model.on_stream_connected(fingerprint)
        self._render_inference()

    def on_inference_payload(self, text):
        self.inference_model.apply_payload(text)
        self._render_inference()

    def on_inference_stream_dropped(self, reason: str = ""):
        self.inference_model.on_stream_lost()
        self._render_inference()

    # -- periodic refresh ----------------------------------------------------
    def refresh(self):
        now = self._clock()
        self.connection.tick(now)
        connected = self.connection.state == gl.CONNECTED
        if connected:
            self.trace_view.set_traces(gl.display_traces(self.buffer))
        self.scalp_map.set_levels(gl.scalp_levels(self.buffer, connected))
        self.orchestrator.tick(service_ready=self.inference_model.service_ready)
        if self.board_run is not None and self.board_run.poll() is not None:
            self.board_run = None
            self._exit_run_view()
        self._render_status()

    def board_mode(self) -> str:
        return self.board_combo.currentData()

    def _on_board_mode_changed(self, *_):
        separate = self.board_mode() == BOARD_SEPARATE
        self.screen_spin.setEnabled(separate)
        self.windowed_check.setEnabled(separate)

    def _enter_run_view(self):
        """Switch to the distraction-free board page (QStackedWidget hides everything else)."""
        self.pages.setCurrentWidget(self.run_page)

    def _exit_run_view(self):
        self.pages.setCurrentWidget(self.pages.widget(0))

    def _start_board(self, settings):
        schedule = bs.build_session_schedule(
            self.characters, settings.targets, n_repetitions=settings.repetitions,
            flash_ms=settings.flash_ms, isi_ms=settings.isi_ms, **self.board_schedule_options)
        run = bq.EmbeddedBoardRun(self.board_view, schedule, self.board_outlet_factory(),
                                  **self.board_run_options)
        run.notice.connect(lambda text: self._on_session_message(text, gl.INFO))
        self.board_run = run
        self._enter_run_view()
        return run.start()

    def _render_status(self):
        self.status_label.setText(self.connection.status_text)
        self.status_label.setStyleSheet(f"color:{self.connection.status_color};")
        self.start_button.setEnabled(self.connection.start_allowed and not self.orchestrator.active)
        self.stop_button.setEnabled(self.orchestrator.phase != gl.IDLE)

    def _render_inference(self):
        m = self.inference_model
        if m.mismatch:
            self.mismatch_banner.setText("⚠ " + m.mismatch_message)
            self.mismatch_banner.show()
            self.composed_view.set_entries([])
            self.gauge.set_state(None)
            self._sync_history([])
        else:
            self.mismatch_banner.hide()
            self.composed_view.set_entries(m.committed)
            self.gauge.set_state(m.gauge)
            self._sync_history(m.committed)
        self.service_label.setText(m.service_text())
        self.rejected_label.setText(m.rejected_text())
        self.malformed_label.setText(m.malformed_text())
        self.malformed_label.setVisible(bool(m.malformed_count))

    def _sync_history(self, entries):
        if self.history_list.count() == len(entries) and all(
                self.history_list.item(i).data(Qt.ItemDataRole.UserRole) == entry
                for i, entry in enumerate(entries)):
            return
        self.history_list.clear()
        for entry in entries:
            marker = "▲ FALLBACK" if entry.is_fallback else "✓"
            item = QListWidgetItem(f"{marker}  {gl.display_isolated(entry.char)}   trial {entry.trial_id} · "
                                   f"{entry.reason} · {entry.n_flashes_used} flashes")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(entry.describe())
            if entry.is_fallback:
                item.setForeground(QColor(gl.COLOR_FALLBACK))
                font = thai_font(16)
                font.setUnderline(True)
                item.setFont(font)
            self.history_list.addItem(item)
        self.history_list.scrollToBottom()

    # -- session control -------------------------------------------------------
    def _on_session_message(self, message: str, level: str):
        color = gl.COLOR_ERROR if level == gl.ERROR else gl.COLOR_MUTED
        self.session_label.setText(message)
        self.session_label.setStyleSheet(f"color:{color};")
        if hasattr(self, "run_status_label"):
            self.run_status_label.setText(message)
            self.run_status_label.setStyleSheet(f"color:{color}; font-size:12px;")
        print(f"[session_gui_qt] {message}")
        if hasattr(self, "start_button"):
            self._render_status()

    def _on_start(self):
        if self.orchestrator.active:
            return
        self.connection.tick(self._clock())
        settings, errors = gl.validate_settings(
            session_id=self.session_edit.text(), targets=self.targets_edit.text(),
            repetitions=self.reps_spin.value(), flash_ms=self.flash_spin.value(),
            isi_ms=self.isi_spin.value(), screen=self.screen_spin.value(),
            windowed=self.windowed_check.isChecked(), characters=self.characters)
        if errors:
            self._on_session_message(" ".join(errors), gl.ERROR)
            return
        if not self.connection.start_allowed:
            if self.connection.state == gl.LOST:
                self._on_session_message("EEG stream lost — check the headset and the LSL bridge before starting.",
                                         gl.ERROR)
            else:
                self._on_session_message("No EEG stream — start the Unicorn LSL Interface first.", gl.ERROR)
            self._render_status()
            return
        use_inference = self.live_check.isChecked() and self.manifest is not None
        out_path = gl.recording_path(settings.session_id, demo=self.demo, now=self._now(),
                                     recordings_dir=self.recordings_dir)
        self.inference_model.reset_session()
        self._render_inference()
        runner = self._start_board if self.board_mode() == BOARD_IN_PANEL else None
        self.orchestrator.start(settings, out_path,
                                manifest_path=self.manifest.path if use_inference else None,
                                stimulus_runner=runner)
        self._render_status()

    def _on_stop(self):
        if self.orchestrator.phase != gl.IDLE:
            self.orchestrator.stop("Stopped.")
        self._render_status()

    def shutdown(self):
        """Stop timers, LSL threads and every child process. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        for reader in (self.eeg_reader, self.inference_reader):
            if reader is not None:
                reader.stop()
        self.orchestrator.shutdown()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thai P300 speller session control panel (PySide6).")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: label the window and prefix recordings with 'demo_'. "
                             "Pair with collection/mock_headset.py to run with no headset.")
    parser.add_argument("--manifest", default=None,
                        help="session manifest for live inference. Default: <repo>/session_manifest.json, "
                             "or <repo>/demo_session_manifest.json with --demo. A relative path is "
                             "relative to the current directory. Missing or invalid = recording only.")
    parser.add_argument("--eeg-type", default=EEG_STREAM_TYPE,
                        help="LSL stream type for EEG (testing only; also passed to the launched "
                             "inference service; the recorder still uses 'EEG'). Default %(default)s")
    parser.add_argument("--inference-type", default=INFERENCE_STREAM_TYPE,
                        help="LSL stream type for inference events (testing only; also passed to the "
                             "launched inference service). Default %(default)s")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = SessionWindow(demo=args.demo, manifest_path=args.manifest,
                           eeg_type=args.eeg_type, inference_type=args.inference_type)
    window.show()

    # Qt's event loop starves Python signal handlers; wake the interpreter
    # periodically so Ctrl+C / SIGTERM close the window (and stop children).
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: window.close())
    wake = QTimer()
    wake.start(200)
    wake.timeout.connect(lambda: None)

    code = app.exec()
    window.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
