"""
PsychoPy-based stimulus presentation for the Thai checkerboard P300
speller. Replaces the reference implementation's tkinter + pre-rendered
PNG approach (github.com/chaklam-silpasuwanchai/EEG-Real-Time-
Applications/Thai-P300-Speller) with PsychoPy's frame-locked Window.flip()
and live TextStim rendering, so stimulus-onset timestamps come from the
actual GPU flip rather than a software `after()` timer estimate.

Flash-group construction (which characters flash together, in what
order) lives in flash_sequence.py and has no PsychoPy dependency -- this
file is purely the rendering/timing/LSL layer on top of it.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

import json as _json

from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet, local_clock

from flash_sequence import CheckerboardConfig, build_flash_sequence

MARKER_STREAM_NAME = "P300Markers"


class TrialAborted(Exception):
    """Raised internally when the user presses Escape mid-trial."""


def load_characters(json_path: str) -> list[str]:
    """
    Load a character list from a characters_th.json-style file. Accepts
    either a plain JSON array of strings, or an object with a
    "characters" key (the format characters_th.json uses so it can carry
    provenance/verification metadata alongside the list).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        characters = data["characters"]
        if not data.get("verified", True):
            warnings.warn(
                f"{json_path}: character list is marked unverified "
                f"({data.get('verification_todo', 'no details given')}). "
                "Confirm against the primary source before real data collection.",
                stacklevel=2,
            )
    else:
        characters = data

    return characters


def placeholder_characters(n_chars: int = 72) -> list[str]:
    """Dummy 'C0'..'C{n-1}' list for tests/demos -- never real Thai content."""
    return [f"C{i}" for i in range(n_chars)]


class ThaiCheckerboardSpeller:
    def __init__(self, characters: list[str], n_rows: int = 8, n_cols: int = 9,
                 font_path: str | None = "THSarabunNew.ttf", font_size: float = 60,
                 flash_duration_ms: float = 62, isi_ms: float = 62,
                 n_repetitions: int = 5, inter_selection_delay_ms: float = 3500,
                 cue_duration_ms: float = 2000,
                 lsl_stream_name: str = MARKER_STREAM_NAME, fullscreen: bool = True,
                 screen_size: tuple[int, int] | None = None, screen_index: int = 0,
                 win_type: str = "pyglet"):
        """
        characters: ordered list of exactly n_rows*n_cols strings.

        fullscreen: True (default) opens a borderless fullscreen window
            covering the whole display -- matches the reference paper's
            flashboard, and avoids window-manager compositing/vsync
            quirks that can blur the frame-lock timing guarantee.
        screen_size: (width, height) in pixels, used only when
            fullscreen=False (ignored otherwise -- PsychoPy fullscreen
            windows always take the full native resolution of the target
            screen). Defaults to PsychoPy's own default (1024x768) if
            None and fullscreen=False.
        screen_index: which physical display to use (0 = primary),
            relevant on multi-monitor setups for both fullscreen and
            windowed mode.
        win_type: PsychoPy window backend. Defaults to "pyglet" rather
            than PsychoPy's PTB backend -- PTB fullscreen windows grab
            exclusive display access on macOS, which blocks Cmd+Tab/
            Mission Control until the window closes. "pyglet" fullscreen
            stays a normal (if borderless) window the OS can still
            switch away from. Only change this to "PTB" if you need
            PTB's hardware-level timing guarantees and are OK trading
            away easy window-switching for it.

        n_rows/n_cols default to 8x9 (not 9x8) deliberately: the
        even/odd checkerboard split in flash_sequence.py only forms a
        true adjacency-safe checkerboard when n_cols is ODD -- see that
        module's docstring. 9x8 was empirically confirmed unsafe
        (adjacent characters co-flash). Don't flip these back without
        re-running validate_no_adjacent_co_flash() against the new shape.

        Font note: PsychoPy's TextStim + THSarabunNew has not been
        verified in this session to shape combining Thai tone marks and
        vowels correctly (Thai text shaping can vary by rendering
        backend -- pyglet/GLFW backends handle complex script shaping
        differently than a native OS text renderer like tkinter's).
        Run verify_thai_font_rendering() below and visually inspect the
        saved screenshot BEFORE trusting this for real characters -- if
        glyphs render incorrectly (marks detached/misplaced), fall back
        to the reference implementation's pre-rendered-PNG approach for
        the affected characters; do not ship silently-broken glyphs.
        """
        if len(characters) != n_rows * n_cols:
            raise ValueError(
                f"characters has {len(characters)} entries, expected "
                f"n_rows*n_cols={n_rows * n_cols}"
            )

        self.characters = characters
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.font_path = self._resolve_font(font_path)
        self.font_size = font_size
        self.flash_duration_ms = flash_duration_ms
        self.isi_ms = isi_ms
        self.n_repetitions = n_repetitions
        self.inter_selection_delay_ms = inter_selection_delay_ms
        self.cue_duration_ms = cue_duration_ms
        self.lsl_stream_name = lsl_stream_name
        self.fullscreen = fullscreen
        self.screen_size = screen_size
        self.screen_index = screen_index
        self.win_type = win_type

        self.win = None
        self.stims: list[visual.TextStim] = []
        self.highlight_rects: list[visual.Rect] = []
        self.outlet = None
        self._lsl_clock_offset = 0.0
        self._trial_id = -1

    @staticmethod
    def _resolve_font(font_path: str | None) -> str | None:
        if font_path is None:
            return None
        if os.path.exists(font_path):
            return font_path
        warnings.warn(
            f"Font file not found: {font_path!r}. Falling back to PsychoPy's "
            "system default font. Thai combining tone marks/vowels WILL "
            "likely render incorrectly with the fallback -- verify manually "
            "before any real session, do not assume this is fine.",
            stacklevel=2,
        )
        return None

    def build_display(self, win: visual.Window | None = None):
        """
        Create one TextStim (+ one background highlight Rect, drawn
        behind it) per character position, laid out in an n_rows x
        n_cols grid on a PsychoPy Window (creates one if not supplied).

        Style matches the reference paper's flashboard (black background,
        light gray idle text, INVERTED white-box-behind-black-text for
        the currently flashing group -- not just a text-color change).
        """
        window_kwargs = dict(fullscr=self.fullscreen, color="black", units="norm",
                              screen=self.screen_index, winType=self.win_type,
                              allowGUI=True)
        if not self.fullscreen and self.screen_size is not None:
            window_kwargs["size"] = self.screen_size
        self.win = win or visual.Window(**window_kwargs)
        self.win.mouseVisible = True
        event.clearEvents()

        # Grid centered in normalized [-1, 1] window units.
        x_step = 2.0 / (self.n_cols + 1)
        y_step = 2.0 / (self.n_rows + 1)
        self.stims = []
        self.highlight_rects = []
        for idx, char in enumerate(self.characters):
            row, col = divmod(idx, self.n_cols)
            x = -1.0 + x_step * (col + 1)
            y = 1.0 - y_step * (row + 1)

            rect = visual.Rect(
                self.win, width=x_step * 0.9, height=y_step * 0.9,
                pos=(x, y), fillColor="white", lineColor=None, opacity=0.0,
            )
            self.highlight_rects.append(rect)

            stim = visual.TextStim(
                self.win, text=char, font=self.font_path, height=self.font_size / 1000.0,
                pos=(x, y), color="gainsboro",
            )
            self.stims.append(stim)
        return self.win

    def _draw_frame(self):
        """Draw all highlight rects, then all text on top of them."""
        for rect in self.highlight_rects:
            rect.draw()
        for stim in self.stims:
            stim.draw()

    def create_lsl_outlet(self):
        """
        Marker stream per the shared interface contract: name='P300Markers',
        type='Markers', channel_count=1, sample_rate=0 (irregular),
        dtype='string' -- each sample is one JSON-encoded event string
        ({"event": "flash"/"trial_start"/"trial_end", ...}), not a bare
        integer code, so downstream consumers (collection/record_session.py)
        get full flash-group/target metadata without needing a second
        side-channel.

        _lsl_clock_offset lets us convert a PsychoPy core.Clock timestamp
        (what win.flip() returns) into pylsl's local_clock() domain, so
        the marker's LSL timestamp reflects the ACTUAL GPU flip time, not
        the time push_sample() happened to be called (which lags the
        flip by however long stimulus-list bookkeeping took). This is
        the whole point of using PsychoPy over tkinter.after() -- see
        _push_marker() below.
        """
        info = StreamInfo(self.lsl_stream_name, "Markers", 1, 0, "string", "thai_speller_1")
        self.outlet = StreamOutlet(info)
        self._lsl_clock_offset = local_clock() - core.getTime()
        return self.outlet

    def _push_marker(self, event_dict: dict, flip_time: float):
        """
        Push one JSON-encoded event, timestamped at `flip_time` (a
        core.getTime()-domain timestamp, as returned by win.flip())
        converted into LSL's clock domain -- NOT timestamped at "now"
        (the time push_sample() is called), which would reintroduce the
        software-timer lag this module exists to avoid.
        """
        self.outlet.push_sample([_json.dumps(event_dict)], timestamp=flip_time + self._lsl_clock_offset)

    @staticmethod
    def _check_escape():
        """Raise TrialAborted if Escape was pressed since the last check."""
        if event.getKeys(keyList=["escape"]):
            raise TrialAborted()

    def _wait_with_escape_check(self, duration_s: float, chunk_s: float = 0.1):
        """core.wait() broken into small chunks so a mid-wait Escape press
        (e.g. during the multi-second inter-selection delay) is noticed
        promptly instead of only being checked once the wait ends."""
        remaining = duration_s
        while remaining > 0:
            self._check_escape()
            core.wait(min(chunk_s, remaining))
            remaining -= chunk_s
        self._check_escape()

    def run_trial(self, target_char: str):
        """
        Run one full character-selection trial: n_repetitions of the
        checkerboard flash sequence from flash_sequence.py, with
        PsychoPy win.flip() timing and an LSL marker pushed at each
        flip's actual onset/offset timestamp.

        Press Escape at any point to abort the trial early (checked once
        per flash, so response is within one flash+ISI cycle) -- this
        does not close the window, just returns from run_trial so the
        caller can decide what to do next (e.g. close the window, or
        start a fresh trial).
        """
        if self.win is None or self.outlet is None:
            raise RuntimeError("Call build_display() and create_lsl_outlet() first.")
        if target_char not in self.characters:
            raise ValueError(f"target_char {target_char!r} not in characters list")

        target_idx = self.characters.index(target_char)
        self._trial_id += 1
        trial_id = self._trial_id

        try:
            # Cue phase: highlight the character the subject should attend
            # to for this trial. Without this the subject has no way to know
            # the target, which makes calibration/training data meaningless
            # (is_target labels would not correspond to attended stimuli).
            self.stims[target_idx].color = "lime"
            self._draw_frame()
            trial_start_flip = self.win.flip()
            self._push_marker({
                "event": "trial_start",
                "trial_id": trial_id,
                "target_char_idx": target_idx,
                "target_char": target_char,
            }, trial_start_flip)
            self._wait_with_escape_check(self.cue_duration_ms / 1000.0)
            self.stims[target_idx].color = "gainsboro"
            self._draw_frame()
            self.win.flip()

            config = CheckerboardConfig(n_chars=len(self.characters), n_repetitions=self.n_repetitions)
            flash_groups = build_flash_sequence(config)

            for flash_idx, group in enumerate(flash_groups):
                self._check_escape()

                for idx in group:
                    self.highlight_rects[idx].opacity = 1.0
                    self.stims[idx].color = "black"
                self._draw_frame()
                # win.flip() returns the actual GPU flip timestamp (seconds,
                # core.Clock domain) -- NOT a pre-flip estimate. This is the
                # frame-lock advantage over tkinter.after() software timers.
                onset_flip = self.win.flip()
                self._push_marker({
                    "event": "flash",
                    "trial_id": trial_id,
                    "flash_idx": flash_idx,
                    "group": list(group),
                    "is_target": target_idx in group,
                }, onset_flip)

                core.wait(self.flash_duration_ms / 1000.0)

                for idx in group:
                    self.highlight_rects[idx].opacity = 0.0
                    self.stims[idx].color = "gainsboro"
                self._draw_frame()
                self.win.flip()

                core.wait(self.isi_ms / 1000.0)

            self._push_marker({"event": "trial_end", "trial_id": trial_id}, core.getTime())
            self._wait_with_escape_check(self.inter_selection_delay_ms / 1000.0)
            return True

        except TrialAborted:
            print("Escape pressed -- trial aborted early.")
            self.stims[target_idx].color = "gainsboro"
            return False

    def wait_for_marker_consumers(self, timeout_s: float = 15.0, settle_s: float = 2.0) -> bool:
        """
        Hold the first trial until a marker consumer subscribes, then wait
        settle_s so the others (recorder, inference service, mock headset)
        can subscribe too. LSL does not replay samples to late subscribers,
        so a trial_start pushed before the recorder connects is lost.
        """
        if self.outlet is None:
            raise RuntimeError("Call create_lsl_outlet() first.")
        if not self.outlet.wait_for_consumers(timeout=timeout_s):
            warnings.warn(f"No marker consumer connected within {timeout_s:g} s; starting anyway. "
                          "Markers pushed before a recorder subscribes are lost.", stacklevel=2)
            return False
        core.wait(settle_s)
        return True

    def run_session(self, targets: str, consumer_wait_s: float = 15.0,
                    consumer_settle_s: float = 2.0) -> int:
        """
        Run one trial per character in `targets`, in order -- the
        calibration/training flow: the experimenter supplies the string
        to spell, each character is cued, then flashed.

        Returns:
            number of trials completed (< len(targets) if aborted early
            via Escape).
        """
        unknown = [c for c in targets if c not in self.characters]
        if unknown:
            raise ValueError(
                f"target string contains characters not in the matrix: {unknown}. "
                f"Every character must exist in characters_th.json."
            )

        self.wait_for_marker_consumers(consumer_wait_s, consumer_settle_s)
        completed = 0
        for i, char in enumerate(targets):
            print(f"[stimulus] Trial {i + 1}/{len(targets)}: target {char!r}")
            if not self.run_trial(char):
                print("[stimulus] Session aborted early.")
                break
            completed += 1
        return completed


def verify_thai_font_rendering(font_path: str = "THSarabunNew.ttf",
                                test_string: str = "ก่อ",
                                out_path: str = "results/thai_font_check.png"):
    """
    Manual verification helper (acceptance criterion #4): renders a Thai
    string containing a combining tone mark and saves a screenshot for
    visual inspection. Requires a real display -- not run automatically
    as part of any test suite. After running, open out_path and confirm
    the tone mark (`่`) sits correctly above `อ`, not detached/offset.
    """
    win = visual.Window(size=(400, 200), color="white", units="norm")
    stim = visual.TextStim(win, text=test_string, font=font_path, height=0.3,
                            color="black", pos=(0, 0))
    stim.draw()
    win.flip()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    win.getMovieFrame()
    win.saveMovieFrames(out_path)
    win.close()
    print(f"Saved font check screenshot to {out_path} -- inspect visually.")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Thai checkerboard P300 speller stimulus presentation.")
    parser.add_argument("--targets", type=str, default=None,
                         help="String of target characters to spell, one trial per "
                              "character (e.g. 'กขค'). Defaults to the first "
                              "character in the matrix if omitted.")
    parser.add_argument("--n-repetitions", type=int, default=5,
                         help="Number of full flash sequences per character "
                              "(each sequence = 24 flashes).")
    parser.add_argument("--flash-ms", type=float, default=62,
                         help="Flash duration in milliseconds.")
    parser.add_argument("--isi-ms", type=float, default=62,
                         help="Inter-stimulus interval in milliseconds.")
    parser.add_argument("--cue-ms", type=float, default=2000,
                         help="Cue duration in milliseconds (target highlighted "
                              "before flashing starts).")
    parser.add_argument("--font", type=str, default="THSarabunNew.ttf",
                         help="Path to a Thai Unicode font file.")
    parser.add_argument("--windowed", action="store_true",
                         help="Run in a resizable window instead of fullscreen.")
    parser.add_argument("--width", type=int, default=1024,
                         help="Window width in pixels (--windowed only).")
    parser.add_argument("--height", type=int, default=768,
                         help="Window height in pixels (--windowed only).")
    parser.add_argument("--screen", type=int, default=0,
                         help="Physical display index for multi-monitor setups (0=primary).")
    args = parser.parse_args()

    # Loads the real (but unverified -- see characters_th.json) Thai
    # character list. Tests/automated code should keep using
    # placeholder_characters() instead, per the original scope.
    characters_path = os.path.join(os.path.dirname(__file__), "characters_th.json")
    characters = load_characters(characters_path)
    targets = args.targets if args.targets else characters[0]

    speller = ThaiCheckerboardSpeller(
        characters=characters,
        font_path=args.font,
        flash_duration_ms=args.flash_ms,
        isi_ms=args.isi_ms,
        n_repetitions=args.n_repetitions,
        cue_duration_ms=args.cue_ms,
        fullscreen=not args.windowed,
        screen_size=(args.width, args.height),
        screen_index=args.screen,
    )
    speller.build_display()
    speller.create_lsl_outlet()
    completed = speller.run_session(targets)
    print(f"[stimulus] Completed {completed}/{len(targets)} trials.")
    speller.win.close()
