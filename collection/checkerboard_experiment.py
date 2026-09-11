"""
Checkerboard P300 speller -- real experiment runner.

IMPORTANT HONESTY NOTE: the flash-grouping logic this imports from
checkerboard.py was fully unit-tested (6 correctness checks, all passing --
see that file). The PsychoPy display code BELOW was NOT execution-tested --
the sandbox this was built in has no display/GL context, so
psychopy.visual couldn't even import. Standard, well-documented PsychoPy
API calls are used throughout, but run test_sync.py FIRST (same folder) on
your actual machine to confirm your display/LSL/hardware chain works before
trusting this file with a real session.

Matrix content is fully configurable via MATRIX_CHARS below -- populate it
with your Thai character set (any Python string works, PsychoPy's TextStim
renders Unicode natively through your system's Thai-supporting font, e.g.
Noto Sans Thai).

Logs BOTH to LSL (for EEG-synced recording via LabRecorder) AND to a local
CSV (redundant backup -- this is the exact information MOABB's public
datasets lost for you; do not depend on a single copy of it).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet

from checkerboard import build_checkerboard_groups, build_flash_sequence

# ---------------------------------------------------------------------------
# Configuration -- edit this section for your actual setup
# ---------------------------------------------------------------------------
N_ROWS, N_COLS = 8, 9

# Placeholder ASCII content -- REPLACE with your actual Thai character set,
# arranged however your protocol specifies. Must have exactly N_ROWS*N_COLS
# entries. Font must support Thai Unicode (e.g. install Noto Sans Thai and
# reference it in FONT_NAME below) if you use real Thai characters here.
MATRIX_CHARS = [[chr(65 + r * N_COLS + c) for c in range(N_COLS)] for r in range(N_ROWS)]
FONT_NAME = None  # None = PsychoPy default font; set to a Thai-supporting font's name/path if using Thai text

N_REPEATS = 15          # flashes per character = N_REPEATS * 24 (see checkerboard.py test 5)
FLASH_DURATION = 0.100  # seconds a flash group stays highlighted
ISI = 0.150             # inter-stimulus interval (gap before next flash)
TARGET_ROW, TARGET_COL = 3, 4  # which character the participant is instructed to attend to THIS trial

LOG_DIR = Path("thai_speller_logs")
LOG_DIR.mkdir(exist_ok=True)


def make_lsl_outlet() -> StreamOutlet:
    info = StreamInfo(
        name="ThaiSpellerMarkers", type="Markers",
        channel_count=1, nominal_srate=0, channel_format="string",
        source_id="thai_speller_checkerboard_v1",
    )
    return StreamOutlet(info)


def make_text_grid(win: visual.Window, matrix_chars: list[list[str]]) -> dict:
    """
    One TextStim per character, positioned in a grid. Returns
    {(row, col): TextStim} so flash logic can look up and highlight
    individual cells.
    """
    n_rows, n_cols = len(matrix_chars), len(matrix_chars[0])
    win_w, win_h = win.size
    cell_w, cell_h = win_w / (n_cols + 1), win_h / (n_rows + 1)

    stims = {}
    for r in range(n_rows):
        for c in range(n_cols):
            x = (c - n_cols / 2 + 0.5) * cell_w
            y = (n_rows / 2 - r - 0.5) * cell_h
            stims[(r, c)] = visual.TextStim(
                win, text=matrix_chars[r][c], pos=(x, y),
                height=cell_w * 0.5, color="grey", font=FONT_NAME,
            )
    return stims


def run_trial(win: visual.Window, stims: dict, groups: dict, outlet: StreamOutlet,
               csv_writer, target_row: int, target_col: int, n_repeats: int, rng: np.random.Generator):
    """
    Runs one full character-selection trial: N_REPEATS repetitions of the
    24-flash checkerboard sequence, with every flash logged to both LSL and
    the local CSV.
    """
    sequence = build_flash_sequence(groups, n_repeats, rng)
    clock = core.Clock()

    # baseline / non-flash frame
    for stim in stims.values():
        stim.color = "grey"
    for stim in stims.values():
        stim.draw()
    win.flip()
    core.wait(0.5)

    for flash_idx, flash_event in enumerate(sequence):
        # inter-stimulus interval -- all characters at rest color
        for stim in stims.values():
            stim.color = "grey"
        for stim in stims.values():
            stim.draw()
        win.flip()
        core.wait(ISI)

        # flash onset -- highlight exactly this event's real cells
        for cell in flash_event.real_cells:
            stims[cell].color = "white"
        for stim in stims.values():
            stim.draw()
        win.flip()
        onset_time = clock.getTime()

        is_target_flash = (target_row, target_col) in flash_event.real_cells
        marker = (
            f"flash|{flash_idx}|{flash_event.phase}|{flash_event.color}|"
            f"{flash_event.virtual_index}|target={int(is_target_flash)}"
        )
        outlet.push_sample([marker])
        csv_writer.writerow([
            time.time(), onset_time, flash_idx, flash_event.phase, flash_event.color,
            flash_event.virtual_index, ";".join(f"{r}-{c}" for r, c in flash_event.real_cells),
            target_row, target_col, int(is_target_flash),
        ])

        core.wait(FLASH_DURATION)

        for cell in flash_event.real_cells:
            stims[cell].color = "grey"

        if event.getKeys(keyList=["escape"]):
            return False  # signal early abort

    return True


def main():
    groups = build_checkerboard_groups(N_ROWS, N_COLS)
    rng = np.random.default_rng()  # real randomness for an actual session -- seed only for debugging

    outlet = make_lsl_outlet()

    win = visual.Window(size=(1280, 800), color="black", units="pix", fullscr=False)
    stims = make_text_grid(win, MATRIX_CHARS)

    log_path = LOG_DIR / f"trial_log_{int(time.time())}.csv"
    with open(log_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "wall_time", "clock_time", "flash_idx", "phase", "color",
            "virtual_index", "real_cells", "target_row", "target_col", "is_target_flash",
        ])

        print(f"Logging to {log_path}")
        print(f"Target character this trial: {MATRIX_CHARS[TARGET_ROW][TARGET_COL]!r} "
              f"(row={TARGET_ROW}, col={TARGET_COL})")

        completed = run_trial(win, stims, groups, outlet, csv_writer,
                                TARGET_ROW, TARGET_COL, N_REPEATS, rng)

    print("Trial completed." if completed else "Trial aborted early (escape pressed).")
    win.close()
    core.quit()


if __name__ == "__main__":
    main()