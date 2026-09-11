"""
Pure-logic checkerboard flash-sequence construction for the Thai P300
speller (Pananookooln et al. 2022 / AIT Brain Lab reference
implementation: github.com/chaklam-silpasuwanchai/EEG-Real-Time-
Applications/Thai-P300-Speller). No PsychoPy/LSL dependency, so this can
be unit-tested and reasoned about independently of rendering -- a bug
here silently corrupts every recorded row/column label.

CHECKERBOARD COLOR ASSIGNMENT -- IMPORTANT CORRECTNESS NOTE, READ BEFORE
CHANGING THE MATRIX ORIENTATION:
Characters are split into two colors by FLAT LIST-INDEX PARITY (even ->
white, odd -> black), matching the reference implementation exactly:
    white = [i for i in range(n_chars) if i % 2 == 0]
    black = [i for i in range(n_chars) if i % 2 == 1]

If character index i is laid out on the physical n_rows x n_cols display
grid row-major (row, col = divmod(i, n_cols)), flat-index parity equals
TRUE 2D checkerboard parity ((row + col) % 2) if and only if n_cols is
ODD -- because row * n_cols has the same parity as row exactly when
n_cols is odd. With n_cols EVEN, flat-index parity collapses to plain
COLUMN parity: every cell in a column shares one color regardless of
row, so two vertically-adjacent characters can land in the same color
and, after random shuffling, the same flash group.

The reference implementation's matrix is oriented 8 rows x 9 COLUMNS
(odd) -- safe. This project's stated spec (9 rows x 8 columns, even
columns) is NOT safe under this exact row-major mapping unless the
character list is reordered so index parity lines up with (row+col)
parity for that orientation. validate_no_adjacent_co_flash() below
performs a real check (it does not assume the construction is safe) --
run it against your actual n_rows/n_cols/character-order before
trusting adjacency safety for real data collection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class CheckerboardConfig:
    n_chars: int = 72
    n_repetitions: int = 5


def _virtual_shape(half: int) -> tuple[int, int]:
    """
    Factor `half` into (v_rows, v_cols) as close to square as possible,
    v_rows <= v_cols. For half=36 (the 72-character reference case) this
    returns (6, 6), matching the reference's virtual 6x6 matrices
    exactly; smaller/odd sizes (toy grids) degrade gracefully to the
    nearest rectangle so they stay testable.
    """
    if half <= 0:
        raise ValueError("half must be positive")
    best = (1, half)
    for v_rows in range(1, int(half ** 0.5) + 1):
        if half % v_rows == 0:
            best = (v_rows, half // v_rows)
    return best


def build_flash_sequence(config: CheckerboardConfig) -> list[list[int]]:
    """
    Returns a list of flash groups (each a list of character indices)
    for ONE character selection, covering config.n_repetitions
    sequences. Each sequence re-shuffles both colors and flashes, in
    order: all white rows, then all black rows, then all white columns,
    then all black columns (row-major reshape into a (v_rows, v_cols)
    virtual matrix per color -- (6, 6) for the reference's 72-character
    case).

    Every character appears in exactly 2 flash groups per repetition
    (one row-group, one column-group, of its own color) -- i.e.
    2 * config.n_repetitions groups total across the full sequence.
    """
    if config.n_chars % 2 != 0:
        raise ValueError("n_chars must be even (splits into two equal colors)")

    white = [i for i in range(config.n_chars) if i % 2 == 0]
    black = [i for i in range(config.n_chars) if i % 2 == 1]
    half = len(white)
    v_rows, v_cols = _virtual_shape(half)

    flash_groups: list[list[int]] = []
    for _ in range(config.n_repetitions):
        white_shuffled = random.sample(white, half)
        black_shuffled = random.sample(black, half)

        white_matrix = [white_shuffled[r * v_cols:(r + 1) * v_cols] for r in range(v_rows)]
        black_matrix = [black_shuffled[r * v_cols:(r + 1) * v_cols] for r in range(v_rows)]

        flash_groups.extend(white_matrix)  # white rows
        flash_groups.extend(black_matrix)  # black rows
        flash_groups.extend([[white_matrix[r][c] for r in range(v_rows)] for c in range(v_cols)])  # white cols
        flash_groups.extend([[black_matrix[r][c] for r in range(v_rows)] for c in range(v_cols)])  # black cols

    return flash_groups


def validate_no_adjacent_co_flash(flash_sequence: list[list[int]], n_rows: int, n_cols: int):
    """
    Sanity check: no two characters that are physically adjacent
    (4-connected: up/down/left/right, NOT diagonal) on the n_rows x
    n_cols display grid (row-major: row, col = divmod(index, n_cols))
    should ever appear together in the same flash group.

    Returns (ok: bool, violations): violations is a list of
    (char_a, char_b) index pairs that co-flashed while physically
    adjacent -- empty if ok is True.
    """
    def neighbors(idx: int) -> set[int]:
        r, c = divmod(idx, n_cols)
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return {rr * n_cols + cc for rr, cc in candidates if 0 <= rr < n_rows and 0 <= cc < n_cols}

    violations = []
    seen = set()
    for group in flash_sequence:
        group_set = set(group)
        for idx in group:
            for nb in neighbors(idx) & group_set:
                pair = tuple(sorted((idx, nb)))
                if pair not in seen:
                    seen.add(pair)
                    violations.append(pair)

    return len(violations) == 0, violations
