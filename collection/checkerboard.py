"""
Checkerboard paradigm construction (Townsend et al. 2010), matching the
8x9 matrix used by both Townsend's original study and Pananookooln et
al.'s Thai speller.

Deliberately pure logic, no PsychoPy/LSL/display dependency -- this is the
part where a bug would silently corrupt an entire real recording session
(wrong flash groupings = wrong ground-truth labels = your row/column
decoding is unrecoverable after the fact). Fully unit-tested below.

CONSTRUCTION NOTE: Townsend's paper describes mapping each checkerboard
color's 36 characters into a virtual 6x6 matrix, but the exact
cell-assignment scan order isn't fully specified in what I could verify
from the paper's figure captions and abstract. This implementation uses
row-major scan order (straightforward, reproducible, deterministic given
a fixed matrix) as a reasonable default -- if exact replication of
Townsend/Pananookooln's specific assignment matters for your methods
section, that's worth confirming against the original paper's full
methods text or supplementary material directly, not assumed from this
code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlashEvent:
    """One flash event -- everything you need to log for it."""
    phase: str          # 'row' or 'col'
    color: str           # 'A' or 'B' (the two checkerboard colors)
    virtual_index: int   # which row/col (0-5) of that color's virtual 6x6 matrix
    real_cells: list[tuple[int, int]]  # actual (row, col) positions on the real matrix that flash together


def build_checkerboard_groups(n_rows: int, n_cols: int) -> dict:
    """
    Assigns every real (row, col) cell a checkerboard color ('A'/'B') and a
    position within that color's virtual matrix.

    REQUIRES: n_rows * n_cols must be even (so the two colors split evenly)
    AND (n_rows * n_cols) // 2 must itself factor into a rectangle -- for
    the standard 8x9 case, 72/2=36=6x6, matching Townsend/Pananookooln
    exactly. Raises ValueError if these don't hold, rather than silently
    producing a lopsided or malformed virtual matrix.

    Returns a dict with:
      'color_grid':    (n_rows, n_cols) array of 'A'/'B'
      'vpos_grid':      (n_rows, n_cols, 2) array -- (virtual_row, virtual_col)
                        within that cell's color's virtual matrix
      'vshape':         (virtual_rows, virtual_cols) shape of EACH color's
                        virtual matrix
    """
    total = n_rows * n_cols
    if total % 2 != 0:
        raise ValueError(
            f"{n_rows}x{n_cols}={total} is odd -- checkerboard colors can't "
            f"split evenly. This construction requires an even total cell count."
        )
    half = total // 2

    # Find a rows'xcols' rectangle for half the cells. For 8x9 -> half=36 -> 6x6.
    vshape = None
    for vr in range(1, half + 1):
        if half % vr == 0:
            vc = half // vr
            # Prefer the most "square" factorization, matching the literature's
            # 6x6 for the 8x9 case -- not load-bearing correctness-wise, just
            # matches convention.
            if vshape is None or abs(vr - vc) < abs(vshape[0] - vshape[1]):
                vshape = (vr, vc)
    if vshape is None:
        raise ValueError(f"Could not factor {half} into a rectangle.")

    color_grid = np.empty((n_rows, n_cols), dtype="<U1")
    vpos_grid = np.zeros((n_rows, n_cols, 2), dtype=int)

    counters = {"A": 0, "B": 0}
    for r in range(n_rows):
        for c in range(n_cols):
            color = "A" if (r + c) % 2 == 0 else "B"
            color_grid[r, c] = color
            idx = counters[color]
            counters[color] += 1
            vr, vc = divmod(idx, vshape[1])
            vpos_grid[r, c] = (vr, vc)

    assert counters["A"] == half and counters["B"] == half, (
        "Color counts didn't split evenly -- this should be unreachable given "
        "the even-total check above; investigate if this assertion ever fires."
    )

    return {"color_grid": color_grid, "vpos_grid": vpos_grid, "vshape": vshape}


def build_flash_events(groups: dict) -> list[FlashEvent]:
    """
    Builds the full list of possible flash events (one per virtual row/col,
    per color) -- NOT yet ordered into a randomized sequence. This is the
    fixed structure; build_flash_sequence() below handles randomization and
    repeats.
    """
    color_grid = groups["color_grid"]
    vpos_grid = groups["vpos_grid"]
    vrows, vcols = groups["vshape"]
    n_rows, n_cols = color_grid.shape

    events = []

    # Row-phase events: for each color, for each virtual row index, collect
    # every real cell whose vpos matches (virtual_row == that index).
    for color in ("A", "B"):
        for vr in range(vrows):
            real_cells = [
                (r, c) for r in range(n_rows) for c in range(n_cols)
                if color_grid[r, c] == color and vpos_grid[r, c, 0] == vr
            ]
            events.append(FlashEvent(phase="row", color=color, virtual_index=vr, real_cells=real_cells))

    # Col-phase events: same idea, grouped by virtual column instead.
    for color in ("A", "B"):
        for vc in range(vcols):
            real_cells = [
                (r, c) for r in range(n_rows) for c in range(n_cols)
                if color_grid[r, c] == color and vpos_grid[r, c, 1] == vc
            ]
            events.append(FlashEvent(phase="col", color=color, virtual_index=vc, real_cells=real_cells))

    return events


def build_flash_sequence(groups: dict, n_repeats: int, rng: np.random.Generator) -> list[FlashEvent]:
    """
    Full randomized flash sequence for ONE character-selection trial:
    n_repeats repetitions, each repetition = all row-phase flashes
    (randomized order) followed by all col-phase flashes (randomized
    order) -- matching Townsend's "rows of each matrix flash first (12
    flashes), then columns (12 flashes)" structure.
    """
    all_events = build_flash_events(groups)
    row_events = [e for e in all_events if e.phase == "row"]
    col_events = [e for e in all_events if e.phase == "col"]

    sequence = []
    for _ in range(n_repeats):
        row_order = row_events.copy()
        rng.shuffle(row_order)
        col_order = col_events.copy()
        rng.shuffle(col_order)
        sequence.extend(row_order)
        sequence.extend(col_order)

    return sequence


def find_target_flashes(groups: dict, target_row: int, target_col: int, sequence: list[FlashEvent]) -> list[int]:
    """
    Returns the indices into `sequence` where the target character's cell
    was actually flashed -- useful for verification/debugging (the target
    should be flashed in EXACTLY 2 events per repeat: one row-phase, one
    col-phase, since it belongs to exactly one virtual row and one virtual
    column of its color).
    """
    return [
        i for i, ev in enumerate(sequence)
        if (target_row, target_col) in ev.real_cells
    ]


# ---------------------------------------------------------------------------
# Self-tests -- run this file directly. These are the checks that matter
# most: get any of these wrong and your entire collected dataset's ground
# truth is silently corrupted.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("Test 1: 8x9 matrix produces two clean 6x6 virtual matrices")
    groups = build_checkerboard_groups(8, 9)
    assert groups["vshape"] == (6, 6), f"expected (6,6), got {groups['vshape']}"
    print(f"  vshape = {groups['vshape']}  [OK]")

    print("\nTest 2: adjacency -- every orthogonal neighbor has the OPPOSITE color")
    color_grid = groups["color_grid"]
    n_rows, n_cols = color_grid.shape
    violations = 0
    for r in range(n_rows):
        for c in range(n_cols):
            for dr, dc in [(0, 1), (1, 0)]:  # right and down neighbors (avoids double-counting)
                nr, nc = r + dr, c + dc
                if nr < n_rows and nc < n_cols:
                    if color_grid[r, c] == color_grid[nr, nc]:
                        violations += 1
    assert violations == 0, f"{violations} adjacent same-color pairs found -- checkerboard construction is broken"
    print(f"  {n_rows*n_cols} cells checked, 0 adjacency violations  [OK]")

    print("\nTest 3: every character belongs to EXACTLY one virtual row and one virtual column")
    vpos_grid = groups["vpos_grid"]
    for color in ("A", "B"):
        mask = color_grid == color
        vpos_flat = vpos_grid[mask]
        assert len(vpos_flat) == 36, f"color {color} has {len(vpos_flat)} cells, expected 36"
        unique_positions = set(map(tuple, vpos_flat))
        assert len(unique_positions) == 36, (
            f"color {color}: {36 - len(unique_positions)} virtual positions are DUPLICATED "
            f"-- two different real characters would be indistinguishable by flash group"
        )
    print("  Each color: 36 cells, 36 unique virtual positions, no collisions  [OK]")

    print("\nTest 4: a full flash sequence flashes the target EXACTLY 2x per repeat (1 row, 1 col)")
    target_row, target_col = 3, 4
    n_repeats = 15
    sequence = build_flash_sequence(groups, n_repeats, rng)
    target_flash_indices = find_target_flashes(groups, target_row, target_col, sequence)
    assert len(target_flash_indices) == 2 * n_repeats, (
        f"expected {2*n_repeats} target flashes ({n_repeats} repeats x 2 phases), "
        f"got {len(target_flash_indices)}"
    )
    # Also confirm exactly 1 row-phase and 1 col-phase hit per repeat, not e.g. 2 row + 0 col
    row_hits = sum(1 for i in target_flash_indices if sequence[i].phase == "row")
    col_hits = sum(1 for i in target_flash_indices if sequence[i].phase == "col")
    assert row_hits == n_repeats and col_hits == n_repeats, (
        f"expected {n_repeats} row hits and {n_repeats} col hits, got {row_hits} row, {col_hits} col"
    )
    print(f"  Target flashed {len(target_flash_indices)} times across {n_repeats} repeats "
          f"({row_hits} row-phase, {col_hits} col-phase)  [OK]")

    print("\nTest 5: total sequence length matches expectation (24 flashes/repeat: 12 row + 12 col)")
    assert len(sequence) == n_repeats * 24, f"expected {n_repeats*24}, got {len(sequence)}"
    print(f"  {len(sequence)} total flashes for {n_repeats} repeats  [OK]")

    print("\nTest 6: adjacent characters NEVER share a flash group (the whole point of CBP)")
    # Pick target's 4 orthogonal neighbors, confirm none of them appear in
    # any flash event that also contains the target.
    neighbors = [(target_row+dr, target_col+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                 if 0 <= target_row+dr < n_rows and 0 <= target_col+dc < n_cols]
    for ev_idx in target_flash_indices:
        ev = sequence[ev_idx]
        shared = set(ev.real_cells) & set(neighbors)
        assert not shared, f"Adjacency violation: target's flash group also contains neighbor(s) {shared}"
    print(f"  Checked {len(neighbors)} orthogonal neighbors of target across all "
          f"{len(target_flash_indices)} target flashes: never co-flashed  [OK]")

    print("\nAll checks passed.")