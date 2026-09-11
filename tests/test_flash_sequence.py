import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "collection"))

from flash_sequence import (
    CheckerboardConfig,
    build_flash_sequence,
    validate_no_adjacent_co_flash,
)

# NOTE on grid choice: flat-index-parity checkerboard coloring is only a
# TRUE 2D checkerboard (safe against both row- and column-adjacency) when
# n_cols is ODD -- see flash_sequence.py's module docstring. A 4x4 toy grid
# (even columns) would NOT satisfy this and could show real adjacency
# violations, which is a genuine property of the construction, not a test
# bug. We use a 4-row x 5-col (odd columns) toy grid here so this test
# actually exercises the "safe" regime; the reference implementation's own
# matrix (8 rows x 9 columns) is odd-column for the same reason.
TOY_N_ROWS = 4
TOY_N_COLS = 5
TOY_N_CHARS = TOY_N_ROWS * TOY_N_COLS  # 20


def test_no_adjacent_co_flash_on_toy_grid():
    config = CheckerboardConfig(n_chars=TOY_N_CHARS, n_repetitions=5)
    flash_sequence = build_flash_sequence(config)
    ok, violations = validate_no_adjacent_co_flash(flash_sequence, TOY_N_ROWS, TOY_N_COLS)
    assert ok, f"Adjacent characters co-flashed: {violations}"


def test_every_character_appears_twice_per_repetition():
    n_repetitions = 5
    config = CheckerboardConfig(n_chars=TOY_N_CHARS, n_repetitions=n_repetitions)
    flash_sequence = build_flash_sequence(config)

    counts = {i: 0 for i in range(TOY_N_CHARS)}
    for group in flash_sequence:
        for idx in group:
            counts[idx] += 1

    for idx, count in counts.items():
        assert count == 2 * n_repetitions, (
            f"char {idx} appeared in {count} flash groups, "
            f"expected {2 * n_repetitions}"
        )


def test_flash_groups_reshuffled_across_repetitions():
    config = CheckerboardConfig(n_chars=TOY_N_CHARS, n_repetitions=5)
    flash_sequence = build_flash_sequence(config)

    # Each repetition contributes the same number of flash groups
    # (2 * v_rows + 2 * v_cols for the toy grid's virtual matrix shape).
    groups_per_repetition = len(flash_sequence) // config.n_repetitions
    first_rep = flash_sequence[:groups_per_repetition]
    second_rep = flash_sequence[groups_per_repetition:2 * groups_per_repetition]

    assert first_rep != second_rep, (
        "Flash groups were identical across repetitions -- shuffling is "
        "not happening (or got an astronomically unlucky random draw)."
    )


def test_build_flash_sequence_requires_even_n_chars():
    config = CheckerboardConfig(n_chars=15, n_repetitions=1)
    try:
        build_flash_sequence(config)
        assert False, "expected ValueError for odd n_chars"
    except ValueError:
        pass


def test_reference_scale_72_chars_uses_6x6_virtual_matrix():
    """
    Sanity check against the reference's stated 72-character / 6x6
    virtual matrix construction: 24 flash groups per repetition
    (6 white rows + 6 black rows + 6 white cols + 6 black cols).
    """
    config = CheckerboardConfig(n_chars=72, n_repetitions=1)
    flash_sequence = build_flash_sequence(config)
    assert len(flash_sequence) == 24
    for group in flash_sequence:
        assert len(group) == 6
