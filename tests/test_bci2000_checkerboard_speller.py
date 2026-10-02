"""
End to end: BCI2000 runs the checkerboard speller (bci2000/CheckerboardSpeller) in copy mode with
UnicornSerialSource reading the fake headset and BCI2000's stock P3SignalProcessing, loading
bci2000/parms/ThaiCheckerboard.prm. BCI2000's own reader checks the .dat flash by flash:

  - the matrix in the file is collection/characters_th.json;
  - each selection cues the next character of TextToSpell (AttendedChar);
  - flashes are numbered 1..24*repetitions, and the FlashChar states give each flash's group,
    lit exactly while the flash is on screen;
  - the groups are the Python flash_sequence.py groups for the same shuffles, adjacency-safe;
  - StimulusType marks exactly the flashes containing the cued character;
  - flash and gap last the configured 64 ms (16 samples);
  - the classifier reports every flash of every selection, and every selection selects.

It opens the speller window for about 20 s, so it runs only with BCI2000_GUI_TESTS=1 (and a BCI2000
build with both modules, and no other BCI2000 source module running). Writes only to a temp dir.
"""

import json
import math
import os
import signal as signal_module
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline.manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import flash_sequence as fs  # noqa: E402
from fake_unicorn_bt import FakeUnicorn  # noqa: E402
from make_bci2000_parms import OUT as PRM, decode, encode, load_characters  # noqa: E402
from test_bci2000_checkerboard import python_groups_with_same_shuffles  # noqa: E402

BCI2000_ROOT = Path(os.environ.get("BCI2000_ROOT", Path.home() / "BCI2000" / "trunk"))
PROG = BCI2000_ROOT / "prog"
pytestmark = [
    pytest.mark.skipif(os.environ.get("BCI2000_GUI_TESTS") != "1", reason="opens a window: set BCI2000_GUI_TESTS=1"),
    pytest.mark.skipif(not all((PROG / m).exists() for m in
                               ("BCI2000Shell", "UnicornSerialSource.app", "CheckerboardSpeller.app",
                                "P3SignalProcessing.app")), reason="no BCI2000 build with the project modules"),
]

TEXT = "ก่อ"            # consonant, tone mark, consonant: a combining mark gets its own cue
REPS = 2
SAMPLES_PER_FLASH = 16  # 64 ms at 250 Hz


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    running = subprocess.run(["pgrep", "-f", str(PROG) + r"/.*(Source|ADC)"], capture_output=True, text=True)
    if running.stdout.strip():
        pytest.skip("another BCI2000 source module is running")
    tmp = tmp_path_factory.mktemp("cbspeller")
    fake = FakeUnicorn(signal=lambda k: [20 * math.sin(k / 7 + c) for c in range(8)]).start()
    script = tmp / "run.bat"
    script.write_text(f"""Change directory $BCI2000LAUNCHDIR
Reset system
Startup system localhost
Start executable UnicornSerialSource --local --TcpAddress=127.0.0.1:{fake.port}
Start executable P3SignalProcessing --local
Start executable CheckerboardSpeller --local
Wait for Connected 30
Load parameterfile "{PRM}"
Set parameter DataDirectory {tmp / 'data'}
Set parameter SubjectName cbtest
Set parameter InterpretMode 2
Set parameter TextToSpell {encode(TEXT)}
Set parameter NumberOfSequences {REPS}
Set parameter PreSequenceDuration 1s
Set parameter PostSequenceDuration 1.5s
Set config
Wait for Resting 30
Sleep 2
Start
Wait for Suspended 120
Quit
""")
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    proc = subprocess.Popen([str(PROG / "BCI2000Shell"), str(script)], cwd=PROG, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True, env=env)
    try:
        out, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal_module.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"BCI2000 did not finish:\n{out}")
    finally:
        fake.stop()
    files = sorted((tmp / "data").rglob("*.dat"))
    assert len(files) == 1, out
    sys.path.insert(0, str(BCI2000_ROOT / "tools" / "python"))
    from BCI2000Tools.FileReader import bcistream
    b = bcistream(str(files[0]))
    _, states = b.decode()
    st = {k: np.asarray(v).ravel().astype(np.int64) for k, v in states.items()}
    applog = next((tmp / "data").rglob("*.applog")).read_text(encoding="utf-8", errors="replace")
    return b, st, applog


def selections(st):
    ph = st["PhaseInSequence"]
    starts = np.flatnonzero((ph == 1) & (np.r_[0, ph[:-1]] != 1))
    return list(zip(starts, np.r_[starts[1:], len(ph)]))


def onsets(st, s, e):
    code = st["StimulusCode"][s:e]
    return s + np.flatnonzero((code > 0) & (np.r_[0, code[:-1]] == 0))


def groups_at(st, idx):
    return [[int(st[f"FlashChar{j}"][i]) - 1 for j in range(1, 7)] for i in idx]


def test_the_matrix_in_the_file_is_characters_th_json(recording):
    b, _, _ = recording
    defs = b.params["TargetDefinitions"]
    cells = [decode(str(row[0])) if not isinstance(row, str) else decode(row) for row in defs]
    assert cells == load_characters()
    assert int(b.params["NumMatrixRows"]) == 8 and int(b.params["NumMatrixColumns"]) == 9


def test_each_selection_cues_the_next_character(recording):
    _, st, _ = recording
    chars = load_characters()
    expected = [chars.index(c) + 1 for c in TEXT]
    sels = selections(st)
    assert len(sels) == len(TEXT)
    cued = [sorted(set(st["AttendedChar"][s:e]) - {0}) for s, e in sels]
    assert cued == [[x] for x in expected]


def test_flashes_are_numbered_and_their_groups_are_the_checkerboard(recording):
    _, st, _ = recording
    for s, e in selections(st):
        on = onsets(st, s, e)
        assert list(st["StimulusCode"][on]) == list(range(1, 24 * REPS + 1))
        groups = groups_at(st, on)
        assert all(min(g) >= 0 and len(set(g)) == 6 and len({c % 2 for c in g}) == 1 for g in groups)
        assert python_groups_with_same_shuffles(groups, 72, REPS) == groups
        assert fs.validate_no_adjacent_co_flash(groups, 8, 9) == (True, [])


def test_flash_states_are_lit_exactly_while_a_flash_is_on(recording):
    _, st, _ = recording
    lit = st["FlashChar1"] > 0
    assert np.array_equal(lit, st["StimulusCode"] > 0)
    for j in range(2, 7):
        assert np.array_equal(st[f"FlashChar{j}"] > 0, lit)


def test_stimulus_type_marks_exactly_the_flashes_with_the_cued_character(recording):
    _, st, _ = recording
    for s, e in selections(st):
        on = onsets(st, s, e)
        cued = int(max(st["AttendedChar"][s:e])) - 1
        has_target = [cued in g for g in groups_at(st, on)]
        assert list(st["StimulusType"][on] == 1) == has_target
        assert sum(has_target) == 2 * REPS


def test_flash_and_gap_last_64_ms(recording):
    _, st, _ = recording
    code = st["StimulusCode"]
    for s, e in selections(st):
        on = onsets(st, s, e)
        lengths = [int(np.argmax(code[i:] == 0)) for i in on]
        assert set(lengths) == {SAMPLES_PER_FLASH}
        assert set(np.diff(on)) == {2 * SAMPLES_PER_FLASH}


def test_the_classifier_reports_every_flash_and_every_selection_selects(recording):
    _, st, applog = recording
    res = st["StimulusCodeRes"]
    for s, e in selections(st):
        codes = res[s:e][res[s:e] > 0]
        assert sorted(set(codes)) == list(range(1, 24 * REPS + 1))
        assert max(st["SelectedChar"][s:e]) > 0
    assert applog.count("flash result(s) for") == len(TEXT)


def test_every_sample_is_real_and_in_order(recording):
    _, st, _ = recording
    assert np.all(st["Validation"] == 1) and np.all(np.diff(st["Counter"]) == 1)
