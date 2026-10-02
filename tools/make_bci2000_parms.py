"""
Write bci2000/parms/ThaiCheckerboard.prm, the BCI2000 parameter file for the Thai checkerboard
speller, from collection/characters_th.json (the single source of the matrix characters).

    $PY tools/make_bci2000_parms.py            # rewrite the file
    $PY tools/make_bci2000_parms.py --check    # exit 1 if the file is out of date

Load it in the BCI2000 Operator (Config > Load) after the modules start. It sets the matrix, the
paradigm timing and the per-flash classifier setting; subject, session and InterpretMode /
TextToSpell are set per run.

Timing. BCI2000 advances stimuli once per sample block, so durations are whole blocks. At 250 Hz
the source's default block of 10 samples is 40 ms; this file sets 4 samples (16 ms, about one
60 Hz frame) and 64 ms flash + 64 ms gap (128 ms SOA), the nearest whole blocks to the paradigm's
nominal 62 + 62 ms. (The PsychoPy stimulus never achieved 62 ms either: CLAUDE.md gotcha 26.)
The .dat file records what actually happened, so analysis never relies on these nominal values.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS = REPO_ROOT / "collection" / "characters_th.json"
OUT = REPO_ROOT / "bci2000" / "parms" / "ThaiCheckerboard.prm"

N_ROWS, N_COLS = 8, 9


def encode(value: str) -> str:
    """BCI2000's parameter encoding (EncodedString::InsertInto): printable ASCII other than space,
    % and the brackets BCI2000 reads as list delimiters ({}[]) passes through; every other byte (each byte
    of a UTF-8 character) becomes %XX."""
    if value == "":
        return "%"
    out = []
    for b in value.encode("utf-8"):
        c = chr(b)
        if 0x21 <= b <= 0x7E and c not in "%{}[]":
            out.append(c)
        else:
            out.append(f"%{b:02X}")
    return "".join(out)


def decode(token: str) -> str:
    """The inverse of encode(), for checks."""
    if token == "%":
        return ""
    raw, i = bytearray(), 0
    while i < len(token):
        hex_digits = token[i + 1:i + 3]
        if token[i] == "%" and len(hex_digits) == 2 and all(c in string.hexdigits for c in hex_digits):
            raw.append(int(hex_digits, 16))
            i += 3
        else:
            raw.append(ord(token[i]))
            i += 1
    return raw.decode("utf-8")


def load_characters() -> list[str]:
    data = json.loads(CHARACTERS.read_text(encoding="utf-8"))
    chars = data["characters"] if isinstance(data, dict) else data
    if len(chars) != N_ROWS * N_COLS:
        raise SystemExit(f"{CHARACTERS} has {len(chars)} characters, expected {N_ROWS * N_COLS}")
    return list(chars)


def render() -> str:
    chars = load_characters()
    entries = " ".join(f"{encode(c)} {encode(c)}" for c in chars)
    lines = [
        "Application:Speller%20Targets matrix TargetDefinitions= "
        f"{len(chars)} {{Display Enter}} {entries} "
        "// generated from collection/characters_th.json by tools/make_bci2000_parms.py",
        f"Application:Speller%20Targets int NumMatrixRows= {N_ROWS} 8 1 % // rows of the display matrix",
        f"Application:Speller%20Targets int NumMatrixColumns= {N_COLS} 9 1 % "
        "// columns; odd, so the checkerboard is adjacency-safe",
        "Application:Sequencing int NumberOfSequences= 5 5 1 % // repetitions per selection",
        "Application:Sequencing float StimulusDuration= 64ms 40ms 0 % // flash (4 blocks of 16 ms)",
        "Application:Sequencing float ISIMinDuration= 64ms 80ms 0 % // gap between flashes",
        "Application:Sequencing float ISIMaxDuration= 64ms 80ms 0 % // fixed gap: same as the minimum",
        "Application:Sequencing float PreSequenceDuration= 2s 2s 0 % // cue: the target lit in CueColor (copy mode)",
        "Application:Sequencing float PostSequenceDuration= 3504ms 2s 0 % "
        "// pause after each selection: the paradigm's 3.5 s, rounded to whole 16 ms blocks",
        "Application:Result%20Processing int InterpretMode= 1 0 0 2 "
        "// 0 none, 1 free, 2 copy (calibration: copy + TextToSpell) (enumeration)",
        "Source:Signal%20Properties int SampleBlockSize= 4 4 1 % // 16 ms blocks at 250 Hz",
        "Filtering int EpochLength= 1400ms 500ms 0 % // time until a flash's result is in: CheckerboardSignalProcessing "
        "reports 1.3 s after onset (0.8 s epoch + 0.5 s zero-phase pad), and the speller waits this long",
        "Filtering int EpochsToAverage= 1 1 0 % // one result per flash: every flash has its own stimulus code",
        "Filtering int SingleEpochMode= 1 0 0 1 "
        "// report every epoch; otherwise P3TemporalFilter drops every other selection's results (boolean)",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--check", action="store_true", help="exit 1 if the file is missing or out of date")
    args = p.parse_args(argv)
    text = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if current != text:
            print(f"{OUT} is out of date: run tools/make_bci2000_parms.py", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
