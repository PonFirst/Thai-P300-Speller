"""
Small helpers for bci2000/run_session.ps1 (Windows), as a script instead of `python -c` code: Windows
PowerShell 5.1 mangles quotes inside arguments to native programs.

    python bci2000/session_helper.py encode <text>
        The text as a BCI2000 parameter value (ASCII, %XX-encoded); exit 2 naming any character that
        is not in the matrix.
    python bci2000/session_helper.py calibration-runs <session folder>
        The count of calibration runs in the folder on the first line, then one path per line, oldest
        first: speller runs (they have flashes) recorded without a classifier.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def encode_text(text: str) -> int:
    from make_bci2000_parms import encode, load_characters
    bad = sorted({c for c in text if c not in load_characters()})
    if bad:
        print("run_session: not in the matrix: " + " ".join(bad), file=sys.stderr)
        return 2
    print(encode(text))
    return 0


def calibration_runs(folder: str) -> int:
    from pipeline.bci2000_dat import read_dat
    runs = []
    for f in sorted(Path(folder).glob("*.dat")):
        d = read_dat(f)
        if "StimulusCode" in d.states and "FlashChar1" in d.states and "RiemannModelID" not in d.params:
            runs.append(f)
    print(len(runs))
    for f in runs:
        print(f)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "encode":
        return encode_text(argv[1])
    if len(argv) == 2 and argv[0] == "calibration-runs":
        return calibration_runs(argv[1])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
