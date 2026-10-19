"""
Small helpers for bci2000/run_session.ps1 and bci2000/windows/menu.ps1 (Windows), as a script instead of
`python -c` code: Windows PowerShell 5.1 mangles quotes inside arguments to native programs.

    python bci2000/session_helper.py encode <text>
        The text as a BCI2000 parameter value (ASCII, %XX-encoded); exit 2 naming any character that
        is not in the matrix.
    python bci2000/session_helper.py calibration-runs <session folder>
        The count of calibration runs in the folder on the first line, then one path per line, oldest
        first: speller runs (they have flashes) recorded without a classifier.
    python bci2000/session_helper.py runs <session folder>
        Every recording in the folder, oldest first, as "<kind><TAB><path>"; kind is check, calibration,
        fixed, transformer or uniform. Empty placeholders and unreadable files are left out.
    python bci2000/session_helper.py find-headset
        The headset's Bluetooth COM port (Windows): the paired outgoing Bluetooth serial port. With
        several candidates, the one whose device answers the start command. Exit 2 if none.
"""

import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "collection"))

PROBE_SECONDS = 10.0          # the headset's first bytes arrive 3.5-4 s after the port opens (measured)
INCOMING_ADDRESS = "000000000000"


def encode_text(text: str) -> int:
    from make_bci2000_parms import encode, load_characters
    bad = sorted({c for c in text if c not in load_characters()})
    if bad:
        print("run_session: not in the matrix: " + " ".join(bad), file=sys.stderr)
        return 2
    print(encode(text))
    return 0


def run_kind(dat) -> str | None:
    """check | calibration | fixed | transformer | uniform, from what the run's .dat records."""
    states, params = dat.states, dat.params
    if "FlashChar1" not in states:
        return "check"
    if "RiemannModelID" not in params:
        return "calibration"
    if "StoppingRule" in params and dat.param_str("StoppingRule") == "0":
        return "fixed"
    if "LanguageModel" in params and dat.param_str("LanguageModel") == "1":
        return "transformer"
    return "uniform"


def list_runs(folder: str) -> list[tuple[str, Path]]:
    from pipeline.bci2000_dat import read_dat
    out = []
    for f in sorted(Path(folder).glob("*.dat")):
        if f.stat().st_size == 0:
            continue                       # BCI2000's placeholder from a configuration without Start
        try:
            out.append((run_kind(read_dat(f)), f))
        except Exception:                  # a run cut off before its header was complete
            continue
    return out


def calibration_runs(folder: str) -> int:
    runs = [f for kind, f in list_runs(folder) if kind == "calibration"]
    print(len(runs))
    for f in runs:
        print(f)
    return 0


def runs(folder: str) -> int:
    for kind, f in list_runs(folder):
        print(f"{kind}\t{f}")
    return 0


def bluetooth_address(hwid: str) -> str | None:
    """The remote device address in a Windows Bluetooth serial port's hardware ID, or None."""
    if not hwid.upper().startswith("BTHENUM"):
        return None
    m = re.search(r"&([0-9A-F]{12})_", hwid.upper())
    return m.group(1) if m else None


def outgoing_bluetooth_ports(ports) -> list[str]:
    """COM ports of paired Bluetooth serial devices. Windows also makes an "incoming" port per device,
    whose hardware ID carries the address 000000000000; that one never receives the headset's data."""
    out = []
    for p in ports:
        address = bluetooth_address(p.hwid or "")
        if address and address != INCOMING_ADDRESS:
            out.append(p.device)
    return out


def answers(port: str, seconds: float = PROBE_SECONDS) -> bool:
    """Whether a Unicorn on this port starts streaming. Sends stop afterwards, so it isn't left running."""
    import serial
    import unicorn_protocol as up
    try:
        s = serial.Serial(port, baudrate=115200, timeout=0.2, write_timeout=2.0)
    except (serial.SerialException, OSError):
        return False
    try:
        s.write(up.START_ACQUISITION)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if s.read(64):
                return True
        return False
    except (serial.SerialException, OSError):
        return False
    finally:
        try:
            s.write(up.STOP_ACQUISITION)
            time.sleep(0.3)
        except Exception:
            pass
        s.close()


def find_headset() -> int:
    from serial.tools import list_ports
    candidates = outgoing_bluetooth_ports(list_ports.comports())
    if len(candidates) == 1:
        print(candidates[0])
        return 0
    if not candidates:
        print("No paired headset found. Switch the headset on and pair it: Settings > Bluetooth & devices > "
              "Add device > Bluetooth > UN-...", file=sys.stderr)
        return 2
    print(f"Several Bluetooth serial ports ({', '.join(candidates)}); asking each for the headset...",
          file=sys.stderr)
    for port in candidates:
        if answers(port):
            print(port)
            return 0
    print(f"None of {', '.join(candidates)} answered. Switch the headset off and on, then try again.",
          file=sys.stderr)
    return 2


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "encode":
        return encode_text(argv[1])
    if len(argv) == 2 and argv[0] == "calibration-runs":
        return calibration_runs(argv[1])
    if len(argv) == 2 and argv[0] == "runs":
        return runs(argv[1])
    if argv == ["find-headset"]:
        return find_headset()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
