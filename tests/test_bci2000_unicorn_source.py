"""
End to end: BCI2000 records through UnicornSerialSource from the protocol-accurate fake headset
(tools/fake_unicorn_bt.py over TCP), and BCI2000's own reader (BCI2000Tools.FileReader) checks
the .dat sample by sample: every counter has its slot, real samples carry exactly the values
sent, lost ones are held copies marked Validation = 0, and the data says it is synthetic.

Needs a BCI2000 build with UnicornSerialSource (bci2000/README.md); skips otherwise. Set
BCI2000_ROOT if it is not ~/BCI2000/trunk. Writes only to a temp dir.
"""

import math
import os
import signal as signal_module
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline.constants import EEG_CHANNELS
from pipeline.manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import unicorn_protocol as up  # noqa: E402
from fake_unicorn_bt import FakeUnicorn  # noqa: E402

BCI2000_ROOT = Path(os.environ.get("BCI2000_ROOT", Path.home() / "BCI2000" / "trunk"))
PROG = BCI2000_ROOT / "prog"
pytestmark = pytest.mark.skipif(
    not (PROG / "BCI2000Shell").exists() or not (PROG / "UnicornSerialSource.app").exists(),
    reason="no BCI2000 build with UnicornSerialSource")

COUNTER_START = 5000
LOST_RUN = range(600, 630)                       # 120 ms in one go
DROPPED = {k for k in range(0, 2400) if k % 97 == 50} | set(LOST_RUN)


def signal(k):
    return [120.0 * math.sin(2 * math.pi * k / 50.0 + ch) + 10.0 * ch for ch in range(8)]


def expected_uv(k):
    return np.array([up.uv_to_counts(v) * up.EEG_UV_PER_COUNT for v in signal(k)], dtype=np.float32)


def run_bci2000(tmp_path, tcp, extra_params="", seconds=8, expect_error=False):
    data = tmp_path / "data"
    script = tmp_path / "run.bat"
    script.write_text(f"""Change directory $BCI2000LAUNCHDIR
Reset system
Startup system localhost
Start executable UnicornSerialSource --local --TcpAddress={tcp}
Start executable DummySignalProcessing --local
Start executable DummyApplication --local
Wait for Connected 30
Set parameter DataDirectory {data}
Set parameter SubjectName unicorntest
{extra_params}
Set config
Wait for Resting 30
Sleep 2
Start
Sleep {seconds}
{"" if expect_error else "Stop"}
{"" if expect_error else "Wait for Suspended 10"}
Quit
""")
    # BCI2000 allows one source module at a time; never disturb a session the user is running.
    running = subprocess.run(["pgrep", "-f", str(PROG) + r"/.*(Source|ADC)"], capture_output=True, text=True)
    if running.stdout.strip():
        pytest.skip("another BCI2000 source module is running")
    # The suite runs with QT_QPA_PLATFORM=offscreen for our GUI tests; BCI2000's Qt only ships cocoa.
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    proc = subprocess.Popen([str(PROG / "BCI2000Shell"), str(script)], cwd=PROG, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True, env=env)
    try:
        out, _ = proc.communicate(timeout=60 + seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal_module.SIGKILL)       # the Operator and the modules it started
        out, _ = proc.communicate()
        pytest.fail(f"BCI2000 did not finish:\n{out}")
    files = sorted(data.rglob("*.dat"))
    return files, out


def read_dat(path):
    sys.path.insert(0, str(BCI2000_ROOT / "tools" / "python"))
    from BCI2000Tools.FileReader import bcistream
    b = bcistream(str(path))
    sig, states = b.decode()
    return b, np.asarray(sig), {k: np.asarray(v).ravel() for k, v in states.items()}


def test_lost_packets_keep_their_slots_and_real_samples_are_exact(tmp_path):
    fake = FakeUnicorn(signal=signal, drop=DROPPED, burst=6, counter_start=COUNTER_START).start()
    try:
        files, log = run_bci2000(tmp_path, f"127.0.0.1:{fake.port}")
    finally:
        fake.stop()
    assert len(files) == 1, log
    b, sig, st = read_dat(files[0])

    assert b.params["SamplingRate"] in (250, "250", "250Hz")
    assert list(b.params["ChannelNames"]) == list(EEG_CHANNELS)
    assert sig.shape[0] == 8 and sig.shape[1] >= 6 * 250

    counter = st["Counter"].astype(np.int64)
    k = counter - COUNTER_START
    assert np.all(np.diff(counter) == 1), "a sample slot is missing or duplicated"
    assert k[0] >= 0

    valid = st["Validation"]
    assert set(np.unique(valid)) <= {0, 1}
    lost = np.array([kk in DROPPED for kk in k])
    assert np.array_equal(valid == 0, lost), "Validation must be 0 exactly on the samples the radio lost"
    assert lost.sum() >= len(LOST_RUN) + 10

    for i in np.flatnonzero(~lost):
        np.testing.assert_array_equal(sig[:, i].astype(np.float32), expected_uv(int(k[i])))
    for i in np.flatnonzero(lost):
        np.testing.assert_array_equal(sig[:, i], sig[:, i - 1])          # held, not invented

    gyro = st["GyroscopeX"].astype(np.int64)
    assert np.all(gyro == 0x5A5A), "synthetic data must identify itself"


def test_a_gap_longer_than_max_gap_fill_stops_the_run(tmp_path):
    fake = FakeUnicorn(signal=signal, drop=set(range(1750, 2150)), counter_start=0).start()
    try:
        files, log = run_bci2000(tmp_path, f"127.0.0.1:{fake.port}", "Set parameter MaxGapFill 250", seconds=9,
                                 expect_error=True)
    finally:
        fake.stop()
    assert len(files) == 1, log
    _, sig, st = read_dat(files[0])
    counter = st["Counter"].astype(np.int64)
    assert counter.max() == 1749, "the recording must end at the last sample before the refused gap"
    assert np.all(np.diff(counter) == 1) and np.all(st["Validation"] == 1)


def test_a_headset_that_takes_seconds_to_start_streaming_is_waited_for(tmp_path):
    # The real headset's first bytes arrive 3.5-4 s after the port opens (Bluetooth link setup, measured
    # 2026-10-06). The module used to give up after DataTimeout (3 s): the first real-headset run failed.
    # 6 s is beyond both old limits (DataTimeout 3 s, BCI2000's initial wait 5 s).
    fake = FakeUnicorn(signal=signal, counter_start=0, start_delay_s=6.0).start()
    try:
        files, log = run_bci2000(tmp_path, f"127.0.0.1:{fake.port}", seconds=6)
    finally:
        fake.stop()
    assert len(files) == 1, log
    _, sig, st = read_dat(files[0])
    counter = st["Counter"].astype(np.int64)
    assert sig.shape[1] >= 4 * 250 and np.all(np.diff(counter) == 1) and np.all(st["Validation"] == 1)
