"""
Unicorn relay tests over real LSL: tools/fake_unicorn_bt.py speaks the real protocol on a TCP socket,
the relay turns it into LSL streams, and an inlet configured like the recorder's reads them back.
Stream types are randomized so nothing here can meet a production stream. Files go under tmp_path.
"""

import json
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pytest
from pylsl import StreamInlet, local_clock, proc_clocksync, resolve_byprop

from pipeline.constants import EEG_CHANNELS, EEG_SFREQ
from pipeline.manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import fake_unicorn_bt as fake  # noqa: E402
import unicorn_protocol as up  # noqa: E402
import unicorn_relay as relay_mod  # noqa: E402

SCALE = up.EEG_UV_PER_COUNT


def known_signal(k):
    """Channel c at counter k: a value the test can recompute, well inside the 24-bit range."""
    return [50.0 * (c + 1) + (k % 40) for c in range(8)]


def counter_offset(data):
    """LSL does not replay to late subscribers, so an inlet's first sample is not counter 0. Channel 0
    is 50 + (counter % 40), which pins the alignment modulo 40, all the expected values need."""
    return int(round(data[0, 0] - 50.0))


class Rig:
    def __init__(self, tmp_path, *, server=None, relay=None):
        tag = uuid.uuid4().hex[:10]
        self.eeg_type, self.aux_type = f"EEGTEST_{tag}", f"AuxTEST_{tag}"
        self.lines = []
        self.server = fake.FakeUnicorn(**{"signal": known_signal, **(server or {})}).start()
        self.report = tmp_path / "relay_report.json"
        kwargs = dict(stream_type=self.eeg_type, aux_type=self.aux_type, warmup_s=0.5, report_path=self.report,
                      log=self.lines.append)
        kwargs.update(relay or {})
        self.relay = relay_mod.UnicornRelay(relay_mod.TcpSource("127.0.0.1", self.server.port), **kwargs)
        self.stop = threading.Event()
        self.codes = []
        self.thread = threading.Thread(target=lambda: self.codes.append(self.relay.run(self.stop)), daemon=True)
        self.thread.start()

    def inlet(self, type_, flags=proc_clocksync):
        found = resolve_byprop("type", type_, timeout=10)
        assert found, f"stream of type {type_} never appeared; relay log: {self.lines}"
        inlet = StreamInlet(found[0], processing_flags=flags)   # default: what the recorder and service use
        inlet.open_stream(timeout=5)
        return inlet

    def pull(self, inlet, n, timeout=15):
        samples, stamps, deadline = [], [], time.monotonic() + timeout
        while len(samples) < n and time.monotonic() < deadline:
            chunk, ts = inlet.pull_chunk(timeout=0.1, max_samples=512)
            samples += chunk
            stamps += ts
        return np.array(samples), np.array(stamps)

    def finish(self, timeout=10):
        self.stop.set()
        self.thread.join(timeout=timeout)
        self.server.stop()
        assert not self.thread.is_alive()
        return self.codes[0]


@pytest.fixture
def rig(tmp_path):
    rigs = []

    def make(**kw):
        r = Rig(tmp_path, **kw)
        rigs.append(r)
        return r

    yield make
    for r in rigs:
        r.stop.set()
        r.server.stop()


def test_streams_carry_the_contract_the_recorder_and_service_expect(rig):
    r = rig(server={"n_samples": 1500, "synthetic": False})
    inlet = r.inlet(r.eeg_type)
    info = inlet.info()
    assert (info.channel_count(), info.nominal_srate(), info.type()) == (8, EEG_SFREQ, r.eeg_type)
    assert info.channel_format() == 1                      # cf_float32
    desc = info.desc()
    assert desc.child_value("manufacturer") == "g.tec" and desc.child_value("synthetic") == "false"
    labels, units, channel = [], set(), desc.child("channels").child("channel")
    while channel.name():
        labels.append(channel.child_value("label"))
        units.add(channel.child_value("unit"))
        channel = channel.next_sibling()
    assert labels == EEG_CHANNELS and units == {"microvolts"}

    data, ts = r.pull(inlet, 800)
    assert data.shape[1] == 8 and len(data) >= 800
    k0 = counter_offset(data)
    expected = np.array([known_signal(k0 + i) for i in range(len(data))])
    quantized = np.round(expected / SCALE) * SCALE
    np.testing.assert_allclose(data, quantized, atol=0.02)         # microvolts, straight from the counts
    assert r.finish() == relay_mod.EXIT_OK


def test_timestamps_are_a_clean_4ms_grid_and_close_to_now(rig):
    r = rig(server={"n_samples": 1500, "burst": 6})                # delivered in radio-like bursts
    # No inlet post-processing here: this checks the relay's own stamps. LSL's clock-sync correction
    # is a client-side effect that adds tens of microseconds of noise under load, and is not the relay's.
    inlet = r.inlet(r.eeg_type, flags=0)
    data, ts = r.pull(inlet, 900)
    steps = np.diff(ts)
    assert np.all(steps > 0)
    assert np.max(np.abs(steps - 0.004)) < 2e-5                    # bursts of 6 do not show in the stamps
    assert 0.0 <= local_clock() - ts[-1] < 1.5
    r.finish()
    assert r.relay.report()["samples"]["timestamps_clamped"] == 0


def test_dropped_packets_are_counted_from_the_devices_own_counter_and_show_as_a_stamp_gap(rig):
    r = rig(server={"n_samples": 1500, "drop": set(range(400, 410)) | {800}})
    inlet = r.inlet(r.eeg_type)
    data, ts = r.pull(inlet, 1100)
    assert len(data) >= 1100                                       # the inlet missed the first second: see counter_offset
    steps = np.diff(ts)
    big = np.flatnonzero(steps > 0.006)
    assert len(big) == 2 and steps[big[0]] == pytest.approx(11 * 0.004, abs=2e-5)
    assert steps[big[1]] == pytest.approx(2 * 0.004, abs=2e-5)
    r.finish()
    rep = json.loads(r.report.read_text(encoding="utf-8"))
    s = rep["samples"]
    assert (s["missing"], s["n_gaps"], s["longest_gap_samples"]) == (11, 2, 10)
    assert s["received"] > 800                                     # the stream was stopped by finish(), not run out
    assert s["received"] + s["missing"] == pytest.approx(rep["duration"]["device_s"] * EEG_SFREQ, abs=1)
    assert s["loss_pct"] == pytest.approx(100 * 11 / (s["received"] + 11), abs=1e-3)
    assert [g["missing"] for g in rep["gaps"]] == [10, 1] and rep["gaps"][0]["counter"] == 410


def test_garbage_on_the_link_is_survived_and_counted(rig):
    r = rig(server={"n_samples": 1000, "garbage": {300: b"\x11\x22\x33\x44\x55", 600: up.HEADER + b"\x00" * 9}})
    inlet = r.inlet(r.eeg_type)
    data, _ = r.pull(inlet, 850)
    k0 = counter_offset(data)
    quantized = np.round(np.array([known_signal(k0 + i) for i in range(len(data))]) / SCALE) * SCALE
    np.testing.assert_allclose(data, quantized, atol=0.02)         # no sample corrupted by the junk
    r.finish()
    f = json.loads(r.report.read_text(encoding="utf-8"))["framing"]
    assert f["bytes_discarded"] == 5 + 11 and f["resyncs"] == 2 and f["frames"] > 850 and f["ack_received"]


def test_the_aux_stream_carries_battery_motion_and_the_counter(rig):
    r = rig(server={"n_samples": 1000, "synthetic": False})
    aux = r.inlet(r.aux_type)
    assert (aux.info().channel_count(), aux.info().channel_format()) == (8, 2)   # cf_double64
    data, _ = r.pull(aux, 500)
    assert data[:, 0] == pytest.approx(80.0)                                      # battery nibble 12 of 15
    assert data[0, 4:7] == pytest.approx([3 / 32.8, -5 / 32.8, 8 / 32.8])
    assert data[:, 7].tolist() == list(range(int(data[0, 7]), int(data[0, 7]) + len(data)))   # the device counter
    r.finish()


def test_synthetic_data_names_itself_and_real_data_keeps_the_real_name(rig):
    fake_rig = rig(server={"n_samples": 800, "synthetic": True}, relay={"stream_name": None})
    inlet = fake_rig.inlet(fake_rig.eeg_type)
    assert inlet.info().name() == "MockUnicornEEG" and inlet.info().desc().child_value("synthetic") == "true"
    assert any("SYNTHETIC" in line for line in fake_rig.lines)
    fake_rig.finish()
    assert json.loads(fake_rig.report.read_text(encoding="utf-8"))["stream"]["synthetic"] is True

    real_rig = rig(server={"n_samples": 800, "synthetic": False}, relay={"stream_name": None})
    assert real_rig.inlet(real_rig.eeg_type).info().name() == "UnicornEEG"
    real_rig.finish()


def test_start_and_stop_commands_are_the_documented_bytes(rig):
    r = rig(server={"n_samples": 800})
    r.inlet(r.eeg_type)
    assert r.finish() == relay_mod.EXIT_OK
    assert r.server.commands == [up.START_ACQUISITION, up.STOP_ACQUISITION]


def test_duration_stops_the_relay_and_the_report_says_why(rig):
    r = rig(server={"n_samples": None}, relay={"duration_s": 1.5})
    assert r.thread.join(timeout=15) is None and not r.thread.is_alive()
    assert r.codes == [relay_mod.EXIT_OK]
    rep = json.loads(r.report.read_text(encoding="utf-8"))
    assert "--duration" in rep["exit"]["reason"] and rep["samples"]["received"] > 500
    assert rep["samples"]["missing"] == 0 and rep["delivery_delay_ms"]["p99"] is not None
    r.server.stop()


def test_a_link_that_closes_exits_4_and_still_writes_its_report(rig):
    r = rig(server={"n_samples": None, "close_after": 400})
    r.thread.join(timeout=15)
    assert r.codes == [relay_mod.EXIT_LINK_LOST] and "link was lost" in r.relay.failure_message
    rep = json.loads(r.report.read_text(encoding="utf-8"))
    assert rep["exit"]["code"] == 4 and 300 < rep["samples"]["received"] <= 400


def test_a_link_that_goes_silent_exits_4(rig):
    r = rig(server={"n_samples": None, "silent_after": 300}, relay={"data_timeout_s": 1.0})
    r.thread.join(timeout=15)
    assert r.codes == [relay_mod.EXIT_LINK_LOST] and "no data from the headset" in r.relay.failure_message


def test_no_headset_exits_3_after_the_connect_timeout(tmp_path):
    lines = []
    dead = fake.FakeUnicorn()
    port = dead.port
    dead.stop()                                                     # nothing listens here any more
    relay = relay_mod.UnicornRelay(relay_mod.TcpSource("127.0.0.1", port, connect_timeout=0.5),
                                   connect_timeout_s=1.5, report_path=tmp_path / "r.json", log=lines.append)
    t0 = time.monotonic()
    assert relay.run(threading.Event()) == relay_mod.EXIT_NO_CONNECT
    assert time.monotonic() - t0 < 6 and "could not connect" in relay.failure_message


def test_ctrl_c_during_connect_is_a_clean_stop(tmp_path):
    dead = fake.FakeUnicorn()
    port = dead.port
    dead.stop()
    relay = relay_mod.UnicornRelay(relay_mod.TcpSource("127.0.0.1", port, connect_timeout=0.5),
                                   connect_timeout_s=30, log=lambda m: None)
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    assert relay.run(stop) == relay_mod.EXIT_OK


def test_the_serial_port_path_works_over_a_pseudo_terminal(tmp_path):
    """SerialSource is the code that talks to /dev/cu.UN-<serial>; a pty behaves like that device."""
    pty = pytest.importorskip("pty")
    import os
    import tty

    master, slave = pty.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    path = os.ttyname(slave)
    got_start = threading.Event()

    def headset():
        buf, deadline = b"", time.monotonic() + 10
        while up.START_ACQUISITION not in buf and time.monotonic() < deadline:
            try:
                buf += os.read(master, 64)
            except BlockingIOError:
                time.sleep(0.002)
        got_start.set()
        os.write(master, up.ACK)
        for k in range(400):
            try:
                os.write(master, up.encode_payload([1000 + k] * 8, k, gyr_counts=up.SYNTHETIC_GYRO_COUNTS))
            except (BlockingIOError, OSError):
                return
            time.sleep(0.004)

    threading.Thread(target=headset, daemon=True).start()
    relay = relay_mod.UnicornRelay(relay_mod.SerialSource(path), stream_type=f"EEGPTY_{uuid.uuid4().hex[:8]}",
                                   aux_type=f"AuxPTY_{uuid.uuid4().hex[:8]}", warmup_s=0.3, duration_s=0.9,
                                   report_path=tmp_path / "pty_report.json", log=lambda m: None)
    t0 = time.monotonic()
    assert relay.run(threading.Event()) == relay_mod.EXIT_OK
    assert time.monotonic() - t0 < 10 and got_start.is_set()          # and no hang writing the stop command
    rep = relay.report()
    assert rep["samples"]["received"] > 200 and rep["samples"]["missing"] == 0 and rep["framing"]["ack_received"]
    assert rep["source"] == f"serial:{path}"
    os.close(master)


def test_the_serial_port_is_opened_exclusively(monkeypatch):
    seen = {}

    class FakeSerial:
        def __init__(self, port, **kwargs):
            seen.update(kwargs, port=port)

    import serial
    monkeypatch.setattr(serial, "Serial", FakeSerial)
    relay_mod.SerialSource("/dev/cu.UN-TEST").open()
    assert seen["exclusive"] is True and seen["port"] == "/dev/cu.UN-TEST" and seen["write_timeout"] == 2.0


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def test_cli_records_a_report(tmp_path, capsys):
    server = fake.FakeUnicorn(signal=known_signal).start()
    report = tmp_path / "logs" / "gap_test.json"
    code = relay_mod.main(["--tcp", f"127.0.0.1:{server.port}", "--duration", "1.5", "--warmup-s", "0.5",
                           "--stream-type", f"EEGTEST_{uuid.uuid4().hex[:8]}",
                           "--aux-type", f"AuxTEST_{uuid.uuid4().hex[:8]}", "--report", str(report)])
    server.stop()
    out = capsys.readouterr().out
    assert code == 0 and "[unicorn_relay] done:" in out and "0 lost" in out
    assert json.loads(report.read_text(encoding="utf-8"))["format"] == relay_mod.REPORT_FORMAT


@pytest.mark.parametrize("argv,message", [
    (["--tcp", "nonsense"], "HOST:PORT"),
    (["--tcp", "host:port"], "HOST:PORT"),
    (["--tcp", "127.0.0.1:9", "--duration", "0"], "--duration must be positive"),
    (["--port", "/dev/x", "--tcp", "h:1"], "not allowed with"),
])
def test_cli_refuses_bad_arguments_with_one_stderr_line(capsys, argv, message):
    try:
        code = relay_mod.main(argv)
    except SystemExit as exc:               # argparse's own errors exit; ours return
        code = exc.code
    assert code == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and message in err and err.startswith("unicorn_relay: exit 2:")


def test_cli_without_a_port_explains_how_to_pair(monkeypatch, capsys):
    monkeypatch.setattr(relay_mod, "find_headset_ports", lambda: [])
    assert relay_mod.main([]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "pair the headset" in err
    monkeypatch.setattr(relay_mod, "find_headset_ports", lambda: ["/dev/cu.UN-A", "/dev/cu.UN-B"])
    assert relay_mod.main([]) == 2 and "several headsets" in capsys.readouterr().err


def test_cli_picks_the_single_paired_headset(monkeypatch):
    monkeypatch.setattr(relay_mod, "find_headset_ports", lambda: ["/dev/cu.UN-ONLY"])
    args = relay_mod.build_parser().parse_args([])
    source = relay_mod.resolve_source(args)
    assert isinstance(source, relay_mod.SerialSource) and source.port == "/dev/cu.UN-ONLY"


def test_list_ports(monkeypatch, capsys):
    monkeypatch.setattr(relay_mod, "find_headset_ports", lambda: ["/dev/cu.UN-1"])
    assert relay_mod.main(["--list-ports"]) == 0 and "/dev/cu.UN-1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the real recorder reads the relay's stream
# ---------------------------------------------------------------------------

def test_the_real_recorder_records_the_relays_stream_without_gaps(rig, tmp_path):
    r = rig(server={"n_samples": None, "burst": 4})
    csv = tmp_path / "rec.csv"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "collection" / "record_session.py"), "--out", str(csv),
         "--eeg-type", r.eeg_type, "--marker-type", f"MarkersNone_{uuid.uuid4().hex[:8]}",
         "--eeg-timeout", "20", "--duration", "5", "--skip-sanity-check"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60)
    r.finish()
    assert proc.returncode == 0, proc.stderr[-800:]
    integrity = json.loads((tmp_path / "rec.csv.integrity.json").read_text(encoding="utf-8"))
    text = csv.read_text(encoding="utf-8").splitlines()
    header, rows = text[0].split(","), [line.split(",") for line in text[1:]]
    assert header[1:9] == EEG_CHANNELS and len(rows) > 700
    fz = np.array([float(row[1]) for row in rows[:300]])
    assert 49.9 <= fz.min() and fz.max() < 90.1                    # known_signal's channel 0: 50 + (k % 40)
    assert integrity                                               # the recorder's exit summary was written
    stamps = np.array([float(row[0]) for row in rows])
    assert np.all(np.diff(stamps) > 0) and np.max(np.abs(np.diff(stamps) - 0.004)) < 2e-4
