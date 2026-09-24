"""
Unicorn LSL relay: reads the Unicorn Hybrid Black's Bluetooth serial stream and publishes it as
LSL, so this project runs on a Mac (or anywhere) without g.tec's Windows-only Unicorn LSL
Interface. It is a stand-in for that program, nothing more: the recorder, the inference service
and the panel find the stream by type, exactly as before.

    python collection/unicorn_relay.py                       # finds /dev/cu.UN-* (pair the headset first)
    python collection/unicorn_relay.py --port /dev/cu.UN-XXXXXX
    python collection/unicorn_relay.py --tcp 127.0.0.1:9000  # e.g. Bumble's rfcomm_bridge, or the fake headset
    python collection/unicorn_relay.py --duration 1800 --report logs/counter_gap_test.json
    python collection/unicorn_relay.py --list-ports

The Mac never uses a g.tec dongle directly: pair the headset in System Settings (built-in
Bluetooth) and it appears as /dev/cu.UN-<serial>. A USB dongle needs a user-space Bluetooth stack
such as Bumble, whose rfcomm_bridge exposes the link as a TCP socket for --tcp.

Streams (same contract as the Unicorn LSL Interface, pipeline/constants.py):
  UnicornEEG   type EEG   8 ch = EEG_CHANNELS, 250 Hz, float32, microvolts
  UnicornAux   type Aux   8 ch float64: battery %, accelerometer x y z (g), gyroscope x y z
               (deg/s), device sample counter. Separate on purpose, so the EEG stream stays 8 channels.
Channel LABELS are written by this program from EEG_CHANNELS: the headset sends no labels. The
service's electrode-label check therefore cannot catch a wrong cap here; check on a live stream that
channels 1-8 are Fz C3 Cz C4 Pz PO7 Oz PO8 (blink at Fz/PO8, alpha at Oz with eyes closed; TESTING.md H2).

Timestamps come from the headset's sample counter (collection/unicorn_protocol.CounterClock), not
from when Bluetooth delivered the bytes, so radio bursts do not smear the ERP. A lost packet is a
jump in the timestamps, which the recorder's gap detection already handles. The first --warmup-s of
data is held back to anchor the clock, so the stream appears about two seconds after the first packet.

Latency: this program adds microseconds of work per packet and hands each sample to LSL as it is
decoded. The delay that matters is the Bluetooth link's own, which is the same on any host; this
program measures how late packets arrive relative to the best ones (`delivery_delay_ms` in the
report). The constant part of the delay cannot be measured in software (TESTING.md H7).

Sample-counter gap test (Bluetooth link quality): run with --duration 1800 and read the report:
`samples.missing` and `loss_pct` come from the device's own counter, so they are true losses.

Synthetic data identifies itself: tools/fake_unicorn_bt.py marks its packets, and the relay then
names the streams MockUnicornEEG / MockUnicornAux and says so in the stream description.

Exit codes: 0 stopped (Ctrl+C, SIGTERM, --duration); 2 bad arguments or no headset port found;
3 could not connect within --connect-timeout; 4 the link was lost or went silent.
Every non-zero exit prints one line to stderr.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import unicorn_protocol as up  # noqa: E402
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ, EEG_STREAM_NAME, EEG_STREAM_TYPE  # noqa: E402

AUX_STREAM_NAME = "UnicornAux"
AUX_STREAM_TYPE = "Aux"
MOCK_PREFIX = "Mock"
AUX_CHANNELS = ["battery_pct", "acc_x_g", "acc_y_g", "acc_z_g", "gyr_x_dps", "gyr_y_dps", "gyr_z_dps", "counter"]
REPORT_FORMAT = "p300-unicorn-relay-report-v1"
PORT_GLOB = "/dev/cu.UN-*"

EXIT_OK, EXIT_ARGS, EXIT_NO_CONNECT, EXIT_LINK_LOST = 0, 2, 3, 4
DELAY_BIN_S = 0.00025
DELAY_BINS = 4000                      # 0 .. 1 s in 0.25 ms bins; the last bin catches anything later
DELAY_SETTLE_S = 1.0                   # creating the outlets stalls the read loop ~60 ms once; not the link's fault
HEALTH_EVERY_S = 10.0
NO_DATA_AFTER_START_S = 5.0


class LinkLost(Exception):
    """The Bluetooth link closed or failed."""


# ---------------------------------------------------------------------------
# byte sources
# ---------------------------------------------------------------------------

class SerialSource:
    """The headset's Bluetooth serial port, e.g. /dev/cu.UN-<serial> on macOS."""

    def __init__(self, port: str):
        self.port = port
        self._ser = None

    def describe(self) -> str:
        return f"serial:{self.port}"

    def open(self) -> None:
        import serial
        # exclusive: a second program reading the same port would steal bytes and look like packet loss
        self._ser = serial.Serial(self.port, baudrate=115200, timeout=0.02, write_timeout=2.0, exclusive=True)

    def read(self) -> bytes:
        import serial
        try:
            return self._ser.read(max(1, self._ser.in_waiting))     # returns as soon as bytes arrive
        except (serial.SerialException, OSError) as exc:
            raise LinkLost(str(exc)) from exc

    def write(self, data: bytes) -> None:
        import serial
        try:
            self._ser.write(data)       # no flush(): tcdrain has no timeout and can hang on a stalled link
        except (serial.SerialException, OSError) as exc:
            raise LinkLost(str(exc)) from exc

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass


class TcpSource:
    """A TCP socket carrying the same bytes: Bumble's rfcomm_bridge, or tools/fake_unicorn_bt.py."""

    def __init__(self, host: str, port: int, connect_timeout: float = 5.0):
        self.host, self.port, self.connect_timeout = host, port, connect_timeout
        self._sock = None

    def describe(self) -> str:
        return f"tcp:{self.host}:{self.port}"

    def open(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(0.02)

    def read(self) -> bytes:
        try:
            data = self._sock.recv(4096)
        except socket.timeout:
            return b""
        except OSError as exc:
            raise LinkLost(str(exc)) from exc
        if not data:
            raise LinkLost("the connection was closed by the other end")
        return data

    def write(self, data: bytes) -> None:
        try:
            self._sock.sendall(data)
        except OSError as exc:
            raise LinkLost(str(exc)) from exc

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass


def find_headset_ports() -> list[str]:
    return sorted(glob.glob(PORT_GLOB))


# ---------------------------------------------------------------------------
# LSL outlets
# ---------------------------------------------------------------------------

def _describe(info: StreamInfo, *, labels, unit, kind, synthetic: bool, source: str) -> None:
    desc = info.desc()
    desc.append_child_value("manufacturer", "g.tec")
    desc.append_child_value("model", "Unicorn Hybrid Black")
    desc.append_child_value("relay", "collection/unicorn_relay.py")
    desc.append_child_value("source", source)
    desc.append_child_value("timestamps", "device sample counter mapped to the LSL clock")
    desc.append_child_value("synthetic", "true" if synthetic else "false")
    channels = desc.append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
        channel.append_child_value("type", kind)


def make_outlets(*, eeg_name, eeg_type, aux_name, aux_type, publish_aux, synthetic, source, uid):
    """(eeg_outlet, aux_outlet or None). Names gain a Mock prefix for synthetic data unless given."""
    eeg = StreamInfo(eeg_name, eeg_type, len(EEG_CHANNELS), EEG_SFREQ, "float32", f"{uid}-eeg")
    _describe(eeg, labels=EEG_CHANNELS, unit="microvolts", kind="EEG", synthetic=synthetic, source=source)
    aux_outlet = None
    if publish_aux:
        aux = StreamInfo(aux_name, aux_type, len(AUX_CHANNELS), EEG_SFREQ, "double64", f"{uid}-aux")
        _describe(aux, labels=AUX_CHANNELS, unit="mixed", kind="Aux", synthetic=synthetic, source=source)
        aux_outlet = StreamOutlet(aux)
    return StreamOutlet(eeg), aux_outlet


# ---------------------------------------------------------------------------
# the relay
# ---------------------------------------------------------------------------

class DelayHistogram:
    def __init__(self):
        self.bins = np.zeros(DELAY_BINS, dtype=np.int64)
        self.n = 0
        self.max_s = 0.0

    def add(self, delay_s: float) -> None:
        self.bins[min(DELAY_BINS - 1, max(0, int(delay_s / DELAY_BIN_S)))] += 1
        self.n += 1
        self.max_s = max(self.max_s, delay_s)

    def percentile_ms(self, q: float) -> float | None:
        if not self.n:
            return None
        idx = int(np.searchsorted(np.cumsum(self.bins), q / 100.0 * self.n))
        return (min(idx, DELAY_BINS - 1) + 1) * DELAY_BIN_S * 1000.0


class UnicornRelay:
    def __init__(self, source, *, stream_name=None, stream_type=EEG_STREAM_TYPE, aux_name=None,
                 aux_type=AUX_STREAM_TYPE, publish_aux=True, warmup_s=2.0, window_s=15.0,
                 duration_s=None, connect_timeout_s=30.0, data_timeout_s=3.0, report_path=None,
                 log=print):
        self.source = source
        self.stream_name, self.stream_type = stream_name, stream_type
        self.aux_name, self.aux_type, self.publish_aux = aux_name, aux_type, publish_aux
        self.warmup_s, self.window_s = warmup_s, window_s
        self.duration_s, self.connect_timeout_s, self.data_timeout_s = duration_s, connect_timeout_s, data_timeout_s
        self.report_path = report_path
        self.log = log
        self.failure_message: str | None = None

        self.framer = up.PacketFramer()
        self.tracker = up.CounterTracker()
        self.clock = up.CounterClock(warmup_s=warmup_s, window_s=window_s)
        self.delays = DelayHistogram()
        self.eeg_outlet = self.aux_outlet = None
        self.published_names: tuple = ()
        self.synthetic = False
        self.published = 0
        self.bad_payloads = 0
        self.clamped = 0
        self.clock_restarts = 0
        self._last_ts = -np.inf
        self._first_arrival = self._last_arrival = None
        self._first_counter = self._last_counter = None
        self._started_utc = datetime.now(timezone.utc)
        self._exit = (EXIT_OK, "stopped")

    # -- connection -------------------------------------------------------

    def _connect(self, stop: threading.Event) -> bool:
        deadline = time.monotonic() + self.connect_timeout_s
        last_error = None
        while not stop.is_set():
            try:
                self.source.open()
                return True
            except Exception as exc:          # OSError, pyserial's SerialException, ...: all mean "not yet"
                last_error = exc
            if time.monotonic() >= deadline:
                break
            stop.wait(1.0)
        if not stop.is_set():
            self.failure_message = (f"could not connect to {self.source.describe()} within "
                                    f"{self.connect_timeout_s:g} s ({last_error})")
        return False

    # -- publishing -------------------------------------------------------

    def _ensure_outlets(self, first_sample: up.Sample) -> None:
        self.synthetic = first_sample.is_synthetic
        eeg_name = self.stream_name or ((MOCK_PREFIX if self.synthetic else "") + EEG_STREAM_NAME)
        aux_name = self.aux_name or ((MOCK_PREFIX if self.synthetic else "") + AUX_STREAM_NAME)
        uid = f"unicorn-relay-{os.getpid()}-{int(time.time())}"
        self.eeg_outlet, self.aux_outlet = make_outlets(
            eeg_name=eeg_name, eeg_type=self.stream_type, aux_name=aux_name, aux_type=self.aux_type,
            publish_aux=self.publish_aux, synthetic=self.synthetic, source=self.source.describe(), uid=uid)
        self.published_names = (eeg_name, aux_name if self.publish_aux else None)
        banner = " SYNTHETIC DATA: the headset stand-in marked its packets." if self.synthetic else ""
        self.log(f"publishing {eeg_name!r} (type {self.stream_type}, {len(EEG_CHANNELS)} ch, {EEG_SFREQ:g} Hz, uV)"
                 + (f" and {aux_name!r}" if self.publish_aux else "") + "." + banner)

    def _publish(self, sample: up.Sample, ts: float) -> None:
        if ts <= self._last_ts:
            ts = self._last_ts + 1e-6          # never hand LSL a non-increasing stamp; counted, not hidden
            self.clamped += 1
        self._last_ts = ts
        if self.eeg_outlet is None:
            self._ensure_outlets(sample)
        self.eeg_outlet.push_sample(list(sample.eeg_uv), ts)
        if self.aux_outlet is not None:
            self.aux_outlet.push_sample(
                [sample.battery_pct, *sample.acc_g, *sample.gyr_dps, float(sample.counter)], ts)
        self.published += 1

    def _handle(self, sample: up.Sample, arrival: float) -> None:
        step = self.tracker.update(sample.counter)
        if step.kind == "duplicate":
            return
        if step.kind == "reset":
            # the counter went backwards: a restarted device or a corrupt frame. Anchor a fresh clock;
            # the new stamps come from arrival times, which are after everything already published.
            self.clock_restarts += 1
            self.clock = up.CounterClock(warmup_s=self.warmup_s, window_s=self.window_s)
            self.log(f"WARNING: sample counter went backwards to {sample.counter}; re-anchoring the clock")
        if step.kind == "gap":
            self.log(f"WARNING: {step.missing} sample(s) lost before counter {sample.counter} "
                     f"({step.missing * 1000 / EEG_SFREQ:.0f} ms)")
        if self._first_counter is None:
            self._first_counter, self._first_arrival = sample.counter, arrival
        self._last_counter, self._last_arrival = sample.counter, arrival
        for released, ts in self.clock.push(sample, sample.counter, arrival):
            self._publish(released, ts)
        if self.clock.ready and sample.counter - self._first_counter >= (self.warmup_s + DELAY_SETTLE_S) * EEG_SFREQ:
            self.delays.add(max(0.0, self.clock.delay_s(sample.counter, arrival)))

    # -- main loop --------------------------------------------------------

    def run(self, stop: threading.Event) -> int:
        try:
            if not self._connect(stop):
                if self.failure_message:
                    self._exit = (EXIT_NO_CONNECT, self.failure_message)
                return self._exit[0]
            self.log(f"connected to {self.source.describe()}; starting acquisition")
            self.source.write(up.START_ACQUISITION)
            t_start = time.monotonic()
            last_data = t_start
            last_health = t_start
            t_first_publish = None
            while not stop.is_set():
                data = self.source.read()
                arrival = local_clock()
                now = time.monotonic()
                if data:
                    last_data = now
                    for payload in self.framer.feed(data):
                        try:
                            sample = up.decode_payload(payload)
                        except up.PayloadError:
                            self.bad_payloads += 1
                            continue
                        self._handle(sample, arrival)
                    if t_first_publish is None and self.published:
                        t_first_publish = now
                else:
                    limit = self.data_timeout_s if self.published or self.framer.frames else NO_DATA_AFTER_START_S
                    if now - last_data > limit:
                        raise LinkLost(f"no data from the headset for {now - last_data:.1f} s")
                if now - last_health >= HEALTH_EVERY_S:
                    last_health = now
                    self._health()
                if (self.duration_s is not None and t_first_publish is not None
                        and now - t_first_publish >= self.duration_s):
                    self._exit = (EXIT_OK, f"--duration {self.duration_s:g} s elapsed")
                    break
            try:
                self.source.write(up.STOP_ACQUISITION)
            except LinkLost:
                pass
        except LinkLost as exc:
            self.failure_message = f"the Bluetooth link was lost: {exc}. Restart the relay."
            self._exit = (EXIT_LINK_LOST, self.failure_message)
        finally:
            self._finish()
        return self._exit[0]

    def _health(self) -> None:
        d99 = self.delays.percentile_ms(99)
        self.log(f"{self.tracker.n_samples} packets, {self.tracker.missing} lost, "
                 f"{self.framer.resyncs} resyncs, delivery delay p99 "
                 f"{'n/a' if d99 is None else f'{d99:.1f} ms'}")

    # -- report -----------------------------------------------------------

    def report(self) -> dict:
        t, fr = self.tracker, self.framer
        received = t.n_samples
        device_s = ((self._last_counter - self._first_counter + 1) / EEG_SFREQ
                    if self._first_counter is not None else 0.0)
        wall_s = (self._last_arrival - self._first_arrival) if self._first_arrival is not None else 0.0
        drift_ppm = None
        if self.clock.ready and device_s > 30:
            drift_ppm = (self.clock.offset - self.clock.initial_offset) / device_s * 1e6
        code, reason = self._exit
        return {
            "format": REPORT_FORMAT,
            "started_utc": self._started_utc.isoformat(timespec="seconds"),
            "ended_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": self.source.describe(),
            "stream": {"names": list(self.published_names), "type": self.stream_type,
                       "synthetic": self.synthetic},
            "duration": {"device_s": round(device_s, 3), "wall_s": round(wall_s, 3)},
            "samples": {
                "received": received, "published": self.published, "missing": t.missing,
                "loss_pct": round(100.0 * t.missing / (received + t.missing), 4) if received else None,
                "n_gaps": t.n_gaps, "longest_gap_samples": t.longest_gap,
                "longest_gap_ms": round(t.longest_gap * 1000 / EEG_SFREQ, 1),
                "duplicates": t.duplicates, "counter_resets": t.resets, "clock_restarts": self.clock_restarts,
                "timestamps_clamped": self.clamped,
            },
            "gaps": [{"counter": c, "missing": n, "ms": round(n * 1000 / EEG_SFREQ, 1)} for c, n in t.gaps],
            "framing": {"frames": fr.frames, "bytes_discarded": fr.bytes_discarded, "resyncs": fr.resyncs,
                        "bad_payloads": self.bad_payloads, "ack_received": fr.ack_received},
            "delivery_delay_ms": {
                "note": "how late packets arrived relative to the best ones seen; not the absolute link delay",
                "p50": self.delays.percentile_ms(50), "p95": self.delays.percentile_ms(95),
                "p99": self.delays.percentile_ms(99), "max": round(self.delays.max_s * 1000, 2)},
            "clock": {"offset_s": self.clock.offset, "initial_offset_s": self.clock.initial_offset,
                      "device_vs_host_drift_ppm": None if drift_ppm is None else round(drift_ppm, 1),
                      "warmup_s": self.warmup_s, "window_s": self.window_s},
            "exit": {"code": code, "reason": reason},
        }

    def _finish(self) -> None:
        self.source.close()
        rep = self.report()
        s = rep["samples"]
        if s["received"]:
            self.log(f"done: {s['received']} packets over {rep['duration']['device_s']:.1f} s of device time; "
                     f"{s['missing']} lost ({s['loss_pct']}%), {s['n_gaps']} gap(s), longest "
                     f"{s['longest_gap_ms']} ms; delivery delay p99 {rep['delivery_delay_ms']['p99']} ms")
        if self.report_path is not None:
            try:
                Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
                Path(self.report_path).write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
                self.log(f"report written to {self.report_path}")
            except OSError as exc:
                self.log(f"WARNING: could not write the report: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class _OneLineParser(argparse.ArgumentParser):
    def error(self, message):
        print(f"unicorn_relay: exit {EXIT_ARGS}: {message}", file=sys.stderr)
        raise SystemExit(EXIT_ARGS)


def build_parser() -> argparse.ArgumentParser:
    p = _OneLineParser(prog="unicorn_relay.py", description="Unicorn Bluetooth serial -> LSL.")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--port", help="headset serial port (default: the one /dev/cu.UN-* device)")
    where.add_argument("--tcp", metavar="HOST:PORT", help="read the same bytes from a TCP socket")
    where.add_argument("--list-ports", action="store_true", help="show candidate serial ports and exit")
    p.add_argument("--stream-name", default=None, help=f"default {EEG_STREAM_NAME} (Mock-prefixed for synthetic data)")
    p.add_argument("--stream-type", default=EEG_STREAM_TYPE)
    p.add_argument("--aux-name", default=None)
    p.add_argument("--aux-type", default=AUX_STREAM_TYPE)
    p.add_argument("--no-aux", action="store_true", help="do not publish the accelerometer/gyroscope/battery stream")
    p.add_argument("--duration", type=float, default=None, help="stop after this many seconds of data")
    p.add_argument("--report", default=None, help="JSON report path (default logs/unicorn_relay_<time>.report.json)")
    p.add_argument("--connect-timeout", type=float, default=30.0)
    p.add_argument("--warmup-s", type=float, default=2.0, help="seconds of data held back to anchor the clock")
    p.add_argument("--window-s", type=float, default=15.0, help="clock offset estimation window")
    return p


class ArgError(Exception):
    """The command line names no usable headset source."""


def resolve_source(args) -> "SerialSource | TcpSource":
    if args.tcp:
        host, sep, port = args.tcp.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ArgError(f"--tcp must be HOST:PORT, got {args.tcp!r}")
        return TcpSource(host, int(port))
    if args.port:
        return SerialSource(args.port)
    found = find_headset_ports()
    if len(found) == 1:
        return SerialSource(found[0])
    if not found:
        raise ArgError(f"no {PORT_GLOB} device found: pair the headset in System Settings > Bluetooth "
                       "first (hold its button until the LED blinks), or pass --port / --tcp")
    raise ArgError(f"several headsets found ({', '.join(found)}); pass --port")


def _fail(message: str) -> int:
    print(f"unicorn_relay: exit {EXIT_ARGS}: {message}", file=sys.stderr)
    return EXIT_ARGS


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_ports:
        ports = find_headset_ports()
        print("\n".join(ports) if ports else f"no {PORT_GLOB} device found")
        return EXIT_OK
    if args.duration is not None and args.duration <= 0:
        return _fail("--duration must be positive")
    try:
        source = resolve_source(args)
    except ArgError as exc:
        return _fail(str(exc))
    report = args.report or f"logs/unicorn_relay_{datetime.now():%Y%m%d_%H%M%S}.report.json"
    relay = UnicornRelay(
        source, stream_name=args.stream_name, stream_type=args.stream_type, aux_name=args.aux_name,
        aux_type=args.aux_type, publish_aux=not args.no_aux, warmup_s=args.warmup_s, window_s=args.window_s,
        duration_s=args.duration, connect_timeout_s=args.connect_timeout, report_path=report,
        log=lambda m: print(f"[unicorn_relay] {m}", flush=True))
    stop = threading.Event()
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda signum, frame: stop.set())
    try:
        code = relay.run(stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if code != EXIT_OK:
        print(f"unicorn_relay: exit {code}: {relay.failure_message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
