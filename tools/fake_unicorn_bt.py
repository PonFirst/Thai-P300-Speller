"""
A stand-in for the Unicorn Hybrid Black's Bluetooth serial link, over TCP: it waits for the real
"start acquisition" command, acknowledges it, and streams real-format 45-byte payloads at 250 Hz
(collection/unicorn_protocol.py). It exists to test collection/unicorn_relay.py end to end, and to
demonstrate the Mac path with no headset:

    python tools/fake_unicorn_bt.py --port 9000
    python collection/unicorn_relay.py --tcp 127.0.0.1:9000

THE DATA IS FAKE. Every payload carries a marker in the gyroscope bytes
(unicorn_protocol.SYNTHETIC_GYRO_COUNTS); the relay sees it and names its streams MockUnicornEEG /
MockUnicornAux, so this can never be recorded as a real UnicornEEG stream.

It can misbehave on request, the way a real link does: drop packets (the counter still advances, so
the receiver sees a gap), deliver them in bursts, insert garbage bytes, and end the link.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "collection"))
import unicorn_protocol as up  # noqa: E402


def noise_signal(seed: int = 0, sd_uv: float = 8.0):
    rng = np.random.default_rng(seed)
    return lambda k: rng.normal(0.0, sd_uv, size=up.N_EEG)


class FakeUnicorn:
    """
    One-connection TCP server speaking the Unicorn protocol.

    signal(k) -> 8 values in uV for counter k. drop: counters whose packets never arrive. garbage:
    {counter: bytes} inserted before that packet. burst: packets delivered together in groups of this
    size, at the moment the last one is due (the shape of Bluetooth delivery). n_samples: end the
    stream after this many counters (None: until told to stop). close_after: hang up abruptly.
    """

    def __init__(self, host="127.0.0.1", port=0, *, signal=None, rate_hz=up.UNICORN_SFREQ, n_samples=None,
                 drop=(), garbage=None, burst=1, synthetic=True, counter_start=0, send_ack=True,
                 close_after=None, silent_after=None, reconnect=False):
        self.signal = signal or noise_signal()
        self.rate_hz, self.n_samples = rate_hz, n_samples
        self.drop = drop if callable(drop) else set(drop).__contains__
        self.garbage, self.burst = dict(garbage or {}), max(1, burst)
        self.synthetic, self.counter_start, self.send_ack = synthetic, counter_start, send_ack
        self.close_after, self.silent_after = close_after, silent_after
        # reconnect: serve connection after connection (BCI2000's UnicornSerialSource reconnects on every
        # Set config, as it would reopen the real headset's port); otherwise one connection only.
        self.reconnect = reconnect
        self.commands: list[bytes] = []
        self.sent = 0
        self._server = socket.socket()
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self.host, self.port = self._server.getsockname()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> "FakeUnicorn":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        if self._thread.ident is not None:                 # never started: nothing to join
            self._thread.join(timeout=3)

    def _serve(self) -> None:
        self._serve_one()
        while self.reconnect and not self._stop.is_set():
            self._serve_one()

    def _serve_one(self) -> None:
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(0.005)
        try:
            self._wait_for(conn, up.START_ACQUISITION)
            if self.send_ack:
                conn.sendall(up.ACK)
            self._stream(conn)
        except (OSError, EOFError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _wait_for(self, conn, command: bytes) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = conn.recv(64)
            except socket.timeout:
                continue
            if not chunk:
                raise EOFError
            buf += chunk
            if command in buf:
                self.commands.append(command)
                return

    def _stop_requested(self, conn) -> bool:
        try:
            data = conn.recv(64)
        except socket.timeout:
            return False
        if not data:
            return True
        if up.STOP_ACQUISITION in data:
            self.commands.append(up.STOP_ACQUISITION)
            return True
        return False

    def _stream(self, conn) -> None:
        t0 = time.perf_counter()
        pending = bytearray()
        gyr = up.SYNTHETIC_GYRO_COUNTS if self.synthetic else (3, -5, 8)
        k = 0
        while not self._stop.is_set() and (self.n_samples is None or k < self.n_samples):
            due = t0 + k / self.rate_hz
            while time.perf_counter() < due:
                if self._stop_requested(conn):
                    conn.sendall(up.ACK)
                    return
                time.sleep(0.0005)
            counter = self.counter_start + k
            if self.close_after is not None and k >= self.close_after:
                return
            if self.silent_after is not None and k >= self.silent_after:
                time.sleep(0.05)
                k += 1
                continue
            if k in self.garbage:
                pending += self.garbage[k]
            if not self.drop(k):
                counts = [up.uv_to_counts(v) for v in self.signal(k)]
                pending += up.encode_payload(counts, counter, battery_nibble=12, gyr_counts=gyr)
                self.sent += 1
            k += 1
            if k % self.burst == 0 or (self.n_samples is not None and k >= self.n_samples):
                if pending:
                    conn.sendall(bytes(pending))
                    pending.clear()
        if pending:
            conn.sendall(bytes(pending))
        # a finished stream stays open (like a headset that is still on) until the peer stops it
        while not self._stop.is_set():
            if self._stop_requested(conn):
                conn.sendall(up.ACK)
                return
            time.sleep(0.01)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fake Unicorn Bluetooth link over TCP (SYNTHETIC data).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--burst", type=int, default=4, help="deliver packets in groups of this many (radio-like)")
    p.add_argument("--drop-every", type=int, default=0, help="drop one packet in every N (0: none)")
    args = p.parse_args(argv)
    every = args.drop_every
    server = FakeUnicorn(args.host, args.port, burst=args.burst, reconnect=True,
                         drop=(lambda k: k % every == every - 1) if every > 0 else ()).start()
    print("=" * 72)
    print(f"FAKE UNICORN on {server.host}:{server.port} -- SYNTHETIC DATA, not a brain.")
    print(f"  python collection/unicorn_relay.py --tcp {server.host}:{server.port}")
    print("=" * 72, flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
