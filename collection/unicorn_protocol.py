"""
Unicorn Hybrid Black Bluetooth serial protocol: the pure logic, no I/O, no LSL.

Source: g.tec "Unicorn Bluetooth Protocol" (UnicornBluetoothProtocol.pdf, manual v1.18.00,
github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black, folder "Unicorn Bluetooth Protocol"). Every
constant below is from that document, and tests/test_unicorn_protocol.py checks the decoder
against the document's own worked example (payload -> 3654.87 uV on CH1, counter 176, ...).

  Commands (3 bytes):  start acquisition 61 7C 87, stop acquisition 63 5C C5; each is
                       acknowledged with 00 00 00 (the stop ack follows the last payload).
  Payload, 45 bytes, 250 Hz once started:
      0..1    header C0 00
      2       battery: low nibble v, percent = 100 * v / 15
      3..26   8 EEG channels, 3 bytes each, BIG-endian two's complement;
              uV = value * 4500000 / 50331642
      27..32  accelerometer X Y Z, int16 LITTLE-endian, g = value / 4096
      33..38  gyroscope X Y Z, int16 LITTLE-endian, deg/s = value / 32.8
      39..42  sample counter, int32 LITTLE-endian
      43..44  footer 0D 0A

Two traps in the document, both handled here: its counter formula prints Counter[1] three times
(the C example beside it, and this code, use bytes 0..3), and the EEG bytes are big-endian while
everything after them is little-endian.

Also here: a stream framer that finds payloads in an arbitrary byte stream and resynchronises
after garbage; a counter tracker that turns the device's sample counter into gap statistics (the
"sample-counter gap test" for Bluetooth link quality); and a clock that stamps samples from the
counter instead of from when Bluetooth happened to deliver them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

UNICORN_SFREQ = 250.0
N_EEG = 8

START_ACQUISITION = bytes([0x61, 0x7C, 0x87])
STOP_ACQUISITION = bytes([0x63, 0x5C, 0xC5])
ACK = bytes([0x00, 0x00, 0x00])

HEADER = bytes([0xC0, 0x00])
FOOTER = bytes([0x0D, 0x0A])
PAYLOAD_LEN = 45

_BATTERY = 2
_EEG = 3
_ACC = 27
_GYR = 33
_COUNTER = 39
_FOOTER = 43

EEG_UV_PER_COUNT = 4_500_000.0 / 50_331_642.0
ACC_G_PER_COUNT = 1.0 / 4096.0
GYR_DPS_PER_COUNT = 1.0 / 32.8

# A real gyroscope does not sit at the same raw value on all three axes. tools/fake_unicorn_bt.py
# writes this pattern so the relay can tell synthetic data from real and name the stream honestly
# (CLAUDE.md: synthetic data must identify itself).
SYNTHETIC_GYRO_COUNTS = (0x5A5A, 0x5A5A, 0x5A5A)


class PayloadError(ValueError):
    """A byte string is not a valid Unicorn payload."""


@dataclass(frozen=True)
class Sample:
    battery_pct: float
    eeg_uv: tuple
    acc_g: tuple
    gyr_dps: tuple
    counter: int
    gyr_counts: tuple

    @property
    def is_synthetic(self) -> bool:
        return self.gyr_counts == SYNTHETIC_GYRO_COUNTS


def decode_payload(payload: bytes) -> Sample:
    if len(payload) != PAYLOAD_LEN:
        raise PayloadError(f"payload must be {PAYLOAD_LEN} bytes, got {len(payload)}")
    if payload[:2] != HEADER:
        raise PayloadError(f"bad header {payload[:2].hex()}")
    if payload[_FOOTER:] != FOOTER:
        raise PayloadError(f"bad footer {payload[_FOOTER:].hex()}")
    eeg = tuple(
        int.from_bytes(payload[_EEG + 3 * i:_EEG + 3 * i + 3], "big", signed=True) * EEG_UV_PER_COUNT
        for i in range(N_EEG))
    acc = tuple(int.from_bytes(payload[_ACC + 2 * i:_ACC + 2 * i + 2], "little", signed=True)
                for i in range(3))
    gyr = tuple(int.from_bytes(payload[_GYR + 2 * i:_GYR + 2 * i + 2], "little", signed=True)
                for i in range(3))
    return Sample(
        battery_pct=100.0 * (payload[_BATTERY] & 0x0F) / 15.0,
        eeg_uv=eeg,
        acc_g=tuple(v * ACC_G_PER_COUNT for v in acc),
        gyr_dps=tuple(v * GYR_DPS_PER_COUNT for v in gyr),
        counter=int.from_bytes(payload[_COUNTER:_COUNTER + 4], "little", signed=True),
        gyr_counts=gyr)


def encode_payload(eeg_counts, counter: int, *, battery_nibble: int = 0x0F,
                   acc_counts=(0, 0, 0), gyr_counts=(0, 0, 0)) -> bytes:
    """The inverse of decode_payload at the raw-integer level (for the fake headset and tests)."""
    if len(eeg_counts) != N_EEG:
        raise ValueError(f"need {N_EEG} EEG values, got {len(eeg_counts)}")
    out = bytearray(HEADER)
    out.append(battery_nibble & 0x0F)
    for v in eeg_counts:
        if not -(1 << 23) <= v < (1 << 23):
            raise ValueError(f"EEG count {v} does not fit 24 bits")
        out += int(v).to_bytes(3, "big", signed=True)
    for v in (*acc_counts, *gyr_counts):
        out += int(v).to_bytes(2, "little", signed=True)
    out += int(counter).to_bytes(4, "little", signed=True)
    out += FOOTER
    assert len(out) == PAYLOAD_LEN
    return bytes(out)


def uv_to_counts(uv: float) -> int:
    return int(round(uv / EEG_UV_PER_COUNT))


class PacketFramer:
    """
    Finds payloads in a byte stream: any chunking, garbage between frames, and the 3-byte
    start-acknowledge in front of the first one. A candidate header is accepted only if the
    footer sits 43 bytes later, so EEG data that happens to contain C0 00 is not mistaken for a
    frame. Skipped bytes are counted (`bytes_discarded`, `resyncs`), never silently dropped.
    """

    def __init__(self, expect_ack: bool = True):
        self._buf = bytearray()
        self._awaiting_ack = expect_ack
        self._in_garbage = False
        self.ack_received = False
        self.frames = 0
        self.bytes_discarded = 0
        self.resyncs = 0          # separate runs of discarded bytes

    def feed(self, data: bytes) -> list[bytes]:
        self._buf += data
        buf = self._buf
        if self._awaiting_ack:
            if not buf:
                return []
            if buf[0] != 0x00 or len(buf) >= len(ACK):
                if bytes(buf[:len(ACK)]) == ACK:
                    del buf[:len(ACK)]
                    self.ack_received = True
                self._awaiting_ack = False
            else:
                return []                       # 1-2 zero bytes: wait for the rest of the ack
        frames, i = [], 0
        while True:
            j = buf.find(HEADER, i)
            if j < 0:
                keep = 1 if buf.endswith(HEADER[:1]) and len(buf) > i else 0    # a split header
                self._discard(len(buf) - keep - i)
                i = len(buf) - keep
                break
            if j > i:
                self._discard(j - i)
                i = j
            if len(buf) - j < PAYLOAD_LEN:
                break                           # a frame is on its way
            if buf[j + _FOOTER:j + PAYLOAD_LEN] == FOOTER:
                frames.append(bytes(buf[j:j + PAYLOAD_LEN]))
                self.frames += 1
                self._in_garbage = False
                i = j + PAYLOAD_LEN
            else:
                self._discard(1)                # a header-looking pair inside data
                i = j + 1
        del buf[:i]
        return frames

    def _discard(self, n: int) -> None:
        if n <= 0:
            return
        self.bytes_discarded += n
        if not self._in_garbage:
            self.resyncs += 1
            self._in_garbage = True


@dataclass
class CounterStep:
    kind: str          # first | ok | gap | duplicate | reset
    missing: int = 0   # samples lost before this one (kind == gap)


@dataclass
class CounterTracker:
    """
    The device numbers every sample, so a Bluetooth drop shows up as a jump in the counter no
    matter when the bytes arrive. This is the ground truth for link quality.
    """
    last: int | None = None
    n_samples: int = 0
    missing: int = 0
    duplicates: int = 0
    resets: int = 0
    gaps: list = field(default_factory=list)      # (counter, n_missing), first MAX_GAPS_KEPT
    longest_gap: int = 0
    n_gaps: int = 0
    MAX_GAPS_KEPT = 500

    def update(self, counter: int) -> CounterStep:
        self.n_samples += 1
        if self.last is None:
            self.last = counter
            return CounterStep("first")
        step = counter - self.last
        self.last = counter
        if step == 1:
            return CounterStep("ok")
        if step == 0:
            self.duplicates += 1
            return CounterStep("duplicate")
        if step < 0:
            self.resets += 1               # the device restarted its counter, or the frame was corrupt
            return CounterStep("reset")
        lost = step - 1
        self.missing += lost
        self.n_gaps += 1
        self.longest_gap = max(self.longest_gap, lost)
        if len(self.gaps) < self.MAX_GAPS_KEPT:
            self.gaps.append((counter, lost))
        return CounterStep("gap", lost)


class CounterClock:
    """
    Stamps samples from the device's sample counter instead of from Bluetooth arrival.

    Arrival times are bursty (the radio delivers a few packets at once), so stamping on arrival
    smears every ERP by the burst size. The counter is a perfect 4 ms grid, so
        timestamp = counter / fs + offset
    where `offset` maps the device's grid onto LSL time. Bluetooth only ever adds delay, so
    arrival - counter/fs is at least the true offset, and its minimum over a window estimates
    it. The first `warmup_s` of samples are held back to fix the offset before anything is
    published; afterwards the offset follows the window minimum but is slew-limited to
    `max_slew_ppm`, which lets it track the device's crystal drifting against the host clock while
    guaranteeing timestamps stay strictly increasing (a step of 4 ms can never be undone by an
    offset change of well under 1 us).

    What this cannot remove is the constant part of the delay (radio, headset firmware): it shifts
    every EEG timestamp late by the same amount in calibration and in live use, so the classifier
    sees a consistent window. Measure it with a photodiode (TESTING.md H7); nothing here can.
    """

    def __init__(self, fs: float = UNICORN_SFREQ, warmup_s: float = 2.0, window_s: float = 15.0,
                 max_slew_ppm: float = 200.0):
        self.fs = fs
        self.warmup_s = warmup_s
        self.window_s = window_s
        self.max_slew = max_slew_ppm * 1e-6
        self._held: list = []
        self._first_counter: int | None = None
        self._min_window: deque = deque()         # (counter, raw), raw strictly increasing
        self.offset: float | None = None
        self.initial_offset: float | None = None
        self._last_counter: int | None = None

    @property
    def ready(self) -> bool:
        return self.offset is not None

    def push(self, item, counter: int, arrival: float) -> list:
        """Returns [(item, timestamp), ...] that are ready to publish, in order."""
        raw = arrival - counter / self.fs
        self._observe(counter, raw)
        if self.offset is None:
            if self._first_counter is None:
                self._first_counter = counter
            self._held.append((item, counter))
            if (counter - self._first_counter) / self.fs < self.warmup_s:
                return []
            self.offset = self.initial_offset = self._min_window[0][1]
            self._last_counter = counter
            released = [(it, c / self.fs + self.offset) for it, c in self._held]
            self._held = []
            return released
        dt = max(0, counter - self._last_counter) / self.fs
        self._last_counter = counter
        target = self._min_window[0][1]
        limit = self.max_slew * dt
        self.offset += max(-limit, min(limit, target - self.offset))
        return [(item, counter / self.fs + self.offset)]

    def _observe(self, counter: int, raw: float) -> None:
        window = self._min_window
        while window and window[-1][1] >= raw:
            window.pop()
        window.append((counter, raw))
        horizon = counter - self.window_s * self.fs
        while window[0][0] < horizon:
            window.popleft()

    def delay_s(self, counter: int, arrival: float) -> float:
        """How late this packet arrived relative to the best case the clock has seen."""
        return arrival - (counter / self.fs + self.offset)
