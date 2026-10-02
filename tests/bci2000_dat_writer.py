"""
A small BCI2000 .dat (format 1.1) writer for tests, written from the file format description
(BCI2000's BCI2000OutputFormat.cpp, State.cpp, Param.cpp, EncodedString.cpp), deliberately NOT
from pipeline/bci2000_dat.py, so the reader is checked against an independent implementation.

  line 1   "BCI2000V= 1.1 HeaderLen= N SourceCh= C StatevectorLen= S DataFormat= fmt"
  header   "[ State Vector Definition ] " + "Name Length Value ByteLocation BitLocation" lines,
           "[ Parameter Definition ] " + parameter lines, then an empty line; N bytes in total
  data     per sample: C little-endian values, then S state-vector bytes; a state's bits start
           at bit ByteLocation*8 + BitLocation of the vector, least significant bit first.

States are packed bit by bit (np.packbits, little bit order), not with shifted words.

Also builds CheckerboardSpeller-shaped recordings (the states bci2000/CheckerboardSpeller and
bci2000/UnicornSerialSource write) for the recording and fit tests.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DTYPES = {"int16": "<i2", "int32": "<i4", "float32": "<f4"}
EEG = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")   # written out, not imported: what the file says
SFREQ = 250


# ---------------------------------------------------------------------------
# values and parameter lines
# ---------------------------------------------------------------------------

def encode(value: str, *, upper: bool = False) -> str:
    """BCI2000 EncodedString: '%' alone for empty; printable non-space ASCII as is (except '%',
    '{', '}'); every other UTF-8 byte as %xx. BCI2000 itself writes lowercase hex."""
    if value == "":
        return "%"
    out = []
    for b in value.encode("utf-8"):
        if 0x21 <= b <= 0x7E and b not in b"%{}":
            out.append(chr(b))
        else:
            out.append(("%%%02X" if upper else "%%%02x") % b)
    return "".join(out)


def scalar(name, value, ptype="string", section="Application:Test", comment="a scalar",
           tail=("%", "%", "%"), raw=False) -> str:
    v = value if raw else encode(str(value))
    return f"{section} {ptype} {name}= {v} {' '.join(tail)} // {comment}"


def plist(name, values, ptype="list", section="Application:Test", labels=None, comment="a list",
          tail=("%", "%", "%"), raw=False) -> str:
    vals = [v if raw else encode(str(v)) for v in values]
    dim = str(len(vals)) if labels is None else "{ " + " ".join(encode(l) for l in labels) + " }"
    return f"{section} {ptype} {name}= {dim} {' '.join(vals)} {' '.join(tail)} // {comment}".replace("  ", " ")


def pmatrix(name, rows, section="Application:Test", row_labels=None, col_labels=None, comment="a matrix",
            raw=False) -> str:
    n_rows, n_cols = len(rows), (len(rows[0]) if rows else 0)
    d1 = str(n_rows) if row_labels is None else "{ " + " ".join(encode(l) for l in row_labels) + " }"
    d2 = str(n_cols) if col_labels is None else "{ " + " ".join(encode(l) for l in col_labels) + " }"
    vals = [v if raw else encode(str(v)) for r in rows for v in r]
    return f"{section} matrix {name}= {d1} {d2} {' '.join(vals)} // {comment}".replace("  ", " ")


def signal_params(n_ch, gains=None, offsets=None, *, sampling_rate="250Hz", channel_names=None) -> list:
    sec = "Source:Signal%20Properties:DataIOFilter"
    gains = ["1muV"] * n_ch if gains is None else list(gains)
    offsets = ["0"] * n_ch if offsets is None else list(offsets)
    names = [f"Ch{i + 1}" for i in range(n_ch)] if channel_names is None else list(channel_names)
    return [
        scalar("SamplingRate", sampling_rate, "int", sec, "sample rate", tail=("256Hz", "0.0", "%"), raw=True),
        plist("SourceChGain", gains, "list", sec, comment="gain for each channel (A/D units -> muV)", raw=True),
        plist("SourceChOffset", offsets, "list", sec, comment="offset for channels in A/D units", raw=True),
        plist("ChannelNames", names, "list", sec, comment="names of channels"),
    ]


# ---------------------------------------------------------------------------
# states and the file
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateSpec:
    name: str
    length: int
    byte: int
    bit: int
    default: int = 0

    @property
    def start(self) -> int:
        return self.byte * 8 + self.bit


def pack_layout(specs, *, start_bit: int = 0, gap_bits: int = 0) -> tuple[list, int]:
    """Lay states out one after another (optionally with unused bits between them).
    Returns (StateSpec list, state-vector length in bytes)."""
    out, pos = [], start_bit
    for name, length in specs:
        out.append(StateSpec(name, length, pos // 8, pos % 8))
        pos += length + gap_bits
    return out, (pos + 7) // 8


def pack_states(n: int, specs, values: dict, sv_len: int, fill_unused: int = 0) -> np.ndarray:
    """(n, sv_len) uint8, built bit by bit. Unused bits are fill_unused (0 or 1)."""
    bits = np.full((n, sv_len * 8), fill_unused, dtype=np.uint8)
    for s in specs:
        v = np.broadcast_to(np.asarray(values.get(s.name, s.default), dtype=np.uint64), (n,))
        if np.any(v >> np.uint64(s.length)):
            raise ValueError(f"state {s.name}: value does not fit {s.length} bits")
        for k in range(s.length):
            bits[:, s.start + k] = ((v >> np.uint64(k)) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="little") if n else np.zeros((0, sv_len), np.uint8)


def write_dat(path, signal, *, data_format="float32", states=(), state_values=None, sv_len=None,
              params=(), version="1.1", eol="\r\n", fill_unused=0, truncate=0, extra=b"",
              header_len_delta=0, state_lines=None) -> Path:
    """Write a .dat. signal: (n_samples, n_channels) raw values (exactly representable in data_format).
    state_lines: raw state definition lines in place of the ones built from `states`."""
    path = Path(path)
    signal = np.asarray(signal)
    n, ch = signal.shape
    dt = np.dtype(DTYPES.get(data_format, "<f4"))
    raw = np.ascontiguousarray(signal.astype(dt))
    if not np.array_equal(raw.astype(np.float64), signal.astype(np.float64), equal_nan=True):
        raise ValueError("signal values are not representable in " + data_format)
    if sv_len is None:
        sv_len = max([(s.start + s.length + 7) // 8 for s in states] + [1])
    sv = pack_states(n, states, state_values or {}, sv_len, fill_unused)

    lines = state_lines if state_lines is not None else [
        f"{s.name} {s.length} {s.default} {s.byte} {s.bit}" for s in states]
    rest = (f" SourceCh= {ch} StatevectorLen= {sv_len} DataFormat= {data_format}{eol}"
            f"[ State Vector Definition ] {eol}" + "".join(l + eol for l in lines)
            + f"[ Parameter Definition ] {eol}" + "".join(p + eol for p in params) + eol).encode("utf-8")
    prefix = (f"BCI2000V= {version} " if version is not None else "") + "HeaderLen= "
    width = 5
    while True:      # BCI2000OutputFormat's scheme: the length field counts itself
        field_ = str(len(prefix) + width + len(rest) + header_len_delta).rjust(width)
        if len(field_) == width:
            break
        width += 1
    header = prefix.encode("ascii") + field_.encode("ascii") + rest
    body = np.concatenate([raw.view(np.uint8).reshape(n, ch * dt.itemsize), sv], axis=1).tobytes()
    data = header + body + extra
    path.write_bytes(data[:len(data) - truncate] if truncate else data)
    return path


# ---------------------------------------------------------------------------
# CheckerboardSpeller-shaped recordings
# ---------------------------------------------------------------------------

FLASH_STATES = tuple(f"FlashChar{i}" for i in range(1, 7))
SPELLER_STATE_SPECS = [
    ("Running", 1), ("Recording", 1), ("SourceTime", 16), ("StimulusTime", 16), ("StimulusCode", 16),
    ("StimulusType", 1), ("StimulusBegin", 1), ("PhaseInSequence", 2),
    *((s, 7) for s in FLASH_STATES), ("AttendedChar", 7), ("SelectedChar", 7),
    ("Counter", 32), ("AccelerationX", 32), ("AccelerationY", 32), ("AccelerationZ", 32),
    ("GyroscopeX", 32), ("GyroscopeY", 32), ("GyroscopeZ", 32), ("Battery", 32), ("Validation", 32),
]
SYNTHETIC_GYRO = 0x5A5A


@dataclass
class Session:
    n: int
    states: dict                         # name -> (n,) int64
    flashes: list                        # dicts: selection, code, onset, group, attended, is_target
    selection_starts: list = field(default_factory=list)


def build_session(selections, *, lead=300, pre=250, post=250, tail=500, on=16) -> Session:
    """
    selections: list of dicts {"attended": 0-based index or None, "flashes": [(group, off_samples), ...]}.
    Mirrors CheckerboardSpellerTask + StimulusTask: PhaseInSequence 1 (pre, AttendedChar set),
    2 (flashes: StimulusCode k = 1..n and FlashChar states for `on` samples, then `off` samples of 0;
    off = 0 means the next flash starts on the very next sample), 3 (post, AttendedChar cleared).
    """
    total = lead + tail + sum(pre + post + sum(on + off for _, off in s["flashes"]) for s in selections)
    st = {name: np.zeros(total, dtype=np.int64) for name, _ in SPELLER_STATE_SPECS}
    flashes, starts, t = [], [], lead
    for sel, s in enumerate(selections):
        att = s["attended"]
        starts.append(t)
        st["PhaseInSequence"][t:t + pre] = 1
        if att is not None:
            st["AttendedChar"][t:t + pre] = att + 1
        t += pre
        for code, (group, off) in enumerate(s["flashes"], start=1):
            st["PhaseInSequence"][t:t + on + off] = 2
            if att is not None:
                st["AttendedChar"][t:t + on + off] = att + 1
            st["StimulusCode"][t:t + on] = code
            is_target = att is not None and att in group
            st["StimulusType"][t:t + on] = int(is_target)
            st["StimulusBegin"][t] = 1
            for j, c in enumerate(group):
                st[FLASH_STATES[j]][t:t + on] = c + 1
            flashes.append({"selection": sel, "code": code, "onset": t, "group": tuple(group),
                            "attended": att, "is_target": is_target})
            t += on + off
        st["PhaseInSequence"][t:t + post] = 3
        t += post
    st["Running"][:] = 1
    st["Recording"][:] = 1
    st["SourceTime"][:] = (np.arange(total) * 4) % 65536
    st["StimulusTime"][:] = (np.arange(total) * 4 + 1) % 65536
    st["Counter"][:] = np.arange(total)
    st["Battery"][:] = 87
    st["Validation"][:] = 1
    return Session(n=total, states=st, flashes=flashes, selection_starts=starts)


def _flash_sequence_module():
    sys.path.insert(0, str(REPO / "collection"))
    import flash_sequence  # noqa: E402
    return flash_sequence


def checkerboard_selections(n_selections, n_reps=1, *, seed=0, off=15, targets=None) -> list:
    """Real checkerboard groups (collection/flash_sequence.py) for each selection, random cued characters."""
    fsq = _flash_sequence_module()
    rng = np.random.default_rng(seed)
    random.seed(seed)
    out = []
    for i in range(n_selections):
        groups = fsq.build_flash_sequence(fsq.CheckerboardConfig(n_chars=72, n_repetitions=n_reps))
        att = int(rng.integers(72)) if targets is None else targets[i]
        out.append({"attended": att, "flashes": [(tuple(int(c) for c in g), off) for g in groups]})
    return out


GAINS = np.array([0.45, 0.65, 0.90, 0.70, 1.00, 0.55, 0.45, 0.55])


def planted_signal(session: Session, *, seed=0, p300_uv=8.0, noise_uv=5.0) -> np.ndarray:
    """(n, 8) float32-exact microvolts: noise plus a P300-like bump after every target flash onset."""
    rng = np.random.default_rng(seed + 1000)
    x = rng.normal(0, noise_uv, size=(session.n, 8))
    bt = np.arange(int(0.8 * SFREQ)) / SFREQ
    bump = p300_uv * np.exp(-0.5 * ((bt - 0.35) / 0.06) ** 2)
    for f in session.flashes:
        if f["is_target"]:
            seg = x[f["onset"]:f["onset"] + len(bump)]
            seg += bump[:len(seg), None] * GAINS
    return x.astype(np.float32).astype(np.float64)


def write_speller_dat(path, session: Session, signal=None, *, lost=(), synthetic_gyro=False, seed=0,
                      channel_names=EEG, sampling_rate="250Hz", text_to_spell="ก่อน", extra_params=(),
                      omit_states=(), omit_params=(), gains=None, data_format="float32",
                      source_filter="UnicornSerialADC", tcp_address="") -> Path:
    """
    lost: (start, stop) sample ranges the source filled in for lost packets: Validation = 0 there,
    and the signal repeats the last real sample (what UnicornSerialSource writes).
    source_filter / tcp_address: what BCI2000 records about the source module (SignalSourceFilterChain,
    TcpAddress); the defaults are a real-headset UnicornSerialSource recording.
    """
    rng = np.random.default_rng(seed + 7)
    st = {k: v.copy() for k, v in session.states.items()}
    sig = np.zeros((session.n, 8)) if signal is None else np.array(signal, dtype=np.float64, copy=True)
    for a, b in lost:
        st["Validation"][a:b] = 0
        sig[a:b] = sig[a - 1]
    for ax in "XYZ":
        st[f"Acceleration{ax}"][:] = rng.integers(0, 4096, session.n)
        st[f"Gyroscope{ax}"][:] = SYNTHETIC_GYRO if synthetic_gyro else rng.integers(0, 2000, session.n)
    specs, sv_len = pack_layout([s for s in SPELLER_STATE_SPECS if s[0] not in omit_states])
    params = [p for p in signal_params(sig.shape[1], gains=gains, sampling_rate=sampling_rate,
                                       channel_names=channel_names)
              if p.split()[2].rstrip("=") not in omit_params]
    app = "Application:Speller"
    params += [
        pmatrix("SignalSourceFilterChain", [["DataIOFilter", "0"], [source_filter, "1"]],
                section="System:Configuration", col_labels=["Filter Name", "Position String"]),
        scalar("TcpAddress", tcp_address, section="Source:Unicorn"),
        scalar("SubjectName", "subj01", section="Storage:Documentation"),
        scalar("SubjectSession", "001", section="Storage:Documentation"),
        scalar("SubjectRun", "02", section="Storage:Documentation"),
        scalar("InterpretMode", "2", "int", app),
        scalar("TextToSpell", text_to_spell, section=app),
        scalar("NumberOfSequences", "2", "int", "Application:Sequencing"),
        scalar("StimulusDuration", "64ms", "float", "Application:Sequencing"),
        scalar("ISIMinDuration", "60ms", "float", "Application:Sequencing"),
        *extra_params,
    ]
    params = [p for p in params if p.split()[2].rstrip("=") not in omit_params]
    return write_dat(path, sig, data_format=data_format, states=specs,
                     state_values={k: v.astype(np.uint64) for k, v in st.items()},
                     sv_len=sv_len, params=params)
