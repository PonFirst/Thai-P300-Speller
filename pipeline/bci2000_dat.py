"""
Read BCI2000 .dat files (format 1.1): signal, per-sample states, parameters.

BCI2000's own Python reader (BCI2000Tools.FileReader, in ~/BCI2000/trunk/tools/python) lives
outside this repo and its environment, so the training path reads .dat itself;
tests/test_bci2000_dat.py checks this reader against BCI2000's on the same files whenever a
BCI2000 build is present.

File layout (BCI2000 file format 1.1):
  line 1     "BCI2000V= 1.1 HeaderLen= N SourceCh= C StatevectorLen= S DataFormat= int16|int32|float32"
  header     "[ State Vector Definition ]" lines "Name Length Value ByteLocation BitLocation",
             then "[ Parameter Definition ]" lines "Section Type Name= values... // comment",
             N bytes in total
  data       per sample: C little-endian values, then S bytes of state vector

A state occupies Length bits starting at bit ByteLocation*8 + BitLocation of the sample's state
vector, least significant bit first (BCI2000 StateVector). Parameter values use BCI2000's
encoding: %XX for any byte that is not printable ASCII, '%' alone for an empty value.

The signal is returned in microvolts: (raw - SourceChOffset) * SourceChGain, with the gain's unit
converted (BCI2000 writes gains like "1muV" or "0.0447uV"). CLAUDE.md gotcha 1: everything on the
collection/pipeline side is microvolts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_VERSION = "1.1"
_DTYPES = {"int16": "<i2", "int32": "<i4", "float32": "<f4"}
# BCI2000 PhysicalUnit prefixes for volts: mu, u and the micro sign all mean 1e-6.
_UNIT_TO_UV = {"V": 1e6, "mV": 1e3, "muV": 1.0, "uV": 1.0, "µV": 1.0, "μV": 1.0, "nV": 1e-3, "kV": 1e9}
_LIST_TYPES = {"list", "intlist", "floatlist", "stringlist"}
_SCALAR_TYPES = {"int", "float", "string", "variant", "blob"}


class DatError(ValueError):
    """The file is not a readable BCI2000 1.1 .dat file, or a value it holds is malformed."""


@dataclass(frozen=True)
class StateDef:
    name: str
    length: int
    default: int
    byte: int
    bit: int


@dataclass(frozen=True)
class Param:
    section: str
    type: str
    name: str
    value: object                 # str (scalar) | list[str] (list) | list[list[str]] (matrix)
    row_labels: tuple = ()
    col_labels: tuple = ()


@dataclass
class DatFile:
    path: Path
    signal_raw: np.ndarray        # (n_samples, n_channels), as stored
    signal_uv: np.ndarray         # (n_samples, n_channels), float64 microvolts
    states: dict                  # name -> (n_samples,) uint32
    state_defs: dict              # name -> StateDef
    params: dict                  # name -> Param
    data_format: str
    truncated_tail_bytes: int = 0  # a partial last sample (a run cut off mid-write), dropped

    @property
    def n_samples(self) -> int:
        return self.signal_raw.shape[0]

    def state(self, name: str) -> np.ndarray:
        if name not in self.states:
            raise DatError(f"{self.path.name}: no state {name!r}")
        return self.states[name]

    def state_float(self, name: str) -> np.ndarray:
        """A 32-bit state written with StateRef::AsFloat(), as float32 values."""
        sd = self.state_defs.get(name)
        if sd is None or sd.length != 32:
            raise DatError(f"{self.path.name}: state {name!r} is not a 32-bit state")
        return self.states[name].astype("<u4").view("<f4")

    def param(self, name: str) -> Param:
        if name not in self.params:
            raise DatError(f"{self.path.name}: no parameter {name!r}")
        return self.params[name]

    def param_str(self, name: str) -> str:
        p = self.param(name)
        if not isinstance(p.value, str):
            raise DatError(f"{self.path.name}: parameter {name} is a {p.type}, not a single value")
        return p.value

    def param_list(self, name: str) -> list:
        p = self.param(name)
        if p.type not in _LIST_TYPES:
            raise DatError(f"{self.path.name}: parameter {name} is a {p.type}, not a list")
        return list(p.value)

    def param_number(self, name: str, unit: str | None = None) -> float:
        """A numeric value, with an optional unit suffix: 'Hz' accepts '250' and '250Hz'."""
        return parse_number(self.param_str(name), unit, where=f"{self.path.name}: {name}")


def decode_value(token: str) -> str:
    """
    BCI2000's EncodedString::ExtractFrom, rule for rule: a '%' is removed and up to two following
    hex digits become one byte (a zero byte is dropped, and the character after it is then taken
    literally, as ExtractFrom resumes its search one past it); '%' followed by 'Z' (or nothing) is the
    empty string; '%' followed by anything else leaves that character literal, so '%%' is '%'.
    Bytes are UTF-8; an invalid sequence is replaced rather than making the whole file unreadable.
    """
    data = token.encode("utf-8")
    out, i, n = bytearray(), 0, len(data)
    hexdigits = b"0123456789abcdefABCDEF"
    while i < n:
        c = data[i]
        if c != 0x25:                                   # not '%'
            out.append(c)
            i += 1
            continue
        i += 1
        digits = 0
        while digits < 2 and i + digits < n and data[i + digits] in hexdigits:
            digits += 1
        if digits:
            value = int(data[i:i + digits], 16)
            i += digits
            if value:
                out.append(value)
            elif i < n:
                out.append(data[i])                     # ExtractFrom resumes past the next character
                i += 1
        elif i < n and data[i] in b"zZ":
            i += 1                                      # alternative empty-string encoding
        elif i < n:
            out.append(data[i])                         # literal, e.g. the second '%' of '%%'
            i += 1
    return out.decode("utf-8", errors="replace")


def parse_number(text: str, unit: str | None = None, *, where: str = "value") -> float:
    m = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zµμ]*)\s*", text)
    if not m:
        raise DatError(f"{where}: {text!r} is not a number")
    value, suffix = float(m.group(1)), m.group(2)
    if suffix and unit is not None and suffix != unit:
        raise DatError(f"{where}: {text!r} has unit {suffix!r}, expected {unit!r}")
    if suffix and unit is None:
        raise DatError(f"{where}: {text!r} has an unexpected unit {suffix!r}")
    return value


def _gain_parts(text: str, where: str) -> tuple[float, str]:
    m = re.fullmatch(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zµμ]*)\s*", text)
    if not m or (m.group(2) and m.group(2) not in _UNIT_TO_UV):
        raise DatError(f"{where}: cannot read gain {text!r}")
    return float(m.group(1)), m.group(2)


def gain_to_uv(text: str, *, where: str = "SourceChGain") -> float:
    """'1muV' -> 1.0, '0.0447uV' -> 0.0447, '1mV' -> 1000.0, '2' -> 2.0 (unitless: microvolts)."""
    value, unit = _gain_parts(text, where)
    return value * (_UNIT_TO_UV[unit] if unit else 1.0)


def _tokens(text: str) -> list[str]:
    """Whitespace tokens, with braces split off ('{Display' -> '{', 'Display'). Values never contain
    a raw brace: BCI2000 writes them as %7B / %7D."""
    out = []
    for t in text.split():
        while t.startswith("{") and len(t) > 1:
            out.append("{")
            t = t[1:]
        tail = []
        while t.endswith("}") and len(t) > 1:
            tail.append("}")
            t = t[:-1]
        out.append(t)
        out.extend(tail)
    return out


def _take_dim(tokens: list[str], i: int, where: str) -> tuple[int, tuple, int]:
    """A matrix dimension: a count, or '{ label ... }'. Returns (size, labels, next index)."""
    if i >= len(tokens):
        raise DatError(f"{where}: matrix dimension missing")
    if tokens[i] == "{":
        j = tokens.index("}", i) if "}" in tokens[i:] else -1
        if j < 0:
            raise DatError(f"{where}: unterminated label list")
        labels = tuple(decode_value(t) for t in tokens[i + 1:j])
        return len(labels), labels, j + 1
    if tokens[i].startswith("{"):
        raise DatError(f"{where}: cannot parse labels {tokens[i]!r}")
    try:
        return int(tokens[i]), (), i + 1
    except ValueError as exc:
        raise DatError(f"{where}: bad matrix dimension {tokens[i]!r}") from exc


def parse_param_line(line: str) -> Param | None:
    """
    One parameter line, or None for a line that is not a parameter definition.

    As BCI2000's Param::ExtractFrom: the declared values are taken by count first; whatever follows
    (default, range, '// comment') is ignored. A value may itself contain '//' (BCI2000 doesn't
    escape '/'), so the line is never cut at the first '//'.
    """
    m = re.match(r"\s*(\S+)\s+(\S+)\s+(\S+?)=\s*(.*)$", line)
    if not m:
        return None
    section, ptype, name, rest = m.groups()
    toks = _tokens(rest)
    where = f"parameter {name}"
    if ptype in _LIST_TYPES:
        if not toks:
            raise DatError(f"{where}: missing list length")
        if toks[0] == "{":                      # labelled list: { labels } then values
            n, labels, i = _take_dim(toks, 0, where)
        else:
            try:
                n, labels, i = int(toks[0]), (), 1
            except ValueError as exc:
                raise DatError(f"{where}: bad list length {toks[0]!r}") from exc
        if len(toks) < i + n:
            raise DatError(f"{where}: {n} values declared, {len(toks) - i} present")
        return Param(section, ptype, name, [decode_value(t) for t in toks[i:i + n]], row_labels=labels)
    if ptype == "matrix":
        rows, row_labels, i = _take_dim(toks, 0, where)
        cols, col_labels, i = _take_dim(toks, i, where)
        values = toks[i:i + rows * cols]
        if len(values) < rows * cols or any(v.startswith("{") for v in values):
            raise DatError(f"{where}: expected {rows}x{cols} plain values (nested matrices are not supported)")
        dec = [decode_value(t) for t in values]
        return Param(section, ptype, name, [dec[r * cols:(r + 1) * cols] for r in range(rows)],
                     row_labels=row_labels, col_labels=col_labels)
    # scalars: value [default low high]
    return Param(section, ptype, name, decode_value(toks[0]) if toks else "")


def read_dat(path) -> DatFile:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatError(f"cannot read {path}: {exc}") from exc
    nl = raw.find(b"\n")
    if nl < 0:
        raise DatError(f"{path.name}: no header line")
    first = raw[:nl].decode("ascii", errors="replace")
    fields = dict(re.findall(r"(\w+)=\s*(\S+)", first))
    if fields.get("BCI2000V") != SUPPORTED_VERSION:
        raise DatError(f"{path.name}: BCI2000 file format {fields.get('BCI2000V', '1.0 or unknown')!r}; "
                       f"only {SUPPORTED_VERSION} is supported")
    try:
        header_len, n_ch, sv_len = int(fields["HeaderLen"]), int(fields["SourceCh"]), int(fields["StatevectorLen"])
        fmt = fields["DataFormat"]
    except (KeyError, ValueError) as exc:
        raise DatError(f"{path.name}: malformed first line {first!r}") from exc
    if fmt not in _DTYPES:
        raise DatError(f"{path.name}: unsupported DataFormat {fmt!r}")
    if header_len > len(raw) or n_ch < 1 or sv_len < 1:
        raise DatError(f"{path.name}: header fields out of range ({first!r})")

    header = raw[:header_len].decode("utf-8", errors="replace").replace("\r\n", "\n").split("\n")
    state_defs, params, section = {}, {}, None
    for line in header[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("["):
            section = s.strip("[] ").lower()
            continue
        if section == "state vector definition":
            parts = s.split()
            try:
                if len(parts) != 5:
                    raise ValueError
                name, length, default, byte, bit = parts[0], *map(int, parts[1:])
            except ValueError as exc:
                raise DatError(f"{path.name}: bad state definition {s!r}") from exc
            if not 1 <= length <= 32 or not 0 <= bit < 8 or byte < 0 or (byte * 8 + bit + length) > sv_len * 8:
                raise DatError(f"{path.name}: state {name} lies outside the state vector")
            state_defs[name] = StateDef(name, length, default, byte, bit)
        elif section == "parameter definition":
            p = parse_param_line(s)
            if p is not None:
                params[p.name] = p

    width = n_ch * np.dtype(_DTYPES[fmt]).itemsize
    rec = width + sv_len
    body = raw[header_len:]
    n = len(body) // rec
    tail = len(body) - n * rec          # a run cut off mid-write leaves a partial last sample
    block = np.frombuffer(body, dtype=np.uint8, count=n * rec).reshape(n, rec)
    signal_raw = np.ascontiguousarray(block[:, :width]).view(_DTYPES[fmt]).reshape(n, n_ch)
    sv = block[:, width:]

    # Little-endian bit order across the state vector; only the bytes a state spans are widened.
    states = {}
    for sd in state_defs.values():
        n_bytes = (sd.bit + sd.length + 7) // 8
        word = np.zeros(n, dtype=np.uint64)
        for k in range(n_bytes):
            word |= sv[:, sd.byte + k].astype(np.uint64) << np.uint64(8 * k)
        states[sd.name] = ((word >> np.uint64(sd.bit)) & np.uint64((1 << sd.length) - 1)).astype(np.uint32)

    gains = _channel_list(params, "SourceChGain", n_ch, path)
    offsets = _channel_list(params, "SourceChOffset", n_ch, path)
    units = [_gain_parts(v, f"{path.name}: SourceChGain")[1] for v in gains]
    if any(units) and not all(units):
        # BCI2000 reads unitless gains as microvolts only when no channel has a unit
        # (DataIOFilter); a mix would be read differently here and there.
        raise DatError(f"{path.name}: SourceChGain mixes gains with and without units {gains}")
    g = np.array([gain_to_uv(v, where=f"{path.name}: SourceChGain") for v in gains])
    o = np.array([parse_number(v, None, where=f"{path.name}: SourceChOffset") for v in offsets])
    signal_uv = (signal_raw.astype(np.float64) - o) * g
    return DatFile(path=path, signal_raw=signal_raw, signal_uv=signal_uv, states=states,
                   state_defs=state_defs, params=params, data_format=fmt, truncated_tail_bytes=tail)


def _channel_list(params: dict, name: str, n_ch: int, path: Path) -> list:
    p = params.get(name)
    if p is None or p.type not in _LIST_TYPES:
        raise DatError(f"{path.name}: missing list parameter {name}")
    values = list(p.value)
    if len(values) < n_ch:                 # BCI2000 needs at least one per channel and uses the first n
        raise DatError(f"{path.name}: {name} has {len(values)} values for {n_ch} channels")
    return values[:n_ch]
