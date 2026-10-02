"""
pipeline/bci2000_dat.py against .dat files built byte by byte by tests/bci2000_dat_writer.py (an
independent writer), and against BCI2000's own Python reader (BCI2000Tools.FileReader) on the same
files whenever a BCI2000 source tree is present. Nothing is written outside tmp_path.
"""

import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import bci2000_dat_writer as w
from pipeline import bci2000_dat as bd

BCI2000_ROOT = Path(os.environ.get("BCI2000_ROOT", Path.home() / "BCI2000" / "trunk"))
SAMPLEFILES = BCI2000_ROOT / "data" / "samplefiles"


def simple_file(path, signal, *, data_format="float32", gains=None, offsets=None, params=(), **kw):
    n_ch = np.asarray(signal).shape[1]
    specs, sv_len = w.pack_layout([("Running", 1)])
    return w.write_dat(path, signal, data_format=data_format, states=specs, sv_len=sv_len,
                       params=[*w.signal_params(n_ch, gains, offsets), *params], **kw)


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------

EXTREMES = {
    "int16": [-32768, -1, 0, 1, 32767],
    "int32": [-2 ** 31, -1, 0, 1, 2 ** 31 - 1],
    "float32": [-3.4028234663852886e38, -1.5, 0.0, 1.401298464324817e-45, 3.4028234663852886e38],
}


@pytest.mark.parametrize("fmt", ["int16", "int32", "float32"])
def test_every_data_format_reads_back_exactly(tmp_path, fmt):
    rng = np.random.default_rng(1)
    ext = np.array(EXTREMES[fmt], dtype=np.float64)
    if fmt == "float32":
        body = rng.normal(0, 100, size=(40, 3)).astype(np.float32).astype(np.float64)
    else:
        lim = 32767 if fmt == "int16" else 2 ** 31 - 1
        body = rng.integers(-lim, lim, size=(40, 3)).astype(np.float64)
    signal = np.vstack([np.column_stack([ext, ext[::-1], ext]), body])
    d = bd.read_dat(simple_file(tmp_path / f"{fmt}.dat", signal, data_format=fmt))
    assert d.data_format == fmt and d.n_samples == len(signal)
    assert d.signal_raw.dtype == np.dtype(w.DTYPES[fmt])
    assert d.signal_raw.shape == signal.shape
    np.testing.assert_array_equal(d.signal_raw.astype(np.float64), signal)
    assert d.signal_uv.dtype == np.float64
    np.testing.assert_array_equal(d.signal_uv, signal)   # gain 1muV, offset 0
    np.testing.assert_array_equal(d.state("Running"), np.zeros(len(signal), np.uint32))


def test_header_only_file_has_zero_samples(tmp_path):
    d = bd.read_dat(simple_file(tmp_path / "empty.dat", np.zeros((0, 2))))
    assert d.n_samples == 0 and d.signal_uv.shape == (0, 2) and d.state("Running").shape == (0,)


def test_lf_line_endings_read_like_crlf(tmp_path):
    sig = np.arange(12, dtype=np.float64).reshape(6, 2)
    a = bd.read_dat(simple_file(tmp_path / "crlf.dat", sig, params=[w.scalar("X", "ก")]))
    b = bd.read_dat(simple_file(tmp_path / "lf.dat", sig, params=[w.scalar("X", "ก")], eol="\n"))
    np.testing.assert_array_equal(a.signal_uv, b.signal_uv)
    assert a.param_str("X") == b.param_str("X") == "ก"


@pytest.mark.parametrize("gain, uv_per_unit", [
    ("1muV", 1.0), ("0.0447uV", 0.0447), ("1mV", 1000.0), ("2", 2.0), ("0.5V", 5e5), ("250nV", 0.25),
    ("1µV", 1.0), ("1μV", 1.0), ("3e-2muV", 0.03), (".5muV", 0.5), ("1kV", 1e9)])
def test_gain_units_are_converted_to_microvolts(tmp_path, gain, uv_per_unit):
    raw = np.array([[-100, 7], [0, 1], [32767, -32768]], dtype=np.float64)
    other = "1muV" if gain[-1:] == "V" else "1"        # never mix unitless and united gains (refused)
    d = bd.read_dat(simple_file(tmp_path / "g.dat", raw, data_format="int16", gains=[gain, other]))
    np.testing.assert_allclose(d.signal_uv[:, 0], raw[:, 0] * uv_per_unit, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(d.signal_uv[:, 1], raw[:, 1])


def test_offsets_are_subtracted_before_the_gain(tmp_path):
    raw = np.array([[10, 10, 10], [-4, 0, 4]], dtype=np.float64)
    d = bd.read_dat(simple_file(tmp_path / "o.dat", raw, data_format="int32",
                                gains=["2muV", "0.5mV", "1muV"], offsets=["3", "-12.5", "0"]))
    expected = (raw - np.array([3, -12.5, 0])) * np.array([2.0, 500.0, 1.0])
    np.testing.assert_array_equal(d.signal_uv, expected)
    np.testing.assert_array_equal(d.signal_raw, raw.astype(np.int32))  # raw stays raw


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------

# Odd offsets, byte-spanning, 1-bit and 32-bit states, a 32-bit state at bit 7 that spans 5 bytes,
# one ending exactly at the end of the vector, listed out of bit order.
ODD_LAYOUT = [
    w.StateSpec("Flag", 1, 0, 0),
    w.StateSpec("Three", 3, 0, 1),
    w.StateSpec("Wide16", 16, 0, 5),          # bits 5..20: bytes 0..2
    w.StateSpec("Word", 32, 2, 7),            # bits 23..54: bytes 2..6 (5 bytes)
    w.StateSpec("Seven", 7, 6, 7),            # bits 55..61
    w.StateSpec("One", 1, 7, 6),              # bit 62 (bit 63 unused)
    w.StateSpec("Last32", 32, 8, 0),          # bytes 8..11, ends at the vector end
    w.StateSpec("Mid5", 5, 13, 3),            # bits 107..111 (byte 12 + low bits of 13 unused)
]
ODD_SV_LEN = 14


def odd_values(n, rng):
    v = {s.name: rng.integers(0, 2 ** s.length, n, dtype=np.uint64) for s in ODD_LAYOUT}
    for s in ODD_LAYOUT:            # every state at its maximum, alone and together, and at zero
        v[s.name][0] = 2 ** s.length - 1
        v[s.name][1] = 0
    for i, s in enumerate(ODD_LAYOUT):
        for t in ODD_LAYOUT:
            v[t.name][2 + i] = 2 ** t.length - 1 if t is s else 0
            v[t.name][2 + len(ODD_LAYOUT) + i] = 0 if t is s else 2 ** t.length - 1
    return v


def write_odd(path, n=40, seed=0, fill_unused=1, shuffle_lines=True, data_format="int16"):
    rng = np.random.default_rng(seed)
    vals = odd_values(n, rng)
    order = list(ODD_LAYOUT)
    if shuffle_lines:
        order = [order[i] for i in rng.permutation(len(order))]
    sig = rng.integers(-2000, 2000, size=(n, 3)).astype(np.float64)
    w.write_dat(path, sig, data_format=data_format, states=order, state_values=vals, sv_len=ODD_SV_LEN,
                fill_unused=fill_unused, params=w.signal_params(3, gains=["0.5muV", "2uV", "1mV"],
                                                                 offsets=["1", "-3", "0"]))
    return vals, sig


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("fill_unused", [0, 1])
def test_states_at_odd_bit_offsets_read_exactly(tmp_path, seed, fill_unused):
    vals, _ = write_odd(tmp_path / "s.dat", seed=seed, fill_unused=fill_unused)
    d = bd.read_dat(tmp_path / "s.dat")
    assert set(d.states) == {s.name for s in ODD_LAYOUT}
    for s in ODD_LAYOUT:
        got = d.state(s.name)
        assert got.dtype == np.uint32
        np.testing.assert_array_equal(got.astype(np.uint64), vals[s.name], err_msg=s.name)
        assert d.state_defs[s.name] == bd.StateDef(s.name, s.length, 0, s.byte, s.bit)


def test_state_bit_order_is_lsb_first_from_byte_and_bit(tmp_path):
    # Hand-computed bytes, no writer involved: "A" 4 bits at byte 0 bit 6 = bits 6..9,
    # value 0b1011 -> bit6=1 bit7=1 bit8=0 bit9=1 -> byte0 = 0b11000000, byte1 = 0b00000010.
    header_rest = ("SourceCh= 1 StatevectorLen= 2 DataFormat= int16\r\n"
                   "[ State Vector Definition ] \r\nA 4 0 0 6\r\n"
                   "[ Parameter Definition ] \r\n"
                   "S list SourceChGain= 1 1muV // g\r\nS list SourceChOffset= 1 0 // o\r\n\r\n")
    prefix = "BCI2000V= 1.1 HeaderLen= "
    n = len(prefix) + 5 + 1 + len(header_rest)
    header = f"{prefix}{n:5d} {header_rest}".encode()
    assert len(header) == n
    body = struct.pack("<h", -7) + bytes([0b11000000, 0b00000010]) + struct.pack("<h", 5) + bytes([0x3F, 0xFC])
    (tmp_path / "h.dat").write_bytes(header + body)
    d = bd.read_dat(tmp_path / "h.dat")
    np.testing.assert_array_equal(d.state("A"), [0b1011, 0])     # 0x3F,0xFC: bits 6..9 are 0,0,0,0
    np.testing.assert_array_equal(d.signal_raw[:, 0], [-7, 5])


def test_float_state_is_reinterpreted_as_float32(tmp_path):
    floats = np.array([0.0, -1.5, 3.14159, 1e-30, -2.5e12, np.inf, -0.0], dtype=np.float32)
    bits = floats.view(np.uint32).astype(np.uint64)
    specs = [w.StateSpec("Pad", 3, 0, 0), w.StateSpec("F", 32, 0, 3), w.StateSpec("Small", 5, 4, 3)]
    p = w.write_dat(tmp_path / "f.dat", np.zeros((len(floats), 1)), states=specs,
                    state_values={"F": bits, "Small": 31, "Pad": 7}, params=w.signal_params(1))
    d = bd.read_dat(p)
    got = d.state_float("F")
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got.view(np.uint32), floats.view(np.uint32))   # bit-exact, incl. -0.0
    with pytest.raises(bd.DatError, match="32-bit"):
        d.state_float("Small")
    with pytest.raises(bd.DatError, match="32-bit"):
        d.state_float("Nope")
    with pytest.raises(bd.DatError, match="no state"):
        d.state("Nope")


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def test_parameters_of_every_shape(tmp_path):
    thai = "ภาษาไทย ก่อ"
    params = [
        w.scalar("Str", "hello"),
        w.scalar("Empty", ""),
        w.scalar("Rate", "250Hz", "int", raw=True),
        w.scalar("Ratio", "0.25", "float", tail=("0.5", "0", "1")),
        w.scalar("Thai", thai),
        w.scalar("ThaiUpper", w.encode(thai, upper=True), raw=True),
        w.scalar("Braces", "{a} 50% done"),
        w.plist("NoItems", []),
        w.plist("Names", ["Fz", "C3", "ก", ""]),
        w.plist("Labelled", ["1", "2", "3"], "intlist", labels=["one", "two words", "ข"]),
        w.pmatrix("Numeric", [["1", "2", "3"], ["4", "5", ""]]),
        w.pmatrix("Labels", [["a", "b"], ["c", "ง"]], row_labels=["r 1", "r2"], col_labels=["c1", "ค"]),
        w.pmatrix("RowLabelsOnly", [["x"], ["y"]], row_labels=["first", "second"]),
        w.pmatrix("ColLabelsOnly", [["x", "y"]], col_labels=["L", "R"]),
        w.pmatrix("Zero", []),
    ]
    d = bd.read_dat(simple_file(tmp_path / "p.dat", np.zeros((2, 1)), params=params))
    assert d.param_str("Str") == "hello"
    assert d.param_str("Empty") == ""
    assert d.param_number("Rate", "Hz") == 250.0
    assert d.param_number("Ratio") == 0.25
    assert d.param_str("Thai") == thai and d.param_str("ThaiUpper") == thai
    assert d.param_str("Braces") == "{a} 50% done"
    assert d.param_list("NoItems") == []
    assert d.param_list("Names") == ["Fz", "C3", "ก", ""]
    lab = d.param("Labelled")
    assert lab.value == ["1", "2", "3"] and lab.row_labels == ("one", "two words", "ข")
    assert d.param("Numeric").value == [["1", "2", "3"], ["4", "5", ""]]
    m = d.param("Labels")
    assert m.value == [["a", "b"], ["c", "ง"]]
    assert m.row_labels == ("r 1", "r2") and m.col_labels == ("c1", "ค")
    assert d.param("RowLabelsOnly").value == [["x"], ["y"]]
    assert d.param("RowLabelsOnly").row_labels == ("first", "second") and d.param("RowLabelsOnly").col_labels == ()
    assert d.param("ColLabelsOnly").value == [["x", "y"]] and d.param("ColLabelsOnly").col_labels == ("L", "R")
    assert d.param("Zero").value == []
    assert d.param_list("SourceChGain") == ["1muV"]
    assert d.param("Str").section == "Application:Test" and d.param("Str").type == "string"


def test_parameter_accessors_refuse_the_wrong_shape(tmp_path):
    d = bd.read_dat(simple_file(tmp_path / "p.dat", np.zeros((1, 1)), params=[
        w.scalar("S", "x"), w.scalar("Rate", "250Hz", "int", raw=True), w.pmatrix("M", [["1"]])]))
    with pytest.raises(bd.DatError, match="not a single value"):
        d.param_str("SourceChGain")
    with pytest.raises(bd.DatError, match="not a list"):
        d.param_list("S")
    with pytest.raises(bd.DatError, match="not a list"):
        d.param_list("M")
    with pytest.raises(bd.DatError, match="no parameter"):
        d.param("Missing")
    with pytest.raises(bd.DatError, match="unit"):
        d.param_number("Rate", "ms")
    with pytest.raises(bd.DatError, match="unexpected unit"):
        d.param_number("Rate")
    with pytest.raises(bd.DatError, match="not a number"):
        d.param_number("S")


@pytest.mark.parametrize("token, value", [
    ("%", ""), ("abc", "abc"), ("%20", " "), ("a%7Bb%7d", "a{b}"), ("%25", "%"),
    ("%e0%b8%81", "ก"), ("%E0%B8%81%E0%B9%88", "ก่"),
    # BCI2000's EncodedString::ExtractFrom, rule for rule: a trailing '%' is dropped, '%Z' is the
    # empty string, one hex digit is a byte, a zero byte is dropped.
    ("50%", "50"), ("%zz", "z"), ("%Z", ""), ("%4", "\x04"), ("a%00b", "ab"), ("%G", "G"),
    # after a zero-valued escape ExtractFrom resumes one past the next character, leaving it literal
    ("%0%41", "%41"), ("5b%0%", "5b%"), ("%0%d", "%d"), ("%00%%", "%")])
def test_decode_value(token, value):
    assert bd.decode_value(token) == value


def test_decode_value_reads_bci2000s_own_percent_escape():
    # BCI2000's EncodedString::InsertInto writes a literal '%' as "%%" (EncodedString.cpp), and both
    # its C++ ExtractFrom and BCI2000Tools.FileReader.Unescape read "%%" back as '%'. A TextToSpell
    # or any string parameter BCI2000 itself wrote with a '%' in it must decode to that '%'.
    assert bd.decode_value("50%%") == "50%"
    assert bd.decode_value("%%") == "%"


def test_parameter_values_may_contain_double_slashes():
    # BCI2000's Param::ReadFromStream reads the declared values as tokens first and only then looks
    # for the comment, with rfind("//") on the remainder (Param.cpp); BCI2000Tools.FileReader also
    # splits at the LAST " //". A value containing "//" (a URL, a UNC-ish path) is therefore kept whole.
    p = bd.parse_param_line("Application:Test string Url= http://example.org // where to look")
    assert p.value == "http://example.org"
    p = bd.parse_param_line("Application:Test list Paths= 2 a//b c // two paths")
    assert p.value == ["a//b", "c"]


def test_parse_param_line_edge_cases():
    assert bd.parse_param_line("this is not a parameter") is None
    assert bd.parse_param_line("") is None
    p = bd.parse_param_line("Sec int N= 3 1 0 10 // with default low high")
    assert (p.section, p.type, p.name, p.value) == ("Sec", "int", "N", "3")
    with pytest.raises(bd.DatError, match="values declared"):
        bd.parse_param_line("Sec list L= 3 a b")
    # As in BCI2000, values are taken by count first, so a '//' token can be a value.
    assert bd.parse_param_line("Sec list L= 3 a b // short").value == ["a", "b", "//"]
    assert bd.parse_param_line("Sec matrix M= {a b} 1 x y // labels without spaces").row_labels == ("a", "b")
    with pytest.raises(bd.DatError, match="list length"):
        bd.parse_param_line("Sec list L= x a b")
    with pytest.raises(bd.DatError, match="unterminated"):
        bd.parse_param_line("Sec list L= { a b")
    with pytest.raises(bd.DatError, match="2x2"):
        bd.parse_param_line("Sec matrix M= 2 2 1 2 3")
    with pytest.raises(bd.DatError, match="nested"):
        bd.parse_param_line("Sec matrix M= 1 1 { matrix 1 1 0 }")
    with pytest.raises(bd.DatError, match="dimension"):
        bd.parse_param_line("Sec matrix M= 2")


@pytest.mark.skipif(not SAMPLEFILES.is_dir(), reason="no BCI2000 sample files")
def test_every_parameter_line_of_bci2000s_sample_files_parses():
    # Real BCI2000-written headers (format 1.0, but the parameter syntax is the same).
    files = sorted(SAMPLEFILES.glob("*.dat"))
    assert files
    n = 0
    for f in files:
        raw = f.read_bytes()
        hl = int(re.search(rb"HeaderLen=\s*(\d+)", raw).group(1))
        lines = raw[:hl].decode("utf-8").replace("\r\n", "\n").split("\n")
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("[ Parameter"))
        for line in lines[start + 1:]:
            if line.strip():
                assert bd.parse_param_line(line) is not None, line
                n += 1
    assert n > 500


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_a_partial_last_sample_is_dropped_and_recorded(tmp_path):
    # A run cut off mid-write leaves a partial last sample; the whole samples before it are kept.
    full = bd.read_dat(simple_file(tmp_path / "f.dat", np.arange(10.0).reshape(5, 2)))
    d = bd.read_dat(simple_file(tmp_path / "t.dat", np.arange(10.0).reshape(5, 2), truncate=1))
    assert d.n_samples == 4 and d.truncated_tail_bytes > 0 and full.truncated_tail_bytes == 0
    np.testing.assert_array_equal(d.signal_uv, full.signal_uv[:4])
    d = bd.read_dat(simple_file(tmp_path / "x.dat", np.ones((5, 2)), extra=b"\x00"))
    assert d.n_samples == 5 and d.truncated_tail_bytes == 1


@pytest.mark.parametrize("version", ["1.0", "1.2", "2", None])
def test_other_format_versions_are_refused(tmp_path, version):
    p = simple_file(tmp_path / "v.dat", np.ones((2, 1)), version=version)
    with pytest.raises(bd.DatError, match="only 1.1"):
        bd.read_dat(p)


@pytest.mark.parametrize("fmt", ["float64", "int8", "uint16"])
def test_unsupported_data_format_is_refused(tmp_path, fmt):
    p = simple_file(tmp_path / "f.dat", np.ones((2, 1)), data_format=fmt)
    with pytest.raises(bd.DatError, match="unsupported DataFormat"):
        bd.read_dat(p)


@pytest.mark.parametrize("drop", ["SourceChGain", "SourceChOffset"])
def test_missing_gain_or_offset_list_is_refused(tmp_path, drop):
    specs, sv = w.pack_layout([("Running", 1)])
    params = [p for p in w.signal_params(2) if f" {drop}=" not in p]
    p = w.write_dat(tmp_path / "g.dat", np.ones((2, 2)), states=specs, sv_len=sv, params=params)
    with pytest.raises(bd.DatError, match=f"missing list parameter {drop}"):
        bd.read_dat(p)


def test_gain_given_as_a_scalar_is_refused(tmp_path):
    specs, sv = w.pack_layout([("Running", 1)])
    params = [p for p in w.signal_params(1) if " SourceChGain=" not in p] + [
        "Source:Signal%20Properties float SourceChGain= 1 // not a list"]
    p = w.write_dat(tmp_path / "g.dat", np.ones((2, 1)), states=specs, sv_len=sv, params=params)
    with pytest.raises(bd.DatError, match="missing list parameter SourceChGain"):
        bd.read_dat(p)


@pytest.mark.parametrize("gains, offsets", [(["1muV"] * 7, None), (None, ["0"] * 7)])
def test_short_gain_or_offset_list_is_refused(tmp_path, gains, offsets):
    specs, sv = w.pack_layout([("Running", 1)])
    p = w.write_dat(tmp_path / "g.dat", np.ones((2, 8)), states=specs, sv_len=sv,
                    params=w.signal_params(8, gains, offsets))
    with pytest.raises(bd.DatError, match="values for 8 channels"):
        bd.read_dat(p)


def test_a_longer_gain_list_uses_the_first_entries_as_bci2000_does(tmp_path):
    specs, sv = w.pack_layout([("Running", 1)])
    p = w.write_dat(tmp_path / "g.dat", np.ones((2, 2)), states=specs, sv_len=sv,
                    params=w.signal_params(2, ["2muV", "3muV", "1000muV"], ["0", "0", "5"]))
    np.testing.assert_array_equal(bd.read_dat(p).signal_uv, [[2.0, 3.0], [2.0, 3.0]])


def test_mixed_unitless_and_united_gains_are_refused(tmp_path):
    # BCI2000 reads unitless gains as microvolts only when no channel has a unit; refuse the mix.
    with pytest.raises(bd.DatError, match="mixes gains"):
        bd.read_dat(simple_file(tmp_path / "m.dat", np.ones((2, 2)), gains=["2", "1muV"]))


@pytest.mark.parametrize("gain", ["abc", "1 muV muV", "", "1xV"])
def test_unreadable_gain_is_refused(tmp_path, gain):
    p = simple_file(tmp_path / "g.dat", np.ones((2, 1)), gains=[w.encode(gain)])
    with pytest.raises(bd.DatError, match="gain"):
        bd.read_dat(p)


@pytest.mark.parametrize("line", [
    "A 8 0 2 1",      # bits 17..24 of a 3-byte (24-bit) vector: one bit past the end
    "A 1 0 3 0",      # starts past the end
    "A 33 0 0 0",     # wider than 32 bits
    "A 0 0 0 0",      # zero width
    "A 1 0 0 8",      # bit location must be 0..7
    "A 1 0 -1 0",     # negative byte
])
def test_states_outside_the_state_vector_are_refused(tmp_path, line):
    p = w.write_dat(tmp_path / "s.dat", np.ones((2, 1)), sv_len=3, state_lines=[line], params=w.signal_params(1))
    with pytest.raises(bd.DatError, match="outside the state vector"):
        bd.read_dat(p)


def test_state_ending_exactly_at_the_vector_end_is_accepted(tmp_path):
    p = w.write_dat(tmp_path / "s.dat", np.ones((2, 1)), sv_len=3, params=w.signal_params(1),
                    states=[w.StateSpec("A", 8, 2, 0)], state_values={"A": np.array([255, 1], np.uint64)})
    np.testing.assert_array_equal(bd.read_dat(p).state("A"), [255, 1])


def test_malformed_headers_are_refused(tmp_path):
    with pytest.raises(bd.DatError, match="cannot read"):
        bd.read_dat(tmp_path / "missing.dat")
    (tmp_path / "e.dat").write_bytes(b"")
    with pytest.raises(bd.DatError, match="no header line"):
        bd.read_dat(tmp_path / "e.dat")
    p = w.write_dat(tmp_path / "s.dat", np.ones((2, 1)), sv_len=1, state_lines=["A 1 0 0"],
                    params=w.signal_params(1))
    with pytest.raises(bd.DatError, match="bad state definition"):
        bd.read_dat(p)
    p = simple_file(tmp_path / "long.dat", np.ones((0, 1)), header_len_delta=1000)
    with pytest.raises(bd.DatError, match="out of range"):
        bd.read_dat(p)
    (tmp_path / "f.dat").write_bytes(b"BCI2000V= 1.1 HeaderLen= 40 SourceCh= x StatevectorLen= 1 DataFormat= int16\r\n")
    with pytest.raises(bd.DatError, match="malformed first line"):
        bd.read_dat(tmp_path / "f.dat")
    p = simple_file(tmp_path / "zero.dat", np.ones((2, 1)))
    p.write_bytes(p.read_bytes().replace(b"SourceCh= 1 ", b"SourceCh= 0 ", 1))
    with pytest.raises(bd.DatError, match="out of range"):
        bd.read_dat(p)


# ---------------------------------------------------------------------------
# against BCI2000's own reader
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bcistream():
    sys.path.insert(0, str(BCI2000_ROOT / "tools" / "python"))
    try:
        from BCI2000Tools.FileReader import bcistream as bs
    except Exception as exc:  # noqa: BLE001 - any import failure means "not available here"
        pytest.skip(f"BCI2000Tools.FileReader not importable: {exc}")
    return bs


def official(bs, path):
    b = bs(str(path))
    try:
        sig, states = b.decode()
        raw, _ = b.read("all", apply_gains=False)
    finally:
        b.close()
    return b, np.asarray(sig), np.asarray(raw), {k: np.asarray(v).ravel() for k, v in states.items()}


def assert_same_as_official(bs, path):
    d = bd.read_dat(path)
    b, sig, raw, states = official(bs, path)
    # official: float32 channels x samples. The test files use values for which float32 arithmetic is exact.
    np.testing.assert_array_equal(d.signal_uv, sig.T.astype(np.float64))
    np.testing.assert_array_equal(d.signal_raw.astype(np.float64), raw.T.astype(np.float64))
    assert set(states) == set(d.states)
    for k, v in states.items():
        np.testing.assert_array_equal(d.states[k].astype(np.int64), v.astype(np.int64), err_msg=k)
    return d, b


# Not int32: BCI2000Tools.FileReader sizes a sample with struct.calcsize('l'), which is the native
# 8 bytes on 64-bit macOS/Linux, while it unpacks with '<l' (4 bytes), so it misreads every int32
# file there. That is its bug; this reader's int32 path is checked byte-exactly above.
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("fmt", ["int16", "float32"])
def test_odd_state_layouts_match_bci2000s_reader(tmp_path, bcistream, seed, fmt):
    write_odd(tmp_path / "s.dat", n=60, seed=seed, data_format=fmt)
    assert_same_as_official(bcistream, tmp_path / "s.dat")


def test_speller_recording_matches_bci2000s_reader(tmp_path, bcistream):
    sess = w.build_session(w.checkerboard_selections(2, 1, seed=3))
    rng = np.random.default_rng(3)
    sig = np.round(rng.normal(0, 20, size=(sess.n, 8)) * 64) / 64      # float32-exact, dyadic
    p = w.write_speller_dat(tmp_path / "rec.dat", sess, sig, lost=[(900, 905)])
    d, b = assert_same_as_official(bcistream, p)
    assert b.params["ChannelNames"] == d.param_list("ChannelNames") == list(w.EEG)
    assert b.samplingrate() == d.param_number("SamplingRate", "Hz") == 250.0
    assert b.params["SubjectName"] == d.param_str("SubjectName")


def test_gains_and_offsets_match_bci2000s_reader(tmp_path, bcistream):
    rng = np.random.default_rng(5)
    raw = rng.integers(-8000, 8000, size=(50, 4)).astype(np.float64)
    p = simple_file(tmp_path / "g.dat", raw, data_format="int16",
                    gains=["0.125uV", "1mV", "2muV", "0.5muV"], offsets=["0", "3", "-8", "1"])
    assert_same_as_official(bcistream, p)


@pytest.mark.skipif(not (SAMPLEFILES / "eeg3_1.dat").is_file(), reason="no BCI2000 sample file eeg3_1.dat")
def test_a_real_bci2000_recording_relabelled_as_1_1_matches_bci2000s_reader(tmp_path, bcistream):
    # BCI2000 writes int16 data in format 1.0 (no BCI2000V field, BCI2000OutputFormat.cpp), which this
    # reader refuses by design. Rewriting only the first line to 1.1 / int16 keeps every real state
    # layout and parameter of the file, so both readers can be compared on genuine BCI2000 output.
    src = (SAMPLEFILES / "eeg3_1.dat").read_bytes()
    nl = src.index(b"\n")
    fields = dict(re.findall(rb"(\w+)=\s*(\S+)", src[:nl]))
    old_len = int(fields[b"HeaderLen"])
    rest = src[nl:old_len]
    tail = f" SourceCh= {int(fields[b'SourceCh'])} StatevectorLen= {int(fields[b'StatevectorLen'])} DataFormat= int16"
    prefix = "BCI2000V= 1.1 HeaderLen= "
    new_len = len(prefix) + 6 + len(tail) + len(rest)
    first = f"{prefix}{new_len:6d}{tail}".encode()
    p = tmp_path / "eeg3_1_v11.dat"
    p.write_bytes(first + rest + src[old_len:])
    d = bd.read_dat(p)
    b, sig, raw, states = official(bcistream, p)
    np.testing.assert_array_equal(d.signal_raw.astype(np.float64), raw.T.astype(np.float64))
    # real gains are arbitrary decimals: BCI2000's reader works in float32, this one in float64
    np.testing.assert_allclose(d.signal_uv, sig.T.astype(np.float64), rtol=1e-6, atol=1e-6)
    assert set(states) == set(d.states) and len(states) > 10
    for k, v in states.items():
        np.testing.assert_array_equal(d.states[k].astype(np.int64), v.astype(np.int64), err_msg=k)
    assert d.param_number("SamplingRate", "Hz") == b.samplingrate()
    assert d.param_list("ChannelNames") == list(b.params["ChannelNames"])
