"""
The C++ Unicorn protocol (bci2000/UnicornSerialSource/UnicornProtocol.cpp, used by the BCI2000
source module) against the Python reference (collection/unicorn_protocol.py, tested on its own in
test_unicorn_protocol.py). Both get the same bytes in the same read sizes; the C++ output has to
match sample for sample, plus the gap filling that only the BCI2000 path needs.

Compiles a small driver with the system C++ compiler into a temp dir; skips if there is none.
"""

import json
import random
import shutil
import subprocess

import pytest

from pipeline.manifest import REPO_ROOT

import sys
sys.path.insert(0, str(REPO_ROOT / "collection"))
import unicorn_protocol as up  # noqa: E402

SRC = REPO_ROOT / "bci2000" / "UnicornSerialSource"
EXAMPLE = bytes([
    0xC0, 0x00, 0x0F, 0x00, 0x9F, 0xAF, 0x00, 0x9F, 0xD4, 0x00, 0xA0, 0x40, 0x00, 0x9F, 0x43, 0x00, 0x9F, 0x9A,
    0x00, 0x9F, 0xE3, 0x00, 0x9F, 0x85, 0x00, 0x9F, 0xBB, 0x2E, 0xF6, 0xE9, 0x02, 0x8D, 0xF2, 0xF3, 0xFF, 0xEF,
    0xFF, 0x23, 0x00, 0xB0, 0x00, 0x00, 0x00, 0x0D, 0x0A])


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("ucli") / "unicorn_protocol_cli"
    subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O1",
                    str(SRC / "UnicornProtocol.cpp"), str(SRC / "tests" / "unicorn_protocol_cli.cpp"),
                    "-o", str(exe)], check=True, capture_output=True)
    return exe


def run_cpp(exe, stream, chunks, *, ack=True, max_fill=2500):
    args = [str(exe)] + ([] if ack else ["--no-ack"]) + ["--max-fill", str(max_fill)] + [str(c) for c in chunks]
    lines = [json.loads(l) for l in subprocess.run(args, input=stream, capture_output=True, check=True)
             .stdout.decode().splitlines()]
    return [l for l in lines if "counter" in l], lines[-1]["stats"]


def run_python(stream, chunks, *, ack=True, max_fill=2500):
    """The reference: Python framer, decoder and tracker, plus the gap rule GapFiller implements."""
    framer, tracker = up.PacketFramer(expect_ack=ack), up.CounterTracker()
    out, last, filled, refused, pos, c = [], None, 0, 0, 0, 0
    while pos < len(stream) and not refused:
        n = max(chunks[c % len(chunks)], 1)
        c += 1
        for payload in framer.feed(stream[pos:pos + n]):
            s = up.decode_payload(payload)
            step = tracker.update(s.counter)
            if step.kind == "duplicate":
                continue
            if step.kind == "gap":
                if step.missing > max_fill:
                    refused = step.missing
                    break
                out += [dict(sample_dict(last), counter=last.counter + k, filled=1)
                        for k in range(1, step.missing + 1)]
                filled += step.missing
            out.append(sample_dict(s))
            last = s
        pos += n
    stats = {"ack": int(framer.ack_received), "frames": framer.frames, "bytes_discarded": framer.bytes_discarded,
             "resyncs": framer.resyncs, "samples": tracker.n_samples, "missing": tracker.missing,
             "duplicates": tracker.duplicates, "resets": tracker.resets, "gaps": tracker.n_gaps,
             "longest_gap": tracker.longest_gap, "filled": filled, "refused_gap": refused}
    return out, stats


def sample_dict(s):
    return {"counter": s.counter, "filled": 0, "battery": s.battery_pct, "eeg": list(s.eeg_uv),
            "acc": [round(v / up.ACC_G_PER_COUNT) for v in s.acc_g], "gyr": list(s.gyr_counts),
            "synthetic": int(s.is_synthetic)}


def assert_same(exe, stream, chunks, **kw):
    cpp, cpp_stats = run_cpp(exe, stream, chunks, **kw)
    py, py_stats = run_python(stream, chunks, **kw)
    assert cpp_stats == py_stats
    assert len(cpp) == len(py)
    for a, b in zip(cpp, py):
        assert a == b                    # exact: same integer counts times the same double constant
    return cpp, cpp_stats


def frame(counter, rng=None, **kw):
    eeg = [rng.randrange(-(1 << 23), 1 << 23) for _ in range(8)] if rng else [1000 + counter % 5000] * 8
    if rng and "acc_counts" not in kw:
        kw["acc_counts"] = tuple(rng.randrange(-32768, 32768) for _ in range(3))
        kw["gyr_counts"] = tuple(rng.randrange(-32768, 32768) for _ in range(3))
        kw["battery_nibble"] = rng.randrange(256)
    return up.encode_payload(eeg, counter, **kw)


def test_the_documents_worked_example(cli):
    cpp, stats = assert_same(cli, up.ACK + EXAMPLE, [7])
    assert cpp[0]["counter"] == 176 and cpp[0]["eeg"][0] == pytest.approx(3654.87, abs=0.006)
    assert stats["ack"] == 1


@pytest.mark.parametrize("seed", range(12))
def test_random_streams_with_gaps_duplicates_resets_and_garbage(cli, seed):
    rng = random.Random(seed)
    parts, counter = [up.ACK], rng.randrange(-5, 1 << 20)
    for _ in range(300):
        r = rng.random()
        if r < 0.05:
            counter += rng.randrange(2, 60)                            # radio loss
        elif r < 0.07:
            parts.append(frame(counter - 1, rng))                      # duplicate of the previous sample
        elif r < 0.08:
            counter -= rng.randrange(1, 500)                           # reset / corrupt counter
        elif r < 0.12:
            parts.append(bytes(rng.randrange(256) for _ in range(rng.randrange(1, 30))))   # garbage
        elif r < 0.14:
            parts.append(up.HEADER + bytes(rng.randrange(256) for _ in range(10)))      # header lookalike
        parts.append(frame(counter, rng))
        counter += 1
    stream = b"".join(parts)
    chunks = [rng.choice([1, 2, 3, 7, 44, 45, 46, 90, 200, 4096]) for _ in range(50)]
    cpp, stats = assert_same(cli, stream, chunks)
    assert stats["frames"] > 250


def test_filled_samples_keep_every_slot_and_hold_the_last_real_values(cli):
    stream = up.ACK + frame(10) + frame(11) + frame(15) + frame(16)
    cpp, stats = assert_same(cli, stream, [4096])
    assert [s["counter"] for s in cpp] == list(range(10, 17))
    assert [s["filled"] for s in cpp] == [0, 0, 1, 1, 1, 0, 0]
    assert all(s["eeg"] == cpp[1]["eeg"] for s in cpp[2:5])
    assert (stats["missing"], stats["filled"]) == (3, 3)


def test_a_gap_beyond_max_fill_is_refused_not_papered_over(cli):
    stream = up.ACK + frame(0) + frame(1) + frame(400) + frame(401)
    cpp, stats = assert_same(cli, stream, [4096], max_fill=250)
    assert [s["counter"] for s in cpp] == [0, 1] and stats["refused_gap"] == 398


def test_no_ack_mode_and_byte_by_byte(cli):
    assert_same(cli, frame(0) + frame(1) + frame(2), [1], ack=False)
    assert_same(cli, up.ACK + frame(0) + frame(1), [1])


def test_tricky_frames_and_the_synthetic_marker(cli):
    tricky = up.encode_payload([0xC00000 - (1 << 24), 0xC000 >> 0, 0xC000, 0, 0, 0, 0, 0], 9)
    fake = up.encode_payload([0] * 8, 10, gyr_counts=up.SYNTHETIC_GYRO_COUNTS)
    cpp, _ = assert_same(cli, b"\x99\x99\x99" + tricky + fake, [5, 45])
    assert [s["synthetic"] for s in cpp] == [0, 1]
