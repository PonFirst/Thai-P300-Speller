"""
Unicorn protocol tests: no hardware, no LSL. The decoder is checked against the worked example in
g.tec's own Bluetooth protocol document, then against the cases a public decoder has been wrong
on (sign extension, the counter's byte order), then the framer, counter tracker and clock.
"""

import random
import sys

import numpy as np
import pytest

from pipeline.manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "collection"))
import unicorn_protocol as up  # noqa: E402

# UnicornBluetoothProtocol.pdf section 1.5, byte for byte.
EXAMPLE = bytes([
    0xC0, 0x00, 0x0F, 0x00, 0x9F, 0xAF, 0x00, 0x9F, 0xD4, 0x00, 0xA0, 0x40, 0x00, 0x9F, 0x43, 0x00, 0x9F, 0x9A,
    0x00, 0x9F, 0xE3, 0x00, 0x9F, 0x85, 0x00, 0x9F, 0xBB, 0x2E, 0xF6, 0xE9, 0x02, 0x8D, 0xF2, 0xF3, 0xFF, 0xEF,
    0xFF, 0x23, 0x00, 0xB0, 0x00, 0x00, 0x00, 0x0D, 0x0A])


def frame(counter, eeg=None, **kw):
    return up.encode_payload(eeg if eeg is not None else [1000 + counter % 5000] * 8, counter, **kw)


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------

def test_example_is_45_bytes_and_matches_the_documents_converted_values():
    assert len(EXAMPLE) == up.PAYLOAD_LEN == 45
    s = up.decode_payload(EXAMPLE)
    assert s.battery_pct == pytest.approx(100.0)
    assert s.eeg_uv == pytest.approx((3654.87, 3658.18, 3667.83, 3645.21, 3652.99, 3659.52, 3651.11, 3655.94),
                                     abs=0.006)
    assert s.acc_g == pytest.approx((-0.614, 0.182, -0.841), abs=0.001)
    assert s.gyr_dps == pytest.approx((-0.397, -0.519, 1.068), abs=0.001)
    assert s.counter == 176


def test_negative_eeg_values_are_sign_extended():
    counts = [-1, -8388608, 8388607, 0, -256, 255, -65536, 1]
    payload = frame(5, counts)
    assert payload[3:6] == b"\xff\xff\xff"                   # -1 as 24-bit two's complement
    s = up.decode_payload(payload)
    assert s.eeg_uv == pytest.approx([c * up.EEG_UV_PER_COUNT for c in counts])
    assert s.eeg_uv[0] < 0 and s.eeg_uv[1] < 0 < s.eeg_uv[2]


def test_counter_uses_all_four_bytes_little_endian():
    # the document's own formula prints Counter[1] three times; its C example, and this decoder, don't
    assert up.decode_payload(frame(0x01020304)).counter == 0x01020304
    assert up.decode_payload(frame(0x00010000)).counter == 65536
    assert up.decode_payload(frame(-1)).counter == -1


def test_eeg_is_big_endian_but_acc_gyr_are_little_endian():
    p = up.encode_payload([0x010203] + [0] * 7, 0, acc_counts=(0x0102, 0, 0), gyr_counts=(0, 0x0304, 0))
    assert p[3:6] == b"\x01\x02\x03" and p[27:29] == b"\x02\x01" and p[35:37] == b"\x04\x03"
    s = up.decode_payload(p)
    assert s.eeg_uv[0] == pytest.approx(0x010203 * up.EEG_UV_PER_COUNT)
    assert s.acc_g[0] == pytest.approx(0x0102 / 4096) and s.gyr_dps[1] == pytest.approx(0x0304 / 32.8)


def test_battery_is_the_low_nibble_as_percent_of_15():
    assert up.decode_payload(frame(0, battery_nibble=0)).battery_pct == 0
    assert up.decode_payload(frame(0, battery_nibble=15)).battery_pct == pytest.approx(100)
    assert up.decode_payload(frame(0, battery_nibble=0xF5)).battery_pct == pytest.approx(100 * 5 / 15)


def test_scale_constants_come_from_the_document():
    assert up.EEG_UV_PER_COUNT == pytest.approx(4_500_000 / 50_331_642)
    assert up.UNICORN_SFREQ == 250.0 and up.START_ACQUISITION == b"\x61\x7c\x87"
    assert up.STOP_ACQUISITION == b"\x63\x5c\xc5" and up.ACK == b"\x00\x00\x00"


@pytest.mark.parametrize("bad,match", [
    (EXAMPLE[:-1], "45 bytes"), (EXAMPLE + b"\x00", "45 bytes"), (b"", "45 bytes"),
    (b"\xc0\x01" + EXAMPLE[2:], "header"), (EXAMPLE[:-2] + b"\x0d\x0b", "footer"),
])
def test_bad_payloads_are_refused(bad, match):
    with pytest.raises(up.PayloadError, match=match):
        up.decode_payload(bad)


@pytest.mark.parametrize("seed", range(5))
def test_encode_decode_roundtrip_on_random_raw_values(seed):
    rng = random.Random(seed)
    eeg = [rng.randrange(-(1 << 23), 1 << 23) for _ in range(8)]
    acc = [rng.randrange(-32768, 32768) for _ in range(3)]
    gyr = [rng.randrange(-32768, 32768) for _ in range(3)]
    counter = rng.randrange(0, 1 << 31)
    s = up.decode_payload(up.encode_payload(eeg, counter, acc_counts=acc, gyr_counts=gyr))
    assert s.counter == counter and list(s.gyr_counts) == gyr
    assert s.eeg_uv == pytest.approx([c * up.EEG_UV_PER_COUNT for c in eeg])
    assert s.acc_g == pytest.approx([c / 4096 for c in acc])


def test_encoder_refuses_out_of_range_input():
    with pytest.raises(ValueError):
        up.encode_payload([1 << 23] + [0] * 7, 0)
    with pytest.raises(ValueError):
        up.encode_payload([0] * 7, 0)


def test_synthetic_marker_is_recognised_and_real_data_is_not():
    fake = up.decode_payload(frame(0, gyr_counts=up.SYNTHETIC_GYRO_COUNTS))
    assert fake.is_synthetic and not up.decode_payload(EXAMPLE).is_synthetic


# ---------------------------------------------------------------------------
# framer
# ---------------------------------------------------------------------------

def test_ack_then_frames_in_one_read():
    f = up.PacketFramer()
    out = f.feed(up.ACK + frame(0) + frame(1) + frame(2))
    assert out == [frame(0), frame(1), frame(2)] and f.ack_received
    assert (f.frames, f.bytes_discarded, f.resyncs) == (3, 0, 0)


def test_a_device_that_skips_the_ack_still_works():
    f = up.PacketFramer()
    assert f.feed(frame(0)) == [frame(0)] and not f.ack_received and f.bytes_discarded == 0


def test_ack_split_across_reads_and_an_empty_first_read():
    f = up.PacketFramer()
    assert f.feed(b"") == [] and f.feed(b"\x00") == [] and f.feed(b"\x00") == []
    assert f.feed(b"\x00" + frame(0)) == [frame(0)] and f.ack_received and f.bytes_discarded == 0


@pytest.mark.parametrize("seed", range(6))
def test_any_chunking_gives_the_same_frames(seed):
    stream = up.ACK + b"".join(frame(i) for i in range(40))
    rng = random.Random(seed)
    f, out, pos = up.PacketFramer(), [], 0
    while pos < len(stream):
        n = rng.choice([1, 2, 3, 7, 44, 45, 46, 90, 200])
        out += f.feed(stream[pos:pos + n])
        pos += n
    assert out == [frame(i) for i in range(40)] and f.bytes_discarded == 0 and f.resyncs == 0


def test_byte_by_byte():
    f, out = up.PacketFramer(), []
    for b in up.ACK + frame(0) + frame(1):
        out += f.feed(bytes([b]))
    assert out == [frame(0), frame(1)]


def test_garbage_between_frames_is_counted_and_recovered_from():
    f = up.PacketFramer()
    out = f.feed(up.ACK + frame(0) + b"\x11" * 7 + frame(1) + b"\x22\x33" + frame(2))
    assert out == [frame(0), frame(1), frame(2)]
    assert f.bytes_discarded == 9 and f.resyncs == 2


def test_one_run_of_garbage_split_over_reads_is_one_resync():
    f = up.PacketFramer()
    assert f.feed(up.ACK + frame(0) + b"\x11" * 5) == [frame(0)]
    assert f.feed(b"\x22" * 5) == []
    assert f.feed(frame(1)) == [frame(1)]
    assert f.bytes_discarded == 10 and f.resyncs == 1


def test_a_header_lookalike_without_a_footer_is_skipped_not_believed():
    lookalike = up.HEADER + b"\x11" * 10
    f = up.PacketFramer()
    assert f.feed(up.ACK + lookalike + frame(7)) == [frame(7)]
    assert f.bytes_discarded == len(lookalike) and f.resyncs == 1


def test_eeg_bytes_that_look_like_a_header_do_not_split_a_frame():
    tricky = up.encode_payload([0xC00000 - (1 << 24), 0x00C000 >> 0, 0xC000, 0, 0, 0, 0, 0], 9)
    assert up.HEADER in tricky[2:]
    f = up.PacketFramer()
    assert f.feed(up.ACK + tricky + frame(10)) == [tricky, frame(10)] and f.bytes_discarded == 0
    f = up.PacketFramer()
    assert f.feed(b"\x99\x99\x99" + tricky) == [tricky]           # recovering into a tricky frame


def test_a_split_header_and_a_partial_frame_are_held():
    f = up.PacketFramer()
    whole = frame(3)
    assert f.feed(up.ACK + whole[:1]) == [] and f.feed(whole[1:20]) == []
    assert f.feed(whole[20:]) == [whole] and f.bytes_discarded == 0


def test_a_run_of_pure_garbage_leaves_nothing_behind():
    f = up.PacketFramer()
    assert f.feed(up.ACK + b"\x11" * 300) == [] and f.bytes_discarded == 300 and f.resyncs == 1
    assert f.feed(frame(0)) == [frame(0)]


# ---------------------------------------------------------------------------
# counter tracker
# ---------------------------------------------------------------------------

def test_counter_tracker_finds_gaps_duplicates_and_resets():
    t = up.CounterTracker()
    kinds = [t.update(c).kind for c in (100, 101, 102, 106, 107, 107, 108, 50, 51)]
    assert kinds == ["first", "ok", "ok", "gap", "ok", "duplicate", "ok", "reset", "ok"]
    assert (t.missing, t.n_gaps, t.longest_gap, t.duplicates, t.resets) == (3, 1, 3, 1, 1)
    assert t.gaps == [(106, 3)] and t.n_samples == 9


def test_counter_tracker_reports_how_many_samples_a_gap_lost():
    t = up.CounterTracker()
    t.update(0)
    step = t.update(251)
    assert (step.kind, step.missing) == ("gap", 250)                     # one second of EEG
    t.update(300)
    assert t.missing == 250 + 48 and t.n_gaps == 2 and t.longest_gap == 250


def test_counter_tracker_caps_the_gap_list_but_not_the_totals():
    t = up.CounterTracker()
    c = 0
    for _ in range(t.MAX_GAPS_KEPT + 20):
        t.update(c)
        c += 3
    assert len(t.gaps) == t.MAX_GAPS_KEPT and t.n_gaps == t.MAX_GAPS_KEPT + 19
    assert t.missing == 2 * (t.MAX_GAPS_KEPT + 19)


# ---------------------------------------------------------------------------
# counter clock
# ---------------------------------------------------------------------------

FS = 250.0


def simulate(n, *, seed, base_delay=0.040, drift_ppm=0.0, interval=0.0163, drop=(), t0=1000.0):
    """
    Device sends counter k at device time k/FS * (1 + drift); the radio delivers in bursts on a
    connection-interval grid with a little extra delay. Returns [(counter, arrival, true_time)]
    where true_time = the moment that sample was taken plus the constant part of the delay.
    The interval is deliberately not a multiple of the 4 ms sample period: the radio's clock is
    independent of the headset's, and a commensurate grid would phase-lock the two.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        if k in drop:
            continue
        sent = t0 + k / FS * (1 + drift_ppm * 1e-6)
        slot = np.ceil((sent - t0) / interval) * interval + t0
        arrival = slot + base_delay + rng.exponential(0.002)
        out.append((k, arrival, sent + base_delay))
    return out


def run_clock(events, **kw):
    clock = up.CounterClock(**kw)
    stamped = {}
    for counter, arrival, _ in events:
        for item, ts in clock.push(counter, counter, arrival):
            stamped[item] = ts
    return clock, stamped


@pytest.mark.parametrize("seed", range(4))
def test_counter_stamps_are_strictly_increasing_and_far_less_jittery_than_arrival(seed):
    events = simulate(6000, seed=seed)
    clock, stamped = run_clock(events)
    counters = sorted(stamped)
    ts = np.array([stamped[c] for c in counters])
    assert len(counters) == len(events) and np.all(np.diff(ts) > 0)
    true = np.array([e[2] for e in events])
    err = ts - true
    arrival = np.array([e[1] for e in events])
    assert np.std(err) < 0.0006                                   # sub-millisecond
    assert abs(np.median(err)) < 0.003
    assert np.std(arrival - true) > 5 * np.std(err)               # stamping on arrival is >5x worse


def test_a_constant_link_delay_stays_a_constant_shift():
    for base in (0.010, 0.080):
        events = simulate(3000, seed=1, base_delay=base)
        _, stamped = run_clock(events)
        err = np.array([stamped[e[0]] - (e[2] - base) for e in events])   # against the send time
        assert np.std(err) < 0.0006 and abs(np.median(err) - base) < 0.003


def test_drift_between_device_and_host_clocks_is_tracked():
    events = simulate(150_000, seed=2, drift_ppm=100.0)             # 10 simulated minutes, 60 ms of drift
    clock, stamped = run_clock(events)
    ts = np.array([stamped[e[0]] for e in events])
    true = np.array([e[2] for e in events])
    assert np.all(np.diff(ts) > 0)
    assert np.max(np.abs(ts - true)[5000:]) < 0.004               # never more than 4 ms off after warm-up
    drift_estimate = (clock.offset - clock.initial_offset) / (events[-1][0] / FS) * 1e6
    assert drift_estimate == pytest.approx(100.0, abs=25.0)       # ppm, and negative-drift devices likewise


def test_a_device_clock_running_slow_is_tracked_too():
    events = simulate(60_000, seed=3, drift_ppm=-80.0)
    _, stamped = run_clock(events)
    err = np.array([stamped[e[0]] - e[2] for e in events])
    assert np.max(np.abs(err)[5000:]) < 0.004


def test_dropped_packets_leave_an_honest_timestamp_gap():
    events = simulate(3000, seed=4, drop=set(range(1000, 1025)))
    _, stamped = run_clock(events)
    assert stamped[1025] - stamped[999] == pytest.approx(26 / FS, abs=1e-9)   # 25 lost + 1 step


def test_samples_are_held_through_warmup_then_released_in_order():
    clock = up.CounterClock(warmup_s=2.0)
    events = simulate(1000, seed=5)
    released, first_release_at = [], None
    for counter, arrival, _ in events:
        out = clock.push(counter, counter, arrival)
        if out and first_release_at is None:
            first_release_at = counter
        released += [item for item, _ in out]
    assert first_release_at == 2 * int(FS) and released == list(range(1000))
    assert not clock.push(0, 0, 0.0) == []                        # after warm-up every push returns its sample


def test_a_long_stall_and_burst_does_not_break_monotonic_stamps():
    events = simulate(5000, seed=6)
    late = [(c, a + (0.6 if 2000 <= c < 2150 else 0.0), t) for c, a, t in events]   # 0.6 s radio stall
    _, stamped = run_clock(late)
    ts = np.array([stamped[c] for c in sorted(stamped)])
    assert np.all(np.diff(ts) > 0)
    true = np.array([e[2] for e in events])
    assert np.max(np.abs(ts - true)[600:]) < 0.004                # the stall did not move the clock


def test_delay_is_measured_against_the_best_packets_and_is_never_far_negative():
    events = simulate(6000, seed=7)
    clock, _ = run_clock(events)
    d = np.array([clock.delay_s(c, a) for c, a, _ in events[600:]])
    assert np.min(d) > -0.003 and np.percentile(d, 99) < 0.030
