"""
pipeline/bci2000_recording.py on synthetic CheckerboardSpeller-shaped .dat files written by
tests/bci2000_dat_writer.py: flash onsets (including flashes with no gap between them), selections,
labels and groups, every refusal of check_recording/flash_events, and NaN masking of the samples the
source filled in for lost packets. Files go to tmp_path only.
"""

import numpy as np
import pytest

import bci2000_dat_writer as w
from pipeline import bci2000_recording as br
from pipeline.bci2000_dat import read_dat
from pipeline.constants import EEG_CHANNELS


def hand_selections():
    """Three selections; off = 0 makes the next flash start on the very next sample (code k -> k+1)."""
    return [
        {"attended": 5, "flashes": [
            ((5, 1, 2, 3, 4, 6), 0),            # target, then immediately ...
            ((7, 8, 9, 10, 11, 12), 15),        # ... a non-target with no gap in between
            ((20, 5, 21, 22, 23, 24), 0),       # target (cued char not first in the group)
            ((30, 31, 32, 33, 34, 35), 0),      # three adjacent flashes in a row
            ((40, 41, 42, 43), 0),              # a short group (FlashChar5/6 = 0)
            ((44, 45, 46, 47, 48, 49), 15)]},
        {"attended": 71, "flashes": [((0, 1, 2, 3, 4, 71), 15), ((60, 61, 62, 63, 64, 65), 0)]},
        {"attended": 0, "flashes": [((0, 70, 69, 68, 67, 66), 15), ((1, 2, 3, 4, 5, 6), 15)]},
    ]


def write(tmp_path, session=None, name="rec.dat", **kw):
    session = session or w.build_session(hand_selections())
    rng = np.random.default_rng(0)
    sig = np.round(rng.normal(0, 10, size=(session.n, 8)) * 64) / 64
    return w.write_speller_dat(tmp_path / name, session, sig, **kw), session


def as_tuples(flashes):
    return [(f.selection, f.code, f.onset, f.is_target, f.group, f.attended) for f in flashes]


def expected_tuples(session):
    return [(f["selection"], f["code"], f["onset"], f["is_target"], f["group"], f["attended"])
            for f in session.flashes]


def test_hand_built_flashes_are_found_exactly(tmp_path):
    path, session = write(tmp_path)
    flashes = br.flash_events(read_dat(path))
    assert as_tuples(flashes) == expected_tuples(session)
    # the adjacent ones really are adjacent: onset k+1 = onset k + 16 samples, no zero code between
    onsets = [f.onset for f in flashes]
    assert onsets[1] - onsets[0] == 16 and onsets[3] - onsets[2] == 16 and onsets[4] - onsets[3] == 16
    assert [f.selection for f in flashes] == [0] * 6 + [1] * 2 + [2] * 2
    assert [f.code for f in flashes] == [1, 2, 3, 4, 5, 6, 1, 2, 1, 2]
    assert [f.is_target for f in flashes] == [True, False, True, False, False, False, True, False, True, False]
    assert flashes[4].group == (40, 41, 42, 43)
    assert {f.attended for f in flashes} == {5, 71, 0}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_checkerboard_recording_flashes_selections_labels_groups(tmp_path, seed):
    sels = w.checkerboard_selections(3, n_reps=2, seed=seed)
    session = w.build_session(sels)
    path, _ = write(tmp_path, session)
    flashes = br.flash_events(read_dat(path))
    assert len(flashes) == 3 * 48
    assert as_tuples(flashes) == expected_tuples(session)
    for s, sel in enumerate(sels):
        mine = [f for f in flashes if f.selection == s]
        assert [f.group for f in mine] == [g for g, _ in sel["flashes"]]
        assert [f.code for f in mine] == list(range(1, 49))
        assert sum(f.is_target for f in mine) == 4        # each character: 1 row + 1 column per repetition
        assert all(f.attended == sel["attended"] for f in mine)
        assert all(f.onset > session.selection_starts[s] for f in mine)


def test_all_adjacent_flashes(tmp_path):
    # every flash immediately followed by the next: StimulusCode never returns to 0 inside a selection
    session = w.build_session(w.checkerboard_selections(2, seed=4, off=0))
    path, _ = write(tmp_path, session)
    flashes = br.flash_events(read_dat(path))
    assert as_tuples(flashes) == expected_tuples(session)
    assert np.all(np.diff([f.onset for f in flashes if f.selection == 0]) == 16)


def test_selection_counts_entries_into_phase_1(tmp_path):
    session = w.build_session(hand_selections())
    st = session.states
    # phase 1 -> 2 -> 3 -> 1 ...; a stray 1 -> 1 continuation must not count twice, a 3 -> 1 must count
    ph = st["PhaseInSequence"]
    entries = np.flatnonzero((ph == 1) & (np.r_[0, ph[:-1]] != 1))
    assert list(entries) == session.selection_starts
    path, _ = write(tmp_path, session)
    assert sorted({f.selection for f in br.flash_events(read_dat(path))}) == [0, 1, 2]


# ---------------------------------------------------------------------------
# check_recording refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", br.REQUIRED_STATES)
def test_missing_required_state_is_refused(tmp_path, state):
    path, _ = write(tmp_path, omit_states=(state,))
    dat = read_dat(path)
    with pytest.raises(br.RecordingError, match=f"no state.*{state}"):
        br.check_recording(dat)
    with pytest.raises(br.RecordingError, match="CheckerboardSpeller"):
        br.flash_events(dat)


@pytest.mark.parametrize("names", [
    ("C3", "Fz", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"),            # two swapped
    ("Fz", "F3", "C4", "Cz", "Pz", "P3", "O2", "O1"),              # the reference paper's montage
    ("1", "2", "3", "4", "5", "6", "7", "8"),                      # generic labels
    ("fz", "c3", "cz", "c4", "pz", "po7", "oz", "po8"),            # case matters
])
def test_wrong_channel_names_are_refused(tmp_path, names):
    path, _ = write(tmp_path, channel_names=names)
    with pytest.raises(br.RecordingError, match="ChannelNames"):
        br.check_recording(read_dat(path))
    with pytest.raises(br.RecordingError, match="ChannelNames"):
        br.flash_events(read_dat(path))


@pytest.mark.parametrize("names", [EEG_CHANNELS[:7], (*EEG_CHANNELS, "EXG")])
def test_channel_names_list_of_the_wrong_length_is_refused(tmp_path, names):
    path, _ = write(tmp_path, channel_names=names)       # 8 signal channels, 7 or 9 names
    with pytest.raises(br.RecordingError, match="ChannelNames"):
        br.check_recording(read_dat(path))


@pytest.mark.parametrize("missing", ["ChannelNames", "SamplingRate"])
def test_missing_channel_names_or_sampling_rate_is_refused(tmp_path, missing):
    path, _ = write(tmp_path, omit_params=(missing,))
    with pytest.raises(br.RecordingError, match=missing):
        br.check_recording(read_dat(path))


@pytest.mark.parametrize("rate, ok", [("250Hz", True), ("250", True), ("250.0Hz", True),
                                      ("256Hz", False), ("500Hz", False), ("125", False)])
def test_sampling_rate_must_be_250_hz(tmp_path, rate, ok):
    path, _ = write(tmp_path, sampling_rate=rate)
    if ok:
        br.check_recording(read_dat(path))
    else:
        with pytest.raises(br.RecordingError, match="SamplingRate"):
            br.check_recording(read_dat(path))


def test_sampling_rate_in_another_unit_is_refused(tmp_path):
    path, _ = write(tmp_path, sampling_rate="250ms")
    with pytest.raises(br.RecordingError, match="SamplingRate"):
        br.check_recording(read_dat(path))


@pytest.mark.parametrize("param", br.SOURCE_FILTER_PARAMS)
@pytest.mark.parametrize("value, ok", [("0", True), ("", True), ("0.0", True), ("1", False), ("3", False),
                                       ("abc", False)])
def test_source_filters_must_be_off(tmp_path, param, value, ok):
    path, _ = write(tmp_path, extra_params=[w.scalar(param, value, "int", "Source:Filtering")])
    if ok:
        br.check_recording(read_dat(path))
        assert len(br.flash_events(read_dat(path))) == 10
    else:
        with pytest.raises(br.RecordingError, match=f"{param} = {value}"):
            br.check_recording(read_dat(path))


def test_absent_filter_parameters_are_fine(tmp_path):
    path, _ = write(tmp_path)
    dat = read_dat(path)
    assert not any(p in dat.params for p in br.SOURCE_FILTER_PARAMS)
    br.check_recording(dat)


# ---------------------------------------------------------------------------
# flash_events refusals
# ---------------------------------------------------------------------------

def test_free_mode_without_cues_is_refused(tmp_path):
    sels = hand_selections()
    for s in sels:
        s["attended"] = None
    path, _ = write(tmp_path, w.build_session(sels))
    with pytest.raises(br.RecordingError, match="copy mode"):
        br.flash_events(read_dat(path))


def test_cue_missing_in_one_selection_is_refused(tmp_path):
    sels = hand_selections()
    sels[2]["attended"] = None
    path, _ = write(tmp_path, w.build_session(sels))
    with pytest.raises(br.RecordingError, match="no cued character"):
        br.flash_events(read_dat(path))


def test_flash_before_any_selection_is_refused(tmp_path):
    session = w.build_session(hand_selections())
    ph = session.states["PhaseInSequence"]
    first = session.selection_starts[0]
    ph[first:first + 250] = 0          # the first pre-sequence phase never happened
    path, _ = write(tmp_path, session)
    with pytest.raises(br.RecordingError, match="before any selection"):
        br.flash_events(read_dat(path))


@pytest.mark.parametrize("flash, stype", [(0, 0), (1, 1), (6, 0), (7, 1)])
def test_label_that_contradicts_the_group_is_refused(tmp_path, flash, stype):
    session = w.build_session(hand_selections())
    f = session.flashes[flash]
    assert f["is_target"] == (not stype)
    session.states["StimulusType"][f["onset"]:f["onset"] + 16] = stype
    path, _ = write(tmp_path, session)
    with pytest.raises(br.RecordingError, match="disagree"):
        br.flash_events(read_dat(path))


def test_flash_without_flash_char_states_is_refused(tmp_path):
    session = w.build_session(hand_selections())
    f = session.flashes[3]
    for s in br.FLASH_STATES:
        session.states[s][f["onset"]:f["onset"] + 16] = 0
    path, _ = write(tmp_path, session)
    with pytest.raises(br.RecordingError, match="no FlashChar"):
        br.flash_events(read_dat(path))


# ---------------------------------------------------------------------------
# lost samples
# ---------------------------------------------------------------------------

def test_validation_zero_samples_become_nan(tmp_path):
    lost = [(400, 403), (1000, 1001), (1500, 1530)]
    path, session = write(tmp_path, lost=lost)
    dat = read_dat(path)
    before = dat.signal_uv.copy()
    x = br.eeg_microvolts(dat)
    mask = np.zeros(session.n, bool)
    for a, b in lost:
        mask[a:b] = True
    assert x.shape == (session.n, 8) and x.dtype == np.float64
    assert np.array_equal(np.all(np.isnan(x), axis=1), mask)
    assert not np.any(np.isnan(x[~mask]))
    np.testing.assert_array_equal(x[~mask], dat.signal_uv[~mask])
    np.testing.assert_array_equal(dat.signal_uv, before)         # the DatFile itself is untouched
    assert not np.any(np.isnan(dat.signal_uv))


def test_validation_nonzero_values_are_valid(tmp_path):
    # Validation is a 32-bit state: anything but 0 is a real sample (BCI2000 rounds 1.0 to 1)
    session = w.build_session(hand_selections())
    session.states["Validation"][:] = 1
    session.states["Validation"][100:110] = 0xFFFFFFFF
    path, _ = write(tmp_path, session)
    assert not np.any(np.isnan(br.eeg_microvolts(read_dat(path))))


def test_without_a_validation_state_nothing_is_masked(tmp_path):
    path, _ = write(tmp_path, omit_states=("Validation",))
    dat = read_dat(path)
    np.testing.assert_array_equal(br.eeg_microvolts(dat), dat.signal_uv)


def test_lost_samples_do_not_change_flash_detection(tmp_path):
    session = w.build_session(hand_selections())
    f = session.flashes[2]
    path, _ = write(tmp_path, session, lost=[(f["onset"] - 2, f["onset"] + 5)])
    assert as_tuples(br.flash_events(read_dat(path))) == expected_tuples(session)


def test_a_counter_break_invalidates_the_first_sample_after_it(tmp_path):
    # A device counter that does not advance by exactly 1 (a reset, or blocks BCI2000 dropped) means
    # samples may be missing there; the online filter rejects windows touching that sample, and so must
    # the training path.
    session = w.build_session(hand_selections())
    session.states["Counter"][700:] += 5
    session.states["Counter"][900:] = np.arange(session.n - 900)       # and a reset to 0
    path, _ = write(tmp_path, session)
    dat = read_dat(path)
    assert list(br.counter_breaks(dat)) == [700, 900]
    bad = br.invalid_samples(dat)
    assert list(np.flatnonzero(bad)) == [700, 900]
    x = br.eeg_microvolts(dat)
    assert np.isnan(x[[700, 900]]).all() and np.isfinite(np.delete(x, [700, 900], axis=0)).all()


def test_a_continuous_counter_and_no_counter_state_have_no_breaks(tmp_path):
    path, _ = write(tmp_path)
    assert len(br.counter_breaks(read_dat(path))) == 0
    path, _ = write(tmp_path, name="nc.dat", omit_states=("Counter",))
    assert len(br.counter_breaks(read_dat(path))) == 0
