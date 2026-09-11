import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

import board_schedule as bs  # noqa: E402
from pipeline.manifest import load_characters_file  # noqa: E402
from record_session import event_to_row_fields  # noqa: E402

CHARS = load_characters_file(REPO_ROOT / "collection" / "characters_th.json")


def _schedule(targets="กข", reps=2, flash=62, isi=62, **kwargs):
    return bs.build_session_schedule(CHARS, targets, n_repetitions=reps, flash_ms=flash, isi_ms=isi, **kwargs)


def _markers(steps):
    return [s.marker for s in steps if s.marker is not None]


def test_trial_structure_matches_the_psychopy_stimulus():
    markers = _markers(_schedule("กข", reps=2))
    per_trial = 1 + 24 * 2 + 1
    assert len(markers) == 2 * per_trial
    assert markers[0]["event"] == "trial_start" and markers[0]["trial_id"] == 0
    assert all(m["event"] == "flash" for m in markers[1:per_trial - 1])
    assert markers[per_trial - 1] == {"event": "trial_end", "trial_id": 0}
    assert markers[per_trial]["event"] == "trial_start" and markers[per_trial]["trial_id"] == 1


def test_marker_payloads_have_contract_keys_and_parse_in_the_recorder():
    by_event = {}
    for marker in _markers(_schedule("ก", reps=1)):
        by_event.setdefault(marker["event"], marker)
    assert set(by_event["trial_start"]) == {"event", "trial_id", "target_char_idx", "target_char"}
    assert set(by_event["flash"]) == {"event", "trial_id", "flash_idx", "group", "is_target"}
    assert set(by_event["trial_end"]) == {"event", "trial_id"}
    assert CHARS[by_event["trial_start"]["target_char_idx"]] == "ก" == by_event["trial_start"]["target_char"]
    for marker in by_event.values():
        assert event_to_row_fields(json.loads(json.dumps(marker)))["trial_id"] == 0


def test_timing_follows_cue_flash_isi_and_inter_selection():
    steps = _schedule("กข", reps=1, flash=62, isi=62, cue_ms=2000, inter_selection_ms=3500)
    cue_on = [s.at_s for s in steps if s.kind == bs.CUE_ON]
    flash_on = np.array([s.at_s for s in steps if s.kind == bs.FLASH_ON and s.trial_id == 0])
    flash_off = np.array([s.at_s for s in steps if s.kind == bs.FLASH_OFF and s.trial_id == 0])
    trial_end = [s.at_s for s in steps if s.kind == bs.TRIAL_END]
    assert cue_on[0] == 0.0
    assert flash_on[0] == pytest.approx(2.0)
    np.testing.assert_allclose(np.diff(flash_on), 0.124)
    np.testing.assert_allclose(flash_off - flash_on, 0.062)
    assert trial_end[0] == pytest.approx(2.0 + 24 * 0.124)
    assert cue_on[1] == pytest.approx(trial_end[0] + 3.5)
    assert steps[-1].kind == bs.DONE and steps[-1].at_s == pytest.approx(trial_end[1] + 3.5)
    assert all(a.at_s <= b.at_s for a, b in zip(steps, steps[1:]))


def test_every_character_flashes_twice_per_repetition_and_target_flags_are_right():
    flashes = [m for m in _markers(_schedule("ข", reps=3)) if m["event"] == "flash"]
    assert [m["flash_idx"] for m in flashes] == list(range(72))
    counts = Counter(i for m in flashes for i in m["group"])
    assert len(counts) == 72 and set(counts.values()) == {6}
    target = CHARS.index("ข")
    assert all(m["is_target"] == (target in m["group"]) for m in flashes)
    assert sum(m["is_target"] for m in flashes) == 6


@pytest.mark.parametrize("kwargs,match", [
    ({"targets": ""}, "empty"),
    ({"targets": "กA"}, "not in the matrix"),
    ({"reps": 0}, "n_repetitions"),
    ({"flash": 0}, "flash_ms"),
    ({"isi": -1}, "isi_ms"),
])
def test_invalid_schedule_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _schedule(**kwargs)
