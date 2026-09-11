import json

import numpy as np
import pytest

from pipeline import events as ev
from pipeline.events import EventSchemaError

CHARS = [f"C{i}" for i in range(72)]
CHARS[15] = "ก"
CHARS[41] = "ถ"

SPEC_EXAMPLES = [
    '{"event": "posterior_update", "trial_id": 42, "flash_idx": 7, '
    '"max_prob": 0.73, "leading_char_idx": 15, "theta_current": 0.90}',
    '{"event": "character_committed", "trial_id": 42, "char_idx": 15, '
    '"char": "ก", "n_flashes_used": 8, "reason": "threshold_crossed"}',
    '{"event": "character_committed", "trial_id": 43, "char_idx": 41, '
    '"char": "ถ", "n_flashes_used": 170, "reason": "max_flashes_reached"}',
]


@pytest.mark.parametrize("text", SPEC_EXAMPLES)
def test_contract_examples_validate(text):
    ev.decode(text, characters=CHARS)


def test_commit_reasons_are_distinct_and_classified():
    assert len(set(ev.COMMIT_REASONS)) == 3
    assert not ev.is_fallback(ev.REASON_THRESHOLD_CROSSED)
    assert ev.is_fallback(ev.REASON_MAX_FLASHES_REACHED)
    assert ev.is_fallback(ev.REASON_TRIAL_ENDED)


def test_roundtrip_preserves_thai():
    event = ev.character_committed(3, 15, "ก", 12, ev.REASON_THRESHOLD_CROSSED, characters=CHARS)
    text = ev.encode(event, characters=CHARS)
    assert "ก" in text
    assert ev.decode(text, characters=CHARS) == event


def test_builders_normalize_numpy_types():
    event = ev.posterior_update(np.int64(1), np.int32(2), np.float32(0.5), np.int64(3), np.float64(0.9))
    assert type(event["trial_id"]) is int
    assert type(event["leading_char_idx"]) is int
    assert type(event["max_prob"]) is float
    json.dumps(event)


def _posterior(**overrides):
    base = {"event": "posterior_update", "trial_id": 1, "flash_idx": 0,
            "max_prob": 0.5, "leading_char_idx": 3, "theta_current": 0.9}
    base.update(overrides)
    return base


def _committed(**overrides):
    base = {"event": "character_committed", "trial_id": 1, "char_idx": 3, "char": "C3",
            "n_flashes_used": 5, "reason": "threshold_crossed"}
    base.update(overrides)
    return base


@pytest.mark.parametrize("event,match", [
    ({k: v for k, v in _posterior().items() if k != "theta_current"}, "missing"),
    (_posterior(extra=1), "unexpected"),
    (_posterior(trial_id=True), "trial_id"),
    (_posterior(trial_id="1"), "trial_id"),
    (_posterior(flash_idx=-1), "flash_idx"),
    (_posterior(max_prob=float("nan")), "max_prob"),
    (_posterior(max_prob=1.5), "max_prob"),
    (_posterior(max_prob=-0.1), "max_prob"),
    (_posterior(theta_current=1.0), "theta_current"),
    (_posterior(leading_char_idx=72), "leading_char_idx"),
    (_posterior(leading_char_idx=-1), "leading_char_idx"),
    (_committed(reason="timeout"), "reason"),
    (_committed(n_flashes_used=0), "n_flashes_used"),
    (_committed(char=""), "char"),
    (_posterior(event="posterior_updated"), "unknown event type"),
    ([1, 2], "JSON object"),
])
def test_invalid_events_rejected(event, match):
    with pytest.raises(EventSchemaError, match=match):
        ev.validate_event(event)


def test_decode_rejects_invalid_json():
    with pytest.raises(EventSchemaError, match="not valid JSON"):
        ev.decode("{not json")


def test_decode_rejects_nan_literal():
    with pytest.raises(EventSchemaError, match="max_prob"):
        ev.decode('{"event": "posterior_update", "trial_id": 1, "flash_idx": 0, '
                  '"max_prob": NaN, "leading_char_idx": 3, "theta_current": 0.9}')


def test_decode_rejects_non_string_payload():
    with pytest.raises(EventSchemaError, match="string"):
        ev.decode(b"{}")


def test_encode_refuses_invalid_event():
    with pytest.raises(EventSchemaError):
        ev.encode(_posterior(max_prob=2.0))


def test_committed_char_must_match_characters_file():
    with pytest.raises(EventSchemaError, match="does not match"):
        ev.character_committed(1, 15, "ข", 5, ev.REASON_THRESHOLD_CROSSED, characters=CHARS)


def test_flash_rejected_and_service_status_roundtrip():
    rejected = ev.flash_rejected(2, 9, ev.REJECT_QUALITY_GATE, ("Pz", "O1"))
    assert ev.decode(ev.encode(rejected)) == rejected
    status = ev.service_status(ev.STATE_READY, "loaded", None)
    assert ev.decode(ev.encode(status)) == status
    with pytest.raises(EventSchemaError, match="state"):
        ev.service_status("crashed", "x", None)
    with pytest.raises(EventSchemaError, match="reason"):
        ev.flash_rejected(2, 9, "blink", ())
