"""
P300Inference event contract: emitted by pipeline/inference_service.py,
consumed by collection/session_gui_qt.py. Both ends call validate_event(),
so a schema drift fails at the boundary instead of rendering garbage.

Transport: a dedicated LSL outlet (see pipeline/constants.py and
docs/INTERFACES.md for why LSL rather than in-process IPC).

Contract events, one JSON object per LSL sample:

    posterior_update     trial_id, flash_idx, max_prob, leading_char_idx, theta_current
    character_committed  trial_id, char_idx, char, n_flashes_used, reason

Additive extensions (flagged in docs/INTERFACES.md):

    reason "trial_ended"  The stimulus finished the trial before either stopping
                          condition fired. The stimulus runs a fixed number of
                          repetitions (5 x 24 = 120 flashes = t_max by default);
                          when flashes were rejected, the flashes run out before t
                          reaches t_max, so this happens; without its own reason
                          it would masquerade as one of the other two.
    flash_rejected       trial_id, flash_idx, reason, channels
    service_status       state, detail, manifest_fingerprint

Validation is strict: unknown event types, missing keys, and extra keys are
all errors. Any new field is a deliberate change made here.
"""

from __future__ import annotations

import json
import math
import numbers
from typing import Any, Callable

from pipeline.constants import N_CHARS

POSTERIOR_UPDATE = "posterior_update"
CHARACTER_COMMITTED = "character_committed"
FLASH_REJECTED = "flash_rejected"
SERVICE_STATUS = "service_status"

REASON_THRESHOLD_CROSSED = "threshold_crossed"
REASON_MAX_FLASHES_REACHED = "max_flashes_reached"
REASON_TRIAL_ENDED = "trial_ended"
COMMIT_REASONS = (REASON_THRESHOLD_CROSSED, REASON_MAX_FLASHES_REACHED, REASON_TRIAL_ENDED)
FALLBACK_REASONS = (REASON_MAX_FLASHES_REACHED, REASON_TRIAL_ENDED)

REJECT_QUALITY_GATE = "quality_gate"
REJECT_EPOCH_INCOMPLETE = "epoch_incomplete"
REJECT_INVALID_SCORE = "classifier_invalid_score"
REJECT_REASONS = (REJECT_QUALITY_GATE, REJECT_EPOCH_INCOMPLETE, REJECT_INVALID_SCORE)

STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_WARNING = "warning"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"
STATES = (STATE_STARTING, STATE_READY, STATE_WARNING, STATE_ERROR, STATE_STOPPED)


class EventSchemaError(ValueError):
    """An event does not match the P300Inference contract."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonneg_int(name, value, ctx):
    if not _is_int(value) or value < 0:
        raise EventSchemaError(f"{name} must be a non-negative integer, got {value!r}")


def _positive_int(name, value, ctx):
    if not _is_int(value) or value < 1:
        raise EventSchemaError(f"{name} must be a positive integer, got {value!r}")


def _probability(name, value, ctx):
    if not _is_number(value) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise EventSchemaError(f"{name} must be a finite number in [0, 1], got {value!r}")


def _open_unit(name, value, ctx):
    if not _is_number(value) or not math.isfinite(value) or not 0.0 < value < 1.0:
        raise EventSchemaError(f"{name} must be a finite number in (0, 1), got {value!r}")


def _char_index(name, value, ctx):
    n_chars = ctx["n_chars"]
    if not _is_int(value) or not 0 <= value < n_chars:
        raise EventSchemaError(f"{name} must be an integer in [0, {n_chars}), got {value!r}")


def _nonempty_str(name, value, ctx):
    if not isinstance(value, str) or not value:
        raise EventSchemaError(f"{name} must be a non-empty string, got {value!r}")


def _string(name, value, ctx):
    if not isinstance(value, str):
        raise EventSchemaError(f"{name} must be a string, got {value!r}")


def _string_list(name, value, ctx):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise EventSchemaError(f"{name} must be a list of strings, got {value!r}")


def _fingerprint(name, value, ctx):
    if value is not None and (not isinstance(value, str) or not value):
        raise EventSchemaError(f"{name} must be null or a non-empty string, got {value!r}")


def _one_of(allowed: tuple) -> Callable:
    def check(name, value, ctx):
        if value not in allowed:
            raise EventSchemaError(f"{name} must be one of {list(allowed)}, got {value!r}")
    return check


_SCHEMAS: dict[str, dict[str, Callable]] = {
    POSTERIOR_UPDATE: {
        "trial_id": _nonneg_int,
        "flash_idx": _nonneg_int,
        "max_prob": _probability,
        "leading_char_idx": _char_index,
        "theta_current": _open_unit,
    },
    CHARACTER_COMMITTED: {
        "trial_id": _nonneg_int,
        "char_idx": _char_index,
        "char": _nonempty_str,
        "n_flashes_used": _positive_int,
        "reason": _one_of(COMMIT_REASONS),
    },
    FLASH_REJECTED: {
        "trial_id": _nonneg_int,
        "flash_idx": _nonneg_int,
        "reason": _one_of(REJECT_REASONS),
        "channels": _string_list,
    },
    SERVICE_STATUS: {
        "state": _one_of(STATES),
        "detail": _string,
        "manifest_fingerprint": _fingerprint,
    },
}


def validate_event(event: Any, *, n_chars: int = N_CHARS,
                   characters: list[str] | None = None) -> dict:
    """
    Raise EventSchemaError unless `event` matches the contract exactly.
    When `characters` is given, a committed event's `char` must equal
    characters[char_idx].
    """
    if characters is not None:
        n_chars = len(characters)
    if not isinstance(event, dict):
        raise EventSchemaError(f"event must be a JSON object, got {type(event).__name__}")
    kind = event.get("event")
    if kind not in _SCHEMAS:
        raise EventSchemaError(f"unknown event type {kind!r}; expected one of {sorted(_SCHEMAS)}")

    schema = _SCHEMAS[kind]
    expected = set(schema) | {"event"}
    missing = expected - event.keys()
    extra = event.keys() - expected
    if missing:
        raise EventSchemaError(f"{kind}: missing field(s) {sorted(missing)}")
    if extra:
        raise EventSchemaError(f"{kind}: unexpected field(s) {sorted(extra)}")

    ctx = {"n_chars": n_chars}
    for field, check in schema.items():
        check(f"{kind}.{field}", event[field], ctx)

    if kind == CHARACTER_COMMITTED and characters is not None:
        expected_char = characters[event["char_idx"]]
        if event["char"] != expected_char:
            raise EventSchemaError(
                f"{kind}.char {event['char']!r} does not match characters[{event['char_idx']}] "
                f"= {expected_char!r}")
    return event


def is_fallback(reason: str) -> bool:
    """True for a commit forced by a limit rather than by crossing theta."""
    return reason in FALLBACK_REASONS


def _as_int(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return value
    return int(value)


def _as_float(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return value
    return float(value)


def posterior_update(trial_id, flash_idx, max_prob, leading_char_idx, theta_current,
                     **ctx) -> dict:
    return validate_event({
        "event": POSTERIOR_UPDATE,
        "trial_id": _as_int(trial_id),
        "flash_idx": _as_int(flash_idx),
        "max_prob": _as_float(max_prob),
        "leading_char_idx": _as_int(leading_char_idx),
        "theta_current": _as_float(theta_current),
    }, **ctx)


def character_committed(trial_id, char_idx, char, n_flashes_used, reason, **ctx) -> dict:
    return validate_event({
        "event": CHARACTER_COMMITTED,
        "trial_id": _as_int(trial_id),
        "char_idx": _as_int(char_idx),
        "char": char,
        "n_flashes_used": _as_int(n_flashes_used),
        "reason": reason,
    }, **ctx)


def flash_rejected(trial_id, flash_idx, reason, channels=(), **ctx) -> dict:
    return validate_event({
        "event": FLASH_REJECTED,
        "trial_id": _as_int(trial_id),
        "flash_idx": _as_int(flash_idx),
        "reason": reason,
        "channels": list(channels),
    }, **ctx)


def service_status(state, detail, manifest_fingerprint, **ctx) -> dict:
    return validate_event({
        "event": SERVICE_STATUS,
        "state": state,
        "detail": detail,
        "manifest_fingerprint": manifest_fingerprint,
    }, **ctx)


def encode(event: dict, **ctx) -> str:
    """Validate, then serialize for one LSL string sample (Thai kept unescaped)."""
    return json.dumps(validate_event(event, **ctx), ensure_ascii=False, allow_nan=False)


def decode(text: Any, **ctx) -> dict:
    """Parse and validate one LSL string sample."""
    if not isinstance(text, str):
        raise EventSchemaError(f"event payload must be a string, got {type(text).__name__}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EventSchemaError(f"not valid JSON: {exc.msg}") from exc
    return validate_event(obj, **ctx)
