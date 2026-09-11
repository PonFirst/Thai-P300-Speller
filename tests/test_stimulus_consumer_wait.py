import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "collection"))
cs = pytest.importorskip("checkerboard_stimulus")

CHARS = [chr(0x100 + i) for i in range(72)]


class FakeOutlet:
    def __init__(self, connects):
        self.connects = connects
        self.timeouts = []

    def wait_for_consumers(self, timeout):
        self.timeouts.append(timeout)
        return self.connects


def _speller(monkeypatch, outlet, waits):
    speller = cs.ThaiCheckerboardSpeller(CHARS, font_path=None)
    speller.outlet = outlet
    monkeypatch.setattr(cs.core, "wait", lambda seconds: waits.append(seconds))
    return speller


def test_waits_for_a_consumer_then_settles(monkeypatch):
    waits, outlet = [], FakeOutlet(connects=True)
    assert _speller(monkeypatch, outlet, waits).wait_for_marker_consumers(15.0, 2.0)
    assert outlet.timeouts == [15.0]
    assert waits == [2.0]


def test_no_consumer_warns_and_starts_anyway(monkeypatch):
    waits, outlet = [], FakeOutlet(connects=False)
    with pytest.warns(UserWarning, match="No marker consumer"):
        assert not _speller(monkeypatch, outlet, waits).wait_for_marker_consumers(3.0, 2.0)
    assert waits == []


def test_session_waits_once_before_its_first_trial(monkeypatch):
    calls = []
    speller = _speller(monkeypatch, FakeOutlet(connects=True), [])
    monkeypatch.setattr(speller, "wait_for_marker_consumers", lambda *args: calls.append("wait") or True)
    monkeypatch.setattr(speller, "run_trial", lambda char: calls.append(char) or True)
    assert speller.run_session(CHARS[0] + CHARS[1]) == 2
    assert calls == ["wait", CHARS[0], CHARS[1]]


def test_wait_requires_an_outlet():
    with pytest.raises(RuntimeError, match="create_lsl_outlet"):
        cs.ThaiCheckerboardSpeller(CHARS, font_path=None).wait_for_marker_consumers()
