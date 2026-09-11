"""
Unit tests for collection/gui_logic.py: everything in the Qt session panel
that does not need Qt, LSL, or real processes.
"""

import ast
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

import gui_logic as gl  # noqa: E402
from pipeline import events as ev  # noqa: E402
from pipeline import manifest as mf  # noqa: E402
from pipeline import quality  # noqa: E402
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ  # noqa: E402

CHARACTERS = mf.load_characters_file(gl.CHARACTERS_PATH)
FP = "a" * 64
OTHER_FP = "b" * 64
KO, MAI_EK, O_ANG = "ก", "่", "อ"
TK_PANEL = REPO_ROOT / "collection" / "session_gui.py"
LIBLSL_LINE = ("2026-09-11 16:09:38.241 (  17.442s) [R_P300Inferenc  ]      data_receiver.cpp:344    ERR| "
               "Stream transmission broke off (Input stream error.); re-connecting...")
NOW = datetime(2026, 9, 11, 14, 5, 9)


# ----------------------------------------------------------------------
# Parity with the Tkinter panel (read from its source; importing it would
# pull in Tk and matplotlib)
# ----------------------------------------------------------------------

def _tk_tree():
    source = TK_PANEL.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _tk_assignment(name):
    _, tree = _tk_tree()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in session_gui.py")


def _tk_brain_hemisphere():
    source, tree = _tk_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_brain_hemisphere":
            segment = " " * node.col_offset + ast.get_source_segment(source, node)
            namespace = {"np": np}
            exec(textwrap.dedent(segment), namespace)
            return namespace["_brain_hemisphere"]
    raise AssertionError("_brain_hemisphere not found in session_gui.py")


def test_electrode_positions_and_display_order_match_tk_panel():
    assert gl.ELECTRODE_POS == _tk_assignment("ELECTRODE_POS")
    assert gl.DISPLAY_ORDER == _tk_assignment("DISPLAY_ORDER") == ["F3", "Fz", "Cz", "C4", "P3", "Pz", "O1", "O2"]
    assert sorted(gl.DISPLAY_ORDER) == sorted(EEG_CHANNELS)
    assert gl.DISPLAY_WINDOW_S == _tk_assignment("DISPLAY_WINDOW_S") == 4.0
    assert gl.QUALITY_WINDOW_S == _tk_assignment("QUALITY_WINDOW_S") == 1.0
    assert gl.REFRESH_MS == _tk_assignment("REFRESH_MS") == 100


@pytest.mark.parametrize("side", [-1, 1])
def test_brain_hemisphere_is_exact_port(side):
    tk_version = _tk_brain_hemisphere()
    np.testing.assert_array_equal(gl.brain_hemisphere(side), tk_version(side))


def test_brain_hemisphere_geometry():
    right, left = gl.brain_hemisphere(1), gl.brain_hemisphere(-1)
    assert right.shape == (400, 2)
    np.testing.assert_allclose(left[:, 0], -right[:, 0])
    np.testing.assert_allclose(left[:, 1], right[:, 1])
    # The inner edge sits exactly at the fissure gap; nothing crosses it.
    assert right[0, 0] == pytest.approx(0.05) and right[-1, 0] == pytest.approx(0.05)
    assert right[:, 0].min() >= 0.05 - 1e-12
    # Radius modulated by 1 + 0.035 sin(11 t).
    t = np.linspace(0, np.pi, 400)
    radius = np.hypot((right[:, 0] - 0.05) / 0.95, right[:, 1])
    np.testing.assert_allclose(radius, 1 + 0.035 * np.sin(11 * t), atol=1e-12)


# ----------------------------------------------------------------------
# Colours, fonts, traces
# ----------------------------------------------------------------------

def test_quality_color_mapping():
    assert gl.quality_color(quality.GREEN) == gl.QUALITY_GREEN == "#22c55e"
    assert gl.quality_color(quality.YELLOW) == gl.QUALITY_YELLOW == "#eab308"
    assert gl.quality_color(quality.RED) == gl.QUALITY_RED == "#ef4444"
    assert gl.quality_color(quality.UNKNOWN) == gl.QUALITY_GRAY
    assert gl.quality_color("something else") == gl.QUALITY_GRAY
    assert len({gl.QUALITY_GREEN, gl.QUALITY_YELLOW, gl.QUALITY_RED, gl.QUALITY_GRAY}) == 4
    assert [label for _, label in gl.QUALITY_LEGEND][:3] == ["Good signal", "Noisy", "Bad / flat"]


def test_pick_thai_family_prefers_sarabun_then_thonburi():
    assert gl.pick_thai_family(["Arial", "Thonburi", "Sarabun"]) == "Sarabun"
    assert gl.pick_thai_family(["Arial", "Thonburi"]) == "Thonburi"
    assert gl.pick_thai_family(["Arial", "Loma"]) is None


def test_display_isolated_gives_combining_marks_a_carrier():
    assert gl.display_isolated(MAI_EK) == "◌" + MAI_EK
    assert gl.display_isolated(KO) == KO
    assert gl.display_isolated("ำ") == "ำ"  # sara am is a spacing letter


def test_normalize_trace_formula():
    rng = np.random.default_rng(3)
    y = rng.normal(5, 30, 1000)
    expected = (y - y.mean()) / (np.std(y) * 4) - 3
    np.testing.assert_allclose(gl.normalize_trace(y, -3), expected)


def test_normalize_trace_flat_and_empty():
    flat = gl.normalize_trace(np.full(100, 7.0), -2)
    assert np.all(np.isfinite(flat)) and np.allclose(flat, -2)
    assert gl.normalize_trace([], 0).size == 0


def test_sample_buffer_trims_pads_and_drops():
    buf = gl.SampleBuffer(n_channels=8, sfreq=250, seconds=4.0, extra=50)
    assert buf.capacity == 1050
    for start in range(0, 2000, 5):
        buf.append(np.full((5, 8), start, dtype=float))
    assert len(buf) == 1050
    assert buf.latest(1)[0, 0] == 1995
    buf.clear()
    buf.append(np.ones((3, 6)))
    assert np.all(np.isnan(buf.latest()[:, 6:])) and np.all(buf.latest()[:, :6] == 1)
    buf.append(np.ones((2, 10)))
    assert buf.latest().shape == (5, 8)
    buf.append(np.ones(8))
    assert len(buf) == 6
    assert buf.append(np.empty((0, 8))) == 0


def test_display_traces_use_display_order_and_4s_window():
    buf = gl.SampleBuffer()
    t = np.arange(2000) / EEG_SFREQ
    data = np.column_stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(8)])
    buf.append(data)
    traces = gl.display_traces(buf)
    assert list(traces) == gl.DISPLAY_ORDER
    for row, ch in enumerate(gl.DISPLAY_ORDER):
        x, y = traces[ch]
        assert x.size == y.size == 1000
        assert x[0] == 0 and x[-1] == pytest.approx(4.0)
        column = data[-1000:, EEG_CHANNELS.index(ch)]
        np.testing.assert_allclose(y, gl.normalize_trace(column, -row))


# ----------------------------------------------------------------------
# Scalp levels: every channel, several seeds, with margin
# ----------------------------------------------------------------------

def _eeg(seed, n=250, faults=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / EEG_SFREQ
    data = 20 * np.sin(2 * np.pi * 10 * t)[:, None] + rng.normal(0, 3, (n, 8))  # ~60 uV ptp: green
    for ch, kind in (faults or {}).items():
        col = EEG_CHANNELS.index(ch)
        if kind == "flat":
            data[:, col] = rng.normal(0, 0.01, n)                                   # ~0.06 uV
        elif kind == "noisy":
            data[:, col] = 150 * np.sin(2 * np.pi * 47 * t) + rng.normal(0, 3, n)   # ~300 uV
        elif kind == "railed":
            data[:, col] = 350 * np.sin(2 * np.pi * 47 * t)                          # ~700 uV
        elif kind == "nan":
            data[n // 2, col] = np.nan
    return data


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("channel", EEG_CHANNELS)
@pytest.mark.parametrize("kind,level", [("flat", quality.RED), ("noisy", quality.YELLOW),
                                        ("railed", quality.RED), ("nan", quality.UNKNOWN)])
def test_scalp_levels_flag_the_faulty_channel_only(seed, channel, kind, level):
    buf = gl.SampleBuffer()
    data = _eeg(seed, n=500, faults={channel: kind})
    for start in range(0, 500, 5):
        buf.append(data[start:start + 5])
    levels = gl.scalp_levels(buf, connected=True)
    assert levels[channel] == level
    assert all(levels[ch] == quality.GREEN for ch in EEG_CHANNELS if ch != channel)
    assert gl.quality_color(levels[channel]) == {quality.RED: gl.QUALITY_RED, quality.YELLOW: gl.QUALITY_YELLOW,
                                                 quality.UNKNOWN: gl.QUALITY_GRAY}[level]


def test_scalp_levels_use_only_the_last_second():
    buf = gl.SampleBuffer()
    buf.append(_eeg(0, n=500, faults={"Pz": "railed"}))
    buf.append(_eeg(1, n=250))
    assert gl.scalp_levels(buf, connected=True)["Pz"] == quality.GREEN


def test_scalp_levels_unknown_when_not_delivering_or_empty():
    buf = gl.SampleBuffer()
    assert set(gl.scalp_levels(buf, connected=True).values()) == {quality.UNKNOWN}
    buf.append(_eeg(0))
    assert set(gl.scalp_levels(buf, connected=False).values()) == {quality.UNKNOWN}
    few = gl.SampleBuffer()
    few.append(_eeg(0, n=4))
    assert set(gl.scalp_levels(few, connected=True).values()) == {quality.UNKNOWN}


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

@pytest.mark.parametrize("sid", ["subject_001", "S-2", "a", "X" * 64, "demos", "demo-1", "my_demo_1"])
def test_valid_session_ids(sid):
    assert gl.validate_session_id(sid) is None


@pytest.mark.parametrize("sid,match", [
    ("", "empty"), ("   ", "empty"), (None, "empty"),
    ("subject 001", "only contain"), ("../etc", "only contain"), ("a/b", "only contain"),
    ("s.1", "only contain"), ("ผู้ทดลอง", "only contain"), ("a:b", "only contain"),
    ("demo_subject", "reserved"), ("DEMO_x", "reserved"), ("Demo_", "reserved"),
    ("demo", "reserved"), ("DEMO", "reserved"),
    ("X" * 65, "longer"),
])
def test_invalid_session_ids(sid, match):
    assert match in gl.validate_session_id(sid)


def test_targets_every_matrix_character_is_accepted():
    assert gl.validate_targets("".join(CHARACTERS), CHARACTERS) is None
    assert gl.validate_targets(KO + MAI_EK + O_ANG, CHARACTERS) is None


@pytest.mark.parametrize("targets,match", [
    ("", "Type the characters"), ("   ", "Type the characters"),
    ("กA", "Not in the 72-character matrix: A"),
    ("ก ข", "(space)"), ("8", ": 8"), ("ฤ", "ฤ"),
])
def test_targets_rejected(targets, match):
    assert match in gl.validate_targets(targets, CHARACTERS)


def _settings(**overrides):
    base = dict(gl.DEFAULT_SETTINGS, targets="กข", characters=CHARACTERS)
    base.update(overrides)
    return gl.validate_settings(**base)


def test_defaults_are_valid():
    settings, errors = _settings()
    assert errors == []
    assert settings == gl.SessionSettings("subject_001", "กข", 5, 62, 62, 0, False)


@pytest.mark.parametrize("key", sorted(gl.BOUNDS))
def test_numeric_bounds(key):
    lo, hi = gl.BOUNDS[key]
    assert _settings(**{key: lo})[1] == []
    assert _settings(**{key: hi})[1] == []
    for bad in (lo - 1, hi + 1, True, 1.5, None, "x"):
        settings, errors = _settings(**{key: bad})
        assert settings is None and len(errors) == 1


def test_numeric_strings_and_whitespace_are_normalized():
    settings, errors = _settings(repetitions=" 7 ", session_id=" s1 ", targets=" ก ", windowed=1)
    assert errors == [] and settings.repetitions == 7 and settings.session_id == "s1"
    assert settings.targets == "ก" and settings.windowed is True


def test_all_errors_are_reported_together():
    settings, errors = _settings(targets="", session_id="demo_x", flash_ms=0, screen=9)
    assert settings is None and len(errors) == 4


# ----------------------------------------------------------------------
# Recording paths
# ----------------------------------------------------------------------

def test_recording_path_real_and_demo(tmp_path):
    real = gl.recording_path("subject_001", demo=False, now=NOW, recordings_dir=tmp_path)
    demo = gl.recording_path("subject_001", demo=True, now=NOW, recordings_dir=tmp_path)
    assert real == tmp_path / "subject_001_20260911_140509.csv"
    assert demo == tmp_path / "demo_subject_001_20260911_140509.csv"
    assert real != demo


def test_recording_path_never_returns_an_existing_file(tmp_path):
    first = gl.recording_path("s1", demo=False, now=NOW, recordings_dir=tmp_path)
    first.write_text("")
    second = gl.recording_path("s1", demo=False, now=NOW, recordings_dir=tmp_path)
    assert second == tmp_path / "s1_20260911_140509_1.csv"
    second.write_text("")
    third = gl.recording_path("s1", demo=False, now=NOW, recordings_dir=tmp_path)
    assert third == tmp_path / "s1_20260911_140509_2.csv" and not third.exists()


def test_demo_path_collision_gets_suffix_and_stays_demo(tmp_path):
    (tmp_path / "demo_s1_20260911_140509.csv").write_text("")
    path = gl.recording_path("s1", demo=True, now=NOW, recordings_dir=tmp_path)
    assert path.name == "demo_s1_20260911_140509_1.csv"


def test_existing_demo_file_does_not_affect_real_path(tmp_path):
    (tmp_path / "demo_s1_20260911_140509.csv").write_text("")
    assert gl.recording_path("s1", demo=False, now=NOW, recordings_dir=tmp_path).name == "s1_20260911_140509.csv"


def test_real_session_cannot_produce_a_demo_named_file(tmp_path):
    # The reserved prefix is what makes demo_ unambiguous on disk: a real
    # session named "demo_s1" would otherwise collide with demo mode's "s1".
    assert gl.validate_session_id("demo_s1") is not None
    assert gl.validate_session_id("demo") is not None     # would yield demo_<timestamp>.csv
    for sid in ("s1", "subject_001", "demos", "demo-1", "x-demo_"):
        assert gl.validate_session_id(sid) is None
        assert not gl.recording_path(sid, demo=False, now=NOW, recordings_dir=tmp_path).name.startswith("demo_")


def test_recording_path_with_injected_exists():
    taken = {Path("/r/s_20260911_140509.csv"), Path("/r/s_20260911_140509_1.csv")}
    path = gl.recording_path("s", demo=False, now=NOW, recordings_dir="/r", exists=lambda p: Path(p) in taken)
    assert path == Path("/r/s_20260911_140509_2.csv")


# ----------------------------------------------------------------------
# EEG connection state machine
# ----------------------------------------------------------------------

def test_connection_searching_blocks_start():
    conn = gl.EegConnection()
    assert conn.state == gl.SEARCHING and not conn.start_allowed
    assert conn.status_text == "● No EEG stream — start Unicorn LSL Interface"
    assert conn.status_color == gl.QUALITY_RED
    assert conn.tick(1e6) == gl.SEARCHING
    assert conn.on_stream_dropped(5.0) == gl.SEARCHING


def test_connection_connected_then_lost_then_recovered():
    conn = gl.EegConnection(lost_after_s=1.0)
    conn.on_stream_found(10.0, "MockUnicornEEG")
    assert conn.state == gl.CONNECTED and conn.start_allowed
    assert conn.status_text == "● EEG connected" and conn.status_color == gl.QUALITY_GREEN
    conn.on_samples(10.5, 5)
    assert conn.tick(11.5) == gl.CONNECTED       # exactly 1.0 s: not yet lost
    assert conn.tick(11.51) == gl.LOST           # > 1 s without samples
    assert not conn.start_allowed
    assert conn.status_text == "● EEG stream lost" and conn.status_color == gl.QUALITY_RED
    assert conn.tick(20.0) == gl.LOST
    conn.on_samples(20.0, 0)                     # an empty pull is not a delivery
    assert conn.state == gl.LOST
    conn.on_samples(20.1, 5)
    assert conn.state == gl.CONNECTED and conn.start_allowed


def test_connection_resolved_but_silent_stream_goes_lost():
    conn = gl.EegConnection()
    conn.on_stream_found(0.0)
    assert conn.tick(0.9) == gl.CONNECTED
    assert conn.tick(1.2) == gl.LOST


def test_connection_dropped_inlet_reports_lost_until_samples():
    conn = gl.EegConnection()
    conn.on_stream_found(0.0)
    conn.on_samples(0.1, 5)
    assert conn.on_stream_dropped(0.2) == gl.LOST
    assert conn.tick(0.3) == gl.LOST
    conn.on_stream_found(5.0)
    assert conn.state == gl.CONNECTED


# ----------------------------------------------------------------------
# Manifest availability
# ----------------------------------------------------------------------

def _write_manifest(root: Path):
    chars = root / "characters_th.json"
    chars.write_text(gl.CHARACTERS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    models = root / "models"
    models.mkdir()
    clf = models / "clf.joblib"
    clf.write_bytes(b"fake-classifier")
    cal = models / "platt.json"
    cal.write_text(
        '{"format": "%s", "a": 1.0, "b": 0.0, "score_method": "%s", "classifier_sha256": "%s"}'
        % (mf.CALIBRATION_FORMAT, mf.SCORE_METHOD, mf.sha256_file(clf)), encoding="utf-8")
    data = mf.build_manifest_dict(base_dir=root, characters_path=chars, classifier_path=clf, calibration_path=cal)
    return mf.write_manifest(data, root / "session_manifest.json")


def test_manifest_missing_disables_inference(tmp_path):
    manifest, reason = gl.load_manifest_for_gui(tmp_path / "session_manifest.json", mf.load_manifest)
    assert manifest is None and reason == gl.MANIFEST_MISSING_MESSAGE
    assert "pipeline.fit_from_recording" in reason
    assert gl.load_manifest_for_gui(None, mf.load_manifest) == (None, gl.MANIFEST_MISSING_MESSAGE)


def test_demo_manifest_missing_message(tmp_path):
    manifest, reason = gl.load_manifest_for_gui(tmp_path / "demo_session_manifest.json", mf.load_manifest,
                                                demo=True)
    assert manifest is None and reason == gl.DEMO_MANIFEST_MISSING_MESSAGE and "--allow-demo" in reason


def test_default_manifest_depends_on_mode_and_ignores_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, demo = gl.default_manifest_path(demo=False), gl.default_manifest_path(demo=True)
    assert real == mf.DEFAULT_MANIFEST_PATH == REPO_ROOT / "session_manifest.json"
    assert demo == REPO_ROOT / "demo_session_manifest.json"
    assert real.is_absolute() and demo.is_absolute() and real != demo
    assert gl.resolve_manifest_arg(None, demo=True) == demo
    assert gl.resolve_manifest_arg(None, demo=False) == real
    assert gl.resolve_manifest_arg("x/m.json", demo=True) == Path("x/m.json")


def test_manifest_invalid_reports_reason(tmp_path):
    path = tmp_path / "session_manifest.json"
    path.write_text("{not json", encoding="utf-8")
    manifest, reason = gl.load_manifest_for_gui(path, mf.load_manifest)
    assert manifest is None and reason.startswith("Inference unavailable:") and "not valid JSON" in reason


def test_manifest_loader_bug_does_not_raise(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{}", encoding="utf-8")

    def broken(_):
        raise RuntimeError("boom")
    manifest, reason = gl.load_manifest_for_gui(path, broken)
    assert manifest is None and "RuntimeError: boom" in reason


def test_valid_manifest_loads(tmp_path):
    written = _write_manifest(tmp_path)
    manifest, reason = gl.load_manifest_for_gui(written.path, mf.load_manifest)
    assert reason is None and manifest.fingerprint == written.fingerprint


# ----------------------------------------------------------------------
# Inference display model
# ----------------------------------------------------------------------

def _model(fp=FP):
    model = gl.InferenceDisplayModel(fp, CHARACTERS)
    model.on_stream_connected(fp)
    return model


def _commit(trial, char, reason=ev.REASON_THRESHOLD_CROSSED, n=10):
    return ev.encode(ev.character_committed(trial, CHARACTERS.index(char), char, n, reason,
                                            characters=CHARACTERS), characters=CHARACTERS)


def _posterior(trial, flash, prob, char=KO, theta=0.9):
    return ev.encode(ev.posterior_update(trial, flash, prob, CHARACTERS.index(char), theta))


def _status(state, fp=FP, detail=""):
    return ev.encode(ev.service_status(state, detail, fp))


def test_committed_characters_compose_into_one_string():
    model = _model()
    for trial, char in enumerate((KO, MAI_EK, O_ANG)):
        assert model.apply_payload(_commit(trial, char)) == ev.CHARACTER_COMMITTED
    assert model.composed_text == "ก่อ"
    assert [c.char for c in model.committed] == [KO, MAI_EK, O_ANG]


def test_gauge_tracks_posterior_and_resets_on_commit():
    model = _model()
    assert model.gauge is None
    model.apply_payload(_posterior(4, 0, 0.2, char="ข", theta=0.85))
    model.apply_payload(_posterior(4, 1, 0.6, char=KO, theta=0.85))
    assert model.gauge == gl.GaugeState(trial_id=4, flash_idx=1, max_prob=0.6, theta=0.85,
                                        leading_char_idx=CHARACTERS.index(KO), leading_char=KO)
    model.apply_payload(_commit(4, KO))
    assert model.gauge is None


def test_posterior_updates_for_committed_trials_are_ignored():
    model = _model()
    model.apply_payload(_commit(1, KO))
    assert model.apply_payload(_posterior(1, 30, 0.99)) is None
    assert model.gauge is None and model.stale_updates_ignored == 1
    model.apply_payload(_posterior(2, 0, 0.1))
    assert model.gauge.trial_id == 2


def test_duplicate_commit_for_a_trial_is_ignored():
    model = _model()
    model.apply_payload(_commit(1, KO))
    assert model.apply_payload(_commit(1, O_ANG)) is None
    assert model.composed_text == KO and model.duplicate_commits_ignored == 1


@pytest.mark.parametrize("reason,fallback", [(ev.REASON_THRESHOLD_CROSSED, False),
                                             (ev.REASON_MAX_FLASHES_REACHED, True),
                                             (ev.REASON_TRIAL_ENDED, True)])
def test_commit_reason_and_fallback_flag(reason, fallback):
    model = _model()
    model.apply_payload(_commit(0, KO, reason=reason, n=37))
    entry = model.committed[0]
    assert entry.reason == reason and entry.is_fallback is fallback and entry.n_flashes_used == 37
    text = entry.describe()
    assert reason in text and "37 flashes" in text and (("FALLBACK" in text) is fallback)


def test_rejected_flash_counters_split_quality_gate_from_other():
    model = _model()
    for reason in (ev.REJECT_QUALITY_GATE, ev.REJECT_QUALITY_GATE, ev.REJECT_EPOCH_INCOMPLETE,
                   ev.REJECT_INVALID_SCORE):
        model.apply_payload(ev.encode(ev.flash_rejected(0, 1, reason, ["Pz", "O1"])))
    assert model.rejected_quality == 2 and model.rejected_other == 2
    assert model.rejected_text() == "Rejected flashes: 2 quality gate (last: Pz, O1) · 2 other"


@pytest.mark.parametrize("payload", [
    "{not json",
    '{"event": "posterior_update", "trial_id": 1}',
    '{"event": "character_committed", "trial_id": 1, "char_idx": 0, "char": "ข", '
    '"n_flashes_used": 5, "reason": "threshold_crossed"}',   # char does not match characters[0]
    b"bytes",
    None,
    '["list"]',
])
def test_malformed_events_are_counted_not_raised(payload):
    model = _model()
    assert model.apply_payload(payload) is None
    assert model.malformed_count == 1 and model.last_malformed_error
    assert model.malformed_text().startswith("1 malformed inference event ")
    model.apply_payload(_commit(0, KO))
    assert model.composed_text == KO


def test_malformed_text_plural_and_empty():
    model = _model()
    assert model.malformed_text() == ""
    model.apply_payload("x")
    model.apply_payload("y")
    assert model.malformed_text().startswith("2 malformed inference events")


def test_service_status_and_readiness():
    model = _model()
    assert not model.service_ready
    model.apply_payload(_status(ev.STATE_STARTING, detail="loading"))
    assert model.service_text() == "Inference service: starting — loading" and not model.service_ready
    model.apply_payload(_status(ev.STATE_READY))
    assert model.service_ready
    model.on_stream_lost()
    assert not model.service_ready
    assert model.service_text() == "Inference service: no P300Inference stream"


def test_ready_heartbeat_alone_is_enough():
    model = _model()
    model.apply_payload(_status(ev.STATE_READY, detail="heartbeat"))   # `starting` was never seen
    assert model.service_ready


def test_ready_from_a_stream_connected_before_start_does_not_count():
    model = _model()                                  # a stream already connected
    model.reset_session()                             # Start pressed
    model.apply_payload(_status(ev.STATE_READY))      # buffered ready / heartbeat from the old service
    assert model.service_state == ev.STATE_READY and not model.service_ready
    model.on_stream_lost()
    model.on_stream_connected(FP)                     # the newly launched service
    assert not model.service_ready
    model.apply_payload(_status(ev.STATE_READY))
    assert model.service_ready


@pytest.mark.parametrize("state", [ev.STATE_STARTING, ev.STATE_ERROR, ev.STATE_STOPPED])
def test_leaving_ready_clears_readiness(state):
    model = _model()
    model.apply_payload(_status(ev.STATE_READY))
    model.apply_payload(_status(state))
    assert not model.service_ready


def test_warning_after_ready_keeps_readiness():
    model = _model()
    model.apply_payload(_status(ev.STATE_READY))
    model.apply_payload(_status(ev.STATE_WARNING, detail="version differs"))
    assert model.service_ready and model.service_state == ev.STATE_WARNING


def test_reset_session_clears_display_but_not_stream_facts():
    model = _model()
    model.apply_payload(_status(ev.STATE_READY))
    model.apply_payload(_commit(0, KO))
    model.apply_payload(ev.encode(ev.flash_rejected(0, 1, ev.REJECT_QUALITY_GATE, [])))
    model.reset_session()
    assert model.composed_text == "" and model.rejected_quality == 0 and model.service_state is None
    assert model.stream_connected and model.stream_fingerprint_ok and not model.service_ready
    model.apply_payload(_posterior(0, 0, 0.3))   # trial ids restart with a new session
    assert model.gauge is not None


@pytest.mark.parametrize("desc_fp", [OTHER_FP, "0000", "", None])
def test_desc_fingerprint_mismatch_hides_stream(desc_fp):
    model = gl.InferenceDisplayModel(FP, CHARACTERS)
    model.on_stream_connected(desc_fp)
    assert model.mismatch and not model.stream_fingerprint_ok
    assert model.mismatch_message.startswith("Inference output hidden:")
    model.apply_payload(_status(ev.STATE_READY))
    model.apply_payload(_commit(0, KO))
    assert model.composed_text == "" and not model.service_ready and model.ignored_after_mismatch == 2


def test_status_fingerprint_mismatch_mid_stream_stops_display():
    model = _model()
    model.apply_payload(_commit(0, KO))
    model.apply_payload(_status(ev.STATE_READY, fp=OTHER_FP))
    assert model.mismatch and "mismatch" in model.mismatch_message
    model.apply_payload(_commit(1, O_ANG))
    assert model.composed_text == KO


def test_status_without_fingerprint_defers_to_the_desc():
    model = _model()
    model.apply_payload(ev.encode(ev.service_status(ev.STATE_READY, "", None)))
    assert not model.mismatch and model.service_ready


def test_desc_is_authoritative_even_when_statuses_match():
    model = gl.InferenceDisplayModel(FP, CHARACTERS)
    model.on_stream_connected(None)
    assert model.mismatch and "no manifest fingerprint" in model.mismatch_message
    model.apply_payload(_status(ev.STATE_READY, fp=FP))
    assert not model.service_ready and model.service_state is None


def test_panel_without_manifest_cannot_verify_any_stream():
    model = gl.InferenceDisplayModel(None, CHARACTERS)
    model.on_stream_connected(FP)
    assert model.mismatch and "no valid session manifest" in model.mismatch_message


def test_new_stream_clears_mismatch():
    model = gl.InferenceDisplayModel(FP, CHARACTERS)
    model.on_stream_connected("0000")
    assert model.mismatch
    model.on_stream_lost()
    model.on_stream_connected(FP)
    assert not model.mismatch and model.stream_fingerprint_ok


def test_malformed_counted_even_after_mismatch():
    model = gl.InferenceDisplayModel(FP, CHARACTERS)
    model.on_stream_connected("0000")
    model.apply_payload("{bad")
    assert model.malformed_count == 1


def test_fake_publisher_script_end_to_end():
    from tools.fake_inference_publisher import build_script
    script = build_script(CHARACTERS, FP, 0.9, 170, 0.0, include_malformed=True)
    model = _model()
    for payload, _pause in script:
        model.apply_payload(payload)
    assert model.composed_text == "ก่อ"
    assert [c.reason for c in model.committed] == [ev.REASON_THRESHOLD_CROSSED, ev.REASON_MAX_FLASHES_REACHED,
                                                   ev.REASON_TRIAL_ENDED]
    assert [c.is_fallback for c in model.committed] == [False, True, True]
    assert model.malformed_count == 1 and model.rejected_quality == 1 and model.rejected_other == 0
    assert model.service_state == ev.STATE_STOPPED and model.gauge is None and not model.mismatch


def test_fake_publisher_with_wrong_fingerprint_is_hidden():
    from tools.fake_inference_publisher import build_script
    model = gl.InferenceDisplayModel(FP, CHARACTERS)
    model.on_stream_connected("0000")
    for payload, _ in build_script(CHARACTERS, "0000", 0.9, 170, 0.0, include_malformed=False):
        model.apply_payload(payload)
    assert model.mismatch and model.composed_text == "" and model.service_state is None


# ----------------------------------------------------------------------
# Exit codes
# ----------------------------------------------------------------------

@pytest.mark.parametrize("code,match", [
    (0, "finished normally"), (1, "crashed"), (2, "output file already exists"),
    (3, "could not find the EEG stream"), (-15, "signal 15"), (9, "unexpected code 9"),
])
def test_recorder_exit_codes(code, match):
    assert match in gl.describe_recorder_exit(code)


@pytest.mark.parametrize("code,match", [
    (0, "finished normally"), (1, "crashed"), (2, "configuration, manifest or checkpoint error"),
    (3, "could not find the EEG stream"), (4, "is not implemented yet"), (-9, "signal 9"),
])
def test_service_exit_codes(code, match):
    assert match in gl.describe_service_exit(code)


def test_exit_reason_picks_the_exit_line_over_liblsl_noise():
    lines = ["Traceback (most recent call last):",
             "inference_service: exit 2: cannot load session from m.json: bad checkpoint (log: l.jsonl)",
             LIBLSL_LINE, "", "  "]
    assert gl.exit_reason(lines) == "cannot load session from m.json: bad checkpoint (log: l.jsonl)"
    assert gl.describe_service_exit(2, lines) == (
        "Live inference refused to start: configuration, manifest or checkpoint error: "
        "cannot load session from m.json: bad checkpoint (log: l.jsonl)")


def test_exit_reason_without_exit_line_skips_liblsl_lines():
    assert gl.exit_reason(["error: output file exists: a.csv", LIBLSL_LINE]) == "error: output file exists: a.csv"
    assert gl.exit_reason([LIBLSL_LINE, "  "]) is None
    assert gl.exit_reason([]) is None
    assert gl.describe_recorder_exit(2, [LIBLSL_LINE]) == (
        "Recorder refused to start: bad arguments or the output file already exists")


@pytest.mark.parametrize("lines,method", [
    (["inference_service: exit 4: BayesianStoppingController.update() raised NotImplementedError at "
      "trial 0, flash 3. (log: x.jsonl)", LIBLSL_LINE], "update"),
    (["NotImplementedError: update(): additive log update on group_idx only"], "update"),
    (["error: BayesianStoppingController.llr() is not implemented"], "llr"),
    (["NotImplementedError: reset(): blend lm_prior"], "reset"),
    ([], "reset/llr/update"),
])
def test_stopping_not_implemented_message(lines, method):
    assert gl.describe_service_exit(4, lines) == (
        f"Live inference stopped: BayesianStoppingController.{method}() is not implemented yet")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeProc:
    def __init__(self, name, cmd, cwd, stubborn=False):
        self.name, self.cmd, self.cwd = name, cmd, cwd
        self.returncode = None
        self.stubborn = stubborn
        self.terminated = self.killed = False
        self.lines = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def stderr_lines(self):
        return list(self.lines)

    def exit(self, code, *lines):
        self.returncode = code
        self.lines.extend(lines)


class Harness:
    def __init__(self, stubborn=(), recorder_started=True):
        self.clock = FakeClock()
        self.launched = {}
        self.order = []
        self.messages = []
        self.stubborn = set(stubborn)
        self.recorder_started = recorder_started
        self.ready_checks = []
        self.orch = gl.SessionOrchestrator(self.launch, self.clock, self.note, python="PY",
                                           recorder_ready=self._recorder_ready)

    def _recorder_ready(self, out_path):
        self.ready_checks.append(out_path)
        return self.recorder_started

    def launch(self, name, cmd, cwd):
        proc = FakeProc(name, cmd, cwd, stubborn=name in self.stubborn)
        self.launched[name] = proc
        self.order.append(name)
        return proc

    def note(self, message, level):
        self.messages.append((level, message))

    @property
    def last(self):
        return self.messages[-1]


SETTINGS = gl.SessionSettings("s1", "กข", 3, 80, 90, 1, True)
OUT = Path("/rec/s1.csv")


def test_commands():
    assert gl.service_command("PY", "/m.json") == ["PY", "-m", "pipeline.inference_service", "--manifest", "/m.json"]
    assert gl.recorder_command("PY", OUT) == ["PY", str(gl.COLLECTION_DIR / "record_session.py"), "--out", str(OUT)]
    cmd = gl.stimulus_command("PY", SETTINGS)
    assert cmd == ["PY", str(gl.COLLECTION_DIR / "checkerboard_stimulus.py"), "--targets", "กข",
                   "--n-repetitions", "3", "--flash-ms", "80", "--isi-ms", "90", "--screen", "1", "--windowed"]
    assert "--windowed" not in gl.stimulus_command("PY", gl.SessionSettings("s", "ก", 5, 62, 62, 0, False))


def test_children_run_with_this_interpreter_not_a_bare_python():
    orch = gl.SessionOrchestrator(lambda *a: None, FakeClock(), lambda *_: None)
    assert orch.python == sys.executable
    assert gl.service_command(orch.python, "/m.json")[0] == sys.executable


def test_service_extra_args_are_forwarded():
    assert gl.service_command("PY", "/m.json", ["--eeg-type", "EEG_x"])[-2:] == ["--eeg-type", "EEG_x"]
    clock, launched = FakeClock(), {}
    orch = gl.SessionOrchestrator(lambda n, c, d: launched.setdefault(n, FakeProc(n, c, d)), clock,
                                  lambda *_: None, python="PY", service_extra_args=["--inference-type", "T"])
    orch.start(SETTINGS, OUT, manifest_path="/m.json")
    assert launched["service"].cmd == ["PY", "-m", "pipeline.inference_service", "--manifest", "/m.json",
                                       "--inference-type", "T"]


def test_orchestrator_defaults_to_reading_the_recorders_log():
    orch = gl.SessionOrchestrator(lambda *a: None, FakeClock(), lambda *_: None)
    assert orch.recorder_ready is gl.recorder_started


def test_recorder_started_reads_the_recorders_structured_log(tmp_path):
    out = tmp_path / "s1.csv"
    assert not gl.recorder_started(out)
    log = gl.recorder_log_path(out)
    assert log.name == "s1.csv.log.jsonl"
    log.write_text('{"event": "eeg_resolve_attempt"}\nnot json\n[1, 2]\n', encoding="utf-8")
    assert not gl.recorder_started(out)
    with log.open("a", encoding="utf-8") as f:
        f.write('{"time": "t", "level": "INFO", "event": "recording_started", "msg": "Recording EEG"}\n')
    assert gl.recorder_started(out)


def test_recorder_that_never_starts_recording_times_out():
    h = Harness(recorder_started=False)
    h.orch.start(SETTINGS, OUT)
    h.clock.advance(gl.RECORDER_READY_TIMEOUT_S - 0.1)
    h.orch.tick()
    assert h.orch.phase == gl.WAITING_RECORDER and "stimulus" not in h.launched
    h.clock.advance(0.2)
    h.orch.tick()
    assert "stimulus" not in h.launched and h.orch.phase == gl.IDLE
    assert h.launched["recorder"].terminated
    level, message = h.last
    assert level == gl.ERROR and "did not report that recording started" in message


def test_calibration_start_without_manifest():
    h = Harness(recorder_started=False)
    assert h.orch.start(SETTINGS, OUT, manifest_path=None)
    assert h.order == ["recorder"] and h.launched["recorder"].cwd == gl.COLLECTION_DIR
    assert h.orch.phase == gl.WAITING_RECORDER and h.orch.active
    h.clock.advance(10.0)
    h.orch.tick()
    assert "stimulus" not in h.launched        # the recorder has not reported recording_started yet
    h.recorder_started = True
    h.orch.tick()
    assert h.ready_checks[-1] == OUT
    stim = h.launched["stimulus"]
    assert stim.cwd == gl.COLLECTION_DIR and stim.cmd == gl.stimulus_command("PY", SETTINGS)
    assert h.orch.phase == gl.RUNNING and h.last[1].startswith("Running — 2 trials")
    assert not h.orch.start(SETTINGS, OUT)   # already running

    stim.exit(0)
    h.orch.tick()
    assert h.launched["recorder"].terminated
    assert h.orch.phase == gl.IDLE and h.last == (gl.INFO, "Stimulus finished — stopping recorder.")
    h.orch.tick()
    assert not h.orch.active


def test_live_inference_waits_for_ready_then_starts_recorder():
    h = Harness()
    h.orch.start(SETTINGS, OUT, manifest_path="/repo/session_manifest.json")
    service = h.launched["service"]
    assert h.order == ["service"] and service.cwd == gl.REPO_ROOT
    assert service.cmd == ["PY", "-m", "pipeline.inference_service", "--manifest", "/repo/session_manifest.json"]
    h.clock.advance(5)
    h.orch.tick(service_ready=False)
    assert h.order == ["service"]
    h.orch.tick(service_ready=True)
    assert h.order == ["service", "recorder"]
    h.clock.advance(3.0)
    h.orch.tick()
    assert h.order == ["service", "recorder", "stimulus"]
    h.launched["stimulus"].exit(0)
    h.orch.tick()
    assert h.launched["recorder"].terminated and service.terminated


def test_service_ready_timeout_aborts():
    h = Harness()
    h.orch.start(SETTINGS, OUT, manifest_path="/m.json")
    h.clock.advance(19.9)
    h.orch.tick()
    assert h.orch.phase == gl.WAITING_SERVICE
    h.clock.advance(0.2)
    h.orch.tick()
    assert h.orch.phase == gl.IDLE and "recorder" not in h.launched
    assert h.launched["service"].terminated
    level, message = h.last
    assert level == gl.ERROR and "within 20 s" in message


def test_service_exit_4_before_ready_aborts_with_method_name():
    h = Harness()
    h.orch.start(SETTINGS, OUT, manifest_path="/m.json")
    h.launched["service"].exit(4, "NotImplementedError: update(): additive log update")
    h.orch.tick()
    level, message = h.last
    assert level == gl.ERROR
    assert message.startswith("Live inference stopped: BayesianStoppingController.update() is not implemented yet")
    assert "recorder" not in h.launched and not h.orch.active


@pytest.mark.parametrize("code,match", [(2, "output file already exists"), (3, "could not find the EEG stream"),
                                        (1, "crashed")])
def test_recorder_failure_during_settle_prevents_stimulus(code, match):
    h = Harness()
    h.orch.start(SETTINGS, OUT)
    h.clock.advance(1.0)
    h.launched["recorder"].exit(code, "reason line")
    h.orch.tick()
    h.clock.advance(5.0)
    h.orch.tick()
    assert "stimulus" not in h.launched and h.orch.phase == gl.IDLE
    level, message = h.last
    assert level == gl.ERROR and match in message and "reason line" in message


def test_recorder_crash_while_running_stops_stimulus():
    h = Harness()
    h.orch.start(SETTINGS, OUT)
    h.clock.advance(3.0)
    h.orch.tick()
    h.launched["recorder"].exit(1)
    h.orch.tick()
    assert h.launched["stimulus"].terminated and h.orch.phase == gl.IDLE and h.last[0] == gl.ERROR


def test_service_exit_while_running_keeps_recording():
    h = Harness()
    h.orch.start(SETTINGS, OUT, manifest_path="/m.json")
    h.orch.tick(service_ready=True)
    h.clock.advance(3.0)
    h.orch.tick()
    h.launched["service"].exit(4, "BayesianStoppingController.reset() is not implemented")
    h.orch.tick()
    assert h.orch.phase == gl.RUNNING and not h.launched["recorder"].terminated
    assert "reset() is not implemented yet" in h.last[1] and "Recording continues" in h.last[1]


def test_stimulus_nonzero_exit_is_an_error_but_still_stops_everything():
    h = Harness()
    h.orch.start(SETTINGS, OUT)
    h.clock.advance(3.0)
    h.orch.tick()
    h.launched["stimulus"].exit(1)
    h.orch.tick()
    assert h.last == (gl.ERROR, "Stimulus exited with code 1 — stopping recorder.")
    assert h.launched["recorder"].terminated


def test_stop_terminates_then_kills_after_timeout():
    h = Harness(stubborn={"recorder"})
    h.orch.start(SETTINGS, OUT)
    h.clock.advance(3.0)
    h.orch.tick()
    h.orch.stop()
    rec, stim = h.launched["recorder"], h.launched["stimulus"]
    assert rec.terminated and stim.terminated and not rec.killed
    assert h.orch.phase == gl.IDLE and h.orch.active      # recorder still shutting down
    h.clock.advance(2.9)
    h.orch.tick()
    assert not rec.killed
    h.clock.advance(0.2)
    h.orch.tick()
    assert rec.killed and not stim.killed and not h.orch.active


def test_shutdown_blocks_until_exit_or_kill():
    h = Harness(stubborn={"recorder", "service"})
    h.orch.start(SETTINGS, OUT, manifest_path="/m.json")
    h.orch.tick(service_ready=True)
    h.orch.shutdown(wait_s=3.0, sleep=h.clock.advance)
    assert h.launched["recorder"].killed and h.launched["service"].killed
    assert not h.orch.active


def test_stop_when_idle_is_harmless():
    h = Harness()
    h.orch.stop()
    h.orch.shutdown(sleep=h.clock.advance)
    assert not h.orch.active and h.launched == {}


def test_stderr_tail_is_bounded():
    tail = gl.StderrTail(maxlen=3)
    for i in range(5):
        tail.append(f"line {i}\n")
    assert tail.lines() == ["line 2", "line 3", "line 4"]
