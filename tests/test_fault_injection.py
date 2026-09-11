"""
Cross-component fault injection. Each scenario here runs through more than one
component; the per-module tests cover each side on its own. TESTING.md maps
every required failure scenario to the tests that cover it.
"""

import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_byprop

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "collection"))
sys.path.insert(0, str(REPO_ROOT))

from gui_logic import (  # noqa: E402
    QUALITY_RED,
    QUALITY_YELLOW,
    REFRESH_MS,
    InferenceDisplayModel,
    SampleBuffer,
    describe_recorder_exit,
    describe_service_exit,
    quality_color,
    recorder_started,
    recording_path,
    scalp_levels,
    validate_session_id,
)
from mock_headset import (  # noqa: E402
    BACKGROUND_SCALE_UV,
    background_noise_step,
    channel_scale,
    interference_uv,
)
from pipeline import events as ev  # noqa: E402
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ, INFERENCE_FINGERPRINT_KEY  # noqa: E402
from pipeline.epoching import GAP_IN_WINDOW, EpochSpec, extract_epochs  # noqa: E402
from pipeline.fit_from_recording import FitError, fit_from_recording, load_recording, output_paths  # noqa: E402
from pipeline.manifest import load_characters_file  # noqa: E402
from pipeline.quality import GATE_SKIP, GREEN, RED, YELLOW, gate_epoch  # noqa: E402
from stopping.bayesian_stopping import BayesianStoppingController  # noqa: E402
from test_fit_from_recording import make_recording  # noqa: E402

CHARACTERS = load_characters_file(REPO_ROOT / "collection" / "characters_th.json")
REFRESH_S = REFRESH_MS / 1000.0


def _unique(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _mock_signal(n, seed, flat=(), noisy=(), railed=()):
    """What mock_headset.py emits for n samples: scaled background plus simulated contact faults."""
    rng = np.random.default_rng(seed)
    noise, _ = background_noise_step(np.zeros(len(EEG_CHANNELS)), rng, n)
    t = np.arange(n) / EEG_SFREQ
    flat, noisy, railed = set(flat), set(noisy), set(railed)
    return np.column_stack([
        noise[:, i] * BACKGROUND_SCALE_UV * channel_scale(ch, flat, 1.0)
        + interference_uv(t, ch, flat, noisy, railed)
        for i, ch in enumerate(EEG_CHANNELS)
    ])


def _stop(*procs):
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _wait_for_log(log_file, needle, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if Path(log_file).exists() and needle in Path(log_file).read_text(encoding="utf-8"):
            return True
        time.sleep(0.2)
    return False


def _start_mock_eeg(eeg_type, marker_type):
    return subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "collection" / "mock_headset.py"),
         "--eeg-type", eeg_type, "--marker-type", marker_type, "--no-p300"],
        cwd=REPO_ROOT / "collection", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _start_recorder(out_csv, eeg_type, marker_type, eeg_timeout=30.0, extra=()):
    return subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "collection" / "record_session.py"), "--out", str(out_csv),
         "--eeg-type", eeg_type, "--marker-type", marker_type, "--eeg-timeout", str(eeg_timeout), *extra],
        cwd=REPO_ROOT / "collection", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _start_service(manifest, log_file, eeg_type, marker_type, inference_type, eeg_timeout=30.0):
    return subprocess.Popen(
        [sys.executable, "-m", "pipeline.inference_service", "--manifest", str(manifest),
         "--eeg-type", eeg_type, "--marker-type", marker_type, "--inference-type", inference_type,
         "--eeg-timeout", str(eeg_timeout), "--log-file", str(log_file)],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


# -- A channel goes flat or railed: scalp map and quality gate -------------------

@pytest.mark.parametrize("fault,level", [("flat", RED), ("railed", RED), ("noisy", YELLOW)])
def test_mock_fault_colours_scalp_map_and_gate_makes_the_same_call(fault, level):
    for seed in range(5):
        data = _mock_signal(250, seed, **{fault: ["Pz"]})
        buffer = SampleBuffer()
        buffer.append(data)
        levels = scalp_levels(buffer, connected=True)
        assert levels["Pz"] == level, seed
        assert all(levels[ch] == GREEN for ch in EEG_CHANNELS if ch != "Pz"), (seed, levels)
        assert quality_color(levels["Pz"]) == (QUALITY_RED if level == RED else QUALITY_YELLOW)

        decision = gate_epoch(data[-200:].T, EEG_CHANNELS, GATE_SKIP)
        assert decision.accept is (level != RED), seed
        assert decision.bad_channels == (("Pz",) if level == RED else ())


def _seconds_until_red(fault, channel="P3", chunk=5, onset_s=1.5):
    """
    Stream mock EEG in LSL-sized chunks into the GUI's buffer, refreshing the
    scalp map every REFRESH_MS as the panel does. Returns seconds from fault
    onset to the first refresh that shows the channel red.
    """
    n = int(3 * EEG_SFREQ)
    onset = int(onset_s * EEG_SFREQ)
    stream = np.vstack([_mock_signal(n, seed=7)[:onset],
                        _mock_signal(n, seed=8, **{fault: [channel]})[onset:]])
    buffer = SampleBuffer()
    next_refresh = REFRESH_S
    for start in range(0, n, chunk):
        buffer.append(stream[start:start + chunk])
        now = (start + chunk) / EEG_SFREQ
        if now + 1e-9 < next_refresh:
            continue
        next_refresh += REFRESH_S
        red = scalp_levels(buffer, connected=True)[channel] == RED
        if start + chunk <= onset:
            assert not red, f"{channel} red at {now:.2f} s, before the fault at {onset_s} s"
        elif red:
            return now - onset_s
    return None


def test_railed_contact_mid_session_shows_red_within_one_refresh():
    latency = _seconds_until_red("railed")
    assert latency is not None and latency <= REFRESH_S + 1e-9


def test_flat_contact_mid_session_shows_red_within_one_refresh_of_100_ms_flatness():
    latency = _seconds_until_red("flat")
    assert latency is not None and latency <= 0.1 + REFRESH_S + 1e-9


# -- Manifest disagreement, t_max fallback, malformed events: publisher -> LSL -> GUI model

def _feed_publisher_into(models, fingerprint, timeout_s=60.0):
    """Run tools/fake_inference_publisher.py on a unique stream type and apply every payload to each model."""
    stream_type = _unique("INFFAULT")
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "tools" / "fake_inference_publisher.py"),
         "--inference-type", stream_type, "--fingerprint", fingerprint,
         "--manifest", str(REPO_ROOT / "tests" / "no_such_manifest.json"),
         "--flash-interval", "0.001", "--include-malformed", "--wait", "15"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        streams = resolve_byprop("type", stream_type, timeout=15)
        assert streams, "fake publisher stream not found"
        inlet = StreamInlet(streams[0])
        inlet.open_stream(timeout=5)
        desc_fingerprint = inlet.info(timeout=5).desc().child_value(INFERENCE_FINGERPRINT_KEY) or None
        for model in models:
            model.on_stream_connected(desc_fingerprint)

        ever_ready = {id(model): False for model in models}
        finished = False
        deadline = time.time() + timeout_s
        while not finished and time.time() < deadline:
            samples, _ = inlet.pull_chunk(timeout=0.5)
            for (payload,) in samples:
                for model in models:
                    model.apply_payload(payload)
                    ever_ready[id(model)] = ever_ready[id(model)] or model.service_ready
                finished = finished or '"state": "stopped"' in payload
        assert finished, "fake publisher did not finish within the timeout"
        return ever_ready
    finally:
        proc.wait(timeout=30)


def test_manifest_disagreement_hides_output_while_matching_panel_shows_fallbacks():
    fingerprint = "ab" * 32
    matching = InferenceDisplayModel(fingerprint, CHARACTERS)
    different = InferenceDisplayModel("cd" * 32, CHARACTERS)
    no_manifest = InferenceDisplayModel(None, CHARACTERS)

    ever_ready = _feed_publisher_into([matching, different, no_manifest], fingerprint)

    assert not matching.mismatch and ever_ready[id(matching)]
    assert matching.composed_text == "ก่อ"
    assert [c.reason for c in matching.committed] == [
        ev.REASON_THRESHOLD_CROSSED, ev.REASON_MAX_FLASHES_REACHED, ev.REASON_TRIAL_ENDED]
    assert [c.is_fallback for c in matching.committed] == [False, True, True]
    assert matching.malformed_count == 1
    assert matching.rejected_quality == 1
    assert matching.service_state == ev.STATE_STOPPED

    for model in (different, no_manifest):
        assert model.mismatch
        assert not ever_ready[id(model)]
        assert model.committed == []
        assert model.ignored_after_mismatch > 0
    assert "mismatch" in different.mismatch_message


# -- Real inference service process <-> GUI logic -------------------------------

@pytest.fixture(scope="module")
def fitted_manifest(tmp_path_factory):
    """A real classifier, calibration and manifest, fitted from a synthetic recording outside the repo."""
    out = tmp_path_factory.mktemp("fault_fit")
    recording = make_recording(out / "calibration_s1.csv")
    result = fit_from_recording(recording, "s1", models_dir=out / "models",
                                manifest_path=out / "session_manifest.json", log=lambda line: None)
    return out / "session_manifest.json", result.fingerprint


def test_service_without_eeg_stream_exits_3_and_panel_says_why(fitted_manifest, tmp_path):
    manifest, _ = fitted_manifest
    eeg_type = _unique("NOEEG")
    service = _start_service(manifest, tmp_path / "service.log.jsonl", eeg_type, _unique("MK"),
                             _unique("INF"), eeg_timeout=2.0)
    try:
        _, stderr = service.communicate(timeout=60)
    finally:
        _stop(service)
    assert service.returncode == 3
    lines = stderr.splitlines()
    assert any(line.startswith("inference_service: exit 3:") for line in lines), stderr
    message = describe_service_exit(service.returncode, lines)
    assert message.startswith("Live inference could not find the EEG stream")
    assert eeg_type in message


def test_late_subscribing_panel_verifies_service_manifest_via_heartbeat(fitted_manifest, tmp_path):
    manifest, fingerprint = fitted_manifest
    eeg_type, marker_type, inference_type = _unique("EEGF"), _unique("MKF"), _unique("INFF")
    log_file = tmp_path / "service.log.jsonl"
    mock = _start_mock_eeg(eeg_type, marker_type)
    service = _start_service(manifest, log_file, eeg_type, marker_type, inference_type)
    try:
        # Subscribe only after the one-shot "ready" went out: any ready seen now is a heartbeat.
        assert _wait_for_log(log_file, '"state": "ready"', timeout_s=60), "service never reported ready"
        streams = resolve_byprop("type", inference_type, timeout=15)
        assert streams, "service inference stream not found"
        inlet = StreamInlet(streams[0])
        inlet.open_stream(timeout=5)
        desc_fingerprint = inlet.info(timeout=5).desc().child_value(INFERENCE_FINGERPRINT_KEY) or None
        assert desc_fingerprint == fingerprint

        matching = InferenceDisplayModel(fingerprint, CHARACTERS)
        different = InferenceDisplayModel("0" * 64, CHARACTERS)
        for model in (matching, different):
            model.on_stream_connected(desc_fingerprint)
        deadline = time.time() + 10
        while time.time() < deadline and not matching.service_ready:
            samples, _ = inlet.pull_chunk(timeout=0.5)
            for (payload,) in samples:
                matching.apply_payload(payload)
                different.apply_payload(payload)

        assert matching.service_ready and not matching.mismatch
        assert different.mismatch and not different.service_ready
        assert "mismatch" in different.mismatch_message
        assert service.poll() is None
    finally:
        _stop(service, mock)
    assert service.returncode == 0


def _reset_is_stub():
    try:
        BayesianStoppingController(n_chars=4).reset(np.full(4, 0.25))
    except NotImplementedError:
        return True
    return False


@pytest.mark.skipif(not _reset_is_stub(), reason="BayesianStoppingController.reset() is implemented now")
def test_unimplemented_controller_stops_service_with_exit_4_naming_the_method(fitted_manifest, tmp_path):
    manifest, _ = fitted_manifest
    eeg_type, marker_type, inference_type = _unique("EEGC"), _unique("MKC"), _unique("INFC")
    markers = StreamOutlet(StreamInfo("P300MarkersFault", marker_type, 1, 0, "string", _unique("stim")))
    mock = _start_mock_eeg(eeg_type, marker_type)
    service = _start_service(manifest, tmp_path / "service.log.jsonl", eeg_type, marker_type, inference_type)
    try:
        assert markers.wait_for_consumers(timeout=60), "service never subscribed to the marker stream"
        markers.push_sample([json.dumps({"event": "trial_start", "trial_id": 0, "target_char_idx": 0,
                                         "target_char": CHARACTERS[0]})], local_clock())
        _, stderr = service.communicate(timeout=60)
    finally:
        _stop(service, mock)
    assert service.returncode == 4
    message = describe_service_exit(service.returncode, stderr.splitlines())
    assert message == "Live inference stopped: BayesianStoppingController.reset() is not implemented yet"


# -- Real recorder process <-> GUI logic and the fit script's loader -------------

def test_recorder_without_eeg_stream_exits_3_and_panel_says_why(tmp_path):
    recorder = _start_recorder(tmp_path / "session.csv", _unique("NOEEG"), _unique("MK"), eeg_timeout=2.0)
    try:
        _, stderr = recorder.communicate(timeout=60)
    finally:
        _stop(recorder)
    assert recorder.returncode == 3
    message = describe_recorder_exit(recorder.returncode, stderr.splitlines())
    assert message.startswith("Recorder could not find the EEG stream")


def test_demo_and_real_recordings_can_never_share_a_file(tmp_path):
    now = datetime(2026, 9, 11, 12, 0, 0)
    real = recording_path("subject_001", demo=False, now=now, recordings_dir=tmp_path)
    demo = recording_path("subject_001", demo=True, now=now, recordings_dir=tmp_path)
    assert demo.name.startswith("demo_") and not real.name.startswith("demo_")
    for session_id in ("demo", "demo_subject_001", "DEMO_x"):
        assert validate_session_id(session_id) is not None, session_id

    real.write_text("real session", encoding="utf-8")
    demo.write_text("demo session", encoding="utf-8")
    for is_demo, taken in ((False, real), (True, demo)):
        fresh = recording_path("subject_001", demo=is_demo, now=now, recordings_dir=tmp_path)
        assert not fresh.exists() and fresh not in (real, demo)

    for existing in (real, demo):
        recorder = _start_recorder(existing, _unique("EEGX"), _unique("MKX"), eeg_timeout=2.0)
        try:
            recorder.communicate(timeout=60)
        finally:
            _stop(recorder)
        assert recorder.returncode == 2
    assert real.read_text(encoding="utf-8") == "real session"
    assert demo.read_text(encoding="utf-8") == "demo session"

    is_demo, checkpoint, calibration, manifest = output_paths(demo, "subject_001", tmp_path / "models",
                                                              allow_demo=True)
    assert is_demo and all(p.name.startswith("demo_") for p in (checkpoint, calibration, manifest))
    with pytest.raises(FitError, match="--allow-demo"):
        output_paths(demo, "subject_001", tmp_path / "models")
    is_demo, checkpoint, calibration, manifest = output_paths(real, "subject_001", tmp_path / "models")
    assert not is_demo and not any(p.name.startswith("demo_") for p in (checkpoint, calibration, manifest))


def test_real_recorder_dropout_is_honoured_by_the_fit_loader(tmp_path):
    """The recorder's actual gap rows, not a synthetic imitation, must keep gapped epochs out of training."""
    eeg_type, marker_type = _unique("EEGG"), _unique("MKG")
    out_csv = tmp_path / "gap_session.csv"
    markers = StreamOutlet(StreamInfo("P300MarkersGap", marker_type, 1, 0, "string", _unique("stim")))
    mock = _start_mock_eeg(eeg_type, marker_type)
    recorder = _start_recorder(out_csv, eeg_type, marker_type, extra=["--skip-sanity-check"])
    restarted = None

    def push_flashes(first, count):
        for i in range(first, first + count):
            markers.push_sample([json.dumps({"event": "flash", "trial_id": 0, "flash_idx": i,
                                             "group": [0, 2, 4, 6, 8, 10], "is_target": i % 12 == 0})],
                                local_clock())
            time.sleep(0.124)

    try:
        # The GUI's own readiness predicate, checked against the real recorder's log.
        deadline = time.time() + 60
        while not recorder_started(out_csv) and time.time() < deadline:
            time.sleep(0.2)
        assert recorder_started(out_csv), "recorder never logged recording_started"
        assert markers.wait_for_consumers(timeout=60), "recorder never subscribed to the marker stream"
        time.sleep(1.5)
        push_flashes(0, 20)
        _stop(mock)                                    # headset drops out
        push_flashes(20, 12)                           # flashes keep coming during the outage
        restarted = _start_mock_eeg(eeg_type, marker_type)   # same source_id: the inlet recovers
        time.sleep(4.0)
        push_flashes(32, 20)
        time.sleep(3.0)                                # past the recorder's holdback
    finally:
        _stop(recorder, *(p for p in (mock, restarted) if p is not None))
    assert recorder.returncode == 0

    recording = load_recording(out_csv)
    assert recording.n_non_eeg_rows > 0
    assert [f.flash_idx for f in recording.flashes] == list(range(52)), "a marker was lost"
    _, kept, reasons = extract_epochs(recording.timestamps, recording.data,
                                      [f.onset for f in recording.flashes], EpochSpec())
    assert GAP_IN_WINDOW in reasons
    assert kept[:8].all(), reasons[:8]
    assert kept[-10:].all(), reasons[-10:]
    assert not kept[20:32].any(), reasons[20:32]
