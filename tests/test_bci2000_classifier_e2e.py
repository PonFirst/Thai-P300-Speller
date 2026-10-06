"""
End to end through real BCI2000: calibrate, fit, then spell online with the fitted classifier, and
check the online scores flash by flash against an offline recomputation of the same recording.

  1. Calibration: UnicornSerialSource (fake headset) + DummySignalProcessing + CheckerboardSpeller in
     copy mode, loading bci2000/parms/ThaiCheckerboard.prm. pipeline.fit_from_recording fits the
     .dat into a temp dir and writes the BCI2000 model .prm.
  2. Online: the same with CheckerboardSignalProcessing loading both .prm files. On its .dat, every
     flash's online result (ClassifierScore, ClassifierOnsetSample, StimulusCodeRes,
     StimulusTypeRes) must equal what the checkpoint pipeline computes offline on epochs cut by
     pipeline.epoching.extract_epochs from the same file: scores within float32 rounding, onsets,
     codes and labels exactly, no flash missing or duplicated, each result inside its own selection.
  3. The speller decides on complete evidence: every selection's applog line counts a result for
     every flash (results arriving after the decision used to leak into the next character), with
     the shipped EpochLength and with the smallest EpochLength the classifier's preflight accepts.
  4. Lost samples: the fake drops single packets every 6 s. The flashes rejected online
     (ClassifierRejectedCode) are exactly those whose padded window holds a filled sample offline
     (NaN in eeg_microvolts); every other flash keeps parity. Likewise for windows that would start
     before the run (a 0.3 s cue at the start).
  5. Preflight refusals, each read from the system log: no model loaded, two channel names swapped,
     EpochLength shorter than the result latency, NotchFilter on. A correct configuration between
     them must be accepted.

It opens BCI2000 windows for about 3 minutes, so it runs only with BCI2000_GUI_TESTS=1 (and a
BCI2000 build with the project modules, and no other BCI2000 source module running). Every output
goes to a temp dir; BCI2000 and the fake headsets are stopped even when a test fails.
"""

import contextlib
import os
import re
import signal as signal_module
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pipeline.bci2000_dat import read_dat
from pipeline.bci2000_recording import eeg_microvolts, flash_events
from pipeline.classifier_io import load_checkpoint
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ
from pipeline.epoching import BEFORE_BUFFER, NON_FINITE, extract_epochs
from pipeline.fit_from_recording import fit_from_recording
from pipeline.manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
from fake_unicorn_bt import FakeUnicorn, noise_signal  # noqa: E402
from make_bci2000_parms import OUT as SPELLER_PRM, encode  # noqa: E402

BCI2000_ROOT = Path(os.environ.get("BCI2000_ROOT", Path.home() / "BCI2000" / "trunk"))
PROG = BCI2000_ROOT / "prog"
MODULES = ("BCI2000Shell", "UnicornSerialSource.app", "CheckerboardSpeller.app", "DummySignalProcessing.app",
           "CheckerboardSignalProcessing.app")
pytestmark = [
    pytest.mark.skipif(os.environ.get("BCI2000_GUI_TESTS") != "1", reason="opens windows: set BCI2000_GUI_TESTS=1"),
    pytest.mark.skipif(not all((PROG / m).exists() for m in MODULES), reason="no BCI2000 build with the project modules"),
]

CALIBRATION_TEXT = "ก่อน"   # 4 selections: enough trials and targets for a 4-fold grouped fit
ONLINE_TEXT = "ไทย"
REPS = 2
FLASHES = 24 * REPS
# The classifier's preflight minimum: 1 block (stimulus set) + (200 + 125 - 1) // 4 blocks (window) + 2 blocks
# (margin), 16 ms each, with the 0.8 s epoch, 0.5 s pad and 4-sample blocks of the fitted model.
MIN_EPOCH_MS = (1 + (200 + 125 - 1) // 4 + 2) * 16
MIN_EPOCH_TEXT = "ยก"
SHORT_PRE_TEXT = "ก"
LOSS_PERIOD, LOSS_PHASE = 1500, 700   # one lost packet every 6 s: each hits ~14 flash windows


# ---------------------------------------------------------------------------
# running BCI2000
# ---------------------------------------------------------------------------

@dataclass
class Session:
    out: str
    system_log: str
    data: Path

    def dat(self):
        files = sorted(self.data.rglob("*.dat"))
        assert len(files) == 1, f"expected one .dat, got {files}\n{self.system_log}"
        return read_dat(files[0])

    def applog(self) -> str:
        files = sorted(self.data.rglob("*.applog"))
        assert len(files) == 1, f"expected one .applog, got {files}"
        return files[0].read_text(encoding="utf-8", errors="replace")


def no_other_bci2000():
    running = subprocess.run(["pgrep", "-f", str(PROG) + r"/.*(Source|ADC)"], capture_output=True, text=True)
    if running.stdout.strip():
        pytest.skip("another BCI2000 source module is running")


def leftover_bci2000(pgid: int) -> list:
    """BCI2000 processes still alive in the shell's process group."""
    found = subprocess.run(["pgrep", "-g", str(pgid), "-f", str(PROG)], capture_output=True, text=True)
    return found.stdout.split()


def run_bci2000(tmp: Path, sp_module: str, body: str, fakes, timeout: float) -> Session:
    """Runs one BCI2000Shell script. fakes: FakeUnicorn servers (the first one feeds the source),
    started here and always stopped, after BCI2000: a source module that loses its headset
    mid-shutdown reports a lost link."""
    proc = None
    log = tmp / "system.log"
    try:
        no_other_bci2000()
        tmp.mkdir(parents=True, exist_ok=True)
        for f in fakes:
            f.start()
        script = tmp / "run.bat"
        script.write_text(f"""Change directory $BCI2000LAUNCHDIR
Reset system
Startup system localhost --SystemLogFile={log}
Start executable UnicornSerialSource --local --TcpAddress=127.0.0.1:{fakes[0].port}
Start executable {sp_module} --local
Start executable CheckerboardSpeller --local
Wait for Connected 30
Load parameterfile "{SPELLER_PRM}"
Set parameter DataDirectory {tmp / 'data'}
Set parameter SubjectName e2e
{body}
Quit
""", encoding="utf-8")
        # The suite runs with QT_QPA_PLATFORM=offscreen for our GUI tests; BCI2000's Qt only ships cocoa.
        env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
        proc = subprocess.Popen([str(PROG / "BCI2000Shell"), str(script)], cwd=PROG, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True, env=env)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal_module.SIGKILL)       # the Operator and the modules it started
            out, _ = proc.communicate()
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            pytest.fail(f"BCI2000 did not finish:\n{out}\n{text}")
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            # Modules the Operator did not take down with it (only ours: the group is the shell's own).
            if leftover_bci2000(proc.pid):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal_module.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.communicate(timeout=10)
        for f in fakes:
            f.stop()
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    return Session(out=out, system_log=text, data=tmp / "data")


def copy_mode(text: str, extra: str = "") -> str:
    """Script body: spell `text` in copy mode, wait until the run ends."""
    return f"""Set parameter InterpretMode 2
Set parameter TextToSpell {encode(text)}
Set parameter NumberOfSequences {REPS}
Set parameter PreSequenceDuration 1s
Set parameter PostSequenceDuration 1.5s
{extra}
Set config
Wait for Resting 30
Sleep 2
Start
Wait for Suspended 120"""


def selection_seconds(text: str) -> float:
    return len(text) * (1 + FLASHES * 0.128 + 1.5 + 1)


# ---------------------------------------------------------------------------
# reading the recordings
# ---------------------------------------------------------------------------

def selection_of_samples(dat) -> np.ndarray:
    phase = dat.state("PhaseInSequence")
    starts = (phase == 1) & (np.r_[0, phase[:-1]] != 1)
    return np.cumsum(starts) - 1


@dataclass
class OfflineFlash:
    selection: int
    code: int
    onset: int
    is_target: bool
    score: float | None
    reason: str | None


def offline_flashes(dat, checkpoint) -> list:
    """Every flash in the .dat, scored the way fit_from_recording trains: extract_epochs on the whole
    recording (lost samples are NaN), then the checkpoint pipeline's decision_function."""
    ckpt = load_checkpoint(checkpoint)
    x = eeg_microvolts(dat)
    flashes = flash_events(dat)
    times = np.arange(dat.n_samples) / EEG_SFREQ
    X, kept, reasons = extract_epochs(times, x, [f.onset / EEG_SFREQ for f in flashes], ckpt.epoch_spec)
    scores = iter(ckpt.pipeline.decision_function(X) if len(X) else [])
    return [OfflineFlash(f.selection, f.code, f.onset, f.is_target, float(next(scores)) if k else None, r)
            for f, k, r in zip(flashes, kept, reasons)]


@dataclass
class OnlineEvents:
    results: list      # (selection at the event, code, type, onset, score as float32)
    rejected: list     # (selection at the event, code)


def online_events(dat) -> OnlineEvents:
    """The classifier's states, one value per sample block (the filter sets them once per block)."""
    block = int(dat.param_number("SampleBlockSize"))
    assert dat.n_samples % block == 0
    idx = np.arange(0, dat.n_samples, block)
    sel = selection_of_samples(dat)[idx]
    code = dat.state("StimulusCodeRes")[idx].astype(int)
    stype = dat.state("StimulusTypeRes")[idx].astype(int)
    onset = dat.state("ClassifierOnsetSample")[idx].astype(int)
    score = dat.state_float("ClassifierScore")[idx]
    rej = dat.state("ClassifierRejectedCode")[idx].astype(int)
    for name in ("StimulusCodeRes", "ClassifierRejectedCode"):   # constant within each block
        full = dat.state(name).reshape(-1, block)
        assert np.all(full == full[:, :1]), f"{name} changes within a block"
    results = [(int(sel[i]), int(code[i]), int(stype[i]), int(onset[i]), score[i]) for i in np.flatnonzero(code > 0)]
    rejected = [(int(sel[i]), int(rej[i])) for i in np.flatnonzero(rej > 0)]
    return OnlineEvents(results, rejected)


def assert_float32_parity(online: np.ndarray, offline: np.ndarray):
    """The online score is a float32 rounding of a double computed like the offline one (Eigen vs
    LAPACK: ~1e-12 relative apart). Allow one float32 ulp, nothing more."""
    online = np.asarray(online, dtype=np.float32)
    ref = np.asarray(offline, dtype=np.float64)
    ulp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
    err = np.abs(online.astype(np.float64) - ref)
    worst = int(np.argmax(err / ulp))
    assert np.all(err <= ulp), (f"{int((err > ulp).sum())} score(s) beyond float32 rounding; worst: online "
                                f"{online[worst]!r} offline {ref[worst]!r} ({err[worst] / ulp[worst]:.1f} ulp)")


def check_parity(dat, checkpoint, reject_reason=None):
    """Shared by every online session. reject_reason: the one offline rejection reason the session is
    built to provoke (it must hit some flashes, not all), or None for none. Returns (offline, online)."""
    off = offline_flashes(dat, checkpoint)
    on = online_events(dat)
    n_sel = len({f.selection for f in off})
    assert Counter(f.selection for f in off) == {s: FLASHES for s in range(n_sel)}

    other = Counter(f.reason for f in off if f.reason not in (None, reject_reason))
    assert not other, f"offline rejected flashes for unexpected reasons: {other}"
    kept = {f.onset: f for f in off if f.reason is None}
    lost = {(f.selection, f.code) for f in off if f.reason is not None}

    onsets = [r[3] for r in on.results]
    dup = [o for o, n in Counter(onsets).items() if n > 1]
    assert not dup, f"flashes reported more than once: onsets {dup}"
    assert set(onsets) == set(kept), (f"online but not offline: {sorted(set(onsets) - set(kept))}; "
                                      f"offline but not online: {sorted(set(kept) - set(onsets))}")
    for sel, code, stype, onset, _ in on.results:
        f = kept[onset]
        assert (code, stype) == (f.code, int(f.is_target)), f"flash at {onset}: online {code}/{stype}, offline {f}"
        assert sel == f.selection, f"the result for flash {f.code} of selection {f.selection} came in selection {sel}"
    assert_float32_parity([r[4] for r in on.results], [kept[r[3]].score for r in on.results])

    rej = Counter(on.rejected)
    assert all(n == 1 for n in rej.values()), f"flashes rejected more than once: {rej}"
    assert set(rej) == lost, (f"rejected online only: {sorted(set(rej) - lost)}; "
                              f"offline only: {sorted(lost - set(rej))}")
    if reject_reason is not None:
        assert 0 < len(lost) < len(off), f"{len(lost)} of {len(off)} flashes rejected: the session must hit some, not all"
    return off, on


def applog_selections(applog: str) -> list:
    return [(int(a), int(b)) for a, b in re.findall(r"Selection \d+: (\d+) flash result\(s\) for (\d+) flashes", applog)]


# ---------------------------------------------------------------------------
# sessions, shared by the tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """Calibration session + fit. Returns (FitResult, calibration Session)."""
    tmp = tmp_path_factory.mktemp("calibration")
    session = run_bci2000(tmp, "DummySignalProcessing", copy_mode(CALIBRATION_TEXT),
                          [FakeUnicorn(signal=noise_signal(1))], 90 + selection_seconds(CALIBRATION_TEXT))
    dat = sorted(session.data.rglob("*.dat"))
    assert len(dat) == 1, session.system_log
    fit = fit_from_recording(dat[0], "e2e", models_dir=tmp / "models", manifest_path=tmp / "models" /
                             "demo_manifest.json", allow_demo=True, log=lambda line: None)
    assert fit.bci2000_prm_path is not None and fit.bci2000_prm_path.is_file()
    return fit, session


def online_session(tmp, fit, fake, text=ONLINE_TEXT, extra=""):
    # StoppingRule 0: these tests check the classifier and the mean-score rule on every flash;
    # Bayesian stopping is tested in test_bci2000_stopping_e2e.py.
    extra = "Set parameter StoppingRule 0\n" + extra
    body = copy_mode(text).replace("Set config", f'Load parameterfile "{fit.bci2000_prm_path}"\n{extra}\nSet config', 1)
    return run_bci2000(tmp, "CheckerboardSignalProcessing", body, [fake], 90 + selection_seconds(text))


@pytest.fixture(scope="module")
def online(model, tmp_path_factory):
    fit, _ = model
    return online_session(tmp_path_factory.mktemp("online"), fit, FakeUnicorn(signal=noise_signal(2)))


@pytest.fixture(scope="module")
def lossy(model, tmp_path_factory):
    fit, _ = model
    fake = FakeUnicorn(signal=noise_signal(3), drop=lambda k: k % LOSS_PERIOD == LOSS_PHASE, burst=3)
    return online_session(tmp_path_factory.mktemp("lossy"), fit, fake)


@pytest.fixture(scope="module")
def min_epoch(model, tmp_path_factory):
    """EpochLength at the smallest value the classifier's preflight accepts."""
    fit, _ = model
    return online_session(tmp_path_factory.mktemp("min_epoch"), fit, FakeUnicorn(signal=noise_signal(4)),
                          MIN_EPOCH_TEXT, f"Set parameter EpochLength {MIN_EPOCH_MS}ms")


@pytest.fixture(scope="module")
def short_pre(model, tmp_path_factory):
    """A 0.3 s cue right after the run starts: the first flashes' padded windows (0.5 s before onset)
    begin before the first sample of the run."""
    fit, _ = model
    return online_session(tmp_path_factory.mktemp("short_pre"), fit, FakeUnicorn(signal=noise_signal(5)),
                          SHORT_PRE_TEXT, "Set parameter PreSequenceDuration 0.3s")


CHANNELS_LINE = "Set parameter Source:Signal%20Properties list ChannelNames= 8 {} // names of amplifier channels"
SWAPPED = list(EEG_CHANNELS)
SWAPPED[1], SWAPPED[2] = SWAPPED[2], SWAPPED[1]
PREFLIGHT_CASES = {
    # name: (parameter changes, restore, the classifier's expected refusals (regexes, one per error line))
    "channels_swapped": (CHANNELS_LINE.format(" ".join(SWAPPED)), CHANNELS_LINE.format(" ".join(EEG_CHANNELS)),
                         [rf'Channel 2 is "{SWAPPED[1]}"; the classifier was fitted on "{EEG_CHANNELS[1]}"',
                          rf'Channel 3 is "{SWAPPED[2]}"; the classifier was fitted on "{EEG_CHANNELS[2]}"']),
    "epoch_length_short": ("Set parameter EpochLength 800ms", "Set parameter EpochLength 1400ms",
                           [rf"EpochLength must be at least {MIN_EPOCH_MS} ms: each result is reported 1300 ms after"]),
    "notch_on": ("Set parameter NotchFilter 1", "Set parameter NotchFilter 0",
                 [r"NotchFilter must be 0: the classifier was fitted on the plain signal"]),
    "align_on": ("Set parameter AlignChannels 1", "Set parameter AlignChannels 0",
                 [r"AlignChannels must be 0: the classifier was fitted on the plain signal"]),
    "file_splitting": ("Set parameter FileSplittingCondition 1:25:00", "Set parameter FileSplittingCondition %",
                       [r"FileSplittingCondition must be empty: each run must be one \.dat file"]),
    # The speller's own refusals: evidence carried between selections would be credited to other characters.
    "accumulate_evidence": ("Set parameter AccumulateEvidence 1", "Set parameter AccumulateEvidence 0",
                            [r"AccumulateEvidence must be 0: flash numbers are reused by every selection"]),
    "minimum_evidence": ("Set parameter MinimumEvidence 2", "Set parameter MinimumEvidence 0",
                         [r"MinimumEvidence must be 0: this speller selects by the mean score per character"]),
}
# Which module's preflight must produce a case's refusals (the classifier's unless listed).
PREFLIGHT_MODULE = {"accumulate_evidence": "CheckerboardSpellerTask::Preflight",
                    "minimum_evidence": "CheckerboardSpellerTask::Preflight"}
GOOD_CONFIG = ["Set config", "Wait for Resting 10", "Log system state: ${Get system state}"]


def ReconnectingFake(**kw):
    """UnicornSerialSource stops and restarts acquisition, over a new connection, on every Set config."""
    return FakeUnicorn(reconnect=True, **kw)


@pytest.fixture(scope="module")
def preflight(model, tmp_path_factory):
    """One BCI2000 session, several configurations: AbortOnError 0 lets the script go on after a refused
    Set config. Order: no model (refused), model loaded (accepted), each bad setting (refused, then
    restored), and all restored (accepted again, so every refusal came from its own setting)."""
    fit, _ = model
    lines = ["Set variable AbortOnError 0", "Set parameter InterpretMode 1",
             "Log ===CASE no_model===", "Set config", "Sleep 1",
             "Log ===CASE loaded===", f'Load parameterfile "{fit.bci2000_prm_path}"', *GOOD_CONFIG]
    for name, (change, restore, _) in PREFLIGHT_CASES.items():
        lines += [f"Log ===CASE {name}===", change, "Set config", "Sleep 1", restore]
    lines += ["Log ===CASE restored===", *GOOD_CONFIG, "Log ===CASE end==="]
    session = run_bci2000(tmp_path_factory.mktemp("preflight"), "CheckerboardSignalProcessing",
                          "\n".join(lines), [ReconnectingFake()], 120)
    cases = dict(re.findall(r"===CASE (\w+)===\n(.*?)(?=^\S+: ===CASE )", session.system_log, re.S | re.M))
    assert set(cases) == {"no_model", "loaded", "restored", *PREFLIGHT_CASES}, session.system_log
    return cases, session


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_calibration_fits_and_exports_a_model(model):
    fit, session = model
    assert fit.demo and fit.n_flashes == len(CALIBRATION_TEXT) * FLASHES
    assert fit.n_epochs == fit.n_flashes and fit.rejected == {}
    assert fit.n_targets == len(CALIBRATION_TEXT) * 2 * REPS
    assert "Error" not in session.system_log, session.system_log
    prm = fit.bci2000_prm_path.read_text(encoding="utf-8")
    assert "RiemannFormat= p300-bci2000-riemann-v1" in prm
    assert f"RiemannChannels= 8 {' '.join(EEG_CHANNELS)} " in prm


def test_online_scores_equal_the_offline_recomputation(online, model):
    fit, _ = model
    dat = online.dat()
    assert "Error" not in online.system_log, online.system_log
    assert not np.any(dat.state("Validation") == 0)
    off, on = check_parity(dat, fit.checkpoint_path)
    assert len(on.results) == len(ONLINE_TEXT) * FLASHES and not on.rejected
    assert len({r[4] for r in on.results}) > len(on.results) // 2, "scores look constant"


def test_the_classifier_reports_its_totals_at_the_end_of_the_run(online):
    n = len(ONLINE_TEXT) * FLASHES
    assert re.search(rf"Riemann P300 classifier: {n} flash\(es\) scored\.?$", online.system_log, re.M), \
        online.system_log


def test_each_selection_decides_on_every_flash(online):
    applog = online.applog()
    assert applog_selections(applog) == [(FLASHES, FLASHES)] * len(ONLINE_TEXT), applog
    assert f"Run ended after {len(ONLINE_TEXT)} selection(s)" in applog


def test_at_the_minimum_epoch_length_each_selection_still_decides_on_every_flash(min_epoch, model):
    fit, _ = model
    assert "Error" not in min_epoch.system_log, min_epoch.system_log
    applog = min_epoch.applog()
    assert applog_selections(applog) == [(FLASHES, FLASHES)] * len(MIN_EPOCH_TEXT), applog
    check_parity(min_epoch.dat(), fit.checkpoint_path)


def test_selections_follow_the_cues(online):
    dat = online.dat()
    sel = selection_of_samples(dat)
    attended = dat.state("AttendedChar")
    selected = dat.state("SelectedChar")
    for s in range(len(ONLINE_TEXT)):
        assert len(set(attended[(sel == s) & (attended > 0)])) == 1
        assert selected[sel == s].max() > 0, f"selection {s} selected nothing"


def test_lost_samples_reject_the_same_flashes_online_and_offline(lossy, model):
    fit, _ = model
    dat = lossy.dat()
    assert np.any(dat.state("Validation") == 0)
    off, on = check_parity(dat, fit.checkpoint_path, NON_FINITE)
    n_lost = len(on.rejected)
    assert re.search(rf"Riemann P300 classifier: {len(on.results)} flash\(es\) scored, {n_lost} rejected "
                     r"\(lost samples in the window\)\.?$", lossy.system_log, re.M), lossy.system_log
    # Each selection's decision counts exactly the flashes that were scored.
    lost_per_sel = Counter(s for s, _ in on.rejected)
    expected = [(FLASHES - lost_per_sel.get(s, 0), FLASHES) for s in range(len(ONLINE_TEXT))]
    assert applog_selections(lossy.applog()) == expected, lossy.applog()


@pytest.mark.parametrize("which", ["online", "lossy"])
def test_each_selection_is_the_character_with_the_best_mean_score(which, request):
    # The speller's rule (CheckerboardSpellerTask::OnClassResult): a character's evidence is the mean of
    # the scores of its scored flashes, so a rejected flash cannot bias the decision (StimulusTask's
    # default summed them, a rejected flash counting as 0). Recomputed here from the recorded scores.
    dat = request.getfixturevalue(which).dat()
    groups = {f.onset: (f.selection, f.group) for f in flash_events(dat)}
    sel = selection_of_samples(dat)
    selected = dat.state("SelectedChar").astype(int)
    n_chars = int(dat.param("TargetDefinitions").value.__len__())
    per_sel = {}
    for _, _, _, onset, score in online_events(dat).results:
        s, group = groups[onset]
        total, count = per_sel.setdefault(s, (np.zeros(n_chars), np.zeros(n_chars)))
        total[list(group)] += float(score)
        count[list(group)] += 1
    assert per_sel, "no results"
    for s, (total, count) in per_sel.items():
        mean = np.where(count > 0, total / np.maximum(count, 1), -np.inf)
        assert selected[sel == s].max() == int(np.argmax(mean)) + 1, f"selection {s}"


def error_lines(text: str) -> list:
    return [line for line in text.splitlines() if "Error:" in line]


def test_windows_reaching_before_the_run_are_rejected_on_both_sides(short_pre, model):
    fit, _ = model
    dat = short_pre.dat()
    off, on = check_parity(dat, fit.checkpoint_path, BEFORE_BUFFER)
    assert {code for _, code in on.rejected} <= {1, 2, 3}, on.rejected       # only the first flashes
    assert re.search(rf"Riemann P300 classifier: {len(on.results)} flash\(es\) scored, {len(on.rejected)} rejected "
                     r"\(window starts before the run\)\.?$", short_pre.system_log, re.M), short_pre.system_log
    assert applog_selections(short_pre.applog()) == [(FLASHES - len(on.rejected), FLASHES)], short_pre.applog()


def test_preflight_refuses_without_a_model(preflight):
    cases, _ = preflight
    text = cases["no_model"]
    errors = error_lines(text)
    assert len(errors) == 1 and "RiemannP300Classifier::Preflight: No classifier loaded" in errors[0], text
    assert "Could not set configuration" in text, text


@pytest.mark.parametrize("name", ["loaded", "restored"])
def test_preflight_accepts_the_fitted_model(preflight, name):
    cases, _ = preflight
    text = cases[name]
    assert not error_lines(text) and "Script error" not in text, text
    assert "system state: Resting" in text, text
    assert "Riemann P300 classifier: model " in text, text


@pytest.mark.parametrize("name", list(PREFLIGHT_CASES))
def test_preflight_refuses(preflight, name):
    cases, _ = preflight
    text = cases[name]
    expected = PREFLIGHT_CASES[name][2]
    errors = error_lines(text)
    module = PREFLIGHT_MODULE.get(name, "RiemannP300Classifier::Preflight")
    assert all(module in e for e in errors), f"expected only {module}'s refusal:\n{text}"
    assert len(errors) == len(expected), text
    for pattern in expected:
        assert any(re.search(pattern, e) for e in errors), f"no error matches {pattern!r}:\n{text}"
    assert "Could not set configuration" in text, text
    assert [line for line in text.splitlines() if "Script error" in line and "Could not set configuration" not in line] \
        == [], f"a parameter change of the case itself failed:\n{text}"
