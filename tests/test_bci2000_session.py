"""
The BCI2000 session tools, without running BCI2000:
  pipeline/bci2000_session.py   a CheckerboardSpeller recording as inference-log records (what
                                stopping/summarize_experiment.py reads), built here from in-memory
                                recordings whose every state is known;
  stopping/summarize_experiment the summary of such a session, incl. the closed-loop "shown" ITR;
  stopping/evaluate_recording   the .dat replay loader;
  tools/h13_montage_check.py    montage and units checks on planted signals;
  bci2000/run_session.sh        its refusals, which happen before BCI2000 starts.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pipeline import bci2000_session as bs
from pipeline import events as ev
from pipeline.bci2000_dat import DatFile, Param, StateDef
from pipeline.constants import EEG_CHANNELS
from pipeline.manifest import REPO_ROOT
from stopping.summarize_experiment import summarize_inference

import sys
sys.path.insert(0, str(REPO_ROOT / "tools"))
import h13_montage_check as h13  # noqa: E402

N = 72
CHARS = [chr(0x0E01 + i) for i in range(N)]
B = 4                                     # SampleBlockSize


def build(selections, *, bayesian=True, lm="transformer", run_id=777, text_result="", block=B):
    """
    selections: list of dicts with keys flashes [(code, group)], results [(code, score)] (arrival order),
    rejected [codes], used (StoppingFlashes), reason (StopReason), selected (0-based or -1), attended
    (0-based or -1), pmax. Lays them out block by block: pre-sequence, flashes (4 blocks each), then
    post-sequence blocks carrying the results.
    """
    st = {k: [] for k in ("PhaseInSequence", "StimulusCode", "StimulusType", "AttendedChar", "SelectedChar",
                          "StimulusCodeRes", "ClassifierScore", "ClassifierRejectedCode", "StoppingFlashes",
                          "StopReason", "PosteriorMax", "SpellerRunID", *(f"FlashChar{i}" for i in range(1, 7)))}

    def emit(nblocks, **vals):
        for _ in range(nblocks * block):
            for k in st:
                st[k].append(vals.get(k, 0))

    emit(2)
    for s in selections:
        common = dict(AttendedChar=s["attended"] + 1, SpellerRunID=run_id)
        emit(2, PhaseInSequence=1, **common)
        for code, group in s["flashes"]:
            lit = {f"FlashChar{i + 1}": g + 1 for i, g in enumerate(group)}
            emit(2, PhaseInSequence=2, StimulusCode=code, StimulusType=int(s["attended"] in group), **lit, **common)
            emit(2, PhaseInSequence=2, **common)
        for code, score in s["results"]:
            emit(1, PhaseInSequence=3, StimulusCodeRes=code,
                 ClassifierScore=int(np.float32(score).view(np.uint32)), **common)
        for code in s.get("rejected", []):
            emit(1, PhaseInSequence=3, ClassifierRejectedCode=code, **common)
        emit(2, PhaseInSequence=3, SelectedChar=s["selected"] + 1, StoppingFlashes=s.get("used", 0),
             StopReason=s.get("reason", 0), PosteriorMax=int(np.float32(s.get("pmax", 0.5)).view(np.uint32)),
             **common)
    states = {k: np.asarray(v, dtype=np.uint32) for k, v in st.items()}
    defs = {k: StateDef(k, 32, 0, 0, 0) for k in states}
    params = {
        "TargetDefinitions": Param("Application", "matrix", "TargetDefinitions", [[c, c] for c in CHARS]),
        "SampleBlockSize": Param("Source", "int", "SampleBlockSize", str(block)),
        "SamplingRate": Param("Source", "int", "SamplingRate", "250Hz"),
        "ChannelNames": Param("Source", "list", "ChannelNames", list(EEG_CHANNELS)),
        "TextResult": Param("Application", "string", "TextResult", text_result),
        "StoppingRule": Param("Application", "int", "StoppingRule", "1" if bayesian else "0"),
        "LanguageModel": Param("Application", "int", "LanguageModel", "1" if lm == "transformer" else "0"),
        "StoppingLambda": Param("Application", "float", "StoppingLambda", "0.7"),
        "StoppingTheta": Param("Application", "float", "StoppingTheta", "0.9"),
        "StoppingTMin": Param("Application", "int", "StoppingTMin", "6"),
        "StoppingTMax": Param("Application", "int", "StoppingTMax", "170"),
        "StoppingLLRClip": Param("Application", "float", "StoppingLLRClip", "8"),
        "RiemannModelID": Param("Filtering", "string", "RiemannModelID", "ab" * 32),
    }
    n = len(states["PhaseInSequence"])
    return DatFile(path=Path("S01S001R01.dat"), signal_raw=np.zeros((n, 8)), signal_uv=np.zeros((n, 8)),
                   states=states, state_defs=defs, params=params, data_format="float32")


def groups(k, start=0):
    return [(code, tuple(range((start + code * 6) % 66, (start + code * 6) % 66 + 6))) for code in range(1, k + 1)]


def two_selections():
    return [
        dict(flashes=groups(10), results=[(c, 1.0 + c) for c in range(1, 9)], rejected=[9], used=6, reason=1,
             selected=6, attended=6, pmax=0.93),
        dict(flashes=groups(12, 3), results=[(c, -1.0) for c in range(1, 12)], used=11, reason=3,
             selected=20, attended=5, pmax=0.4),
    ]


def prior_line(sel, peak):
    p = np.full(N, 0.5 / (N - 1))
    p[peak] = 0.5
    return f"PRIOR {sel} " + " ".join(f"{v:.9g}" for v in p)


APPLOG = "\n".join(["RUN 111", prior_line(1, 0), prior_line(2, 1),           # another run: must be ignored
                    "RUN 777", "Selection 1 prior ...", prior_line(1, 6), prior_line(2, 9)])


def test_records_follow_the_recording():
    recs = bs.session_records(build(two_selections()), APPLOG)
    cfg = recs[0]
    assert cfg["event"] == "service_config" and cfg["characters"] == CHARS
    assert cfg["manifest"]["language_model"] == {"type": "transformer"}
    assert cfg["manifest"]["stopping"] == {"lam": 0.7, "theta": 0.9, "t_min": 6, "t_max": 170, "llr_clip": 8.0}
    starts = [r for r in recs if r["event"] == "trial_start"]
    commits = [r for r in recs if r["event"] == "char_committed"]
    assert [s["context"] for s in starts] == ["", CHARS[6]]
    assert [s["target_char_idx"] for s in starts] == [6, 5]
    # this run's priors (RUN 777), not the other run's
    assert starts[0]["target_prior_rank"] == 1 and starts[0]["target_prior_prob"] == pytest.approx(0.5)
    assert starts[1]["target_prior_rank"] == 2
    c1, c2 = commits
    assert (c1["char_idx"], c1["reason"], c1["n_flashes_used"], c1["n_flashes_presented"], c1["correct"]) == \
        (6, ev.REASON_THRESHOLD_CROSSED, 6, 10, True)
    assert (c2["char_idx"], c2["reason"], c2["n_flashes_used"], c2["n_flashes_presented"], c2["correct"]) == \
        (20, ev.REASON_TRIAL_ENDED, 11, 12, False)
    assert c1["max_prob"] == pytest.approx(0.93, rel=1e-6)
    ups = [r for r in recs if r["event"] == "flash_update" and r["trial_id"] == 0]
    assert [u["t"] for u in ups] == list(range(1, 7))                      # only the results the controller used
    assert [u["score"] for u in ups] == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert ups[0]["group"] == list(groups(10)[0][1]) and ups[0]["pi"] == pytest.approx(6 / N)
    rej = [r for r in recs if r["event"] == "flash_rejected"]
    assert [(r["trial_id"], r["flash_idx"]) for r in rej] == [(0, 9)]


def test_summary_of_a_bci2000_session_reports_used_and_shown_flashes():
    summary = summarize_inference(bs.session_records(build(two_selections()), APPLOG), seconds_per_flash=0.128)
    assert (summary["n_correct"], summary["n_scored"]) == (1, 2)
    assert summary["mean_flashes"] == pytest.approx(8.5)                    # (6 + 11) / 2, used
    assert summary["mean_flashes_presented"] == pytest.approx(11.0)         # (10 + 12) / 2, shown
    assert summary["itr_presented_bits_per_min"] < summary["itr_bits_per_min"]
    assert summary["reasons"] == {ev.REASON_THRESHOLD_CROSSED: 1, ev.REASON_TRIAL_ENDED: 1}
    assert summary["lm"]["target_ranked_first"] == 1


def test_uniform_prior_and_the_mean_score_rule():
    dat = build(two_selections(), lm="uniform")
    starts = [r for r in bs.session_records(dat, None) if r["event"] == "trial_start"]
    assert all(s["target_prior_prob"] == pytest.approx(1 / N) for s in starts)
    dat = build(two_selections(), bayesian=False)
    recs = bs.session_records(dat, None)
    commits = [r for r in recs if r["event"] == "char_committed"]
    assert {c["reason"] for c in commits} == {bs.MEAN_RULE_REASON}
    assert [c["n_flashes_used"] for c in commits] == [8, 11]               # every result counts
    assert recs[0]["manifest"]["stopping"] is None
    assert all("target_prior_prob" not in r for r in recs if r["event"] == "trial_start")   # no log: no prior


def test_without_a_run_id_or_applog_there_are_no_logged_priors():
    assert bs.logged_priors(build(two_selections(), run_id=0), APPLOG) == {}
    assert bs.logged_priors(build(two_selections()), None) == {}
    assert sorted(bs.logged_priors(build(two_selections()), APPLOG)) == [0, 1]


def test_a_selection_with_nothing_selected_has_no_commit():
    sels = two_selections()
    sels[1]["selected"] = -1
    commits = [r for r in bs.session_records(build(sels), APPLOG) if r["event"] == "char_committed"]
    assert len(commits) == 1


# ---------------------------------------------------------------------------
# replay loader
# ---------------------------------------------------------------------------

def test_replay_loads_a_bci2000_recording(tmp_path):
    import bci2000_dat_writer as w
    from stopping.evaluate_recording import load_recording_for_replay
    session = w.build_session(w.checkerboard_selections(2, 1, seed=3))
    path = w.write_speller_dat(tmp_path / "s.dat", session, w.planted_signal(session, seed=3), lost=[(900, 903)])
    rec = load_recording_for_replay(path, N)
    assert len(rec.flashes) == len(session.flashes)
    assert rec.targets == {f["selection"]: f["attended"] for f in session.flashes}
    assert [(f.trial_id, f.flash_idx, f.group) for f in rec.flashes] == \
        [(f["selection"], f["code"], tuple(f["group"])) for f in session.flashes]
    assert np.isnan(rec.data[900:903]).all() and np.isfinite(np.delete(rec.data, range(900, 903), axis=0)).all()
    assert rec.timestamps[250] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# montage and units checks
# ---------------------------------------------------------------------------

def planted(fs=250.0, seconds=60, *, swap=None, seed=0):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = rng.normal(0, 4, size=(n, 8))
    third = n // 3
    ch = {c: i for i, c in enumerate(EEG_CHANNELS)}
    if swap:
        a, b = swap
        ch[a], ch[b] = ch[b], ch[a]
    blinks = np.zeros(n)
    for k in range(0, third, int(fs)):
        blinks[k:k + 50] += np.hanning(50) * 80
    x[:, ch["Fz"]] += blinks
    x[:, ch["PO8"]] += blinks * 0.2
    x[third:2 * third, ch["Oz"]] += 25 * np.sin(2 * np.pi * 10 * t[third:2 * third])
    x[third:2 * third, ch["PO7"]] += 12 * np.sin(2 * np.pi * 10 * t[third:2 * third])
    for c, g in (("C3", 30), ("C4", 28)):
        x[2 * third:, ch[c]] += rng.normal(0, g, size=(n - 2 * third)) * (np.sin(2 * np.pi * 60 * t[2 * third:]) ** 2)
    return x, fs


def test_the_montage_check_passes_a_correct_cap_and_catches_a_swapped_one():
    x, fs = planted()
    m = h13.montage_report(x, fs, list(EEG_CHANNELS))
    assert m["largest"] == {"blink": "Fz", "alpha": "Oz", "jaw": "C3"} and all(m["passed"].values())
    m = h13.montage_report(*planted(swap=("Fz", "Pz")), list(EEG_CHANNELS))
    assert m["largest"]["blink"] == "Pz" and not m["passed"]["blink"]
    with pytest.raises(ValueError, match="too short"):
        h13.montage_report(x[:1000], fs, list(EEG_CHANNELS))


def test_the_units_check_rates_channels_with_the_quality_thresholds():
    x, fs = planted()
    u = h13.units_report(x[len(x) // 3:2 * len(x) // 3], fs, list(EEG_CHANNELS))
    assert set(u["levels"]) == {"green"}
    assert set(h13.units_report(x * 1e-6, fs, list(EEG_CHANNELS))["levels"]) == {"red"}   # volts, not uV
    assert set(h13.units_report(x * 1e3, fs, list(EEG_CHANNELS))["levels"]) == {"red"}


def test_the_montage_tool_reads_a_dat(tmp_path):
    import bci2000_dat_writer as w
    session = w.build_session(w.checkerboard_selections(2, 1, seed=1))
    x, _ = planted(seconds=session.n / 250 + 1)
    path = w.write_speller_dat(tmp_path / "c.dat", session, x[:session.n].astype(np.float32), lost=[(500, 504)])
    y, fs, channels, filled, synthetic = h13.load(path)
    assert fs == 250 and channels == list(EEG_CHANNELS) and filled == 4 and np.isfinite(y).all()
    assert synthetic is None
    path = w.write_speller_dat(tmp_path / "f.dat", session, x[:session.n].astype(np.float32), synthetic_gyro=True)
    assert h13.load(path)[4]


# ---------------------------------------------------------------------------
# the launcher refuses before starting anything
# ---------------------------------------------------------------------------

SCRIPT = REPO_ROOT / "bci2000" / "run_session.sh"
PROG = Path(os.environ.get("BCI2000_ROOT", Path.home() / "BCI2000" / "trunk")) / "prog"


@pytest.mark.skipif(not (PROG / "CheckerboardSpeller.app").exists(), reason="no BCI2000 build")
@pytest.mark.parametrize("args, message", [
    ([], "Run one BCI2000 session"),
    (["calibrate", "--subject", "S 1", "--text", "ก"], "letters and digits"),
    (["calibrate", "--subject", "S1"], "needs --text"),
    (["calibrate", "--subject", "S1", "--text", "กAB"], "not in the matrix: A B"),
    (["online", "--subject", "S1", "--text", "ก"], "needs --model"),
    (["online", "--subject", "S1", "--model", "/nonexistent/x.prm", "--text", "ก"], "No such file|no model"),
    (["online", "--subject", "S1", "--model", "x.prm", "--free", "--text", "ก"], "exclude each other|no model|No such"),
    (["calibrate", "--subject", "S1", "--text", "ก", "--condition", "best"], "--condition must be"),
    (["calibrate", "--subject", "S1", "--text", "ก", "--reps", "0"], "--reps must be"),
])
def test_the_launcher_refuses_bad_requests(args, message, tmp_path):
    out = subprocess.run(["bash", str(SCRIPT), *args, "--fake", "--data", str(tmp_path)] if args and "--data" not in args
                         else ["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=60)
    assert out.returncode == 2, out.stdout + out.stderr
    import re
    assert re.search(message, out.stdout + out.stderr), out.stdout + out.stderr
