"""
End to end, closed loop: BCI2000 spells with Bayesian dynamic stopping and the Thai Transformer prior
(CheckerboardSpeller, StoppingRule 1, LanguageModel 1), fed by the fake headset through
UnicornSerialSource and scored by CheckerboardSignalProcessing, after a calibration session and a fit
with --lm-transformer-checkpoint. Every decision is then replayed from the recording with the Python
originals -- stopping/transformer_lm.ThaiTransformerLM for the prior of the text spelled so far, and
stopping/bayesian_stopping.BayesianStoppingController fed the recorded scores in the order the speller
received them -- and must agree: the prior (logged in the .applog), the flash at which it stopped, why,
and the character selected. Early stops must also end the flashing early.

Three sessions cover the three ways a character ends: the threshold (a low theta), the forced stop at
t_max, and the flashes running out (argmax). Opt-in like the other BCI2000 tests (BCI2000_GUI_TESTS=1);
also needs models/thai_gpt_checkpoint.pt.
"""

import re

import numpy as np
import pytest

from pipeline.manifest import REPO_ROOT, load_characters_file
from stopping.bayesian_stopping import BayesianStoppingController
from test_bci2000_classifier_e2e import (FakeUnicorn, copy_mode, noise_signal, online_events, pytestmark,  # noqa: F401
                                         run_bci2000, selection_of_samples, selection_seconds, flash_events)
from pipeline.fit_from_recording import fit_from_recording

CHECKPOINT = REPO_ROOT / "models" / "thai_gpt_checkpoint.pt"
CHARACTERS = load_characters_file(REPO_ROOT / "collection" / "characters_th.json")
CALIBRATION_TEXT = "ก่อน"
FLASHES_PER_REP = 24


@pytest.fixture(scope="module")
def lm():
    if not CHECKPOINT.is_file():
        pytest.skip("no trained checkpoint at models/thai_gpt_checkpoint.pt")
    pytest.importorskip("torch")
    from stopping.transformer_lm import ThaiTransformerLM
    return ThaiTransformerLM(str(CHECKPOINT), CHARACTERS)


@pytest.fixture(scope="module")
def model(lm, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("calibration_lm")
    session = run_bci2000(tmp, "DummySignalProcessing", copy_mode(CALIBRATION_TEXT),
                          [FakeUnicorn(signal=noise_signal(11))], 90 + selection_seconds(CALIBRATION_TEXT))
    dat = sorted(session.data.rglob("*.dat"))
    assert len(dat) == 1, session.system_log
    fit = fit_from_recording(dat[0], "e2e", models_dir=tmp / "models", manifest_path=tmp / "models" /
                             "demo_manifest.json", allow_demo=True, lm_transformer_checkpoint=CHECKPOINT,
                             log=lambda line: None)
    prm = fit.bci2000_prm_path.read_text(encoding="utf-8")
    assert "LanguageModel= 1" in prm and "StoppingRule= 1" in prm
    return fit


def session(model, tmp_path_factory, name, text, extra, seed):
    body = copy_mode(text).replace("Set config", f'Load parameterfile "{model.bci2000_prm_path}"\n{extra}\nSet config', 1)
    return run_bci2000(tmp_path_factory.mktemp(name), "CheckerboardSignalProcessing", body,
                       [FakeUnicorn(signal=noise_signal(seed))], 90 + selection_seconds(text))


@pytest.fixture(scope="module")
def threshold(model, tmp_path_factory):
    # theta below the LM-blended starting maximum: stops as soon as t_min allows
    return session(model, tmp_path_factory, "threshold", "ไทย", "Set parameter StoppingTheta 0.05", 21)


@pytest.fixture(scope="module")
def t_max(model, tmp_path_factory):
    return session(model, tmp_path_factory, "tmax", "กิน",
                   "Set parameter StoppingTheta 0.999\nSet parameter StoppingTMax 10", 22)


@pytest.fixture(scope="module")
def ran_out(model, tmp_path_factory):
    return session(model, tmp_path_factory, "ranout", "ก",
                   "Set parameter StoppingTheta 0.999\nSet parameter NumberOfSequences 1", 23)


def replay(dat, lm):
    """Every selection, decided again by the Python controller from the recording. Returns a list of
    dicts: the speller's recorded decision next to the replay's."""
    p = lambda name: dat.param_str(name)  # noqa: E731
    params = dict(lam=float(p("StoppingLambda")), theta=float(p("StoppingTheta")), t_min=int(p("StoppingTMin")),
                  t_max=int(p("StoppingTMax")), llr_clip=float(p("StoppingLLRClip")),
                  sigmoid_a=float(p("RiemannPlattA")), sigmoid_b=float(p("RiemannPlattB")))
    enter = [row[1] for row in dat.param("TargetDefinitions").value]
    groups = {(f.selection, f.code): f.group for f in flash_events(dat)}
    onsets = {}
    for f in flash_events(dat):
        onsets.setdefault(f.selection, []).append(f.onset)
    results = {}
    for s, code, _, _, score in online_events(dat).results:
        results.setdefault(s, []).append((code, float(score)))
    sel = selection_of_samples(dat)
    selected, flashes_used, reason = (dat.state(n).astype(int) for n in ("SelectedChar", "StoppingFlashes", "StopReason"))
    context = p("TextResult")
    out = []
    for s in range(int(sel.max()) + 1):
        mask = sel == s
        prior = lm.get_prior(context)
        c = BayesianStoppingController(n_chars=len(CHARACTERS), **params)
        c.reset(prior)
        stop_at, decision, why = None, None, 0
        for code, score in results.get(s, []):
            group = groups[(s, code)]
            stop, idx = c.update(list(group), score, len(group) / len(CHARACTERS))
            if stop:
                stop_at, decision = c.t, idx
                why = 1 if (c.t >= c.t_min and np.exp(c.log_p.max()) >= c.theta) else 2
                break
        if decision is None and c.t >= 1:
            decision, why = int(np.argmax(c.log_p)), 3
        rec_sel = int(selected[mask].max())
        out.append(dict(prior=prior, decision=decision, why=why, stop_at=stop_at, used=c.t,
                        rec_selected=rec_sel - 1, rec_used=int(flashes_used[mask].max()),
                        rec_reason=int(reason[mask].max()), presented=len(onsets.get(s, []))))
        if rec_sel:
            context += enter[rec_sel - 1]
    return out


def logged_priors(applog: str) -> dict:
    return {int(m.group(1)): np.array(m.group(2).split(), dtype=float)
            for m in re.finditer(r"^PRIOR (\d+) (.*)$", applog, re.M)}


@pytest.mark.parametrize("which", ["threshold", "t_max", "ran_out"])
def test_every_decision_replays_identically_in_python(which, request, lm):
    s = request.getfixturevalue(which)
    dat = s.dat()
    rows = replay(dat, lm)
    priors = logged_priors(s.applog())
    assert rows and len(priors) == len(rows), s.applog()
    for i, row in enumerate(rows):
        np.testing.assert_allclose(priors[i + 1], row["prior"], rtol=2e-5, atol=2e-6)   # C++ vs PyTorch prior
        assert row["rec_selected"] == row["decision"], (i, row)
        assert row["rec_reason"] == row["why"], (i, row)
        assert row["rec_used"] == row["used"], (i, row)
    return rows


def test_the_threshold_stops_early_and_ends_the_flashing(threshold, lm):
    dat = threshold.dat()
    for row in replay(dat, lm):
        assert row["why"] == 1 and row["stop_at"] == int(dat.param_str("StoppingTMin"))
        # Results lag their flashes by ~1.3 s (about 10 flashes), so a few more flashes are shown after
        # the stopping flash, but far fewer than the full 2 x 24.
        assert row["stop_at"] < row["presented"] < 2 * FLASHES_PER_REP, row


def test_t_max_forces_the_stop(t_max, lm):
    for row in replay(t_max.dat(), lm):
        assert row["why"] == 2 and row["stop_at"] == 10 and row["presented"] < 2 * FLASHES_PER_REP, row


def test_running_out_of_flashes_selects_the_argmax(ran_out, lm):
    for row in replay(ran_out.dat(), lm):
        assert row["why"] == 3 and row["stop_at"] is None and row["presented"] == FLASHES_PER_REP, row
        assert row["used"] == FLASHES_PER_REP, row
