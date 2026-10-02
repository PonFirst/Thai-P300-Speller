"""
fit_from_recording on a BCI2000 .dat (CheckerboardSpeller-shaped, written by
tests/bci2000_dat_writer.py) with a planted target ERP: artifacts incl. <id>_bci2000.prm, synthetic
(fake-headset gyroscope) data handling, --force protection of the .prm, lost-sample rejections, the
fit report, and the exported parameter file reproducing the fitted model when parsed back.

Every output goes under tmp_path; the CLI tests also point REPO_ROOT at an empty directory and check
it stays empty.
"""

import json
import random
import string
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bci2000_dat_writer as w
from pipeline import bci2000_dat as bd
from pipeline import bci2000_export as bx
from pipeline import experiment_log as xlog
from pipeline import fit_from_recording as ffr
from pipeline import manifest as mf
from pipeline.classifier_io import load_checkpoint
from pipeline.constants import EEG_CHANNELS
from pipeline.epoching import EpochSpec, extract_epochs, preprocess_segment

SPEC = EpochSpec()
N_SEL, N_REPS = 5, 2


def session_and_signal(seed=11):
    session = w.build_session(w.checkerboard_selections(N_SEL, N_REPS, seed=seed))
    return session, w.planted_signal(session, seed=seed)


def windows_touching(session, lost):
    """Flashes whose padded epoch window [onset - n_pad, onset + n_times + n_pad) contains a lost sample."""
    bad = np.zeros(session.n, bool)
    for a, b in lost:
        bad[a:b] = True
    return sum(bool(bad[f["onset"] - SPEC.n_pad:f["onset"] + SPEC.n_times + SPEC.n_pad].any())
               for f in session.flashes)


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    d = tmp_path_factory.mktemp("dat")
    session, sig = session_and_signal()
    f = session.flashes[2 * 48 + 20]
    lost = [(f["onset"] + 10, f["onset"] + 13)]          # 3 lost samples inside selection 2
    path = w.write_speller_dat(d / "subject_001.dat", session, sig, lost=lost, text_to_spell="ก่อนนี้")
    return SimpleNamespace(path=path, session=session, lost=lost, dat=bd.read_dat(path))


@pytest.fixture(scope="module")
def fitted(rec, tmp_path_factory):
    out = tmp_path_factory.mktemp("fit")
    lines = []
    result = ffr.fit_from_recording(rec.path, "subject_001", models_dir=out / "models",
                                    manifest_path=out / "session_manifest.json", n_splits=3, log=lines.append)
    return SimpleNamespace(result=result, out=out, models=out / "models", lines=lines)


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------

def test_artifacts_are_written_including_the_bci2000_parameter_file(fitted, rec):
    r, models = fitted.result, fitted.models
    assert sorted(p.name for p in models.iterdir()) == [
        "subject_001_bci2000.prm", "subject_001_classifier.joblib", "subject_001_fit_report.json",
        "subject_001_platt.json"]
    assert r.bci2000_prm_path == models / "subject_001_bci2000.prm"
    assert r.checkpoint_path == models / "subject_001_classifier.joblib"
    assert not r.demo and r.warnings == []
    manifest = mf.load_manifest(fitted.out / "session_manifest.json")
    assert manifest.fingerprint == r.fingerprint and manifest.lm_config == {"type": "uniform"}
    cal = mf.load_calibration_file(manifest.calibration_path)
    assert cal["classifier_sha256"] == mf.sha256_file(r.checkpoint_path)
    assert cal["source_recording_sha256"] == mf.sha256_file(rec.path)
    ckpt = load_checkpoint(r.checkpoint_path)
    assert ckpt.epoch_spec == SPEC
    assert ckpt.trained_on == {"recording_sha256": mf.sha256_file(rec.path), "n_epochs": r.n_epochs,
                               "n_targets": r.n_targets, "n_trials": N_SEL}
    assert any("BCI2000 classifier parameters" in line for line in fitted.lines)
    summary = ffr.format_summary(r)
    assert "BCI2000 model" in summary and str(r.bci2000_prm_path) in summary


def test_planted_erp_is_found(fitted):
    r = fitted.result
    assert r.auc > 0.8 and r.a > 0 and 0 <= r.ece < 0.2
    assert r.n_trials == N_SEL and r.n_splits == 3


def test_lost_samples_reject_exactly_the_epochs_whose_window_they_touch(fitted, rec):
    r = fitted.result
    expected = windows_touching(rec.session, rec.lost)
    assert expected >= 10                                   # the gap really sits inside epoch windows
    assert r.rejected == {"lost_samples": expected}
    assert "non_finite" not in r.rejected
    assert r.n_flashes == N_SEL * 48
    assert r.n_epochs + expected == r.n_flashes


def test_fit_report_records_the_bci2000_recording(fitted, rec):
    r = fitted.result
    report = json.loads(r.report_path.read_text(encoding="utf-8"))
    assert r.report_path == fitted.models / "subject_001_fit_report.json"
    assert report["format"] == xlog.FIT_REPORT_FORMAT and report["demo"] is False
    recd = report["recording"]
    assert recd["kind"] == "bci2000"
    assert recd["sha256"] == mf.sha256_file(rec.path)
    assert recd["n_rows"] == recd["n_eeg_samples"] == rec.session.n
    assert recd["n_non_eeg_rows"] == 3 and recd["n_flash_markers"] == N_SEL * 48
    assert recd["bci2000"] == {
        "text_to_spell": "ก่อนนี้", "n_selections": N_SEL, "n_samples": rec.session.n, "lost_samples": 3,
        "synthetic": False, "synthetic_reason": None, "counter_breaks": 0, "truncated_tail_bytes": 0,
        "subject": "subj01", "session": "001", "run": "02",
        "number_of_sequences": "2", "stimulus_duration": "64ms", "isi": "60ms"}
    assert report["epochs"]["rejected"] == r.rejected
    art = report["artifacts"]["bci2000_parameters"]
    assert art == {"path": r.bci2000_prm_path.as_posix(), "sha256": mf.sha256_file(r.bci2000_prm_path)}
    assert report["artifacts"]["classifier"]["sha256"] == mf.sha256_file(r.checkpoint_path)
    assert any("BCI2000 classifier parameters" in line for line in report["console"])


def test_epochs_are_cut_at_the_flash_onset_sample(rec):
    loaded = ffr.load_bci2000_recording(rec.path)
    assert loaded.kind == "bci2000"
    assert [round(f.onset * 250) for f in loaded.flashes] == [f["onset"] for f in rec.session.flashes]
    assert [f.trial_id for f in loaded.flashes] == [f["selection"] for f in rec.session.flashes]
    assert [f.flash_idx for f in loaded.flashes] == [f["code"] for f in rec.session.flashes]
    assert [f.is_target for f in loaded.flashes] == [f["is_target"] for f in rec.session.flashes]
    X, kept, _ = extract_epochs(loaded.timestamps, loaded.data, [f.onset for f in loaded.flashes], SPEC)
    raw = rec.dat.signal_uv
    kept_flashes = [f for f, k in zip(rec.session.flashes, kept) if k]
    for j in (0, 1, 47, 100, len(kept_flashes) - 1):
        o = kept_flashes[j]["onset"]
        np.testing.assert_array_equal(X[j], preprocess_segment(raw[o - SPEC.n_pad:o + SPEC.n_times + SPEC.n_pad], SPEC))


# ---------------------------------------------------------------------------
# the exported parameter file
# ---------------------------------------------------------------------------

def parse_prm(path) -> dict:
    params = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = bd.parse_param_line(line)
            assert p is not None, line
            assert p.section == bx.PRM_SECTION
            params[p.name] = p
    return params


def _matrix(p) -> np.ndarray:
    rows = [[float(v) for v in row] for row in p.value]
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 0))


def model_from_prm(params) -> bx.ExportedModel:
    base = params["RiemannBaseline"].value
    return bx.ExportedModel(
        channels=tuple(params["RiemannChannels"].value),
        sfreq=float(params["RiemannSamplingRate"].value),
        n_times=int(params["RiemannEpochSamples"].value),
        n_pad=int(params["RiemannPadSamples"].value),
        baseline=None if not base else (int(base[0]), int(base[1])),
        sos=_matrix(params["RiemannFilterSOS"]), zi=_matrix(params["RiemannFilterZI"]),
        padlen=int(params["RiemannFilterPadlen"].value),
        xdawn_filters=_matrix(params["RiemannXdawnFilters"]), prototypes=_matrix(params["RiemannPrototypes"]),
        ref_invsqrt=_matrix(params["RiemannRefInvSqrt"]),
        lda_weights=np.array([float(v) for v in params["RiemannLDAWeights"].value]),
        lda_intercept=float(params["RiemannLDAIntercept"].value),
        platt_a=float(params["RiemannPlattA"].value), platt_b=float(params["RiemannPlattB"].value),
        model_id=params["RiemannModelID"].value)


def test_exported_prm_parses_back_to_the_exact_model(fitted):
    r = fitted.result
    params = parse_prm(r.bci2000_prm_path)
    assert params["RiemannFormat"].value == bx.EXPORT_FORMAT
    assert params["RiemannModelID"].value == mf.sha256_file(r.checkpoint_path)
    got = model_from_prm(params)
    ref = bx.extract_model(r.checkpoint_path, r.calibration_path)
    for name in ref.__dataclass_fields__:
        a, b = getattr(got, name), getattr(ref, name)
        if isinstance(b, np.ndarray):
            assert a.shape == b.shape and np.array_equal(a, b), name      # repr() floats round-trip exactly
        else:
            assert a == b, name
    assert got.channels == tuple(EEG_CHANNELS) and got.sfreq == 250.0
    assert (got.n_times, got.n_pad, got.baseline) == (SPEC.n_times, SPEC.n_pad, None)
    cal = mf.load_calibration_file(r.calibration_path)
    assert (got.platt_a, got.platt_b) == (cal["a"], cal["b"]) == (r.a, r.b)
    n = got.prototypes.shape[0] + got.xdawn_filters.shape[0]
    assert got.ref_invsqrt.shape == (n, n) and got.lda_weights.shape == (n * (n + 1) // 2,)


def test_exported_provenance_decodes_to_the_fit(fitted, rec):
    prov = json.loads(parse_prm(fitted.result.bci2000_prm_path)["RiemannProvenance"].value)
    assert prov == {"recording": "subject_001.dat", "recording_sha256": mf.sha256_file(rec.path),
                    "subject_id": "subject_001", "demo": False, "n_epochs": fitted.result.n_epochs,
                    "n_targets": fitted.result.n_targets, "out_of_fold_auc": round(fitted.result.auc, 4)}


def test_parsed_prm_scores_like_the_checkpoint_pipeline(fitted, rec):
    r = fitted.result
    m = model_from_prm(parse_prm(r.bci2000_prm_path))
    ckpt = load_checkpoint(r.checkpoint_path)
    raw = rec.dat.signal_uv
    targets = [f for f in rec.session.flashes if f["is_target"]][:4]
    others = [f for f in rec.session.flashes if not f["is_target"]][::37][:4]
    rng = np.random.default_rng(0)
    windows = [raw[f["onset"] - m.n_pad:f["onset"] + m.n_times + m.n_pad] for f in targets + others]
    windows.append(rng.normal(0, 30, size=windows[0].shape))           # an unrelated window
    windows.append(windows[0] * 3.0 + 50.0)                             # scaled and offset
    scores = []
    for win in windows:
        expected = float(ckpt.pipeline.decision_function(preprocess_segment(win, ckpt.epoch_spec)[None])[0])
        got = bx.reference_score(m, win)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)
        scores.append(got)
    assert np.mean(scores[:4]) > np.mean(scores[4:8])                   # targets score higher


ODD_STRINGS = [
    "", " ", "plain", "with space", "%", "%%", "%25", "%zz", "{", "}", "{ a }", "}{", '"quoted"', "it's",
    "ก่อน", "ภาษาไทย 50%", "a//b", "tab\there", "new\nline\r\n", "[x]", "emoji 😀", "é", "\x7f",
    "~!@#$^&*()_+`-=[]\\|;:,.<>/?", json.dumps({"recording": "x y.dat", "n": [1, 2], "ok": True}),
]


def random_strings(n, seed=0):
    rnd = random.Random(seed)
    pool = string.printable + "%{}[]ก่อนภาษาไทย😀é\x7f "
    return ["".join(rnd.choice(pool) for _ in range(rnd.randint(0, 20))) for _ in range(n)]


@pytest.mark.parametrize("s", ODD_STRINGS + random_strings(200))
def test_bci_encode_round_trips_through_decode_value(s):
    enc = bx._bci_encode(s)
    assert bd.decode_value(enc) == s
    assert enc and enc.isascii() and enc.isprintable()
    assert len(enc.split()) == 1 and not set(enc) & set("{}[]")


def test_bci_encode_refuses_nul_which_bci2000_cannot_store():
    with pytest.raises(bx.ExportError, match="NUL"):
        bx._bci_encode("a\x00b")


@pytest.mark.parametrize("s", [s for s in ODD_STRINGS + random_strings(100, seed=1) if "//" not in s])
def test_bci_encoded_string_survives_a_parameter_line(s):
    # (A value containing "//" is covered by test_bci2000_dat.py::test_parameter_values_may_contain_double_slashes.)
    line = f"{bx.PRM_SECTION} string X= {bx._bci_encode(s)} // a comment"
    assert bd.parse_param_line(line).value == s


def test_bci_encode_agrees_with_bci2000s_python_unescape():
    sys.path.insert(0, str(Path.home() / "BCI2000" / "trunk" / "tools" / "python"))
    try:
        from BCI2000Tools.FileReader import Unescape
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BCI2000Tools not importable: {exc}")
    for s in ODD_STRINGS + random_strings(200, seed=2):
        if s.isascii():          # Unescape maps each %XX byte to one character (no UTF-8 decoding)
            assert Unescape(bx._bci_encode(s)) == s, s


# ---------------------------------------------------------------------------
# synthetic data, demo prefixing, --force, refusals
# ---------------------------------------------------------------------------

def _isolate(tmp_path, monkeypatch):
    fake_root = tmp_path / "fake_repo_root"
    fake_root.mkdir()
    monkeypatch.setattr(ffr, "REPO_ROOT", fake_root)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return tmp_path / "out" / "models", tmp_path / "out" / "demo_session_manifest.json", cwd, fake_root


def test_dat_is_synthetic(tmp_path):
    session = w.build_session(w.checkerboard_selections(1, seed=0))
    yes = bd.read_dat(w.write_speller_dat(tmp_path / "a.dat", session, synthetic_gyro=True))
    no = bd.read_dat(w.write_speller_dat(tmp_path / "b.dat", session))
    assert ffr.dat_is_synthetic(yes) and not ffr.dat_is_synthetic(no)
    session.states["Validation"][:] = 1
    one_real = w.write_speller_dat(tmp_path / "c.dat", session, synthetic_gyro=True)
    data = bd.read_dat(one_real)
    data.states["GyroscopeY"] = data.states["GyroscopeY"].copy()
    data.states["GyroscopeY"][17] = 3                   # one real gyroscope reading: not the fake headset
    assert not ffr.dat_is_synthetic(data)
    # Absence of evidence is not evidence of a real headset (CLAUDE.md: synthetic data must identify itself).
    no_gyro = bd.read_dat(w.write_speller_dat(tmp_path / "d.dat", session, omit_states=("GyroscopeZ",)))
    assert ffr.dat_is_synthetic(no_gyro) and "gyroscope" in ffr.dat_synthetic_reason(no_gyro)
    empty = bd.read_dat(w.write_dat(tmp_path / "e.dat", np.zeros((0, 8)), params=w.signal_params(8),
                                    states=w.pack_layout([("GyroscopeX", 32), ("GyroscopeY", 32),
                                                          ("GyroscopeZ", 32)])[0]))
    assert ffr.dat_is_synthetic(empty)
    simulated = bd.read_dat(w.write_speller_dat(tmp_path / "f.dat", session, source_filter="SignalGeneratorADC"))
    assert "SignalGeneratorADC" in ffr.dat_synthetic_reason(simulated)
    fake_tcp = bd.read_dat(w.write_speller_dat(tmp_path / "g.dat", session, tcp_address="127.0.0.1:9000"))
    assert "TcpAddress" in ffr.dat_synthetic_reason(fake_tcp)
    assert ffr.dat_synthetic_reason(bd.read_dat(w.write_speller_dat(tmp_path / "h.dat", session))) is None


def test_paths_helpers():
    assert ffr.is_bci2000("x/rec.dat") and ffr.is_bci2000("REC.DAT") and not ffr.is_bci2000("rec.csv")
    assert ffr.bci2000_prm_path(Path("m/s1_classifier.joblib")) == Path("m/s1_bci2000.prm")
    assert ffr.bci2000_prm_path(Path("m/demo_s1_classifier.joblib")) == Path("m/demo_s1_bci2000.prm")
    assert ffr.bci2000_prm_path(Path("m/other.joblib")) == Path("m/other_bci2000.prm")


def test_synthetic_gyro_refused_without_allow_demo_and_prefixed_with_it(tmp_path, monkeypatch, capsys):
    session, sig = session_and_signal(seed=12)
    path = w.write_speller_dat(tmp_path / "subject_002.dat", session, sig, synthetic_gyro=True)
    models, man, cwd, fake_root = _isolate(tmp_path, monkeypatch)
    argv = ["--recording", str(path), "--id", "s1", "--models-dir", str(models), "--manifest", str(man),
            "--n-splits", "3"]

    with pytest.raises(ffr.FitError, match="fake headset.*--allow-demo"):
        ffr.fit_from_recording(path, "s1", models_dir=models, manifest_path=man, log=lambda s: None)
    assert ffr.main(argv) == 2
    err = capsys.readouterr().err
    assert "--allow-demo" in err and err.count("\n") == 1
    assert not models.exists() and not man.exists()

    assert ffr.main([*argv, "--allow-demo"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "BCI2000 model" in out
    assert sorted(p.name for p in models.iterdir()) == [
        "demo_s1_bci2000.prm", "demo_s1_classifier.joblib", "demo_s1_fit_report.json", "demo_s1_platt.json"]
    report = json.loads((models / "demo_s1_fit_report.json").read_text(encoding="utf-8"))
    assert report["demo"] is True and report["recording"]["bci2000"]["synthetic"] is True
    prov = json.loads(parse_prm(models / "demo_s1_bci2000.prm")["RiemannProvenance"].value)
    assert prov["demo"] is True and prov["subject_id"] == "s1"
    assert list(cwd.iterdir()) == [] and list(fake_root.iterdir()) == []


def test_demo_prefixed_dat_name_is_refused_without_allow_demo(tmp_path, rec):
    demo = tmp_path / "demo_subject_001.dat"
    demo.write_bytes(rec.path.read_bytes())
    with pytest.raises(ffr.FitError, match="demo_ prefix"):
        ffr.fit_from_recording(demo, "s1", models_dir=tmp_path / "models", manifest_path=tmp_path / "m.json",
                               log=lambda s: None)
    assert not (tmp_path / "models").exists()


def test_force_protection_covers_the_prm(tmp_path, rec):
    models = tmp_path / "models"
    models.mkdir()
    prm = models / "s1_bci2000.prm"
    prm.write_text("old\n", encoding="utf-8")
    kw = dict(models_dir=models, manifest_path=tmp_path / "session_manifest.json", n_splits=2, log=lambda s: None)
    with pytest.raises(ffr.FitError, match="refusing to overwrite.*s1_bci2000.prm.*--force"):
        ffr.fit_from_recording(rec.path, "s1", **kw)
    assert prm.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in models.iterdir()] == ["s1_bci2000.prm"]
    assert not (tmp_path / "session_manifest.json").exists()

    result = ffr.fit_from_recording(rec.path, "s1", force=True, **kw)
    assert result.bci2000_prm_path == prm
    params = parse_prm(prm)
    assert params["RiemannModelID"].value == mf.sha256_file(result.checkpoint_path)


def test_a_recording_cut_off_mid_sample_still_fits_and_says_so(tmp_path, rec):
    # A session that crashed mid-write leaves a partial last sample: drop it, keep the session.
    trunc = tmp_path / "trunc.dat"
    trunc.write_bytes(rec.path.read_bytes()[:-1])
    result = ffr.fit_from_recording(trunc, "s1", models_dir=tmp_path / "models",
                                    manifest_path=tmp_path / "m.json", n_splits=2, log=lambda s: None)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["recording"]["bci2000"]["truncated_tail_bytes"] > 0


def test_unusable_dat_files_are_refused_before_writing_anything(tmp_path, rec):
    kw = dict(models_dir=tmp_path / "models", manifest_path=tmp_path / "m.json", log=lambda s: None)
    with pytest.raises(ffr.FitError, match="not found"):
        ffr.fit_from_recording(tmp_path / "missing.dat", "s1", **kw)
    session, sig = session_and_signal(seed=13)
    wrong = w.write_speller_dat(tmp_path / "wrong.dat", session, sig,
                                channel_names=("Fz", "F3", "C4", "Cz", "Pz", "P3", "O2", "O1"))
    with pytest.raises(ffr.FitError, match="ChannelNames"):
        ffr.fit_from_recording(wrong, "s1", **kw)
    free = w.build_session([{**s, "attended": None} for s in w.checkerboard_selections(3, seed=1)])
    with pytest.raises(ffr.FitError, match="copy mode"):
        ffr.fit_from_recording(w.write_speller_dat(tmp_path / "free.dat", free), "s1", **kw)
    assert not (tmp_path / "models").exists() and not (tmp_path / "m.json").exists()


def test_too_few_selections_is_refused(tmp_path):
    session = w.build_session(w.checkerboard_selections(2, 2, seed=5))
    path = w.write_speller_dat(tmp_path / "short.dat", session, w.planted_signal(session))
    with pytest.raises(ffr.FitError, match="too little usable data: 2 trials"):
        ffr.fit_from_recording(path, "s1", models_dir=tmp_path / "models", manifest_path=tmp_path / "m.json",
                               log=lambda s: None)
    assert not (tmp_path / "models").exists()


def test_epochs_with_linearly_dependent_channels_are_not_trained_on(tmp_path):
    # Two bridged electrodes (identical channels) make every covariance singular; such epochs have no
    # score online either, so the fit must refuse them by name instead of training on garbage.
    session, sig = session_and_signal(seed=21)
    sig[:, 3] = sig[:, 2]
    path = w.write_speller_dat(tmp_path / "bridged.dat", session, sig)
    with pytest.raises(ffr.FitError, match="rank_deficient"):
        ffr.fit_from_recording(path, "s1", models_dir=tmp_path / "models", manifest_path=tmp_path / "m.json",
                               n_splits=2, log=lambda s: None)
