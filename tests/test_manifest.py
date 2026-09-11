import inspect
import json
from pathlib import Path

import pytest

from pipeline import manifest as mf
from pipeline.manifest import ManifestError, ManifestMismatchError


def _write_artifacts(root: Path, n_chars: int = 72):
    models = root / "models"
    models.mkdir(exist_ok=True)
    chars = root / "characters_th.json"
    chars.write_text(json.dumps({"characters": [f"C{i}" for i in range(n_chars)]}), encoding="utf-8")
    clf = models / "clf.joblib"
    clf.write_bytes(b"fake-classifier")
    cal = models / "platt.json"
    cal.write_text(json.dumps({
        "format": mf.CALIBRATION_FORMAT, "a": 1.2, "b": -0.3,
        "score_method": mf.SCORE_METHOD, "classifier_sha256": mf.sha256_file(clf),
    }), encoding="utf-8")
    return chars, clf, cal


def _manifest(root: Path, **kwargs):
    chars, clf, cal = _write_artifacts(root)
    data = mf.build_manifest_dict(base_dir=root, characters_path=chars, classifier_path=clf,
                                  calibration_path=cal, **kwargs)
    return mf.write_manifest(data, root / "session_manifest.json")


def test_roundtrip_resolves_paths_against_manifest_dir(tmp_path, monkeypatch):
    written = _manifest(tmp_path)
    assert written.data["classifier"]["path"] == "models/clf.joblib"
    monkeypatch.chdir(Path("/"))
    loaded = mf.load_manifest(tmp_path / "session_manifest.json")
    assert loaded.fingerprint == written.fingerprint
    assert loaded.classifier_path == (tmp_path / "models" / "clf.joblib").resolve()
    assert loaded.lm_config == {"type": "uniform"}
    assert loaded.corpus_path is None
    assert loaded.gate_policy == "skip"
    assert loaded.stopping == mf.DEFAULT_STOPPING


def test_default_stopping_matches_controller_defaults():
    from stopping.bayesian_stopping import BayesianStoppingController
    params = inspect.signature(BayesianStoppingController.__init__).parameters
    for key, value in mf.DEFAULT_STOPPING.items():
        assert params[key].default == value, key


def test_default_ngram_settings_match_lm_defaults():
    from stopping.lm_prior import ThaiNgramLM
    params = inspect.signature(ThaiNgramLM.__init__).parameters
    for key, value in mf.DEFAULT_NGRAM.items():
        assert params[key].default == value, key


def test_fingerprint_changes_with_any_setting(tmp_path):
    m = _manifest(tmp_path)
    changed = json.loads(json.dumps(m.data))
    changed["stopping"]["theta"] = 0.95
    assert mf.fingerprint_of(changed) != m.fingerprint


def test_fingerprint_ignores_key_order_and_whitespace(tmp_path):
    m = _manifest(tmp_path)
    reordered = dict(reversed(list(m.data.items())))
    (tmp_path / "session_manifest.json").write_text(json.dumps(reordered, indent=4), encoding="utf-8")
    assert mf.load_manifest(tmp_path / "session_manifest.json").fingerprint == m.fingerprint


def test_classifier_swapped_behind_same_path_is_detected(tmp_path):
    _manifest(tmp_path)
    (tmp_path / "models" / "clf.joblib").write_bytes(b"a different classifier")
    with pytest.raises(ManifestError, match="changed since the manifest was written"):
        mf.load_manifest(tmp_path / "session_manifest.json")


def test_calibration_fitted_for_another_classifier_is_detected(tmp_path):
    chars, clf, cal = _write_artifacts(tmp_path)
    cal_data = json.loads(cal.read_text())
    cal_data["classifier_sha256"] = "0" * 64
    cal.write_text(json.dumps(cal_data))
    data = mf.build_manifest_dict(base_dir=tmp_path, characters_path=chars, classifier_path=clf,
                                  calibration_path=cal)
    with pytest.raises(ManifestError, match="different classifier"):
        mf.write_manifest(data, tmp_path / "session_manifest.json")


def test_missing_artifact_is_reported(tmp_path):
    _manifest(tmp_path)
    (tmp_path / "models" / "platt.json").unlink()
    with pytest.raises(ManifestError, match="calibration artifact missing"):
        mf.load_manifest(tmp_path / "session_manifest.json")


def test_verify_hashes_false_skips_file_checks(tmp_path):
    _manifest(tmp_path)
    (tmp_path / "models" / "platt.json").unlink()
    mf.load_manifest(tmp_path / "session_manifest.json", verify_hashes=False)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d.update(schema_version=True), "schema_version"),
    (lambda d: d["stopping"].update(theta=1.0), "theta"),
    (lambda d: d["stopping"].update(t_min=200), "t_min"),
    (lambda d: d["stopping"].update(t_max=10.5), "t_max"),
    (lambda d: d["stopping"].update(llr_clip=0), "llr_clip"),
    (lambda d: d["stopping"].pop("lam"), "lam"),
    (lambda d: d.update(language_model={"type": "transformer"}), "language_model.type"),
    (lambda d: d.update(language_model={"type": "ngram"}), "corpus"),
    (lambda d: d.update(quality_gate={"policy": "downweight"}), "policy"),
    (lambda d: d.update(extra_key=1), "extra_key"),
    (lambda d: d["classifier"].update(sha256="abc"), "sha256"),
    (lambda d: d.pop("calibration"), "calibration"),
])
def test_invalid_manifest_rejected(tmp_path, mutate, match):
    data = json.loads(json.dumps(_manifest(tmp_path).data))
    mutate(data)
    with pytest.raises(ManifestError, match=match):
        mf.validate_manifest_dict(data)


def test_characters_file_must_hold_72_entries(tmp_path):
    chars, clf, cal = _write_artifacts(tmp_path)
    chars.write_text(json.dumps({"characters": [f"C{i}" for i in range(71)]}))
    data = mf.build_manifest_dict(base_dir=tmp_path, characters_path=chars, classifier_path=clf,
                                  calibration_path=cal)
    with pytest.raises(ManifestError, match="exactly 72"):
        mf.write_manifest(data, tmp_path / "session_manifest.json")


def test_ngram_corpus_is_hashed_and_resolved(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("กขค" * 10, encoding="utf-8")
    m = _manifest(tmp_path, language_model={"type": "ngram", "corpus_path": corpus})
    assert m.corpus_path == corpus.resolve()
    assert m.lm_config["order"] == 4
    corpus.write_text("changed", encoding="utf-8")
    with pytest.raises(ManifestError, match="language_model.corpus artifact changed"):
        mf.load_manifest(m.path)


def test_missing_manifest_points_to_fit_script(tmp_path):
    with pytest.raises(ManifestError, match="fit_from_recording"):
        mf.load_manifest(tmp_path / "nope.json")


def test_malformed_manifest_json(tmp_path):
    path = tmp_path / "session_manifest.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        mf.load_manifest(path)


def test_calibration_file_requires_decision_function_scores(tmp_path):
    path = tmp_path / "platt.json"
    path.write_text(json.dumps({"format": mf.CALIBRATION_FORMAT, "a": 1.0, "b": 0.0,
                                "score_method": "predict_proba", "classifier_sha256": "0" * 64}))
    with pytest.raises(ManifestError, match="score_method"):
        mf.load_calibration_file(path)


def test_assert_same_fingerprint():
    mf.assert_same_fingerprint("a" * 64, "a" * 64, context="gui")
    with pytest.raises(ManifestMismatchError, match="aaaaaaaaaaaa.*bbbbbbbbbbbb"):
        mf.assert_same_fingerprint("a" * 64, "b" * 64, context="gui")
    with pytest.raises(ManifestMismatchError, match="no manifest fingerprint"):
        mf.assert_same_fingerprint("a" * 64, None, context="gui")


def test_write_leaves_no_temp_files(tmp_path):
    _manifest(tmp_path)
    assert not list(tmp_path.glob(".session_manifest.*"))
