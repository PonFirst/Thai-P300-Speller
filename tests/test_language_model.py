import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import language_model as lmod
from pipeline import manifest as mf
from pipeline.language_model import LanguageModelError
from stopping.lm_prior import ThaiNgramLM

REPO_CHARACTERS = mf.REPO_ROOT / "collection" / "characters_th.json"


def _manifest(root: Path, language_model=None):
    models = root / "models"
    models.mkdir(exist_ok=True)
    clf = models / "clf.joblib"
    clf.write_bytes(b"fake-classifier")
    cal = models / "platt.json"
    cal.write_text(json.dumps({"format": mf.CALIBRATION_FORMAT, "a": 1.0, "b": 0.0,
                               "score_method": mf.SCORE_METHOD,
                               "classifier_sha256": mf.sha256_file(clf)}), encoding="utf-8")
    data = mf.build_manifest_dict(base_dir=root, characters_path=REPO_CHARACTERS, classifier_path=clf,
                                  calibration_path=cal, language_model=language_model)
    return mf.write_manifest(data, root / "session_manifest.json")


@pytest.fixture(scope="module")
def characters():
    return mf.load_characters_file(REPO_CHARACTERS)


def test_uniform_prior_is_exactly_one_over_n(tmp_path, characters):
    manifest = _manifest(tmp_path)
    lm = lmod.build_language_model(manifest, characters)
    for context in ("", characters[0], "".join(characters[:5])):
        prior = lm.get_prior(context)
        assert prior.shape == (72,)
        assert np.all(prior == 1.0 / 72)


def test_ngram_built_from_manifest_corpus(tmp_path, characters):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(characters[i] for i in (0, 1, 2, 0, 1, 3)) * 50, encoding="utf-8")
    manifest = _manifest(tmp_path, {"type": "ngram", "corpus_path": corpus, "order": 3, "alpha": 0.5})
    lm = lmod.build_language_model(manifest, characters)
    reference = ThaiNgramLM.from_corpus_path(str(corpus), characters, order=3, alpha=0.5, backoff_weight=0.4)
    for context in ("", characters[0], characters[0] + characters[1]):
        np.testing.assert_array_equal(lm.get_prior(context), reference.get_prior(context))
    after_01 = lm.get_prior(characters[0] + characters[1])
    assert after_01.argmax() in (2, 3)
    assert abs(after_01.sum() - 1) < 1e-9


def test_prior_is_a_fresh_copy(tmp_path, characters):
    lm = lmod.build_language_model(_manifest(tmp_path), characters)
    prior = lm.get_prior("")
    prior[:] = 0
    assert np.all(lm.get_prior("") == 1.0 / 72)


@pytest.mark.parametrize("prior,match", [
    (np.full(71, 1 / 71), "shape"),
    (np.full((72, 1), 1 / 72), "shape"),
    (np.r_[np.nan, np.full(71, 1 / 71)], "non-finite"),
    (np.r_[np.inf, np.zeros(71)], "non-finite"),
    (np.r_[-0.1, np.full(71, 1.1 / 71)], "negative"),
    (np.full(72, 1 / 70), "sum to 1"),
    (np.r_[1.0 + 2e-6, np.zeros(71)], "sum to 1"),
    (["a"] * 72, "not numeric"),
])
def test_validate_prior_rejects(prior, match):
    with pytest.raises(LanguageModelError, match=match):
        lmod.validate_prior(prior, 72)


def test_validate_prior_accepts_within_tolerance():
    lmod.validate_prior(np.r_[1.0 + 5e-7, np.zeros(71)], 72)


def test_bad_model_output_caught_at_the_boundary(characters, monkeypatch):
    class Broken:
        calls = 0

        def get_prior(self, context):
            Broken.calls += 1
            return np.full(72, 1 / 72) if Broken.calls == 1 else np.full(72, 0.5)

    monkeypatch.setitem(lmod.LM_BUILDERS, "uniform", lambda cfg, m, chars: Broken())
    lm = lmod.build_language_model(SimpleNamespace(lm_config={"type": "uniform"}, corpus_path=None), characters)
    with pytest.raises(LanguageModelError, match="uniform language model: prior must sum to 1"):
        lm.get_prior("x")


def test_invalid_model_fails_at_construction(characters, monkeypatch):
    monkeypatch.setitem(lmod.LM_BUILDERS, "uniform", lambda cfg, m, chars: SimpleNamespace(
        get_prior=lambda context: np.ones(5)))
    with pytest.raises(LanguageModelError, match="shape"):
        lmod.build_language_model(SimpleNamespace(lm_config={"type": "uniform"}, corpus_path=None), characters)


def test_unknown_type_and_bad_context(tmp_path, characters):
    with pytest.raises(LanguageModelError, match="transformer"):
        lmod.build_language_model(SimpleNamespace(lm_config={"type": "transformer"}, corpus_path=None), characters)
    lm = lmod.build_language_model(_manifest(tmp_path), characters)
    with pytest.raises(LanguageModelError, match="context"):
        lm.get_prior(None)
