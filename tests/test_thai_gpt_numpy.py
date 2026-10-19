"""
stopping/thai_gpt_numpy.py: the Transformer prior without PyTorch, from the .lmbin export.
  - equals the PyTorch model (a small random one always; the trained one when models/ has it);
  - refuses a file for another matrix or a damaged file;
  - the LM factory picks it for an .lmbin, and the fit copies an .lmbin instead of exporting.
The parity tests need torch and skip without it (the Windows package has none); the rest don't.
"""

import json

import numpy as np
import pytest

import bci2000_dat_writer as w
from pipeline import fit_from_recording as ffr
from pipeline.language_model import build_language_model
from pipeline.manifest import REPO_ROOT, load_characters_file, load_manifest
from stopping.thai_gpt_numpy import LmbinError, ThaiGPTNumpy, read_lmbin

CHARS = load_characters_file(REPO_ROOT / "collection" / "characters_th.json")
UNK = ""
SMALL = {"vocab_size": len(CHARS) + 4, "block_size": 16, "n_embd": 32, "n_head": 4, "n_layer": 2}
CONTEXTS = ["", "ก", "กิน", "วันนี้ผมไปกินข้าวที่ตลาด", "abc", "ภาษาไทย" * 6]


def write_lmbin(path, *, matrix=CHARS, vocab=None, seed=0, config=SMALL):
    """A valid .lmbin written directly (no torch): random weights, the exporter's header and layout."""
    vocab = vocab or (sorted(set(CHARS)) + ["ฤ", "ๅ", "!"] + [UNK])
    v, b, c, h, n = (config[k] for k in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"))
    assert len(vocab) == v
    rng = np.random.default_rng(seed)
    shapes = [(v, c), (b, c)]
    for _ in range(n):
        shapes += [(c,), (c,)] + [(c // h, c)] * (3 * h) + [(c, c), (c,), (c,), (c,), (4 * c, c), (4 * c,), (c, 4 * c), (c,)]
    shapes += [(c,), (c,), (v, c), (v,)]
    header = ["THAIGPT 1", f"{v} {b} {c} {h} {n} {v - 1}", " ".join(f"{ord(x):x}" for x in vocab),
              "checkpoint " + "0" * 64, "matrix " + " ".join(f"{ord(x):x}" for x in matrix), "DATA"]
    data = b"".join(rng.normal(0, 0.3, s).astype("<f4").tobytes() for s in shapes)
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + data)
    return path


# ---------------------------------------------------------------------------
# equal to PyTorch
# ---------------------------------------------------------------------------

def test_a_random_model_equals_pytorch_through_the_real_exporter(tmp_path):
    torch = pytest.importorskip("torch")
    from pipeline.bci2000_lm_export import export_lm
    from stopping.transformer_lm import ThaiGPT, ThaiTransformerLM
    vocab = sorted(set(CHARS)) + ["ฤ", "ๅ", "!"] + [UNK]
    torch.manual_seed(3)
    model = ThaiGPT(SMALL["vocab_size"], SMALL["block_size"], SMALL["n_embd"], SMALL["n_head"], SMALL["n_layer"], 0.0)
    ckpt = tmp_path / "small.pt"
    torch.save({"config": {**SMALL, "dropout": 0.0}, "vocab": vocab, "model_state_dict": model.state_dict()}, ckpt)
    lmbin = export_lm(ckpt, tmp_path / "small.lmbin")
    ref, mine = ThaiTransformerLM(str(ckpt), CHARS), ThaiGPTNumpy(lmbin, CHARS)
    for c in CONTEXTS:
        np.testing.assert_allclose(mine.get_prior(c), ref.get_prior(c), rtol=2e-5, atol=2e-6, err_msg=repr(c))


def test_the_trained_model_equals_pytorch(tmp_path):
    ckpt = REPO_ROOT / "models" / "thai_gpt_checkpoint.pt"
    if not ckpt.is_file():
        pytest.skip("no trained checkpoint in models/")
    pytest.importorskip("torch")
    from pipeline.bci2000_lm_export import export_lm
    from stopping.transformer_lm import ThaiTransformerLM
    ref, mine = ThaiTransformerLM(str(ckpt), CHARS), ThaiGPTNumpy(export_lm(ckpt, tmp_path / "t.lmbin"), CHARS)
    for c in CONTEXTS:
        a, b = mine.get_prior(c), ref.get_prior(c)
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6, err_msg=repr(c))
        assert int(np.argmax(a)) == int(np.argmax(b))


# ---------------------------------------------------------------------------
# refusals and shape (no torch)
# ---------------------------------------------------------------------------

def test_priors_are_distributions_over_the_matrix(tmp_path):
    lm = ThaiGPTNumpy(write_lmbin(tmp_path / "m.lmbin"), CHARS)
    for c in CONTEXTS:
        p = lm.get_prior(c)
        assert p.shape == (len(CHARS),) and np.all(p > 0) and abs(p.sum() - 1) < 1e-12


def test_a_file_for_another_matrix_or_a_damaged_file_is_refused(tmp_path):
    good = write_lmbin(tmp_path / "m.lmbin")
    with pytest.raises(LmbinError, match="different matrix"):
        ThaiGPTNumpy(write_lmbin(tmp_path / "other.lmbin", matrix=list(reversed(CHARS))), CHARS)
    vocab = sorted(set(CHARS))[1:] + ["ฤ", "ๅ", "!", "?"] + [UNK]
    with pytest.raises(LmbinError, match="missing matrix"):
        ThaiGPTNumpy(write_lmbin(tmp_path / "missing.lmbin", vocab=vocab), CHARS)
    data = good.read_bytes()
    for name, blob in [("short", data[:-4]), ("long", data + b"\0\0\0\0"), ("format", b"THAIGPT 2" + data[9:])]:
        (tmp_path / name).write_bytes(blob)
        with pytest.raises(LmbinError):
            read_lmbin(tmp_path / name)


# ---------------------------------------------------------------------------
# the fit and the LM factory use it
# ---------------------------------------------------------------------------

def test_the_fit_copies_an_lmbin_and_the_manifest_loads_it_without_torch(tmp_path):
    lmbin = write_lmbin(tmp_path / "thai.lmbin")
    session = w.build_session(w.checkerboard_selections(4, 2, seed=8))
    dat = w.write_speller_dat(tmp_path / "S1S001R01.dat", session, w.planted_signal(session, seed=8))
    result = ffr.fit_from_recording(dat, "s1", models_dir=tmp_path / "models", manifest_path=tmp_path / "m.json",
                                    lm_transformer_checkpoint=lmbin, n_splits=3, log=lambda s: None)
    copied = tmp_path / "models" / "s1_lm.lmbin"
    assert copied.read_bytes() == lmbin.read_bytes()
    assert "LanguageModel= 1" in result.bci2000_prm_path.read_text(encoding="utf-8")
    lm = build_language_model(load_manifest(result.manifest_path), CHARS)
    np.testing.assert_allclose(lm.get_prior("กิน"), ThaiGPTNumpy(lmbin, CHARS).get_prior("กิน"))


def test_the_fit_refuses_an_lmbin_for_another_matrix(tmp_path):
    lmbin = write_lmbin(tmp_path / "other.lmbin", matrix=list(reversed(CHARS)))
    session = w.build_session(w.checkerboard_selections(4, 2, seed=8))
    dat = w.write_speller_dat(tmp_path / "S1S001R01.dat", session, w.planted_signal(session, seed=8))
    with pytest.raises(ffr.FitError, match="different matrix"):
        ffr.fit_from_recording(dat, "s1", models_dir=tmp_path / "models", manifest_path=tmp_path / "m.json",
                               lm_transformer_checkpoint=lmbin, n_splits=3, log=lambda s: None)
    assert not (tmp_path / "m.json").exists()
