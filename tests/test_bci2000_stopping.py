"""
The BCI2000 speller's Bayesian stopping and language-model prior against their Python originals.

  BayesianStopping.h  vs stopping/bayesian_stopping.BayesianStoppingController (owner-implemented):
                      the same priors and flash sequences must give the same posterior after every
                      flash, the same stop flash and the same choice.
  ThaiGPT.h           vs stopping/transformer_lm.ThaiTransformerLM.get_prior (PyTorch), through the
                      file pipeline/bci2000_lm_export.py writes: the same prior for every context.

Compiles a driver with BCI2000's bundled Eigen; skips without a compiler or BCI2000 source tree.
The language-model tests also need the trained checkpoint (models/thai_gpt_checkpoint.pt, gitignored)
and torch; they only load it and run forward passes (CLAUDE.md: never train here).
"""

import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pipeline.manifest import REPO_ROOT, load_characters_file
from stopping.bayesian_stopping import BayesianStoppingController

BCI2000_ROOT = Path(os.environ.get("BCI2000_ROOT", Path.home() / "BCI2000" / "trunk"))
EIGEN = BCI2000_ROOT / "src" / "extlib" / "eigen" / "eigen-3.3.7"
SRC = REPO_ROOT / "bci2000" / "CheckerboardSpeller"
CHECKPOINT = REPO_ROOT / "models" / "thai_gpt_checkpoint.pt"
CHARACTERS = load_characters_file(REPO_ROOT / "collection" / "characters_th.json")


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None or not (EIGEN / "Eigen").is_dir():
        pytest.skip("needs a C++ compiler and BCI2000's bundled Eigen")
    exe = tmp_path_factory.mktemp("scli") / "stopping_cli"
    subprocess.run([cxx, "-std=c++17", "-O2", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", "-isystem", str(EIGEN),
                    str(SRC / "tests" / "stopping_cli.cpp"), "-o", str(exe)], check=True, capture_output=True)
    return exe


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------

def run_controller(exe, params, prior, flashes):
    lines = [" ".join(repr(float(v)) if isinstance(v, float) else str(v) for v in params),
             " ".join(repr(float(p)) for p in prior), str(len(flashes))]
    for group, s, pi in flashes:
        lines.append(f"{len(group)} {' '.join(map(str, group))} {s!r} {pi!r}")
    out = subprocess.run([str(exe), "stop"], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         check=True).stdout.splitlines()
    steps = [line.split() for line in out[:-1]]
    return [(int(t), float(p), int(a), bool(int(st)), int(i)) for t, p, a, st, i in steps], \
        np.array(out[-1].split(), dtype=float)


def python_controller(params, prior, flashes):
    n, lam, theta, t_min, t_max, a, b, clip = params
    c = BayesianStoppingController(n_chars=n, lam=lam, theta=theta, t_min=t_min, t_max=t_max,
                                   sigmoid_a=a, sigmoid_b=b, llr_clip=clip)
    c.reset(np.asarray(prior, dtype=float))
    steps, logs = [], []
    for group, s, pi in flashes:
        stop, idx = c.update(list(group), s, pi)
        steps.append((c.t, float(np.exp(c.log_p.max())), int(np.argmax(c.log_p)), stop, -1 if idx is None else idx))
        logs.append(c.log_p.copy())
    return steps, c.log_p, logs


def same_choice(a_cpp, a_py, log_p):
    """The same character, or two characters tied in exact arithmetic: characters that have shared every
    flash under a uniform prior. numpy's vectorised log rounds array positions differently, so Python's
    copies of a tie can differ by an ulp where the C++ keeps them equal; either choice is then correct."""
    return a_cpp == a_py or abs(log_p[a_cpp] - log_p[a_py]) <= 1e-12 * abs(log_p[a_py])


def random_case(seed):
    rng = np.random.default_rng(seed)
    n = 72
    prior = rng.dirichlet(np.full(n, rng.choice([0.05, 0.5, 5.0])))
    params = (n, float(rng.choice([0.0, 0.3, 0.7, 1.0])), float(rng.uniform(0.3, 0.99)), int(rng.integers(0, 12)),
              int(rng.integers(20, 200)), float(rng.uniform(0.01, 2.0)), float(rng.uniform(-3, 1)),
              float(rng.choice([0.5, 2.0, 8.0])))
    target = int(rng.integers(n))
    flashes = []
    for _ in range(int(rng.integers(1, 150))):
        group = sorted(rng.choice(n, size=6, replace=False).tolist())
        s = float(rng.normal(3.0 if target in group else -1.0, 4.0))
        flashes.append((group, s, 6 / n))
    return params, prior, flashes


@pytest.mark.parametrize("seed", range(40))
def test_controller_equals_the_python_controller(cli, seed):
    params, prior, flashes = random_case(seed)
    cpp, cpp_logp = run_controller(cli, params, prior, flashes)
    py, py_logp, logs = python_controller(params, prior, flashes)
    assert len(cpp) == len(py)
    for (t1, p1, a1, s1, i1), (t2, p2, a2, s2, i2), lp in zip(cpp, py, logs):
        assert (t1, s1) == (t2, s2)
        assert same_choice(a1, a2, lp) and (i1 == i2 == -1 or same_choice(i1, i2, lp))
        assert p1 == pytest.approx(p2, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(cpp_logp, py_logp, rtol=1e-12, atol=1e-12)


def test_controller_edges_threshold_tmax_and_clip(cli):
    n = 72
    uniform = np.full(n, 1 / n)
    big = [([0, 1, 2, 3, 4, 5], 1e6, 6 / n)] * 3                       # clipped LLRs
    for params in [(n, 0.7, 0.5, 2, 170, 1.0, 0.0, 8.0),                 # threshold after t_min
                   (n, 0.7, 0.999999, 0, 2, 1.0, 0.0, 8.0),              # forced at t_max
                   (n, 0.0, 0.9, 0, 170, 1.0, 0.0, 0.5)]:                # tiny clip, lam 0
        cpp, cpp_logp = run_controller(cli, params, uniform, big)
        py, py_logp, _ = python_controller(params, uniform, big)
        assert [(t, a, s, i) for t, _, a, s, i in cpp] == [(t, a, s, i) for t, _, a, s, i in py]
        np.testing.assert_allclose(cpp_logp, py_logp, rtol=1e-12, atol=1e-12)
    # ties: argmax is the first index, as np.argmax
    cpp, _ = run_controller(cli, (n, 0.7, 0.99, 0, 170, 1.0, 0.0, 8.0), uniform, [([0, 1], 0.0, 6 / n)])
    assert cpp[0][2] == 0


# ---------------------------------------------------------------------------
# the language model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lm(tmp_path_factory):
    if not CHECKPOINT.is_file():
        pytest.skip("no trained checkpoint at models/thai_gpt_checkpoint.pt")
    pytest.importorskip("torch")
    from pipeline.bci2000_lm_export import export_lm
    from stopping.transformer_lm import ThaiTransformerLM
    out = export_lm(CHECKPOINT, tmp_path_factory.mktemp("lm") / "thai.lmbin")
    return out, ThaiTransformerLM(str(CHECKPOINT), CHARACTERS)


def contexts():
    rnd = random.Random(0)
    thai = "".join(CHARACTERS)
    out = ["", "ก", "กิน", "สวัสดีครับ", "ภาษาไทย", "คอมพิวเตอร์", "ประเทศไทย", "abc", "😀", "กิน ข้าว",
           "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์และสระหลายตัวซึ่งทำให้การสะกดคำมีความซับซ้อนมากกว่าภาษาอังกฤษมาก"]
    out += ["".join(rnd.choice(thai) for _ in range(rnd.randint(1, 90))) for _ in range(25)]
    return out


def run_lm(exe, path, ctxs):
    out = subprocess.run([str(exe), "lm", str(path)], input="\n".join(ctxs) + "\n", capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out[-1].startswith("us ")
    return [np.array(line.split(), dtype=float) for line in out[:-1]], float(out[-1].split()[1])


def test_prior_equals_pytorch_for_every_context(cli, lm):
    path, model = lm
    ctxs = contexts()
    cpp, us = run_lm(cli, path, ctxs)
    assert len(cpp) == len(ctxs)
    for c, p in zip(ctxs, cpp):
        ref = model.get_prior(c)
        assert p.shape == ref.shape and abs(p.sum() - 1) < 1e-9
        np.testing.assert_allclose(p, ref, rtol=2e-5, atol=2e-6, err_msg=repr(c))
        assert int(np.argmax(p)) == int(np.argmax(ref)), repr(c)
    assert us < 50_000, f"{us:.0f} us per prior: too slow for the speller's selection start"


def test_export_refuses_a_matrix_the_checkpoint_was_not_trained_for(tmp_path, lm):
    from pipeline.bci2000_lm_export import LMExportError, export_lm
    data = json.loads((REPO_ROOT / "collection" / "characters_th.json").read_text(encoding="utf-8"))
    data["characters"][0] = "☃"                                   # not in the Thai vocabulary
    bad = tmp_path / "chars.json"
    bad.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(LMExportError, match="missing matrix character"):
        export_lm(CHECKPOINT, tmp_path / "x.lmbin", bad)


def test_a_damaged_file_is_refused_not_used(cli, lm, tmp_path):
    path, _ = lm
    data = path.read_bytes()
    for name, blob in [("short", data[:-4]), ("long", data + b"\x00"), ("header", b"THAIGPT 2\n" + data[10:])]:
        f = tmp_path / f"{name}.lmbin"
        f.write_bytes(blob)
        out = subprocess.run([str(cli), "lm", str(f)], input="ก\n", capture_output=True, text=True)
        assert out.stdout.startswith("error"), (name, out.stdout[:200])
