"""
The C++ checkerboard (bci2000/CheckerboardSpeller/CheckerboardSequence.h, used by the BCI2000
speller) against the Python reference (collection/flash_sequence.py).

The two draw random numbers differently, so the comparison is exact given the same shuffles:
the C++ output's row groups spell out each repetition's white and black orders, those are fed to
the Python builder in place of random.sample, and the two group lists must be identical.
Adjacency checks are compared on the same groups. Compiles a driver into a temp dir; skips
without a C++ compiler.
"""

import json
import shutil
import subprocess
import sys
from collections import Counter
from unittest import mock

import pytest

from pipeline.manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "collection"))
import flash_sequence as fs  # noqa: E402

SRC = REPO_ROOT / "bci2000" / "CheckerboardSpeller"


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("cbcli") / "checkerboard_cli"
    subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O1", str(SRC / "tests" / "checkerboard_cli.cpp"),
                    "-o", str(exe)], check=True, capture_output=True)
    return exe


def run_cpp(exe, n_chars, reps, seed, grid=None):
    args = [str(exe), str(n_chars), str(reps), str(seed)] + ([str(g) for g in grid] if grid else [])
    out = subprocess.run(args, capture_output=True, text=True)
    return json.loads(out.stdout)


def python_groups_with_same_shuffles(cpp_groups, n_chars, reps):
    half = n_chars // 2
    v_rows, v_cols = fs._virtual_shape(half)
    per_rep = 2 * v_rows + 2 * v_cols
    draws = []
    for r in range(reps):
        rep = cpp_groups[r * per_rep:(r + 1) * per_rep]
        draws.append([c for g in rep[:v_rows] for c in g])                  # white order
        draws.append([c for g in rep[v_rows:2 * v_rows] for c in g])        # black order
    it = iter(draws)
    with mock.patch.object(fs.random, "sample", side_effect=lambda population, k: next(it)):
        return fs.build_flash_sequence(fs.CheckerboardConfig(n_chars=n_chars, n_repetitions=reps))


@pytest.mark.parametrize("seed", range(8))
def test_same_shuffles_give_the_same_groups_as_python(cli, seed):
    cpp = run_cpp(cli, 72, 5, seed)["groups"]
    assert python_groups_with_same_shuffles(cpp, 72, 5) == cpp


@pytest.mark.parametrize("n_chars,reps", [(72, 1), (72, 7), (12, 3), (8, 2), (2, 1)])
def test_other_sizes_match_python_too(cli, n_chars, reps):
    cpp = run_cpp(cli, n_chars, reps, 3)["groups"]
    assert python_groups_with_same_shuffles(cpp, n_chars, reps) == cpp


@pytest.mark.parametrize("seed", range(5))
def test_every_character_twice_per_repetition_in_same_colour_groups_of_six(cli, seed):
    groups = run_cpp(cli, 72, 5, seed)["groups"]
    assert len(groups) == 5 * 24 and all(len(g) == 6 for g in groups)
    for r in range(5):
        counts = Counter(c for g in groups[r * 24:(r + 1) * 24] for c in g)
        assert sorted(counts) == list(range(72)) and set(counts.values()) == {2}
    assert all(len({c % 2 for c in g}) == 1 for g in groups)


def test_different_seeds_differ_and_one_seed_repeats(cli):
    assert run_cpp(cli, 72, 5, 1)["groups"] == run_cpp(cli, 72, 5, 1)["groups"]
    assert run_cpp(cli, 72, 5, 1)["groups"] != run_cpp(cli, 72, 5, 2)["groups"]


@pytest.mark.parametrize("seed", range(5))
def test_8x9_is_adjacency_safe_and_9x8_is_not_same_answer_as_python(cli, seed):
    safe = run_cpp(cli, 72, 5, seed, (8, 9))
    assert safe["violations"] == []
    assert fs.validate_no_adjacent_co_flash(safe["groups"], 8, 9) == (True, [])
    unsafe = run_cpp(cli, 72, 5, seed, (9, 8))
    ok, py_violations = fs.validate_no_adjacent_co_flash(unsafe["groups"], 9, 8)
    assert not ok and sorted(map(tuple, unsafe["violations"])) == sorted(py_violations)


def test_odd_character_count_is_refused(cli):
    assert "error" in run_cpp(cli, 71, 1, 0)


def test_the_parameter_file_is_generated_from_characters_th_json():
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import make_bci2000_parms as mp
    assert mp.main(["--check"]) == 0, "run tools/make_bci2000_parms.py"
    chars = mp.load_characters()
    assert [mp.decode(mp.encode(c)) for c in chars] == chars
    assert mp.encode("") == "%" and mp.encode("A b%") == "A%20b%25"
