"""
Phase 1 offline validation (design doc Sec. 4/5): exercise
BayesianStoppingController + run_sweep() end-to-end using REAL classifier
scores from BNCI2014-008, with SYNTHETIC row/column groups layered on top.

IMPORTANT LIMITATION -- read before trusting any number this prints:
BNCI2014-008 does not expose real row/column flash-group identity
(docs/CLAUDE.md gotcha #3): only target/non-target labels are available.
So the "characters" and "groups" here are fabricated (a made-up 8x9 grid),
NOT real Thai speller trials. This script validates that the stopping
code (log_normalize / llr / update / run_sweep / wolpaw_itr) behaves
sensibly on real classifier score distributions -- it is a code-validation
exercise, not a research result, and must not be reported as one.

It is also a NO-LM baseline: lm_prior is uniform here (a real
ThaiNgramLM can be swapped in via stopping/lm_prior.py). Under uniform
lm_prior, p0 = lam * uniform + (1 - lam) * uniform = uniform regardless
of lam, so this sweep only varies theta and t_min meaningfully -- lam is
included in the param grid for plumbing purposes but has no effect until
a real (non-uniform) lm_prior is passed in.

NOTE: BayesianStoppingController.reset()/llr()/update() are left as
NotImplementedError stubs in bayesian_stopping.py (fixed threshold,
boundary-aware thresholding descoped) -- this script will raise
NotImplementedError until those are implemented by hand. That is
expected, not a bug in this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import numpy as np
from sklearn.model_selection import GroupKFold

from data_loader import load_arrays
from classifiers.xdawn_cov_tangent import build_pipeline
from stopping.bayesian_stopping import BayesianStoppingController
from stopping.calibrate import PlattCalibrator, expected_calibration_error, plot_reliability_diagram
from stopping.replay import run_sweep


def extract_out_of_fold_scores(dataset: str = "BNCI2014_008",
                                subjects: list[int] = (1, 2, 3, 4, 5),
                                classifier_name: str = "FgMDM",
                                nfilter: int = 4,
                                n_splits: int = 5,
                                seed: int = 0):
    """
    Fit classifier_name (from classifiers/xdawn_cov_tangent.py, same
    build_pipeline used in benchmark_all.py) via GroupKFold-by-subject CV,
    and return REAL out-of-fold P(Target) scores for every epoch -- i.e.
    each subject's epochs are scored by a model that never saw that
    subject during training (same leakage-avoidance as benchmark_all.py).

    Returns:
        scores: np.ndarray, shape (N,), P(Target) from predict_proba.
        labels: np.ndarray of str, shape (N,), "Target" / "NonTarget".
    """
    X, y, metadata = load_arrays(dataset, subjects=list(subjects))
    groups = metadata["subject"].values

    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    scores = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        pipe = build_pipeline(classifier_name, nfilter=nfilter)
        pipe.fit(X[train_idx], y[train_idx])
        classes = list(pipe.named_steps["clf"].classes_)
        scores[test_idx] = pipe.predict_proba(X[test_idx])[:, classes.index("Target")]

    return scores, y


def build_synthetic_groups(n_rows: int = 8, n_cols: int = 9):
    """
    Fabricate an n_rows x n_cols matrix's row/column flash groups (no real
    Thai speller matrix layout backs this -- purely for exercising
    run_sweep()/simulate_trial() end-to-end).

    Returns:
        groups: list of lists, groups[g] = char indices in group g
            (n_rows row-groups, then n_cols column-groups).
        char_groups: dict, char_idx -> (row_group_idx, col_group_idx).
        n_chars: n_rows * n_cols.
    """
    n_chars = n_rows * n_cols

    def char_idx(r, c):
        return r * n_cols + c

    row_groups = [[char_idx(r, c) for c in range(n_cols)] for r in range(n_rows)]
    col_groups = [[char_idx(r, c) for r in range(n_rows)] for c in range(n_cols)]
    groups = row_groups + col_groups

    char_groups = {}
    for r in range(n_rows):
        for c in range(n_cols):
            char_groups[char_idx(r, c)] = (r, n_rows + c)

    return groups, char_groups, n_chars


def build_synthetic_trials(n_trials: int, target_scores: np.ndarray,
                            nontarget_scores: np.ndarray, groups: list,
                            char_groups: dict, n_chars: int,
                            n_repetitions: int = 10, seed: int = 0):
    """
    Build synthetic replay trials: for each trial, pick a random true
    character, then simulate n_repetitions full cycles of all groups
    flashing once (in random order), drawing each flash's score from the
    REAL empirical target/non-target score distributions depending on
    whether that group contains the true character.

    Returns:
        trials: list of dicts as expected by stopping.replay.run_sweep().
    """
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    trials = []

    for _ in range(n_trials):
        true_char = int(rng.integers(0, n_chars))
        true_row_group, true_col_group = char_groups[true_char]

        scores, group_ids = [], []
        for _ in range(n_repetitions):
            order = rng.permutation(n_groups)
            for g in order:
                is_target_group = g in (true_row_group, true_col_group)
                pool = target_scores if is_target_group else nontarget_scores
                scores.append(float(rng.choice(pool)))
                group_ids.append(int(g))

        trials.append({
            "scores": scores,
            "group_ids": group_ids,
            "true_char_idx": true_char,
            "lm_prior": np.full(n_chars, 1.0 / n_chars),
        })

    return trials


def main():
    print("Extracting real out-of-fold classifier scores from BNCI2014-008 "
          "(xDAWNcov+FgMDM, subjects 1-5)...")
    scores, labels = extract_out_of_fold_scores()
    target_scores = scores[labels == "Target"]
    nontarget_scores = scores[labels == "NonTarget"]
    print(f"  {len(target_scores)} target flashes, {len(nontarget_scores)} "
          f"non-target flashes")
    print(f"  target mean/std: {target_scores.mean():.4f}/{target_scores.std():.4f}  "
          f"non-target mean/std: {nontarget_scores.mean():.4f}/{nontarget_scores.std():.4f}")

    print("Fitting Platt scaling (a, b) on these real scores -- the "
          "uncalibrated defaults (a=1, b=0) make the LLR meaningless "
          "given how close target/non-target scores are here.")
    binary_labels = (labels == "Target").astype(int)
    calibrator = PlattCalibrator(calibration_path="unused")
    a, b = calibrator.fit(scores, binary_labels)
    probs = calibrator.predict_proba(scores)
    ece = expected_calibration_error(probs, binary_labels, n_bins=10)
    out_dir = "results/route_a"
    os.makedirs(out_dir, exist_ok=True)
    plot_reliability_diagram(probs, binary_labels, n_bins=10,
                              out_path=os.path.join(out_dir, "phase1_reliability_diagram.png"))
    print(f"  Fitted: a={a:.4f}, b={b:.4f}, ECE={ece:.4f}")

    groups, char_groups, n_chars = build_synthetic_groups(n_rows=8, n_cols=9)
    n_flashes_per_repetition = len(groups)  # 8 + 9 = 17

    print("Building 200 synthetic replay trials on a fabricated 8x9 grid "
          "(NOT a real Thai speller layout -- see module docstring)...")
    trials = build_synthetic_trials(
        n_trials=200,
        target_scores=target_scores,
        nontarget_scores=nontarget_scores,
        groups=groups,
        char_groups=char_groups,
        n_chars=n_chars,
        n_repetitions=10,
    )

    # pi = target-in-group prior = group size / n_chars (constant here
    # since all row groups are size n_cols, all col groups size n_rows;
    # BayesianStoppingController.llr() needs the actual pi per flash, but
    # run_sweep()/simulate_trial() compute pi internally from group size).
    param_grid = {
        "lam": [0.7],          # inert: lm_prior is uniform (no LM yet)
        "theta": [0.7, 0.8, 0.9, 0.95],
        "t_min": [1, 3, 6],
        "t_max": [10 * n_flashes_per_repetition],
        "sigmoid_a": [a],
        "sigmoid_b": [b],
    }

    print("Running run_sweep() over theta x t_min, using the fitted "
          "Platt (a, b) above (lam held fixed -- see docstring for why)...")
    df = run_sweep(trials, groups, param_grid,
                    seconds_per_flash=250e-3,  # config.BNCI2014_008_SOA_MS
                    controller_cls=BayesianStoppingController)

    out_dir = "results/route_a"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1_no_lm_baseline_sweep.csv")
    df.to_csv(out_path, index=False)

    print(df.sort_values("itr_bits_per_min", ascending=False).to_string(
        index=False, float_format="{:.3f}".format))
    print(f"\nSaved to {out_path}")
    print("\nReminder: fabricated grid + groups. This validates the code "
          "path (accuracy/flashes/ITR trade-off responds sensibly to "
          "theta_base and t_min), it is not a Thai-speller research result.")


if __name__ == "__main__":
    main()
