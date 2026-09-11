"""
Offline replay harness for the Bayesian stopping controller.

Replays recorded classifier scores (no new EEG / hardware) through
BayesianStoppingController to evaluate accuracy, mean flashes-to-stop,
and Wolpaw ITR across a grid of controller hyperparameters.

Standard replay/ITR math -- implemented in full here.
"""

import math

import numpy as np
import pandas as pd


def simulate_trial(scores, group_ids, groups, true_char_idx, ctrl, lm_prior):
    """
    Replay one recorded trial (one target character) through the
    stopping controller, flash by flash, until it stops or runs out of
    recorded flashes.

    Args:
        scores: sequence of raw classifier scores, one per flash, in
            the order they were presented.
        group_ids: sequence, same length as scores, giving the row/col
            group index (into `groups`) flashed at each step.
        groups: list of lists -- groups[g] gives the character indices
            belonging to group g.
        true_char_idx: int, the ground-truth target character index
            for this trial.
        ctrl: a BayesianStoppingController instance (already
            constructed; reset() is called here).
        lm_prior: shape (n_chars,) passed to ctrl.reset().

    Returns:
        (predicted_char: int, n_flashes: int, correct: bool)
        If the controller never stops before the recorded flashes run
        out, the character with the highest posterior at the end is
        used as the prediction and n_flashes is the total number of
        recorded flashes replayed.
    """
    ctrl.reset(lm_prior)

    n_flashes_used = 0
    predicted_char = None
    pi = len(groups[group_ids[0]]) / ctrl.n_chars if len(scores) > 0 else None

    for s_t, g_id in zip(scores, group_ids):
        group_idx = groups[g_id]
        pi = len(group_idx) / ctrl.n_chars
        stop, char_idx = ctrl.update(group_idx, s_t, pi)
        n_flashes_used += 1
        if stop:
            predicted_char = char_idx
            break

    if predicted_char is None:
        predicted_char = int(np.argmax(ctrl.log_p))

    correct = predicted_char == true_char_idx
    return predicted_char, n_flashes_used, correct


def wolpaw_itr(acc: float, n_classes: int, seconds_per_selection: float) -> float:
    """
    Standard Wolpaw bits/min formula.

    Bits per selection:
        B = log2(N) + acc*log2(acc) + (1-acc)*log2((1-acc)/(N-1))
    (with the convention 0*log2(0) = 0, and B = log2(N) at acc == 1)

    Bits/min = B * (60 / seconds_per_selection).

    Args:
        acc: selection accuracy in [0, 1].
        n_classes: number of possible target classes (e.g. 72).
        seconds_per_selection: mean wall-clock time per character
            selection, in seconds.

    Returns:
        Information transfer rate in bits/minute.
    """
    if n_classes <= 1:
        raise ValueError("n_classes must be > 1")
    if not (0.0 <= acc <= 1.0):
        raise ValueError("acc must be in [0, 1]")
    if seconds_per_selection <= 0:
        raise ValueError("seconds_per_selection must be positive")

    log2_n = math.log2(n_classes)

    if acc == 1.0:
        bits_per_selection = log2_n
    elif acc == 0.0:
        bits_per_selection = log2_n + math.log2(1.0 / (n_classes - 1))
    else:
        bits_per_selection = (
            log2_n
            + acc * math.log2(acc)
            + (1 - acc) * math.log2((1 - acc) / (n_classes - 1))
        )

    selections_per_min = 60.0 / seconds_per_selection
    return bits_per_selection * selections_per_min


def run_sweep(trials: list, groups: list, param_grid: dict,
              seconds_per_flash: float = 0.2,
              controller_cls=None) -> pd.DataFrame:
    """
    Sweep controller hyperparameters over recorded trials and summarize
    performance.

    Args:
        trials: list of dicts, one per recorded trial, each with keys:
            "scores" (sequence of raw scores), "group_ids" (sequence of
            group indices, aligned with scores), "true_char_idx" (int),
            "lm_prior" (shape (n_chars,)).
        groups: list of lists mapping group index -> character indices
            (shared across all trials, i.e. the flash matrix layout).
        param_grid: dict mapping BayesianStoppingController constructor
            kwarg names to lists of values to try, e.g.
            {"theta": [0.8, 0.85, 0.9], "t_min": [4, 6]}.
            The Cartesian product of all values is swept.
        seconds_per_flash: wall-clock seconds per flash, used to convert
            mean flashes-to-stop into seconds-per-selection for ITR.
        controller_cls: the controller class to instantiate (defaults
            to BayesianStoppingController; overridable for testing).

    Returns:
        pd.DataFrame, one row per hyperparameter configuration, with
        columns for each swept param plus "mean_accuracy",
        "mean_flashes", and "itr_bits_per_min".
    """
    if controller_cls is None:
        from stopping.bayesian_stopping import BayesianStoppingController
        controller_cls = BayesianStoppingController

    param_names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in param_names]

    n_chars = len(next(iter(trials), {}).get("lm_prior", [])) or 72

    rows = []
    for combo in _cartesian_product(value_lists):
        kwargs = dict(zip(param_names, combo))
        ctrl = controller_cls(n_chars=n_chars, **kwargs)

        n_correct = 0
        total_flashes = 0
        for trial in trials:
            _, n_flashes, correct = simulate_trial(
                trial["scores"],
                trial["group_ids"],
                groups,
                trial["true_char_idx"],
                ctrl,
                trial["lm_prior"],
            )
            n_correct += int(correct)
            total_flashes += n_flashes

        n_trials = len(trials)
        mean_accuracy = n_correct / n_trials if n_trials else float("nan")
        mean_flashes = total_flashes / n_trials if n_trials else float("nan")
        seconds_per_selection = mean_flashes * seconds_per_flash
        itr = wolpaw_itr(mean_accuracy, n_chars, seconds_per_selection) \
            if seconds_per_selection > 0 else float("nan")

        row = dict(zip(param_names, combo))
        row["mean_accuracy"] = mean_accuracy
        row["mean_flashes"] = mean_flashes
        row["itr_bits_per_min"] = itr
        rows.append(row)

    return pd.DataFrame(rows)


def _cartesian_product(value_lists):
    """Yield the Cartesian product of a list of value lists."""
    if not value_lists:
        yield ()
        return
    head, *rest = value_lists
    for value in head:
        for rest_combo in _cartesian_product(rest):
            yield (value,) + rest_combo
