"""
Chooses which pre-defined flash group to present next, given the
current posterior. Does NOT alter group composition -- only ordering
(design doc Sec. 2.4b). Must be called using the posterior AFTER the LM
prior has already been folded in (BayesianStoppingController.log_p);
this module never queries the LLM directly.
"""

from __future__ import annotations

import numpy as np


class InfoGainFlashSelector:
    def __init__(self, fixed_groups: list):
        """
        Args:
            fixed_groups: list of lists, each inner list = char indices
                belonging to one static row/column/checkerboard group.
                Composition must match what the classifier was trained/
                validated on.
        """
        self.groups = fixed_groups

    def entropy(self, posterior: np.ndarray) -> float:
        """
        Shannon entropy of the posterior, in bits, with a log(0) guard.
        Standard formula -- implemented in full.
        """
        p = np.asarray(posterior, dtype=float)
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def expected_posterior_entropy(self, posterior: np.ndarray,
                                    group_indices: list) -> float:
        """
        Estimate E[H(posterior | flash this group)] by simulating both
        branches (P300 evoked / not evoked) using the calibrated sigmoid
        likelihood, weighted by their current marginal probability under
        `posterior`.

        TODO (design doc Sec. 2.4b -- core contribution):
            p_evoke = marginal P(P300 evoked | posterior, group_indices)
                     (sum of posterior mass inside group_indices, folded
                      through the calibrated likelihood)
            For each branch (evoked / not evoked): compute the resulting
            posterior via the same Bayesian update as
            BayesianStoppingController.update(), then its entropy().
            Return p_evoke * H(posterior_if_evoked)
                 + (1 - p_evoke) * H(posterior_if_not_evoked)
        """
        raise NotImplementedError(
            "expected_posterior_entropy(): implement per design doc Sec. 2.4b"
        )

    def select_next_group(self, posterior: np.ndarray,
                           already_flashed: set) -> int:
        """
        Returns index into self.groups for the next group to flash.

        TODO (design doc Sec. 2.4b -- core contribution):
            argmax over groups not yet flashed this trial of
            [entropy(posterior) - expected_posterior_entropy(posterior, group)]
        """
        raise NotImplementedError(
            "select_next_group(): implement information-gain ordering "
            "per design doc Sec. 2.4b"
        )
