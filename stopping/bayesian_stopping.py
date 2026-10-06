"""
Bayesian dynamic stopping controller (fixed-threshold version).

Character-level posterior over N Thai matrix characters, updated in the
log domain after each flash via Bayes' rule, with a FIXED confidence
threshold theta (no boundary-aware term -- boundary-probability
thresholding was descoped, may return later, do not build it here).

reset(), llr(), and update() below were implemented by hand by the author
-- the Bayes posterior update, the score-to-LLR conversion, and the
stopping-condition check -- deliberately, to maintain mastery of the math
for technical interviews, and are verified against a hand-worked toy
example in tests/test_bayesian_stopping.py. Do not rewrite or "fix" them
without an explicit ask; that protection doesn't end just because they're
filled in now.
"""

import numpy as np


def log_normalize(log_p: np.ndarray) -> np.ndarray:
    """
    Numerically stable log-domain normalization (log-sum-exp trick).

    Input: unnormalized log-probabilities, shape (N,).
    Output: log-probabilities that exponentiate to sum to 1.
    """
    m = np.max(log_p)
    log_sum = m + np.log(np.sum(np.exp(log_p - m)))
    return log_p - log_sum


class BayesianStoppingController:
    """
    Character-level posterior over n_chars Thai matrix characters,
    updated via log-domain Bayes rule after each flash, with a fixed
    confidence threshold theta.

    Attributes (set by reset()):
        log_p: np.ndarray, shape (n_chars,) -- current log-posterior.
        t: int -- number of flashes seen for the current character.
    """

    def __init__(self, n_chars=72, lam=0.7, theta=0.90, t_min=6,
                 t_max=120, sigmoid_a=1.0, sigmoid_b=0.0, llr_clip=8.0):
        """
        Args:
            n_chars: size of the character matrix (72 for this project).
            lam: blend weight for the language-model prior in reset()
                (p0 = lam * lm_prior + (1 - lam) / n_chars).
            theta: fixed posterior-probability stopping threshold, in
                (0, 1). No boundary term -- this is the entire stopping
                condition's confidence bar.
            t_min: minimum number of flashes before stopping is allowed.
            t_max: maximum number of flashes before a forced stop
                (argmax of the current posterior is returned).
            sigmoid_a, sigmoid_b: Platt-scaling slope/intercept (fitted
                by stopping.calibrate.PlattCalibrator) used by llr() to
                convert a raw classifier score into P(target | score).
            llr_clip: symmetric clip bound applied to each flash's LLR,
                to keep a single noisy flash from swinging the posterior
                to numerical extremes.
        """
        self.n_chars = n_chars
        self.lam = lam
        self.theta = theta
        self.t_min = t_min
        self.t_max = t_max
        self.sigmoid_a = sigmoid_a
        self.sigmoid_b = sigmoid_b
        self.llr_clip = llr_clip

        self.log_p = None
        self.t = 0

    def reset(self, lm_prior: np.ndarray):
        """
        Call once per character, before the first flash.

        Args:
            lm_prior: shape (n_chars,), sums to 1. Language-model prior
                over the next character (from stopping.lm_prior).

        TODO (author-implemented): blend lm_prior with a uniform floor,
            p0 = lam * lm_prior + (1 - lam) / n_chars
        store log(p0) in self.log_p, and reset self.t to 0.
        """
        p0 = self.lam * lm_prior + (1 - self.lam) / self.n_chars
        self.log_p = np.log(p0)
        self.t = 0

    def llr(self, s_t: float, pi: float) -> float:
        """
        Convert a raw classifier score to a clipped log-likelihood ratio.

        Args:
            s_t: raw classifier score for this flash.
            pi: target-in-group prior probability (e.g. 1/6 for a
                row/column flash in a 6x12 virtual matrix), used in the
                likelihood-ratio correction term.

        TODO (author-implemented): the Platt-scaled classifier gives
        P(target | s_t) = sigmoid(sigmoid_a * s_t + sigmoid_b). By
        Bayes' rule with prior pi = P(target) for this flash's group,
        derive
            ell = log[P(s_t | target) / P(s_t | non-target)]
                = (sigmoid_a * s_t + sigmoid_b) + log((1 - pi) / pi)
        clip to +/- self.llr_clip, and return as a float.
        """
        ell = (self.sigmoid_a * s_t + self.sigmoid_b) + np.log((1 - pi) / pi)
        clipped_ell = np.clip(ell, -self.llr_clip, self.llr_clip)
        return float(clipped_ell)

    def update(self, group_idx: list[int], s_t: float, pi: float):
        """
        Process one flash and update the posterior in place.

        Args:
            group_idx: indices of characters included in this flash's
                row/column group.
            s_t: raw classifier score for this flash.
            pi: target-in-group prior probability, passed through to
                llr().

        Returns:
            (stop: bool, char_idx: int or None) -- whether the stopping
            condition has been met and, if so, the winning character
            index (else None).

        TODO (author-implemented): additive log-domain update on
        group_idx only (self.log_p[group_idx] += self.llr(s_t, pi)),
        renormalize via log_normalize(), increment self.t, then check
        the fixed-threshold stopping condition: once self.t >= t_min,
        stop if the posterior max >= self.theta, OR unconditionally
        once self.t >= t_max (forced stop, argmax wins either way).
        """
        self.log_p[group_idx] += self.llr(s_t, pi)
        self.log_p = log_normalize(self.log_p)
        self.t += 1
        max_idx = int(np.argmax(self.log_p))
        max_p = np.exp(self.log_p[max_idx])
        if self.t >= self.t_min and max_p >= self.theta:
            return True, max_idx
        if self.t >= self.t_max:
            return True, max_idx
        return False, None
