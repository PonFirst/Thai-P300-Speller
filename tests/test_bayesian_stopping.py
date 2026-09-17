import numpy as np
from stopping.bayesian_stopping import log_normalize, BayesianStoppingController


def test_log_normalize_sums_to_one():
    log_p = np.log(np.array([0.1, 0.2, 0.3, 0.4]))
    result = log_normalize(log_p)
    assert np.isclose(np.exp(result).sum(), 1.0, atol=1e-9)


def test_log_normalize_no_nan_at_extreme_llr():
    log_p = np.array([-50.0, 0.0, 50.0, -50.0])
    result = log_normalize(log_p)
    assert not np.any(np.isnan(result))


# reset()/llr()/update() are now filled in by hand in bayesian_stopping.py
# (see CLAUDE.md). The three tests below check each of them directly
# against the formulas documented in their own docstrings; the fourth is a
# hand-verifiable end-to-end toy example of update() with llr() mocked out.


def test_reset_blends_lm_prior_with_uniform_floor():
    """reset(): p0 = lam * lm_prior + (1 - lam) / n_chars, log_p = log(p0), t = 0."""
    lm_prior = np.array([0.1, 0.2, 0.3, 0.4])
    ctrl = BayesianStoppingController(n_chars=4, lam=0.7)
    ctrl.reset(lm_prior)
    expected_p0 = 0.7 * lm_prior + 0.3 / 4
    assert np.allclose(np.exp(ctrl.log_p), expected_p0, atol=1e-9)
    assert np.isclose(np.exp(ctrl.log_p).sum(), 1.0, atol=1e-9)
    assert ctrl.t == 0


def test_llr_converts_score_to_clipped_log_likelihood_ratio():
    """llr(): Platt-scaled logit plus the group-prior correction term, clipped to +/- llr_clip."""
    ctrl = BayesianStoppingController(n_chars=4, sigmoid_a=2.0, sigmoid_b=-0.5, llr_clip=8.0)
    s_t, pi = 1.0, 1 / 6
    expected = np.clip(2.0 * s_t - 0.5 + np.log((1 - pi) / pi), -8.0, 8.0)
    assert np.isclose(ctrl.llr(s_t, pi), expected, atol=1e-9)

    # clipping actually engages for an extreme score, in both directions
    ctrl2 = BayesianStoppingController(n_chars=4, sigmoid_a=50.0, sigmoid_b=0.0, llr_clip=8.0)
    assert ctrl2.llr(s_t=10.0, pi=0.5) == 8.0
    assert ctrl2.llr(s_t=-10.0, pi=0.5) == -8.0


def test_update_advances_flash_count_and_returns_a_valid_stop_contract():
    """update(): increments t, keeps the posterior normalized, and returns (stop, char_idx)."""
    ctrl = BayesianStoppingController(n_chars=4, t_min=1, t_max=10, theta=0.99)
    ctrl.reset(np.full(4, 0.25))
    stop, pred = ctrl.update(group_idx=[0, 1], s_t=0.0, pi=0.5)
    assert ctrl.t == 1
    assert isinstance(stop, bool)
    assert pred is None or (isinstance(pred, int) and 0 <= pred < 4)
    assert np.isclose(np.exp(ctrl.log_p).sum(), 1.0, atol=1e-9)


def test_toy_bayes_update_four_characters():
    """
    Hand-verifiable toy example (uniform prior, N=4, group={0,1},
    mocked llr() returning log(3)):
        log p_tilde = log(0.25) + [log3, log3, 0, 0]
        p_tilde = [0.75, 0.75, 0.25, 0.25], sum=2
        p = [0.375, 0.375, 0.125, 0.125]
    """
    ctrl = BayesianStoppingController(n_chars=4, lam=1.0, theta=0.99,
                                       t_min=0, t_max=10)
    ctrl.log_p = np.log(np.array([0.25, 0.25, 0.25, 0.25]))
    ctrl.t = 0
    ctrl.llr = lambda s_t, pi: np.log(3)

    stop, pred = ctrl.update(group_idx=[0, 1], s_t=0.0, pi=0.5)
    posterior = np.exp(ctrl.log_p)
    expected = np.array([0.375, 0.375, 0.125, 0.125])
    assert np.allclose(posterior, expected, atol=1e-9), \
        f"Expected {expected}, got {posterior}"
