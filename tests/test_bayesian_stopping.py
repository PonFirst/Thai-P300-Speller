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


# The four tests below are EXPECTED TO FAIL with NotImplementedError until
# reset()/llr()/update() are filled in by hand in bayesian_stopping.py --
# that failure is correct, not a bug in this test file or a regression.


def test_reset_not_implemented():
    """reset() is left for the author to implement -- see bayesian_stopping.py."""
    ctrl = BayesianStoppingController(n_chars=4)
    ctrl.reset(np.array([0.25, 0.25, 0.25, 0.25]))


def test_llr_not_implemented():
    """llr() is left for the author to implement -- see bayesian_stopping.py."""
    ctrl = BayesianStoppingController(n_chars=4)
    ctrl.llr(s_t=0.0, pi=0.5)


def test_update_not_implemented():
    """update() is left for the author to implement -- see bayesian_stopping.py."""
    ctrl = BayesianStoppingController(n_chars=4)
    ctrl.log_p = np.log(np.array([0.25, 0.25, 0.25, 0.25]))
    ctrl.t = 0
    ctrl.update(group_idx=[0, 1], s_t=0.0, pi=0.5)


def test_toy_bayes_update_four_characters():
    """
    Hand-verifiable toy example (uniform prior, N=4, group={0,1},
    mocked llr() returning log(3)):
        log p_tilde = log(0.25) + [log3, log3, 0, 0]
        p_tilde = [0.75, 0.75, 0.25, 0.25], sum=2
        p = [0.375, 0.375, 0.125, 0.125]

    This test is EXPECTED TO FAIL with NotImplementedError until
    update() is filled in by hand -- that failure is correct, not a bug.
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
