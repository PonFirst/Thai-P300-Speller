import numpy as np
from stopping.lm_prior import ThaiNgramLM

TOY_CHARS = list("กขคงจ")


def test_get_prior_sums_to_one_and_matches_length():
    lm = ThaiNgramLM(TOY_CHARS, order=3, alpha=0.1)
    lm.fit("กขคกขงกขคจกขคงกขคกขงกขคจ" * 5)
    prior = lm.get_prior("กข")
    assert prior.shape == (len(TOY_CHARS),)
    assert np.isclose(prior.sum(), 1.0, atol=1e-9)
    assert np.all(prior >= 0)


def test_unseen_context_backs_off_to_uniform():
    lm = ThaiNgramLM(TOY_CHARS, order=3, alpha=0.1)
    lm.fit("กขคกขงกขคจ")
    prior = lm.get_prior("จจจ")  # context never seen in training text
    assert prior.shape == (len(TOY_CHARS),)
    assert np.isclose(prior.sum(), 1.0, atol=1e-9)


def test_empty_model_returns_uniform_floor():
    lm = ThaiNgramLM(TOY_CHARS, order=3, alpha=0.1)
    prior = lm.get_prior("ก")
    assert np.allclose(prior, np.full(len(TOY_CHARS), 1.0 / len(TOY_CHARS)))
