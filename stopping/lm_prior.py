"""
Thai character-level n-gram language model prior for the Bayesian
stopping controller's reset(lm_prior) input.

Fully implemented (off-the-shelf n-gram counting + add-alpha smoothing +
backoff) -- this is plumbing, not the novel research contribution. The
novel piece is BayesianStoppingController's Bayes update/LLR/stopping
logic in bayesian_stopping.py, left for the author to implement there.

No boundary-probability method in this version -- boundary-aware
thresholding was descoped, see stopping/bayesian_stopping.py.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np


class ThaiNgramLM:
    """
    Character-level n-gram model with add-alpha (Laplace-family)
    smoothing and stupid backoff to lower orders on unseen context.

    get_prior(context) returns a length-len(characters) distribution
    aligned to the same index order as characters_th.json, so it can be
    passed directly to BayesianStoppingController.reset().
    """

    def __init__(self, characters: list[str], order: int = 4, alpha: float = 0.1,
                 backoff_weight: float = 0.4):
        """
        Args:
            characters: ordered list of matrix characters (index order
                must match characters_th.json's "characters" list --
                this defines get_prior()'s output index alignment).
            order: max n-gram order (context length + 1), e.g. order=4
                means up to trigram context (3 preceding chars) predicting
                the 4th. Backs off to order-1, order-2, ... down to the
                unigram (order=1, no context) on unseen context.
            alpha: add-alpha smoothing constant added to every count
                before normalizing, so unseen (context, next-char) pairs
                still get nonzero probability at the order they're
                evaluated at.
            backoff_weight: multiplicative discount applied to a lower
                order's distribution when the higher order has no counts
                for the given context at all (stupid backoff, Brants et
                al. 2007 -- simple and robust rather than proper
                Katz/Kneser-Ney backoff).
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        self.characters = list(characters)
        self.char_to_idx = {c: i for i, c in enumerate(self.characters)}
        self.n_chars = len(self.characters)
        self.order = order
        self.alpha = alpha
        self.backoff_weight = backoff_weight

        # counts[k][context] -> Counter(next_char -> count), for context
        # lengths k = 0 (unigram) .. order - 1.
        self.counts: list[dict[str, Counter]] = [defaultdict(Counter) for _ in range(order)]

    @classmethod
    def from_corpus_path(cls, corpus_path: str, characters: list[str],
                          order: int = 4, alpha: float = 0.1,
                          backoff_weight: float = 0.4) -> "ThaiNgramLM":
        """Load a plain-text UTF-8 Thai corpus and fit an n-gram model on it."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        model = cls(characters, order=order, alpha=alpha, backoff_weight=backoff_weight)
        model.fit(text)
        return model

    def fit(self, text: str) -> None:
        """
        Count n-gram occurrences of every order from the given text.
        Characters not in self.char_to_idx are treated as unigram-only
        noise and skipped when they'd be the *predicted* character
        (context characters outside the matrix are still kept, since
        they're still valid preceding context for e.g. spaces/digits).
        """
        for i, ch in enumerate(text):
            if ch not in self.char_to_idx:
                continue
            for k in range(self.order):
                if i - k < 0:
                    continue
                context = text[i - k:i]
                self.counts[k][context][ch] += 1

    def _order_distribution(self, k: int, context: str) -> np.ndarray | None:
        """
        Smoothed distribution over self.characters at context length k
        (0 = unigram), or None if this context was never observed at
        this order (nothing to smooth from -- caller backs off).
        """
        bucket = self.counts[k].get(context)
        if not bucket:
            return None
        counts = np.array([bucket.get(c, 0) for c in self.characters], dtype=float)
        counts += self.alpha
        return counts / counts.sum()

    def get_prior(self, context: str) -> np.ndarray:
        """
        Returns shape (n_chars,) distribution over matrix characters
        given recent context, aligned to self.characters' index order.

        Tries the longest available context (up to order - 1 preceding
        characters) first; backs off to shorter contexts (discounted by
        backoff_weight per level dropped) whenever a given context was
        never observed in the training corpus, all the way down to the
        add-alpha-smoothed unigram distribution, which is always
        defined (every matrix character has a nonzero floor).
        """
        max_k = min(self.order - 1, len(context))
        for k in range(max_k, -1, -1):
            ctx = context[len(context) - k:] if k > 0 else ""
            dist = self._order_distribution(k, ctx)
            if dist is not None:
                discount = self.backoff_weight ** (max_k - k)
                dist = discount * dist + (1 - discount) * self._unigram_floor()
                return dist / dist.sum()
        return self._unigram_floor()

    def _unigram_floor(self) -> np.ndarray:
        """Uniform distribution -- the base case when there are no counts at all."""
        return np.full(self.n_chars, 1.0 / self.n_chars)


def load_characters(characters_json_path: str) -> list[str]:
    """Load the ordered character list from a characters_th.json-style file."""
    with open(characters_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["characters"] if isinstance(data, dict) else data
