"""
Language-model priors for the stopping controller, built from the session
manifest.

Every type is a builder registered in LM_BUILDERS, taking
(lm_config, manifest, characters) and returning any object with
get_prior(context: str) -> np.ndarray. build_language_model() wraps it in
ValidatedLanguageModel, which checks every prior at the boundary, so a buggy
model fails loudly instead of feeding the controller a distribution that
doesn't sum to 1.

"transformer" loads a stopping/gpt_dev.ipynb checkpoint via
stopping.transformer_lm.ThaiTransformerLM -- same factory pattern as "ngram",
just a different builder registered in LM_BUILDERS. That module imports
torch, so it's imported lazily inside _build_transformer(): a manifest using
"uniform" or "ngram" (the common case on a fresh install, before a Transformer
checkpoint exists) never needs torch to import cleanly.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from stopping.lm_prior import ThaiNgramLM

PRIOR_SUM_TOLERANCE = 1e-6


class LanguageModelError(Exception):
    """A language model is misconfigured or produced an invalid prior."""


class LanguageModel(Protocol):
    def get_prior(self, context: str) -> np.ndarray: ...


def validate_prior(prior, n_chars: int, *, source: str = "language model") -> np.ndarray:
    """Return prior as a fresh float array, or raise LanguageModelError."""
    try:
        arr = np.array(prior, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LanguageModelError(f"{source}: prior is not numeric ({exc})") from exc
    if arr.shape != (n_chars,):
        raise LanguageModelError(f"{source}: prior must have shape ({n_chars},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        bad = np.flatnonzero(~np.isfinite(arr))[:5].tolist()
        raise LanguageModelError(f"{source}: prior has non-finite entries at indices {bad}")
    if np.any(arr < 0):
        bad = np.flatnonzero(arr < 0)[:5].tolist()
        raise LanguageModelError(f"{source}: prior has negative entries at indices {bad}")
    total = float(arr.sum())
    if abs(total - 1.0) > PRIOR_SUM_TOLERANCE:
        raise LanguageModelError(
            f"{source}: prior must sum to 1 within {PRIOR_SUM_TOLERANCE:g}, sums to {total!r}")
    return arr


class UniformLanguageModel:
    """Exactly 1/n_chars for every character, whatever the context."""

    def __init__(self, n_chars: int):
        self.n_chars = n_chars

    def get_prior(self, context: str) -> np.ndarray:
        return np.full(self.n_chars, 1.0 / self.n_chars)


class ValidatedLanguageModel:
    def __init__(self, inner, n_chars: int, lm_type: str):
        self.inner = inner
        self.n_chars = n_chars
        self.lm_type = lm_type

    def get_prior(self, context: str) -> np.ndarray:
        if not isinstance(context, str):
            raise LanguageModelError(f"context must be a str, got {type(context).__name__}")
        return validate_prior(self.inner.get_prior(context), self.n_chars,
                              source=f"{self.lm_type} language model")


def _build_uniform(lm_config: dict, manifest, characters: list[str]):
    return UniformLanguageModel(len(characters))


def _build_ngram(lm_config: dict, manifest, characters: list[str]):
    corpus = manifest.corpus_path
    if corpus is None or not corpus.is_file():
        raise LanguageModelError(f"ngram language model: corpus not found at {corpus}")
    return ThaiNgramLM.from_corpus_path(str(corpus), characters, order=lm_config["order"],
                                        alpha=lm_config["alpha"],
                                        backoff_weight=lm_config["backoff_weight"])


def _build_transformer(lm_config: dict, manifest, characters: list[str]):
    checkpoint = manifest.transformer_checkpoint_path
    if checkpoint is None or not checkpoint.is_file():
        raise LanguageModelError(f"transformer language model: checkpoint not found at {checkpoint}")
    if checkpoint.suffix.lower() == ".lmbin":
        # The BCI2000 export of the same weights, run in NumPy: no torch (the Windows package)
        from stopping.thai_gpt_numpy import LmbinError, ThaiGPTNumpy
        try:
            return ThaiGPTNumpy(checkpoint, characters)
        except LmbinError as exc:
            raise LanguageModelError(f"transformer language model: {exc}") from exc
    from stopping.transformer_lm import ThaiTransformerLM  # torch: only needed for a .pt checkpoint
    return ThaiTransformerLM.from_checkpoint_path(str(checkpoint), characters)


LM_BUILDERS: dict[str, Callable] = {
    "uniform": _build_uniform,
    "ngram": _build_ngram,
    "transformer": _build_transformer,
}


def build_language_model(manifest, characters: list[str]) -> ValidatedLanguageModel:
    """
    manifest: a pipeline.manifest.SessionManifest (uses .lm_config and .corpus_path).
    characters: the manifest's character list, which fixes the prior's index order.
    """
    lm_config = manifest.lm_config
    lm_type = lm_config.get("type")
    builder = LM_BUILDERS.get(lm_type)
    if builder is None:
        raise LanguageModelError(
            f"language model type {lm_type!r} has no builder; known types {sorted(LM_BUILDERS)}")
    if not characters:
        raise LanguageModelError("characters must be a non-empty list")
    model = ValidatedLanguageModel(builder(lm_config, manifest, list(characters)), len(characters), lm_type)
    model.get_prior("")  # fail at construction, not at the first trial
    return model
