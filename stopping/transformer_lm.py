"""
Thai character-level Transformer language model prior for the Bayesian stopping
controller's reset(lm_prior) input -- the trained counterpart to stopping/lm_prior.py's
ThaiNgramLM, loaded from a checkpoint produced by stopping/gpt_dev.ipynb rather than fit
at runtime.

Architecture matches gpt_dev.ipynb's ThaiGPT exactly (same Head/MultiHeadAttention/
FeedForward/Block/ThaiGPT classes, config-parameterized instead of reading notebook
globals, so this is the single place that definition lives -- stopping/compare_transformers.py
and stopping/model_comparison.py both import it from here instead of redefining it).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embd, block_size, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, block_size, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, block_size, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, block_size, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, block_size, n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ThaiGPT(nn.Module):
    """Same architecture as gpt_dev.ipynb's ThaiGPT, config-parameterized so checkpoints
    trained with different hyperparameters (e.g. the first-pass vs. full-corpus runs) can
    all be reconstructed correctly from their own saved config."""

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, block_size, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        return self.lm_head(x)


class ThaiTransformerLM:
    """
    Loads a gpt_dev.ipynb checkpoint and exposes get_prior(context) -> length-len(characters)
    distribution, aligned to the same index order as characters_th.json -- the same contract
    as ThaiNgramLM.get_prior(), so this can be passed directly to
    BayesianStoppingController.reset() or plugged into pipeline/language_model.py's
    LM_BUILDERS registry as a drop-in alternative to the n-gram model.

    Runs on CPU: get_prior() is called once per trial (pipeline/inference_service.py's
    _start_trial(), not once per flash), and a CPU forward pass through this model's size
    takes single-digit milliseconds -- far under the stimulus's multi-second cue-period
    budget (see stopping/model_comparison.py) -- so there's no need for GPU inference here.
    """

    def __init__(self, checkpoint_path: str, characters: list[str]):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        self.vocab = ckpt["vocab"]
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        # gpt_dev.ipynb always builds vocab as sorted(...) + [UNK], so UNK is the last entry.
        self.unk_id = self.stoi[self.vocab[-1]]
        self.block_size = self.config["block_size"]
        self.characters = list(characters)

        # gpt_dev.ipynb's tokenization cell force-includes every matrix character in the
        # vocab regardless of corpus frequency, so a checkpoint trained against the current
        # characters_th.json always has every matrix character as a real token. If the matrix
        # is edited after a checkpoint is trained (it has been, once, this session -- commit
        # b93def5), a stale checkpoint's vocab could be missing a character; get_prior() would
        # then need to fall back to the UNK slot for it, producing a shape/sum-valid but
        # silently wrong distribution that ValidatedLanguageModel's boundary check can't catch
        # (it only checks shape/finiteness/sum-to-1, not per-character correctness). Fail loud
        # here instead, once at construction, matching this project's checkpoint-loading
        # philosophy elsewhere (pipeline/inference_service.py never silently relaxes a
        # channel/rate mismatch to "get something running").
        missing = [c for c in self.characters if c not in self.stoi]
        if missing:
            raise ValueError(
                f"checkpoint {checkpoint_path!r} vocab is missing {len(missing)} matrix "
                f"character(s) {missing!r} -- this checkpoint was trained against a different "
                "characters_th.json than the one this manifest uses. Retrain against the "
                "current matrix, or use a checkpoint that matches it.")

        self.model = ThaiGPT(self.config["vocab_size"], self.config["block_size"],
                              self.config["n_embd"], self.config["n_head"],
                              self.config["n_layer"], self.config["dropout"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: str, characters: list[str]) -> "ThaiTransformerLM":
        return cls(checkpoint_path, characters)

    def _encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in s]

    def get_prior(self, context: str) -> np.ndarray:
        """
        Truncates context to the last block_size characters, runs one forward pass, and
        marginalizes the model's full training-vocabulary softmax down to just the matrix
        characters (same steps as gpt_dev.ipynb's own get_prior(), see its section 9).
        """
        ids = self._encode(context)[-self.block_size:]
        if not ids:
            ids = [self.unk_id]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
        full_probs = F.softmax(logits[0, -1, :], dim=-1).numpy()
        # Direct indexing, not .get() with a fallback: __init__ already guarantees every
        # matrix character is a real vocab token, so a KeyError here would mean that
        # invariant broke somehow -- better to crash loudly than silently reuse UNK's mass.
        raw = np.array([full_probs[self.stoi[c]] for c in self.characters], dtype=float)
        raw = raw + 1e-8
        return raw / raw.sum()
