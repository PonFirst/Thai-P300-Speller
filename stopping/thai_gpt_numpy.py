"""
The Thai Transformer LM without PyTorch: a NumPy forward pass over the .lmbin export
(pipeline/bci2000_lm_export.py), the same weights BCI2000's CheckerboardSpeller runs in C++
(bci2000/CheckerboardSpeller/ThaiGPT.h).

get_prior(context) follows stopping/transformer_lm.ThaiTransformerLM.get_prior step for step,
in float32 like PyTorch on the CPU: context truncated to the last block_size characters (an
empty context is the <unk> token), one forward pass, softmax over the full vocabulary,
marginalized to the matrix characters plus 1e-8 and renormalized. tests/test_thai_gpt_numpy.py
checks it against the PyTorch model.

It exists so a machine without PyTorch (the Windows package) can fit a model with the
Transformer prior and replay recordings under it: pipeline/language_model.py builds this class
for a "transformer" language model whose file is an .lmbin rather than a .pt checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FORMAT_LINE = "THAIGPT 1"
LAYER_NORM_EPS = 1e-5


class LmbinError(ValueError):
    """The file is not a ThaiGPT export, or not one made for this matrix."""


def read_lmbin(path) -> tuple[dict, list[str], list[str], str, list[np.ndarray]]:
    """(config, vocab, matrix, checkpoint sha256, tensors in pipeline.bci2000_lm_export.tensor_order)."""
    raw = Path(path).read_bytes()
    marker = b"\nDATA\n"
    end = raw.find(marker)
    if not raw.startswith(FORMAT_LINE.encode() + b"\n") or end < 0:
        raise LmbinError(f"{path} is not a Thai GPT export (pipeline/bci2000_lm_export.py)")
    lines = raw[:end].decode("ascii").split("\n")
    if len(lines) != 5 or not lines[3].startswith("checkpoint ") or not lines[4].startswith("matrix "):
        raise LmbinError(f"{path}: damaged header")
    try:
        v, b, c, h, n, unk = (int(x) for x in lines[1].split())
        vocab = [chr(int(x, 16)) for x in lines[2].split()]
        matrix = [chr(int(x, 16)) for x in lines[4].split()[1:]]
    except ValueError as exc:
        raise LmbinError(f"{path}: damaged header ({exc})") from exc
    if len(vocab) != v or unk != v - 1 or c % h:
        raise LmbinError(f"{path}: header sizes are inconsistent")
    config = {"vocab_size": v, "block_size": b, "n_embd": c, "n_head": h, "n_layer": n}
    hs = c // h
    shapes = [(v, c), (b, c)]
    for _ in range(n):
        shapes += [(c,), (c,)] + [(hs, c)] * (3 * h) + [(c, c), (c,), (c,), (c,), (4 * c, c), (4 * c,), (c, 4 * c), (c,)]
    shapes += [(c,), (c,), (v, c), (v,)]
    data = np.frombuffer(raw, dtype="<f4", offset=end + len(marker))
    if data.size != sum(int(np.prod(s)) for s in shapes):
        raise LmbinError(f"{path}: weight data has {data.size} values, the header implies a different size")
    tensors, at = [], 0
    for s in shapes:
        k = int(np.prod(s))
        tensors.append(data[at:at + k].reshape(s).astype(np.float32))
        at += k
    return config, vocab, matrix, lines[3].split()[1], tensors


def _layer_norm(x, w, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + np.float32(LAYER_NORM_EPS)) * w + b


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class ThaiGPTNumpy:
    """Same contract as stopping.transformer_lm.ThaiTransformerLM: get_prior(context) -> len(characters)."""

    def __init__(self, lmbin_path, characters: list[str]):
        self.path = Path(lmbin_path)
        self.config, self.vocab, matrix, self.checkpoint_sha256, t = read_lmbin(self.path)
        self.characters = list(characters)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.unk_id = len(self.vocab) - 1
        self.block_size = self.config["block_size"]
        # The same refusals as ThaiTransformerLM and ThaiGPT.h: a model for another matrix is never used.
        missing = [ch for ch in self.characters if ch not in self.stoi]
        if missing:
            raise LmbinError(f"{self.path}: vocabulary is missing matrix character(s) {missing!r}; "
                             "retrain against the current characters_th.json")
        if matrix != self.characters:
            raise LmbinError(f"{self.path} was exported for a different matrix than characters_th.json")
        c, h = self.config["n_embd"], self.config["n_head"]
        self.tok, self.pos = t[0], t[1]
        self.layers, at = [], 2
        for _ in range(self.config["n_layer"]):
            ln1 = (t[at], t[at + 1]); at += 2
            heads = [(t[at + 3 * i], t[at + 3 * i + 1], t[at + 3 * i + 2]) for i in range(h)]; at += 3 * h
            proj = (t[at], t[at + 1]); at += 2
            ln2 = (t[at], t[at + 1]); at += 2
            ffn = (t[at], t[at + 1], t[at + 2], t[at + 3]); at += 4
            self.layers.append((ln1, heads, proj, ln2, ffn))
        self.ln_f = (t[at], t[at + 1])
        self.head = (t[at + 2], t[at + 3])
        self.scale = np.float32(c ** -0.5)   # PyTorch's Head scales by C ** -0.5 with C = n_embd

    def _logits_last(self, ids: list[int]) -> np.ndarray:
        n = len(ids)
        x = self.tok[ids] + self.pos[:n]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        for (ln1, heads, proj, ln2, ffn) in self.layers:
            y = _layer_norm(x, *ln1)
            outs = []
            for wk, wq, wv in heads:
                k, q, v = y @ wk.T, y @ wq.T, y @ wv.T
                wei = (q @ k.T) * self.scale
                wei[mask] = -np.inf
                outs.append(_softmax(wei) @ v)
            x = x + (np.concatenate(outs, axis=-1) @ proj[0].T + proj[1])
            y = _layer_norm(x, *ln2)
            x = x + (np.maximum(y @ ffn[0].T + ffn[1], 0) @ ffn[2].T + ffn[3])
        x = _layer_norm(x[-1:], *self.ln_f)
        return (x @ self.head[0].T + self.head[1])[0]

    def get_prior(self, context: str) -> np.ndarray:
        ids = [self.stoi.get(ch, self.unk_id) for ch in context][-self.block_size:] or [self.unk_id]
        full = _softmax(self._logits_last(ids).astype(np.float32))
        raw = np.array([full[self.stoi[ch]] for ch in self.characters], dtype=float) + 1e-8
        return raw / raw.sum()
