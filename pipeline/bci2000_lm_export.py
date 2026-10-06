"""
Export the Thai Transformer LM (a stopping/gpt_dev.ipynb checkpoint) for BCI2000's
CheckerboardSpeller, which runs the forward pass in C++ (bci2000/CheckerboardSpeller/ThaiGPT.h)
and must produce the same prior as stopping/transformer_lm.ThaiTransformerLM.get_prior.

    python -m pipeline.bci2000_lm_export --checkpoint models/thai_gpt_checkpoint.pt \\
        --out models/thai_gpt.lmbin

Only loads the finished checkpoint (CLAUDE.md: the model is trained on Colab, never here).

File format (little-endian), read by ThaiGPT.h:
  line  "THAIGPT 1"
  line  "vocab_size block_size n_embd n_head n_layer unk_id"
  line  the vocabulary as hex code points, in token-id order (every entry is one code point)
  line  "checkpoint <sha256 of the .pt file>"
  line  "matrix <hex code points of characters_th.json, in order>" (the matrix the file was checked against)
  line  "DATA"
  float32 tensors, row-major as PyTorch stores them ([out, in] for linear layers), in this order:
    token embedding (V, C); position embedding (B, C);
    per layer: ln1 weight, bias (C); per head: key, query, value (hs, C); proj weight (C, C), bias (C);
               ln2 weight, bias (C); ffn1 weight (4C, C), bias (4C); ffn2 weight (C, 4C), bias (C);
    ln_f weight, bias (C); lm_head weight (V, C), bias (V).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pipeline.manifest import REPO_ROOT, load_characters_file, sha256_file

FORMAT_LINE = "THAIGPT 1"
DEFAULT_CHARACTERS = REPO_ROOT / "collection" / "characters_th.json"


class LMExportError(ValueError):
    """The checkpoint cannot be exported for this matrix."""


def tensor_order(config: dict) -> list[str]:
    names = ["token_embedding_table.weight", "position_embedding_table.weight"]
    for layer in range(config["n_layer"]):
        p = f"blocks.{layer}"
        names += [f"{p}.ln1.weight", f"{p}.ln1.bias"]
        for h in range(config["n_head"]):
            names += [f"{p}.sa.heads.{h}.{w}.weight" for w in ("key", "query", "value")]
        names += [f"{p}.sa.proj.weight", f"{p}.sa.proj.bias", f"{p}.ln2.weight", f"{p}.ln2.bias",
                  f"{p}.ffwd.net.0.weight", f"{p}.ffwd.net.0.bias", f"{p}.ffwd.net.2.weight", f"{p}.ffwd.net.2.bias"]
    return names + ["ln_f.weight", "ln_f.bias", "lm_head.weight", "lm_head.bias"]


def export_lm(checkpoint_path, out_path, characters_path=DEFAULT_CHARACTERS) -> Path:
    import torch  # only here: the BCI2000 side needs no torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config, vocab, state = ckpt["config"], list(ckpt["vocab"]), ckpt["model_state_dict"]
    if len(vocab) != config["vocab_size"] or any(len(v) != 1 for v in vocab):
        raise LMExportError("every vocabulary entry must be exactly one character (code point)")
    if config["n_embd"] % config["n_head"]:
        raise LMExportError(f"n_embd {config['n_embd']} is not divisible by n_head {config['n_head']}")
    characters = load_characters_file(characters_path)
    stoi = {c: i for i, c in enumerate(vocab)}
    missing = [c for c in characters if c not in stoi]
    if missing:   # the same refusal as ThaiTransformerLM: the checkpoint was trained for another matrix
        raise LMExportError(f"checkpoint vocab is missing matrix character(s) {missing!r}; retrain on Colab "
                            "against the current characters_th.json")
    names = tensor_order(config)
    unused = sorted(k for k in state if k not in names and not k.endswith(".tril"))
    if unused or any(n not in state for n in names):
        raise LMExportError(f"checkpoint layout differs from ThaiGPT: unexpected {unused}, "
                            f"missing {[n for n in names if n not in state]}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    with open(tmp, "wb") as f:
        header = [
            FORMAT_LINE,
            " ".join(str(config[k]) for k in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"))
            + f" {len(vocab) - 1}",                                  # unk = the last entry, as ThaiTransformerLM
            " ".join(f"{ord(c):x}" for c in vocab),
            f"checkpoint {sha256_file(checkpoint_path)}",
            "matrix " + " ".join(f"{ord(c):x}" for c in characters),
            "DATA",
        ]
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for n in names:
            f.write(np.ascontiguousarray(state[n].detach().cpu().numpy(), dtype="<f4").tobytes())
    tmp.replace(out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export the Thai Transformer LM for BCI2000.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--characters", default=str(DEFAULT_CHARACTERS))
    args = p.parse_args(argv)
    try:
        out = export_lm(args.checkpoint, args.out, args.characters)
    except (LMExportError, OSError, KeyError) as exc:
        print(f"bci2000_lm_export: error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
