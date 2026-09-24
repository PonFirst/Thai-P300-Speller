"""
Compare the two Thai character-level GPT checkpoints trained in stopping/gpt_dev.ipynb:

  - "old" (first pass): 0.88M params, a 25M-character subsample, block_size=128 --
    stopping/data/thai_gpt_checkpoint.pt. Trained locally to prove the pipeline was correct
    before spending real compute on it.
  - "new" (full corpus): trained on Colab per the notebook's Colab-adapted cells --
    models/thai_gpt_checkpoint.pt.

Falls back to comparing "new" against the n-gram LM (stopping/lm_prior.py) if the old checkpoint
isn't present, since that's the only other LM prior this project has.

Both checkpoints marginalize down to the same 72-character speller matrix
(collection/characters_th.json) via get_prior(), so their outputs are directly comparable even
though the two models have different internal vocabularies (275 vs 1661 tokens) and different
architectures. Their own self-reported training losses are NOT comparable this way -- different
corpora, different vocab sizes, different train/val splits -- see the note this script prints.

CPU-only, tiny forward passes -- safe to run locally.
Run: PY="$(conda info --base)/envs/p300speller/bin/python"; $PY stopping/compare_transformers.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from stopping.transformer_lm import ThaiTransformerLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_CHECKPOINT = os.path.join(REPO_ROOT, "stopping", "data", "thai_gpt_checkpoint.pt")
NEW_CHECKPOINT = os.path.join(REPO_ROOT, "models", "thai_gpt_checkpoint.pt")
CHARACTERS_PATH = os.path.join(REPO_ROOT, "collection", "characters_th.json")

TEST_CONTEXTS = ["ประเทศไ", "สวัส", "ขอบ", "ผม", "อยาก", "ไป", ""]

# Kept as an alias -- stopping/model_comparison.py and any external callers imported this
# name before the model definition moved to stopping/transformer_lm.py.
LoadedTransformer = ThaiTransformerLM


def top3_str(p, matrix_chars):
    top3 = np.argsort(p)[::-1][:3]
    return ", ".join(f"{matrix_chars[i]!r}={p[i]:.2f}" for i in top3)


def compare_transformers(matrix_chars):
    print(f"old checkpoint: {OLD_CHECKPOINT}")
    print(f"new checkpoint: {NEW_CHECKPOINT}\n")
    old = LoadedTransformer(OLD_CHECKPOINT, matrix_chars)
    new = LoadedTransformer(NEW_CHECKPOINT, matrix_chars)

    print(f"{'':20s} {'old (first pass)':>20s} {'new (full corpus)':>20s}")
    for label, key in [("parameters (M)", None), ("block_size", "block_size"),
                        ("n_embd", "n_embd"), ("n_head", "n_head"), ("n_layer", "n_layer")]:
        if key is None:
            print(f"{label:20s} {old.n_params/1e6:>19.2f}M {new.n_params/1e6:>19.2f}M")
        else:
            print(f"{label:20s} {old.config[key]:>20d} {new.config[key]:>20d}")
    print(f"{'vocab size':20s} {len(old.vocab):>20d} {len(new.vocab):>20d}")

    print(f"\n{'context':22s} | {'old top-3':38s} | {'new top-3':38s}")
    print("-" * 105)
    for ctx in TEST_CONTEXTS:
        label = repr(ctx) if ctx else "(empty)"
        s_old = top3_str(old.get_prior(ctx), matrix_chars)
        s_new = top3_str(new.get_prior(ctx), matrix_chars)
        print(f"{label:22s} | {s_old:38s} | {s_new:38s}")

    print(
        "\nNote: each model's own reported training loss (old: val ~2.01 after 5000 steps on a "
        "25M-char\nsubsample; new: val ~1.16 after 20000 steps on the full corpus) is NOT a fair "
        "apples-to-apples\nnumber -- different corpora, different vocab sizes, different splits. "
        "The top-3 table above is\nthe fair comparison: both marginalize to the same 72-character "
        "matrix from characters_th.json,\nso their get_prior() outputs occupy the same space and "
        "are directly comparable."
    )


def compare_against_ngram(matrix_chars):
    from stopping.lm_prior import ThaiNgramLM

    print(f"No old-pass checkpoint at {OLD_CHECKPOINT} -- comparing the new transformer against "
          f"the n-gram LM instead.\n")
    print(f"new checkpoint: {NEW_CHECKPOINT}\n")
    new = LoadedTransformer(NEW_CHECKPOINT, matrix_chars)

    corpus_path = os.path.join(REPO_ROOT, "stopping", "data", "thai_wiki_sample.txt")
    if not os.path.isfile(corpus_path):
        raise SystemExit(
            f"No cached corpus at {corpus_path} to fit the n-gram LM against -- run gpt_dev.ipynb "
            f"at least through its data-download cell first, or point corpus_path at a corpus you "
            f"already have.")
    ngram = ThaiNgramLM.from_corpus_path(corpus_path, matrix_chars, order=3, alpha=0.5,
                                          backoff_weight=0.4)

    print(f"{'context':22s} | {'transformer top-3':38s} | {'n-gram top-3':38s}")
    print("-" * 105)
    for ctx in TEST_CONTEXTS:
        label = repr(ctx) if ctx else "(empty)"
        s_new = top3_str(new.get_prior(ctx), matrix_chars)
        s_ngram = top3_str(ngram.get_prior(ctx), matrix_chars)
        print(f"{label:22s} | {s_new:38s} | {s_ngram:38s}")


def main():
    matrix_chars = json.load(open(CHARACTERS_PATH, encoding="utf-8"))["characters"]
    if os.path.isfile(OLD_CHECKPOINT):
        compare_transformers(matrix_chars)
    else:
        compare_against_ngram(matrix_chars)


if __name__ == "__main__":
    main()
