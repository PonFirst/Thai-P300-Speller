"""
Three-way comparison of this project's LM prior candidates: the old (first-pass) transformer,
the new (full-corpus) transformer -- now the production default, selected via
--lm-transformer-checkpoint in pipeline/fit_from_recording.py -- and the n-gram LM
(stopping/lm_prior.py), still available as an alternative LM_BUILDERS["ngram"] type.

Produces stopping/model_comparison_results.json, consumed by a standalone HTML report. Three
axes, matched to what actually matters for get_prior() as used by pipeline/inference_service.py:

  1. Size (parameter/table count) -- informative but not itself a constraint here.
  2. Prediction quality -- top-1/top-3 accuracy and mean probability assigned to the correct
     next character, over a small hand-picked set of common Thai word completions (not derived
     from any corpus, so it's an external check, not a training-data-overlap-biased one).
  3. Inference latency -- wall-clock time for one get_prior(context) call on CPU, which is what
     pipeline/inference_service.py calls exactly once per trial (_start_trial(), before the
     first flash), not once per flash. The real-time budget for it is the stimulus's
     cue_duration_ms (collection/checkerboard_stimulus.py, default 2000ms) between trial_start
     and the first flash -- NOT the 124ms flash SOA, which is a much looser budget than a naive
     "real-time BCI" assumption would suggest.

CPU-only, no corpus download, no training -- safe to run locally.
Run: PY="$(conda info --base)/envs/p300speller/bin/python"; $PY stopping/model_comparison.py
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from stopping.compare_transformers import OLD_CHECKPOINT, NEW_CHECKPOINT, CHARACTERS_PATH
from stopping.lm_prior import ThaiNgramLM
from stopping.transformer_lm import ThaiTransformerLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NGRAM_CORPUS = os.path.join(REPO_ROOT, "stopping", "data", "thai_wiki_sample.txt")
OUTPUT_PATH = os.path.join(REPO_ROOT, "stopping", "model_comparison_results.json")

CUE_DURATION_MS = 2000  # collection/checkerboard_stimulus.py's cue_duration_ms default

# Hand-picked common Thai word completions -- external ground truth (my own knowledge of the
# language), not derived from any training corpus, so accuracy here isn't just "did the model
# memorize its own training text."
LABELED_CONTEXTS = [
    ("ประเทศไท", "ย", "ประเทศไทย (Thailand)"),
    ("สวัส", "ด", "สวัสดี (hello)"),
    ("ขอบคุ", "ณ", "ขอบคุณ (thank you)"),
    ("อยากจ", "ะ", "อยากจะ (want to)"),
    ("สบา", "ย", "สบาย (comfortable/fine)"),
    ("ขอโท", "ษ", "ขอโทษ (sorry)"),
    ("น้ำใ", "จ", "น้ำใจ (kindness)"),
    ("เด็", "ก", "เด็ก (child)"),
    ("โรงเรีย", "น", "โรงเรียน (school)"),
    ("ภาษาไท", "ย", "ภาษาไทย (Thai language)"),
    ("กรุงเท", "พ", "กรุงเทพ (Bangkok)"),
    ("อากา", "ศ", "อากาศ (weather)"),
    ("ผมชื่", "อ", "ผมชื่อ (my name is)"),
    ("ทำงา", "น", "ทำงาน (work)"),
    ("จังหวั", "ด", "จังหวัด (province)"),
    ("รัฐบา", "ล", "รัฐบาล (government)"),
]


def top_k_accuracy(get_prior_fn, matrix_chars, k):
    char_to_idx = {c: i for i, c in enumerate(matrix_chars)}
    hits, probs = 0, []
    for ctx, correct_char, _ in LABELED_CONTEXTS:
        if correct_char not in char_to_idx:
            continue
        p = get_prior_fn(ctx)
        correct_idx = char_to_idx[correct_char]
        topk = np.argsort(p)[::-1][:k]
        if correct_idx in topk:
            hits += 1
        probs.append(float(p[correct_idx]))
    n = len(LABELED_CONTEXTS)
    return hits / n, float(np.mean(probs)), float(np.mean(-np.log2(np.clip(probs, 1e-12, 1))))


def time_inference(get_prior_fn, contexts, n_repeats=20):
    # Warm up once (first call can pay one-time lazy-init cost); then time n_repeats real calls.
    get_prior_fn(contexts[0])
    times_ms = []
    for _ in range(n_repeats):
        ctx = contexts[_ % len(contexts)]
        t0 = time.perf_counter()
        get_prior_fn(ctx)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def main():
    matrix_chars = json.load(open(CHARACTERS_PATH, encoding="utf-8"))["characters"]
    timing_contexts = [ctx for ctx, _, _ in LABELED_CONTEXTS]

    results = {"cue_duration_ms": CUE_DURATION_MS, "models": {}}

    # --- old transformer ---
    print("loading old transformer...")
    t0 = time.perf_counter()
    old = ThaiTransformerLM(OLD_CHECKPOINT, matrix_chars)
    old_load_s = time.perf_counter() - t0
    top1, mean_p, bits = top_k_accuracy(old.get_prior, matrix_chars, k=1)
    top3, _, _ = top_k_accuracy(old.get_prior, matrix_chars, k=3)
    results["models"]["old_transformer"] = {
        "label": "Old transformer (first pass)",
        "n_params": old.n_params,
        "vocab_size": len(old.vocab),
        "config": old.config,
        "load_time_s": old_load_s,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "mean_prob_correct": mean_p,
        "mean_surprisal_bits": bits,
        "latency_ms": time_inference(old.get_prior, timing_contexts),
    }

    # --- new transformer (production default as of pipeline/language_model.py's "transformer" type) ---
    print("loading new transformer...")
    t0 = time.perf_counter()
    new = ThaiTransformerLM(NEW_CHECKPOINT, matrix_chars)
    new_load_s = time.perf_counter() - t0
    top1, mean_p, bits = top_k_accuracy(new.get_prior, matrix_chars, k=1)
    top3, _, _ = top_k_accuracy(new.get_prior, matrix_chars, k=3)
    results["models"]["new_transformer"] = {
        "label": "New transformer (full corpus, production default)",
        "n_params": new.n_params,
        "vocab_size": len(new.vocab),
        "config": new.config,
        "load_time_s": new_load_s,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "mean_prob_correct": mean_p,
        "mean_surprisal_bits": bits,
        "latency_ms": time_inference(new.get_prior, timing_contexts),
    }

    # --- n-gram (the one actually wired into pipeline/language_model.py today) ---
    print(f"fitting n-gram LM on {NGRAM_CORPUS} (this is the slow one-time step)...")
    t0 = time.perf_counter()
    ngram = ThaiNgramLM.from_corpus_path(NGRAM_CORPUS, matrix_chars, order=4, alpha=0.1,
                                          backoff_weight=0.4)
    ngram_fit_s = time.perf_counter() - t0
    top1, mean_p, bits = top_k_accuracy(ngram.get_prior, matrix_chars, k=1)
    top3, _, _ = top_k_accuracy(ngram.get_prior, matrix_chars, k=3)
    n_entries = sum(len(bucket) for order_dict in ngram.counts for bucket in [order_dict])
    n_context_entries = sum(len(d) for d in ngram.counts)
    results["models"]["ngram"] = {
        "label": "N-gram (order=4, alternative LM_BUILDERS type)",
        "n_params": None,
        "n_context_entries": n_context_entries,
        "config": {"order": ngram.order, "alpha": ngram.alpha, "backoff_weight": ngram.backoff_weight},
        "load_time_s": ngram_fit_s,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "mean_prob_correct": mean_p,
        "mean_surprisal_bits": bits,
        "latency_ms": time_inference(ngram.get_prior, timing_contexts),
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {OUTPUT_PATH}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
