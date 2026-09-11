"""
Runs all 5 classifier pipelines (via benchmark_all.py's get_pipelines(),
not redefined here) on BOTH BNCI2014_008 and BNCI2014_009, and produces one
comparison table covering both datasets at once.

Flash-level metrics only (accuracy / balanced accuracy / AUC) -- row/column
identity isn't available through data_loader.py's standard loading path for
these two datasets, so character-level accuracy/ITR is out of scope here
(see CLAUDE.md's known gotchas and character_decoder.py's docstring).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for data_loader/config

import argparse
import os

import numpy as np
import pandas as pd

from benchmark_all import run_benchmark
from data_loader import load_arrays

DATASETS = ["BNCI2014_008", "BNCI2014_009"]


def summarize_dataset(name: str, y: np.ndarray, results: dict) -> None:
    n_target = int((y == "Target").sum())
    n_nontarget = int((y == "NonTarget").sum())
    imbalance = n_nontarget / n_target if n_target else float("nan")
    winner = max(results, key=lambda p: results[p]["mean_auc"])
    print(f"\n{name} summary: {len(y)} epochs, "
          f"NonTarget:Target = {imbalance:.1f}:1, "
          f"best pipeline = {winner} (AUC={results[winner]['mean_auc']:.3f})")


def main(subjects: list[int], n_splits: int, nfilter: int) -> pd.DataFrame:
    per_dataset_results = {}

    for name in DATASETS:
        print(f"\n{'#'*70}\nBenchmarking {name}, subjects={subjects}\n{'#'*70}")
        X, y, metadata = load_arrays(name, subjects=subjects)
        groups = metadata["subject"].values
        print(f"X: {X.shape}, groups (subjects): {sorted(set(groups))}")

        results = run_benchmark(X, y, groups, n_splits=n_splits, nfilter=nfilter)
        per_dataset_results[name] = results
        summarize_dataset(name, y, results)

    pipeline_names = list(per_dataset_results[DATASETS[0]].keys())
    rows = []
    for pipeline in pipeline_names:
        row = {"Pipeline": pipeline}
        for name in DATASETS:
            res = per_dataset_results[name][pipeline]
            row[f"{name} AUC"] = res["mean_auc"]
            row[f"{name} BalAcc"] = res["mean_balanced_accuracy"]
        rows.append(row)

    df = pd.DataFrame(rows)
    auc_cols = [f"{name} AUC" for name in DATASETS]
    df["_mean_auc"] = df[auc_cols].mean(axis=1)
    df = df.sort_values("_mean_auc", ascending=False).drop(columns="_mean_auc")

    per_dataset_winner = {
        name: max(per_dataset_results[name], key=lambda p: per_dataset_results[name][p]["mean_auc"])
        for name in DATASETS
    }
    combined_winner = df.iloc[0]["Pipeline"]
    if len(set(per_dataset_winner.values())) > 1 or combined_winner not in per_dataset_winner.values():
        print(f"\nNOTE: best pipeline differs across datasets/combined ranking -- "
              f"per-dataset winners: {per_dataset_winner}, combined-ranking winner: {combined_winner}. "
              f"Worth flagging as a real finding, not averaging away.")

    print(f"\n{'#'*70}\nCombined comparison (both datasets, sorted by mean AUC)\n{'#'*70}")
    print(df.to_string(index=False, float_format="{:.3f}".format))

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/all_classifiers_results.csv", index=False)
    print(f"\nSaved to results/all_classifiers_results.csv")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark all 5 classifier pipelines on both BNCI2014_008 and BNCI2014_009."
    )
    parser.add_argument("--subjects", type=int, nargs="*", default=[1, 2, 3, 4, 5],
                          help="Subject IDs, applied to both datasets.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--nfilter", type=int, default=4)
    args = parser.parse_args()

    main(args.subjects, args.n_splits, args.nfilter)
