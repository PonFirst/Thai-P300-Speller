"""
Class balance + trial-structure EDA.

Goes beyond the Day 2 sanity check: this module is meant to produce the
actual numbers you cite in your methods section (imbalance ratio, per-subject
trial counts) and to flag the grouping structure your later cross-validation
must respect (see the CV-leakage note in config.py's docstring and your
earlier terminology discussion).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR


def summarize_labels(y: np.ndarray) -> pd.DataFrame:
    labels, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({"label": labels, "count": counts})
    df["pct"] = 100 * df["count"] / df["count"].sum()
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def imbalance_ratio(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.min()


def per_group_counts(metadata: pd.DataFrame, y: np.ndarray, group_cols=("subject", "session")) -> pd.DataFrame:
    """
    Trial counts broken down by subject/session, split further by label.
    This is what tells you whether every subject/session has *both* classes
    represented, and whether counts are balanced enough across subjects to
    avoid one subject dominating a pooled analysis.
    """
    df = metadata.copy()
    df["label"] = y
    cols = [c for c in group_cols if c in df.columns]
    return df.groupby(cols + ["label"]).size().unstack(fill_value=0)


def plot_class_balance(y: np.ndarray, title: str, save_path: str | None = None):
    summary = summarize_labels(y)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(summary["label"], summary["count"], color=["#4c72b0", "#dd8452"])
    for i, row in summary.iterrows():
        ax.text(i, row["count"], f"{row['count']}\n({row['pct']:.1f}%)",
                 ha="center", va="bottom")
    ax.set_ylabel("Epoch count")
    ax.set_title(f"Class balance -- {title}\nimbalance ratio ~{imbalance_ratio(y):.1f}:1")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def run(metadata: pd.DataFrame, y: np.ndarray, dataset_name: str, out_dir: str = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    summary = summarize_labels(y)
    print(f"\n--- Class balance: {dataset_name} ---")
    print(summary.to_string(index=False))
    print(f"Imbalance ratio: {imbalance_ratio(y):.2f}:1")

    counts = per_group_counts(metadata, y)
    print("\nPer-subject/session counts:")
    print(counts.to_string())

    # Flag any subject/session missing a class entirely -- would break
    # per-subject classifier training/evaluation downstream.
    if (counts == 0).any().any():
        print("[!] Some subject/session groups are missing a class entirely -- "
              "check before running per-subject evaluation.")

    save_path = os.path.join(out_dir, f"class_balance_{dataset_name}.png")
    plot_class_balance(y, dataset_name, save_path=save_path)
    print(f"Saved: {save_path}")

    return {"summary": summary, "counts": counts, "imbalance_ratio": imbalance_ratio(y)}


if __name__ == "__main__":
    from data_loader import load_arrays

    X, y, metadata = load_arrays("BNCI2014_008", subjects=[1, 2])
    run(metadata, y, "BNCI2014_008")
