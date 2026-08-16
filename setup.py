"""
MOABB install/download check + metadata exploration.

Scope note: this file only covers *data plumbing* (download, shape/metadata
verification). No preprocessing, spatial filtering, or classification logic
lives here on purpose -- that's yours to implement starting Day 4.

Run:
    python day2_moabb_setup.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_008, BNCI2014_009
from moabb.paradigms import P300


# ---------------------------------------------------------------------------
# 1. Dataset registry
#    Keeping this as a dict (rather than hardcoding one dataset everywhere)
#    means your Day 6 evaluation harness can loop over both datasets with
#    zero code changes -- worth setting up this way now.
# ---------------------------------------------------------------------------
DATASETS = {
    "BNCI2014_008": BNCI2014_008,  # 8ch, ALS patients, matches your hardware channel count
    "BNCI2014_009": BNCI2014_009,  # 16ch, healthy subjects, more sessions/subject
}


def load_dataset(name: str, subjects: list[int] | None = None):
    """
    Download (if needed) and load a MOABB P300 dataset.

    Parameters
    ----------
    name : one of DATASETS.keys()
    subjects : subset of subject IDs to pull. None = all subjects.
               Start with 1-2 subjects while you're debugging the pipeline --
               full-dataset pulls are slow and you don't need them yet.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
    y : ndarray of str, shape (n_epochs,)         -- 'Target' / 'NonTarget'
    metadata : DataFrame with columns ['subject', 'session', 'run']
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")

    dataset = DATASETS[name]()
    paradigm = P300()  # MOABB's P300 paradigm handles epoching/resampling defaults for you

    subjects = subjects or dataset.subject_list
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    return X, y, metadata


# ---------------------------------------------------------------------------
# 2. Sanity-check report
#    This is the "how do I know it worked" checklist from Day 2 turned into
#    code you can rerun on any dataset/subject subset.
# ---------------------------------------------------------------------------
def sanity_check(name: str, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    # --- shapes ---
    n_epochs, n_channels, n_times = X.shape
    print(f"X.shape           = {X.shape}  (epochs, channels, timepoints)")
    print(f"y.shape            = {y.shape}")
    print(f"metadata.shape     = {metadata.shape}")

    # --- label distribution ---
    labels, counts = np.unique(y, return_counts=True)
    print(f"\nLabel distribution:")
    for lab, cnt in zip(labels, counts):
        pct = 100 * cnt / len(y)
        print(f"  {lab:12s} {cnt:6d}  ({pct:5.1f}%)")
    ratio = counts.max() / counts.min()
    print(f"  imbalance ratio  ~{ratio:.1f}:1")

    # --- metadata structure ---
    print(f"\nMetadata columns   = {list(metadata.columns)}")
    for col in ("subject", "session", "run"):
        if col in metadata.columns:
            uniq = sorted(metadata[col].unique().tolist())
            print(f"  unique {col:8s} = {uniq}")
        else:
            print(f"  [!] expected column '{col}' not found -- check paradigm/dataset version")

    # --- per-subject epoch counts (catches silent partial-download issues) ---
    if "subject" in metadata.columns:
        print(f"\nEpochs per subject:")
        print(metadata.groupby("subject").size().to_string())

    # --- consistency check ---
    assert len(y) == len(metadata) == n_epochs, "X/y/metadata length mismatch -- investigate before proceeding"
    print("\n[OK] X, y, metadata lengths are consistent.")


if __name__ == "__main__":
    # Start small: 2 subjects, one dataset, to keep the first run fast.
    X, y, metadata = load_dataset("BNCI2014_008", subjects=[1, 2])
    sanity_check("BNCI2014_008", X, y, metadata)

    # Uncomment once the above looks right:
    # X9, y9, meta9 = load_dataset("BNCI2014_009", subjects=[1, 2])
    # sanity_check("BNCI2014_009", X9, y9, meta9)