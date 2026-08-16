"""
Run the full EDA suite end to end.

Usage:
    python run_eda.py --dataset BNCI2014_008 --subjects 1 2

Each module (class_balance, signal_quality, erp_analysis, topography) can
also be run standalone via `python <module>.py` for faster iteration while
you're tweaking one piece -- this script just chains all four and prints a
consolidated summary at the end.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import class_balance
import erp_analysis
import signal_quality
import topography
from config import DATASET_CONFIG, OUTPUT_DIR
from data_loader import load_arrays, load_epochs


def main(dataset_name: str, subjects: list[int] | None):
    print(f"{'#'*70}\nEDA run: {dataset_name}, subjects={subjects or 'all'}\n{'#'*70}")

    X, y, metadata = load_arrays(dataset_name, subjects=subjects)
    epochs = load_epochs(dataset_name, subjects=subjects)
    ch_names = epochs.info["ch_names"]
    sfreq = DATASET_CONFIG.get(dataset_name, {}).get("sfreq", epochs.info["sfreq"])
    times = np.linspace(0, X.shape[2] / sfreq, X.shape[2])

    # Infer target/nontarget label strings from the data itself rather than
    # hardcoding, so this script generalizes to a differently-labeled
    # dataset (e.g. your self-collected Thai speller data) without editing.
    labels = np.unique(y)
    target_label = next((l for l in labels if "target" == str(l).lower()), None)
    nontarget_label = next((l for l in labels if "nontarget" in str(l).lower().replace("-", "").replace("_", "")), None)
    if target_label is None or nontarget_label is None:
        raise ValueError(
            f"Could not infer Target/NonTarget labels from {labels}. "
            f"Pass them explicitly if your label scheme differs."
        )

    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    results["class_balance"] = class_balance.run(metadata, y, dataset_name, out_dir=out_dir)
    results["signal_quality"] = signal_quality.run(X, epochs, ch_names, dataset_name, out_dir=out_dir)
    results["erp_analysis"] = erp_analysis.run(
        X, y, times, ch_names, target_label, nontarget_label, dataset_name, out_dir=out_dir
    )
    results["topography"] = topography.run(
        X, y, times, epochs.info, target_label, nontarget_label, dataset_name, out_dir=out_dir
    )

    print(f"\n{'#'*70}\nSUMMARY -- {dataset_name}\n{'#'*70}")
    print(f"Epochs: {X.shape[0]} | Channels: {len(ch_names)} | "
          f"Imbalance: {results['class_balance']['imbalance_ratio']:.1f}:1")
    print(f"Would reject {results['signal_quality']['rejection']['pct_rejected']:.1f}% "
          f"of epochs at current PTP threshold")
    print(f"Largest target-ERP peak channel: {results['erp_analysis']['best_channel']}")
    print(f"Target-ERP topomap peak latency: {results['topography']['peak_time']*1000:.0f}ms")
    print(f"\nAll figures saved to: {out_dir}/")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the P300 EDA suite.")
    parser.add_argument("--dataset", default="BNCI2014_008", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--subjects", type=int, nargs="*", default=[1, 2],
                          help="Subject IDs to include. Omit to use all subjects (slower).")
    args = parser.parse_args()

    main(args.dataset, args.subjects or None)
