"""
Scalp topography EDA: where on the scalp is the P300 effect largest?

Classical P300 topography peaks parietal/centro-parietal (Pz/Cz region).
Confirming this in your own data is a cheap, high-value sanity check --
if your topomap peaks somewhere anatomically implausible (frontal pole,
one lone electrode), that's a red flag for an artifact or a
channel-labeling bug, not a novel finding.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mne
import numpy as np

from config import OUTPUT_DIR, P300_SEARCH_WINDOW
from erp_analysis import find_peak, grand_average


def make_evoked(mean_waveform: np.ndarray, info: mne.Info, tmin: float) -> mne.EvokedArray:
    return mne.EvokedArray(mean_waveform, info, tmin=tmin)


def plot_target_topomap(
    X: np.ndarray, y: np.ndarray, times: np.ndarray, info: mne.Info,
    target_label, dataset_name: str, save_path: str | None = None,
):
    """Topomap at the grand-average target-ERP peak latency."""
    t_mean, _, n_t = grand_average(X, y, target_label, times)

    # Use the channel-averaged peak latency as a single representative
    # timepoint (rather than each channel's own peak, which would mix
    # different times into one map and be uninterpretable).
    channel_avg_waveform = t_mean.mean(axis=0)
    peak_t, _ = find_peak(channel_avg_waveform, times)

    evoked = make_evoked(t_mean, info, tmin=times[0])
    fig = evoked.plot_topomap(times=[peak_t], show=False, colorbar=True, size=4)
    fig.suptitle(f"Target-ERP topomap at {peak_t*1000:.0f}ms -- {dataset_name} (n={n_t})", fontsize=10)
    fig.subplots_adjust(top=0.82)

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path, peak_t
    return fig, peak_t


def plot_difference_topomap(
    X: np.ndarray, y: np.ndarray, times: np.ndarray, info: mne.Info,
    target_label, nontarget_label, peak_t: float, dataset_name: str, save_path: str | None = None,
):
    """
    Target-minus-NonTarget difference topomap at the target peak latency.
    This isolates the P300-specific effect from any shared visual-evoked
    response common to both target and non-target flashes.
    """
    t_mean, _, _ = grand_average(X, y, target_label, times)
    nt_mean, _, _ = grand_average(X, y, nontarget_label, times)
    diff = t_mean - nt_mean

    evoked = make_evoked(diff, info, tmin=times[0])
    fig = evoked.plot_topomap(times=[peak_t], show=False, colorbar=True, size=4)
    fig.suptitle(f"Target-minus-NonTarget difference topomap at {peak_t*1000:.0f}ms -- {dataset_name}", fontsize=10)
    fig.subplots_adjust(top=0.82)

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def run(X: np.ndarray, y: np.ndarray, times: np.ndarray, info: mne.Info,
        target_label, nontarget_label, dataset_name: str, out_dir: str = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    target_path = os.path.join(out_dir, f"topomap_target_{dataset_name}.png")
    _, peak_t = plot_target_topomap(X, y, times, info, target_label, dataset_name, save_path=target_path)
    print(f"Saved: {target_path} (peak at {peak_t*1000:.0f}ms)")

    diff_path = os.path.join(out_dir, f"topomap_difference_{dataset_name}.png")
    plot_difference_topomap(X, y, times, info, target_label, nontarget_label, peak_t, dataset_name, save_path=diff_path)
    print(f"Saved: {diff_path}")

    return {"peak_time": peak_t, "target_topomap": target_path, "difference_topomap": diff_path}


if __name__ == "__main__":
    from data_loader import load_arrays, load_epochs

    X, y, metadata = load_arrays("BNCI2014_008", subjects=[1, 2])
    epochs = load_epochs("BNCI2014_008", subjects=[1, 2])
    times = np.linspace(0, X.shape[2] / 256.0, X.shape[2])
    run(X, y, times, epochs.info, target_label="Target", nontarget_label="NonTarget", dataset_name="BNCI2014_008")
