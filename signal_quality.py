"""
Signal quality EDA: amplitude distributions, artifact-rejection impact,
and power spectral density (PSD).

Purpose: before you trust any ERP result, you need to know (a) how noisy
the raw signal is, (b) how many epochs a given rejection threshold would
throw out, and (c) whether the spectral content looks like EEG (1/f-ish,
alpha bump ~8-12Hz) rather than dominated by line noise or muscle artifact.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mne
import numpy as np

from config import OUTPUT_DIR, PTP_REJECT_THRESHOLD_V


def peak_to_peak_per_epoch(X: np.ndarray) -> np.ndarray:
    """
    Peak-to-peak amplitude per epoch, per channel.

    X : (n_epochs, n_channels, n_times)
    Returns : (n_epochs, n_channels) -- max(x) - min(x) over the time axis.
    """
    return X.max(axis=2) - X.min(axis=2)


def rejection_summary(X: np.ndarray, threshold_v: float = PTP_REJECT_THRESHOLD_V) -> dict:
    """
    How many epochs would a simple peak-to-peak threshold rejection drop?
    An epoch is flagged if ANY channel exceeds the threshold -- matches the
    common single-bad-channel-rejects-the-epoch convention for low channel
    counts (with only 8 channels, interpolating a bad channel is less
    defensible than with 64+, so whole-epoch rejection is the standard
    choice here).
    """
    ptp = peak_to_peak_per_epoch(X)  # (n_epochs, n_channels)
    bad_per_channel = ptp > threshold_v
    bad_epoch_mask = bad_per_channel.any(axis=1)
    return {
        "threshold_v": threshold_v,
        "n_epochs": X.shape[0],
        "n_rejected": int(bad_epoch_mask.sum()),
        "pct_rejected": 100 * bad_epoch_mask.mean(),
        "bad_epoch_mask": bad_epoch_mask,
        "worst_channel_rejection_rate": bad_per_channel.mean(axis=0),  # per channel
    }


def plot_ptp_distribution(X: np.ndarray, ch_names: list[str], threshold_v: float, save_path: str | None = None):
    ptp = peak_to_peak_per_epoch(X) * 1e6  # convert to uV for readability
    threshold_uv = threshold_v * 1e6

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ptp.flatten(), bins=80, color="#4c72b0", alpha=0.8)
    ax.axvline(threshold_uv, color="crimson", linestyle="--",
               label=f"reject threshold ({threshold_uv:.0f}uV)")
    ax.set_xlabel("Peak-to-peak amplitude (uV)")
    ax.set_ylabel("Count (epoch x channel)")
    ax.set_title("Peak-to-peak amplitude distribution, all channels pooled")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def compute_psd(epochs: mne.Epochs, fmin: float = 0.5, fmax: float = 40.0):
    """
    Welch PSD per channel, averaged across epochs.
    Useful sanity check: EEG PSD should roughly follow a 1/f-like decay,
    with a visible alpha bump (~8-12Hz) in occipital/parietal channels if
    subjects had any eyes-open/closed variance, and a spike at 50/60Hz if
    line-noise filtering wasn't applied or didn't fully work.
    """
    spectrum = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)
    return psds.mean(axis=0), freqs  # average over epochs -> (n_channels, n_freqs)


def plot_psd(psd_mean: np.ndarray, freqs: np.ndarray, ch_names: list[str], save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for ch_idx, ch_name in enumerate(ch_names):
        ax.semilogy(freqs, psd_mean[ch_idx], label=ch_name, alpha=0.8)
    for line_freq in (50, 60):
        if freqs.min() <= line_freq <= freqs.max():
            ax.axvline(line_freq, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (log scale)")
    ax.set_title("Power spectral density by channel")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def run(X: np.ndarray, epochs: mne.Epochs, ch_names: list[str], dataset_name: str, out_dir: str = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    rej = rejection_summary(X)
    print(f"\n--- Signal quality: {dataset_name} ---")
    print(f"Threshold: {rej['threshold_v']*1e6:.0f}uV peak-to-peak")
    print(f"Would reject {rej['n_rejected']}/{rej['n_epochs']} epochs "
          f"({rej['pct_rejected']:.1f}%)")
    print("Per-channel rejection contribution:")
    for ch, rate in zip(ch_names, rej["worst_channel_rejection_rate"]):
        print(f"  {ch:6s} {100*rate:5.1f}%")

    ptp_path = os.path.join(out_dir, f"ptp_distribution_{dataset_name}.png")
    plot_ptp_distribution(X, ch_names, rej["threshold_v"], save_path=ptp_path)
    print(f"Saved: {ptp_path}")

    psd_mean, freqs = compute_psd(epochs)
    psd_path = os.path.join(out_dir, f"psd_{dataset_name}.png")
    plot_psd(psd_mean, freqs, ch_names, save_path=psd_path)
    print(f"Saved: {psd_path}")

    return {"rejection": rej, "psd_mean": psd_mean, "freqs": freqs}


if __name__ == "__main__":
    from data_loader import load_arrays, load_epochs

    X, y, metadata = load_arrays("BNCI2014_008", subjects=[1, 2])
    epochs = load_epochs("BNCI2014_008", subjects=[1, 2])
    run(X, epochs, epochs.info["ch_names"], "BNCI2014_008")
