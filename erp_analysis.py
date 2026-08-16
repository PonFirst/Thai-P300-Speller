"""
Core ERP EDA: grand-average target vs. non-target waveforms, P300 peak
detection, and single-trial vs. ensemble-average SNR.

This is the module that answers the central question of Day 3: "is there
actually a visible, plausible P300 in this data, and does it match what
Pananookooln et al. and the wider P300 literature report?"
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from config import BASELINE_WINDOW, OUTPUT_DIR, P300_SEARCH_WINDOW


def grand_average(X: np.ndarray, y: np.ndarray, target_label, times: np.ndarray):
    """
    Grand-average waveform for one class, all channels.

    X : (n_epochs, n_channels, n_times)
    y : (n_epochs,) labels (string or int)
    Returns : (n_channels, n_times) mean waveform, and (n_channels, n_times)
              standard error of the mean (for confidence bands).
    """
    mask = y == target_label
    n = mask.sum()
    if n == 0:
        raise ValueError(f"No epochs found with label {target_label!r}")
    subset = X[mask]  # (n, n_channels, n_times)
    mean = subset.mean(axis=0)
    sem = subset.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, sem, n


def find_peak(waveform: np.ndarray, times: np.ndarray, window: tuple[float, float] = P300_SEARCH_WINDOW):
    """
    Find the maximum-amplitude peak within a search window (defaults to the
    canonical 250-550ms P300 window). Returns (peak_time_s, peak_amplitude).

    Note: this looks for the *maximum positive* deflection in-window, which
    assumes non-inverted polarity. If your peak comes back negative or at
    the window edge, that is itself informative -- don't silently trust it.
    """
    in_window = (times >= window[0]) & (times <= window[1])
    if not in_window.any():
        raise ValueError(f"Window {window} does not overlap with epoch times {times.min()}-{times.max()}")
    idx_in_window = np.where(in_window)[0]
    local_idx = np.argmax(waveform[idx_in_window])
    global_idx = idx_in_window[local_idx]
    return times[global_idx], waveform[global_idx]


def plot_channel_erps(
    X: np.ndarray, y: np.ndarray, times: np.ndarray, ch_names: list[str],
    target_label, nontarget_label, dataset_name: str, save_path: str | None = None,
):
    """
    Grid of target-vs-nontarget grand-average ERP waveforms, one subplot per
    channel, with SEM confidence bands and detected P300 peak marked.
    """
    n_channels = len(ch_names)
    ncols = 4
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    t_mean, t_sem, n_t = grand_average(X, y, target_label, times)
    nt_mean, nt_sem, n_nt = grand_average(X, y, nontarget_label, times)

    for i, ch in enumerate(ch_names):
        ax = axes[i]
        ax.plot(times, t_mean[i] * 1e6, color="crimson", label=f"Target (n={n_t})")
        ax.fill_between(times, (t_mean[i]-t_sem[i])*1e6, (t_mean[i]+t_sem[i])*1e6,
                          color="crimson", alpha=0.2)
        ax.plot(times, nt_mean[i] * 1e6, color="steelblue", label=f"NonTarget (n={n_nt})")
        ax.fill_between(times, (nt_mean[i]-nt_sem[i])*1e6, (nt_mean[i]+nt_sem[i])*1e6,
                          color="steelblue", alpha=0.2)

        peak_t, peak_amp = find_peak(t_mean[i], times)
        ax.axvline(peak_t, color="crimson", linestyle=":", alpha=0.6)
        ax.scatter([peak_t], [peak_amp * 1e6], color="crimson", zorder=5, s=25)
        ax.annotate(f"{peak_t*1000:.0f}ms", (peak_t, peak_amp * 1e6),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_title(ch)
        if i == 0:
            ax.legend(fontsize=7)

    for ax in axes[n_channels:]:
        ax.axis("off")

    fig.supxlabel("Time (s)")
    fig.supylabel("Amplitude (uV)")
    fig.suptitle(f"Grand-average ERPs -- {dataset_name}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def single_trial_vs_ensemble_snr(
    X: np.ndarray, y: np.ndarray, times: np.ndarray, target_label,
    channel_idx: int, n_repeats_list: list[int] = (1, 2, 5, 10, 15),
    n_bootstrap: int = 200, seed: int = 0,
):
    """
    Empirically demonstrate WHY P300 spellers repeat flash sequences and
    ensemble-average: estimate how peak amplitude SNR improves as you
    average more target trials together.

    Method: for each n_repeats, bootstrap-sample n_repeats target epochs,
    average them, measure peak-window amplitude; repeat n_bootstrap times
    and report the mean and std of that estimate. SNR is approximated as
    peak-window mean amplitude / across-bootstrap std -- i.e. how
    consistently you land on a similar (hopefully P300-shaped) estimate as
    you add more trials.
    """
    rng = np.random.default_rng(seed)
    mask = y == target_label
    target_epochs = X[mask][:, channel_idx, :]  # (n_target, n_times)
    n_target = target_epochs.shape[0]

    in_window = (times >= P300_SEARCH_WINDOW[0]) & (times <= P300_SEARCH_WINDOW[1])

    results = {}
    for n_rep in n_repeats_list:
        if n_rep > n_target:
            continue
        peak_amps = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(n_target, size=n_rep, replace=False)
            avg_waveform = target_epochs[idx].mean(axis=0)
            peak_amps[b] = avg_waveform[in_window].max()
        results[n_rep] = {
            "mean_peak_uv": peak_amps.mean() * 1e6,
            "std_peak_uv": peak_amps.std() * 1e6,
        }
    return results


def plot_snr_curve(snr_results: dict, ch_name: str, save_path: str | None = None):
    n_reps = sorted(snr_results.keys())
    means = [snr_results[n]["mean_peak_uv"] for n in n_reps]
    stds = [snr_results[n]["std_peak_uv"] for n in n_reps]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(n_reps, means, yerr=stds, marker="o", capsize=4, color="#4c72b0")
    ax.set_xlabel("Number of trials averaged")
    ax.set_ylabel("Peak-window amplitude (uV), mean +/- bootstrap std")
    ax.set_title(f"Ensemble-averaging effect on peak estimate stability -- {ch_name}\n"
                  f"(this is WHY spellers use multiple repetitions per character)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return fig


def run(X: np.ndarray, y: np.ndarray, times: np.ndarray, ch_names: list[str],
        target_label, nontarget_label, dataset_name: str, out_dir: str = OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    erp_path = os.path.join(out_dir, f"grand_average_erps_{dataset_name}.png")
    plot_channel_erps(X, y, times, ch_names, target_label, nontarget_label, dataset_name, save_path=erp_path)
    print(f"Saved: {erp_path}")

    t_mean, _, n_t = grand_average(X, y, target_label, times)
    print(f"\n--- ERP peak summary: {dataset_name} (n_target={n_t}) ---")
    for i, ch in enumerate(ch_names):
        peak_t, peak_amp = find_peak(t_mean[i], times)
        print(f"  {ch:6s}  peak={peak_amp*1e6:6.2f}uV at {peak_t*1000:.0f}ms")

    # Pick the channel with the largest target-peak amplitude for the SNR
    # demo -- typically Pz/Cz for P300, but let the data decide rather than
    # hardcoding, in case your Thai-script speller shows a different
    # topography later.
    peak_amps_by_ch = [find_peak(t_mean[i], times)[1] for i in range(len(ch_names))]
    best_ch_idx = int(np.argmax(peak_amps_by_ch))
    best_ch = ch_names[best_ch_idx]
    print(f"\nUsing {best_ch} (largest peak) for single-trial-vs-ensemble SNR demo...")

    snr_results = single_trial_vs_ensemble_snr(X, y, times, target_label, best_ch_idx)
    for n_rep, stats in snr_results.items():
        print(f"  n={n_rep:3d} repeats: {stats['mean_peak_uv']:6.2f} +/- {stats['std_peak_uv']:5.2f} uV")

    snr_path = os.path.join(out_dir, f"snr_curve_{dataset_name}.png")
    plot_snr_curve(snr_results, best_ch, save_path=snr_path)
    print(f"Saved: {snr_path}")

    return {"peak_summary": dict(zip(ch_names, peak_amps_by_ch)), "snr_results": snr_results, "best_channel": best_ch}


if __name__ == "__main__":
    from data_loader import load_arrays

    X, y, metadata = load_arrays("BNCI2014_008", subjects=[1, 2])
    times = np.linspace(0, X.shape[2] / 256.0, X.shape[2])
    ch_names = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]
    run(X, y, times, ch_names, target_label="Target", nontarget_label="NonTarget", dataset_name="BNCI2014_008")
