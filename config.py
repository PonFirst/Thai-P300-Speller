"""
Shared config for the P300 EDA suite.

Values here (channel names/order, SOA/ISI) are sourced from the dataset
description in Korczowski et al. 2022 ("End-to-end P300 BCI using Bayesian
accumulation of Riemannian probabilities") and the original BNCI 2014-008
release notes -- not guessed. If you self-collect Thai speller data later,
this is the file to fork/edit for that hardware's channel layout.
"""

from __future__ import annotations

# --- BNCI2014_008 -----------------------------------------------------------
# 8-channel g.MOBIlab amplifier, active electrodes, right-ear-lobe reference,
# left-mastoid ground. Matches your 8-channel Thai speller hardware in
# electrode *count* (not necessarily placement -- check your own montage).
BNCI2014_008_CHANNELS = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]
BNCI2014_008_SFREQ = 256.0  # Hz
BNCI2014_008_SOA_MS = 250  # stimulus onset asynchrony (125ms flash + 125ms ISI)

# --- BNCI2014_009 ------------------------------------------------------------
BNCI2014_009_CHANNELS = [
    "Fz", "FCz", "Cz", "CPz", "Pz", "Oz",
    "F3", "F4", "C3", "C4", "CP3", "CP4", "P3", "P4", "PO7", "PO8",
]
BNCI2014_009_SFREQ = 256.0  # Hz

DATASET_CONFIG = {
    "BNCI2014_008": {
        "channels": BNCI2014_008_CHANNELS,
        "sfreq": BNCI2014_008_SFREQ,
        "soa_ms": BNCI2014_008_SOA_MS,
    },
    "BNCI2014_009": {
        "channels": BNCI2014_009_CHANNELS,
        "sfreq": BNCI2014_009_SFREQ,
        "soa_ms": None,  # fill in if you pull this dataset -- not verified here
    },
}

# P300 window conventions used throughout the EDA (seconds, relative to
# stimulus onset). These bound where we *expect* to find the P300 peak --
# used for peak search windows and topomap timepoints, not for filtering.
P300_SEARCH_WINDOW = (0.25, 0.55)

# Artifact rejection threshold (peak-to-peak, volts). Matches the ±75-100uV
# convention widely used in P300 literature (e.g. Krusienski et al., and the
# feedback-latency BCI paper's ±75uV / 150uV-step criterion). Start here;
# revisit once you've looked at your own amplitude distributions in
# signal_quality.py -- this is a starting assumption, not a fixed truth.
PTP_REJECT_THRESHOLD_V = 100e-6

# Baseline window for baseline-correction sanity checks (seconds).
BASELINE_WINDOW = (-0.2, 0.0)

# Output directory for all EDA figures.
OUTPUT_DIR = "eda_outputs"
