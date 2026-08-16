"""
Data loading for the EDA suite.

Builds on your Day 2 setup script: same paradigm.get_data() call, but this
version also fetches MNE Epochs objects (return_epochs=True) since several
EDA modules (topography.py, erp_analysis.py) need MNE's Evoked/topomap
machinery, not just raw arrays.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_008, BNCI2014_009
from moabb.paradigms import P300

from config import DATASET_CONFIG

DATASETS = {
    "BNCI2014_008": BNCI2014_008,
    "BNCI2014_009": BNCI2014_009,
}


def load_arrays(name: str, subjects: list[int] | None = None):
    """
    Array-based load (same as Day 2). Returns X, y, metadata.

    Unit note: MOABB's array-return path (return_epochs=False) deliberately
    multiplies by dataset.unit_factor (1e6 for BNCI2014-008/009), returning
    microvolts -- unlike the Epochs-return path, which stays in volts (MNE's
    native convention). Every downstream file in this EDA suite assumes
    volts (matching PTP_REJECT_THRESHOLD_V in config.py, and the *1e6
    uV-conversion in erp_analysis.py/signal_quality.py's plotting code), so
    we divide back out by unit_factor here to keep load_arrays() and
    load_epochs() consistent with each other. Confirmed via
    moabb/paradigms/base.py's _get_array_pipeline -- not guessed.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    dataset = DATASETS[name]()
    paradigm = P300()
    subjects = subjects or dataset.subject_list
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X = X / dataset.unit_factor  # undo MOABB's built-in V -> uV scaling
    return X, y, metadata


def load_epochs(name: str, subjects: list[int] | None = None) -> mne.Epochs:
    """
    MNE-Epochs-based load, with a standard 10-20 montage attached so
    downstream topomap plotting works out of the box.

    Note: MOABB assigns generic channel positions from the recording;
    we override with mne's standard_1020 montage matched by channel name,
    which is a reasonable approximation for scalp-topography visualization
    (it will NOT be picked up automatically -- MOABB Epochs objects don't
    ship a montage by default, hence this explicit step).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    dataset = DATASETS[name]()
    paradigm = P300()
    subjects = subjects or dataset.subject_list
    epochs, y, metadata = paradigm.get_data(
        dataset=dataset, subjects=subjects, return_epochs=True
    )

    expected_channels = DATASET_CONFIG.get(name, {}).get("channels")
    if expected_channels is not None:
        actual = epochs.info["ch_names"]
        if list(actual) != expected_channels:
            print(
                f"[warn] channel order/names differ from config.py expectation.\n"
                f"  expected: {expected_channels}\n"
                f"  actual:   {actual}\n"
                f"  Proceeding with actual names -- update config.py if this is "
                f"the new ground truth."
            )

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing="warn")

    epochs.metadata = epochs.metadata.assign(label=y)

    return epochs


if __name__ == "__main__":
    # Smoke test: confirms montage attaches and shapes look right, without
    # requiring you to run the full EDA suite.
    epochs = load_epochs("BNCI2014_008", subjects=[1, 2])
    print(epochs)
    print("montage channels with positions:", len(epochs.get_montage().ch_names))