"""
Cross-process constants: the one place stream identities, channel order,
sample rate, and signal-quality thresholds are defined.

collection/record_session.py re-exports EEG_CHANNELS and the SANITY_PTP_*
thresholds from here, so existing `from record_session import ...` imports
keep working.
"""

# Unicorn LSL channel index order. This is stream indexing, not display
# order -- the GUI keeps its own front-to-back DISPLAY_ORDER. A list, not a
# tuple, because existing code concatenates it with other lists.
#
# This is the Unicorn Hybrid Black's factory-default 8-channel cap, verified
# against g.tec's own Unicorn Suite Hybrid Black user manual
# (github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black-User-Manual), and
# cross-checked against a MathWorks third-party support-package listing and
# multiple published papers using this exact headset. It replaces an earlier
# assumption (Fz, F3, C4, Cz, Pz, P3, O2, O1) inherited from the Pananookooln
# et al. reference paper's own EEG system, which was never actually the same
# hardware and was never verified against this project's device -- see
# CLAUDE.md's hard constraints for the correction record. Four of the eight
# positions are genuinely different scalp locations, not just relabelled:
# C3 vs. the old F3, PO7 vs. P3, Oz vs. O2, PO8 vs. O1.
EEG_CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
EEG_SFREQ = 250.0
N_CHARS = 72

EEG_STREAM_TYPE = "EEG"
EEG_STREAM_NAME = "UnicornEEG"
MARKER_STREAM_TYPE = "Markers"
MARKER_STREAM_NAME = "P300Markers"

# Deliberately NOT type "Markers": the recorder and the inference service find
# the stimulus stream by type "Markers", and would latch onto this one instead.
INFERENCE_STREAM_TYPE = "P300Inference"
INFERENCE_STREAM_NAME = "P300Inference"
INFERENCE_FINGERPRINT_KEY = "manifest_fingerprint"

# One switch for recorder and inference service EEG inlets, so train and serve stay consistent.
EEG_INLET_DEJITTER = False

# Peak-to-peak plausibility band in microvolts. Chosen from the recorder's
# sanity check, not tuned on real Unicorn recordings.
SANITY_PTP_MIN_UV = 1.0
SANITY_PTP_MAX_UV = 200.0
RAILED_PTP_UV = 2 * SANITY_PTP_MAX_UV
