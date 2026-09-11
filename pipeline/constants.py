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
EEG_CHANNELS = ["Fz", "F3", "C4", "Cz", "Pz", "P3", "O2", "O1"]
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
