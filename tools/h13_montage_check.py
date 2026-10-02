"""TESTING.md H13 readout: is channel order Fz C3 Cz C4 Pz PO7 Oz PO8 physically right?

Record 60 s wearing the headset: blink hard for the first 20 s, eyes closed for the
next 20 s, clench your jaw for the last 20 s. Then:
    $PY tools/h13_montage_check.py /tmp/h13.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.constants import EEG_CHANNELS, EEG_SFREQ  # noqa: E402


df = pd.read_csv(sys.argv[1]); df = df[pd.to_numeric(df["Fz"], errors="coerce").notna()]
ch = list(EEG_CHANNELS); fs = int(EEG_SFREQ)
x = df[ch].to_numpy(float); n = len(x) // 3
blink, closed, jaw = x[:n], x[n:2*n], x[2*n:3*n]
def rng(seg, lo, hi):
    y = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), seg, axis=0)
    return np.percentile(y, 99, axis=0) - np.percentile(y, 1, axis=0)
def band(seg, lo, hi):
    f, p = welch(seg, fs=fs, nperseg=fs*2, axis=0); return p[(f>=lo)&(f<=hi)].mean(axis=0)
b = rng(blink, 1, 10)
a = band(closed, 8, 12) / band(blink, 8, 12)
j = band(jaw, 20, 100) / band(closed, 20, 100)
print(f"{'':6}{'blink uV':>10}{'alpha x':>10}{'jaw x':>10}")
for i, c in enumerate(ch): print(f"{c:6}{b[i]:10.1f}{a[i]:10.2f}{j[i]:10.2f}")
print("\nexpect: biggest blink = Fz | biggest alpha = Oz (or PO7/PO8) | jaw strongest at C3/C4")
print("largest:", ch[b.argmax()], "|", ch[a.argmax()], "|", ch[j.argmax()])
