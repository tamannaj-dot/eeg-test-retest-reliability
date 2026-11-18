"""Look at one participant/session: do the files load, and does eyes-closed show more alpha?

Usage (from the project folder):
    python3 inspect_one.py                 # defaults to sub-01, session1
    python3 inspect_one.py sub-05 session2

Prints basic facts about the recordings and saves psd_<sub>_<ses>.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

sub = sys.argv[1] if len(sys.argv) > 1 else "sub-01"
ses = sys.argv[2] if len(sys.argv) > 2 else "session1"
root = Path(__file__).resolve().parent
eeg_dir = root / sub / f"ses-{ses}" / "eeg"
mne.set_log_level("WARNING")


def load(task):
    matches = sorted(eeg_dir.glob(f"*task-{task}_eeg.vhdr"))
    if len(matches) != 1:
        sys.exit(f"Expected one {task} file in {eeg_dir}, found {len(matches)}")
    return mne.io.read_raw_brainvision(matches[0], preload=True)


raws = {t: load(t) for t in ("eyesclosed", "eyesopen")}

r = raws["eyesclosed"]
print(f"Sampling rate: {r.info['sfreq']} Hz")
print(f"Channels: {len(r.ch_names)}")
print(f"First 10 channel names: {r.ch_names[:10]}")
print(f"Channel types: {sorted(set(r.get_channel_types()))}")
for t, raw in raws.items():
    print(f"{t}: {raw.times[-1]:.0f} s long, annotations: {len(raw.annotations)}")

# Power spectrum per condition, averaged over channels
alpha = (8, 13)
band_power = {}
fig, ax = plt.subplots(figsize=(7, 4))
for t, raw in raws.items():
    psd = raw.compute_psd(method="welch", fmin=1, fmax=40, n_fft=int(raw.info["sfreq"] * 4))
    data, freqs = psd.get_data(return_freqs=True)
    mean_psd = data.mean(axis=0)
    ax.semilogy(freqs, mean_psd, label=t)
    mask = (freqs >= alpha[0]) & (freqs <= alpha[1])
    band_power[t] = mean_psd[mask].mean()
ax.axvspan(*alpha, alpha=0.15)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (V^2/Hz)")
ax.set_title(f"{sub} {ses}: mean over channels")
ax.legend()
out = root / f"psd_{sub}_{ses}.png"
fig.tight_layout()
fig.savefig(out, dpi=150)

ratio = band_power["eyesclosed"] / band_power["eyesopen"]
print(f"\nAlpha (8-13 Hz) power, eyes closed / eyes open: {ratio:.2f}")
print("Expected: above 1 (usually 1.5 to 3+). Below 1 would be worth investigating.")
print(f"Saved {out.name}")
