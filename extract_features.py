"""Extract resting-state band power for every participant / session / condition.

Usage (from the project folder, with the virtual environment active):
    python extract_features.py 3        # test run: first 3 participants only
    python extract_features.py          # everything (sub-01 ... sub-60)

For each eyes-open / eyes-closed recording:
  1. band-pass filter 1-40 Hz (this also removes 50 Hz mains noise)
  2. Welch power spectrum with 4-second windows
  3. mean power in theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz)
  4. averaged over scalp regions: frontal, central, temporal, parietal, occipital, plus 'all'

Writes features.csv in long format (one row per recording x region x band).
No artifact rejection beyond the filter -- say so in the README.
"""
import re
import sys
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parent
TASKS = ("eyesclosed", "eyesopen")
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
REGION_BY_PREFIX = {
    "FP": "frontal", "AF": "frontal", "F": "frontal",
    "FC": "central", "C": "central", "CP": "central",
    "FT": "temporal", "T": "temporal", "TP": "temporal",
    "P": "parietal", "PO": "parietal",
    "O": "occipital",
}
NAME_RE = re.compile(r"^([A-Za-z]+?)(z|\d+)?$")


def region_of(ch_name):
    m = NAME_RE.match(ch_name)
    return REGION_BY_PREFIX.get(m.group(1).upper()) if m else None


def features_for(vhdr):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    raw.filter(1.0, 40.0, fir_design="firwin")
    sf = raw.info["sfreq"]
    psd = raw.compute_psd(method="welch", fmin=1, fmax=40,
                          n_fft=int(4 * sf), n_overlap=int(2 * sf))
    data, freqs = psd.get_data(return_freqs=True)        # channels x freqs

    groups = {"all": list(range(len(raw.ch_names)))}
    unmatched = []
    for i, ch in enumerate(raw.ch_names):
        reg = region_of(ch)
        if reg is None:
            unmatched.append(ch)
        else:
            groups.setdefault(reg, []).append(i)

    rows = []
    for reg, idx in groups.items():
        spec = data[idx].mean(axis=0)
        total = spec.sum()                      # total power, 1-40 Hz
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            rows.append({"region": reg, "band": band,
                         "power": spec[mask].mean(),        # mean power density in band
                         "rel_power": spec[mask].sum() / total,   # share of 1-40 Hz power (0 to 1)
                         "n_channels": len(idx)})
    return rows, unmatched


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    subs = sorted(p for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"sub-\d{2}", p.name))
    if limit:
        subs = subs[:limit]

    all_rows, failures, unmatched_seen = [], [], set()
    t0 = time.time()
    for n, sub in enumerate(subs, 1):
        for ses_dir in sorted(sub.glob("ses-*")):
            for task in TASKS:
                files = sorted((ses_dir / "eeg").glob(f"*task-{task}_eeg.vhdr"))
                if len(files) != 1:
                    failures.append(f"{sub.name}/{ses_dir.name}/{task}: found {len(files)} files")
                    continue
                try:
                    rows, unmatched = features_for(files[0])
                except Exception as e:  # keep going; report at the end
                    failures.append(f"{sub.name}/{ses_dir.name}/{task}: {type(e).__name__}: {e}")
                    continue
                unmatched_seen.update(unmatched)
                for r in rows:
                    r.update(subject=sub.name, session=ses_dir.name.replace("ses-", ""), task=task)
                all_rows.extend(rows)
        print(f"[{n}/{len(subs)}] {sub.name} done ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    df = pd.DataFrame(all_rows)[["subject", "session", "task", "region", "band",
                                 "power", "rel_power", "n_channels"]]
    out = ROOT / "features.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out.name}: {len(df)} rows from {df[['subject', 'session', 'task']].drop_duplicates().shape[0]} recordings")
    if unmatched_seen:
        print(f"Channels not assigned to a region (only in the 'all' average): {sorted(unmatched_seen)}")
    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print("  -", f)


if __name__ == "__main__":
    main()
