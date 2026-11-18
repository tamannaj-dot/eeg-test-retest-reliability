"""Inventory check for the ds004148 EEG download.

Usage:  python3 check_dataset.py /path/to/ds004148_folder

Looks only at folders named sub-01 ... sub-60 (two digits), so stray folders
like sub-001 or 'derivatives' are ignored. Reports, for each participant,
which sessions and recordings (tasks) exist, whether each BrainVision
recording has all three files (.vhdr, .vmrk, .eeg), and file sizes.
Writes inventory.csv next to the script. Uses only the standard library.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("Usage: python3 check_dataset.py /path/to/dataset_folder")

root = Path(sys.argv[1]).expanduser()
if not root.is_dir():
    sys.exit(f"Folder not found: {root}")

SUB_RE = re.compile(r"^sub-(\d{2})$")
subs = sorted(p for p in root.iterdir() if p.is_dir() and SUB_RE.match(p.name))
print(f"Found {len(subs)} participant folders named sub-01..sub-60\n")

rows = []                       # one row per recording
tasks_seen = defaultdict(int)   # task name -> count of recordings
problems = []

for sub in subs:
    sessions = sorted(d for d in sub.iterdir() if d.is_dir() and d.name.startswith("ses-"))
    if len(sessions) != 3:
        problems.append(f"{sub.name}: has {len(sessions)} session folders (expected 3)")
    for ses in sessions:
        vhdrs = sorted(ses.rglob("*.vhdr"))
        if not vhdrs:
            problems.append(f"{sub.name}/{ses.name}: no .vhdr recordings found")
        for vhdr in vhdrs:
            m = re.search(r"task-([A-Za-z0-9]+)", vhdr.name)
            task = m.group(1) if m else "UNKNOWN"
            tasks_seen[task] += 1
            eeg, vmrk = vhdr.with_suffix(".eeg"), vhdr.with_suffix(".vmrk")
            ok = eeg.exists() and vmrk.exists()
            size_mb = eeg.stat().st_size / 1e6 if eeg.exists() else 0.0
            if not ok:
                problems.append(f"{vhdr.name}: missing .eeg or .vmrk")
            elif size_mb < 1:
                problems.append(f"{vhdr.name}: .eeg is only {size_mb:.2f} MB (partial download?)")
            rows.append({"subject": sub.name, "session": ses.name, "task": task,
                         "complete": ok, "eeg_mb": round(size_mb, 1), "vhdr": str(vhdr)})

print("Recordings found per task name:")
for t, n in sorted(tasks_seen.items()):
    print(f"  {t:20s} {n}")

total_mb = sum(r["eeg_mb"] for r in rows)
resting = [r for r in rows if re.search(r"open|closed", r["task"], re.I)]
print(f"\nTotal .eeg size on disk: {total_mb/1000:.1f} GB")
print(f"Resting-state (open/closed) only: {sum(r['eeg_mb'] for r in resting)/1000:.1f} GB "
      f"across {len(resting)} recordings")

print("\nProblems to look at:" if problems else "\nNo problems found.")
for p in problems:
    print("  -", p)

out = Path(__file__).with_name("inventory.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subject", "session", "task", "complete", "eeg_mb", "vhdr"])
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {out}")
