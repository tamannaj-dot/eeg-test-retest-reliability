"""Reliability and validation analysis on features.csv.

Usage (project folder, virtual environment active):
    python analyze.py

Needs features.csv (from extract_features.py) and participants.tsv (from the dataset).
Writes results/*.csv and figures/*.png and prints a summary.

Parts:
  1. Sanity check: is eyes-closed alpha higher than eyes-open?
  2. Test-retest reliability (ICC) between session 1 and session 2 (90 min later)
     and between session 1 and session 3 (one month later).
  3. Smallest detectable change: how big a change must be to exceed normal
     session-to-session variation.
  4. Classification evaluated with and without leakage between people/sessions.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]
BANDS = ["theta", "alpha", "beta"]
PAIRS = {"90 min": ("session1", "session2"), "1 month": ("session1", "session3")}
N_BOOT = 1000

df = pd.read_csv(ROOT / "features.csv")
df["logp"] = np.log10(df["power"])          # log scale: band power is strongly skewed
print(f"Loaded {len(df)} rows, {df.subject.nunique()} participants\n")

# ---------------------------------------------------------------- 1. sanity check
occ = df[(df.region == "occipital") & (df.band == "alpha")]
wide = occ.pivot_table(index=["subject", "session"], columns="task", values="power")
ratio = (wide["eyesclosed"] / wide["eyesopen"]).dropna()
print("1. SANITY CHECK: occipital alpha, eyes closed / eyes open")
print(f"   median ratio {ratio.median():.2f}; higher with eyes closed in "
      f"{(ratio > 1).mean() * 100:.0f}% of {len(ratio)} participant-sessions\n")


# ---------------------------------------------------------------- 2-3. reliability
def icc2_1(X):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.
    X is participants x sessions."""
    n, k = X.shape
    gm = X.mean()
    ms_rows = k * ((X.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    ms_cols = n * ((X.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    resid = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + gm
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n)


rng = np.random.default_rng(0)
wide_all = df.pivot_table(index="subject", columns=["task", "region", "band", "session"], values="logp")
rows = []
for task in ("eyesclosed", "eyesopen"):
    for region in REGIONS:
        for band in BANDS:
            for label, (a, b) in PAIRS.items():
                X = wide_all[(task, region, band)][[a, b]].dropna().to_numpy()
                n = len(X)
                icc = icc2_1(X)
                boots = [icc2_1(X[rng.integers(0, n, n)]) for _ in range(N_BOOT)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                sd = X.std(ddof=1)
                sem = sd * np.sqrt(max(0.0, 1 - icc))
                sdc = 1.96 * np.sqrt(2) * sem                      # in log10 units
                rows.append({"task": task, "region": region, "band": band, "gap": label, "n": n,
                             "icc": icc, "ci_low": lo, "ci_high": hi,
                             "sdc_log10": sdc, "sdc_pct": (10 ** sdc - 1) * 100})
rel = pd.DataFrame(rows)
rel.to_csv(ROOT / "results" / "reliability.csv", index=False)

print("2. TEST-RETEST RELIABILITY (ICC; <0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, >0.9 excellent)")
for task in ("eyesclosed", "eyesopen"):
    t = rel[rel.task == task].pivot_table(index=["region", "band"], columns="gap", values="icc")
    t = t.reindex(pd.MultiIndex.from_product([REGIONS, BANDS]))[["90 min", "1 month"]]
    print(f"\n   {task}:")
    print(t.round(2).to_string().replace("\n", "\n   "))
print("\n3. SMALLEST DETECTABLE CHANGE (occipital alpha, eyes closed)")
for _, r in rel[(rel.task == "eyesclosed") & (rel.region == "occipital") & (rel.band == "alpha")].iterrows():
    print(f"   {r.gap:8s} ICC {r.icc:.2f} (95% CI {r.ci_low:.2f} to {r.ci_high:.2f}); "
          f"a change must exceed about +{r.sdc_pct:.0f}% (or the matching decrease) to beat normal variation")

# figure 1: ICC heatmaps
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, (task, gap) in zip(axes.ravel(), [(t, g) for t in ("eyesclosed", "eyesopen") for g in PAIRS]):
    m = rel[(rel.task == task) & (rel.gap == gap)].pivot_table(index="region", columns="band", values="icc")
    m = m.loc[REGIONS, BANDS]
    im = ax.imshow(m.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(3), BANDS)
    ax.set_yticks(range(5), REGIONS)
    ax.set_title(f"{task.replace('eyesclosed', 'eyes closed').replace('eyesopen', 'eyes open')}, {gap} apart", fontsize=10)
    for i in range(5):
        for j in range(3):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.6 else "black")
fig.colorbar(im, ax=axes, shrink=0.8, label="ICC")
fig.savefig(ROOT / "figures" / "fig1_icc.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# figure 2: drift across sessions
d = df[(df.task == "eyesclosed") & (df.region == "occipital") & (df.band == "alpha")]
w = d.pivot_table(index="subject", columns="session", values="logp")
fig, ax = plt.subplots(figsize=(5.5, 4))
for _, r in w.iterrows():
    ax.plot(range(w.shape[1]), r.to_numpy(), color="gray", alpha=0.35, lw=1)
ax.plot(range(w.shape[1]), w.mean().to_numpy(), color="C0", lw=3, label="group mean")
ax.set_xticks(range(w.shape[1]), ["session 1", "session 2\n(+90 min)", "session 3\n(+1 month)"])
ax.set_ylabel("log10 occipital alpha power (eyes closed)")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig2_drift.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 4. classification
feat = df[df.region.isin(REGIONS)].pivot_table(index=["subject", "session", "task"],
                                               columns=["region", "band"], values="logp")
feat.columns = [f"{r}_{b}" for r, b in feat.columns]
feat = feat.dropna().reset_index()
cols = [c for c in feat.columns if c not in ("subject", "session", "task")]

part = pd.read_csv(ROOT / "participants.tsv", sep="\t")[["participant_id", "sex"]]
part["sex"] = part["sex"].astype(str).str.strip().str.lower()
feat = feat.merge(part, left_on="subject", right_on="participant_id", how="left")

MODELS = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "random forest": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
}


def evaluate(X, y, groups_subject, groups_session, schemes, model_name):
    out = []
    for scheme in schemes:
        if scheme == "random split (leaky)":
            scores = []
            for seed in range(5):
                cv = KFold(10, shuffle=True, random_state=seed)
                p = cross_val_predict(MODELS[model_name](), X, y, cv=cv)
                scores.append(balanced_accuracy_score(y, p))
            score = float(np.mean(scores))
        elif scheme == "leave participants out":
            p = cross_val_predict(MODELS[model_name](), X, y, cv=GroupKFold(10), groups=groups_subject)
            score = balanced_accuracy_score(y, p)
        else:  # leave one session out
            p = cross_val_predict(MODELS[model_name](), X, y, cv=LeaveOneGroupOut(), groups=groups_session)
            score = balanced_accuracy_score(y, p)
        out.append((scheme, score))
    return out


results = []
# Task A: eyes closed vs eyes open
XA, yA = feat[cols].to_numpy(), (feat.task == "eyesclosed").to_numpy()
for m in MODELS:
    for scheme, s in evaluate(XA, yA, feat.subject, feat.session,
                              ["random split (leaky)", "leave participants out", "leave one session out"], m):
        results.append({"task": "eyes closed vs open", "model": m, "validation": scheme, "balanced_accuracy": s})
# Task B: sex, from eyes-closed recordings (a participant-level trait: leakage-prone)
fb = feat[(feat.task == "eyesclosed") & feat.sex.isin(["m", "f"])]
XB, yB = fb[cols].to_numpy(), (fb.sex == "m").to_numpy()
for m in MODELS:
    for scheme, s in evaluate(XB, yB, fb.subject, fb.session,
                              ["random split (leaky)", "leave participants out"], m):
        results.append({"task": "sex (demo of leakage)", "model": m, "validation": scheme, "balanced_accuracy": s})
res = pd.DataFrame(results)
res.to_csv(ROOT / "results" / "classification.csv", index=False)

print("\n4. CLASSIFICATION (balanced accuracy; 0.50 = chance)")
print(res.pivot_table(index=["task", "model"], columns="validation", values="balanced_accuracy")
      .round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, (task, g) in zip(axes, res.groupby("task")):
    pv = g.pivot_table(index="validation", columns="model", values="balanced_accuracy")
    pv.plot.bar(ax=ax, rot=20)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_title(task, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.15)
    ax.legend(title="model", loc="upper center", ncol=2, fontsize=8, title_fontsize=8, frameon=False)
axes[0].set_ylabel("balanced accuracy")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig3_validation.png", dpi=150)
plt.close(fig)
print("\nSaved results/reliability.csv, results/classification.csv and figures/fig1-3.")
