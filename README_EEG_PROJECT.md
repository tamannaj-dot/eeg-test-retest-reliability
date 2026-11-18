# EEG test-retest reliability and cross-session validation

How stable is resting-state EEG within a person across hours and weeks, and does the way you validate a classifier change what you conclude?

> Draft. Items marked **[CHECK]** need you to confirm against your own scripts or the dataset page before this goes public.

## Question

1. How reliable (ICC) are band-power measures between session 1 and session 2 (90 minutes later) and session 3 (about one month later)?
2. How large must a change be to exceed normal session-to-session variation (smallest detectable change)?
3. How much does validation scheme (leave-one-session-out, leave-participants-out, random split) change classifier performance?

## Data

- Wang, Y., Duan, W., Ding, L., Dong, D. & Lei, X. A test-retest resting and cognitive state EEG dataset. OpenNeuro, ds004148 v1.0.1 (2022), https://doi.org/10.18112/openneuro.ds004148.v1.0.1. Described in Wang, Y., Duan, W., Dong, D., Ding, L. & Lei, X. A test-retest resting, and cognitive state EEG dataset during multiple subject-driven states. *Scientific Data* 9, 566 (2022), https://doi.org/10.1038/s41597-022-01607-9.
- 60 participants, 3 sessions (session 2 within 90 minutes of session 1, session 3 about one month later), 5 EEG states.
- This analysis uses only the two resting states, eyes closed and eyes open.
- `features.csv` has 6,480 rows: 60 participants x 3 sessions x 2 states x 6 regions (the five below plus an average over all channels, "all") x 3 bands. Reliability and classification use the five regional averages only.
- License: see the dataset's `dataset_description.json` on OpenNeuro. **[CHECK: copy the license name here]**
- The raw data (about 33 GB) is not included here. Download it from the original source.

## Methods

- Preprocessing: each eyes-closed and eyes-open recording is band-pass filtered 1 to 40 Hz (MNE FIR filter, which also removes 50 Hz mains noise). No artifact rejection, no ICA, no bad-channel handling and no re-referencing is applied in this pipeline; the whole recording is used, with no epoching.
- Power: Welch power spectral density (MNE `compute_psd`) with 4-second windows and 2-second overlap, 1 to 40 Hz.
- Features: mean power density in theta (4 to 8 Hz), alpha (8 to 13 Hz) and beta (13 to 30 Hz), averaged across the channels in each region, then log10-transformed because band power is strongly skewed.
- Regions are assigned from channel-name prefixes: frontal (Fp, AF, F), central (FC, C, CP), temporal (FT, T, TP), parietal (P, PO), occipital (O). Channels that match no prefix are excluded from the regional averages and only enter "all". `extract_features.py` prints any such channels when it runs. `features.csv` also stores relative power, which the analysis does not use.
- Reliability: ICC(2,1) (two-way random effects, absolute agreement, single measurement) between session 1 and session 2, and between session 1 and session 3. 95% confidence intervals come from 1,000 bootstrap resamples of participants.
- Smallest detectable change (SDC): SDC = 1.96 x sqrt(2) x SEM, where SEM = SD x sqrt(1 - ICC) and SD is the standard deviation of the log10 values pooled across the two sessions. It is computed in log10 units and converted to a percentage change with (10^SDC - 1) x 100. Reported here for occipital alpha, eyes closed; `results/reliability.csv` has it for every measure.
- Classification: all 15 region-band log-power values as features. Logistic regression (standardized features) and random forest (300 trees), scored with balanced accuracy under three schemes: leave-one-session-out; leave-participants-out (10-fold, each person entirely in train or test); and a random 10-fold split (leaky, since one person's recordings appear in both train and test), averaged over 5 random seeds.
- Tasks: eyes closed vs eyes open, and participant sex (from `participants.tsv`) as a deliberately leakage-prone demonstration. The sex task uses eyes-closed recordings only and has no leave-one-session-out scheme, since sex is fixed per person.

## Results

**Sanity check.** Occipital alpha was higher with eyes closed in 99% of 180 participant-sessions (median ratio 4.12), as expected for real EEG.

**Reliability.**

| Measure | 90 min ICC | 1 month ICC |
|---|---|---|
| Occipital alpha, eyes closed | 0.95 | 0.85 |
| Central alpha, eyes closed | 0.96 | 0.86 |
| Parietal alpha, eyes closed | 0.96 | 0.86 |
| Frontal theta, eyes closed | 0.88 | 0.31 |
| Frontal beta, eyes closed | 0.76 | 0.14 |
| Temporal beta, eyes closed | 0.71 | 0.27 |

- Over 90 minutes, eyes-closed alpha was excellent in every region (0.90 to 0.96); eyes-open alpha was lower and more variable (0.75 to 0.88).
- Over one month, posterior and central measures stayed good, while frontal and beta measures dropped sharply.
- Full table: `results/reliability.csv`. Figures: `figures/fig1_icc.png`, `fig2_drift.png`, `fig3_validation.png`.

**Smallest detectable change (occipital alpha, eyes closed).** Over 90 minutes, a change must exceed about +71% (or −41%) to exceed normal variation. Over one month, about +141% (or −58%). A high ICC does not mean small variability: ICC compares within-person variation to between-person differences, and alpha differs enormously between people.

**Classification (balanced accuracy).**

| Task / model | Leave-one-session-out | Leave-participants-out | Random split (leaky) |
|---|---|---|---|
| Eyes closed vs open, logistic regression | 0.91 | 0.90 | 0.90 |
| Eyes closed vs open, random forest | 0.88 | 0.85 | 0.87 |
| Sex, logistic regression | n/a | 0.79 | 0.81 |
| Sex, random forest | n/a | 0.77 | 0.86 |

- Eyes closed vs open is so consistent across people that leakage barely matters.
- For sex, the random forest scored 9 points higher under the leaky split than the honest one; the logistic regression barely changed. This is suggestive from a single dataset and was not tested statistically.

## Limitations

- Minimal cleaning (band-pass filter only; no artifact rejection or ICA). Frontal and beta measures are the most vulnerable to blink and muscle artifact, which may explain their poor one-month reliability. This is a hypothesis, not tested.
- Some eyes-open measures were less reliable at 90 minutes than at one month (e.g. temporal theta 0.52 vs 0.77; occipital theta 0.62 vs 0.79). Unexplained. **[CHECK: update after the follow-up checks]**
- 60 healthy participants gives wide confidence intervals; ICCs with wide intervals should not be over-interpreted.
- One dataset, one feature type (band power). No multiple-comparison correction.

## Relevance to other modalities

The numbers here do not transfer to other recording methods such as functional ultrasound. The approach does: quantify test-retest reliability and the smallest detectable change before interpreting any intervention effect, and validate across sessions and across people rather than within them.

## Reproduce

```
pip install numpy pandas matplotlib scikit-learn mne
```

1. Download ds004148 from OpenNeuro. Copy `extract_features.py` into the dataset folder (next to the `sub-01 ... sub-60` folders) and run `python extract_features.py` to build `features.csv`. `python extract_features.py 3` is a quick test on the first 3 participants. This step is slow and needs the full dataset (about 33 GB).
2. To skip step 1, use the `features.csv` included here. Put `participants.tsv` from the dataset next to `analyze.py`, then run `python analyze.py`. It writes `results/` and `figures/`.

## Citation

If you use this repo, please cite the dataset (OpenNeuro ds004148) and the Scientific Data paper listed under Data, and follow the dataset's license terms.
