# Are Fairness Scores in Crime Prediction Telling Us the Truth?

> **STATUS — read `STATUS.md` before using any number from this repo.**
> The modelling results were **retracted on 2026-09-07**. The measurement
> critique is the paper. The physics model does **not** work.

Reported fairness gaps in crime-prediction models are largely explained by
**how often crime happens**, not by how well a model serves each neighbourhood.
Correcting for that is **not enough**, because a model can close a corrected gap
by moving a threshold while getting genuinely worse — and because **pooling a
metric across regions lets a model score by ranking regions instead of
forecasting weeks.**

We found the second failure in **our own headline result**, and retracted it.

---

## The findings that survive

**1. The standard fairness metric moves without the model improving.**
On Chicago, an intervention raised Tail-region F1 by **+16.95** and closed our
base-rate-corrected gap from 40.97 to −2.39 — while Tail AUC *fell* 4.01 points
and positive-prediction rate went 23% → 80%. Moving a **single decision
threshold** on a model with no graph exceeded that whole "improvement"
(+13.63 vs +12.87), with AUC provably unchanged at 61.99 on all 19 rows.

**2. Published fairness gaps track crime volume.**
Across six published models, reported scores correlate with crime volume at
**r = 0.87–0.90**, Fisher combined **p = 1.3 × 10⁻⁵**, holding with no
distributional assumption (rank-only 0.875). ST-HSL is non-significant and acts
as an internal negative control. FedCrime's own tables: **r = 0.998**.

**2b. It holds in five cities, and the size of it is predictable.**
No model trained — a scorer built to have *identical* skill in every group, so
the honest gap is 0.00 everywhere:

| city | window | base gap | F1 gap | macro AUC gap |
|---|---|---|---|---|
| Chicago | 2015–19 | +55.1% | +36.09 | +1.74 |
| Los Angeles | 2020–24 | +64.1% | +39.40 | +0.15 |
| New York | 2015–19 | +71.9% | +48.70 | −0.66 |
| Seattle | 2015–19 | +51.5% | +43.64 | −0.26 |
| Cincinnati | 2015–19 | +32.7% | +39.26 | +1.36 |
| **mean** | | +55.1% | **+41.42 ± 4.87** | **+0.46 ± 1.04** |

Macro AUC sits at **76.21 ± 0.66** across all 15 city-group cells. And
`A_F1(p)`, computed from the base rates alone, **predicts** the observed F1 gap
at **r = +0.94** (exact permutation p = 0.025), observed/predicted
**0.786 ± 0.052**. The artifact isn't just present — its magnitude follows a
closed-form function of crime volume.

Cities use their own year windows because portals don't publish the same period;
that is sound here because the audit compares Head vs Tail *within* a city.

**3. It changes who gets help.**
Model *selection* is unaffected (0/5 groups change). Remediation *targeting*
flips for **6/6 models**.

**4. Pooling inflates AUC — this one caught us.**
A scorer that knows only *which region it is looking at*, with zero forecasting
skill, gets **pooled AUC 66.67 / macro AUC 50.45**. Our own "+11.11 Tail
advantage" and "+2.81 physics gain" were this artifact. Under within-region
(macro) AUC the physics gain is **+0.16 ± 2.76, winning 4 of 8 bins** — nothing.

**5. The fairness libraries have the same problem.**
On data built to have equal skill, Fairlearn's **default** `demographic_parity`
equalises the flag rate at ~24% everywhere and in doing so opens a
**−21.26 ± 0.52** Head-minus-Tail TPR gap — the *sparse* group ends up with the
higher true-positive rate (56.33 vs 35.50) **and** the higher false-alarm rate
(19.84 vs 7.12). In plain terms: the low-crime area is over-flagged while the
busy area starts missing real crime (TPR 48.00 → 35.50). `equalized_odds` and
`true_positive_rate_parity` behave correctly — the failure is specific to the
parity-of-selection family, which happens to be the default.

AIF360 `RejectOptionClassification` on statistical parity takes wrongful
flagging of sparse regions from **13.59% → 41.09%**. `CalibratedEqOdds(fnr)`
changes nothing at all. Per-group AUC is unmoved on every row — post-processing
re-thresholds, it does not re-rank.

**6. No neural model beat a 4-week moving average.**
On the base-rate-free target, in any region-density bin.

---

## What we tried, and what happened

Every method was tested against an **identical model with the new component
switched off** — same size, same seeds, same training.

| # | Method | Result |
|---|---|---|
| 1 | Reweighting | ❌ gap 41 → 52, worse |
| 2 | Group-DRO (τ = 3, 6, 10) | ❌ no change |
| 3 | Adaptive per-region thresholds | ❌ plateaued ~41 |
| 4 | Plain GCN, depths 1–4 | ❌ up to −57; collapses at depth ≥3 |
| 5 | Gated GCN, depths 1–4 | ❌ −0.5 to −4.5; learns to switch itself off |
| 6 | Attention GCN, depths 1–4 | ❌ −0.5 to −7.4 |
| 7 | Neural ODE, α swept | ❌ flat at control |
| 8 | ~~PINN (Short et al. PDE)~~ | ⚠️ **RETRACTED** — +18.3 AUC was on the confounded target; **+0.16 under macro AUC** |
| 9 | Learned thresholds, Arm A | ❌ *created* a 35.56-point TPR gap |
| 10 | Learned thresholds, Arm B | ✅ correct, but adds no skill — a measuring tool |

**22 of 24** graph comparisons negative, 9 significantly so. Not one positive.
**One of ten** interventions appeared to add skill, and it did not survive
honest evaluation.

---

## Scope — what we do NOT claim

- **FedCrime's actual claim was federated training without data sharing.** We
  reproduced it and do not challenge it. We do not say "FedCrime is wrong."
- **The Head/Mid/Tail split was ours**, not theirs. The 41-point gap came from
  our own analysis, using a standard tool that gave a wrong answer.
- Davis & Goadrich (2006) own the base-rate mathematics; Kleinberg et al. (2016)
  and Chouldechova (2017) own the DP-vs-equalized-odds impossibility. We cite
  both as background and claim neither.
- We improved nothing about policing. That is the point of the paper.

---

## Quick start

```
git clone https://github.com/zda25m005-netizen/crime-fairness.git
cd crime-fairness
pip install -r requirements.txt
```

No GPU and no download needed:

```
python analysis/theory.py               # the maths, verified numerically
python analysis/literature_audit.py     # 6 published models vs crime volume
python analysis/decision_impact.py      # does it change decisions? (targeting: yes)
python analysis/reanalysis_published.py # FedCrime's own tables
python analysis/audit_fairness_tools.py # Fairlearn + AIF360 audit
python analysis/multicity_audit.py --probe   # check the portals respond
python analysis/multicity_audit.py --year0 2015 --year1 2019 \
       --grid 24 --category BURGLARY        # the 5-city table above
```

The measurement critique needs **no model and no GPU**. That is deliberate.

**Portal caveats**, recorded because they cost us a day: Austin publishes no
lat/lon, Baltimore's endpoint no longer returns JSON, San Francisco 403s
automated clients, Los Angeles's current file starts in 2020, and Seattle's
burglary label is in `offense_sub_category`. `pandas.read_json(url)` sets no
timeout, so a slow portal hangs indefinitely — fetch year by year through a
wrapper with a timeout, a User-Agent and retries.

---

## Repository structure

```
src/
  fedcrime/     federated ST-GNN pipeline (TCN + ZINB + graph variants)
  fairaudit/    metric taxonomy, skill scores, one-call audit
  pinn/
    build_field.py               point records -> field u(x,y,t)
    crime_pinn.py                the PINN and the Short et al. PDE
    pinn_history.py              macro_auc() -- THE FIX. also the ablations
    crossover.py                 density-binned audit, macro AUC throughout
    signal_ceiling.py            is there any weekly signal? (3-way validated)
    physics_normalized_eval.py   base-rate-free target; the A0 idea FAILED
    pne_validate.py              synthetic validation of the transform

analysis/       standalone analyses, most needing no GPU
  multicity_audit.py             replicates the CRITIQUE across cities, no training
  audit_fairness_tools.py        Fairlearn + AIF360
docs/           written record, incl. STATUS.md and MISTAKES.md
figures/  presentations/  report/  results/  scripts/  notebooks/  tests/
```

---

## Reproducibility

- **Full determinism** — Python, NumPy, torch CPU and CUDA seeded; cuDNN
  autotuning disabled; `torch.use_deterministic_algorithms(True)`. Verify with
  `--repro-check`, which trains twice and asserts bit-identical output.
- **Capacity-matched controls** — every treatment against an identical-size
  model with the component disabled, never a smaller one.
- **5 seeds**, mean ± std, Welch t-tests.
- **Time-based splits** — never random.
- **Leakage tests** — features verified to use only data strictly before the
  target week, three separate ways.

⚠️ Results from before the determinism fix are **void**. Results from before the
macro-AUC fix (2026-09-07) are **retracted** — see `STATUS.md`.

**Cross-hardware note.** Runs are bit-identical within one machine. Across GPUs
the direction and significance are stable; exact values vary slightly.

---

## Ethics

1. **Recorded crime is not crime.** It measures where police went and what they
   wrote down. Base rates are themselves shaped by policing, so the confound
   compounds: a more heavily policed area has a higher recorded rate, which
   inflates model scores there, which makes the fairness gap look larger, which
   sends remediation to the wrong place.
2. **The burglary model derives from Broken Windows theory**, which is contested
   — Sampson & Raudenbush (2004), Goodson & Hoyer-Leitzel (2021). We implement
   it to *measure* what it does, not because we think it is correct.
3. **We built nothing that improves policing, and we are not trying to.** This
   work argues that the reported numbers in this literature should not be
   trusted, including our own.

---

## Key references

Davis & Goadrich (2006) · Boyd et al. (2012) · Chouldechova (2017) ·
Kleinberg, Mullainathan & Raghavan (2016) · Lipton et al. (2014) ·
Short et al. (2008) · Lum & Isaac (2016) · Richardson et al. (2019) ·
Ensign et al. (2018) · Sampson & Raudenbush (2004)

MIT licensed. Data is public; see `scripts/` to rebuild it.
