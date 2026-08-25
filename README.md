# Are Fairness Scores in Crime Prediction Telling Us the Truth?

Reported fairness gaps in crime-prediction models are largely explained by **how
often crime happens**, not by how well the model serves each neighbourhood — and
correcting for that is **not enough**, because a model can close a corrected gap
by moving its decision threshold while getting genuinely worse.

This repository contains everything: the code, the data pipelines, every method
we tried (including the six that failed), the mistakes we found in our own work,
and the results that survived.

---

## The three findings

**1. The standard fairness metric can be moved without the model improving.**
On Chicago, an intervention raised Tail-region F1 by **+16.95** and closed our
base-rate-corrected gap from 40.97 to −2.39 — while Tail AUC *fell* 4.01 points
and the positive-prediction rate went from 23% to 80%. Moving a **single decision
threshold** on a model with no graph exceeded that entire "improvement"
(+13.63 vs +12.87) with AUC provably unchanged at 61.99.

**2. Published fairness gaps track crime volume.**
Across six published crime-prediction models, reported scores correlate with
crime volume at **r = 0.87–0.90**, Fisher combined **p = 1.3 × 10⁻⁵**. The result
holds with **no distributional assumption at all** (rank-only correlation 0.875).

**3. It changes who gets help.**
Model *selection* is unaffected (0/5 groups change). Remediation *targeting*
flips for **6/6 models**. The flaw does not mislead you about which method to
use — it misleads you about *where the model is failing*.

**Plus a positive modelling result.** A physics-informed neural network using the
Short et al. (2008) burglary hotspot equations beat its capacity-matched control
by **+18.3 AUC** (t = 9.6, 5 seeds) and cut over-flagging of poor neighbourhoods
from 3.2× to 1.8× reality.

---

## Quick start

```bash
git clone <this-repo>
cd crime-fairness-repo
pip install -r requirements.txt
```

Analyses that need **no GPU and no data download** — run these first:

```bash
python analysis/theory.py               # the maths, verified numerically
python analysis/literature_audit.py     # 6 published models vs crime volume
python analysis/sensitivity.py          # does the result depend on assumptions? (no)
python analysis/decision_impact.py      # does it change decisions? (yes, for targeting)
python analysis/reanalysis_published.py # FedCrime's own tables
```

Everything else needs data and a GPU — see [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Repository structure

```
src/
  fedcrime/          federated ST-GNN pipeline (TCN + ZINB + graph variants)
    robust_fair_gnn.py   the main experiment script; all modes live here
  fairaudit/         the auditing tool — metric taxonomy, skill scores, one-call audit
  pinn/              physics-informed model
    build_field.py       point crime records -> continuous field u(x,y,t)
    crime_pinn.py        the PINN, the Short et al. PDE, and the ablations
    signal_check.py      is there week-to-week signal at all? (run before tuning)

analysis/            standalone analyses, most needing no GPU
scripts/             data preparation from city open-data portals
notebooks/           Colab/Kaggle notebooks, self-contained
docs/                the full written record — see below
figures/             every chart in the paper and slides
presentations/       slides and speaking scripts
results/             saved JSONL result files
tests/               unit tests
```

---

## The written record

| Document | What it covers |
|---|---|
| [`docs/TIMELINE.md`](docs/TIMELINE.md) | Day 1 to now. Every phase, in order, with what changed and why |
| [`docs/METHODS.md`](docs/METHODS.md) | Every method tried, how it works, why we tried it, why it failed or worked |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Every number, with seeds and significance tests |
| [`docs/MISTAKES.md`](docs/MISTAKES.md) | Errors we found in our own work, and what we did about them |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | How to re-run everything from scratch |

---

## What we tried, and what happened

Every method was tested against an **identical model with the new component
switched off** — same size, same seeds, same training. Any difference is
attributable to the component and nothing else.

| # | Method | Result |
|---|---|---|
| 1 | Reweighting (higher loss weight on sparse regions) | ❌ gap 41 → 52, **worse** |
| 2 | Group-DRO (optimise the worst group) | ❌ no change at τ = 3, 6, 10 |
| 3 | Adaptive per-region thresholds | ❌ plateaued at ~41 |
| 4 | Plain GCN, depths 1–4 | ❌ up to −57 AUC; collapses at depth ≥3 |
| 5 | Gated GCN, depths 1–4 | ❌ −0.5 to −4.5; learns to switch itself off |
| 6 | Attention GCN, depths 1–4 | ❌ −0.5 to −7.4 |
| 7 | **PINN (Short et al. PDE)** | ✅ **+18.3 AUC, t = 9.6** |

**22 of 24** graph comparisons negative, 9 significantly so. Not one positive.

---

## Reproducibility

Every model result in `docs/RESULTS.md` comes from code with:

- **Full determinism** — Python, NumPy, torch CPU and CUDA seeded; cuDNN
  autotuning disabled; `torch.use_deterministic_algorithms(True)`.
  Verify with `--repro-check`, which trains the same config twice and asserts
  bit-identical output.
- **Capacity-matched controls** — every treatment is compared against an
  identical-size model with the component disabled, never against a smaller one.
- **5 seeds** with mean ± std and Welch t-tests.
- **Time-based splits** — never random, so the future cannot leak into training.

**Cross-hardware note.** Runs are bit-identical within one machine. Across
different GPUs the direction and significance are stable but exact values vary
by a small amount. Report hardware alongside numbers.

⚠️ **Results produced before the determinism fix are void** and are not included
here. See [`docs/MISTAKES.md`](docs/MISTAKES.md) for what was discarded and why.

---

## Ethics

This work analyses predictive-policing systems. Two points we consider essential:

1. **Recorded crime is not crime.** It measures where police went and what they
   wrote down. Base rates are themselves shaped by policing patterns, so the
   confound we describe compounds: a more heavily policed area has a higher
   recorded rate, which inflates model scores there, which makes the fairness gap
   look larger, which sends remediation to the wrong place.
2. **The burglary model we implement derives from Broken Windows theory**, which
   is contested — see Sampson & Raudenbush (2004) and Goodson & Hoyer-Leitzel
   (2021), who argue the crime-hotspot modelling framework encodes systemic
   racism. We implement it in order to *measure* what it does, particularly in
   low-crime neighbourhoods. We do not claim it is a correct description of crime.

---

## Key references

The base-rate dependence of F1 and the invariance of AUC are **established**, and
we do not claim them:

- Davis & Goadrich (2006), *The relationship between Precision-Recall and ROC curves*
- Boyd, Costa, Davis & Page (2012), *Unachievable region in precision-recall space*
- Chouldechova (2017), *Fair prediction with disparate impact*

Crime modelling and predictive policing:

- Short, D'Orsogna, Pasour, Tita, Brantingham, Bertozzi & Chayes (2008),
  *A statistical model of criminal behavior*, Math. Models Methods Appl. Sci. 18
- Lum & Isaac (2016), *To predict and serve?*, Significance
- Ensign, Friedler, Neville, Scheidegger & Venkatasubramanian (2018),
  *Runaway feedback loops in predictive policing*, FAccT
- Richardson, Schultz & Crawford (2019), *Dirty data, bad predictions*, NYU Law Review
- Goodson & Hoyer-Leitzel (2021), *Examining the modeling framework of crime
  hotspot models in predictive policing*, arXiv:2103.11757

---

## Data

No crime data is committed to this repository. All of it is public:

| Source | Used for |
|---|---|
| [Chicago Open Data](https://data.cityofchicago.org/) (`ijzp-q8t2`) | Chicago crime, burglary field |
| [FedCrime repo](https://github.com/vanetlabiitj/FedCrime) | Los Angeles 2018 |
| [NYC Open Data](https://data.cityofnewyork.us/) (`qgea-i56i`) | New York base rates |
| [SF Open Data](https://data.sfgov.org/) (`wg3w-h783`) | San Francisco base rates |
| Chicago 311 (`v6vf-nfxy`) | Second (non-crime) domain |

`scripts/` rebuilds all of them.

---

## License

MIT — see [LICENSE](LICENSE).
