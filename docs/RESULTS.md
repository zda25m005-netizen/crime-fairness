# Results

Every number, with seeds and significance. Anything produced before the
determinism fix is excluded — see [`MISTAKES.md`](MISTAKES.md).

---

## 1. The theory, verified numerically

`analysis/theory.py`

| Check | Result |
|---|---|
| `A_F1(p) = 2p/(1+p)` vs empirical F1 | matches across 8 base rates |
| AUC of a skill-free model | 0.5 at every base rate |
| Controlled sim, **equal skill by construction** | F1 gap **38.69**, AUC gap **0.19** |

A skill-free predictor scores 66.7 at p=0.50 and 9.5 at p=0.05 — a 57-point
"fairness gap" from a model that never reads the data.

---

## 2. Base rates across cities

`analysis/baserate_analysis.py`

| City | Artifact gap |
|---|---|
| Los Angeles | 42.8 |
| Chicago | 45.9 |
| New York | 19.9 |
| San Francisco | 48.8 |

Correlation of artifact gap with sparsity: **r = +0.835**.
Chicago 311 (non-crime domain, same spatial units): same pattern.

---

## 3. The published literature

`analysis/literature_audit.py` · `analysis/sensitivity.py`

Correlation between reported F1 and the base-rate proxy, six models:

| Assumption for base rate | macro-F1 | micro-F1 |
|---|---|---|
| Poisson (as originally submitted) | 0.642 | 0.868 |
| Negative binomial, k=5 | 0.651 | 0.875 |
| Negative binomial, k=1 | 0.662 | 0.884 |
| Raw rate, no distribution | 0.696 | 0.899 |
| Log crime volume | 0.672 | 0.890 |
| **Rank of crime volume only** | **0.719** | **0.875** |

The last row assumes **nothing** — only the ordering of groups, which the source
papers state directly. It is *higher* than the Poisson version.

**Inference** (permutation tests, 200k permutations):

| Model | r (micro) | permutation p |
|---|---|---|
| DeepCrime | 0.888 | 0.0085 |
| MiST | 0.985 | 0.0084 |
| CrimeForecaster | 0.986 | 0.0332 |
| HAGEN | 0.995 | 0.0166 |
| ST-HSL | 0.463 | 0.3594 ← negative control |
| AIST | 0.893 | 0.0165 |

Fisher combined: **chi2 = 44.4, df = 12, p = 1.32 × 10⁻⁵**
Pooled correlation +0.868, bootstrap 95% CI across models **[+0.694, +0.974]**

FedCrime's own tables (`analysis/reanalysis_published.py`): **r = 0.998**,
mean |reported − skill-free baseline| = 1.07 F1 points.

---

## 4. Decision impact

`analysis/decision_impact.py`

| Decision | Changes? |
|---|---|
| Which model is best, within a group | **0 of 5** |
| Which group receives remediation | **6 of 6** |

Mean Spearman ρ between the two rankings: **−0.700**.

---

## 5. Graph depth ablation — capacity-matched, 5 seeds

`analysis/threshold_sweep.py`, `--depth-sweep` mode

Delta = graph minus its own no-graph control at the same layer count.

### Los Angeles

| depth | plain | gated | attention |
|---|---|---|---|
| 1 | −12.92 ** | −1.03 ** | −0.47 |
| 2 | −6.51 ** | −1.04 ** | −1.10 ** |
| 3 | −44.49 ** | −0.51 | −0.71 ** |
| 4 | −52.82 ** | −2.11 | −0.85 ** |

### Chicago

| depth | plain | gated | attention |
|---|---|---|---|
| 1 | −16.29 ** | −2.30 | +0.37 |
| 2 | −3.55 ** | −2.83 | +0.07 |
| 3 | −57.52 ** | −2.79 | −1.15 |
| 4 | −56.01 ** | −4.46 ** | −7.35 |

`**` = p < 0.05, Welch t-test.
**22 of 24 negative. 9 significantly negative. 0 significantly positive.**

Absolute scores (no-graph control, LA): 53.68, 53.66, 53.57, 53.52 — stable
across depth, std ≤ 0.45.

---

## 6. Threshold sweep — the artifact without any architecture

`analysis/threshold_sweep.py`, Chicago, no graph, 5 seeds

| threshold | Tail F1 | says "crime" | Tail AUC |
|---|---|---|---|
| 0.05 | 39.75 | 99.6% | 61.99 |
| 0.25 | 41.09 | 86.4% | 61.99 |
| **0.35** | **41.59** | 63.9% | 61.99 |
| 0.50 (default) | 27.96 | 22.7% | 61.99 |
| 0.75 | 4.55 | 1.9% | 61.99 |

**AUC is 61.99 on all nineteen rows** — it cannot depend on a threshold.

| | Tail F1 | says "crime" | Tail AUC |
|---|---|---|---|
| no graph, threshold 0.50 | 27.96 | 22.7% | 61.99 |
| graph model | 40.83 | 89.7% | 58.94 |
| **no graph, threshold 0.35** | **41.59** | 63.9% | **61.99** |

Graph intervention: **+12.87**. Threshold change alone: **+13.63** — larger,
with higher AUC and less over-prediction.

---

## 7. PINN — 5 seeds

`src/pinn/crime_pinn.py` · Chicago burglary 2011–2015, 24×24 weekly

| configuration | n | pooled AUC | F1 | RMSE | learned eta |
|---|---|---|---|---|---|
| Short PDE + Poisson | 5 | **71.49 ± 2.93** | 70.52 ± 1.49 | 1.354 ± 0.036 | 0.0121 |
| Short PDE + MSE | 5 | **71.12 ± 1.50** | 69.99 ± 1.12 | 1.379 ± 0.029 | 0.0172 |
| no physics (control) | 5 | 53.17 ± 2.46 | 66.01 ± 0.93 | 1.750 ± 0.033 | — |
| generic smoothness | 1 | **44.63** | 66.90 | 1.523 | — |

**Versus control:**

| | ΔAUC | t | ΔF1 | t |
|---|---|---|---|---|
| Short + Poisson | **+18.32** | 9.57 ** | +4.51 | 5.15 ** |
| Short + MSE | **+17.95** | 12.46 ** | +3.98 | 5.49 ** |

The generic-smoothness ablation sits **below the control and below chance (50)**.
The gain is the burglary equations, not regularisation.

### Tail regions (crime occurs in 28.5% of weeks)

| configuration | Tail F1 | Tail AUC | says "crime" | × too often |
|---|---|---|---|---|
| Short + Poisson | 43.64 ± 2.75 | **58.35 ± 3.29** | 52.2 ± 5.3% | **1.8×** |
| Short + MSE | 43.36 ± 3.14 | 57.81 ± 3.05 | 56.1 ± 8.6% | 2.0× |
| no physics | 43.45 ± 0.98 | 51.14 ± 1.72 | 90.2 ± 7.6% | 3.2× |

---

## 8. The headline — does the fairness metric track real skill?

| configuration | pooled AUC | Head−Tail F1 gap | AUC gap |
|---|---|---|---|
| Short + Poisson | 71.49 | 47.32 ± 2.65 | **−1.92 ± 3.51** |
| Short + MSE | 71.12 | 47.60 ± 3.21 | **−3.50 ± 3.32** |
| no physics | 53.17 | 45.54 ± 1.91 | +0.33 ± 2.73 |

**Real skill (AUC) spans 18.32 points. The F1 "fairness gap" spans 2.06 points.**

Sharper still, **within the Tail group alone**: F1 reads 43.64 / 43.36 / 43.45 —
a spread of **0.28** — while Tail AUC spans **7.2 points**. Same group, same
base rate, same threshold procedure. One metric sees a large real difference in
how well these models serve poor neighbourhoods; the other sees nothing.

**And the sign reverses.** The F1 gap says Head regions are ~47 points better
served. The AUC gap for both physics models says Head is *worse* served. The two
metrics disagree about **who is being failed**.

---

## 9. Reproducibility

| Check | Result |
|---|---|
| Same config, twice, same machine | difference **0.00000000** — PASS |
| Same config across sessions | byte-identical (LA depth sweep re-run) |
| Across different GPUs | direction and significance stable; exact values vary slightly |

Example of cross-hardware variation: the Chicago AUC check gave Tail F1 +16.95 /
Tail AUC −4.01 on one machine and +12.87 / −3.05 on another. Same conclusion,
different third digit. Report hardware alongside numbers.
