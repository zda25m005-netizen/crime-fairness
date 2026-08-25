# Timeline — day 1 to now

The project in the order it actually happened, including the parts that went
wrong. Each phase records what we did, what we found, and what it changed.

---

## Phase 1 — Reproduce the paper

**Goal.** Rebuild FedCrime (Neurocomputing 679, 2026): federated crime
prediction across six police stations, with a zero-inflated negative binomial
(ZINB) loss to handle the fact that most days in most places have no crime.

**Built.** TCN temporal encoder (dilated causal convolutions + residuals), ZINB
heads (Eqs. 6–8), FedAvg aggregation, Head/Mid/Tail region grouping at 20/30/50
percentiles by training-period crime volume.

**Result.** Reproduced the paper's trend on real Los Angeles 2018 data:
72.16 → 48.43 F1 as the data subset became sparser.

**Files.** `src/fedcrime/` · `notebooks/FedCrime_FullStudy.ipynb`

---

## Phase 2 — Comparison baselines

**Goal.** Situate the work against what the field uses.

**Built.** Nine federated methods: FedAvg, FedProx, SCAFFOLD, MOON, FBLG,
FedAvgM, FedTrimAvg, Multi-Krum, Bulyan.

**Note.** A typo in the Bulyan implementation (`sset_w` / `sel_size`) was found
and fixed during testing.

---

## Phase 3 — The fairness problem appears

**Goal.** Measure performance per region group rather than in aggregate.

**Result.** Head regions scored ≈ 69.4, Tail regions ≈ 28.2 — a **41-point gap**
on Los Angeles. This became the research question.

**Files.** `figures/s1_gap.png`

---

## Phase 4 — Three fixes, all failed

| Method | Setting | Result |
|---|---|---|
| Reweighting | sparse-region weight clipped to [1, 4] | gap 41 → **52** (worse) |
| Group-DRO | τ = 3, 6, 10 | no change, ~41 at every setting |
| Adaptive thresholds | per region × crime type, tuned on validation | improved to ~41, then plateaued |

**What it changed.** Three reasonable interventions failing in three different
ways suggested the problem was not what we had assumed.

See [`METHODS.md`](METHODS.md) for why each one cannot work.

---

## Phase 5 — The diagnosis

**Idea.** A model with **zero skill** — one that predicts "crime" everywhere,
every day — has F1 = `2p/(1+p)` where `p` is the base rate. In busy areas that is
66.7; in quiet areas 9.5. A **57-point gap from a model that never looks at the
data.**

**Verified numerically** (`analysis/theory.py`):
- `A_F1(p) = 2p/(1+p)` matches empirical F1 across 8 base rates
- AUC = 0.5 at every base rate, as expected
- A controlled simulation with **equal skill by construction** gives an F1 gap of
  **38.69** and an AUC gap of **0.19**

**Extended:**
- **Four cities** (`analysis/baserate_analysis.py`) — artifact gap LA 42.8,
  Chicago 45.9, NYC 19.9, SF 48.8; correlation with sparsity r = +0.835
- **Six published models** (`analysis/literature_audit.py`) — r = 0.899;
  ST-HSL at r = −0.43 acts as an internal negative control
- **FedCrime's own tables** (`analysis/reanalysis_published.py`) — r = 0.998
- **A second, non-crime domain** — Chicago 311 service requests, same spatial
  units, same pattern

**Important.** This mathematics is **not new** — Davis & Goadrich (2006), Boyd
et al. (2012), Chouldechova (2017). We cite it as background. What is new is
showing it operating across this literature.

---

## Phase 6 — Does it change decisions?

`analysis/decision_impact.py`

| Decision | Result |
|---|---|
| Which model is best, per group | **0 of 5** groups change |
| Which group needs remediation | **6 of 6** models change |

Mean Spearman ρ between raw-F1 ranking and skill-corrected ranking: **−0.700**.

The negative half matters as much as the positive half: the confound does *not*
invalidate model comparisons.

---

## Phase 7 — Robustness work (excluded from the paper)

Novel *sparsity-camouflaged* attack, plus a density/graph "vouching" aggregator
(`strust2`). Result was weak — won 1 of 6 tests, significantly worse twice.

⚠️ **All of these numbers are void.** They predate the determinism fix in
Phase 9. Kept in the repository for continuity; excluded from any write-up.

---

## Phase 8 — Graphs, and a result we had to withdraw

**Goal.** Test whether letting neighbouring regions share information helps —
plain GCN, gated, attention, at depths 1–4.

**First run** looked clean: plain degraded with depth, gated stayed flat. We
nearly wrote it up.

**Then we re-ran the same command.** Completely different numbers. The same
configuration produced 48.86 one day and 54.03 the next; one setting gave 46.79
once and 0.00 the next time.

**The results were not reproducible.** Everything from this phase was discarded.

---

## Phase 9 — Two mistakes found and fixed

**Mistake 1 — non-determinism.** Only `torch.manual_seed` was being set. cuDNN
autotuning and non-deterministic CUDA kernels were free to vary between runs.

*Fix:* `seed_everything()` — Python, NumPy, torch CPU and CUDA, cuDNN
determinism, `use_deterministic_algorithms(True)`, plus gradient clipping and a
non-finite-loss guard. Added `--repro-check`, which trains the same config twice
and asserts bit-identical output. **PASS: difference 0.00000000.**

**Mistake 2 — mismatched controls.** Graph models at depth 4 were being compared
against a no-graph control with only 1 dense layer. Any difference could have
come from size rather than from the graph.

*Fix:* every depth is now paired with its own no-graph control at the same layer
count. The `delta` column isolates the graph.

See [`MISTAKES.md`](MISTAKES.md).

---

## Phase 10 — The graph result, properly measured

3 architectures × 4 depths × 2 cities × 5 seeds, capacity-matched, deterministic.

| | Los Angeles | Chicago |
|---|---|---|
| Negative deltas | 12/12 | 10/12 |
| Significant at p<0.05 | 9/12 | — |
| Significantly positive | **0** | **0** |

Plain GCN collapses at depth ≥ 3. Gated and attention track the control from
just below — they learn to switch the graph off.

**Files.** `analysis/threshold_sweep.py` · `figures/g1_absolute.png`,
`g2_zoom.png`, `g3_all24.png`

---

## Phase 11 — The threshold sweep

**Question a reviewer would ask.** Is the artifact a property of the metric, or
a quirk of one architecture?

**Test.** Take the model with **no graph**, change **only the decision
threshold**, sweep it from 0.05 to 0.95.

| | Tail F1 | says "crime" | Tail AUC |
|---|---|---|---|
| no graph, threshold 0.50 | 27.96 | 22.7% | 61.99 |
| graph model | 40.83 | 89.7% | 58.94 |
| **no graph, threshold 0.35** | **41.59** | 63.9% | **61.99** |

Moving one threshold gives **+13.63**, *exceeding* the entire architectural
intervention's +12.87 — with higher AUC and less over-prediction. AUC reads
**61.99 on all nineteen rows**, because AUC cannot depend on a threshold.

This became the strongest single result in the project.

---

## Phase 12 — Physics-informed model

**Prompted by** the supervisor's suggestion to try PINNs.

**Built.**
- `src/pinn/build_field.py` — point crime records → continuous field `u(x,y,t)`.
  Chicago burglary 2011–2015, 24×24 grid, weekly: 94,992 events, 183 training
  weeks, 41.5% empty cells
- `src/pinn/crime_pinn.py` — the Short et al. (2008) reaction–diffusion system,
  with the attractiveness baseline `A0` learned as a **spatial field** (faithful
  to the original paper, which states A0 "is not necessarily uniform"),
  a **Poisson** likelihood for count data, collocation points masked to the city,
  and a **generic-smoothness ablation**

**Results** (5 seeds):

| | pooled AUC | vs control | t |
|---|---|---|---|
| Short PDE + Poisson | 71.49 ± 2.93 | +18.32 | **9.57** |
| Short PDE + MSE | 71.12 ± 1.50 | +17.95 | **12.46** |
| no physics (control) | 53.17 ± 2.46 | — | — |
| **generic smoothness** | **44.63** | −6.85 | — |

The ablation is the point: generic smoothing scores *below* the control and
*below chance*. The gain is not regularisation — it is the burglary equations.

Tail over-flagging fell from **3.2× to 1.8×** reality.

**And the measurement finding reappears.** Across these configurations pooled AUC
spans **18.32 points** while the Head−Tail F1 gap spans **2.06**. Within the Tail
group alone, F1 is 43.64 / 43.36 / 43.45 — a spread of 0.28 — while Tail AUC
spans 7.2 points.

**Honest limit.** Per-group AUC sits at 51–58. The model learned *where*, not
*when*.

---

## Where things stand

| | Status |
|---|---|
| Experiments | **complete** — no further runs needed |
| Determinism and controls | verified |
| Written paper | **not started** |
| Ethics and limitations section | not started |
| Target | ACM FAccT 2027 — abstract 27 Oct, paper 3 Nov |
