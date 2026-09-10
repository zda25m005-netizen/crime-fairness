# PROJECT STATUS — read this first

Last updated: 2026-09-10

## One line
Crime-prediction fairness. **The model results are dead. The measurement
critique is the paper.** Target: ACM FAccT 2027 (abstract 27 Oct, paper 3 Nov).

## RETRACTED — do not use these numbers
| claim | was | corrected |
|---|---|---|
| Tail advantage over trivial baseline | +11.11 | ~+0.9 |
| Physics vs no-physics (Tail) | +2.81, 8/8 bins | **+0.16, 4/8 bins** |
| Density crossover for physics | r = -0.78 | **r = -0.02** |
| PINN "+18.32 AUC" headline | on raw target | dies under macro AUC |

All three were the same artifact: **pooled AUC over heterogeneous cells lets a
model win by ranking CELLS instead of ranking WEEKS.** Fixed by `macro_auc()`
in `src/pinn/pinn_history.py` — per-cell AUC, then averaged. Proven invariant:
a scorer that only knows cell identity gets pooled 66.67 / macro 50.45.

**The physics prior does not work. Do not try to revive it.**

## What is TRUE and survives

1. `A_F1(p) = 2p/(1+p)` — a do-nothing model scores 72 at 56% base rate.
   FedCrime's reported 72.16 / 48.43 is that curve (our reanalysis: r = 0.998).

2. Six published models: reported F1 vs base rate r = 0.899, Fisher p = 1.3e-5.
   ST-HSL non-significant = internal negative control.

3. Threshold sweep: no-graph at tau=0.35 beats the graph model (41.59 vs 40.83)
   at HIGHER AUC. AUC flat at 61.99 on all 19 rows.

4. Graphs: 22/24 capacity-matched configs negative, 0 significantly positive.

5. **MULTI-CITY (2026-09-10, `analysis/multicity_audit.py`).** No model trained.
   A scorer with IDENTICAL skill in every group, so the honest gap is 0.00:

   | city | window | base gap | F1 gap | macro AUC gap |
   |---|---|---|---|---|
   | Chicago | 2015-19 | +55.1% | +36.09 | +1.74 |
   | Los Angeles | 2020-24 | +64.1% | +39.40 | +0.15 |
   | New York | 2015-19 | +71.9% | +48.70 | -0.66 |
   | Seattle | 2015-19 | +51.5% | +43.64 | -0.26 |
   | Cincinnati | 2015-19 | +32.7% | +39.26 | +1.36 |
   | **mean** | | +55.1% | **+41.42 +/- 4.87** | **+0.46 +/- 1.04** |

   Macro AUC across all 15 city-group cells: **76.21 +/- 0.66** — flat, as it
   must be. Pooling inflation up to **+13.73**.

6. **The artifact is quantitatively PREDICTED, not just present.** `A_F1(p)`
   computed from base rates alone vs the observed F1 gap across the 5 cities:
   **r = +0.9355, exact permutation p = 0.025**, observed/predicted
   **0.786 +/- 0.052**. This is the strongest single claim in the project.

7. **The fairness libraries do the same thing** (`analysis/audit_fairness_tools.py`,
   re-verified 2026-09-10 on fairlearn 0.14.0 / aif360). On equal-skill data:
   - Fairlearn `demographic_parity` (**the DEFAULT**) equalises the flag rate at
     ~24% in every group. **Direction matters and we previously stated it
     loosely:** the gap is Head-minus-Tail **-21.26 +/- 0.52**, i.e. the SPARSE
     group ends up with the HIGHER TPR (56.33 vs 35.50) and the higher
     false-alarm rate (FPR 19.84 vs 7.12, +11.03). Concretely: the low-crime
     area gets over-flagged while the busy area misses real crime
     (TPR 48.00 -> 35.50).
   - `true_positive_rate_parity` (-0.02) and `equalized_odds` (+0.18) behave
     correctly. The failure is the parity-of-selection family, which is the
     default — it is not a blanket indictment.
   - AIF360 `RejectOptionClassification` on statistical parity: Tail
     FPR **13.59% -> 41.09%**, Head FPR 14.99% -> 27.87%.
     `CalibratedEqOdds(fnr)` changes essentially nothing (13.52 vs 13.59).
   - Per-group AUC is unmoved on every row. Post-processing re-thresholds; it
     does not re-rank.

8. **REAL CHICAGO DATA, pooled vs macro (2026-09-10, PART 6 of the toolkit
   audit).** 61,640 burglaries, 262 weeks, 249 cells, logistic regression on
   lag features, operating point = a 20% patrol budget (NOT a 0.50 threshold —
   at 0.50 the model flags 100% of Head and every number downstream is junk).

   | group | base | pooled AUC | macro AUC | inflation |
   |---|---|---|---|---|
   | Head | 78.3% | 61.90 | 55.77 | +6.13 |
   | Mid | 58.5% | 58.84 | 54.84 | +4.00 |
   | Tail | 24.2% | **71.72** | **53.54** | **+18.18** |
   | mean | | 64.15 | 54.72 | +9.44 |

   Two things to say about this in the paper:
   - **The inflation is LARGEST in the Tail** — the group the fairness
     literature says is under-served is where the pooled metric flatters the
     model most. Pooled says the Tail is the best-served group (71.72); within
     cells it is the worst (53.54).
   - **Pooled and macro disagree about the SIGN.** Pooled AUC gap −9.82,
     macro AUC gap +2.23. They point at different groups for remediation.

   Dropping the `cell_mean` feature moves pooled only 64.15 → 61.00, so do NOT
   claim one feature supplied all the skill. The claim is the vertical gap:
   macro AUC sits near 50 either way.

   Under `demographic_parity` on this real data the Tail goes from **0 to
   3,528** crime-free cell-weeks flagged, on a model with macro AUC 53.54 —
   i.e. it has no idea which of those places is worth visiting.

9. On the fair target NO neural model beats a 4-week moving average anywhere.

## Scope — do not overclaim
- FedCrime's actual claim was **federated training without data sharing**. Not
  challenged. Do not write "FedCrime is wrong."
- The Head/Mid/Tail split was **ours**, not theirs. The 41-point gap was our
  own analysis artifact. Say so.
- Kleinberg 2016 / Chouldechova 2017 own the DP-vs-EO impossibility. Our claim
  is narrower: it is the library DEFAULT and the cost is quantifiable.
- Davis & Goadrich 2006 own the base-rate math. Cite prominently.
- The multi-city run uses **per-city year windows**, not one common window.
  Cities do not publish the same period. This is fine because the audit compares
  Head vs Tail WITHIN a city, never city against city. Say this in the paper.

## Data access reality (2026-09-10)
Open-data portals are not uniformly reachable. Record this honestly:
- **Austin** (`fdj4-gpfu`) publishes no lat/lon columns.
- **Baltimore** (`wsfq-mvij`) no longer returns JSON.
- **San Francisco** (`wg3w-h783`) returns HTTP 403 to automated clients.
- **Los Angeles** (`2nrs-mtv8`) begins in 2020, not 2015.
- **Seattle**'s burglary label lives in `offense_sub_category`, not
  `offense_category` (which is only PROPERTY / VIOLENT / ALL OTHER).
- `pandas.read_json(url)` passes **no timeout** to urllib, so a slow portal
  hangs for ever. Deep `$offset` paging over a filtered Socrata table gets
  slower every page and times out. Fetch **year by year**, always through a
  wrapper with a timeout, a User-Agent, and retries.

## Files
| path | what |
|---|---|
| `src/pinn/pinn_history.py` | `macro_auc()` — THE fix. Also the ablation runner. |
| `src/pinn/crossover.py` | density-binned audit, macro AUC throughout |
| `src/pinn/signal_ceiling.py` | is there any weekly signal? (validated 3 ways) |
| `src/pinn/physics_normalized_eval.py` | base-rate-free target; A0 idea FAILED (r=-0.45) |
| `analysis/multicity_audit.py` | FAccT item 1 — replicates the CRITIQUE, no training |
| `analysis/audit_fairness_tools.py` | Fairlearn + AIF360 audit |
| `results/multicity_audit.json` | the 5-city numbers above |
| `report/report_v2.tex` | 17-page technical report, method+results per phase |

## NEXT (FAccT plan, ~30% -> ~50%)
1. [DONE 2026-09-10] Multi-city replication — 5 cities, committed.
2. [DONE 2026-09-10] Deepened Fairlearn/AIF360 audit, 7 parts. All three
   method families fail, in different ways:
   - **post** (ThresholdOptimizer, RejectOption): AUC unchanged, harm
     redistributed. Tail FPR 13.62 → 38.53.
   - **in** (ExponentiatedGradient): accuracy 72.11 → 65.79 at a tight
     demographic-parity bound, by flagging almost nobody.
   - **pre** (DisparateImpactRemover r=1.0): AUC 76.35 → 69.68 and it
     *creates* a +19.32 equal-opportunity gap where there was −2.16.
   - `Reweighing` does nothing at all (76.35 → 76.35).
   - `PrejudiceRemover` at low eta reports a HIGHER AUC than unconstrained
     (86.01 vs 76.05) by using the group indicator AIF360 hands it as a
     feature; at high eta it flags nothing. Both ends are degenerate and no
     standard metric catches either.
   - `equalized_odds` and `true_positive_rate_parity` cost nothing. The
     failure is the parity-of-SELECTION family, which is the default.
   - PART 7: with skill identical throughout, the F1 gap runs 2.16 → 61.88 as
     the base-rate spread widens from 30/20 to 90/4, tracking `A_F1`, while
     the AUC gap stays near 0. The analyst's grid choice sets the reported
     unfairness.
3. Ethics section — engage Lum & Isaac, Richardson et al., Ferguson.
   State plainly: we improved nothing about policing, and that is the point.
4. Ship the macro-vs-pooled inflation check as a pip-installable tool.
5. Write. 4 weeks.

Then ICLR (~10% -> ~20-25%): demonstrate the pooling inflation OUTSIDE crime on
a standard public benchmark, plus the synthetic controlled-heterogeneity sweep.
**Before any ICLR submission the repo needs an anonymous mirror** — README and
CITATION.cff carry the author's name, and a link would be a desk reject.

## Rules learned the hard way
- Never trust a pooled metric over heterogeneous units. Report macro too.
- Every claim needs a capacity-matched control and >=5 seeds.
- Validate new code on synthetic data with a KNOWN answer before real data.
- State the DIRECTION of a gap, not just its magnitude. We described the
  Fairlearn DP result by magnitude alone for a week and it read backwards.
- Numbers from before the determinism fix (Phase 7) are void.
