# PROJECT STATUS — read this first

Last updated: 2026-09-07

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
5. Fairlearn `demographic_parity` (the DEFAULT) opens a 21.26 +/- 0.52 TPR gap
   on equal-skill data. AIF360 RejectOption: Tail FPR 13.49% -> 41.55%.
6. On the fair target NO neural model beats a 4-week moving average anywhere.

## Scope — do not overclaim
- FedCrime's actual claim was **federated training without data sharing**. Not
  challenged. Do not write "FedCrime is wrong."
- The Head/Mid/Tail split was **ours**, not theirs. The 41-point gap was our
  own analysis artifact. Say so.
- Kleinberg 2016 / Chouldechova 2017 own the DP-vs-EO impossibility. Our claim
  is narrower: it is the library DEFAULT and the cost is quantifiable.
- Davis & Goadrich 2006 own the base-rate math. Cite prominently.

## Files
| path | what |
|---|---|
| `src/pinn/pinn_history.py` | `macro_auc()` — THE fix. Also the ablation runner. |
| `src/pinn/crossover.py` | density-binned audit, macro AUC throughout |
| `src/pinn/signal_ceiling.py` | is there any weekly signal? (validated 3 ways) |
| `src/pinn/physics_normalized_eval.py` | base-rate-free target; A0 idea FAILED (r=-0.45) |
| `analysis/multicity_audit.py` | FAccT item 1 — replicates the CRITIQUE, no training |
| `analysis/audit_fairness_tools.py` | Fairlearn + AIF360 audit |
| `report/report_v2.tex` | 17-page technical report, method+results per phase |

## NEXT (FAccT plan, ~30% -> ~50%)
1. [IN PROGRESS] Run `analysis/multicity_audit.py --probe` then full, 6-8 cities.
2. Deepen the Fairlearn/AIF360 audit; add real Chicago scores.
3. Ethics section — engage Lum & Isaac, Richardson et al., Ferguson.
   State plainly: we improved nothing about policing, and that is the point.
4. Ship the macro-vs-pooled inflation check as a pip-installable tool.
5. Write. 4 weeks.

## Rules learned the hard way
- Never trust a pooled metric over heterogeneous units. Report macro too.
- Every claim needs a capacity-matched control and >=5 seeds.
- Validate new code on synthetic data with a KNOWN answer before real data.
- Numbers from before the determinism fix (Phase 7) are void.
