# Mistakes we found in our own work

Every one of these was caught internally, before submission. They are documented
because the corrections are the reason the remaining results can be defended,
and because a reader deserves to know what was discarded.

---

## 1. Results were not reproducible

**Found by.** Re-running an experiment we had already "finished".

**Symptom.** The same command produced different answers:

| config | first run | second run |
|---|---|---|
| no-graph control, LA | 48.86 ± 0.14 | **54.03 ± 0.13** |
| plain GCN, depth 3 | 46.79 ± 0.00 | **0.00 ± 0.00** |

Both had tight standard deviations, so this was not sampling noise. Seeds 0–2
were shared between the runs and could not have produced both sets of numbers.

**Cause.** Only `torch.manual_seed()` was being set. cuDNN chooses algorithms by
autotuning, and several CUDA kernels accumulate non-deterministically, so the
same seed gave different results between sessions.

**Fix.** `seed_everything()` in `src/fedcrime/robust_fair_gnn.py`:

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

Plus gradient clipping (norm 1.0) and a non-finite-loss guard, since the depth-3
collapse was partly divergence.

**Verification.** `--repro-check` trains the same configuration twice and
asserts bit-identical output. **PASS — difference 0.00000000.** It later held
across a session restart on different hardware.

**Consequence.** Every model result produced before this fix was discarded,
including the entire first depth ablation and all attack/defence experiments.

---

## 2. Controls were not capacity-matched

**Found by.** Investigating why a control number had moved between experiments.

**Symptom.** In the depth sweep, `depth = 0` used `max(1, 0) = 1` dense layer,
while graph models at depth 4 used four. We were comparing a 4-layer graph model
against a 1-layer control.

**Why it matters.** Any difference could have come from model size rather than
from the graph. The comparison could not support the conclusion drawn from it.

**Fix.** Every depth is now paired with its **own** no-graph control at the same
layer count, and the reported `delta` isolates the graph:

```
layers | graph F1      | no-graph F1   | delta
     1 | 40.76 ± 2.62  | 53.68 ± 0.45  | -12.92
     2 | 47.15 ± 0.49  | 53.66 ± 0.27  |  -6.51
```

**Consequence.** The direction of the original conclusion survived, but the
magnitudes changed and the "no-graph = 48.86" baseline was wrong.

---

## 3. A result that looked like a breakthrough

**What happened.** After the fixes, plain GCN at depth 2 appeared to improve
Tail-region F1 substantially: **+7.97 on LA, +19.24 on Chicago**, both
significant. Our base-rate correction reported the gap closing from 40.97 to
−2.39 — an apparently perfect fairness result.

**The check.** Evaluated with AUC, which cannot be moved by base rate or by the
decision threshold.

| | control | graph |
|---|---|---|
| Tail F1 | 23.42 | 40.38 |
| **Tail AUC** | **62.03** | **58.02** |
| Tail TPR | 23.46 | 79.91 |

The model had not improved. It had started predicting "crime" in 80% of poor
neighbourhoods instead of 23%, and its ranking ability had *fallen*.

**Consequence.** Withdrawn as a positive result — and it became the paper's
central finding instead. It also showed that our own skill-score correction was
fooled, because `2p/(1+p)` corrects for base rate but **not** for operating
point.

---

## 4. An evaluation bug that produced NaN

**Symptom.** Every per-group F1 and AUC came out `NaN`.

**Cause.** Binary labels were derived from the **Gaussian-smoothed** field, so
almost every in-city cell was non-zero and every label was 1. With only one
class present, F1 and AUC are undefined.

**Fix.** Labels now come from the **raw counts** (`U_raw`) — "did a burglary
actually occur in this cell this week" — while the smoothed field is used only
where the PDE needs a differentiable surface.

---

## 5. An adaptive sampler that was not the cited method

**Symptom.** `--sampling adaptive` gave no benefit (66.57 vs 66.61).

**Cause.** It implemented plain residual-based resampling. The method in the
paper we were testing (Lin & Chen, causality-guided adaptive sampling) requires
**causal weighting** — `exp(−ε · Σ earlier residuals)` — so that early times must
be fitted before later ones. Without that, it is not their method.

**Status.** Documented as not-yet-implemented rather than reported as a negative
result for their approach.

---

## 6. Collocation points outside the city

**Symptom.** Found during review of the PINN, not from a failing number.

**Cause.** PDE residuals were evaluated at points sampled uniformly from the
lat/lon bounding box — which includes Lake Michigan and areas outside Chicago.
Burglary physics was being enforced over water.

**Fix.** Collocation points are now sampled from in-city grid cells with
sub-cell jitter.

---

## Claims we corrected in our own writing

| Claim made | Correction |
|---|---|
| `2p/(1+p)` described as a "ceiling" or "maximum possible F1" | It is the **skill-free baseline**, not an upper bound. Models can and do exceed it. |
| "The theory is a novel contribution" | It is established — Davis & Goadrich 2006, Boyd 2012, Chouldechova 2017. Cited as background. |
| "Attention improves Tail performance 24.8 → 28.0" | Single-seed noise. Did not survive 5 seeds. |
| "Re-analyse FairTP / FairDRL-ST with the F1 derivation" | Those papers do regression (MAE/RMSE). The derivation does not apply. |
| "Runs are bit-identical" | True **within** one machine. Across GPUs, direction and significance hold but exact values vary. |
