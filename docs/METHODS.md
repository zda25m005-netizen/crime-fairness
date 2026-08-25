# Methods — every approach we tried

For each: what it is, why we tried it, what happened, and **why**.

Every method was compared against an **identical model with the component
switched off** — same parameter count, same seeds, same training budget.

---

## Part 1 — Fairness interventions

### 1. Reweighting

**What it is.** During training the model is penalised for mistakes. Normally
every mistake counts equally. Reweighting makes mistakes in sparse regions count
more:

```python
rw = np.clip(dens_r.mean() / (dens_r + 1e-6), 1.0, 4.0)   # up to 4x weight
```

**Why we tried it.** The standard first intervention for group disparity.

**Result.** Gap 41 → **52**. Worse.

**Why it failed.** The only way the model can "care more" about a region is to
predict positive there more often. Recall rises slightly; precision falls
sharply; F1 drops. Increasing the loss weight does not add information — it just
shifts the operating point, which is exactly the mechanism this project later
documented in the threshold sweep.

---

### 2. Group-DRO

**What it is.** Instead of minimising average loss, minimise the loss of the
*worst-performing group*, via soft weighting:

```
w_g = softmax(tau * loss_g)      # tau = 3, 6, 10 tested
```

**Why we tried it.** The standard method when one group is systematically
underserved.

**Result.** No change at any τ. Gap stayed ≈ 41.

**Why it failed.** Group-DRO helps when a group is underserved because the
optimiser is ignoring it. Here the Tail group scores low because its events are
**rare**, not because it is being ignored. The attainable F1 in that group is
capped near `2p/(1+p)` regardless of how much attention it receives. Reallocating
optimiser attention cannot change a base rate.

---

### 3. Adaptive thresholds

**What it is.** The model outputs a probability. A threshold converts it to a
decision. Instead of 0.5 everywhere, tune a separate threshold per region and
crime type on validation data.

**Why we tried it.** Sparse regions plausibly need a lower bar.

**Result.** Improved to ≈ 41, then plateaued.

**Why it failed.** Moving a threshold slides you along the precision–recall
curve. It cannot lift the curve. No information enters the model.

> **This failure turned out to be the most important one.** It is the first
> evidence of what the threshold sweep later proved: the decision threshold can
> move the fairness score a long way without any change in model quality.

---

## Part 2 — Graph architectures

All three share the same TCN encoder and ZINB/classifier heads. Only the mixing
step differs. Depth is configurable (`--gnn-layers`), and the no-graph control
uses the **same number of dense layers**.

### 4. Plain GCN

```python
h = relu(einsum('ij,bjh->bih', A, lin(h)))    # repeated n_layers times
```

**Result.** LA: −12.92, −6.51, −44.49, −52.82 at depths 1–4.
Chicago: −16.29, −3.55, −57.52, −56.01. Collapses at depth ≥ 3.

**Why it failed — over-smoothing.** Every layer averages a region with its
neighbours, and there is no mechanism to stop. After 3–4 layers every region has
been mixed with every other and they become indistinguishable, so the model
cannot predict them differently.

Verified numerically (region diversity, cosine spread):

| | diversity |
|---|---|
| input | 1.3495 |
| discrete GCN, 4 layers | **0.0359** |

97% of the distinction between regions is destroyed.

---

### 5. Gated GCN

```python
agg = einsum('ij,bjh->bih', A, lin(h))
g   = sigmoid(gate(h))
h   = relu(g * agg + (1 - g) * h)      # learnable mix
```

**Result.** LA: −1.03, −1.04, −0.51, −2.11. Chicago: −2.30, −2.83, −2.79, −4.46.
Flat across depth — no collapse.

**Why it "failed".** The gate can suppress propagation entirely, and that is what
it learned to do. The model converges to the no-graph solution, minus a small
cost for unused parameters.

**This is the informative result.** When a model is free to choose how much to
use its neighbours, it chooses ≈ none. The limitation is *informational*, not
architectural.

---

### 6. Attention GCN

```python
scores = einsum('bih,bjh->bij', Wh, Wh) / sqrt(d)
scores = scores.masked_fill(A == 0, -inf)
h      = relu(einsum('bij,bjh->bih', softmax(scores), Wh))
```

**Result.** LA: −0.47, −1.10, −0.71, −0.85. Chicago: +0.37, +0.07, −1.15, −7.35.

**Why.** Attention can down-weight neighbours but cannot switch them off entirely
the way gating can, so it degrades slightly more than gated at depth. The two
positive values are far below run-to-run variation and are not significant.

---

### Also tried

**Neural ODE / continuous depth** (`--gnn ode`). Implemented as forward-Euler
integration of `dh/dt = σ(A·h·W) − h` with one **shared** weight matrix across
steps. Swept α ∈ {0.05 … 1.0}. Flat at the control. α = 1.0 exactly reproduces
the discrete GCN, confirming the parameterisation spans the space correctly.

**A note on the graph itself.** Our region graph is built from **crime-pattern
correlation** (top-k similar regions), not physical adjacency. A
physically-grounded street-network graph is a natural next test — this is stated
as a limitation, not answered here.

---

## Part 3 — Physics-informed model

### 7. PINN with the Short et al. (2008) equations

**The physics.** Two coupled fields over the city:

- `A(x,y,t)` — attractiveness: how appealing a location is to burgle
- `rho(x,y,t)` — offender density

```
dA/dt   =  eta * Lap(A)  -  A  +  A0(x,y)  +  rho*A
drho/dt =  Lap(rho)  -  2*div( (rho/A) grad A )  -  rho*A  +  A  -  A0(x,y)
```

| Term | Meaning |
|---|---|
| `eta * Lap(A)` | attractiveness spreads to nearby locations |
| `- A` | it decays over time |
| `+ A0(x,y)` | each place has a static baseline appeal |
| `+ rho*A` | **a crime raises local attractiveness** (repeat victimisation) |
| `Lap(rho)` | offenders move randomly |
| `- 2*div((rho/A) grad A)` | offenders drift *up* the attractiveness gradient |
| `- rho*A` | an offender leaves after committing a crime |
| `+ A - A0` | replacement offenders enter |

The **observed** quantity is the crime rate `rho * A`.

**Verification of the transcription.** At a uniform steady state both residuals
are exactly zero when `rho*A = A - A0` — the two equations are mutually
consistent. The divergence expansion used in the code matches finite differences
to 3.9 × 10⁻⁵.

**The loss.**

```
Loss = Loss_data + lambda * Loss_physics

Loss_data    = Poisson NLL:  rate - count * log(rate),  rate = scale * rho * A
Loss_physics = mean(residual_A^2) + mean(residual_rho^2)
```

Residuals are evaluated at random collocation points **inside the city**
(sampling the bounding box would enforce burglary physics over Lake Michigan).
Derivatives come from automatic differentiation, not finite differences.

**Design choices that mattered:**

| Choice | Why | Effect |
|---|---|---|
| **Poisson likelihood** | counts are Poisson, not Gaussian; MSE over-weights busy cells and bakes a Head bias into the objective | Tail over-flagging 3.2× → 1.8× |
| **A0 as a spatial field** | Short et al. state A0 "is not necessarily uniform over the lattice grids" | gives the network a place to store geography, freeing the dynamics |
| **Collocation masked to the city** | otherwise physics is enforced over water | correctness |
| **Generic-smoothness ablation** | rules out "any regulariser would do" | the decisive control |

**Result** (5 seeds): +18.32 AUC over the capacity-matched control, t = 9.57.
Generic smoothness scored **44.63** — below the control *and* below chance —
which is what proves the burglary equations are doing specific work.

**Learned constants.** `eta ≈ 0.012–0.017`, consistently across seeds. A small
diffusion coefficient is consistent with criminology: near-repeat effects are
measured over metres and weeks, not kilometres and years.

**Limitation.** Per-group AUC is 51–58. Most of the pooled AUC comes from telling
neighbourhoods apart, not weeks apart. Run `src/pinn/signal_check.py` before
attempting to improve this — it measures whether week-to-week signal exists at
all.

---

## Metrics

| Metric | Base-rate dependent? | Used for |
|---|---|---|
| F1, precision, accuracy | **yes** | reported because the field reports them |
| AUC-ROC, balanced accuracy | **no** | the primary evidence |
| Equal opportunity, equalized odds | no | taxonomy in `src/fairaudit/metrics.py` |
| Demographic parity, predictive parity | **yes** | taxonomy |
| Skill score `(M − A)/(1 − A)` | corrected for base rate — **but not for threshold** | shown to be insufficient |

The last row is the paper's central methodological point: correcting a
threshold-dependent metric for base rate does **not** make it safe.
