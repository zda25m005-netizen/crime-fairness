# The findings, in one page

For someone who has five minutes.

---

## The problem

A model that has **zero skill** — one that predicts "crime" every day everywhere —
scores **66.7** in a busy area and **9.5** in a quiet one. That is a 57-point
"fairness gap" produced by a model that never reads the data.

So when a paper reports "69 in busy areas, 28 in quiet areas", most of that gap
was going to be there regardless of how good the model was.

*(This mathematics is known — Davis & Goadrich 2006, Chouldechova 2017. We cite
it; we do not claim it.)*

---

## Finding 1 — it is happening across the literature

Six published crime-prediction models. Reported scores correlate with crime
volume at **r = 0.87–0.90**, Fisher combined **p = 1.3 × 10⁻⁵**. Holds with no
distributional assumption at all.

---

## Finding 2 — correcting for it is not enough

We built the obvious correction. It reported our fairness gap closing from
**40.97 to −2.39** — apparently perfect.

It was wrong:

| | control | "improved" model |
|---|---|---|
| Tail F1 | 23.42 | **40.38** |
| Tail AUC | 62.03 | **58.02** |
| says "crime" | 23% | **80%** |

The model had become **worse at ranking**. It simply started predicting crime in
80% of poor neighbourhoods.

**Why the correction failed.** `2p/(1+p)` corrects for base rate. It does *not*
correct for a shifted decision threshold. Those are two different confounds.

---

## Finding 3 — no architecture needed

Take the model with **no graph**. Change **only the decision threshold**.

| | Tail F1 | says "crime" | Tail AUC |
|---|---|---|---|
| threshold 0.50 (default) | 27.96 | 22.7% | 61.99 |
| the graph model | 40.83 | 89.7% | 58.94 |
| **threshold 0.35** | **41.59** | 63.9% | **61.99** |

One knob **exceeds** the entire architectural intervention, with higher AUC and
less over-prediction. AUC reads 61.99 on all nineteen rows of the sweep.

---

## Finding 4 — the metric does not track model quality

Across four PINN configurations:

- **Real skill (AUC) spans 18.32 points**
- **The Head−Tail F1 gap spans 2.06 points**

Within the Tail group alone, F1 reads 43.64 / 43.36 / 43.45 — a spread of
**0.28** — while Tail AUC spans **7.2**.

And the sign reverses: F1 says Head is ~47 points better served; AUC says Head is
*worse* served. The two metrics disagree about **who is being failed**.

---

## What we recommend

1. Never report a threshold-dependent fairness gap alone.
2. Base-rate correction is **not** sufficient — state the operating point too.
3. Report an invariant metric (AUC, balanced accuracy) beside any F1 gap.
4. Report the positive-prediction rate per group.
5. Treat any gap closure not matched by an invariant metric as **unverified**.

---

## The positive result

A physics-informed model using the Short et al. (2008) burglary equations beat
its capacity-matched control by **+18.3 AUC** (t = 9.6, 5 seeds) and cut
over-flagging of poor neighbourhoods from **3.2× to 1.8×** reality.

A generic smoothness prior scored **44.63** — worse than no physics and below
chance — which is what shows the burglary equations are doing specific work.

**Limitation:** the model learned *where* burglary happens, not *when*.
