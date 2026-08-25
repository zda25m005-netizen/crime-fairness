r"""
Extending the attainability decomposition beyond binary F1.
================================================================================
The core result (binary case) is

        Gap_M = [A_M(p_H) - A_M(p_T)]  +  skill term,     A_F1(p) = 2p/(1+p).

This module derives and NUMERICALLY VERIFIES the analogous confound for three
further settings that are common in spatio-temporal prediction.

--------------------------------------------------------------------------------
E1. MULTI-LABEL (macro-averaged) metrics
--------------------------------------------------------------------------------
With C labels of rates p_1..p_C, macro-F1 attainability is the MEAN of the
per-label ceilings:

        A_macro(p_1..p_C) = (1/C) sum_c 2 p_c / (1 + p_c).

IMPORTANT: this is NOT 2 pbar/(1 + pbar) for the pooled rate pbar. Because
x -> 2x/(1+x) is concave, Jensen gives

        (1/C) sum_c A(p_c)  <=  A(pbar),

so using the pooled rate OVERSTATES the ceiling. Papers reporting only pooled
sparsity therefore cannot recover the correct ceiling; per-label rates are
required.

--------------------------------------------------------------------------------
E2. RANKING metrics (Precision@k, NDCG@k)
--------------------------------------------------------------------------------
For a skill-free ranker over N items with positive rate p, the expected
Precision@k is p for every k. Hence

        A_P@k(p) = p,

which is linear (not concave) in p, and the group gap under equal skill is
exactly p_H - p_T. Ranking metrics are therefore ALSO confounded, and the
confound is larger than for F1 in the low-rate regime (since 2p/(1+p) > p).

--------------------------------------------------------------------------------
E3. REGRESSION metrics (MAE, RMSE)
--------------------------------------------------------------------------------
For counts with mean mu and variance sigma^2, a skill-free predictor emitting
the group mean attains

        A_MAE  = E|Y - mu|,      A_RMSE = sigma.

For Poisson counts sigma = sqrt(mu), so BOTH error metrics grow with the group
mean. Reporting raw MAE/RMSE gaps across groups with different intensities is
confounded in the OPPOSITE direction to F1: busier groups look WORSE.
This explains a pattern noted in the crime-prediction benchmark literature,
where models appear to degrade on high-volume regions under MAE/RMSE while
appearing to improve on them under F1.

Normalised alternatives (relative error MAE/mu, or a skill score
1 - MAE/A_MAE) remove the scale term.

Run:  python -m fairaudit.extensions
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
def a_macro_f1(rates):
    r = np.asarray(rates, dtype=float)
    return float(np.mean(2 * r / (1 + r)))


def a_pooled_f1(rates):
    pbar = float(np.mean(rates))
    return 2 * pbar / (1 + pbar)


def a_precision_at_k(p):
    return float(p)


def a_mae_poisson(mu, n=200_000):
    y = rng.poisson(mu, n)
    return float(np.mean(np.abs(y - mu)))


def a_rmse_poisson(mu):
    return float(np.sqrt(mu))


def hdr(t):
    print("\n" + "=" * 76); print(t); print("=" * 76)


def main():
    # ---------------------------------------------------------------- E1
    hdr("E1 — MULTI-LABEL: per-label mean vs pooled rate (Jensen gap)")
    cases = {
        "LA S-Omega (FedCrime Tab.1)": [1 - s / 100 for s in
                                        [37.21, 71.54, 72.50, 73.88,
                                         58.98, 65.43, 82.73, 88.16]],
        "CHI S-Alpha (FedCrime Tab.1)": [1 - s / 100 for s in
                                         [47.26, 5.83, 38.11, 38.50,
                                          29.94, 4.50, 17.60, 34.14]],
    }
    print(f"{'case':32s} {'correct A':>10s} {'pooled A':>10s} {'overstatement':>14s}")
    print("-" * 70)
    for name, rates in cases.items():
        a, ap = 100 * a_macro_f1(rates), 100 * a_pooled_f1(rates)
        print(f"{name:32s} {a:10.2f} {ap:10.2f} {ap - a:+14.2f}")
    print("\n  Using the pooled rate overstates the ceiling (Jensen). Papers that")
    print("  publish only aggregate sparsity cannot recover the correct ceiling.")

    # ---------------------------------------------------------------- E2
    hdr("E2 — RANKING: Precision@k is confounded, and worse than F1 at low p")
    print(f"{'p':>6} {'A_P@k = p':>11} {'A_F1 = 2p/(1+p)':>17} {'ratio':>8}")
    print("-" * 46)
    for p in [0.05, 0.10, 0.16, 0.30, 0.50]:
        af1 = 2 * p / (1 + p)
        print(f"{p:6.2f} {p:11.3f} {af1:17.3f} {af1 / p:8.2f}")
    print("\n  Verification (skill-free ranker, N=20000, k=100, 200 trials):")
    for p in [0.05, 0.20, 0.50]:
        vals = []
        for _ in range(200):                        # average away sampling noise
            y = (rng.random(20000) < p).astype(int)
            s = rng.random(20000)                   # scores independent of y
            vals.append(y[np.argsort(-s)[:100]].mean())
        print(f"    p={p:.2f}  empirical P@100 = {np.mean(vals):.3f} "
              f"+/- {np.std(vals):.3f}  (theory {p:.3f})")

    # ---------------------------------------------------------------- E3
    hdr("E3 — REGRESSION: MAE/RMSE grow with the group mean (opposite direction)")
    print(f"{'mu':>6} {'A_MAE':>9} {'A_RMSE':>9} {'relative MAE (A/mu)':>21}")
    print("-" * 48)
    for mu in [0.5, 1.0, 2.0, 5.0, 10.0]:
        am, ar = a_mae_poisson(mu), a_rmse_poisson(mu)
        print(f"{mu:6.1f} {am:9.3f} {ar:9.3f} {am / mu:21.3f}")
    print("\n  A skill-free predictor's error RISES with intensity, so busy regions")
    print("  look worse under raw MAE/RMSE while looking better under F1. Both are")
    print("  attainability effects, not skill differences.")

    hdr("SUMMARY — attainability by metric family")
    print("  binary F1        A(p) = 2p/(1+p)          confounded (concave)")
    print("  macro-F1         mean_c 2p_c/(1+p_c)      confounded; needs per-label p")
    print("  precision        A(p) = p                 confounded (linear)")
    print("  Precision@k      A(p) = p                 confounded (linear)")
    print("  MAE (Poisson)    A(mu) = E|Y-mu|          confounded, increasing in mu")
    print("  RMSE (Poisson)   A(mu) = sqrt(mu)         confounded, increasing in mu")
    print("  AUC / bal. acc.  A = 0.5                  INVARIANT")


if __name__ == "__main__":
    main()
