"""
Downstream decision impact: does the metric choice change who gets resources?
================================================================================
A fairness audit is only consequential if it changes DECISIONS.  Two decisions
are common in deployed crime-prediction systems:

  D1  MODEL SELECTION — pick the best model per region group.
  D2  REMEDIATION TARGETING — rank regions by "how badly the model serves them"
      and direct extra effort (data collection, retraining, human review) to the
      worst-served ones.

Both are usually driven by raw F1.  We compare the decision made under raw F1
against the decision made under a base-rate-corrected criterion, and report how
often they disagree.

This uses ONLY published numbers (no model training required).

Run:  python decision_impact.py
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Benchmark data (arXiv:2407.19324, Tables 6 & 8) — 6 models x 5 area groups
# --------------------------------------------------------------------------- #
GROUPS = ["Very Small", "Small", "Medium", "Large", "Very Large"]
N_COMMUNITIES = np.array([13, 17, 15, 18, 14])
N_CRIMES = np.array([18070, 47813, 52979, 78933, 63409])
CRIMES_PER_COMM = N_CRIMES / N_COMMUNITIES
N_CATEGORIES, N_DAYS = 4, 365

MACRO_F1 = {
    "DeepCrime":       [0.24, 0.30, 0.33, 0.38, 0.42],
    "MiST":            [0.18, 0.22, 0.28, 0.31, 0.35],
    "CrimeForecaster": [0.20, 0.39, 0.38, 0.47, 0.43],
    "HAGEN":           [0.25, 0.36, 0.37, 0.42, 0.41],
    "ST-HSL":          [0.39, 0.37, 0.29, 0.32, 0.38],
    "AIST":            [0.46, 0.50, 0.48, 0.52, 0.61],
}


def base_rate():
    lam = CRIMES_PER_COMM / (N_DAYS * N_CATEGORIES)
    return 1.0 - np.exp(-lam)


def attainable(p):
    return 2 * p / (1 + p)


def skill_score(f1, A, perfect=1.0):
    """(F1 - A) / (1 - A): 0 = no better than skill-free, 1 = perfect."""
    return (np.asarray(f1) - A) / (perfect - A)


def main():
    p = base_rate()
    A = attainable(p)

    print("=" * 78)
    print("DECISION IMPACT — does correcting the metric change what we would do?")
    print("=" * 78)

    # ------------------------------------------------------------ D1
    print("\nD1. MODEL SELECTION — which model is 'best' for each group?")
    print(f"{'Group':12s} {'best by raw F1':>18s} {'best by skill score':>21s} "
          f"{'agree?':>8s}")
    print("-" * 64)
    names = list(MACRO_F1)
    disagree_d1 = 0
    for i, g in enumerate(GROUPS):
        raw = [MACRO_F1[m][i] for m in names]
        sk = [skill_score(MACRO_F1[m][i], A[i]) for m in names]
        b_raw, b_sk = names[int(np.argmax(raw))], names[int(np.argmax(sk))]
        ok = b_raw == b_sk
        disagree_d1 += (not ok)
        print(f"{g:12s} {b_raw:>18s} {b_sk:>21s} {('yes' if ok else 'NO'):>8s}")
    print(f"\n  -> model selection changes in {disagree_d1}/{len(GROUPS)} groups.")
    print("     (Expected: within a group the ceiling is shared, so ranking is")
    print("      preserved. This is a NEGATIVE result and it is important: the")
    print("      confound does NOT invalidate model comparisons.)")

    # ------------------------------------------------------------ D2
    print("\nD2. REMEDIATION TARGETING — which group is worst served, per model?")
    print("    (this is where resources / extra data collection would be sent)")
    print(f"{'Model':18s} {'worst by raw F1':>17s} {'worst by skill':>16s} "
          f"{'agree?':>8s}")
    print("-" * 62)
    disagree_d2, flips = 0, []
    for m in names:
        raw = np.array(MACRO_F1[m])
        sk = skill_score(raw, A)
        w_raw, w_sk = GROUPS[int(np.argmin(raw))], GROUPS[int(np.argmin(sk))]
        ok = w_raw == w_sk
        disagree_d2 += (not ok)
        if not ok:
            flips.append((m, w_raw, w_sk))
        print(f"{m:18s} {w_raw:>17s} {w_sk:>16s} {('yes' if ok else 'NO'):>8s}")
    print(f"\n  -> remediation target changes in {disagree_d2}/{len(names)} models.")
    for m, a, b in flips:
        print(f"     {m}: raw F1 sends help to '{a}', skill score sends it to '{b}'")

    # ------------------------------------------------------------ magnitude
    print("\nRANK CORRELATION between the two criteria (per model, across groups)")
    print(f"{'Model':18s} {'Spearman rho':>14s}")
    print("-" * 34)
    rhos = []
    for m in names:
        raw = np.array(MACRO_F1[m]); sk = skill_score(raw, A)
        r1 = np.argsort(np.argsort(raw)); r2 = np.argsort(np.argsort(sk))
        rho = float(np.corrcoef(r1, r2)[0, 1])
        rhos.append(rho)
        print(f"{m:18s} {rho:14.3f}")
    print(f"{'MEAN':18s} {np.mean(rhos):14.3f}")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("  * Model COMPARISON is robust to the confound (D1): within a group all")
    print("    models share the same ceiling, so their ranking is unchanged.")
    print(f"  * Group TARGETING is not (D2): the identified worst-served group")
    print(f"    changes for {disagree_d2}/{len(names)} models once the ceiling is removed.")
    print("    Under raw F1, the sparsest group looks worst almost by construction,")
    print("    so remediation is systematically directed at low-crime regions")
    print("    regardless of whether the model actually serves them poorly.")
    print("\n  This is the practical cost of the confound: it does not mislead us")
    print("  about which METHOD to use, but it does mislead us about WHERE the")
    print("  model is failing -- which is precisely the fairness question.")

    print("\nCAVEATS")
    print("  * Base rates are estimated from published crime totals (Poisson")
    print("    assumption); see literature_audit.py for the same caveat.")
    print("  * D2 assumes remediation targets the worst-served group; other")
    print("    policies (e.g. proportional allocation) would be affected")
    print("    differently, though the ordering issue is the same.")


if __name__ == "__main__":
    main()
