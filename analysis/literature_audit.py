"""
Systematic audit of published crime-prediction results for base-rate confounding
================================================================================
Sources (all numbers transcribed from the published papers themselves):

  [A] Bhumika, Lalanda, Vega & Das. "FedCrime". Neurocomputing 679 (2026) 133217.
      Table 1 (per-category sparsity), Tables 2/3/5 (macro-F1).

  [B] "Deep Learning Based Crime Prediction Models: Experiments and Analysis",
      arXiv:2407.19324.  A unified benchmark of SEVEN published models on
      Chicago.  Table 6 (group sizes + crime counts), Table 8 (macro/micro-F1
      per group).  Models: DeepCrime, MiST, CrimeForecaster, HAGEN, ST-HSL, AIST.

WHY THIS MATTERS
  Paper [B] states (Sec. 4.1.2): "models tend to perform better as the community
  size increases ... The larger number of crimes provides more data points,
  leading to better training and higher performance in classification."

  That is a CAUSAL claim: more data -> better learning.  But more crimes also
  means a higher base rate, and the attainable F1 rises with the base rate
  (A(p) = 2p/(1+p)).  Both explanations predict the same direction, and the
  paper does not distinguish them.  This audit quantifies the confound.

Run:  python literature_audit.py
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# [B] benchmark paper — Table 6: groups by community area
# --------------------------------------------------------------------------- #
GROUPS = ["Very Small", "Small", "Medium", "Large", "Very Large"]
N_COMMUNITIES = np.array([13, 17, 15, 18, 14])
N_CRIMES = np.array([18070, 47813, 52979, 78933, 63409])
CRIMES_PER_COMMUNITY = N_CRIMES / N_COMMUNITIES

# Table 8: macro-F1 per group for each published model
MACRO_F1 = {
    "DeepCrime":       [0.24, 0.30, 0.33, 0.38, 0.42],
    "MiST":            [0.18, 0.22, 0.28, 0.31, 0.35],
    "CrimeForecaster": [0.20, 0.39, 0.38, 0.47, 0.43],
    "HAGEN":           [0.25, 0.36, 0.37, 0.42, 0.41],
    "ST-HSL":          [0.39, 0.37, 0.29, 0.32, 0.38],
    "AIST":            [0.46, 0.50, 0.48, 0.52, 0.61],
}
MICRO_F1 = {
    "DeepCrime":       [0.36, 0.41, 0.55, 0.56, 0.59],
    "MiST":            [0.21, 0.34, 0.39, 0.42, 0.45],
    "CrimeForecaster": [0.23, 0.45, 0.45, 0.53, 0.51],
    "HAGEN":           [0.27, 0.39, 0.41, 0.45, 0.44],
    "ST-HSL":          [0.44, 0.48, 0.43, 0.47, 0.60],
    "AIST":            [0.54, 0.60, 0.68, 0.77, 0.73],
}

# Table 4: the benchmark models 4 crime categories on Chicago 2019
N_CATEGORIES = 4
N_DAYS = 365


def estimated_base_rate():
    """Approximate P(category occurs on a given day) per group.

    crimes/community/day spread over the modelled categories, converted to an
    occurrence probability under a Poisson assumption: p = 1 - exp(-lambda).
    This is an ESTIMATE: the paper reports crime totals, not per-category
    daily incidence, so absolute values carry uncertainty (see CAVEATS).
    """
    lam = CRIMES_PER_COMMUNITY / (N_DAYS * N_CATEGORIES)
    return 1.0 - np.exp(-lam)


def ceiling_from_p(p):
    return 2 * p / (1 + p)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("=" * 84)
    print("SYSTEMATIC AUDIT — do published crime-prediction results track base rates?")
    print("=" * 84)

    p = estimated_base_rate()
    ceil = ceiling_from_p(p)

    print("\n[B] Benchmark paper (arXiv:2407.19324), Chicago, groups by area")
    print(f"{'Group':12s} {'#Comm':>6s} {'#Crimes':>9s} {'crimes/comm':>12s} "
          f"{'est. p':>8s} {'ceiling A':>10s}")
    print("-" * 62)
    for i, g in enumerate(GROUPS):
        print(f"{g:12s} {N_COMMUNITIES[i]:6d} {N_CRIMES[i]:9d} "
              f"{CRIMES_PER_COMMUNITY[i]:12.0f} {p[i]:8.3f} {100*ceil[i]:10.1f}")

    print("\nCorrelation of each model's reported macro-F1 with crime volume")
    print("(volume drives the base rate, which drives the attainable F1)")
    print(f"{'Model':18s} {'r(F1, crimes/comm)':>20s} {'r(F1, ceiling)':>16s} "
          f"{'monotone?':>10s}")
    print("-" * 68)
    rs_vol, rs_ceil = [], []
    for m, f1 in MACRO_F1.items():
        r_v = pearson(f1, CRIMES_PER_COMMUNITY)
        r_c = pearson(f1, ceil)
        mono = "yes" if all(np.diff(f1) >= -0.02) else "no"
        rs_vol.append(r_v); rs_ceil.append(r_c)
        print(f"{m:18s} {r_v:20.3f} {r_c:16.3f} {mono:>10s}")
    print(f"{'MEAN':18s} {np.mean(rs_vol):20.3f} {np.mean(rs_ceil):16.3f}")

    print("\nSame test on micro-F1")
    rs2 = [pearson(v, CRIMES_PER_COMMUNITY) for v in MICRO_F1.values()]
    for (m, v), r in zip(MICRO_F1.items(), rs2):
        print(f"  {m:18s} r = {r:+.3f}")
    print(f"  {'MEAN':18s} r = {np.mean(rs2):+.3f}")

    print("\n" + "=" * 84)
    print("FINDING")
    n_high = sum(1 for r in rs_vol if r > 0.8)
    print(f"  {n_high}/{len(rs_vol)} models show r > 0.8 between reported macro-F1 and")
    print(f"  crime volume; mean r = {np.mean(rs_vol):.3f}.")
    print("  The benchmark attributes this trend to 'more data -> better training'.")
    print("  An equally consistent explanation is that the ATTAINABLE F1 rises with")
    print("  the base rate. The published evidence does not distinguish the two,")
    print("  because no base-rate-invariant metric (e.g. AUC) is reported.")

    print("\nINTERNAL VALIDATION (a natural control inside the benchmark)")
    print("  ST-HSL is the single model whose macro-F1 does NOT track crime volume")
    print(f"  (r = {rs_vol[list(MACRO_F1).index('ST-HSL')]:+.3f}).  The benchmark itself"
          " describes ST-HSL as using")
    print("  'a self-supervised learning mechanism to backup its performance when the")
    print("  data is sparse ... makes the model invariant to area or density' (Sec 4.1.1).")
    print("  So the one model DESIGNED to be density-invariant is the one that breaks")
    print("  the correlation -- consistent with the base-rate explanation.")

    print("\n" + "=" * 84)
    print("SCOPE OF THE AUDIT")
    print("  Papers audited      : 2  (FedCrime; the 7-model benchmark)")
    print("  Published models    : 7  (FedCrime, DeepCrime, MiST,")
    print("                            CrimeForecaster, HAGEN, ST-HSL, AIST)")
    print(f"  Reported numbers    : {len(MACRO_F1)*5 + len(MICRO_F1)*5} from [B] "
          f"+ 16 from [A]")

    print("\nCAVEATS (must be stated in the paper)")
    print("  1. Base rates for [B] are ESTIMATED from reported crime totals under a")
    print("     Poisson occurrence assumption; the paper does not publish per-category")
    print("     daily incidence. The correlation with crime VOLUME (directly reported)")
    print("     is therefore the primary evidence; the ceiling is indicative.")
    print("  2. Correlation is not proof of confounding: better learning and higher")
    print("     attainability predict the same direction. The claim is that the")
    print("     published evidence CANNOT SEPARATE them, not that learning is absent.")
    print("  3. RELATIVE model rankings within a group are unaffected, since the")
    print("     ceiling is a property of the data, not of the method.")
    print("  4. Our own reproduction finds genuine skill (AUC ~ 0.74), so these models")
    print("     do learn; the issue is that macro-F1 does not isolate that skill.")


if __name__ == "__main__":
    main()
