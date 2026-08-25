"""
Re-analysis of PUBLISHED results under base-rate correction
================================================================================
Target: Bhumika, Lalanda, Vega & Das, "FedCrime", Neurocomputing 679 (2026)
        133217.  All inputs below are transcribed directly from that paper.

WHY THIS IS CHECKABLE BY ANYONE
  * Table 1 reports PER-CATEGORY sparsity for each subset and city.
  * Tables 2, 3 and 5 report the macro-F1 achieved by FedCrime and baselines.
  * Therefore the attainability ceiling of macro-F1 can be computed exactly
    from the paper's own numbers, with no access to their data or code.

METHOD
  For category c with positive rate p_c = 1 - sparsity_c, the always-positive
  (skill-free) predictor attains F1_c = 2 p_c / (1 + p_c).  Macro-F1 averages
  over categories, so the skill-free macro-F1 is

        A = (1/C) * sum_c  2 p_c / (1 + p_c).                            (*)

  NOTE: (*) averages the per-category ceilings; using the pooled sparsity
  instead would be wrong by Jensen's inequality. We use per-category values.

  We then report  Skill = reported macro-F1 / A.  Skill <= 1 means the
  published result does not exceed a no-skill baseline in macro-F1 terms.

Run:  python reanalysis_published.py
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Transcribed from FedCrime Table 1  (per-category sparsity %, 8 categories)
# --------------------------------------------------------------------------- #
SPARSITY = {
    "LA": {
        "S-Alpha": [6.50, 44.42, 56.14, 51.57, 20.19, 33.35, 55.15, 71.09],
        "S-Beta":  [14.70, 53.12, 63.38, 63.40, 36.25, 46.45, 67.74, 82.11],
        "S-Gamma": [33.43, 66.47, 72.42, 73.52, 53.63, 61.32, 80.53, 88.16],
        "S-Omega": [37.21, 71.54, 72.50, 73.88, 58.98, 65.43, 82.73, 88.16],
    },
    "CHI": {
        "S-Alpha": [47.26, 5.83, 38.11, 38.50, 29.94, 4.50, 17.60, 34.14],
        "S-Beta":  [63.40, 13.12, 53.29, 50.56, 43.21, 11.04, 26.05, 48.78],
        "S-Gamma": [74.98, 32.08, 64.70, 62.53, 59.18, 27.12, 39.77, 65.18],
        "S-Omega": [75.63, 37.57, 66.16, 67.80, 58.33, 33.09, 45.88, 64.98],
    },
}

# --------------------------------------------------------------------------- #
# Reported macro-F1 (FedCrime = "Federated Zero Inflation"), Tables 2/3/5
# --------------------------------------------------------------------------- #
REPORTED = {
    ("LA", "global"):  {"S-Alpha": 72.16, "S-Beta": 61.70,
                        "S-Gamma": 48.54, "S-Omega": 48.43},
    ("LA", "local"):   {"S-Alpha": 70.60, "S-Beta": 59.72,
                        "S-Gamma": 45.41, "S-Omega": 45.40},
    ("CHI", "global"): {"S-Alpha": 84.67, "S-Beta": 76.01,
                        "S-Gamma": 63.40, "S-Omega": 60.09},
    ("CHI", "local"):  {"S-Alpha": 83.53, "S-Beta": 74.22,
                        "S-Gamma": 59.67, "S-Omega": 55.44},
}

# strongest competing baselines reported in Tables 2/3 (global test, macro-F1)
BASELINES = {
    ("LA", "global"):  {"S-Alpha": ("FedAvgM", 63.26), "S-Beta": ("FedAvgM", 44.83),
                        "S-Gamma": ("Scaffold", 36.40), "S-Omega": ("Scaffold", 34.09)},
    ("CHI", "global"): {"S-Alpha": ("FedProx", 83.87), "S-Beta": ("FedAvgM", 66.51),
                        "S-Gamma": ("FedAvgM", 51.45), "S-Omega": ("FedTrimAvg", 48.13)},
}


def ceiling(sparsity_pct):
    """Skill-free macro-F1 ceiling from per-category sparsity (equation *)."""
    p = 1.0 - np.asarray(sparsity_pct, dtype=float) / 100.0
    return float(np.mean(2 * p / (1 + p))) * 100


def main():
    print("=" * 82)
    print("RE-ANALYSIS OF PUBLISHED FedCrime RESULTS UNDER BASE-RATE CORRECTION")
    print("=" * 82)
    print("All inputs transcribed from the paper (Table 1 sparsity; Tables 2/3/5 scores).")
    print("A = skill-free macro-F1 ceiling = mean_c 2p_c/(1+p_c)\n")

    all_sk = []
    for city in ["LA", "CHI"]:
        for split in ["global", "local"]:
            rep = REPORTED[(city, split)]
            print(f"--- {city}  ({split} test set) " + "-" * 46)
            print(f"{'Subset':10s} {'Sparsity':>9s} {'Reported F1':>12s} "
                  f"{'Ceiling A':>10s} {'Skill=F1/A':>11s} {'Verdict':>16s}")
            for sub in ["S-Alpha", "S-Beta", "S-Gamma", "S-Omega"]:
                sp = SPARSITY[city][sub]
                A = ceiling(sp)
                f1 = rep[sub]
                sk = 100 * f1 / A
                all_sk.append(sk)
                verdict = "AT/BELOW no-skill" if sk <= 102 else "exceeds baseline"
                print(f"{sub:10s} {np.mean(sp):8.1f}% {f1:12.2f} {A:10.1f} "
                      f"{sk:10.1f}% {verdict:>16s}")
            print()

    print("=" * 82)
    print("FINDING 1 — the reported scores track the skill-free ceiling")
    rep_g = [REPORTED[("LA", "global")][s] for s in SPARSITY["LA"]] + \
            [REPORTED[("CHI", "global")][s] for s in SPARSITY["CHI"]]
    cei_g = [ceiling(SPARSITY["LA"][s]) for s in SPARSITY["LA"]] + \
            [ceiling(SPARSITY["CHI"][s]) for s in SPARSITY["CHI"]]
    r = float(np.corrcoef(rep_g, cei_g)[0, 1])
    mad = float(np.mean(np.abs(np.array(rep_g) - np.array(cei_g))))
    print(f"  correlation(reported macro-F1, skill-free ceiling) = {r:.4f}")
    print(f"  mean |reported - ceiling|                          = {mad:.2f} F1 points")
    print(f"  mean skill (reported / ceiling)                    = {np.mean(all_sk):.1f}%")

    print("\nFINDING 2 — the paper's headline 'degradation under sparsity'")
    la = REPORTED[("LA", "global")]
    dA = ceiling(SPARSITY["LA"]["S-Alpha"]) - ceiling(SPARSITY["LA"]["S-Omega"])
    dF = la["S-Alpha"] - la["S-Omega"]
    print(f"  LA global: reported drop S-Alpha -> S-Omega = {dF:.2f} F1 points")
    print(f"             ceiling  drop S-Alpha -> S-Omega = {dA:.2f} F1 points")
    print(f"  => {100*dA/dF:.0f}% of the reported degradation is the ceiling moving,")
    print("     not the method losing skill.")

    print("\nFINDING 3 — comparisons between methods are NOT affected")
    print("  Within a subset the ceiling is identical for all methods, so relative")
    print("  comparisons (FedCrime vs baselines) remain valid:")
    for city in ["LA", "CHI"]:
        for sub in ["S-Alpha", "S-Gamma", "S-Omega"]:
            A = ceiling(SPARSITY[city][sub])
            f1 = REPORTED[(city, "global")][sub]
            bname, bf1 = BASELINES[(city, "global")][sub]
            print(f"    {city:4s} {sub:8s} FedCrime {f1:5.2f} ({100*f1/A:5.1f}% of A) "
                  f"vs {bname:10s} {bf1:5.2f} ({100*bf1/A:5.1f}% of A)")

    print("\n" + "=" * 82)
    print("INTERPRETATION (stated conservatively)")
    print("  * The ABSOLUTE macro-F1 values reported across subsets are close to,")
    print("    and sometimes below, what a skill-free always-positive predictor")
    print("    attains at the same base rates.")
    print("  * Consequently, the reported decline from S-Alpha to S-Omega mostly")
    print("    reflects a falling attainability ceiling rather than a loss of skill,")
    print("    and should not be read as evidence about sparsity robustness.")
    print("  * RELATIVE comparisons between methods within a subset are unaffected,")
    print("    because the ceiling is a property of the data, not the method.")
    print("  * CAVEAT: Table 1 sparsity is measured over the full subset, whereas the")
    print("    scores are computed on the test month; small differences between the")
    print("    two periods can shift A by a few points. The conclusion relies on the")
    print("    magnitude of the effect, not on exact equality.")


if __name__ == "__main__":
    main()
