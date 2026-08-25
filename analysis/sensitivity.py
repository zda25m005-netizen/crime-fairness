"""
Robustness of the literature-audit correlation  (answers reviewer W2 and W3)
================================================================================
W2: base rates are ESTIMATED from published crime totals under a Poisson
    assumption.  If the headline correlation depends on that assumption, the
    claim is fragile.  We therefore recompute it under six different
    assumptions, including ones that make no distributional assumption at all.

W3: n is small (6 models x 5 area groups).  A Pearson r on 5 points per model
    is easy to over-read, so we add a permutation test and a bootstrap CI
    rather than reporting r alone.

Run:  python sensitivity.py
"""
from __future__ import annotations
import numpy as np

rng = np.random.default_rng(0)

GROUPS = ["Very Small", "Small", "Medium", "Large", "Very Large"]
N_COMMUNITIES = np.array([13, 17, 15, 18, 14])
N_CRIMES = np.array([18070, 47813, 52979, 78933, 63409])
CPC = N_CRIMES / N_COMMUNITIES              # crimes per community per year
N_CAT, N_DAYS = 4, 365

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

lam = CPC / (N_DAYS * N_CAT)                # mean events per cell per day

# ---------------------------------------------------------------- W2 variants
def nb_rate(lam, k):
    """Negative-binomial P(at least one event); k = dispersion. k->inf = Poisson.
    Crime clusters in time, so real data is OVER-dispersed relative to Poisson."""
    return 1.0 - (k / (k + lam)) ** k

ASSUMPTIONS = {
    "Poisson (as submitted)":        1 - np.exp(-lam),
    "Neg-binomial k=5 (clustered)":  nb_rate(lam, 5.0),
    "Neg-binomial k=1 (very clustered)": nb_rate(lam, 1.0),
    "Raw rate, no distribution":     lam,
    "Log crime volume":              np.log(CPC),
    "RANK of crime volume only":     np.argsort(np.argsort(CPC)).astype(float),
}

def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)

print("=" * 78)
print("W2 — DOES THE RESULT DEPEND ON THE POISSON ASSUMPTION?")
print("=" * 78)
print("Mean correlation between reported F1 and the base-rate proxy, over 6 models\n")
print(f"{'assumption for base rate':>34} {'macro-F1':>10} {'micro-F1':>10}")
print("-" * 58)
for name, x in ASSUMPTIONS.items():
    ma = np.mean([pearson(x, v) for v in MACRO_F1.values()])
    mi = np.mean([pearson(x, v) for v in MICRO_F1.values()])
    print(f"{name:>34} {ma:10.3f} {mi:10.3f}")
print("\nThe last two rows use NO distributional assumption at all -- 'rank of")
print("crime volume' only needs the ORDER of the groups, which is given directly")
print("in the source papers. If the finding survives there, it does not depend")
print("on how we estimated the base rate.")

# ---------------------------------------------------------------- W3 inference
print("\n" + "=" * 78)
print("W3 — IS THE CORRELATION REAL, GIVEN ONLY 5 POINTS PER MODEL?")
print("=" * 78)

x = ASSUMPTIONS["Poisson (as submitted)"]

def perm_p(x, y, n=200_000):
    """Exact-ish permutation test: how often does a random re-ordering of the
    5 groups produce a correlation at least this strong?"""
    obs = abs(pearson(x, y))
    idx = np.array([rng.permutation(len(y)) for _ in range(n)])
    perms = np.asarray(y)[idx]
    xm = x - x.mean()
    num = perms @ xm
    den = np.sqrt(((perms - perms.mean(1, keepdims=True)) ** 2).sum(1) * (xm ** 2).sum())
    return float(np.mean(np.abs(num / den) >= obs - 1e-12))

print(f"\n{'model':>18} {'r (macro)':>10} {'perm p':>9}   {'r (micro)':>10} {'perm p':>9}")
print("-" * 64)
pa_all, pi_all = [], []
for m in MACRO_F1:
    ya, yi = np.array(MACRO_F1[m]), np.array(MICRO_F1[m])
    ra, ri = pearson(x, ya), pearson(x, yi)
    pa, pi = perm_p(x, ya), perm_p(x, yi)
    pa_all.append(pa); pi_all.append(pi)
    print(f"{m:>18} {ra:10.3f} {pa:9.4f}   {ri:10.3f} {pi:9.4f}")

# pool the 6 models: Fisher's method on the micro-F1 p-values
from math import log
chi2 = -2 * sum(log(max(p, 1e-12)) for p in pi_all)
df = 2 * len(pi_all)
# survival function of chi2 without scipy
def chi2_sf(c, k):
    if k % 2: raise ValueError
    m = k // 2
    t = np.exp(-c / 2); s = t
    for i in range(1, m):
        t *= (c / 2) / i; s += t
    return float(s)
print(f"\nFisher combined test over the 6 models (micro-F1): "
      f"chi2 = {chi2:.1f}, df = {df}, p = {chi2_sf(chi2, df):.2e}")

# pooled correlation with bootstrap CI over MODELS (the unit of generalisation)
names = list(MICRO_F1)
def pooled(sample):
    return float(np.mean([pearson(x, MICRO_F1[n]) for n in sample]))
obs = pooled(names)
boot = [pooled(list(rng.choice(names, len(names), replace=True))) for _ in range(20000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"Pooled micro-F1 correlation = {obs:+.3f}   "
      f"95% bootstrap CI over models [{lo:+.3f}, {hi:+.3f}]")

print("\n" + "=" * 78)
print("HOW TO REPORT THIS")
print("  * Do not lead with a single Pearson r on 5 points.")
print("  * Lead with the RANK-based result: it needs only the ordering of the")
print("    groups, which the source papers state directly, so it is assumption-free.")
print("  * Report the permutation p per model, the Fisher combined p, and the")
print("    bootstrap CI across models -- models are the unit of generalisation.")
print("  * ST-HSL is the negative control and should be reported, not hidden.")
print("=" * 78)
