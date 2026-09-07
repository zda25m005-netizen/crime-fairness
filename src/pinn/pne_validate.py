"""Does Anscombe + a global kappa really equalise base rates across groups
with very different baseline intensities? Test before trusting the method."""
import numpy as np
rng = np.random.default_rng(0)

def anscombe(y, mu0, floor=0.05):
    mu0 = np.maximum(np.asarray(mu0,float), floor); y=np.asarray(y,float)
    return 1.5*(y**(2/3) - mu0**(2/3))/(mu0**(1/6))

# 3 groups of cells with very different baseline rates, like Head/Mid/Tail
WEEKS = 60
spec = [("Head", 40, 3.0), ("Mid", 60, 0.9), ("Tail", 100, 0.15)]
mu0_cell, g_cell = [], []
for name, ncell, lam in spec:
    mu0_cell.append(rng.gamma(shape=8, scale=lam/8, size=ncell))  # cell-level variation
    g_cell.append(np.full(ncell, name))
mu0_cell = np.concatenate(mu0_cell); g_cell = np.concatenate(g_cell)

counts = rng.poisson(np.tile(mu0_cell, WEEKS))
mu0 = np.tile(mu0_cell, WEEKS); g = np.tile(g_cell, WEEKS)

print("="*70); print("BASELINE INTENSITIES BY GROUP"); print("="*70)
for k in ("Head","Mid","Tail"):
    m = g_cell==k
    print(f"  {k:>5}  mean mu0 {mu0_cell[m].mean():6.3f}   {m.sum()} cells")

def rates(y):
    return {k: 100*y[g==k].mean() for k in ("Head","Mid","Tail")}

print("\n"+"="*70); print("ARM 1 — RAW TARGET  (count > 0)"); print("="*70)
r = rates((counts>0).astype(int))
for k,v in r.items(): print(f"  {k:>5} base rate {v:6.2f}%")
print(f"  SPREAD Head-Tail: {r['Head']-r['Tail']:+.2f} points   <-- the confound")

target = float((counts>0).mean())
print("\n"+"="*70); print(f"ARM 3 — ANSCOMBE-NORMALISED  (kappa set to match {100*target:.1f}%)"); print("="*70)
z = anscombe(counts, mu0)
kappa = float(np.quantile(z, 1-target))
r2 = rates((z>kappa).astype(int))
for k,v in r2.items(): print(f"  {k:>5} base rate {v:6.2f}%")
print(f"  SPREAD Head-Tail: {r2['Head']-r2['Tail']:+.2f} points")

print("\n"+"="*70); print("VERDICT"); print("="*70)
before = abs(r['Head']-r['Tail']); after = abs(r2['Head']-r2['Tail'])
print(f"  base-rate gap  {before:.2f}  ->  {after:.2f}   "
      f"({100*(1-after/before):.0f}% removed)")
if after < 5:
    print("  MECHANISM WORKS: base rates are near-equal after normalisation.")
else:
    print("  WARNING: residual gap remains. Likely cause: Poisson discreteness")
    print("  at very low mu0 -- most Tail cells are 0 or 1, so a continuous")
    print("  quantile cannot land on the target rate. Report this honestly.")

# how bad is the discreteness?
print("\n  diagnostic — share of cell-weeks with count 0:")
for k in ("Head","Mid","Tail"):
    m=g==k; print(f"    {k:>5} {100*(counts[m]==0).mean():5.1f}%   distinct z values: {len(np.unique(z[m]))}")
