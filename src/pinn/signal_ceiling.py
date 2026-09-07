"""
Is there ANY week-to-week signal, once geography is removed?
================================================================================
WHY THIS SCRIPT EXISTS
----------------------
On the base-rate-free target, the PINN scored AUC 54.97 / 52.87 / 40.39
(Head / Mid / Tail). Barely better than a coin flip, and below one in the
sparsest group. Two possible reasons:

    (a) the model is too weak          -> a better model would help
    (b) there is no weekly signal      -> NO model will help

This script decides between them, and it trains nothing. It asks how well the
simplest possible predictors do on the same fair target. If none of them beats
50, the ceiling is the DATA, not the architecture, and every further modelling
attempt is wasted effort.

THE PREDICTORS
--------------
    constant        sanity check -- must come out at 50
    persistence     last week's excess in this same cell
    AR(1)           last week's excess, scaled by a coefficient fit on train
    NEIGHBOURS      last week's excess in the 8 surrounding cells  <-- the key one
    neigh + self    both together
    seasonal        the week-of-year average, learned on train
    roll4           average excess over the last 4 weeks

THE ONE THAT MATTERS IS "NEIGHBOURS". The Short et al. model this project
implements says a burglary raises the risk NEARBY and SOON -- the near-repeat
effect. If last week's neighbouring crime cannot predict this week's unusual
crime, then near-repeat is not visible at this resolution, and that is the
cleanest possible explanation for why the PINN has no temporal skill.

    python signal_ceiling.py --data data/burg_w24.npz
"""
from __future__ import annotations

import argparse

import numpy as np

GROUPS = ("Head", "Mid", "Tail")


def auc_np(y, s):
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(s, float).ravel()
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    _, first, cnt = np.unique(s[o], return_index=True, return_counts=True)
    for st, ct in zip(first[cnt > 1], cnt[cnt > 1]):
        idx = o[st:st + ct]; r[idx] = r[idx].mean()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def anscombe(y, mu0, floor=0.05):
    mu0 = np.maximum(np.asarray(mu0, float), floor)
    return 1.5 * (np.asarray(y, float) ** (2 / 3) - mu0 ** (2 / 3)) / (mu0 ** (1 / 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/burg_w24.npz")
    ap.add_argument("--kappa-rate", type=float, default=None,
                    help="positive rate for the fair label; default 0.8 x ceiling")
    ap.add_argument("--floor", type=float, default=0.05)
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    Uraw, mask, group, split = d["U_raw"], d["mask"], d["group"], d["split"]
    T, H, W = Uraw.shape
    iy, ix = np.where(mask)
    ncell = len(iy)
    tr = np.where(split == "train")[0]
    te = np.where(split == "test")[0]
    te = te[te > 0]                      # need t-1 to exist

    X = Uraw[:, iy, ix].astype(np.float64)          # (T, cells)
    cell_mean = X[tr].mean(0)                        # baseline, TRAIN only

    # fair target: Anscombe excess over each cell's own baseline
    Z = anscombe(X, cell_mean[None, :], a.floor)     # (T, cells)

    # ceiling: a label can only fire where count >= 1
    g = group[iy, ix]
    ceil_g = {k: float((X[te][:, g == k] >= 1).mean()) for k in GROUPS}
    ceiling = min(ceil_g.values())
    target = a.kappa_rate if a.kappa_rate else 0.8 * ceiling

    kappa = float(np.quantile(Z[te], 1 - target))
    Y = (Z[te] > kappa).astype(int)                  # (len(te), cells)

    print("=" * 74)
    print("SIGNAL CEILING — can anything beat a coin flip on the fair target?")
    print("=" * 74)
    print(f"  cells {ncell}   train {len(tr)}w   test {len(te)}w")
    print(f"  mean count per cell-week: " +
          "  ".join(f"{k} {X[te][:, g==k].mean():.2f}" for k in GROUPS))
    print(f"  ceiling {100*ceiling:.1f}%   target rate {100*target:.1f}%   "
          f"kappa {kappa:.3f}")
    print(f"  positive rate achieved: " +
          "  ".join(f"{k} {100*Y[:, g==k].mean():.1f}%" for k in GROUPS))

    # ---------------- build the neighbour operator ----------------------- #
    # 8-neighbour sum on the city grid, restricted to in-city cells
    grid_id = -np.ones((H, W), int)
    grid_id[iy, ix] = np.arange(ncell)
    nbr = [[] for _ in range(ncell)]
    for c in range(ncell):
        r0, c0 = iy[c], ix[c]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < H and 0 <= cc < W and grid_id[rr, cc] >= 0:
                    nbr[c].append(grid_id[rr, cc])
    nbr_deg = np.array([max(1, len(v)) for v in nbr], float)

    def neigh_sum(zrow):
        out = np.zeros(ncell)
        for c in range(ncell):
            if nbr[c]:
                out[c] = zrow[list(nbr[c])].sum()
        return out / nbr_deg

    # ---------------- AR(1) coefficient, fit on train only ---------------- #
    ztr_prev = Z[tr[:-1]].ravel(); ztr_now = Z[tr[1:]].ravel()
    denom = (ztr_prev ** 2).sum()
    ar = float((ztr_prev * ztr_now).sum() / denom) if denom > 0 else 0.0

    # ---------------- seasonal profile, fit on train only ----------------- #
    woy_tr = tr % 52
    seasonal = np.zeros(52)
    for w in range(52):
        m = woy_tr == w
        if m.any():
            seasonal[w] = Z[tr[m]].mean()

    # ---------------- assemble predictors on the test weeks --------------- #
    P = {k: np.zeros_like(Z[te]) for k in
         ("constant", "persistence", "AR(1)", "NEIGHBOURS",
          "neigh+self", "seasonal", "roll4")}
    for i, t in enumerate(te):
        zp = Z[t - 1]
        ns = neigh_sum(zp)
        P["constant"][i] = 0.0
        P["persistence"][i] = zp
        P["AR(1)"][i] = ar * zp
        P["NEIGHBOURS"][i] = ns
        P["neigh+self"][i] = ns + zp
        P["seasonal"][i] = seasonal[t % 52]
        lo = max(0, t - 4)
        P["roll4"][i] = Z[lo:t].mean(0) if t > lo else zp

    # ---------------- score ------------------------------------------------ #
    print(f"\n  AR(1) coefficient fit on train: {ar:+.4f}")
    print("  (near zero means last week barely predicts this week at all)\n")
    print(f"  {'predictor':>14} {'Head':>8} {'Mid':>8} {'Tail':>8} {'ALL':>8}")
    print("  " + "-" * 52)
    best = {}
    for name, pred in P.items():
        row = []
        for k in GROUPS:
            m = g == k
            row.append(100 * auc_np(Y[:, m].ravel(), pred[:, m].ravel()))
        allv = 100 * auc_np(Y.ravel(), pred.ravel())
        best[name] = allv
        star = "  <--" if name == "NEIGHBOURS" else ""
        print(f"  {name:>14} {row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f} "
              f"{allv:8.2f}{star}")

    # ---------------- verdict ---------------------------------------------- #
    top = max((v for k, v in best.items() if k != "constant"))
    topname = [k for k, v in best.items() if v == top][0]
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  best simple predictor: {topname} at AUC {top:.2f}")
    if top < 52:
        print("""
  NOTHING BEATS CHANCE.

  No trained model can beat this, because these predictors already use the
  only information available: what happened last week, nearby, and at this
  time of year. The ceiling is the DATA, not the architecture.

  This is a publishable negative result. It says the whole literature's
  reported performance is spatial recall -- knowing which neighbourhoods are
  busy -- and contains essentially no week-ahead forecasting.

  NEXT STEP: do not build a better model. Change the resolution
  (--grid / --freq in build_field.py) and re-run this script. If signal
  appears at some scale, that scale is your finding.""")
    elif top < 58:
        print(f"""
  WEAK BUT REAL SIGNAL ({top:.2f}).

  Something is there, but it is small. A model could plausibly reach the
  high 50s and no further. Check whether the PINN (54.97 / 52.87 / 40.39)
  already matches this ceiling -- if so it is not underperforming, it is
  at the limit of the data.""")
    else:
        print(f"""
  REAL SIGNAL ({top:.2f}), and the PINN is NOT capturing it.

  A trivial predictor beats the trained model. That points to a modelling
  problem, not a data problem -- which is good news and worth chasing.""")

    n_auc = best["NEIGHBOURS"]
    print(f"\n  NEAR-REPEAT CHECK: neighbours-last-week scores {n_auc:.2f}.")
    if n_auc < 52:
        print("  The near-repeat effect -- the entire basis of the Short et al.")
        print("  equations -- is NOT VISIBLE at this grid size and time step.")
        print("  Cells are ~1.5 km wide and steps are a week; near-repeat operates")
        print("  over hundreds of metres and days. The measuring box is very likely")
        print("  larger than the effect. Test this with a finer grid.")
    else:
        print("  Near-repeat IS visible. The physics prior has something real to")
        print("  work with, and the PINN should be able to exploit it.")
    print("=" * 74)


if __name__ == "__main__":
    main()
