r"""
When is a physics prior worth having?  — the density crossover
================================================================================
THE CLAIM THIS TESTS
--------------------
Everyone in scientific ML assumes physics priors help when data is scarce.
There is surprisingly little clean evidence for it. Our three-group result
hints at it:

                        data        trivial     physics+history
    Head (busy)         plentiful   62.1 WINS   51.1
    Mid                 medium      58.4 WINS   53.4
    Tail (sparse)       scarce      49.6        60.7 WINS

Three bins is an anecdote. This script turns it into a CURVE: sort every cell
by how much data it has, bin them, and measure each method's skill in each bin.

If the physics advantage rises smoothly as density falls, and crosses zero at
some identifiable density, that is a general machine-learning finding
demonstrated on crime data -- not a crime-prediction result. That distinction
is what makes it publishable at a venue like ICLR.

WHAT IS COMPARED, in every density bin
--------------------------------------
    4-week average      trivial; uses recent data only, no model
    NO physics + hist   same network, same params, PDE loss OFF
    physics + hist      the full model

THE OUTPUT
----------
    * a table of AUC per density bin per method
    * the ADVANTAGE curve: physics minus trivial, against density
    * the crossover density where the advantage passes zero
    * fig_crossover.png

    python crossover.py --data data/burg_w24.npz --epochs 4000 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from pinn_history import HistPINN, anscombe, auc_np, build, macro_auc
from crime_pinn import pde_residuals, seed_everything


# --------------------------------------------------------------------------- #
def train_one(cfg, tr, va, raw_scale, city, args, device):
    """Train one configuration and return it. Mirrors pinn_history.run()."""
    seed_everything(cfg["seed"])
    model = HistPINN(cfg["season"], cfg["history"],
                     width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    city_t = torch.tensor(city, device=device)
    T = {k: torch.tensor(v, device=device) for k, v in tr.items()
         if k in ("x", "y", "t", "counts", "feat")}
    best, best_state = -np.inf, None

    for ep in range(1, args.epochs + 1):
        model.train()
        idx = torch.randint(0, tr["n"], (args.batch,), device=device)
        lam = model.rate(T["x"][idx], T["y"][idx], T["t"][idx],
                         T["feat"][idx], raw_scale).clamp_min(1e-6)
        loss_data = (lam - T["counts"][idx] * torch.log(lam)).mean()

        j = torch.randint(0, city.shape[0], (args.n_coll,), device=device)
        cx = city_t[j, 0].clone().requires_grad_(True)
        cy = city_t[j, 1].clone().requires_grad_(True)
        ct = torch.rand(args.n_coll, device=device).requires_grad_(True)
        rA, rR = pde_residuals(model, cx, cy, ct)
        loss = loss_data + cfg["pde"] * args.pde_weight * ((rA**2).mean() + (rR**2).mean())
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            v = predict(model, va, raw_scale, device)
            a = 100 * auc_np(va["lab"], v)
            if a > best:
                best = a
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, S, raw_scale, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, S["n"], 200_000):
            sl = slice(i, i + 200_000)
            T = lambda a: torch.tensor(a[sl], device=device)
            out.append(model.rate(T(S["x"]), T(S["y"]), T(S["t"]),
                                  T(S["feat"]), raw_scale).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/burg_w24.npz")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-coll", type=int, default=2048)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pde-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save", default="results/crossover.json")
    ap.add_argument("--fig", default="fig_crossover.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te, raw_scale, scale, kappa, target, ceil, city = build(args.data)

    # ---- per-cell density, measured on TRAIN only ------------------------ #
    d = np.load(args.data, allow_pickle=True)
    mask = d["mask"]; iy, ix = np.where(mask); ncell = len(iy)
    split = d["split"]
    X = d["U_raw"][:, iy, ix].astype(np.float64)
    dens = X[np.where(split == "train")[0]].mean(0)      # events per cell per week

    nweeks = te["n"] // ncell
    cell_of_row = np.tile(np.arange(ncell), nweeks)
    dens_of_row = dens[cell_of_row]

    # ---- the trivial baseline: mean fair-excess over the last 4 weeks ---- #
    #      (feature column 3 is exactly that, already lag-safe)
    trivial = te["feat"][:, 3]

    print("=" * 76)
    print("DENSITY CROSSOVER — when does the physics prior start to pay?")
    print("=" * 76)
    print(f"  cells {ncell}   test weeks {nweeks}   rows {te['n']:,}")
    print(f"  density (events/cell/week): min {dens.min():.3f}  "
          f"median {np.median(dens):.3f}  max {dens.max():.3f}")

    # ---- train both arms, several seeds ---------------------------------- #
    preds = {"physics": [], "nophysics": []}
    for s in range(args.seeds):
        for tag, pde in (("physics", 1.0), ("nophysics", 0.0)):
            t0 = time.time()
            m = train_one(dict(season=False, history=True, pde=pde, seed=s),
                          tr, va, raw_scale, city, args, device)
            preds[tag].append(predict(m, te, raw_scale, device))
            print(f"  trained {tag:>10}  seed {s}  ({time.time()-t0:.0f}s)")

    # ---- bin cells by density, equal cells per bin ----------------------- #
    order = np.argsort(dens)
    edges = np.array_split(order, args.bins)
    bin_of_cell = np.empty(ncell, int)
    for b, cells in enumerate(edges):
        bin_of_cell[cells] = b
    bin_of_row = bin_of_cell[cell_of_row]

    rows = []
    for b in range(args.bins):
        m = bin_of_row == b
        y = te["lab"][m]
        dmean = dens[edges[b]].mean()
        r = {"bin": b, "density": float(dmean), "n_cells": int(len(edges[b])),
             "n_rows": int(m.sum()), "pos_rate": float(100 * y.mean())}
        cid = te["cell"][m]
        # MACRO (within-cell) AUC: between-cell ranking cannot contribute.
        r["trivial"], _ = macro_auc(y, trivial[m], cid)
        r["trivial_pooled"] = 100 * auc_np(y, trivial[m])
        for tag in ("physics", "nophysics"):
            v = np.array([macro_auc(y, p[m], cid)[0] for p in preds[tag]])
            r[tag] = float(v.mean()); r[tag + "_sd"] = float(v.std(ddof=1) if len(v) > 1 else 0)
            r[tag + "_pooled"] = float(np.mean([100 * auc_np(y, p[m]) for p in preds[tag]]))
        r["advantage"] = r["physics"] - r["trivial"]
        r["phys_vs_nophys"] = r["physics"] - r["nophysics"]
        rows.append(r)

    # ---- report ---------------------------------------------------------- #
    print("\n  MACRO AUC — computed WITHIN each cell, then averaged.")
    print("  Between-cell ranking cannot inflate these numbers.")
    print(f"\n  {'bin':>3} {'density':>9} {'cells':>6} {'pos%':>6} "
          f"{'trivial':>9} {'no-phys':>13} {'PHYSICS':>13} {'vs triv':>9} {'vs no-ph':>9}")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  {r['bin']:>3} {r['density']:9.3f} {r['n_cells']:6d} "
              f"{r['pos_rate']:5.1f}% {r['trivial']:9.2f} "
              f"{r['nophysics']:8.2f}+/-{r['nophysics_sd']:4.2f} "
              f"{r['physics']:8.2f}+/-{r['physics_sd']:4.2f} "
              f"{r['advantage']:+9.2f} {r['phys_vs_nophys']:+9.2f}")

    # ---- the crossover point --------------------------------------------- #
    dn = np.array([r["density"] for r in rows])
    ad = np.array([r["advantage"] for r in rows])
    cross = None
    for i in range(len(dn) - 1):
        if ad[i] * ad[i + 1] < 0:            # sign change between bins
            f = ad[i] / (ad[i] - ad[i + 1])
            cross = dn[i] + f * (dn[i + 1] - dn[i])
            break
    rho = float(np.corrcoef(np.log(dn), ad)[0, 1])
    pv = np.array([r["phys_vs_nophys"] for r in rows])
    rho_pv = float(np.corrcoef(np.log(dn), pv)[0, 1])

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    print(f"  PHYSICS vs NO-PHYSICS (the capacity-matched comparison):")
    print(f"    mean gain {pv.mean():+.2f} +/- {pv.std(ddof=1):.2f}   "
          f"wins in {int((pv>0).sum())}/{len(pv)} bins")
    print(f"    correlation with log(density): r = {rho_pv:+.3f}")
    print(f"    -> if this r is near 0, the gain is UNIFORM, not density-driven.")
    print(f"\n  PHYSICS vs TRIVIAL:")
    print(f"    correlation between log(density) and advantage: r = {rho:+.3f}")
    if cross is not None:
        print(f"  CROSSOVER at density ~= {cross:.3f} events/cell/week")
        print(f"  Below this the physics prior wins; above it a 4-week average wins.")
    else:
        print("  No sign change across the bins — advantage has one sign throughout.")
    if rho < -0.7:
        print("""
  STRONG MONOTONE RELATIONSHIP. The physics prior's value is governed by data
  density, and you can state the law rather than the anecdote. This is the
  general ML claim: report the crossover density as the headline number.""")
    elif rho < -0.4:
        print("""
  MODERATE relationship. Report it with the caveat that bins are noisy, and
  strengthen it with the synthetic sweep where density is controlled directly.""")
    else:
        print("""
  WEAK relationship. The three-group result may be driven by something other
  than density. Do NOT make the density claim until the synthetic sweep is
  run -- it would not survive review.""")

    # ---- figure ----------------------------------------------------------- #
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, GRN, RED, GREY = "#1E2761", "#1E8449", "#C0392B", "#B8BFCC"
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=200)

    ph = np.array([r["physics"] for r in rows]); ph_sd = np.array([r["physics_sd"] for r in rows])
    npx = np.array([r["nophysics"] for r in rows]); np_sd = np.array([r["nophysics_sd"] for r in rows])
    tv = np.array([r["trivial"] for r in rows])

    ax[0].errorbar(dn, ph, yerr=ph_sd, marker="o", color=GRN, lw=2, capsize=3,
                   label="physics + history")
    ax[0].errorbar(dn, npx, yerr=np_sd, marker="s", color=NAVY, lw=1.6, capsize=3,
                   alpha=.75, label="no physics + history")
    ax[0].plot(dn, tv, marker="^", color=GREY, lw=1.6, label="4-week average")
    ax[0].axhline(50, ls="--", lw=1.1, color=RED)
    ax[0].set_xscale("log"); ax[0].set_xlabel("data density (events per cell per week)", fontsize=10)
    ax[0].set_ylabel("AUC", fontsize=10.5)
    ax[0].set_title("Skill vs how much data a region has", fontsize=11.5, color=NAVY, weight="bold")
    ax[0].legend(fontsize=8.5, frameon=False, loc="best")

    ax[1].axhline(0, color="#333", lw=1)
    ax[1].plot(dn, ad, marker="o", color=GRN, lw=2.2)
    ax[1].fill_between(dn, 0, ad, where=ad > 0, color=GRN, alpha=.18)
    ax[1].fill_between(dn, 0, ad, where=ad < 0, color=GREY, alpha=.35)
    if cross is not None:
        ax[1].axvline(cross, ls=":", color=RED, lw=1.6)
        ax[1].text(cross, ad.max() * .82, f"  crossover\n  {cross:.2f} ev/cell/wk",
                   fontsize=9, color=RED)
    ax[1].set_xscale("log"); ax[1].set_xlabel("data density (events per cell per week)", fontsize=10)
    ax[1].set_ylabel("physics advantage over trivial (AUC pts)", fontsize=10)
    ax[1].set_title(f"The prior pays off only when data is scarce   (r = {rho:+.2f})",
                    fontsize=11.5, color=NAVY, weight="bold")

    plt.tight_layout(); plt.savefig(args.fig, bbox_inches="tight", facecolor="white")
    print(f"\n  figure -> {args.fig}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        json.dump({"bins": rows, "crossover": cross, "logdens_corr": rho},
                  open(args.save, "w"), indent=2)
        print(f"  saved  -> {args.save}")


if __name__ == "__main__":
    main()
