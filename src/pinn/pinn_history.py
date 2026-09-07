"""
PINN + season + recent history  (Fix A and Fix B)
================================================================================
WHY
---
signal_ceiling.py showed that trivial predictors BEAT the trained PINN on the
base-rate-free target:

                        Head     Mid    Tail
    PINN               54.97   52.87   40.39
    4-week average     62.14   58.43   49.59
    week-of-year       60.84   55.91   51.99

The reason is structural. The PINN is a function of (x, y, t) only. It never
sees "three burglaries happened here last week", so it cannot compete with a
moving average, which does. And with t spanning five years in [0,1], the
Fourier features cannot resolve the 52-week seasonal cycle.

WHAT THIS DOES
--------------
Keeps the physics EXACTLY as it was, and adds an observation-level correction:

    lambda_hat = raw_scale * rho(x,y,t) * A(x,y,t) * exp( g(features) )
                 \_______ physics, unchanged _______/   \___ new ___/

CRITICAL DESIGN POINT: g does NOT enter the PDE residuals. A and rho still
obey the Short et al. equations exactly as before; autodiff still differentiates
a clean function of (x,y,t). This is the standard separation of a PROCESS model
(the PDE, governing the latent fields) from an OBSERVATION model (what we
actually record, which also depends on season and recent activity).

Say it that way to a reviewer. "We bolted an LSTM onto a PINN" is indefensible.
"The PDE governs the latent process; seasonality and near-term activity are
observation-level effects the PDE does not claim to model" is defensible.

FEATURES (all computed from TRAIN-period statistics only, no leakage)
    season_sin, season_cos   sin/cos of week-of-year          <- Fix B
    z_lag1                   this cell's excess last week     <- Fix A
    z_lag4                   this cell's mean excess, last 4 weeks
    z_nbr1                   neighbours' mean excess last week (near-repeat)

ABLATION — the script runs four configurations so the claim is testable:
    physics only            (the current model, reproduced)
    physics + season        (Fix B alone)
    physics + history       (Fix A alone)
    physics + season + history

and prints the trivial baselines alongside, so you can see immediately whether
the model has actually earned its place.

    python pinn_history.py --data data/burg_w24.npz --epochs 4000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from crime_pinn import CrimePINN, pde_residuals, seed_everything

GROUPS = ("Head", "Mid", "Tail")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
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


def macro_auc(y, s, cell_id, min_pos=3):
    """Mean of PER-CELL AUCs.

    THIS IS THE HONEST METRIC. Pooling all cells into one AUC lets a model
    score by ranking CELLS against each other -- which is the base-rate
    confound this whole project is about, sneaking back in through the metric.
    Computing AUC inside each cell and then averaging makes that structurally
    impossible: every comparison is between two WEEKS OF THE SAME CELL.

    Cells with fewer than `min_pos` positives (or negatives) are skipped --
    their AUC is undefined or meaningless, not zero.
    """
    vals = []
    for c in np.unique(cell_id):
        m = cell_id == c
        yy = y[m]
        npos = int(yy.sum())
        if npos < min_pos or (len(yy) - npos) < min_pos:
            continue
        vals.append(auc_np(yy, s[m]))
    if not vals:
        return float("nan"), 0
    return 100 * float(np.mean(vals)), len(vals)


def anscombe(y, mu0, floor=0.05):
    mu0 = np.maximum(np.asarray(mu0, float), floor)
    return 1.5 * (np.asarray(y, float) ** (2 / 3) - mu0 ** (2 / 3)) / (mu0 ** (1 / 6))


# --------------------------------------------------------------------------- #
# data: field + features + the FAIR target
# --------------------------------------------------------------------------- #
def build(path, floor=0.05, kappa_rate=None, lag_start=4,
          per_cell_kappa=True):
    d = np.load(path, allow_pickle=True)
    U, Uraw = d["U"], d["U_raw"]
    x2, y2, t1 = d["x"], d["y"], d["t"]
    mask, group, split = d["mask"], d["group"], d["split"]
    T, H, W = Uraw.shape
    iy, ix = np.where(mask)
    n = len(iy)
    g = group[iy, ix]

    X = Uraw[:, iy, ix].astype(np.float64)          # (T, cells) raw counts
    Xs = U[:, iy, ix].astype(np.float64)            # smoothed, for the PDE scale
    tr_all = np.where(split == "train")[0]
    cell_mean = X[tr_all].mean(0)                   # baseline, TRAIN ONLY
    Z = anscombe(X, cell_mean[None, :], floor)      # (T, cells) fair excess

    # 8-neighbour averaging operator
    gid = -np.ones((H, W), int); gid[iy, ix] = np.arange(n)
    nbr = []
    for c in range(n):
        v = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = iy[c] + dr, ix[c] + dc
                if 0 <= rr < H and 0 <= cc < W and gid[rr, cc] >= 0:
                    v.append(gid[rr, cc])
        nbr.append(v)
    deg = np.array([max(1, len(v)) for v in nbr], float)
    NB = np.zeros((n, n), np.float32)
    for c, v in enumerate(nbr):
        for j in v:
            NB[c, j] = 1.0 / deg[c]

    # ---- fair label threshold ------------------------------------------- #
    # A SINGLE GLOBAL kappa leaves per-cell positive rates unequal (we measured
    # 7.5% to 38.0%), which lets a pooled AUC be won by ranking cells rather
    # than weeks. A PER-CELL kappa, fitted on the TRAIN period only, removes
    # most of that. Combined with macro_auc() the loophole is closed.
    te_all = np.where(split == "test")[0]
    te_all = te_all[te_all >= lag_start]
    ceil = min(float((X[te_all][:, g == k] >= 1).mean()) for k in GROUPS)
    target = kappa_rate if kappa_rate else 0.8 * ceil

    if per_cell_kappa:
        # one threshold per cell, from the training period
        kappa = np.quantile(Z[tr_all], 1 - target, axis=0)      # (cells,)
    else:
        kappa = np.full(n, float(np.quantile(Z[te_all], 1 - target)))

    scale = float(Xs[tr_all].mean())
    raw_scale = float(X[tr_all].mean())

    def pack(which):
        ts = np.where(split == which)[0]
        ts = ts[ts >= lag_start]
        nT = len(ts)
        xs = np.tile(x2[iy, ix], nT).astype(np.float32)
        ys = np.tile(y2[iy, ix], nT).astype(np.float32)
        tt = np.repeat(t1[ts], n).astype(np.float32)
        counts = X[ts].reshape(-1)
        # ---- features, all from data strictly BEFORE the target week ---- #
        woy = (ts % 52).astype(np.float64)
        s_sin = np.repeat(np.sin(2 * np.pi * woy / 52), n)
        s_cos = np.repeat(np.cos(2 * np.pi * woy / 52), n)
        z1 = Z[ts - 1].reshape(-1)
        z4 = np.stack([Z[ts - k] for k in (1, 2, 3, 4)]).mean(0).reshape(-1)
        zn = (Z[ts - 1] @ NB.T).reshape(-1)
        feat = np.stack([s_sin, s_cos, z1, z4, zn], 1).astype(np.float32)
        lab = (Z[ts] > kappa[None, :]).reshape(-1).astype(int)
        cell = np.tile(np.arange(n), nT)          # which cell each row is
        return dict(x=xs, y=ys, t=tt, counts=counts.astype(np.float32),
                    feat=feat, lab=lab, g=np.tile(g, nT), cell=cell,
                    n=len(counts))

    return (pack("train"), pack("val"), pack("test"),
            raw_scale, scale, kappa, target, ceil,
            np.stack([x2[iy, ix], y2[iy, ix]], 1).astype(np.float32))


# --------------------------------------------------------------------------- #
# model: physics core (unchanged) + observation correction
# --------------------------------------------------------------------------- #
class HistPINN(CrimePINN):
    """CrimePINN with an OPTIONAL multiplicative observation correction.

    forward(x, y, t) is inherited untouched and still returns (A, rho), so
    pde_residuals() differentiates exactly the same function it always did.
    The correction is applied only in rate(), never in the physics.
    """
    SEASON = [0, 1]
    HIST = [2, 3, 4]

    def __init__(self, use_season, use_history, width=128, depth=5,
                 fourier=32, f_scale=3.0, corr_width=64):
        super().__init__(width=width, depth=depth, fourier=fourier,
                         f_scale=f_scale, spatial_A0=True)
        cols = []
        if use_season:
            cols += self.SEASON
        if use_history:
            cols += self.HIST
        self.cols = cols
        self.corr = None
        if cols:
            self.corr = nn.Sequential(
                nn.Linear(len(cols), corr_width), nn.Tanh(),
                nn.Linear(corr_width, corr_width), nn.Tanh(),
                nn.Linear(corr_width, 1))
            # start as identity: exp(0) = 1, so training begins from the
            # unmodified physics model and any gain is attributable to g
            nn.init.zeros_(self.corr[-1].weight); nn.init.zeros_(self.corr[-1].bias)

    def rate(self, x, y, t, feat, raw_scale):
        A, rho = self(x, y, t)
        lam = raw_scale * rho * A
        if self.corr is not None:
            adj = self.corr(feat[:, self.cols])[..., 0]
            lam = lam * torch.exp(torch.clamp(adj, -3.0, 3.0))
        return lam


# --------------------------------------------------------------------------- #
def evaluate(model, S, raw_scale, device):
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, S["n"], 200_000):
            sl = slice(i, i + 200_000)
            T = lambda a: torch.tensor(a[sl], device=device)
            out.append(model.rate(T(S["x"]), T(S["y"]), T(S["t"]),
                                  T(S["feat"]), raw_scale).cpu().numpy())
        lam = np.concatenate(out)
    res = {}
    for k in GROUPS:
        m = S["g"] == k
        res[k] = 100 * auc_np(S["lab"][m], lam[m])                 # pooled
        res[k + "_macro"], res[k + "_ncell"] = macro_auc(
            S["lab"][m], lam[m], S["cell"][m])                     # within-cell
    res["ALL"] = 100 * auc_np(S["lab"], lam)
    res["ALL_macro"], res["ALL_ncell"] = macro_auc(S["lab"], lam, S["cell"])
    return res


def run(cfg, name, tr, va, te, raw_scale, scale, city, args, device):
    seed_everything(args.seed)
    model = HistPINN(cfg["season"], cfg["history"],
                     width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    npar = sum(p.numel() for p in model.parameters())
    city_t = torch.tensor(city, device=device)

    Ttr = {k: torch.tensor(v, device=device) for k, v in tr.items()
           if k in ("x", "y", "t", "counts", "feat")}
    t0 = time.time(); best = np.inf; best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        idx = torch.randint(0, tr["n"], (args.batch,), device=device)
        lam = model.rate(Ttr["x"][idx], Ttr["y"][idx], Ttr["t"][idx],
                         Ttr["feat"][idx], raw_scale).clamp_min(1e-6)
        yb = Ttr["counts"][idx]
        loss_data = (lam - yb * torch.log(lam)).mean()      # Poisson NLL

        # ---- physics on A and rho only; the correction is not involved --- #
        j = torch.randint(0, city.shape[0], (args.n_coll,), device=device)
        cx = city_t[j, 0].clone().requires_grad_(True)
        cy = city_t[j, 1].clone().requires_grad_(True)
        ct = torch.rand(args.n_coll, device=device).requires_grad_(True)
        rA, rR = pde_residuals(model, cx, cy, ct)
        loss_pde = (rA ** 2).mean() + (rR ** 2).mean()

        loss = loss_data + cfg.get("pde", 1.0) * args.pde_weight * loss_pde
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            v = evaluate(model, va, raw_scale, device)
            if -v["ALL"] < best:
                best = -v["ALL"]
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
            print(f"    ep {ep:5d} | data {loss_data.item():.4f} "
                  f"| pde {loss_pde.item():.4f} | val AUC {v['ALL']:.2f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    r = evaluate(model, te, raw_scale, device)
    r.update(name=name, params=npar, **cfg)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/burg_w24.npz")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-coll", type=int, default=2048)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pde-weight", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa-rate", type=float, default=None)
    ap.add_argument("--control-only", action="store_true",
                    help="run ONLY the no-physics controls (saves time when the "
                         "physics arms are already done)")
    ap.add_argument("--save", default="results/pinn_history.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te, raw_scale, scale, kappa, target, ceil, city = build(
        args.data, kappa_rate=args.kappa_rate)

    print("=" * 74)
    print("PINN + SEASON + HISTORY — evaluated on the BASE-RATE-FREE target")
    print("=" * 74)
    print(f"  device {device}   train {tr['n']:,}  val {va['n']:,}  test {te['n']:,}")
    kap = np.atleast_1d(kappa)
    print(f"  ceiling {100*ceil:.1f}%   target {100*target:.1f}%   "
          f"kappa: {kap.mean():.3f} mean, [{kap.min():.3f}, {kap.max():.3f}] "
          f"({'per-cell' if len(kap) > 1 else 'global'})")
    # how equal did the per-cell positive rates actually come out?
    prates = np.array([100 * te["lab"][te["cell"] == c].mean()
                       for c in np.unique(te["cell"])])
    print(f"  per-cell positive rate: {prates.min():.1f}% to {prates.max():.1f}%  "
          f"(median {np.median(prates):.1f}%)")
    print("  NOTE: these are not equal -- sparse cells cannot reach the target")
    print("  rate at all. That is why macro AUC, not the label, is the real fix.")
    print(f"  positive rate: " +
          "  ".join(f"{k} {100*te['lab'][te['g']==k].mean():.1f}%" for k in GROUPS))
    print("\n  REFERENCE — trivial baselines from signal_ceiling.py:")
    print(f"    {'4-week average':>16}   62.14  58.43  49.59")
    print(f"    {'week-of-year':>16}   60.84  55.91  51.99")
    print(f"    {'PINN (old)':>16}   54.97  52.87  40.39")
    print(f"\n  TARGET TO BEAT — physics + history, 5 seeds:")
    print(f"    {'Head 51.12+/-0.63   Mid 53.42+/-1.16   Tail 60.70+/-0.75'}")
    print("    If the NO-physics control matches Tail 60.70, the physics adds")
    print("    nothing and the paper must say so plainly.")

    ALL = [
        (dict(season=False, history=False, pde=1.0), "physics only"),
        (dict(season=True,  history=False, pde=1.0), "physics + season"),
        (dict(season=False, history=True,  pde=1.0), "physics + history"),
        (dict(season=True,  history=True,  pde=1.0), "physics + season + history"),
        # ---- CAPACITY-MATCHED CONTROLS: identical network, PDE loss OFF ---- #
        # If these match "physics + history", the physics is doing nothing and
        # the gain is entirely from the history features. This is the first
        # question any reviewer will ask.
        (dict(season=False, history=True,  pde=0.0), "NO physics + history"),
        (dict(season=True,  history=True,  pde=0.0), "NO physics + season + hist"),
    ]
    cfgs = ALL[4:] if args.control_only else ALL
    rows = []
    for cfg, name in cfgs:
        print(f"\n  --- {name} ---")
        rows.append(run(cfg, name, tr, va, te, raw_scale, scale, city, args, device))

    print("\n" + "=" * 74)
    print("RESULTS — AUC on the fair target (higher is better, 50 = coin flip)")
    print("=" * 74)
    print(f"  {'configuration':>28} {'Head':>8} {'Mid':>8} {'Tail':>8} {'ALL':>8} {'params':>9}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['name']:>28} {r['Head']:8.2f} {r['Mid']:8.2f} "
              f"{r['Tail']:8.2f} {r['ALL']:8.2f} {r['params']:9,}")
    print("\n" + "=" * 74)
    print("MACRO AUC — averaged WITHIN cells (the honest metric)")
    print("=" * 74)
    print(f"  {'configuration':>28} {'Head':>8} {'Mid':>8} {'Tail':>8} {'ALL':>8} {'cells':>7}")
    print("  " + "-" * 72)
    for r in rows:
        print(f"  {r['name']:>28} {r['Head_macro']:8.2f} {r['Mid_macro']:8.2f} "
              f"{r['Tail_macro']:8.2f} {r['ALL_macro']:8.2f} {r['ALL_ncell']:7d}")
    print("\n  Pooled AUC can be won by ranking CELLS; macro AUC compares only")
    print("  weeks within the same cell. Where the two disagree, believe macro.")
    print(f"  {'4-week average (trivial)':>28} {62.14:8.2f} {58.43:8.2f} "
          f"{49.59:8.2f} {55.06:8.2f} {0:9,}")

    print("\nHOW TO READ THIS")
    print("-" * 74)
    print("  * If 'physics only' still sits near 50, the reproduction is correct")
    print("    and the earlier finding stands.")
    print("  * If season or history lifts it ABOVE the 4-week average, the model")
    print("    has earned its place. If it merely matches, say so honestly: the")
    print("    physics is adding nothing a moving average does not already give.")
    print("  * Tail is the column that matters. Every trivial method is near 50")
    print("    there, so any real gain in Tail is the genuinely new result.")
    print("  * The correction head starts at exp(0)=1, so every configuration")
    print("    begins from the unmodified physics model. Any difference is")
    print("    attributable to the added features, not to a different init.")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
