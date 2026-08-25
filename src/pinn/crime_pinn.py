"""
STEP 2 — a Physics-Informed Neural Network for the Short et al. crime model
================================================================================
THE PHYSICS
-----------
Short, D'Orsogna, Pasour, Tita, Brantingham, Bertozzi & Chayes (2008),
"A statistical model of criminal behavior", Math. Models Methods Appl. Sci. 18.

Two coupled fields on the city:

    A(x,y,t)   attractiveness  - how appealing a location is to burgle
    rho(x,y,t) offender density

    dA/dt   = eta * lap(A) - A + A0 + rho*A
    drho/dt = div( grad(rho) - 2*(rho/A)*grad(A) ) - rho*A + A - A0

Reading the terms:
  eta*lap(A)          attractiveness spreads to neighbouring locations
  -A + A0             it decays back to a baseline A0
  +rho*A              a crime raises local attractiveness (repeat victimisation)
  -2*(rho/A)*grad(A)  offenders drift UP the attractiveness gradient
  -rho*A              an offender leaves after committing a crime
  +A - A0             replacement of offenders

The OBSERVED quantity is the crime rate, which in this model is  rho * A.
That is what our field U from build_field.py measures.

Self-consistency: at a uniform steady state both equations give rho*A = A - A0,
so the system is consistent (a useful check that we transcribed it correctly).

WHAT THIS SCRIPT DOES
---------------------
* network  (x, y, t) -> (A, rho), both forced positive by softplus
* data loss     :  (rho*A - U_observed)^2  on training time steps
* physics loss  :  the two PDE residuals at random collocation points
* eta and A0 are LEARNED, not fixed - the model discovers them from Chicago
* --pde-weight 0 gives the capacity-matched control: identical network, no physics
* evaluation reports F1 AND AUC per Head/Mid/Tail group, because this project
  has already shown that F1 alone can move without any gain in real skill

WHAT IT DOES NOT DO
-------------------
It does not claim the physics is a correct description of crime. The
attractiveness mechanism comes from Broken Windows theory, which is contested
(Sampson & Raudenbush 2004; Goodson & Hoyer-Leitzel 2021). This code lets you
MEASURE what that assumption does to predictions, particularly in low-crime
neighbourhoods. That measurement is the point.

Usage
-----
    python crime_pinn.py --data data/burg_w24.npz --pde-weight 1.0 --epochs 8000
    python crime_pinn.py --data data/burg_w24.npz --pde-weight 0.0 --epochs 8000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# determinism  (this project learned the hard way that seeding torch is not enough)
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
class FourierFeatures(nn.Module):
    """Random Fourier features. Plain MLPs are biased toward low frequencies and
    struggle to fit sharp hotspots; this is the standard fix (Tancik et al. 2020,
    used for PINNs by Wang et al.)."""
    def __init__(self, in_dim=3, n=32, scale=3.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n) * scale)

    def forward(self, z):
        p = 2 * np.pi * z @ self.B
        return torch.cat([torch.sin(p), torch.cos(p)], -1)


class A0Field(nn.Module):
    """A0(x, y): the static baseline attractiveness of a location.

    Short et al. state explicitly that A0 "is not necessarily uniform over the
    lattice grids". Learning it as a field rather than one constant is therefore
    FAITHFUL to the original model, not an extension. It also gives the network
    a proper place to store the spatial crime map, so the dynamics are free to
    model change over time instead of re-learning geography.
    """
    def __init__(self, width=64, depth=3, fourier=16, f_scale=3.0):
        super().__init__()
        self.ff = FourierFeatures(2, fourier, f_scale) if fourier else None
        d = 2 * fourier if fourier else 2
        L = []
        for _ in range(depth):
            L += [nn.Linear(d, width), nn.Tanh()]; d = width
        L += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x, y):
        z = torch.stack([x, y], -1)
        h = self.ff(z) if self.ff is not None else z
        return torch.nn.functional.softplus(self.net(h)[..., 0]) + 1e-3


class CrimePINN(nn.Module):
    """(x, y, t) -> (A, rho).  Both outputs are positive: A appears in a
    denominator (rho/A) and rho is a density."""
    def __init__(self, width=128, depth=5, fourier=32, f_scale=3.0,
                 spatial_A0=True):
        super().__init__()
        self.spatial_A0 = spatial_A0
        self.A0_net = A0Field(64, 3, 16, f_scale) if spatial_A0 else None
        self.ff = FourierFeatures(3, fourier, f_scale) if fourier else None
        d_in = 2 * fourier if fourier else 3
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
        # learned physical constants, kept positive through softplus
        self.raw_eta = nn.Parameter(torch.tensor(-1.0))
        self.raw_A0 = nn.Parameter(torch.tensor(0.0))

    @property
    def eta(self): return torch.nn.functional.softplus(self.raw_eta)

    @property
    def A0(self): return torch.nn.functional.softplus(self.raw_A0)

    def A0_at(self, x, y):
        """Baseline attractiveness: a field if enabled, otherwise the scalar."""
        if self.A0_net is not None:
            return self.A0_net(x, y)
        return self.A0.expand_as(x)

    def forward(self, x, y, t):
        z = torch.stack([x, y, t], -1)
        h = self.ff(z) if self.ff is not None else z
        out = self.net(h)
        A = torch.nn.functional.softplus(out[..., 0]) + 1e-3   # strictly > 0
        rho = torch.nn.functional.softplus(out[..., 1])
        return A, rho


# --------------------------------------------------------------------------- #
# PDE residuals
# --------------------------------------------------------------------------- #
def grad(f, v):
    return torch.autograd.grad(f, v, torch.ones_like(f), create_graph=True)[0]


def smooth_residuals(model, x, y, t):
    """ABLATION: a generic smoothness prior with NO burglary content.

    Penalises how fast the predicted crime rate changes in time and how curved
    it is in space. If this reproduces the gain from the Short PDE, then the
    physics is doing nothing a plain regulariser could not do -- which is the
    control this project has learned to always build.
    """
    x = x.requires_grad_(True); y = y.requires_grad_(True); t = t.requires_grad_(True)
    A, rho = model(x, y, t)
    u = rho * A
    u_t = grad(u, t)
    u_x, u_y = grad(u, x), grad(u, y)
    lap = grad(u_x, x) + grad(u_y, y)
    return u_t, lap


def pde_residuals(model, x, y, t):
    """Both residuals of the Short et al. system. Zero when the physics holds."""
    x = x.requires_grad_(True); y = y.requires_grad_(True); t = t.requires_grad_(True)
    A, rho = model(x, y, t)

    A_t = grad(A, t)
    A_x, A_y = grad(A, x), grad(A, y)
    A_xx, A_yy = grad(A_x, x), grad(A_y, y)
    lapA = A_xx + A_yy

    rho_t = grad(rho, t)
    rho_x, rho_y = grad(rho, x), grad(rho, y)
    rho_xx, rho_yy = grad(rho_x, x), grad(rho_y, y)
    lap_rho = rho_xx + rho_yy

    # div( (rho/A) grad A ) = grad(rho/A) . grad A + (rho/A) lap A
    w = rho / A
    w_x, w_y = grad(w, x), grad(w, y)
    div_term = w_x * A_x + w_y * A_y + w * lapA

    eta = model.eta
    A0 = model.A0_at(x, y)          # spatial baseline, per Short et al.
    rA = rho * A
    rA_res = A_t - (eta * lapA - A + A0 + rA)
    rr_res = rho_t - (lap_rho - 2.0 * div_term - rA + A - A0)
    return rA_res, rr_res


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(path, device):
    d = np.load(path, allow_pickle=True)
    U, mask, group = d["U"], d["mask"], d["group"]
    # U is Gaussian-smoothed so the PDE has a differentiable field to act on.
    # For BINARY labels we must use the RAW counts: "did a burglary actually
    # happen in this cell this week". Using the smoothed field makes almost
    # every cell non-zero and the labels degenerate.
    U_raw = d["U_raw"]
    x2, y2, t1, split = d["x"], d["y"], d["t"], d["split"]
    T = U.shape[0]

    # scale so the field is O(1): the PDE is written in nondimensional units
    scale = float(U[:, mask].mean())
    Us = U / max(scale, 1e-8)
    # for the Poisson likelihood we need the RATE on the raw count scale, so we
    # keep the conversion factor: rate = raw_scale * (rho*A)
    raw_scale = float(U_raw[:, mask].mean())

    iy, ix = np.where(mask)
    XY = np.stack([x2[iy, ix], y2[iy, ix]], 1).astype(np.float32)   # (M,2)
    G = group[iy, ix]

    def pack(which):
        ts = np.where(split == which)[0]
        n = len(ts) * len(iy)
        xs = np.tile(XY[:, 0], len(ts)); ys = np.tile(XY[:, 1], len(ts))
        tt = np.repeat(t1[ts], len(iy))
        rr = np.repeat(ts, len(iy)); cc = np.tile(iy, len(ts)); dd = np.tile(ix, len(ts))
        uu = Us[rr, cc, dd]
        ur = U_raw[rr, cc, dd]
        gg = np.tile(G, len(ts))
        to = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
        return dict(x=to(xs), y=to(ys), t=to(tt), u=to(uu),
                    u_raw=ur.astype(np.float32), u_raw_t=to(ur), g=gg, n=n)

    meta = json.loads(str(d["meta"]))
    # in-city cell centres, so collocation points never land in the lake
    city = torch.tensor(XY, dtype=torch.float32, device=device)
    return (pack("train"), pack("val"), pack("test"), scale, meta,
            raw_scale, city)


# --------------------------------------------------------------------------- #
# evaluation — F1 AND AUC per group
# --------------------------------------------------------------------------- #
def _predict(model, S, raw_scale=1.0):
    """Predicted burglary RATE per cell per week, on the raw count scale."""
    model.eval()
    with torch.no_grad():
        A, rho = model(S["x"], S["y"], S["t"])
        return (raw_scale * rho * A).cpu().numpy()


def tune_threshold(model, S, raw_scale=1.0):
    """Pick the threshold that maximises F1 on VALIDATION.

    Each model gets its own threshold, which is what a deployer would do. The
    positive-prediction rate is reported alongside every score so that a change
    driven by the operating point cannot be mistaken for a change in skill --
    the failure mode documented earlier in this project.
    """
    p = _predict(model, S, raw_scale)
    y = (S["u_raw"] > 0).astype(int)
    if y.sum() in (0, len(y)):
        return 0.5
    best, bt = -1.0, 0.5
    for q in np.linspace(1, 99, 99):
        t = float(np.percentile(p, q))
        pb = (p > t).astype(int)
        tp = ((pb == 1) & (y == 1)).sum(); fp = ((pb == 1) & (y == 0)).sum()
        fn = ((pb == 0) & (y == 1)).sum()
        f1 = 0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn)
        if f1 > best:
            best, bt = f1, t
    return bt


def evaluate(model, S, thr=None, raw_scale=1.0):
    pred = _predict(model, S, raw_scale)
    true = S["u_raw"]                      # compare rate against actual counts
    ybin = (S["u_raw"] > 0).astype(int)          # RAW counts, not smoothed
    res = {"rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
           "mae": float(np.mean(np.abs(pred - true))),
           "base_rate": float(100 * ybin.mean()),
           "thr": float(thr) if thr is not None else float("nan")}
    t = 0.5 if thr is None else thr
    for g in ("Head", "Mid", "Tail", "ALL"):
        m = np.ones_like(ybin, bool) if g == "ALL" else (S["g"] == g)
        yy, pp = ybin[m], np.nan_to_num(pred[m])
        res[f"{g}_base"] = float(100 * yy.mean()) if len(yy) else float("nan")
        res[f"{g}_rmse"] = float(np.sqrt(np.mean((pp - true[m]) ** 2))) if len(yy) else float("nan")
        if len(yy) == 0 or yy.sum() in (0, len(yy)):
            res[f"{g}_f1"] = res[f"{g}_auc"] = res[f"{g}_rate"] = float("nan")
            continue
        pb = (pp > t).astype(int)
        tp = ((pb == 1) & (yy == 1)).sum(); fp = ((pb == 1) & (yy == 0)).sum()
        fn = ((pb == 0) & (yy == 1)).sum()
        res[f"{g}_f1"] = float(0 if 2*tp+fp+fn == 0 else 200*tp/(2*tp+fp+fn))
        res[f"{g}_auc"] = float(100 * roc_auc_score(yy, pp))
        res[f"{g}_rate"] = float(100 * pb.mean())
    res["gap_f1"] = res["Head_f1"] - res["Tail_f1"]
    res["gap_auc"] = res["Head_auc"] - res["Tail_auc"]
    return res


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(args):
    seed_everything(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te, scale, meta, raw_scale, city = load(args.data, dev)
    print(f"device {dev} | field scale {scale:.4f} | raw scale {raw_scale:.4f} | "
          f"train pts {tr['n']:,} | test pts {te['n']:,}")
    print(f"data: {meta}")

    model = CrimePINN(args.width, args.depth, args.fourier, args.f_scale,
                      spatial_A0=not args.scalar_A0).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"model: {npar:,} parameters | physics={args.physics} "
          f"| loss={args.loss} | A0={'field' if not args.scalar_A0 else 'scalar'} "
          f"| pde-weight {args.pde_weight}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    mse = nn.MSELoss()

    # collocation points must lie INSIDE the city. Sampling uniformly over the
    # bounding box would enforce burglary physics over Lake Michigan.
    ncell = city.shape[0]
    jitter = 2.0 / max(meta.get("grid", 24), 1)      # half a cell, in [-1,1] units
    def collocation(n):
        i = torch.randint(0, ncell, (n,), device=dev)
        cx = city[i, 0] + (torch.rand(n, device=dev) - .5) * jitter
        cy = city[i, 1] + (torch.rand(n, device=dev) - .5) * jitter
        return cx.clamp(-1, 1), cy.clamp(-1, 1), torch.rand(n, device=dev)

    # adaptive sampling: keep a large pool, train on the points with the biggest
    # residuals. This is the idea from the Lin & Chen paper (Causal AS).
    pool = collocation(args.pool) if args.sampling == "adaptive" else None

    best, best_state, t0 = float("inf"), None, time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()

        idx = torch.randint(0, tr["n"], (args.batch,), device=dev)
        A, rho = model(tr["x"][idx], tr["y"][idx], tr["t"][idx])
        if args.loss == "poisson":
            # burglaries per cell per week are COUNTS. MSE treats them as
            # Gaussian, which over-weights busy cells -- a Head/Tail bias baked
            # straight into the objective. Poisson is the correct likelihood.
            lam = raw_scale * rho * A + 1e-6
            target = tr["u_raw_t"][idx]
            loss_data = (lam - target * torch.log(lam)).mean()
        else:
            loss_data = mse(rho * A, tr["u"][idx])

        loss_pde = torch.tensor(0.0, device=dev)
        if args.pde_weight > 0 and args.physics != "none":
            if args.sampling == "adaptive" and ep % args.resample == 0:
                fn = pde_residuals if args.physics == "short" else smooth_residuals
                with torch.enable_grad():
                    r1, r2 = fn(model, *[p.clone() for p in pool])
                    sev = (r1.abs() + r2.abs()).detach()
                keep = torch.topk(sev, args.n_coll).indices
                cx, cy, ct = pool[0][keep], pool[1][keep], pool[2][keep]
            else:
                cx, cy, ct = collocation(args.n_coll)
            if args.physics == "short":
                r1, r2 = pde_residuals(model, cx, cy, ct)
            else:                                   # generic smoothness ablation
                r1, r2 = smooth_residuals(model, cx, cy, ct)
            loss_pde = (r1 ** 2).mean() + (r2 ** 2).mean()

        loss = loss_data + args.pde_weight * loss_pde
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            v = evaluate(model, va, raw_scale=raw_scale)
            if v["rmse"] < best:
                best = v["rmse"]
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            print(f"  ep {ep:6d} | data {loss_data.item():.4f} "
                  f"| pde {loss_pde.item():.4f} | val rmse {v['rmse']:.4f} "
                  f"| eta {model.eta.item():.4f} A0 {model.A0.item():.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    thr = tune_threshold(model, va, raw_scale)   # tuned on validation
    r = evaluate(model, te, thr, raw_scale)
    r.update(pde_weight=args.pde_weight, seed=args.seed, scale=scale,
             physics=args.physics, loss=args.loss,
             spatial_A0=not args.scalar_A0,
             eta=float(model.eta.item()), A0=float(model.A0.item()),
             params=npar, sampling=args.sampling, data=args.data)

    print("\n" + "=" * 70)
    tag = ("CONTROL (no physics)" if args.physics == "none" or args.pde_weight == 0
           else f"PINN — physics={args.physics}")
    print(f"TEST — {tag}  [loss={args.loss}]")
    print("=" * 70)
    print(f"{'group':>6} {'F1':>8} {'AUC':>8} {'RMSE':>8} {'says yes':>9} {'actual':>8}")
    for g in ("Head", "Mid", "Tail", "ALL"):
        print(f"{g:>6} {r[g+'_f1']:8.2f} {r[g+'_auc']:8.2f} "
              f"{r[g+'_rmse']:8.3f} {r[g+'_rate']:8.1f}% {r[g+'_base']:7.1f}%")
    print(f"\n  threshold {r['thr']:.4f} (tuned on validation)")
    print(f"\n  Head-Tail gap:  F1 {r['gap_f1']:+.2f}   AUC {r['gap_auc']:+.2f}")
    print(f"  learned eta {r['eta']:.4f}   A0 {r['A0']:.4f}")
    print("  eta is the attractiveness diffusion rate the model inferred from data.")
    print("  Compare F1 and AUC gaps: if only F1 moves, the change is threshold,")
    print("  not skill -- the failure mode this project already documented.")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(f"\nappended -> {args.save}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/burg_w24.npz")
    ap.add_argument("--physics", default="short", choices=["short", "smooth", "none"],
                    help="short = Short et al. PDE; smooth = generic smoothness "
                         "ablation with no burglary content; none = control")
    ap.add_argument("--loss", default="poisson", choices=["poisson", "mse"],
                    help="poisson is the correct likelihood for counts")
    ap.add_argument("--scalar-A0", action="store_true",
                    help="revert A0 to a single constant (old behaviour)")
    ap.add_argument("--pde-weight", type=float, default=1.0,
                    help="0 = capacity-matched control with no physics")
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-coll", type=int, default=4096)
    ap.add_argument("--pool", type=int, default=100_000,
                    help="candidate pool for adaptive sampling")
    ap.add_argument("--sampling", default="uniform", choices=["uniform", "adaptive"])
    ap.add_argument("--resample", type=int, default=100)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--fourier", type=int, default=32, help="0 disables")
    ap.add_argument("--f-scale", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="results/pinn.jsonl")
    train(ap.parse_args())


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        print("=" * 70)
        print("This file was RUN as a notebook cell instead of being SAVED.")
        print("Add  %%writefile crime_pinn.py  as the FIRST line, then Enter.")
        print("=" * 70)
    else:
        main()
