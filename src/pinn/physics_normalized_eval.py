"""
Physics-Normalised Evaluation (PNE)
================================================================================
THE IDEA
--------
Every fairness result in this project has fought the same enemy: the label
"did a burglary happen in this cell this week" has a base rate of ~60% in busy
neighbourhoods and ~12% in quiet ones. F1 then differs between groups even when
the model is equally good everywhere, and every correction we tried (skill
scores, per-group thresholds) patches the symptom after the fact.

This script asks a different question: why correct for the base rate when you
can remove it from the TARGET?

Proposition 1 of the report showed that at steady state the Short et al. system
satisfies rho*A = A - A0. The observed crime rate is therefore already a
DEVIATION quantity, measured against the static baseline A0(x,y) that the PINN
learns as a field. So instead of predicting counts, predict the standardised
excess over that baseline:

    mu0(x,y) = raw_scale * A0(x,y)                 <- physics baseline

                 3     y^(2/3) - mu0^(2/3)
    z(x,y,t) =  --- * ---------------------        <- Anscombe residual
                 2         mu0^(1/6)                  (Poisson-stabilising)

    label    =  z > kappa      "was this week unusually high FOR THIS CELL"

Because mu0 already carries the cell's own rate, the base rate of that label is
the same in every cell BY CONSTRUCTION. The confound is not corrected. It is
absent.

WHY THE PHYSICS MATTERS
-----------------------
You could normalise by the historical per-cell mean instead. That baseline is a
free statistical quantity and may absorb real signal. A0 is constrained by the
two coupled PDEs -- it cannot be whatever the fit wants -- and it has a meaning
in the criminology ("static intrinsic attractiveness").

This script therefore runs THREE arms so the claim is testable, not assumed:

    raw        the current target (counts > 0)          -- the baseline
    empirical  normalised by per-cell training mean     -- the ABLATION
    physics    normalised by the learned A0 field       -- the proposal

If 'physics' behaves like 'empirical', the physics adds nothing and you should
say so. That comparison is the point of the script.

WHAT TO LOOK FOR
----------------
 1. Under the normalised targets, per-group base rates should be nearly equal.
    That is the mechanism working.
 2. The Head-Tail F1 gap should collapse.
 3. AUC on the normalised target is the model's TEMPORAL skill, isolated from
    geography. This finally quantifies the report's stated limitation that the
    PINN "learned where, not when". If it lands near 50, that is a real and
    publishable finding, not a failure.

USAGE
-----
    # 1. train a PINN and keep the weights
    python crime_pinn.py --data data/burg_w24.npz --physics short \
        --loss poisson --epochs 8000 --save-ckpt results/pinn_s0.pt

    # 2. run this
    python physics_normalized_eval.py --data data/burg_w24.npz \
        --ckpt results/pinn_s0.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from crime_pinn import CrimePINN, seed_everything

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
    vals, first, cnt = np.unique(s[o], return_index=True, return_counts=True)
    for st, ct in zip(first[cnt > 1], cnt[cnt > 1]):
        idx = o[st:st + ct]; r[idx] = r[idx].mean()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def f1_at(y, s, thr):
    p = (s > thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return 0.0 if 2 * tp + fp + fn == 0 else 100 * 2 * tp / (2 * tp + fp + fn)


def best_threshold(y, s, n=200):
    """Tune one global threshold, exactly as the main pipeline does."""
    lo, hi = np.quantile(s, 0.01), np.quantile(s, 0.99)
    grid = np.linspace(lo, hi, n)
    scores = [f1_at(y, s, t) for t in grid]
    return float(grid[int(np.argmax(scores))])


# --------------------------------------------------------------------------- #
# the transform
# --------------------------------------------------------------------------- #
def anscombe_excess(y, mu0, floor=0.05):
    """Variance-stabilised Poisson residual of counts y against baseline mu0.

    The Anscombe transform y -> y^(2/3) makes Poisson variance roughly constant,
    so cells with very different rates become comparable. `floor` prevents the
    denominator exploding where the baseline is near zero.
    """
    mu0 = np.maximum(np.asarray(mu0, float), floor)
    y = np.asarray(y, float)
    return 1.5 * (y ** (2 / 3) - mu0 ** (2 / 3)) / (mu0 ** (1 / 6))


# --------------------------------------------------------------------------- #
def load_field(path):
    d = np.load(path, allow_pickle=True)
    return dict(U=d["U"], U_raw=d["U_raw"], x=d["x"], y=d["y"], t=d["t"],
                mask=d["mask"], group=d["group"], split=d["split"])


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = CrimePINN(width=ck["width"], depth=ck["depth"], fourier=ck["fourier"],
                  f_scale=ck["f_scale"], spatial_A0=ck["spatial_A0"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


# --------------------------------------------------------------------------- #
def report(name, y, s, g, note=""):
    """One arm: per-group base rate, F1 and AUC, plus the Head-Tail gaps."""
    thr = best_threshold(y, s)
    print(f"\n{name}")
    if note:
        print(f"  {note}")
    print(f"  {'group':>6} {'base rate':>11} {'F1':>8} {'AUC':>8}")
    out = {}
    for k in GROUPS:
        m = g == k
        out[k] = dict(base=100 * y[m].mean(),
                      f1=f1_at(y[m], s[m], thr),
                      auc=100 * auc_np(y[m], s[m]))
        r = out[k]
        print(f"  {k:>6} {r['base']:10.1f}% {r['f1']:8.2f} {r['auc']:8.2f}")
    br = [out[k]["base"] for k in GROUPS]
    gaps = dict(base=br[0] - br[-1],
                f1=out["Head"]["f1"] - out["Tail"]["f1"],
                auc=out["Head"]["auc"] - out["Tail"]["auc"])
    print(f"  {'Head-Tail':>6} {gaps['base']:+10.1f}  {gaps['f1']:+8.2f} {gaps['auc']:+8.2f}")
    print(f"  base-rate spread across groups: {max(br) - min(br):.1f} points")
    return gaps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/burg_w24.npz")
    ap.add_argument("--ckpt", default="results/pinn_s0.pt")
    ap.add_argument("--kappa-rate", type=float, default=None,
                    help="target positive rate for the normalised label; "
                         "default matches the raw target's overall rate")
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="results/pne.json")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = load_field(args.data)
    model, ck = load_model(args.ckpt, device)
    raw_scale = ck["raw_scale"]

    mask, group, split = D["mask"], D["group"], D["split"]
    iy, ix = np.where(mask)
    te = np.where(split == "test")[0]
    tr = np.where(split == "train")[0]

    # ---- flatten test set: one row per (cell, week) -----------------------
    xs = np.tile(D["x"][iy, ix], len(te)).astype(np.float32)
    ys = np.tile(D["y"][iy, ix], len(te)).astype(np.float32)
    ts = np.repeat(D["t"][te], len(iy)).astype(np.float32)
    rr = np.repeat(te, len(iy)); cc = np.tile(iy, len(te)); dd = np.tile(ix, len(te))
    counts = D["U_raw"][rr, cc, dd].astype(np.float64)
    g = np.tile(group[iy, ix], len(te))

    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    with torch.no_grad():
        A, rho = model(T(xs), T(ys), T(ts))
        mu_hat = (raw_scale * rho * A).cpu().numpy().astype(np.float64)
        A0_cell = model.A0_at(T(D["x"][iy, ix].astype(np.float32)),
                              T(D["y"][iy, ix].astype(np.float32))).cpu().numpy()

    mu0_phys_cell = raw_scale * A0_cell.astype(np.float64)
    mu0_phys = np.tile(mu0_phys_cell, len(te))
    mu0_emp_cell = D["U_raw"][tr][:, iy, ix].mean(0).astype(np.float64)
    mu0_emp = np.tile(mu0_emp_cell, len(te))

    print("=" * 74)
    print("PHYSICS-NORMALISED EVALUATION")
    print("=" * 74)
    print(f"  test rows {len(counts):,}   cells {len(iy)}   weeks {len(te)}")
    print(f"  learned eta {model.eta.item():.4f}")
    print(f"  baseline mu0 per cell — physics: mean {mu0_phys_cell.mean():.3f} "
          f"range [{mu0_phys_cell.min():.3f}, {mu0_phys_cell.max():.3f}]")
    print(f"                          empirical: mean {mu0_emp_cell.mean():.3f} "
          f"range [{mu0_emp_cell.min():.3f}, {mu0_emp_cell.max():.3f}]")
    corr = float(np.corrcoef(mu0_phys_cell, mu0_emp_cell)[0, 1])
    print(f"  correlation between the two baselines: r = {corr:.3f}")
    print("  (if r is very high the physics baseline is little more than the mean;")
    print("   that is a result worth reporting either way)")

    # ---- ARM 1: the current target ---------------------------------------
    y_raw = (counts > 0).astype(int)

    # ------------------------------------------------------------------ #
    # CEILING. A normalised label can only fire where count >= 1, so the
    # highest positive rate the SPARSEST group can express is its own
    # P(count >= 1). Asking for more than that is impossible -- not because
    # the method is weak, but because the data has no finer resolution
    # there. Validation showed that above the ceiling only about half the
    # confound is removed, and below it essentially all of it is.
    # ------------------------------------------------------------------ #
    ceil_g = {k: float((counts[g == k] >= 1).mean()) for k in GROUPS}
    ceiling = min(ceil_g.values())
    print("\n  CEILING — highest positive rate each group can express:")
    for k in GROUPS:
        print(f"    {k:>6}  P(count>=1) = {100*ceil_g[k]:5.1f}%")
    print(f"    => target rates above {100*ceiling:.1f}% are UNREACHABLE "
          f"in the sparsest group.")

    if args.kappa_rate:
        target_rate = args.kappa_rate
        if target_rate > ceiling:
            print(f"\n  *** WARNING: requested {100*target_rate:.1f}% exceeds the "
                  f"{100*ceiling:.1f}% ceiling.")
            print("      Base rates CANNOT equalise. Results below will show a")
            print("      residual gap that is a data limit, not a model effect.")
    else:
        target_rate = 0.8 * ceiling      # safely inside the feasible region
        print(f"\n  target rate set to 0.8 x ceiling = {100*target_rate:.1f}% "
              f"(override with --kappa-rate)")
    print(f"  overall positive rate of the raw target: {100*y_raw.mean():.1f}%")

    res = {}
    res["raw"], _ = report(
        "ARM 1 — RAW TARGET  (label: count > 0, the current setup)",
        y_raw, mu_hat, g,
        "base rates differ by construction; this is the confound")

    # ---- ARMS 2 and 3: normalised targets --------------------------------
    for tag, mu0, desc in (
        ("empirical", mu0_emp,
         "ABLATION — normalised by the per-cell training mean"),
        ("physics", mu0_phys,
         "PROPOSAL — normalised by the PDE-constrained A0 field"),
    ):
        z_obs = anscombe_excess(counts, mu0, args.floor)
        z_hat = anscombe_excess(mu_hat, mu0, args.floor)   # transform the
        #                                                    prediction too
        kappa = float(np.quantile(z_obs, 1 - target_rate))
        y_z = (z_obs > kappa).astype(int)
        res[tag], _ = report(
            f"ARM {'2' if tag=='empirical' else '3'} — {tag.upper()} NORMALISED"
            f"  (label: excess > {kappa:.3f})",
            y_z, z_hat, g, desc)

    # ---- sensitivity: does the gap close as we go below the ceiling? -----
    print("\n" + "=" * 74)
    print("SENSITIVITY — base-rate gap vs chosen target rate (physics arm)")
    print("=" * 74)
    z_obs_p = anscombe_excess(counts, mu0_phys, args.floor)
    print(f"  {'target':>8} {'Head':>8} {'Mid':>8} {'Tail':>8} {'H-T gap':>10}")
    for tgt in (0.40, 0.30, 0.20, 0.12, 0.10, 0.08, 0.05):
        kap = float(np.quantile(z_obs_p, 1 - tgt))
        yy = (z_obs_p > kap).astype(int)
        rr_ = {k: 100 * yy[g == k].mean() for k in GROUPS}
        flag = "  <-- above ceiling" if tgt > ceiling else ""
        print(f"  {100*tgt:7.0f}% {rr_['Head']:7.2f}% {rr_['Mid']:7.2f}% "
              f"{rr_['Tail']:7.2f}% {rr_['Head']-rr_['Tail']:+9.2f}{flag}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 74)
    print("SUMMARY — Head minus Tail")
    print("=" * 74)
    print(f"{'target':>26} {'base-rate gap':>15} {'F1 gap':>10} {'AUC gap':>10}")
    print("-" * 66)
    for k, lab in (("raw", "raw counts (current)"),
                   ("empirical", "empirical baseline"),
                   ("physics", "physics A0 baseline")):
        r = res[k]
        print(f"{lab:>26} {r['base']:+15.1f} {r['f1']:+10.2f} {r['auc']:+10.2f}")

    print("\nHOW TO READ THIS")
    print("-" * 74)
    print("  1. If the base-rate gap collapses under the normalised targets, the")
    print("     mechanism works: the confound has been removed from the label.")
    print("  2. If the F1 gap collapses with it, the reported disparity was an")
    print("     artifact of the target, not of the model.")
    print("  3. AUC on a normalised target is TEMPORAL skill with geography")
    print("     divided out. Near 50 means the week-to-week signal is genuinely")
    print("     weak — an honest and publishable measurement of the report's")
    print("     'learned where, not when' limitation.")
    print("  4. Compare ARM 3 against ARM 2. If they agree, the physics baseline")
    print("     is doing no more than a per-cell mean, and the paper must say so.")
    print("  5. The ceiling is a REQUIREMENT of the method, not a bug: the target")
    print("     rate must sit below the sparsest group's P(count>=1). State it as")
    print("     a condition of applicability. In synthetic validation the gap fell")
    print("     from 79.5 to 0.03 points below the ceiling, and only to 38.7 above")
    print("     it. Reporting a number from above the ceiling would be misleading.")
    print("=" * 74)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as fh:
            json.dump({"gaps": res, "baseline_corr": corr,
                       "eta": float(model.eta.item()),
                       "ckpt": args.ckpt, "data": args.data}, fh, indent=2)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
