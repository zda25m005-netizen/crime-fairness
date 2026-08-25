"""
Is there any week-to-week signal to predict?  (run this BEFORE improving anything)
================================================================================
The PINN's per-group AUC sat near 50, which says it tells neighbourhoods apart
but not weeks apart.  Two explanations:

    (a) the model is not good enough yet        -> worth improving
    (b) the data has no week-to-week signal     -> no model will find it

This script decides between them WITHOUT training anything, by asking how much
of the variance three trivial predictors explain on the test period:

    1. global mean            - one number for everything
    2. per-cell mean          - the spatial map, no dynamics at all
    3. per-cell mean + lag-1  - the map plus last week's deviation

If (3) is barely better than (2), there is nothing temporal to learn and the
ceiling is the data, not the architecture.

    python signal_check.py --data data/burg_w24.npz
"""
import argparse
import numpy as np


def r2(true, pred):
    ss = ((true - true.mean()) ** 2).sum()
    return float(1 - ((true - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/burg_w24.npz")
    ap.add_argument("--raw", action="store_true",
                    help="use unsmoothed counts (harder, more honest)")
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    U = d["U_raw"] if a.raw else d["U"]
    mask, group, split = d["mask"], d["group"], d["split"]
    iy, ix = np.where(mask)
    X = U[:, iy, ix]                       # (T, cells)
    tr = np.where(split == "train")[0]
    te = np.where(split == "test")[0]

    cell_mean = X[tr].mean(0)              # the spatial map, fit on train only
    glob = X[tr].mean()

    print(f"{'field':>10}: {'raw counts' if a.raw else 'smoothed'}   "
          f"cells {X.shape[1]}   train {len(tr)}w   test {len(te)}w\n")

    # ---- how well can we do with no dynamics at all? ---------------------- #
    yt = X[te]
    p_glob = np.full_like(yt, glob)
    p_map = np.tile(cell_mean, (len(te), 1))
    # lag-1: map + a fraction of last week's deviation from the map
    prev = X[te - 1]
    dev = prev - cell_mean
    # fit the AR coefficient on TRAIN only
    dtr = X[tr[1:]] - cell_mean
    dtr_prev = X[tr[:-1]] - cell_mean
    beta = float((dtr_prev * dtr).sum() / max((dtr_prev ** 2).sum(), 1e-9))
    p_ar = cell_mean + beta * dev

    print(f"{'predictor':>34} {'R2 on test':>11}")
    print("-" * 47)
    print(f"{'1. global mean (one number)':>34} {r2(yt, p_glob):11.4f}")
    print(f"{'2. per-cell mean (the map)':>34} {r2(yt, p_map):11.4f}")
    print(f"{'3. map + last week (AR-1)':>34} {r2(yt, p_ar):11.4f}")
    print(f"\n  fitted lag-1 coefficient beta = {beta:+.4f}")
    gain = r2(yt, p_ar) - r2(yt, p_map)
    print(f"  temporal gain over the plain map = {gain:+.4f} R2")

    # ---- same question, per group ---------------------------------------- #
    G = group[iy, ix]
    print(f"\n{'group':>6} {'cells':>6} {'R2 map':>9} {'R2 +lag1':>9} {'gain':>8}")
    print("-" * 42)
    for g in ("Head", "Mid", "Tail"):
        m = (G == g)
        if m.sum() == 0:
            continue
        a_ = r2(yt[:, m], p_map[:, m]); b_ = r2(yt[:, m], p_ar[:, m])
        print(f"{g:>6} {m.sum():6d} {a_:9.4f} {b_:9.4f} {b_-a_:+8.4f}")

    # ---- lag-1 autocorrelation of the deviation --------------------------- #
    dall = X - cell_mean
    ac = float((dall[:-1] * dall[1:]).sum() /
               max(np.sqrt((dall[:-1] ** 2).sum() * (dall[1:] ** 2).sum()), 1e-9))
    print(f"\n  lag-1 autocorrelation of the deviation-from-map: {ac:+.4f}")

    print("\n" + "=" * 62)
    if gain < 0.01:
        print("VERDICT: essentially NO week-to-week signal beyond the spatial map.")
        print("  Improving the PINN will not create temporal skill that is not")
        print("  in the data. Report the map-learning honestly and stop tuning.")
    elif gain < 0.05:
        print("VERDICT: WEAK temporal signal. Some room, but small. Expect modest")
        print("  gains at best -- set expectations before spending GPU time.")
    else:
        print("VERDICT: real temporal signal exists. The PINN is leaving skill on")
        print("  the table and the Tier-1 changes are worth making.")
    print("=" * 62)


if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.modules:
        print("Add  %%writefile signal_check.py  as the FIRST line of this cell.")
    else:
        main()
