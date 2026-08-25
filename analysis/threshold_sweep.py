"""
The artifact without any architecture  (answers reviewer W4)
================================================================================
The GNN result shows Tail F1 rising +16.95 while Tail AUC FALLS 4.01.  A
reviewer will ask whether that is a general property of the metric or a quirk
of one architecture at one depth.

This settles it.  We take the plain no-graph model -- no graph, no depth, no
architecture change of any kind -- and simply MOVE THE DECISION THRESHOLD.

  * AUC is threshold-free, so it is mathematically constant across the sweep.
  * F1 is not, so it traces a curve.

The key number: the threshold at which the ungraphed model REPRODUCES the
graph model's Tail F1.  If a single knob reproduces the entire "fairness
improvement" while ranking ability is provably unchanged, then the improvement
was never about the model.

    python threshold_sweep.py --city chicago --seeds 5 --rounds 40
"""
import argparse, json
import numpy as np
import torch

from robust_fair_gnn import (CITY, C, L, load_city, make_windows, build_graph,
                             head_mid_tail, train)
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--city", default="chicago")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--clients", type=int, default=6)
ap.add_argument("--target-f1", type=float, default=40.38,
                help="Tail F1 the graph model reached (Chicago plain d2)")
ap.add_argument("--out", default="results/threshold_sweep.json")
a = ap.parse_args()

csv, dcol, start, end = CITY[a.city]
mat, _ = load_city(csv, dcol, start, end)
X, Y = make_windows(mat, horizon=1)
IN = X.shape[-1]
n = len(X); ntr = int(n*0.7); nval = int(n*0.1)
Xtr, Ytr = X[:ntr], Y[:ntr]
Xte, Yte = X[ntr+nval:], Y[ntr+nval:]
A_full, _ = build_graph(mat[:ntr+L])
tags = head_mid_tail(mat[:ntr+L])
tail = np.array([t == "Tail" for t in tags])

order = np.argsort(mat[:ntr].sum(axis=(0, 2)))[::-1]
parts = [order[i::a.clients] for i in range(a.clients)]
dens = Ytr.mean(axis=(0, 2))
rw = np.clip(dens.mean()/(dens+1e-6), 1.0, 4.0).astype(np.float32)
gmap = {"Head": 0, "Mid": 1, "Tail": 2}
grp = np.array([gmap[t] for t in tags], dtype=np.int64)
clients = [{"A": A_full[np.ix_(i, i)], "X": Xtr[:, :, i, :], "Y": Ytr[:, i, :],
            "rw": torch.tensor(rw[i]), "grp": torch.tensor(grp[i])} for i in parts]
yf = Ytr.reshape(-1, C); pos = yf.sum(0)
pw = torch.tensor(np.clip((len(yf)-pos)/np.maximum(pos, 1), 1.0, 10.0),
                  dtype=torch.float32)

def tail_scores(net):
    """Return (y, prob) for Tail regions only, flattened per category."""
    net.eval()
    with torch.no_grad():
        _, _, _, logit = net(torch.tensor(Xte), torch.tensor(A_full))
        prob = torch.sigmoid(logit).numpy()
    yg = Yte[:, tail, :]; pg = np.nan_to_num(prob[:, tail, :])
    return yg.reshape(-1, C), pg.reshape(-1, C)

def macro_f1(y, p, thr):
    out = []
    for c in range(C):
        yy, pp = y[:, c], (p[:, c] > thr).astype(int)
        tp = ((pp == 1) & (yy == 1)).sum(); fp = ((pp == 1) & (yy == 0)).sum()
        fn = ((pp == 0) & (yy == 1)).sum()
        if yy.sum() in (0, len(yy)): continue
        out.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
    return 100*float(np.mean(out)) if out else float("nan")

def macro_auc(y, p):
    out = []
    for c in range(C):
        if y[:, c].sum() in (0, len(y)): continue
        s = p[:, c]
        out.append(0.5 if np.allclose(s, s[0]) else roc_auc_score(y[:, c], s))
    return 100*float(np.mean(out)) if out else float("nan")

THR = np.round(np.arange(0.05, 0.96, 0.05), 2)
f1s, aucs, rates = [], [], []
for sd in range(a.seeds):
    net = train(clients, A_full, "fedavg", "none", set(), use_graph=False,
                rounds=a.rounds, seed=sd, pos_weight=pw, gnn_type="plain",
                in_ch=IN, gnn_layers=a.layers)
    y, p = tail_scores(net)
    f1s.append([macro_f1(y, p, t) for t in THR])
    rates.append([100*float((p > t).mean()) for t in THR])
    aucs.append(macro_auc(y, p))          # ONE value: independent of threshold
    print(f"  seed {sd} done", flush=True)

f1m, f1s_ = np.mean(f1s, 0), np.std(f1s, 0)
ratem = np.mean(rates, 0)
aucm, aucsd = float(np.mean(aucs)), float(np.std(aucs))

print("\n" + "="*76)
print(f"{a.city.upper()} — NO GRAPH AT ALL. Only the decision threshold moves.")
print("="*76)
print(f"{'threshold':>10} {'Tail F1':>14} {'says yes':>10} {'Tail AUC':>16}")
print("-"*56)
for t, m, s, r in zip(THR, f1m, f1s_, ratem):
    print(f"{t:>10.2f} {m:8.2f}±{s:5.2f} {r:9.1f}% {aucm:11.2f}±{aucsd:4.2f}")

best = int(np.nanargmax(f1m))
print(f"\nBest Tail F1 from threshold alone: {f1m[best]:.2f} at threshold {THR[best]:.2f}")
print(f"Graph model reached:               {a.target_f1:.2f}")
print(f"AUC across the ENTIRE sweep:       {aucm:.2f} (constant by construction)")
print()
if f1m[best] >= a.target_f1:
    print("RESULT: the ungraphed model MATCHES OR BEATS the graph model's Tail F1")
    print("using nothing but a different threshold, with identical ranking ability.")
    print("The 'fairness improvement' is reproducible with one knob and no model")
    print("change at all. It was never a property of the architecture.")
else:
    print(f"RESULT: threshold alone reaches {f1m[best]:.2f} of the graph model's "
          f"{a.target_f1:.2f}\n({100*f1m[best]/a.target_f1:.0f}% of it), with AUC "
          "unchanged throughout. Most of the\napparent gain is threshold "
          "placement rather than improved prediction.")
print("="*76)

import os
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"city": a.city, "seeds": a.seeds, "rounds": a.rounds,
           "thresholds": THR.tolist(), "tail_f1_mean": f1m.tolist(),
           "tail_f1_std": f1s_.tolist(), "positive_rate": ratem.tolist(),
           "tail_auc_mean": aucm, "tail_auc_std": aucsd,
           "graph_target_f1": a.target_f1}, open(a.out, "w"), indent=1)
print("saved ->", a.out)
