"""
Does the plain-GCN Tail F1 gain reflect real ranking skill, or a moved threshold?
================================================================================
Trains ONLY the two configurations in question -- plain GCN at depth 2 and its
capacity-matched no-graph control -- and reports base-rate DEPENDENT metrics
(F1, precision) beside base-rate INVARIANT ones (AUC, balanced accuracy, TPR).

  Tail AUC rises      -> the graph genuinely ranks sparse regions better.
  Tail AUC flat       -> F1 only slid toward its 2p/(1+p) ceiling; the "gain"
                         is the very artifact this project is about.

Imports robust_fair_gnn rather than editing it, so a "Run all" that rewrites
that file cannot silently undo this.

    python auc_check.py --city chicago --depth 2 --seeds 5 --rounds 40
"""
import argparse, json
import numpy as np
import torch

from robust_fair_gnn import (CITY, C, L, load_city, make_windows, build_graph,
                             head_mid_tail, train, evaluate)

ap = argparse.ArgumentParser()
ap.add_argument("--city", default="chicago")
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--gnn", default="plain")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--clients", type=int, default=6)
ap.add_argument("--out", default="results/auc_check.json")
a = ap.parse_args()

csv, dcol, start, end = CITY[a.city]
mat, _ = load_city(csv, dcol, start, end)
print(f"{a.city.upper()}: {mat.shape[1]} regions, {mat.shape[0]} days, "
      f"sparsity {100*(1-mat.mean()):.1f}% zeros")

X, Y = make_windows(mat, horizon=1)
IN = X.shape[-1]
n = len(X); ntr = int(n*0.7); nval = int(n*0.1)
Xtr, Ytr = X[:ntr], Y[:ntr]
Xte, Yte = X[ntr+nval:], Y[ntr+nval:]
A_full, _ = build_graph(mat[:ntr+L])
tags = head_mid_tail(mat[:ntr+L])

order = np.argsort(mat[:ntr].sum(axis=(0, 2)))[::-1]
parts = [order[i::a.clients] for i in range(a.clients)]
dens_r = Ytr.mean(axis=(0, 2))
rw = np.clip(dens_r.mean()/(dens_r+1e-6), 1.0, 4.0).astype(np.float32)
gmap = {"Head": 0, "Mid": 1, "Tail": 2}
grp = np.array([gmap[t] for t in tags], dtype=np.int64)
clients = [{"A": A_full[np.ix_(i, i)], "X": Xtr[:, :, i, :], "Y": Ytr[:, i, :],
            "rw": torch.tensor(rw[i]), "grp": torch.tensor(grp[i])}
           for i in parts]
yflat = Ytr.reshape(-1, C); pos = yflat.sum(0)
pos_weight = torch.tensor(np.clip((len(yflat)-pos)/np.maximum(pos, 1), 1.0, 10.0),
                          dtype=torch.float32)

KEYS = ["Head", "Mid", "Tail", "Head_prec", "Tail_prec",
        "Head_auc", "Mid_auc", "Tail_auc", "gap_auc",
        "Head_bal", "Tail_bal", "gap_bal", "Head_tpr", "Tail_tpr", "gap_tpr",
        "overall", "fairness_gap", "skill_gap"]

def run(use_graph):
    acc = {k: [] for k in KEYS}
    for sd in range(a.seeds):
        net = train(clients, A_full, "fedavg", "none", set(),
                    use_graph=use_graph, rounds=a.rounds, seed=sd,
                    pos_weight=pos_weight, gnn_type=a.gnn, in_ch=IN,
                    gnn_layers=a.depth)
        r = evaluate(net, Xte, Yte, A_full, tags)
        for k in acc:
            acc[k].append(r.get(k, float("nan")))
        print(f"   seed {sd} done", flush=True)
    return ({k: float(np.nanmean(v)) for k, v in acc.items()},
            {k: float(np.nanstd(v)) for k, v in acc.items()})

print(f"\ntraining GRAPH ({a.gnn}, depth {a.depth})...")
mg, sg = run(True)
print(f"training CONTROL (no graph, {a.depth} dense layers)...")
mc, sc = run(False)

BAND = [("BASE-RATE DEPENDENT  (can move without real skill)",
         [("Tail F1", "Tail"), ("Head F1", "Head"),
          ("Tail precision", "Tail_prec"), ("Head precision", "Head_prec"),
          ("raw gap", "fairness_gap"), ("skill gap", "skill_gap")]),
        ("BASE-RATE INVARIANT  (real ranking skill)",
         [("Tail AUC", "Tail_auc"), ("Head AUC", "Head_auc"),
          ("AUC gap", "gap_auc"), ("Tail bal-acc", "Tail_bal"),
          ("bal-acc gap", "gap_bal"), ("Tail TPR", "Tail_tpr"),
          ("TPR gap", "gap_tpr")])]

print("\n" + "="*72)
print(f"{a.city.upper()} — {a.gnn} depth {a.depth} vs matched control "
      f"({a.seeds} seeds, {a.rounds} rounds)")
print("="*72)
for title, items in BAND:
    print("\n  " + title)
    print(f"  {'metric':>15} {'control':>14} {'graph':>14} {'delta':>9}")
    print("  " + "-"*56)
    for lab, k in items:
        print(f"  {lab:>15} {mc[k]:8.2f}±{sc[k]:5.2f} {mg[k]:8.2f}±{sg[k]:5.2f} "
              f"{mg[k]-mc[k]:+9.2f}")

d_f1, d_auc = mg["Tail"]-mc["Tail"], mg["Tail_auc"]-mc["Tail_auc"]
print("\n" + "="*72)
print(f"VERDICT   Tail F1 {d_f1:+.2f}   Tail AUC {d_auc:+.2f}")
if d_auc > 2.0:
    print("  -> AUC moved with F1: the graph genuinely ranks tail regions")
    print("     better. This is a real fairness result.")
elif abs(d_auc) <= 2.0 and d_f1 > 5:
    print("  -> F1 jumped while AUC stayed flat: the model is predicting MORE")
    print("     POSITIVES in sparse regions, sliding F1 toward 2p/(1+p) without")
    print("     gaining skill. That is the artifact, reproduced inside our own")
    print("     model. Check Tail precision above: it should have fallen.")
else:
    print("  -> mixed / no clear effect; do not claim either story.")
print("="*72)

import os
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"city": a.city, "gnn": a.gnn, "depth": a.depth, "seeds": a.seeds,
           "rounds": a.rounds, "graph_mean": mg, "graph_std": sg,
           "control_mean": mc, "control_std": sc}, open(a.out, "w"), indent=1)
print("saved ->", a.out)
