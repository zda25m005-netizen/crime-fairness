"""
Final PINN table: mean +/- std over seeds, with Welch t-tests.
================================================================================
Reads results/pinn.jsonl and compares each configuration against the
no-physics control. Reports the base-rate INVARIANT metric (AUC) beside the
base-rate DEPENDENT one (F1), because this project has already shown that F1
can move a long way without any change in real skill.

    python analyse_pinn.py --file results/pinn.jsonl
"""
import argparse, json, math
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="results/pinn.jsonl")
ap.add_argument("--min-seeds", type=int, default=2)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.file) if l.strip()]
print(f"{len(rows)} runs in {a.file}\n")

def key(r):
    ph = r.get("physics", "short" if r.get("pde_weight", 1) > 0 else "none")
    return f"{ph:6s} + {r.get('loss','mse'):7s}"

G = defaultdict(list)
for r in rows:
    G[key(r)].append(r)

def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"),)*2 + (0,)

def welch(a1, a2):
    a1 = np.asarray([x for x in a1 if x is not None]); a2 = np.asarray([x for x in a2 if x is not None])
    if len(a1) < 2 or len(a2) < 2: return float("nan"), float("nan")
    s1, s2 = a1.var(ddof=1)/len(a1), a2.var(ddof=1)/len(a2)
    se = math.sqrt(s1 + s2)
    if se == 0: return float("nan"), float("nan")
    t = (a1.mean() - a2.mean()) / se
    df = (s1+s2)**2 / (s1**2/(len(a1)-1) + s2**2/(len(a2)-1))
    return t, df

# ---------------------------------------------------------------- main table
print("=" * 84)
print("OVERALL  (mean +/- std over seeds)")
print("=" * 84)
print(f"{'configuration':>18} {'n':>3} {'AUC':>14} {'F1':>14} {'RMSE':>14} {'eta':>9}")
print("-" * 84)
order = sorted(G, key=lambda k: -ms([r['ALL_auc'] for r in G[k]])[0])
for k in order:
    R = G[k]
    if len(R) < a.min_seeds: continue
    au, aus, n = ms([r['ALL_auc'] for r in R])
    f1, f1s, _ = ms([r['ALL_f1'] for r in R])
    rm, rms, _ = ms([r['rmse'] for r in R])
    et, ets, _ = ms([r.get('eta') for r in R])
    print(f"{k:>18} {n:3d} {au:8.2f}±{aus:5.2f} {f1:8.2f}±{f1s:5.2f} "
          f"{rm:8.3f}±{rms:5.3f} {et:9.4f}")

# ---------------------------------------------------------------- vs control
ctrl = next((k for k in G if k.strip().startswith("none")), None)
if ctrl:
    print("\n" + "=" * 84)
    print(f"VERSUS CONTROL  ({ctrl.strip()})   ** = p<0.05")
    print("=" * 84)
    print(f"{'configuration':>18} {'dAUC':>8} {'t':>8} {'sig':>4} "
          f"{'dF1':>8} {'t':>8} {'sig':>4}")
    print("-" * 66)
    for k in order:
        if k == ctrl or len(G[k]) < a.min_seeds: continue
        for name, fld in (("auc", "ALL_auc"), ("f1", "ALL_f1")):
            pass
        t1, d1 = welch([r['ALL_auc'] for r in G[k]], [r['ALL_auc'] for r in G[ctrl]])
        t2, d2 = welch([r['ALL_f1'] for r in G[k]], [r['ALL_f1'] for r in G[ctrl]])
        crit = 2.78
        da = ms([r['ALL_auc'] for r in G[k]])[0] - ms([r['ALL_auc'] for r in G[ctrl]])[0]
        df_ = ms([r['ALL_f1'] for r in G[k]])[0] - ms([r['ALL_f1'] for r in G[ctrl]])[0]
        print(f"{k:>18} {da:+8.2f} {t1:8.2f} {'**' if abs(t1)>crit else '':>4} "
              f"{df_:+8.2f} {t2:8.2f} {'**' if abs(t2)>crit else '':>4}")

# ---------------------------------------------------------------- the point
print("\n" + "=" * 84)
print("DOES THE FAIRNESS METRIC TRACK REAL SKILL?")
print("=" * 84)
print(f"{'configuration':>18} {'pooled AUC':>13} {'Head-Tail F1 gap':>19} {'AUC gap':>11}")
print("-" * 66)
aucs, gaps = [], []
for k in order:
    if len(G[k]) < a.min_seeds: continue
    au = ms([r['ALL_auc'] for r in G[k]])[0]
    gf, gfs, _ = ms([r['gap_f1'] for r in G[k]])
    ga, gas, _ = ms([r['gap_auc'] for r in G[k]])
    aucs.append(au); gaps.append(gf)
    print(f"{k:>18} {au:13.2f} {gf:12.2f}±{gfs:5.2f} {ga:+7.2f}±{gas:4.2f}")
if len(aucs) > 1:
    print(f"\n  real skill (AUC) spans      {max(aucs)-min(aucs):6.2f} points")
    print(f"  the F1 'fairness gap' spans {max(gaps)-min(gaps):6.2f} points")
    print("\n  If the second number is small while the first is large, the metric a")
    print("  fairness audit would report is insensitive to genuine model quality.")

# ---------------------------------------------------------------- tail detail
print("\n" + "=" * 84)
print("TAIL REGIONS  (crime actually occurs in ~28.5% of weeks)")
print("=" * 84)
print(f"{'configuration':>18} {'Tail F1':>14} {'Tail AUC':>14} {'says yes':>14} {'x too often':>12}")
print("-" * 78)
for k in order:
    if len(G[k]) < a.min_seeds: continue
    f1, f1s, _ = ms([r['Tail_f1'] for r in G[k]])
    au, aus, _ = ms([r['Tail_auc'] for r in G[k]])
    ry, rys, _ = ms([r['Tail_rate'] for r in G[k]])
    br = ms([r['Tail_base'] for r in G[k]])[0]
    print(f"{k:>18} {f1:8.2f}±{f1s:5.2f} {au:8.2f}±{aus:5.2f} "
          f"{ry:8.1f}±{rys:4.1f}% {ry/max(br,1e-9):11.1f}x")
print("=" * 84)
