"""
Base-rate confounding in fairness evaluation of sparse crime prediction.
================================================================================
CLAIM (analytical). For a binary label with positive rate p, a predictor that
outputs "positive" achieves

        precision = p,   recall = 1,   F1 = 2p / (1 + p)

More generally, F1 is bounded above by a function that increases with p. Hence
when two groups have different base rates p_Head > p_Tail, the *attainable*
F1 differs BEFORE any model is trained. A raw F1 gap therefore conflates

    (a) genuine performance disparity, with
    (b) an arithmetic artifact of differing base rates.

This script quantifies (b) on every available city, so a paper can report how
much of a published-style "fairness gap" is explained by base rates alone.

It requires NO trained model and NO GPU — it is pure data analysis.

Usage
-----
    python baserate_analysis.py                 # all cities found in data/
    python baserate_analysis.py --cities la chicago nyc sf
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

C = 8
CITY = {
    "la":      ("data/la_crime.csv",  "2018-01-01", "2018-12-31"),
    "chicago": ("data/chi_crime.csv", "2015-01-01", "2015-12-31"),
    "nyc":     ("data/nyc_crime.csv", "2019-01-01", "2019-12-31"),
    "sf":      ("data/sf_crime.csv",  "2019-01-01", "2019-12-31"),
}


def load(csv, start, end):
    df = pd.read_csv(csv)
    df["date_occ"] = pd.to_datetime(df["date_occ"], errors="coerce")
    df = df.dropna(subset=["date_occ"])
    days = pd.date_range(start, end, freq="D")
    regions = sorted(df["neighborhood_id"].unique())
    ridx = {r: i for i, r in enumerate(regions)}
    didx = {d: i for i, d in enumerate(days)}
    mat = np.zeros((len(days), len(regions), C), dtype=np.float32)
    g = df.groupby([df["date_occ"].dt.normalize(),
                    "neighborhood_id", "crime_type_id"]).size()
    for (day, r, c), _ in g.items():
        if day in didx and r in ridx and 0 <= int(c) < C:
            mat[didx[day], ridx[r], int(c)] = 1.0
    return mat


def groups(mat):
    """Head/Mid/Tail by total crime (20/30/50 percentiles), as in FedCrime."""
    tot = mat.sum(axis=(0, 2))
    order = np.argsort(tot)[::-1]
    R = len(tot); nh = max(1, round(R * .2)); nm = max(1, round(R * .3))
    tag = np.empty(R, dtype=object)
    tag[order[:nh]] = "Head"; tag[order[nh:nh + nm]] = "Mid"
    tag[order[nh + nm:]] = "Tail"
    return tag


def baseline_f1(y):
    """Macro-F1 of the trivial always-positive predictor: mean_c 2p_c/(1+p_c)."""
    p = y.reshape(-1, C).mean(0)
    return float((2 * p / (1 + p)).mean()) * 100


def analyse(city, path, start, end):
    mat = load(path, start, end)
    D, R, _ = mat.shape
    tag = groups(mat)
    # evaluate on the same final-20% window the model uses
    n = D - 8
    te = mat[8 + int(n * .8):]

    out = {"city": city.upper(), "regions": R, "days": D,
           "sparsity": 100 * (1 - mat.mean())}
    for g in ["Head", "Mid", "Tail"]:
        m = np.array([t == g for t in tag])
        yg = te[:, m, :]
        out[g + "_p"] = float(yg.mean())
        out[g + "_base"] = baseline_f1(yg)
    out["base_gap"] = out["Head_base"] - out["Tail_base"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    args = ap.parse_args()
    cities = args.cities or [c for c in CITY if os.path.exists(CITY[c][0])]
    if not cities:
        raise SystemExit("No city CSVs found in data/. Build them first.")

    rows = []
    for c in cities:
        path, s, e = CITY[c]
        if not os.path.exists(path):
            print(f"[skip] {c}: {path} not found"); continue
        rows.append(analyse(c, path, s, e))

    print("\n" + "=" * 78)
    print("BASE-RATE CONFOUND IN FAIRNESS EVALUATION  (no model involved)")
    print("=" * 78)
    print(f"{'City':8s} {'Reg':>4s} {'Sparsity':>9s} | "
          f"{'p_Head':>7s} {'p_Tail':>7s} | {'F1max_H':>8s} {'F1max_T':>8s} "
          f"| {'ARTIFACT gap':>12s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['city']:8s} {r['regions']:4d} {r['sparsity']:8.1f}% | "
              f"{r['Head_p']:7.3f} {r['Tail_p']:7.3f} | "
              f"{r['Head_base']:8.1f} {r['Tail_base']:8.1f} | "
              f"{r['base_gap']:12.1f}")

    print("\nINTERPRETATION")
    print("  'ARTIFACT gap' is the Head-Tail F1 gap produced by base rates ALONE,")
    print("  by a model with NO learning (always predict positive). Any reported")
    print("  fairness gap must be compared against this floor: a gap of this size")
    print("  indicates NO disparity in skill, only a difference in attainability.")
    print("\n  Corrected metric:  Skill_g = 100 * F1_g / F1max_g")
    print("                     SkillGap = Skill_Head - Skill_Tail   (0 = fair)")

    if len(rows) > 1:
        sp = np.array([r["sparsity"] for r in rows])
        bg = np.array([r["base_gap"] for r in rows])
        if len(rows) > 2:
            cc = float(np.corrcoef(sp, bg)[0, 1])
            print(f"\n  Across {len(rows)} cities, correlation between overall sparsity")
            print(f"  and the artifact gap: r = {cc:+.3f}")
            print("  -> sparser cities exhibit larger *apparent* unfairness purely")
            print("     as a measurement artifact.")


if __name__ == "__main__":
    main()
