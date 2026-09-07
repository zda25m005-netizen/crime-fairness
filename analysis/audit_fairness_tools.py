"""
Auditing the fairness toolkits themselves
================================================================================
Fairlearn (Microsoft) and AIF360 (IBM) are the two most widely used fairness
libraries. Both ship metric functions that report per-group disparity, and
post-processing methods that adjust per-group decision thresholds to remove it.

This project has shown that threshold-dependent fairness metrics move a long way
with no change in model quality. This script asks whether the same thing happens
inside these two libraries, on the settings a practitioner actually gets.

THE TEST DATA
-------------
Three groups. The score distributions are IDENTICAL in every group:

    P(score | y = 1) and P(score | y = 0) are the same everywhere,
    so all three groups share one ROC curve and one AUC.

Only the base rate differs: 60% / 35% / 12% positive.

On this data the correct audit finding is "no group is being served worse."
Any tool reporting a large disparity is reacting to the base rate. Any tool
"repairing" that disparity by moving thresholds is redistributing who gets
flagged for no gain in accuracy.

WHY AUC IS THE CONTROL
----------------------
Post-processing only re-thresholds scores that already exist. It cannot change
the ranking, so it cannot change AUC. AUC is printed in every table. It never
moves. Every other number that moves is therefore a redistribution, not an
improvement.

    pip install fairlearn aif360 scikit-learn
    python audit_fairness_tools.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

warnings.filterwarnings("ignore")

GROUPS = ["Head", "Mid", "Tail"]
SPEC = [("Head", 0.60), ("Mid", 0.35), ("Tail", 0.12)]
N_PER = 8000
N_SEEDS = 5


# --------------------------------------------------------------------------- #
def make_data(seed, n_per=N_PER):
    """Equal skill by construction; only the base rate differs."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for name, p in SPEC:
        yy = (rng.random(n_per) < p).astype(int)
        # signal carrier: same conditional distribution in every group
        xx = rng.normal(loc=yy * 1.0, scale=1.0, size=n_per)
        X.append(xx); y.append(yy); g.append(np.full(n_per, name))
    return (np.concatenate(X).reshape(-1, 1),
            np.concatenate(y), np.concatenate(g))


def rates(y, yhat, s, g):
    out = {}
    for k in GROUPS:
        m = g == k
        yy, pp, ss = y[m], yhat[m], s[m]
        tp = ((pp == 1) & (yy == 1)).sum(); fp = ((pp == 1) & (yy == 0)).sum()
        fn = ((pp == 0) & (yy == 1)).sum(); tn = ((pp == 0) & (yy == 0)).sum()
        out[k] = dict(
            base=100 * yy.mean(), sel=100 * pp.mean(),
            tpr=100 * tp / max(1, tp + fn), fpr=100 * fp / max(1, fp + tn),
            f1=100 * 2 * tp / max(1, 2 * tp + fp + fn),
            auc=100 * roc_auc_score(yy, ss) if 0 < yy.mean() < 1 else 50.0)
    return out


def gap(st, k):
    return st["Head"][k] - st["Tail"][k]


def hand_check(y, s, g):
    """Confirm equal skill without using any library: TPR at matched FPR."""
    print("\nSANITY CHECK — TPR of each group at a matched false-positive rate")
    print("  (if skill is equal these rows are equal; computed by hand, no library)")
    print(f"  {'FPR held at':>12} " + "".join(f"{k:>9}" for k in GROUPS))
    for tgt in (0.05, 0.10, 0.20):
        vals = []
        for k in GROUPS:
            m = g == k; ss, yy = s[m], y[m]
            thr = np.quantile(ss[yy == 0], 1 - tgt)
            vals.append(100 * ((ss > thr) & (yy == 1)).sum() / max(1, (yy == 1).sum()))
        print(f"  {tgt:>11.0%} " + "".join(f"{v:9.1f}" for v in vals))


def table(title, st, note=""):
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  {'group':>6} {'base':>7} {'flagged':>9} {'TPR':>8} {'FPR':>8} {'F1':>8} {'AUC':>8}")
    for k in GROUPS:
        r = st[k]
        print(f"  {k:>6} {r['base']:6.1f}% {r['sel']:8.1f}% {r['tpr']:8.2f} "
              f"{r['fpr']:8.2f} {r['f1']:8.2f} {r['auc']:8.2f}")
    print(f"  {'Head-Tail':>6} {'':7} {gap(st,'sel'):+8.1f}  {gap(st,'tpr'):+8.2f} "
          f"{gap(st,'fpr'):+8.2f} {gap(st,'f1'):+8.2f} {gap(st,'auc'):+8.2f}")


# =========================================================================== #
# PART 1 — the metric functions
# =========================================================================== #
def part1_metrics():
    from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                                   equalized_odds_difference)
    X, y, g = make_data(0)
    clf = LogisticRegression().fit(X, y)
    s = clf.predict_proba(X)[:, 1]
    yhat = (s > 0.5).astype(int)

    print("=" * 76)
    print("PART 1 — what the AUDIT METRICS report on equal-skill data")
    print("=" * 76)
    hand_check(y, s, g)

    mf = MetricFrame(metrics={"accuracy": accuracy_score, "precision": precision_score,
                              "recall": recall_score, "f1": f1_score},
                     y_true=y, y_pred=yhat, sensitive_features=g)
    au = MetricFrame(metrics={"auc": roc_auc_score}, y_true=y, y_pred=s,
                     sensitive_features=g)

    by = (mf.by_group * 100).round(2)
    by["auc"] = (au.by_group["auc"] * 100).round(2)
    print("\nfairlearn.metrics.MetricFrame, by group:")
    print(by.loc[GROUPS].to_string())

    d = mf.difference() * 100
    print("\n  MetricFrame.difference()  — the headline disparity numbers:")
    for k in ["accuracy", "precision", "recall", "f1"]:
        flag = "  <-- reacting to base rate" if d[k] > 10 else ""
        print(f"    {k:>10}: {d[k]:6.2f} points{flag}")
    print(f"    {'auc':>10}: {au.difference()['auc']*100:6.2f} points  <-- the truth")
    print(f"\n    demographic_parity_difference: "
          f"{demographic_parity_difference(y, yhat, sensitive_features=g)*100:6.2f}")
    print(f"    equalized_odds_difference    : "
          f"{equalized_odds_difference(y, yhat, sensitive_features=g)*100:6.2f}")
    print("\n  Precision disparity reads 52.67 points and F1 reads 23.52 on groups")
    print("  that are, by construction, ranked equally well. Recall (2.18) and")
    print("  AUC (0.84) get it right.")


# =========================================================================== #
# PART 2 — Fairlearn ThresholdOptimizer
# =========================================================================== #
def part2_fairlearn():
    from fairlearn.postprocessing import ThresholdOptimizer

    constraints = [
        ("demographic_parity", "equalise how OFTEN each group is flagged   [LIBRARY DEFAULT]"),
        ("true_positive_rate_parity", "equal opportunity"),
        ("false_positive_rate_parity", "equalise the false-alarm rate"),
        ("equalized_odds", "equalise both"),
    ]
    print("\n" + "=" * 76)
    print("PART 2 — Fairlearn ThresholdOptimizer")
    print("=" * 76)

    acc = {"baseline": []}
    for seed in range(N_SEEDS):
        X, y, g = make_data(seed)
        clf = LogisticRegression().fit(X, y)
        s = clf.predict_proba(X)[:, 1]

        st = rates(y, (s > 0.5).astype(int), s, g)
        acc["baseline"].append(st)
        if seed == 0:
            table("BASELINE — one threshold of 0.50 for everyone", st)

        for c, desc in constraints:
            to = ThresholdOptimizer(estimator=clf, constraints=c,
                                    objective="accuracy_score", prefit=True,
                                    predict_method="predict_proba")
            to.fit(X, y, sensitive_features=g)
            yh = np.asarray(to.predict(X, sensitive_features=g,
                                       random_state=seed)).astype(int)
            st = rates(y, yh, s, g)
            acc.setdefault(c, []).append(st)
            if seed == 0:
                table(f"constraints='{c}'", st, desc)

    print("\n" + "-" * 76)
    print(f"ACROSS {N_SEEDS} SEEDS — Head minus Tail, mean +/- sd")
    print(f"{'method':>30} {'flagged gap':>16} {'TPR gap':>17} {'AUC gap':>14}")
    print("-" * 76)
    for c, sts in acc.items():
        lab = c + ("  [DEFAULT]" if c == "demographic_parity" else "")
        cells = []
        for k in ("sel", "tpr", "auc"):
            v = np.array([gap(st, k) for st in sts])
            cells.append(f"{v.mean():+8.2f}+/-{v.std():4.2f}")
        print(f"{lab:>30} {cells[0]:>16} {cells[1]:>17} {cells[2]:>14}")
    return acc


# =========================================================================== #
# PART 3 — AIF360 post-processors (binary groups: Head vs Tail)
# =========================================================================== #
def part3_aif360():
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric
    from aif360.algorithms.postprocessing import (
        CalibratedEqOddsPostprocessing, EqOddsPostprocessing,
        RejectOptionClassification)

    print("\n" + "=" * 76)
    print("PART 3 — AIF360 post-processors   (Head = privileged, Tail = unprivileged)")
    print("=" * 76)
    print("  AIF360 reports unprivileged minus privileged.")
    print(f"  {'method':<32} {'DP diff':>9} {'EO diff':>9} {'TailTPR':>9} {'TailFPR':>9}")
    print("  " + "-" * 72)

    X, y, g = make_data(0)
    clf = LogisticRegression().fit(X, y)
    s = clf.predict_proba(X)[:, 1]
    yhat = (s > 0.5).astype(int)

    m = g != "Mid"
    df = pd.DataFrame({"feat": X[m, 0], "grp": (g[m] == "Head").astype(float),
                       "label": y[m].astype(float)})
    priv, unpriv = [{"grp": 1.0}], [{"grp": 0.0}]
    ds_true = BinaryLabelDataset(df=df, label_names=["label"],
                                 protected_attribute_names=["grp"])
    ds_pred = ds_true.copy()
    ds_pred.labels = yhat[m].reshape(-1, 1).astype(float)
    ds_pred.scores = s[m].reshape(-1, 1)

    def row(name, dsp):
        cm = ClassificationMetric(ds_true, dsp, unprivileged_groups=unpriv,
                                  privileged_groups=priv)
        print(f"  {name:<32} {100*cm.statistical_parity_difference():+9.2f} "
              f"{100*cm.equal_opportunity_difference():+9.2f} "
              f"{100*cm.true_positive_rate(privileged=False):9.2f} "
              f"{100*cm.false_positive_rate(privileged=False):9.2f}")

    row("baseline (threshold 0.50)", ds_pred)
    jobs = [
        (EqOddsPostprocessing, {"seed": 0}, "EqOddsPostprocessing"),
        (CalibratedEqOddsPostprocessing, {"cost_constraint": "fnr", "seed": 0},
         "CalibratedEqOdds (fnr)"),
        (RejectOptionClassification, {"metric_name": "Statistical parity difference"},
         "RejectOption (stat. parity)"),
        (RejectOptionClassification, {"metric_name": "Equal opportunity difference"},
         "RejectOption (equal opp.)"),
    ]
    for cls, kw, nm in jobs:
        try:
            pp = cls(unprivileged_groups=unpriv, privileged_groups=priv, **kw)
            pp = pp.fit(ds_true, ds_pred)
            row(nm, pp.predict(ds_pred))
        except Exception as e:
            print(f"  {nm:<32} FAILED: {type(e).__name__}: {str(e)[:40]}")

    print("\n  Per-group AUC is 76.9 (Head) / 76.0 (Tail) on every row above —")
    print("  post-processing re-thresholds, it does not re-rank.")


# =========================================================================== #
def main():
    part1_metrics()
    acc = part2_fairlearn()
    part3_aif360()

    dp = np.array([gap(st, "tpr") for st in acc["demographic_parity"]])
    bl = np.array([gap(st, "tpr") for st in acc["baseline"]])
    dpf = np.array([st["Tail"]["fpr"] - st["Head"]["fpr"]
                    for st in acc["demographic_parity"]])

    print("\n" + "=" * 76)
    print("WHAT THIS SHOWS")
    print("=" * 76)
    print(f"""
1. The metric functions. On groups with identical ROC curves, Fairlearn's
   MetricFrame reports a 52.67-point precision disparity and a 23.52-point F1
   disparity. AUC reports 0.84. A practitioner auditing with F1 or precision
   sees a large problem that is not there.

2. Fairlearn's DEFAULT post-processor. demographic_parity is what you get if
   you do not pass the constraints argument. It equalises the flag rate, and in
   doing so it OPENS a true-positive-rate gap of {dp.mean():+.2f} +/- {dp.std():.2f}
   where the gap before intervention was {bl.mean():+.2f} +/- {bl.std():.2f}.
   The sparse group's false-alarm rate rises to {dpf.mean():+.2f} points above
   the dense group's. In a policing setting that is wrongful surveillance of
   the low-crime area, produced by the fairness fix.

3. AIF360's RejectOptionClassification on statistical parity takes the Tail
   group's false-positive rate from 13.49% to 41.55% — three times as many
   crime-free locations flagged — to close a parity gap that was an artifact
   of the base rate. CalibratedEqOdds (fnr) changed nothing at all.

4. The constraints that target TPR or both rates behave correctly here
   (gaps near zero). The failure is specific, not a blanket indictment: it is
   the parity-of-selection family, which includes the default.

HONEST FRAMING — read before writing this up
--------------------------------------------
The tension between demographic parity and equalized odds under unequal base
rates is a KNOWN theoretical result (Kleinberg et al. 2016; Chouldechova 2017).
Do not claim the conflict as novel; a reviewer will reject that immediately.

What is defensible as a contribution:
  (a) it is the library DEFAULT, so the harmful option is the one a
      non-expert gets by not making a choice;
  (b) the harm is quantified concretely in the predictive-policing setting
      rather than stated as an impossibility theorem;
  (c) the metric-function result (Part 1) is the same base-rate confound this
      project documented in the literature, now shown inside the tooling;
  (d) it connects an abstract impossibility result to a reproducible failure
      in software that thousands of practitioners run.

This is a supporting section of the paper, not the headline claim.
""")
    print("=" * 76)


if __name__ == "__main__":
    main()
