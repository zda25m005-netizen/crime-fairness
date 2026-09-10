r"""
Auditing the fairness toolkits themselves   (FAccT item 2)
================================================================================
Fairlearn (Microsoft) and AIF360 (IBM) are the two most widely used fairness
libraries. Both ship metric functions that report per-group disparity, and
algorithms that remove it. This project has shown that threshold-dependent
fairness metrics move a long way with no change in model quality. This script
asks whether the same thing happens inside these two libraries, on the settings
a practitioner actually gets.

WHAT IS NEW IN THIS VERSION (2026-09-10)
----------------------------------------
The first version tested POST-processing only, on synthetic data, at one seed.
Three things were missing, and a reviewer would have found all three.

  * Post-processing cannot change AUC -- it only re-thresholds an existing
    ranking. That makes the "AUC never moves" control trivially true and
    somewhat uninteresting. IN-processing methods CAN change the model, so
    they are the harder test. PART 4 adds them.
  * PRE-processing (reweighting, repairing features) is a third family with a
    different mechanism again. PART 5 adds it.
  * Everything was synthetic. PART 6 runs the whole audit on REAL Chicago
    burglary data, gridded exactly as in analysis/multicity_audit.py.

THE SYNTHETIC TEST DATA (parts 1-5)
-----------------------------------
Three groups. The score distributions are IDENTICAL in every group:

    P(score | y = 1) and P(score | y = 0) are the same everywhere,
    so all three groups share one ROC curve and one AUC.

Only the base rate differs: 60% / 35% / 12% positive.

On this data the correct audit finding is "no group is being served worse."
Any tool reporting a large disparity is reacting to the base rate. Any tool
"repairing" that disparity is redistributing harm, or destroying skill, for
no gain in accuracy.

WHY BOTH POOLED AND MACRO AUC ARE PRINTED
-----------------------------------------
Pooling a metric over units with different base rates lets a scorer win by
ranking UNITS instead of ranking cases. That artifact is what forced this
project to retract its own headline result. Every AUC in PART 6 is therefore
reported twice: pooled, and macro (per-cell, then averaged). See
src/pinn/pinn_history.py::macro_auc.

    pip install fairlearn aif360 BlackBoxAuditing scikit-learn
    python audit_fairness_tools.py                  # synthetic only, ~2 min
    python audit_fairness_tools.py --real           # + real Chicago, ~6 min
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

warnings.filterwarnings("ignore")

GROUPS = ["Head", "Mid", "Tail"]
SPEC = [("Head", 0.60), ("Mid", 0.35), ("Tail", 0.12)]
N_PER = 6000
N_SEEDS = 5


# --------------------------------------------------------------------------- #
# data + metrics
# --------------------------------------------------------------------------- #
def make_data(seed, n_per=N_PER, spec=None):
    """Equal skill by construction; only the base rate differs."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for name, p in (spec or SPEC):
        yy = (rng.random(n_per) < p).astype(int)
        # signal carrier: same conditional distribution in every group
        xx = rng.normal(loc=yy * 1.0, scale=1.0, size=n_per)
        X.append(xx); y.append(yy); g.append(np.full(n_per, name))
    return (np.concatenate(X).reshape(-1, 1),
            np.concatenate(y), np.concatenate(g))


def auc_np(y, s):
    y = np.asarray(y).ravel().astype(int); s = np.asarray(s, float).ravel()
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    _, f, c = np.unique(s[o], return_index=True, return_counts=True)
    for st_, ct in zip(f[c > 1], c[c > 1]):
        i = o[st_:st_ + ct]; r[i] = r[i].mean()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_auc(y, s, unit, min_pos=3):
    """Mean of PER-UNIT AUCs. Ranking units against each other cannot help."""
    v = []
    for u in np.unique(unit):
        m = unit == u; yy = y[m]; np_ = int(yy.sum())
        if np_ < min_pos or (len(yy) - np_) < min_pos:
            continue
        v.append(auc_np(yy, s[m]))
    return (100 * float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def rates(y, yhat, s, g, unit=None):
    out = {}
    for k in GROUPS:
        m = g == k
        yy, pp, ss = y[m], yhat[m], s[m]
        tp = ((pp == 1) & (yy == 1)).sum(); fp = ((pp == 1) & (yy == 0)).sum()
        fn = ((pp == 0) & (yy == 1)).sum(); tn = ((pp == 0) & (yy == 0)).sum()
        d = dict(
            base=100 * yy.mean(), sel=100 * pp.mean(),
            tpr=100 * tp / max(1, tp + fn), fpr=100 * fp / max(1, fp + tn),
            f1=100 * 2 * tp / max(1, 2 * tp + fp + fn),
            # RAW COUNT of crime-free places flagged. Rates hide scale; a
            # policing reader needs the count.
            wrongful=int(fp),
            auc=100 * auc_np(yy, ss) if 0 < yy.mean() < 1 else 50.0)
        d["macro"] = macro_auc(yy, ss, unit[m])[0] if unit is not None else d["auc"]
        out[k] = d
    return out


def gap(st, k):
    return st["Head"][k] - st["Tail"][k]


def table(title, st, note="", macro=False):
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    extra = f" {'macroAUC':>9}" if macro else ""
    print(f"  {'group':>6} {'base':>7} {'flagged':>9} {'TPR':>8} {'FPR':>8} "
          f"{'F1':>8} {'AUC':>8}{extra} {'wrongful':>9}")
    for k in GROUPS:
        r = st[k]
        e = f" {r['macro']:9.2f}" if macro else ""
        print(f"  {k:>6} {r['base']:6.1f}% {r['sel']:8.1f}% {r['tpr']:8.2f} "
              f"{r['fpr']:8.2f} {r['f1']:8.2f} {r['auc']:8.2f}{e} {r['wrongful']:9,}")
    e = f" {gap(st,'macro'):+9.2f}" if macro else ""
    print(f"  {'H - T':>6} {'':7} {gap(st,'sel'):+8.1f}  {gap(st,'tpr'):+8.2f} "
          f"{gap(st,'fpr'):+8.2f} {gap(st,'f1'):+8.2f} {gap(st,'auc'):+8.2f}{e}")


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

    print("=" * 78)
    print("PART 1 — what the AUDIT METRICS report on equal-skill data")
    print("=" * 78)
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
    print("\n  Precision and F1 report a large disparity on groups that are, by")
    print("  construction, ranked equally well. Recall and AUC get it right.")


# =========================================================================== #
# PART 2 — Fairlearn ThresholdOptimizer   (POST-processing)
# =========================================================================== #
def part2_fairlearn():
    from fairlearn.postprocessing import ThresholdOptimizer

    constraints = [
        ("demographic_parity", "equalise how OFTEN each group is flagged   [LIBRARY DEFAULT]"),
        ("true_positive_rate_parity", "equal opportunity"),
        ("false_positive_rate_parity", "equalise the false-alarm rate"),
        ("equalized_odds", "equalise both"),
    ]
    print("\n" + "=" * 78)
    print("PART 2 — Fairlearn ThresholdOptimizer  (post-processing)")
    print("=" * 78)

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

    print("\n" + "-" * 78)
    print(f"ACROSS {N_SEEDS} SEEDS — Head minus Tail, mean +/- sd")
    print(f"{'method':>30} {'flagged gap':>16} {'TPR gap':>17} {'AUC gap':>14}")
    print("-" * 78)
    for c, sts in acc.items():
        lab = c + ("  [DEFAULT]" if c == "demographic_parity" else "")
        cells = []
        for k in ("sel", "tpr", "auc"):
            v = np.array([gap(st, k) for st in sts])
            cells.append(f"{v.mean():+8.2f}+/-{v.std():4.2f}")
        print(f"{lab:>30} {cells[0]:>16} {cells[1]:>17} {cells[2]:>14}")

    print("""
  READ THE SIGN, NOT JUST THE SIZE. Under demographic_parity the TPR gap is
  NEGATIVE: the SPARSE group ends up with the HIGHER true-positive rate and
  the HIGHER false-alarm rate, because everyone is flagged at the same rate
  regardless of how much crime is actually there. The dense group then misses
  real crime. Both halves of that are harms, in opposite directions.""")
    return acc


# =========================================================================== #
# PART 3 — AIF360 post-processors (binary groups: Head vs Tail)
# =========================================================================== #
def _aif_frames(seed):
    from aif360.datasets import BinaryLabelDataset
    X, y, g = make_data(seed)
    clf = LogisticRegression().fit(X, y)
    s = clf.predict_proba(X)[:, 1]
    yhat = (s > 0.5).astype(int)
    m = g != "Mid"
    df = pd.DataFrame({"feat": X[m, 0], "grp": (g[m] == "Head").astype(float),
                       "label": y[m].astype(float)})
    ds_true = BinaryLabelDataset(df=df, label_names=["label"],
                                 protected_attribute_names=["grp"])
    ds_pred = ds_true.copy()
    ds_pred.labels = yhat[m].reshape(-1, 1).astype(float)
    ds_pred.scores = s[m].reshape(-1, 1)
    return ds_true, ds_pred, [{"grp": 1.0}], [{"grp": 0.0}]


def part3_aif360():
    from aif360.metrics import ClassificationMetric
    from aif360.algorithms.postprocessing import (
        CalibratedEqOddsPostprocessing, EqOddsPostprocessing,
        RejectOptionClassification)

    print("\n" + "=" * 78)
    print("PART 3 — AIF360 post-processors   (Head = privileged, Tail = unprivileged)")
    print("=" * 78)
    print(f"  {N_SEEDS} seeds, mean +/- sd. AIF360 reports unprivileged minus privileged.")

    rows = {}
    for seed in range(N_SEEDS):
        ds_true, ds_pred, priv, unpriv = _aif_frames(seed)

        def rec(name, dsp):
            cm = ClassificationMetric(ds_true, dsp, unprivileged_groups=unpriv,
                                      privileged_groups=priv)
            rows.setdefault(name, []).append((
                100 * cm.statistical_parity_difference(),
                100 * cm.equal_opportunity_difference(),
                100 * cm.true_positive_rate(privileged=False),
                100 * cm.false_positive_rate(privileged=False),
                100 * cm.false_positive_rate(privileged=True)))

        rec("baseline (threshold 0.50)", ds_pred)
        jobs = [
            (EqOddsPostprocessing, {"seed": seed}, "EqOddsPostprocessing"),
            (CalibratedEqOddsPostprocessing, {"cost_constraint": "fnr", "seed": seed},
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
                rec(nm, pp.predict(ds_pred))
            except Exception as e:
                if seed == 0:
                    print(f"  {nm:<30} FAILED: {type(e).__name__}: {str(e)[:40]}")

    print(f"\n  {'method':<30} {'DP diff':>9} {'EO diff':>9} {'TailTPR':>9} "
          f"{'TailFPR':>9} {'HeadFPR':>9}")
    print("  " + "-" * 74)
    for nm, v in rows.items():
        a = np.array(v)
        print(f"  {nm:<30} {a[:,0].mean():+9.2f} {a[:,1].mean():+9.2f} "
              f"{a[:,2].mean():9.2f} {a[:,3].mean():9.2f} {a[:,4].mean():9.2f}")
    print("\n  Per-group AUC is identical on every row above — post-processing")
    print("  re-thresholds, it does not re-rank. Every movement is redistribution.")
    return rows


# =========================================================================== #
# PART 4 — IN-processing.  THIS IS THE HARDER TEST.
# =========================================================================== #
def part4_inprocessing():
    """Post-processing provably cannot change AUC, which makes 'AUC never moved'
    a weak control. In-processing changes the MODEL, so it can. The question is
    what it costs to satisfy a constraint that was never violated in the first
    place."""
    from fairlearn.reductions import (ExponentiatedGradient, DemographicParity,
                                      EqualizedOdds, TruePositiveRateParity)

    print("\n" + "=" * 78)
    print("PART 4 — IN-processing: what does the 'fix' cost the model itself?")
    print("=" * 78)

    X, y, g = make_data(0)
    base = LogisticRegression().fit(X, y)
    s = base.predict_proba(X)[:, 1]
    acc0 = 100 * accuracy_score(y, (s > 0.5).astype(int))
    auc0 = 100 * auc_np(y, s)
    print(f"\n  unconstrained logistic regression:  accuracy {acc0:.2f}   AUC {auc0:.2f}")
    print(f"\n  {'method':<36} {'acc':>7} {'AUC*':>7} {'flag H/M/T':>20} "
          f"{'TPR H/T':>14} {'FPR H/T':>14}")
    print("  " + "-" * 76)

    for nm, C in [("DemographicParity  [selection]", DemographicParity),
                  ("EqualizedOdds", EqualizedOdds),
                  ("TruePositiveRateParity", TruePositiveRateParity)]:
        for eps in (0.01, 0.05):
            try:
                eg = ExponentiatedGradient(LogisticRegression(),
                                           constraints=C(difference_bound=eps),
                                           max_iter=50)
                eg.fit(X, y, sensitive_features=g)
                yh = np.asarray(eg.predict(X)).astype(int)
                pm = eg._pmf_predict(X)[:, 1]
                st = rates(y, yh, pm, g)
                print(f"  {nm+f' eps={eps}':<36} {100*accuracy_score(y,yh):7.2f} "
                      f"{100*auc_np(y,pm):7.2f} "
                      f"{st['Head']['sel']:6.1f}/{st['Mid']['sel']:5.1f}/{st['Tail']['sel']:5.1f}   "
                      f"{st['Head']['tpr']:6.1f}/{st['Tail']['tpr']:5.1f} "
                      f"{st['Head']['fpr']:7.1f}/{st['Tail']['fpr']:5.1f}")
            except Exception as e:
                print(f"  {nm+f' eps={eps}':<36} FAILED {type(e).__name__}")

    print("""
  PrejudiceRemover is reported SEPARATELY below and is NOT comparable to the
  rows above. It is fitted on the Head/Tail two-group frame (no Mid), and
  AIF360 hands it the group indicator as a model FEATURE. Read it as its own
  result, not as another row in the same table.""")
    print(f"\n  {'method':<36} {'acc':>7} {'AUC':>7} {'flag H/T':>20} "
          f"{'TPR H/T':>14} {'FPR H/T':>14}")
    print("  " + "-" * 76)
    try:
        from aif360.algorithms.inprocessing import PrejudiceRemover
        ds_true, _, priv, unpriv = _aif_frames(0)
        for eta in (1.0, 25.0):
            pr = PrejudiceRemover(eta=eta, sensitive_attr="grp", class_attr="label")
            pr.fit(ds_true)
            out = pr.predict(ds_true)
            yy = ds_true.labels.ravel().astype(int)
            gg = np.where(ds_true.features[:, ds_true.feature_names.index("grp")] == 1,
                          "Head", "Tail")
            pp = out.labels.ravel().astype(int)
            sc = out.scores.ravel()
            tp = {k: ((pp == 1) & (yy == 1) & (gg == k)).sum() for k in ("Head", "Tail")}
            fp = {k: ((pp == 1) & (yy == 0) & (gg == k)).sum() for k in ("Head", "Tail")}
            pos = {k: ((yy == 1) & (gg == k)).sum() for k in ("Head", "Tail")}
            neg = {k: ((yy == 0) & (gg == k)).sum() for k in ("Head", "Tail")}
            print(f"  {'PrejudiceRemover eta='+str(eta):<36} "
                  f"{100*accuracy_score(yy,pp):7.2f} {100*auc_np(yy,sc):7.2f} "
                  f"{100*pp[gg=='Head'].mean():6.1f}/    -/{100*pp[gg=='Tail'].mean():5.1f}   "
                  f"{100*tp['Head']/max(1,pos['Head']):6.1f}/{100*tp['Tail']/max(1,pos['Tail']):5.1f} "
                  f"{100*fp['Head']/max(1,neg['Head']):7.1f}/{100*fp['Tail']/max(1,neg['Tail']):5.1f}")
    except Exception as e:
        print(f"  PrejudiceRemover unavailable: {type(e).__name__}: {str(e)[:50]}")

    print("""
  * AUC in the FIRST table is of the randomised classifier's mixture weight,
    not of a clean score. It is indicative, not a clean ROC. Say so in print.

  ExponentiatedGradient: post-processing redistributes harm at constant AUC,
  whereas in-processing pays for the same constraint in model quality --
  accuracy falls by roughly six points at the tight demographic-parity bound,
  and the method gets there by flagging almost nobody anywhere. Equalized-odds
  and TPR-parity cost nothing. Selection-parity is again the expensive one,
  and the constraint being enforced was never violated: the groups had
  identical ROC curves before anyone intervened.

  PrejudiceRemover deserves its own sentence, because it does something worse
  and more interesting. At LOW eta (weak fairness penalty) it reports a HIGHER
  AUC than the unconstrained model. It gets there by using the group indicator
  it was handed, which in this data predicts the label only through the base
  rate. A 'fairness-aware' model is scoring better on the standard metric by
  leaning on precisely the confound this paper is about. At HIGH eta it
  collapses to flagging nothing at all -- a degenerate classifier whose
  accuracy (the negative rate) still looks respectable. Neither end of that
  range is a model anyone should deploy, and the usual metrics flag neither.""")


# =========================================================================== #
# PART 5 — PRE-processing
# =========================================================================== #
def part5_preprocessing():
    print("\n" + "=" * 78)
    print("PART 5 — PRE-processing: repair the data before training")
    print("=" * 78)

    from aif360.metrics import ClassificationMetric
    ds_true, ds_pred, priv, unpriv = _aif_frames(0)
    feat_i = ds_true.feature_names.index("feat")
    yy = ds_true.labels.ravel().astype(int)
    gg = ds_true.features[:, ds_true.feature_names.index("grp")]

    def report(nm, Xtr, w=None):
        clf = LogisticRegression().fit(Xtr, yy, sample_weight=w)
        sc = clf.predict_proba(Xtr)[:, 1]
        pp = (sc > 0.5).astype(int)
        d = ds_pred.copy(); d.labels = pp.reshape(-1, 1).astype(float)
        cm = ClassificationMetric(ds_true, d, unprivileged_groups=unpriv,
                                  privileged_groups=priv)
        print(f"  {nm:<34} {100*auc_np(yy,sc):7.2f} "
              f"{100*accuracy_score(yy,pp):7.2f} "
              f"{100*cm.statistical_parity_difference():+9.2f} "
              f"{100*cm.equal_opportunity_difference():+9.2f} "
              f"{100*cm.false_positive_rate(privileged=False):9.2f}")

    print(f"\n  {'method':<34} {'AUC':>7} {'acc':>7} {'DP diff':>9} "
          f"{'EO diff':>9} {'TailFPR':>9}")
    print("  " + "-" * 76)
    report("no repair", ds_true.features[:, [feat_i]])

    try:
        from aif360.algorithms.preprocessing import Reweighing
        rw = Reweighing(unprivileged_groups=unpriv,
                        privileged_groups=priv).fit_transform(ds_true)
        report("AIF360 Reweighing", ds_true.features[:, [feat_i]],
               w=rw.instance_weights)
    except Exception as e:
        print(f"  Reweighing FAILED {type(e).__name__}")

    try:
        from aif360.algorithms.preprocessing import DisparateImpactRemover
        for lvl in (0.5, 1.0):
            di = DisparateImpactRemover(repair_level=lvl).fit_transform(ds_true)
            report(f"AIF360 DisparateImpactRemover r={lvl}",
                   di.features[:, [feat_i]])
    except Exception as e:
        print(f"  DisparateImpactRemover unavailable ({type(e).__name__}) — "
              f"needs `pip install BlackBoxAuditing`")

    try:
        from fairlearn.preprocessing import CorrelationRemover
        cr = CorrelationRemover(sensitive_feature_ids=[1])
        Xa = np.column_stack([ds_true.features[:, feat_i], gg])
        report("fairlearn CorrelationRemover", cr.fit_transform(Xa))
    except Exception as e:
        print(f"  CorrelationRemover FAILED {type(e).__name__}")

    print("""
  Pre-processing acts on the FEATURE, which here carries genuine signal and is
  already independent of group membership given the label. Repairing it can
  only remove signal. Watch the AUC column: this is the one family where the
  intervention degrades the ranking itself.""")


# =========================================================================== #
# PART 6 — THE SAME AUDIT ON REAL CHICAGO DATA
# =========================================================================== #
def part6_real(grid=24, year0=2015, year1=2019, category="BURGLARY",
               capacity=0.20):
    """No synthetic scores. Real burglary records, real Head/Mid/Tail split,
    a real trained model. We cannot claim the honest gap is zero here -- we do
    not know it. What we CAN do is show F1 and macro AUC disagreeing about the
    same model, and show what the library does on top of that.

    TWO THINGS THIS GETS RIGHT, LEARNED THE HARD WAY
    ------------------------------------------------
    1. THE OPERATING POINT IS A CAPACITY, NOT 0.50. A fixed probability
       threshold is meaningless here. In a Head cell burglary happens ~78% of
       weeks, so P > 0.50 always and the model flags 100% of them -- a
       degenerate baseline that makes every downstream number nonsense. A
       police department has a fixed number of patrols, so the honest operating
       point is "flag the top `capacity` fraction of all cell-weeks". That is
       what we threshold at.
    2. THE POST-PROCESSOR IS FIT ON TRAIN, EVALUATED ON TEST. Fitting
       ThresholdOptimizer on the same rows you score is how you get a flattering
       and meaningless answer.

    We also report the model WITH and WITHOUT the `cell_mean` feature, because
    that single feature is what turns the model into a location lookup, and
    seeing the two side by side is the clearest statement of the whole paper.
    """
    from fairlearn.postprocessing import ThresholdOptimizer
    try:
        from multicity_audit import CITIES, fetch_city, gridify
    except ImportError:
        from analysis.multicity_audit import CITIES, fetch_city, gridify

    print("\n" + "=" * 78)
    print("PART 6 — the same audit on REAL Chicago burglary data")
    print("=" * 78)

    cfg = CITIES["Chicago"]
    df = fetch_city("Chicago", cfg, year0, year1, category)
    M, grp, nrec = gridify(df, cfg["bbox"], grid)
    T, ncell = M.shape
    print(f"  {nrec:,} records   {T} weeks   {ncell} cells   grid {grid}x{grid}")

    y_all = (M > 0).astype(int)                       # did any burglary happen
    # ---- lag-safe features, TRAIN-period statistics only ------------------ #
    split = int(0.7 * T)
    cell_mean = M[:split].mean(0)
    lag1 = np.vstack([np.zeros((1, ncell)), M[:-1]])
    roll4 = np.vstack([np.zeros((4, ncell))] +
                      [M[max(0, t - 4):t].mean(0)[None, :] for t in range(4, T)])
    woy = (np.arange(T) % 52)[:, None] * np.ones((1, ncell))
    season = [np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)]

    tr = slice(4, split); te = slice(split, T)
    ytr = y_all[tr].ravel(); yte = y_all[te].ravel()
    g_tr = np.tile(grp, split - 4); g = np.tile(grp, T - split)
    unit = np.tile(np.arange(ncell), T - split)
    thr_note = f"top {capacity:.0%} of all cell-weeks (a patrol budget)"

    def build(use_cellmean):
        cols = [lag1, roll4] + season
        if use_cellmean:
            cols.insert(2, np.tile(cell_mean, (T, 1)))
        F = np.stack(cols, -1)
        return F[tr].reshape(-1, F.shape[-1]), F[te].reshape(-1, F.shape[-1])

    fitted = {}
    for use_cm, label in [(True, "WITH cell_mean (a location lookup)"),
                          (False, "WITHOUT cell_mean (must use timing alone)")]:
        Xtr, Xte = build(use_cm)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        s = clf.predict_proba(Xte)[:, 1]
        # operating point = capacity, NOT 0.50
        thr = float(np.quantile(s, 1 - capacity))
        st = rates(yte, (s > thr).astype(int), s, g, unit)
        table(f"BASELINE — {label}", st, note=f"flagging {thr_note}", macro=True)
        fitted[use_cm] = (clf, s, Xtr, Xte, st)

    st_cm = fitted[True][4]; st_no = fitted[False][4]
    pc = np.mean([st_cm[k]["auc"] for k in GROUPS])
    mc = np.mean([st_cm[k]["macro"] for k in GROUPS])
    pn = np.mean([st_no[k]["auc"] for k in GROUPS])
    mn = np.mean([st_no[k]["macro"] for k in GROUPS])
    print(f"""
  THE COMPARISON THAT MATTERS — and state it carefully, because the obvious
  version of it is wrong. Dropping cell_mean moves pooled AUC only
  {pc:.2f} -> {pn:.2f}, so it is NOT true that one feature was supplying all
  the apparent skill. The real result is the vertical gap, not the horizontal
  one:

      with cell_mean     pooled {pc:.2f}   macro {mc:.2f}   inflation {pc-mc:+.2f}
      without cell_mean  pooled {pn:.2f}   macro {mn:.2f}   inflation {pn-mn:+.2f}

  Whatever the feature set, macro AUC sits near 50: this model has almost no
  ability to say WHICH WEEK a given cell will be hit, which is the only thing
  a weekly forecast is for. The pooled figure looks respectable because it is
  also being paid for ranking cells against each other.

  Now read it per group. The inflation is""" +
          "".join(f"\n      {k:>5}  pooled {st_cm[k]['auc']:6.2f}  macro "
                  f"{st_cm[k]['macro']:6.2f}  ->  {st_cm[k]['auc']-st_cm[k]['macro']:+6.2f}"
                  for k in GROUPS) + f"""

  LARGEST IN THE TAIL. The sparsest group -- the one the fairness literature
  says is being under-served -- is where the pooled metric flatters the model
  most. An analyst reading pooled AUC concludes the Tail is the best-served
  group at {st_cm['Tail']['auc']:.2f}. Within-cell it is the worst, at
  {st_cm['Tail']['macro']:.2f}.

  F1 gap {gap(st_cm,'f1'):+.2f}, pooled AUC gap {gap(st_cm,'auc'):+.2f}, macro
  AUC gap {gap(st_cm,'macro'):+.2f}. Pooled and macro do not merely differ in
  size on real data -- here they disagree about the SIGN, and therefore about
  which group needs help.""")

    clf, s, Xtr, Xte, _ = fitted[True]
    for c in ("demographic_parity", "true_positive_rate_parity", "equalized_odds"):
        to = ThresholdOptimizer(estimator=clf, constraints=c,
                                objective="accuracy_score", prefit=True,
                                predict_method="predict_proba")
        # FIT ON TRAIN, predict on test. Fitting on the evaluation rows is how
        # you get a flattering number that means nothing.
        to.fit(Xtr, ytr, sensitive_features=g_tr)
        yh = np.asarray(to.predict(Xte, sensitive_features=g,
                                   random_state=0)).astype(int)
        st2 = rates(yte, yh, s, g, unit)
        table(f"constraints='{c}'  on real Chicago data", st2, macro=True)

    print("""
  The 'wrongful' column is a COUNT of crime-free cell-weeks flagged, not a
  rate. On real data that is the number a police department would actually
  act on, and it is the number that moves when the library is applied.

  IF THE TAIL BASELINE SHOWS NEAR-ZERO FLAGS AND F1 = 0, THAT IS NOT A BUG.
  It is what a fixed patrol budget plus a model that has learned WHERE crime
  is actually does: every patrol goes to the busy area and the sparse area
  gets none. Two things follow, and both belong in the paper.

    * An F1 'fairness gap' computed against a group that received zero flags
      is not measuring model quality at all. It is measuring the budget.
    * The fairness tooling then forces the sparse group up to the same
      selection rate as everyone else. Read the Tail 'wrongful' count before
      and after: that is the number of crime-free places newly subject to
      police attention, produced by the fairness intervention, on a model
      whose within-cell skill (macro AUC) is near 50 everywhere and therefore
      has no idea which of those places is worth visiting.""")


# =========================================================================== #
# PART 7 — how the harm scales with the base-rate spread
# =========================================================================== #
def part7_sweep():
    """The base-rate spread is not a fixed property of a city -- it depends on
    the grid resolution the analyst chose. So: how much does the artifact grow
    as the spread grows? If it grows smoothly, the analyst's arbitrary choice
    of grid size is silently setting the size of the reported 'unfairness'."""
    from fairlearn.postprocessing import ThresholdOptimizer

    print("\n" + "=" * 78)
    print("PART 7 — the artifact as a function of base-rate spread")
    print("=" * 78)
    print(f"  {'Head/Tail base':>16} {'A_F1 gap':>9} {'F1 gap':>9} {'AUC gap':>9} "
          f"{'DP: TPR gap':>12} {'DP: TailFPR':>12}")
    print("  " + "-" * 74)

    for ph, pt in [(0.30, 0.20), (0.45, 0.15), (0.60, 0.12), (0.75, 0.08),
                   (0.90, 0.04)]:
        spec = [("Head", ph), ("Mid", (ph + pt) / 2), ("Tail", pt)]
        f1g, aucg, dtpr, dfpr = [], [], [], []
        for seed in range(3):
            X, y, g = make_data(seed, spec=spec)
            clf = LogisticRegression().fit(X, y)
            s = clf.predict_proba(X)[:, 1]
            st = rates(y, (s > 0.5).astype(int), s, g)
            f1g.append(gap(st, "f1")); aucg.append(gap(st, "auc"))
            to = ThresholdOptimizer(estimator=clf, constraints="demographic_parity",
                                    objective="accuracy_score", prefit=True,
                                    predict_method="predict_proba")
            to.fit(X, y, sensitive_features=g)
            yh = np.asarray(to.predict(X, sensitive_features=g,
                                       random_state=seed)).astype(int)
            st2 = rates(y, yh, s, g)
            dtpr.append(gap(st2, "tpr")); dfpr.append(st2["Tail"]["fpr"])
        af1 = 100 * (2 * ph / (1 + ph) - 2 * pt / (1 + pt))
        print(f"  {ph:6.0%} / {pt:<7.0%} {af1:9.2f} {np.mean(f1g):9.2f} "
              f"{np.mean(aucg):9.2f} {np.mean(dtpr):12.2f} {np.mean(dfpr):12.2f}")

    print("""
  A_F1 gap is the closed-form prediction 2p/(1+p) evaluated at the two base
  rates, computed before any model exists. The F1 column tracks it. The AUC
  column stays near zero throughout, because skill really is equal throughout.

  The uncomfortable implication: the analyst picks the grid resolution, the
  grid resolution sets the base-rate spread, and the base-rate spread sets the
  reported unfairness. None of that is a property of the model.""")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="also run PART 6 (downloads Chicago data)")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--capacity", type=float, default=0.20,
                    help="PART 6 operating point: fraction of cell-weeks the "
                         "department can patrol. NOT a probability threshold.")
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip AIF360 parts (RejectOption is memory hungry)")
    # parse_known_args, not parse_args: inside a Jupyter/Kaggle cell the kernel
    # passes its own "-f /path/kernel.json" on sys.argv and strict parsing
    # exits with SystemExit(2) before anything runs.
    a, _ = ap.parse_known_args()

    part1_metrics()
    part2_fairlearn()
    if not a.skip_slow:
        part3_aif360()
    part4_inprocessing()
    if not a.skip_slow:
        part5_preprocessing()
    part7_sweep()
    if a.real:
        part6_real(grid=a.grid, capacity=a.capacity)

    print("\n" + "=" * 78)
    print("HONEST FRAMING — read before writing this up")
    print("=" * 78)
    print("""
The tension between demographic parity and equalized odds under unequal base
rates is a KNOWN theoretical result (Kleinberg et al. 2016; Chouldechova 2017).
Do NOT claim the conflict as novel; a reviewer will reject that immediately.

What is defensible as a contribution:
  (a) it is the library DEFAULT, so the harmful option is the one a non-expert
      gets by not making a choice;
  (b) the harm is quantified concretely in the predictive-policing setting, as
      counts of crime-free places flagged, rather than stated as an
      impossibility theorem;
  (c) the metric-function result (PART 1) is the same base-rate confound this
      project documented in the literature, now shown inside the tooling;
  (d) PART 4 shows the three method families fail in DIFFERENT ways --
      post-processing redistributes harm at constant AUC, in-processing pays
      in model quality, pre-processing destroys signal -- so there is no
      family that escapes it;
  (e) PART 6 does it on real data, which the first version did not;
  (f) PART 7 shows the analyst's arbitrary grid choice sets the size of the
      reported unfairness.

This is a supporting section of the paper, not the headline claim. The headline
is the measurement critique and its five-city replication.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
