"""
Fairness and performance metrics, classified by base-rate dependence.
================================================================================
Every metric below is annotated with whether its ATTAINABLE value depends on the
group's base rate p.  Metrics whose attainability depends on p produce non-zero
group gaps even under equal skill, and must not be used alone to claim
unfairness.

    CONFOUNDED  : F1, precision, accuracy, average precision,
                  demographic parity, predictive parity
    INVARIANT   : AUC-ROC, balanced accuracy, equal opportunity (TPR gap),
                  equalized odds (TPR+FPR gaps)
    CONDITIONAL : recall/TPR and FPR alone are attainability-invariant but
                  OPERATING-POINT dependent; report them as a pair, or fix the
                  operating point across groups.
"""
from __future__ import annotations

import numpy as np

DEPENDENCE = {
    "f1": "confounded",
    "precision": "confounded",
    "accuracy": "confounded",
    "average_precision": "confounded",
    "demographic_parity": "confounded",
    "predictive_parity": "confounded",
    "auc": "invariant",
    "balanced_accuracy": "invariant",
    "equal_opportunity": "invariant",
    "equalized_odds": "invariant",
    "tpr": "conditional",
    "fpr": "conditional",
}


def _conf(y, pred):
    y = np.asarray(y).ravel(); pred = np.asarray(pred).ravel()
    tp = float(np.sum((pred == 1) & (y == 1))); fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1))); tn = float(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def _safe(a, b):
    return 0.0 if b == 0 else a / b


# ------------------------------- performance ------------------------------- #
def f1(y, pred):
    tp, fp, fn, _ = _conf(y, pred); return _safe(2 * tp, 2 * tp + fp + fn)


def precision(y, pred):
    tp, fp, _, _ = _conf(y, pred); return _safe(tp, tp + fp)


def tpr(y, pred):
    tp, _, fn, _ = _conf(y, pred); return _safe(tp, tp + fn)


def fpr(y, pred):
    _, fp, _, tn = _conf(y, pred); return _safe(fp, fp + tn)


def accuracy(y, pred):
    tp, fp, fn, tn = _conf(y, pred); return _safe(tp + tn, tp + fp + fn + tn)


def balanced_accuracy(y, pred):
    return 0.5 * (tpr(y, pred) + (1.0 - fpr(y, pred)))


def auc(y, score):
    """Mann-Whitney AUC with tie correction; 0.5 = no skill, base-rate invariant."""
    y = np.asarray(y).ravel(); s = np.asarray(score, dtype=float).ravel()
    s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    s_sorted = s[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for st, ct in zip(first[counts > 1], counts[counts > 1]):
        idx = order[st:st + ct]; ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------- group-fairness criteria ----------------------- #
def demographic_parity(pred_a, pred_b):
    """P(pred=1 | A) - P(pred=1 | B).  CONFOUNDED: depends on base rates."""
    return float(np.mean(pred_a) - np.mean(pred_b))


def equal_opportunity(y_a, p_a, y_b, p_b):
    """TPR_A - TPR_B.  INVARIANT: each TPR conditions on the positive class."""
    return tpr(y_a, p_a) - tpr(y_b, p_b)


def equalized_odds(y_a, p_a, y_b, p_b):
    """(TPR gap, FPR gap).  INVARIANT: both terms condition on a single class."""
    return (tpr(y_a, p_a) - tpr(y_b, p_b), fpr(y_a, p_a) - fpr(y_b, p_b))


def predictive_parity(y_a, p_a, y_b, p_b):
    """Precision gap.  CONFOUNDED: precision is bounded by the base rate.

    This is the criterion at the heart of Chouldechova (2017): it cannot hold
    simultaneously with equalized odds when base rates differ.
    """
    return precision(y_a, p_a) - precision(y_b, p_b)


# ------------------------------- attainability ----------------------------- #
def attainable_f1(p):
    """A_F1(p) = 2p/(1+p): F1 of the skill-free always-positive predictor."""
    p = np.asarray(p, dtype=float)
    return 2 * p / (1 + p)


def attainable(metric: str, p):
    """Skill-free attainable value of `metric` at base rate p."""
    p = np.asarray(p, dtype=float)
    if metric in ("f1",):
        return attainable_f1(p)
    if metric in ("precision", "average_precision"):
        return p                      # always-positive precision = p
    if metric == "accuracy":
        return np.maximum(p, 1 - p)   # majority-class rule
    if metric in ("auc", "balanced_accuracy"):
        return np.full_like(p, 0.5)   # invariant
    if metric in ("tpr",):
        return np.ones_like(p)
    raise ValueError(f"no attainability defined for {metric!r}")


def skill(metric: str, value, p):
    """RATIO form: value / A(p).  1.0 = no better than skill-free.

    CAUTION: unstable when A(p) is small (sparse groups), because a small
    absolute gain becomes a large ratio.  Prefer `skill_score` for group
    comparisons; this form is kept for interpretability ("% of attainable").
    """
    a = attainable(metric, p)
    return float(value) / float(np.asarray(a).mean() if np.ndim(a) else a)


def skill_score(metric: str, value, p, perfect: float = 1.0):
    """SKILL SCORE form:  (M - A) / (perfect - A).

    The standard forecasting-skill normalisation (cf. Brier Skill Score).
    0 = matches the skill-free baseline, 1 = perfect, <0 = worse than baseline.
    Unlike the ratio it stays bounded and comparable across groups with very
    different base rates, so it is the recommended statistic for group gaps.
    """
    a = attainable(metric, p)
    a = float(np.asarray(a).mean() if np.ndim(a) else a)
    denom = perfect - a
    return float("nan") if abs(denom) < 1e-12 else (float(value) - a) / denom
