"""Metrics. Pure numpy, no dependencies beyond it.

Everything here is deliberately small and readable, because the whole point of
the package is that people should be able to check that the diagnostic itself
is not doing something clever behind their back.
"""
from __future__ import annotations

import numpy as np

__all__ = ["auc", "macro_auc", "f1_at", "best_f1", "a_f1", "base_rate"]


def auc(y, score):
    """Area under the ROC curve, with correct handling of tied scores.

    Ties matter more than people expect. A model that outputs a handful of
    distinct values (a shallow tree, a quantised score, a randomised
    classifier) will have many ties, and an implementation that breaks them
    arbitrarily can move the AUC by several points. We assign mid-ranks.

    Returns nan when one class is absent, rather than raising or silently
    returning 0.5 -- an undefined AUC should be visibly undefined.
    """
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y has {y.shape} but score has {s.shape}")
    npos = int(y.sum())
    nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    # mid-rank the ties
    _, first, count = np.unique(s[order], return_index=True, return_counts=True)
    for start, n in zip(first[count > 1], count[count > 1]):
        idx = order[start:start + n]
        ranks[idx] = ranks[idx].mean()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_auc(y, score, unit, min_count=3):
    """Mean of per-unit AUCs. THE point of this package.

    A pooled AUC computed across units with different base rates can be won by
    ranking UNITS against each other rather than ranking cases within a unit.
    If your units are hospitals, schools, city blocks, users or sessions, and
    the outcome is more common in some than others, then a model that has
    learned nothing except "which unit is this" scores above chance -- and the
    pooled number will not tell you.

    Computing AUC inside each unit and averaging makes that structurally
    impossible: every comparison is between two cases in the SAME unit, so a
    constant per-unit offset cannot contribute. This is invariant to any
    monotone per-unit transformation of the score.

    Units with fewer than `min_count` positives or negatives are skipped, since
    their AUC is undefined or dominated by noise. The count of units actually
    used is returned so you can see how much was dropped.

    Returns
    -------
    (value, n_units) : the macro AUC and how many units contributed.
    """
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(score, dtype=float).ravel()
    u = np.asarray(unit).ravel()
    if not (y.shape == s.shape == u.shape):
        raise ValueError(
            f"y {y.shape}, score {s.shape} and unit {u.shape} must match")
    vals = []
    for key in np.unique(u):
        m = u == key
        yy = y[m]
        npos = int(yy.sum())
        if npos < min_count or (yy.size - npos) < min_count:
            continue
        v = auc(yy, s[m])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def f1_at(y, pred):
    y = np.asarray(y).ravel().astype(int)
    p = np.asarray(pred).ravel().astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def best_f1(y, score, n_thresholds=200):
    """Best F1 over a threshold sweep.

    Reported because a group's F1 depends on where you put the threshold, and
    comparing groups at one shared threshold conflates 'worse model' with
    'different base rate'. The best achievable F1 removes the threshold as a
    free variable -- and, as base_rate_floor shows, it still does not remove
    the base rate.
    """
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(score, dtype=float).ravel()
    if y.size == 0:
        return float("nan")
    grid = np.quantile(s, np.linspace(0.01, 0.99, n_thresholds))
    return max(f1_at(y, (s > t).astype(int)) for t in np.unique(grid))


def a_f1(p):
    """F1 of the skill-free predictor that answers YES to everything: 2p/(1+p).

    This is NOT a ceiling and NOT a bound -- a real model can and should beat
    it. It is a FLOOR that costs nothing to reach, and it rises steeply with
    the base rate. Two groups with base rates of 60% and 12% have skill-free
    F1 scores of 75.0 and 21.4. A 53-point 'fairness gap' between them is
    available before any model exists.

    Always report a group's F1 next to its a_f1. An F1 of 60 at p = 0.6 is
    below the do-nothing baseline.
    """
    p = np.asarray(p, dtype=float)
    return 2 * p / (1 + p)


def base_rate(y):
    y = np.asarray(y).ravel().astype(int)
    return float(y.mean()) if y.size else float("nan")
