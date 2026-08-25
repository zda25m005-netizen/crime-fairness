"""
fairaudit — detect base-rate confounding in group-fairness evaluation.
================================================================================
Group performance gaps computed with base-rate-DEPENDENT metrics (F1, precision,
accuracy, Precision@k, MAE) are non-zero even when two groups receive EQUAL
skill, because the attainable value of those metrics depends on the group's base
rate.  This package reports the confounded gap, the attainability floor, the
skill-normalised gap, and base-rate-INVARIANT gaps side by side.

Quick start
-----------
    import numpy as np
    from fairaudit import audit

    report = audit(y_true, y_score, groups, threshold=0.5)
    print(report)

`y_true`  (N,) or (N, C) binary labels
`y_score` same shape, model scores in [0, 1]
`groups`  (N,) group label per row (e.g. "Head"/"Tail")

Theory and proofs: see fairaudit.extensions and the accompanying paper.
Builds on Davis & Goadrich (2006), Boyd et al. (2012), Chouldechova (2017).
"""
from .metrics import (DEPENDENCE, auc, balanced_accuracy, accuracy, f1,
                      precision, tpr, fpr, attainable, attainable_f1, skill,
                      demographic_parity, equal_opportunity, equalized_odds,
                      predictive_parity)
from .audit import audit, AuditReport

__version__ = "0.1.0"
__all__ = ["audit", "AuditReport", "DEPENDENCE", "auc", "balanced_accuracy",
           "accuracy", "f1", "precision", "tpr", "fpr", "attainable",
           "attainable_f1", "skill", "demographic_parity", "equal_opportunity",
           "equalized_odds", "predictive_parity"]
