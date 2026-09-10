"""gapcheck — is that reported group gap real, or is it arithmetic?

Two failure modes turn up constantly in applied fairness evaluation, and both
produce large, confident, entirely spurious group disparities.

1. THRESHOLD METRICS TRACK BASE RATES. F1, precision and accuracy all rise
   with how often the outcome happens. A model with identical skill in every
   group still shows a large F1 "gap" if the groups differ in base rate. The
   size of the artifact is predictable in advance: a do-nothing predictor
   scores 2p/(1+p).

2. POOLING A METRIC OVER NESTED UNITS INFLATES IT. If your rows are nested
   inside units -- city blocks, hospitals, schools, users, sessions -- and the
   outcome is more common in some units than others, then a pooled AUC can be
   won by ranking UNITS rather than ranking cases within a unit. A scorer that
   knows only which unit a row came from, and nothing else, scores above
   chance.

    >>> from gapcheck import audit
    >>> print(audit(y, scores, unit=cell_id, group=region))

Both checks are cheap, need no retraining, and are worth running before any
disparity is reported.
"""
from .audit import GroupResult, Report, audit
from .metrics import a_f1, auc, base_rate, best_f1, f1_at, macro_auc

__version__ = "0.1.0"
__all__ = [
    "audit", "Report", "GroupResult",
    "auc", "macro_auc", "f1_at", "best_f1", "a_f1", "base_rate",
]
