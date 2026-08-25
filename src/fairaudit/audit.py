"""The main audit entry point: one call, a full confounding report."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import metrics as M


@dataclass
class AuditReport:
    groups: List[str]
    base_rate: Dict[str, float] = field(default_factory=dict)
    confounded: Dict[str, Dict[str, float]] = field(default_factory=dict)
    invariant: Dict[str, Dict[str, float]] = field(default_factory=dict)
    attainable: Dict[str, float] = field(default_factory=dict)
    skill: Dict[str, float] = field(default_factory=dict)
    skill_score: Dict[str, float] = field(default_factory=dict)
    gaps: Dict[str, float] = field(default_factory=dict)
    verdict: str = ""

    def __str__(self) -> str:
        hi, lo = self.groups[0], self.groups[-1]
        L = ["=" * 74,
             f"FAIRNESS AUDIT — base-rate confounding check ({hi} vs {lo})",
             "=" * 74,
             f"{'group':10s} {'base rate':>10s} {'A_F1(p)':>9s} {'F1':>8s} "
             f"{'%attain':>8s} {'skillsc':>8s} {'AUC':>8s}",
             "-" * 70]
        for g in self.groups:
            L.append(f"{g:10s} {self.base_rate[g]:10.3f} "
                     f"{100*self.attainable[g]:9.1f} "
                     f"{100*self.confounded['f1'][g]:8.2f} "
                     f"{100*self.skill[g]:8.1f} "
                     f"{100*self.skill_score[g]:8.2f} "
                     f"{100*self.invariant['auc'][g]:8.2f}")
        L += ["", "GAPS (first group minus last)",
              f"{'metric':22s} {'gap':>9s}   class"]
        L.append("-" * 46)
        for k, v in self.gaps.items():
            L.append(f"{k:22s} {100*v:9.2f}   {M.DEPENDENCE.get(k, '-')}")
        L += ["", self.verdict, "=" * 74]
        return "\n".join(L)


def _as2d(a):
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a.reshape(len(a), -1)


def audit(y_true, y_score, groups, threshold=0.5, per_label_threshold=None
          ) -> AuditReport:
    """Audit group gaps for base-rate confounding.

    Parameters
    ----------
    y_true  : (N,) or (N, C) binary labels
    y_score : same shape, scores in [0, 1]
    groups  : (N,) group label per row
    threshold : scalar decision threshold (ignored where per_label_threshold set)
    per_label_threshold : optional (C,) per-label thresholds

    Returns
    -------
    AuditReport with base rates, attainability, confounded and invariant gaps.
    """
    Y, S = _as2d(y_true), _as2d(y_score)
    S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)
    g = np.asarray(groups).ravel()
    if len(Y) != len(g):
        raise ValueError("y_true and groups must have the same length")
    C = Y.shape[1]
    thr = (np.full(C, threshold) if per_label_threshold is None
           else np.asarray(per_label_threshold, dtype=float))
    P = (S > thr.reshape(1, -1)).astype(int)

    # order groups by base rate, highest first (Head -> Tail convention)
    uniq = list(dict.fromkeys(g.tolist()))
    uniq.sort(key=lambda u: -float(Y[g == u].mean()))

    rep = AuditReport(groups=uniq)
    rep.confounded = {k: {} for k in ("f1", "precision", "accuracy")}
    rep.invariant = {k: {} for k in ("auc", "balanced_accuracy")}

    for u in uniq:
        m = g == u
        yg, pg, sg = Y[m], P[m], S[m]
        rates = yg.mean(axis=0)
        rep.base_rate[u] = float(rates.mean())
        rep.attainable[u] = float(np.mean(M.attainable_f1(rates)))  # macro (E1)

        def per_label(fn, use_score=False):
            vals = []
            for c in range(C):
                if yg[:, c].sum() in (0, len(yg)):
                    continue
                vals.append(fn(yg[:, c], sg[:, c] if use_score else pg[:, c]))
            return float(np.mean(vals)) if vals else float("nan")

        rep.confounded["f1"][u] = per_label(M.f1)
        rep.confounded["precision"][u] = per_label(M.precision)
        rep.confounded["accuracy"][u] = per_label(M.accuracy)
        rep.invariant["auc"][u] = per_label(M.auc, use_score=True)
        rep.invariant["balanced_accuracy"][u] = per_label(M.balanced_accuracy)
        rep.skill[u] = rep.confounded["f1"][u] / max(rep.attainable[u], 1e-9)
        rep.skill_score[u] = M.skill_score("f1", rep.confounded["f1"][u], rates)

    hi, lo = uniq[0], uniq[-1]
    for k in ("f1", "precision", "accuracy"):
        rep.gaps[k] = rep.confounded[k][hi] - rep.confounded[k][lo]
    for k in ("auc", "balanced_accuracy"):
        rep.gaps[k] = rep.invariant[k][hi] - rep.invariant[k][lo]
    rep.gaps["skill_ratio_f1"] = rep.skill[hi] - rep.skill[lo]
    rep.gaps["skill_score_f1"] = rep.skill_score[hi] - rep.skill_score[lo]

    # verdict
    gf1, gauc = abs(rep.gaps["f1"]), abs(rep.gaps["auc"])
    floor = float(rep.attainable[hi] - rep.attainable[lo])
    ratio = gf1 / max(gauc, 1e-9)
    if gf1 > 0.05 and ratio > 3:
        rep.verdict = (
            f"VERDICT: likely CONFOUNDED. The F1 gap ({100*gf1:.1f} pts) is "
            f"{ratio:.0f}x the AUC gap ({100*gauc:.1f} pts),\n"
            f"and the attainability floor alone predicts {100*floor:.1f} pts. "
            "Report an invariant\nmetric or the skill-normalised gap before "
            "claiming a disparity in model quality.")
    else:
        rep.verdict = (
            "VERDICT: no strong evidence of base-rate confounding; the "
            "confounded and\ninvariant metrics broadly agree.")
    return rep
