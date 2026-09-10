"""The audit itself: is this reported group gap real, or is it arithmetic?"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import a_f1, auc, base_rate, best_f1, macro_auc

__all__ = ["GroupResult", "Report", "audit"]


@dataclass
class GroupResult:
    name: str
    n: int
    base: float
    f1: float
    a_f1: float
    auc_pooled: float
    auc_macro: float
    n_units: int

    @property
    def f1_over_floor(self):
        """How much F1 the model earned ABOVE the do-nothing predictor."""
        return self.f1 - self.a_f1

    @property
    def inflation(self):
        return self.auc_pooled - self.auc_macro


@dataclass
class Report:
    groups: list = field(default_factory=list)
    auc_pooled: float = float("nan")
    auc_macro: float = float("nan")
    n_units: int = 0
    has_units: bool = False
    has_groups: bool = False

    # ---- headline comparisons ------------------------------------------- #
    @property
    def inflation(self):
        """Pooled minus macro. How much the pooled number is flattered by
        being allowed to rank units against each other."""
        return self.auc_pooled - self.auc_macro

    def _hi_lo(self):
        """Highest- and lowest-base-rate groups. The gap is defined between
        these two, not between whichever pair looks worst."""
        g = [x for x in self.groups if not np.isnan(x.base)]
        if len(g) < 2:
            return None, None
        g = sorted(g, key=lambda x: x.base)
        return g[-1], g[0]

    @property
    def f1_gap(self):
        hi, lo = self._hi_lo()
        return float("nan") if hi is None else hi.f1 - lo.f1

    @property
    def predicted_f1_gap(self):
        """What the gap would be for a model with NO SKILL AT ALL, from the
        base rates alone. If the observed gap is close to this, the gap is
        telling you about the base rates."""
        hi, lo = self._hi_lo()
        return float("nan") if hi is None else hi.a_f1 - lo.a_f1

    @property
    def auc_gap_pooled(self):
        hi, lo = self._hi_lo()
        return float("nan") if hi is None else hi.auc_pooled - lo.auc_pooled

    @property
    def auc_gap_macro(self):
        hi, lo = self._hi_lo()
        return float("nan") if hi is None else hi.auc_macro - lo.auc_macro

    @property
    def explained(self):
        """Fraction of the observed F1 gap that the skill-free baseline
        already accounts for. Near or above 1.0 means the gap is arithmetic."""
        p = self.predicted_f1_gap
        if np.isnan(p) or abs(p) < 1e-9:
            return float("nan")
        return self.f1_gap / p

    # ---- the verdicts ---------------------------------------------------- #
    def warnings(self):
        # NOTE: every quantity on this object is a FRACTION in [0, 1]. Only the
        # printed strings multiply by 100. An earlier version compared
        # `inflation > 2.0` intending "2 points", so the warning could never
        # fire; the test suite caught it. Keep thresholds in fractions.
        w = []
        if self.has_units and self.inflation > 0.02:
            w.append(
                f"POOLING INFLATION {self.inflation*100:+.2f} points. Your "
                f"pooled AUC ({self.auc_pooled*100:.2f}) is higher than the "
                f"within-unit AUC ({self.auc_macro*100:.2f}). Some of what it "
                f"is rewarding is knowing WHICH UNIT a case belongs to, not "
                f"ranking cases inside a unit. Report the macro figure.")
        if self.has_groups:
            e = self.explained
            if not np.isnan(e) and e > 0.5 and abs(self.f1_gap) > 0.05:
                w.append(
                    f"F1 GAP LOOKS LIKE A BASE-RATE ARTIFACT. Observed "
                    f"{self.f1_gap*100:+.2f}; a model with zero skill would "
                    f"show {self.predicted_f1_gap*100:+.2f} from the base "
                    f"rates alone ({e*100:.0f}% of it). Compare AUC instead: "
                    f"pooled gap {self.auc_gap_pooled*100:+.2f}, macro gap "
                    f"{self.auc_gap_macro*100:+.2f}.")
            if self.has_units and not np.isnan(self.auc_gap_pooled):
                a, b = self.auc_gap_pooled, self.auc_gap_macro
                if not np.isnan(b) and a * b < 0 and max(abs(a), abs(b)) > 0.01:
                    w.append(
                        f"POOLED AND MACRO DISAGREE ON THE SIGN of the gap "
                        f"({a*100:+.2f} vs {b*100:+.2f}). They nominate "
                        f"DIFFERENT groups as the worse-served one. Any "
                        f"remediation decision here depends entirely on which "
                        f"metric you happened to pick.")
            for g in self.groups:
                if g.f1_over_floor < 0:
                    w.append(
                        f"GROUP {g.name!r} SCORES BELOW THE DO-NOTHING "
                        f"BASELINE: F1 {g.f1*100:.2f} against a skill-free "
                        f"{g.a_f1*100:.2f} at base rate {g.base*100:.1f}%. "
                        f"Predicting the positive class everywhere would score "
                        f"higher.")
        if self.has_units and self.n_units == 0:
            w.append(
                "NO UNIT had enough positives and negatives to compute a "
                "within-unit AUC. The macro figure is undefined. Either your "
                "units are too small or the outcome is too rare -- in both "
                "cases the pooled AUC is not measuring what you think.")
        return w

    def __str__(self):
        L = []
        L.append("=" * 72)
        L.append("gapcheck")
        L.append("=" * 72)
        if self.has_groups:
            head = (f"  {'group':>14} {'n':>9} {'base':>8} {'F1':>8} "
                    f"{'F1 floor':>9} {'vs floor':>9} {'AUC':>8}"
                    + (f" {'macro':>8} {'infl':>7}" if self.has_units else ""))
            L.append(head)
            L.append("  " + "-" * (len(head) - 2))
            for g in sorted(self.groups, key=lambda x: -x.base):
                row = (f"  {g.name:>14} {g.n:>9,} {g.base*100:7.1f}% "
                       f"{g.f1*100:8.2f} {g.a_f1*100:9.2f} "
                       f"{g.f1_over_floor*100:+9.2f} {g.auc_pooled*100:8.2f}")
                if self.has_units:
                    row += f" {g.auc_macro*100:8.2f} {g.inflation*100:+7.2f}"
                L.append(row)
            L.append("")
            L.append(f"  F1 gap (highest minus lowest base rate) : "
                     f"{self.f1_gap*100:+8.2f}")
            L.append(f"  ... of which a ZERO-SKILL model explains : "
                     f"{self.predicted_f1_gap*100:+8.2f}")
            L.append(f"  AUC gap, pooled                         : "
                     f"{self.auc_gap_pooled*100:+8.2f}")
            if self.has_units:
                L.append(f"  AUC gap, macro (within-unit)            : "
                         f"{self.auc_gap_macro*100:+8.2f}")
        L.append("")
        L.append(f"  overall pooled AUC : {self.auc_pooled*100:8.2f}")
        if self.has_units:
            L.append(f"  overall macro  AUC : {self.auc_macro*100:8.2f}   "
                     f"({self.n_units} units)")
            L.append(f"  pooling inflation  : {self.inflation*100:+8.2f}")
        w = self.warnings()
        L.append("")
        if w:
            for i, msg in enumerate(w, 1):
                L.append(f"  [{i}] " + _wrap(msg, 66, "      "))
                L.append("")
        else:
            L.append("  No base-rate or pooling artifact detected. This does "
                     "not mean the")
            L.append("  model is fair; it means these two specific failure "
                     "modes are absent.")
        L.append("=" * 72)
        return "\n".join(L)


def _wrap(text, width, indent):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return ("\n" + indent).join(out)


def audit(y_true, y_score, unit=None, group=None, min_count=3,
          threshold=None):
    """Check whether a reported group gap survives contact with the base rates.

    Parameters
    ----------
    y_true : array of 0/1
    y_score : array of continuous scores (higher = more likely positive)
    unit : array, optional
        The repeated entity each row belongs to -- cell, hospital, school,
        user, session. Supply this whenever rows are nested, which is almost
        always. Without it the macro AUC cannot be computed and the pooling
        check is skipped.
    group : array, optional
        The protected or comparison group. Without it the per-group table and
        the base-rate check are skipped.
    min_count : int
        Minimum positives AND negatives for a unit to contribute to the macro
        AUC.
    threshold : float, optional
        Score threshold for F1. Default is to report each group's BEST F1 over
        a threshold sweep, which is the fairest comparison: it stops a shared
        threshold from being mistaken for a difference in model quality.

    Returns
    -------
    Report -- print it, or read the attributes.
    """
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true {y.shape} and y_score {s.shape} must match")
    if y.size == 0:
        raise ValueError("empty input")
    bad = set(np.unique(y)) - {0, 1}
    if bad:
        raise ValueError(f"y_true must be 0/1, found {sorted(bad)}")

    rep = Report()
    rep.has_units = unit is not None
    rep.has_groups = group is not None
    rep.auc_pooled = auc(y, s)
    if unit is not None:
        u = np.asarray(unit).ravel()
        rep.auc_macro, rep.n_units = macro_auc(y, s, u, min_count=min_count)

    def _f1(yy, ss):
        if threshold is None:
            return best_f1(yy, ss)
        from .metrics import f1_at
        return f1_at(yy, (ss > threshold).astype(int))

    if group is not None:
        g = np.asarray(group).ravel()
        if g.shape != y.shape:
            raise ValueError(f"group {g.shape} and y_true {y.shape} must match")
        for name in np.unique(g):
            m = g == name
            yy, ss = y[m], s[m]
            p = base_rate(yy)
            if unit is not None:
                mac, nu = macro_auc(yy, ss, np.asarray(unit).ravel()[m],
                                    min_count=min_count)
            else:
                mac, nu = float("nan"), 0
            rep.groups.append(GroupResult(
                name=str(name), n=int(m.sum()), base=p,
                f1=_f1(yy, ss), a_f1=float(a_f1(p)),
                auc_pooled=auc(yy, ss), auc_macro=mac, n_units=nu))
    return rep
