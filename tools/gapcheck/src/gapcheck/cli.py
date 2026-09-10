"""Command line: point it at a CSV.

    gapcheck preds.csv --y label --score p --unit cell --group region

Reads the CSV with the standard library, so pandas is not required.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .audit import audit


def _read_csv(path, cols):
    """Return {name: list} for the requested columns, or raise with a message
    that says which columns actually exist."""
    want = [c for c in cols.values() if c]
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"gapcheck: {path} looks empty")
        missing = [c for c in want if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                f"gapcheck: column(s) {missing} not in {path}\n"
                f"  available: {reader.fieldnames}")
        out = {c: [] for c in want}
        for row in reader:
            for c in want:
                out[c].append(row[c])
    return out


def _numeric(values, colname):
    try:
        return np.array([float(v) for v in values])
    except ValueError as e:
        raise SystemExit(f"gapcheck: column {colname!r} is not numeric ({e})")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="gapcheck",
        description="Is that reported group gap real, or is it arithmetic?")
    ap.add_argument("csv", help="path to a CSV of predictions")
    ap.add_argument("--y", required=True, help="column of 0/1 outcomes")
    ap.add_argument("--score", required=True,
                    help="column of continuous scores (higher = more likely)")
    ap.add_argument("--unit", default=None,
                    help="column identifying the nested unit (cell, hospital, "
                         "user...). Supply this if rows repeat within units.")
    ap.add_argument("--group", default=None,
                    help="column identifying the comparison group")
    ap.add_argument("--min-count", type=int, default=3,
                    help="min positives AND negatives for a unit to count "
                         "toward the macro AUC (default 3)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed score threshold for F1; default sweeps for "
                         "each group's best F1")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a table")
    a = ap.parse_args(argv)

    cols = dict(y=a.y, score=a.score, unit=a.unit, group=a.group)
    data = _read_csv(a.csv, cols)

    y = _numeric(data[a.y], a.y).astype(int)
    s = _numeric(data[a.score], a.score)
    unit = np.array(data[a.unit]) if a.unit else None
    group = np.array(data[a.group]) if a.group else None

    rep = audit(y, s, unit=unit, group=group, min_count=a.min_count,
                threshold=a.threshold)

    if a.json:
        import json
        print(json.dumps({
            "auc_pooled": rep.auc_pooled,
            "auc_macro": rep.auc_macro,
            "inflation": rep.inflation,
            "n_units": rep.n_units,
            "f1_gap": rep.f1_gap,
            "predicted_f1_gap": rep.predicted_f1_gap,
            "explained": rep.explained,
            "auc_gap_pooled": rep.auc_gap_pooled,
            "auc_gap_macro": rep.auc_gap_macro,
            "groups": [
                {"name": g.name, "n": g.n, "base": g.base, "f1": g.f1,
                 "a_f1": g.a_f1, "auc_pooled": g.auc_pooled,
                 "auc_macro": g.auc_macro, "n_units": g.n_units}
                for g in rep.groups],
            "warnings": rep.warnings(),
        }, indent=2, default=float))
    else:
        print(rep)

    # non-zero exit when something was flagged, so this can gate CI
    return 1 if rep.warnings() else 0


if __name__ == "__main__":
    sys.exit(main())
