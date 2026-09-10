# gapcheck

**Is that reported group gap real, or is it arithmetic?**

Two failure modes turn up constantly in applied fairness evaluation. Both
produce large, confident, entirely spurious group disparities. Both are cheap
to check, need no retraining, and are usually not checked.

```
pip install gapcheck
```

```python
from gapcheck import audit

print(audit(y_true, y_score, unit=cell_id, group=region))
```

```
========================================================================
gapcheck
========================================================================
           group         n     base       F1  F1 floor  vs floor      AUC    macro    infl
  ----------------------------------------------------------------------------------
            Head     8,000    60.3%    78.77     75.24     +3.52    76.87    76.91   -0.04
             Mid     8,000    34.8%    61.52     51.69     +9.84    75.48    75.51   -0.03
            Tail     8,000    12.4%    40.56     22.04    +18.52    77.24    77.12   +0.12

  F1 gap (highest minus lowest base rate) :   +38.21
  ... of which a ZERO-SKILL model explains :   +53.20
  AUC gap, pooled                         :    -0.37
  AUC gap, macro (within-unit)            :    -0.21

  [1] F1 GAP LOOKS LIKE A BASE-RATE ARTIFACT. Observed +38.21; a model
      with zero skill would show +53.20 from the base rates alone (72%
      of it). Compare AUC instead: pooled gap -0.37, macro gap -0.21.
========================================================================
```

Those three groups were generated with **identical** score distributions. The
honest gap is exactly zero. A 38-point F1 disparity is available before any
model exists.

---

## The two checks

### 1. Threshold metrics track base rates

F1, precision and accuracy all rise with how often the outcome happens. The
skill-free predictor that answers *yes* to everything scores

```
A_F1(p) = 2p / (1 + p)
```

At a 60% base rate that is **75.0**. At 12% it is **21.4**. So two groups with
identical models, differing only in base rate, show a 53-point "fairness gap"
for free. `gapcheck` reports each group's F1 next to this floor, and tells you
how much of your observed gap the floor already accounts for.

This is not a bound — a good model should beat it. It is a floor that costs
nothing, and reporting F1 without it is close to meaningless.

### 2. Pooling a metric over nested units inflates it

If rows are nested inside units — city blocks, hospitals, schools, users,
sessions — and the outcome is more common in some units than others, then a
**pooled AUC can be won by ranking units rather than ranking cases within a
unit**. A scorer that knows only which unit a row came from, and nothing else
about it, scores well above chance.

`macro_auc` computes AUC inside each unit and averages. Every comparison is
then between two cases in the *same* unit, so a constant per-unit offset
cannot contribute. This is a property, not a heuristic, and the test suite
asserts it:

```python
before, _ = macro_auc(y, s, unit)
after,  _ = macro_auc(y, s + arbitrary_per_unit_offset, unit)
assert before == after          # exactly, to floating point
```

The same offsets move the pooled figure by more than 12 points.

`gapcheck` also flags the case that actually changes decisions: when pooled
and macro **disagree about the sign** of the gap, and therefore nominate
different groups for remediation.

---

## Command line

```
gapcheck preds.csv --y label --score p --unit cell --group region
```

`--json` emits machine-readable output. The exit code is non-zero when
something is flagged, so it can gate CI:

```yaml
- run: gapcheck preds.csv --y label --score p --unit cell --group region
```

---

## API

```python
audit(y_true, y_score, unit=None, group=None, min_count=3, threshold=None)
```

`unit` and `group` are both optional; supply what you have. Without `unit` the
pooling check is skipped, without `group` the base-rate check is. Returns a
`Report` — print it, or read `.auc_pooled`, `.auc_macro`, `.inflation`,
`.f1_gap`, `.predicted_f1_gap`, `.explained`, `.auc_gap_pooled`,
`.auc_gap_macro`, `.groups`, `.warnings()`.

By default each group's **best** F1 over a threshold sweep is reported, so a
shared threshold cannot masquerade as a difference in model quality. Pass
`threshold=` to fix it instead.

Individual metrics are importable: `auc`, `macro_auc`, `f1_at`, `best_f1`,
`a_f1`, `base_rate`. `auc` mid-ranks ties, and returns `nan` rather than 0.5
when a class is absent.

---

## When to reach for this

- Any grouped evaluation where the groups differ in outcome prevalence.
- Any evaluation where rows repeat within an entity — which is most panel,
  spatial, longitudinal and per-user data.
- Before reporting a disparity. Especially before reporting one you like.

## What it does not do

It checks two specific failure modes. **A clean `gapcheck` report does not mean
a model is fair.** It means these two artifacts are absent. Real disparities
exist and this tool will not find them for you.

---

## Provenance

This came out of a crime-prediction fairness project that used it to find and
**retract its own headline result**. A reported +11.11 AUC advantage for
sparse regions, and a +2.81 gain attributed to a physics prior, were both
artifacts of pooling AUC across cells with positive rates ranging from 7.5% to
38%. Under macro AUC the effect was +0.16 ± 2.76, winning 4 of 8 bins — that
is, nothing.

The base-rate half was replicated across five cities (Chicago, Los Angeles,
New York, Seattle, Cincinnati). With a scorer built to have identical skill in
every group — so the honest gap is 0.00 — the mean F1 gap was **+41.42 ± 4.87**
and the mean macro-AUC gap **+0.46 ± 1.04**. `A_F1` computed from base rates
alone predicted the observed F1 gap at **r = +0.94**.

## Testing

```
pip install -e ".[test]"
pytest
```

The interesting tests are the invariance ones. `macro_auc` is asserted to be
exactly unchanged under arbitrary per-unit offsets, and pooled AUC is asserted
to move — because if pooled AUC were robust, this package would have no reason
to exist.

MIT licensed.
