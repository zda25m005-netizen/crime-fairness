"""Tests.

The important ones are the invariance tests. macro_auc is not merely "usually
better" than a pooled AUC -- it is provably immune to the specific artifact,
and that is a property we can assert rather than hope for.
"""
import numpy as np
import pytest

from gapcheck import a_f1, audit, auc, best_f1, f1_at, macro_auc


# --------------------------------------------------------------------------- #
# auc
# --------------------------------------------------------------------------- #
def test_auc_perfect_and_inverted():
    y = [0, 0, 1, 1]
    assert auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_all_ties_is_half():
    """Every score identical -> no ranking information -> exactly 0.5.
    An implementation that breaks ties arbitrarily gets this wrong."""
    assert auc([0, 1, 0, 1], [7.0] * 4) == 0.5


def test_auc_partial_ties_midranked():
    # y=[0,1] with tied scores contributes 0.5; the clean pair contributes 1
    assert auc([0, 1, 0, 1], [1.0, 1.0, 2.0, 3.0]) == pytest.approx(0.625)


def test_auc_undefined_when_one_class_missing():
    assert np.isnan(auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_auc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        auc([0, 1, 1], [0.1, 0.2])


# --------------------------------------------------------------------------- #
# macro_auc — the invariance that justifies the package
# --------------------------------------------------------------------------- #
def _nested(seed=0, n_units=40, n_per=60, d_prime=1.0):
    """Units with WILDLY different base rates but IDENTICAL within-unit skill."""
    rng = np.random.default_rng(seed)
    y, s, u = [], [], []
    for k in range(n_units):
        p = 0.05 + 0.9 * (k / (n_units - 1))         # 5% .. 95%
        yy = (rng.random(n_per) < p).astype(int)
        ss = d_prime * yy + rng.normal(0, 1, n_per)  # same signal everywhere
        y.append(yy); s.append(ss); u.append(np.full(n_per, k))
    return (np.concatenate(y), np.concatenate(s), np.concatenate(u))


def test_macro_auc_is_invariant_to_per_unit_offsets():
    """Adding an arbitrary constant to every score within a unit cannot change
    any within-unit comparison, so macro AUC must be EXACTLY unchanged."""
    y, s, u = _nested()
    rng = np.random.default_rng(1)
    offset = rng.normal(0, 5, u.max() + 1)[u]
    before, _ = macro_auc(y, s, u)
    after, _ = macro_auc(y, s + offset, u)
    assert before == pytest.approx(after, abs=1e-12)


def test_pooled_auc_is_NOT_invariant_to_per_unit_offsets():
    """The same offsets move the pooled number a long way. This is the bug the
    package exists to catch, so we assert it happens."""
    y, s, u = _nested()
    base_rates = np.array([y[u == k].mean() for k in np.unique(u)])
    leak = 4.0 * base_rates[u]          # offset correlated with the base rate
    assert auc(y, s + leak) - auc(y, s) > 0.05


def test_unit_identity_alone_beats_chance_when_pooled():
    """A scorer that knows ONLY which unit a row came from, and nothing about
    the row, has zero forecasting skill. Pooled says otherwise; macro does
    not."""
    y, s, u = _nested()
    base_rates = np.array([y[u == k].mean() for k in np.unique(u)])
    cheat = base_rates[u].astype(float)          # no row-level information
    assert auc(y, cheat) > 0.7                   # pooled is fooled
    mac, _ = macro_auc(y, cheat, u)
    assert np.isnan(mac) or mac == pytest.approx(0.5, abs=1e-9)


def test_macro_auc_skips_units_without_both_classes():
    y = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    s = np.arange(12, dtype=float)
    u = np.array([0] * 4 + [1] * 8)
    _, n = macro_auc(y, s, u, min_count=3)
    assert n == 1                                # unit 0 has no negatives


def test_macro_auc_undefined_when_no_unit_qualifies():
    y = np.array([0, 1, 0, 1])
    s = np.array([1.0, 2.0, 3.0, 4.0])
    u = np.array([0, 0, 1, 1])
    val, n = macro_auc(y, s, u, min_count=3)
    assert np.isnan(val) and n == 0


def test_macro_auc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        macro_auc([0, 1], [0.1, 0.2], [0, 0, 1])


# --------------------------------------------------------------------------- #
# base-rate arithmetic
# --------------------------------------------------------------------------- #
def test_a_f1_known_values():
    assert a_f1(0.0) == 0.0
    assert a_f1(1.0) == 1.0
    assert a_f1(0.5) == pytest.approx(2 / 3)
    # 2(0.564)/1.564 = 0.72123. A model reporting F1 = 72 at this base rate
    # has earned nothing at all.
    assert a_f1(0.564) * 100 == pytest.approx(72.12, abs=0.01)


def test_a_f1_is_achievable_by_predicting_everything():
    """Not a bound -- a floor. Saying YES to every case really does score it."""
    rng = np.random.default_rng(0)
    for p in (0.1, 0.35, 0.8):
        y = (rng.random(20000) < p).astype(int)
        assert f1_at(y, np.ones_like(y)) == pytest.approx(a_f1(y.mean()),
                                                          abs=1e-9)


def test_f1_at_all_zero_predictions_is_zero_not_nan():
    assert f1_at([0, 1, 1], [0, 0, 0]) == 0.0


def test_best_f1_at_least_matches_predicting_everything():
    rng = np.random.default_rng(3)
    y = (rng.random(5000) < 0.3).astype(int)
    s = y + rng.normal(0, 1, y.size)
    assert best_f1(y, s) >= a_f1(y.mean()) - 0.02


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def _equal_skill_groups(seed=0, n=8000):
    """Identical ROC in every group; only the base rate differs."""
    rng = np.random.default_rng(seed)
    y, s, g, u = [], [], [], []
    for gi, (name, p) in enumerate([("high", 0.60), ("mid", 0.35),
                                    ("low", 0.12)]):
        yy = (rng.random(n) < p).astype(int)
        ss = yy + rng.normal(0, 1, n)
        y.append(yy); s.append(ss); g.append(np.full(n, name))
        u.append(rng.integers(gi * 20, gi * 20 + 20, n))
    return (np.concatenate(y), np.concatenate(s),
            np.concatenate(g), np.concatenate(u))


def test_audit_flags_a_base_rate_artifact():
    y, s, g, u = _equal_skill_groups()
    r = audit(y, s, unit=u, group=g)
    assert r.f1_gap > 0.15                       # big apparent gap
    assert abs(r.auc_gap_pooled) < 0.03          # no real gap
    assert r.explained > 0.5                     # mostly arithmetic
    assert any("BASE-RATE ARTIFACT" in w for w in r.warnings())


def test_audit_flags_pooling_inflation():
    y, s, u = _nested()
    base_rates = np.array([y[u == k].mean() for k in np.unique(u)])
    r = audit(y, s + 4.0 * base_rates[u], unit=u)
    assert r.inflation > 0.02
    assert any("POOLING INFLATION" in w for w in r.warnings())


def test_audit_stays_quiet_on_clean_data():
    """Equal base rates, no nesting artifact -> no warnings. A diagnostic that
    always fires is useless."""
    rng = np.random.default_rng(7)
    n = 6000
    y = (rng.random(3 * n) < 0.4).astype(int)
    s = y + rng.normal(0, 1, 3 * n)
    g = np.repeat(["a", "b", "c"], n)
    u = rng.integers(0, 60, 3 * n)
    assert audit(y, s, unit=u, group=g).warnings() == []


def test_audit_flags_scoring_below_the_do_nothing_baseline():
    rng = np.random.default_rng(11)
    n = 4000
    y = (rng.random(n) < 0.85).astype(int)
    s = rng.normal(0, 1, n)                      # pure noise, no skill
    r = audit(y, s, group=np.full(n, "only"))
    g = r.groups[0]
    assert g.f1_over_floor < 0
    assert any("BELOW THE DO-NOTHING" in w for w in r.warnings())


def test_audit_reports_sign_disagreement():
    """The failure mode that actually changes decisions.

    Group A genuinely ranks better WITHIN its units. Group B ranks poorly
    within units, but its units differ a lot in base rate and its scores carry
    a matching per-unit offset -- so pooling lets B win on between-unit
    ranking alone. Pooled and macro then nominate DIFFERENT groups as the
    worse-served one, and remediation goes to whichever the analyst computed.
    """
    rng = np.random.default_rng(5)
    y, s, g, u = [], [], [], []
    for k in range(25):                          # A: real within-unit skill
        yy = (rng.random(200) < 0.55).astype(int)
        ss = 0.7 * yy + rng.normal(0, 1, 200)
        y.append(yy); s.append(ss)
        g.append(np.full(200, "A")); u.append(np.full(200, f"A{k}"))
    for k in range(25):                          # B: almost none, big offset
        p = 0.02 + 0.56 * (k / 24)
        yy = (rng.random(200) < p).astype(int)
        ss = 0.1 * yy + rng.normal(0, 1, 200) + 20.0 * p
        y.append(yy); s.append(ss)
        g.append(np.full(200, "B")); u.append(np.full(200, f"B{k}"))
    y = np.concatenate(y); s = np.concatenate(s)
    g = np.concatenate(g); u = np.concatenate(u)

    r = audit(y, s, unit=u, group=g)
    by = {x.name: x for x in r.groups}
    assert by["B"].auc_macro < by["A"].auc_macro        # truth: A is better
    assert by["B"].auc_pooled > by["A"].auc_pooled      # pooled says otherwise
    assert r.auc_gap_pooled * r.auc_gap_macro < 0
    assert any("DISAGREE ON THE SIGN" in w for w in r.warnings())


def test_audit_works_without_unit_or_group():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.3).astype(int)
    s = y + rng.normal(0, 1, 500)
    r = audit(y, s)
    assert not np.isnan(r.auc_pooled)
    assert np.isnan(r.auc_macro)
    assert "gapcheck" in str(r)


def test_audit_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="0/1"):
        audit([0, 1, 2], [0.1, 0.2, 0.3])


def test_audit_rejects_empty_and_mismatched():
    with pytest.raises(ValueError):
        audit([], [])
    with pytest.raises(ValueError):
        audit([0, 1], [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def test_cli_round_trip(tmp_path, capsys):
    from gapcheck.cli import main
    y, s, g, u = _equal_skill_groups(n=1500)
    p = tmp_path / "preds.csv"
    with open(p, "w") as fh:
        fh.write("label,score,cell,region\n")
        for a, b, c, d in zip(y, s, u, g):
            fh.write(f"{a},{b},{c},{d}\n")
    code = main([str(p), "--y", "label", "--score", "score",
                 "--unit", "cell", "--group", "region"])
    out = capsys.readouterr().out
    assert "gapcheck" in out
    assert code == 1              # artifact present -> non-zero exit


def test_cli_json(tmp_path, capsys):
    import json
    from gapcheck.cli import main
    y, s, g, u = _equal_skill_groups(n=800)
    p = tmp_path / "preds.csv"
    with open(p, "w") as fh:
        fh.write("label,score,cell,region\n")
        for a, b, c, d in zip(y, s, u, g):
            fh.write(f"{a},{b},{c},{d}\n")
    main([str(p), "--y", "label", "--score", "score", "--unit", "cell",
          "--group", "region", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "auc_macro" in payload and len(payload["groups"]) == 3


def test_cli_missing_column_says_what_exists(tmp_path):
    from gapcheck.cli import main
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit) as e:
        main([str(p), "--y", "nope", "--score", "b"])
    assert "available" in str(e.value)
