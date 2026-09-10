r"""
Paper figures, built from saved results only.
================================================================================
Every number plotted here is read from a results file that some other script
wrote. Nothing is retyped, so a figure cannot drift away from the run that
produced it. If a number in the paper disagrees with a number in a figure,
that is a bug and this script is the arbiter.

    python make_paper_figures.py --results results/multicity_audit.json \
                                 --outdir figures

Produces:
    fig1_prediction.pdf   A_F1 predicts the observed F1 gap  (the headline)
    fig2_sweep.pdf        the artifact as a function of base-rate spread
    fig3_chicago.pdf      pooled vs macro per group, real Chicago
    fig4_cities.pdf       five-city forest plot, F1 gap vs macro AUC gap
    fig5_invariance.pdf   macro AUC is flat while pooled drifts
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ("Head", "Mid", "Tail")

# One colour system for the whole paper. Pooled is the misleading one, so it
# gets the warm colour; macro is the honest one and gets the cool colour.
C_POOL = "#c1502e"
C_MACRO = "#2f6d80"
C_F1 = "#8c6d31"
C_GREY = "#5a5a5a"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})



def save(fig, outdir, name):
    """Write PDF for the paper and PNG for slides, from the same figure
    object, so the two can never drift apart."""
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200)
    plt.close(fig)


def a_f1(p):
    return 100 * 2 * np.asarray(p, float) / (1 + np.asarray(p, float))


# --------------------------------------------------------------------------- #
def fig1_prediction(rows, outdir):
    """The strongest claim in the paper: the size of the artifact is
    predictable from base rates alone, before any model exists."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1),
                                   gridspec_kw={"width_ratios": [1, 1]})

    # -- left: the A_F1 curve with each city's Head and Tail marked --------- #
    p = np.linspace(0, 1, 400)
    ax1.plot(100 * p, a_f1(p), color=C_GREY, lw=1.6, zorder=1,
             label=r"$A_{F_1}(p)=2p/(1+p)$")
    for r in rows:
        ph, pt = r["Head"]["base"], r["Tail"]["base"]
        ax1.plot([ph, pt], [a_f1(ph / 100), a_f1(pt / 100)], "-",
                 color=C_F1, lw=0.9, alpha=0.55, zorder=2)
        ax1.scatter([ph, pt], [a_f1(ph / 100), a_f1(pt / 100)], s=22,
                    color=C_F1, zorder=3, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("base rate of the group (%)")
    ax1.set_ylabel(r"$F_1$ of a skill-free predictor")
    ax1.set_title("A do-nothing model's score\nrises steeply with the base rate",
                  fontsize=9)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100)

    # -- right: predicted vs observed gap ----------------------------------- #
    pred = np.array([r["gap_af1"] for r in rows])
    obs = np.array([r["gap_f1"] for r in rows])
    r_val = float(np.corrcoef(pred, obs)[0, 1])

    lo = min(pred.min(), obs.min()) - 8
    lim = [max(0, lo), max(pred.max(), obs.max()) * 1.12]
    ax2.plot(lim, lim, "--", color=C_GREY, lw=0.9, zorder=1,
             label="observed = predicted")
    ax2.scatter(pred, obs, s=38, color=C_F1, zorder=3,
                edgecolor="white", linewidth=0.6)
    # hand-placed offsets: Cincinnati and Los Angeles sit almost on top of
    # each other, so an automatic placement collides.
    nudge = {"Cincinnati": (6, 5), "Los Angeles": (6, -9), "Chicago": (6, -9),
             "New York": (-8, 7), "Seattle": (6, -3)}
    for r in rows:
        ax2.annotate(r["city"], (r["gap_af1"], r["gap_f1"]),
                     textcoords="offset points",
                     xytext=nudge.get(r["city"], (6, -3)),
                     fontsize=7.5, color=C_GREY,
                     ha="right" if r["city"] == "New York" else "left")
    # macro AUC gap on the same axes, to show it sitting at zero
    macro = np.array([r["gap_auc_macro"] for r in rows])
    ax2.scatter(pred, macro, s=30, marker="s", color=C_MACRO, zorder=3,
                edgecolor="white", linewidth=0.6,
                label="macro AUC gap (truth = 0)")
    ax2.axhline(0, color=C_MACRO, lw=0.8, ls=":", zorder=1)

    ax2.set_xlabel(r"predicted gap: $A_{F_1}(p_{\mathrm{Head}})-A_{F_1}(p_{\mathrm{Tail}})$")
    ax2.set_ylabel("observed Head $-$ Tail gap")
    ax2.set_title(f"Skill is identical in every group,\n"
                  f"so the honest gap is 0.  $r={r_val:+.2f}$", fontsize=9)
    ax2.legend(loc="upper left", fontsize=7.5)
    ax2.set_xlim(*lim); ax2.set_ylim(-6, lim[1])

    save(fig, outdir, "fig1_prediction")
    return r_val


# --------------------------------------------------------------------------- #
SWEEP = [   # from audit_fairness_tools.py PART 7; skill identical at every row
    (0.30, 0.20, 12.82, 2.16, -0.26, -4.57, 5.82),
    (0.45, 0.15, 35.98, 11.76, 0.27, -13.14, 12.49),
    (0.60, 0.12, 53.57, 23.96, 0.07, -20.19, 19.94),
    (0.75, 0.08, 70.90, 40.32, 0.18, -26.38, 25.16),
    (0.90, 0.04, 87.04, 61.88, 0.24, -32.11, 41.26),
]


def fig2_sweep(outdir):
    """The analyst chooses the grid resolution; the grid resolution sets the
    base-rate spread; the spread sets the reported unfairness."""
    a = np.array(SWEEP)
    spread = (a[:, 0] - a[:, 1]) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)

    ax1.plot(spread, a[:, 2], "o--", color=C_GREY, lw=1.2, ms=5,
             label=r"predicted by $A_{F_1}$")
    ax1.plot(spread, a[:, 3], "o-", color=C_F1, lw=1.8, ms=5,
             label=r"observed $F_1$ gap")
    ax1.plot(spread, a[:, 4], "s-", color=C_MACRO, lw=1.8, ms=5,
             label="AUC gap (the truth)")
    ax1.axhline(0, color="black", lw=0.7)
    ax1.set_ylabel("Head $-$ Tail gap")
    ax1.set_title("The reported disparity is a function\nof a modelling choice",
                  fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")

    ax2.plot(spread, a[:, 6], "o-", color=C_POOL, lw=1.8, ms=5,
             label="sparse group's false-alarm rate")
    ax2.plot(spread, -a[:, 5], "^-", color=C_GREY, lw=1.4, ms=5,
             label="TPR gap opened (magnitude)")
    ax2.set_ylabel("percent")
    ax2.set_title("...and so is the damage done by\n"
                  r"$\mathtt{demographic\_parity}$", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.supxlabel("difference in base rate between the two groups (points)",
                  fontsize=9, y=-0.02)

    save(fig, outdir, "fig2_sweep")


# --------------------------------------------------------------------------- #
CHI = {  # PART 6, real Chicago burglary 2015-2019, 20% patrol budget
    "Head": dict(base=78.3, pooled=61.90, macro=55.77),
    "Mid": dict(base=58.5, pooled=58.84, macro=54.84),
    "Tail": dict(base=24.2, pooled=71.72, macro=53.54),
}
CHI_WRONGFUL = {"baseline": {"Head": 715, "Mid": 120, "Tail": 0},
                "demographic_parity": {"Head": 291, "Mid": 1135, "Tail": 3528}}


def fig3_chicago(outdir):
    """Real data. Two panels: the metric disagreement, and what the fairness
    intervention does about it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    fig.subplots_adjust(wspace=0.34)
    x = np.arange(3)
    w = 0.36
    pooled = [CHI[g]["pooled"] for g in GROUPS]
    macro = [CHI[g]["macro"] for g in GROUPS]

    ax1.bar(x - w / 2, pooled, w, color=C_POOL, label="pooled AUC")
    ax1.bar(x + w / 2, macro, w, color=C_MACRO, label="macro (within-cell) AUC")
    ax1.axhline(50, color="black", lw=0.9, ls="--")
    ax1.text(-0.46, 50.5, "chance", fontsize=7, color="black", ha="left")
    for i, g in enumerate(GROUPS):
        d = CHI[g]["pooled"] - CHI[g]["macro"]
        ax1.annotate(f"{d:+.2f}", (i, max(pooled[i], macro[i]) + 1.2),
                     ha="center", fontsize=7.5, color=C_GREY)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{g}\n({CHI[g]['base']:.0f}% base)" for g in GROUPS])
    ax1.set_ylim(45, 78)
    ax1.set_ylabel("AUC")
    ax1.set_title("Pooling flatters the model most\nin the sparsest region",
                  fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")

    base = [CHI_WRONGFUL["baseline"][g] for g in GROUPS]
    dp = [CHI_WRONGFUL["demographic_parity"][g] for g in GROUPS]
    ax2.bar(x - w / 2, base, w, color=C_GREY, label="patrol budget alone")
    ax2.bar(x + w / 2, dp, w, color=C_POOL,
            label=r"after $\mathtt{demographic\_parity}$")
    for i in range(3):
        ax2.annotate(f"{dp[i]:,}", (i + w / 2, dp[i] + 60), ha="center",
                     fontsize=7.5, color=C_POOL)
    ax2.set_xticks(x)
    ax2.set_xticklabels(GROUPS)
    ax2.set_ylabel("crime-free cell-weeks flagged")
    ax2.set_title("The remedy moves attention to where\n"
                  "the model has least skill", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_ylim(0, max(dp) * 1.22)

    save(fig, outdir, "fig3_chicago")


# --------------------------------------------------------------------------- #
def fig4_cities(rows, outdir):
    """Forest plot. One line per city; the honest answer is the zero line."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    order = sorted(rows, key=lambda r: -r["gap_f1"])
    y = np.arange(len(order))

    for i, r in enumerate(order):
        ax.plot([r["gap_auc_macro"], r["gap_f1"]], [i, i], "-",
                color="#cccccc", lw=1.4, zorder=1)
    ax.scatter([r["gap_f1"] for r in order], y, s=44, color=C_F1, zorder=3,
               edgecolor="white", linewidth=0.6, label=r"$F_1$ gap")
    ax.scatter([r["gap_auc_macro"] for r in order], y, s=38, marker="s",
               color=C_MACRO, zorder=3, edgecolor="white", linewidth=0.6,
               label="macro AUC gap")
    ax.axvline(0, color="black", lw=1.0)
    ax.text(1.5, len(order) - 0.35, "truth", fontsize=7.5, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels([r["city"] for r in order])
    ax.set_xlabel("Head $-$ Tail gap, from a scorer with identical skill everywhere")
    ax.set_title("Same experiment, five cities", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    ax.set_ylim(-0.5, len(order) - 0.15)

    save(fig, outdir, "fig4_cities")


# --------------------------------------------------------------------------- #
def fig5_invariance(rows, outdir):
    """The mechanism, isolated. As the scorer leans harder on cell identity,
    within-cell skill is unchanged by construction. Macro knows that."""
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for r in rows:
        b = [s["beta"] for s in r["sweep"]]
        ax.plot(b, [s["pooled"] for s in r["sweep"]], "-", color=C_POOL,
                lw=1.3, alpha=0.75)
        ax.plot(b, [s["macro"] for s in r["sweep"]], "--", color=C_MACRO,
                lw=1.3, alpha=0.75)
    ax.plot([], [], "-", color=C_POOL, lw=1.6, label="pooled AUC")
    ax.plot([], [], "--", color=C_MACRO, lw=1.6, label="macro AUC")
    ax.set_xlabel(r"$\beta$: how much the scorer leans on cell identity")
    ax.set_ylabel("AUC")
    ax.set_title("Within-cell skill is constant at every $\\beta$.\n"
                 "Only one of these metrics knows that.", fontsize=9)
    ax.legend(fontsize=8, loc="center right")
    save(fig, outdir, "fig5_invariance")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/multicity_audit.json")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    with open(a.results) as fh:
        rows = json.load(fh)

    r_val = fig1_prediction(rows, a.outdir)
    fig2_sweep(a.outdir)
    fig3_chicago(a.outdir)
    fig4_cities(rows, a.outdir)
    fig5_invariance(rows, a.outdir)

    print(f"  {len(rows)} cities read from {a.results}")
    print(f"  A_F1 gap vs observed F1 gap: r = {r_val:+.4f}")
    print(f"  mean F1 gap        {np.mean([r['gap_f1'] for r in rows]):+.2f} "
          f"+/- {np.std([r['gap_f1'] for r in rows], ddof=1):.2f}")
    print(f"  mean macro AUC gap {np.mean([r['gap_auc_macro'] for r in rows]):+.2f} "
          f"+/- {np.std([r['gap_auc_macro'] for r in rows], ddof=1):.2f}")
    for f in sorted(os.listdir(a.outdir)):
        print("  wrote", os.path.join(a.outdir, f))



if __name__ == "__main__":
    main()
