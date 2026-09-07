# ICLR 2027 — paper outline

**Deadlines.** Abstract 18 Sept 2026 · Full paper 25 Sept 2026 · 9 pages main
text, unlimited references · double-blind (desk reject if identity leaks) ·
AI-use statement mandatory.

---

## The framing decision

The FAccT version of this work says: *fairness metrics are confounded by base
rate, and this harms sparse communities.* That is the right paper for FAccT and
the wrong paper for ICLR. An ICLR reviewer answers it with "Davis & Goadrich
2006" and scores it 3.

The ICLR version says something an ML audience has to care about:

> **A subfield is reporting architectural gains that a single decision
> threshold reproduces or exceeds. We give an evaluation protocol that
> separates the two, show that it dissolves the reported gains across 24
> capacity-matched configurations, and show that it still detects a real gain
> when one exists.**

The difference is that this is a claim about **evaluation methodology in ML**,
not about social harm. Same experiments, different load-bearing sentence.

**Why the protocol framing beats a pure negative result.** ICLR reviewers
distrust "nothing works" papers — they read as failed engineering. The PINN is
what makes this publishable: it is a **positive control**. It proves the
protocol has power to detect a genuine improvement, so the null results are
informative rather than a sign of a weak experimental setup. Lead with that
symmetry.

**Title candidates**

1. Threshold Artifacts in Spatiotemporal Prediction
2. A Decision Threshold Outperforms the Architecture: Capacity-Matched Controls for Spatiotemporal Forecasting
3. What Survives a Capacity-Matched Control? Re-evaluating Graph and Physics Priors in Crime Forecasting

---

## Section plan (9 pages)

### 1. Introduction — 1.25 pp

Open with the single most arresting fact, in the first 100 words:

> On the sparse-region subset where improvement is hardest, adding a graph
> neural network raises F1 by **+12.87**. Moving the decision threshold from
> 0.50 to 0.35 — changing no parameter, adding no information — raises it by
> **+13.63**, with *higher* AUC (61.99 vs 58.94) and less over-prediction.

Then: this is not a quirk of one model. State the three contributions.

**Contributions (write these as a bulleted list, reviewers look for it):**

1. A four-part evaluation protocol: capacity-matched controls, bit-level
   determinism verification, threshold sweeps, and threshold-free primary
   metrics.
2. Applying it to graph priors for spatiotemporal crime forecasting: **0 of 24**
   capacity-matched configurations show a significant gain; gated variants learn
   to switch the graph off.
3. Applying it to a physics prior: **+18.32 AUC** (t = 9.57), which no threshold
   can produce, with an ablation that isolates the mechanism.

### 2. Related work — 0.75 pp

Three short paragraphs. Be generous, not defensive.

- **Spatiotemporal crime forecasting** — FedCrime, ST-HSL, the six audited papers.
- **Metric behaviour under class imbalance** — Davis & Goadrich (2006), Boyd
  et al. (2012), Chouldechova (2017). **Cite these prominently and early.**
  Claiming novelty here is the fastest route to rejection; claiming that the
  field has not *applied* them is defensible and true.
- **Evaluation critiques in ML** — the GNN-baselines line (Shchur et al.,
  Dwivedi et al.), recommender-systems reproducibility (Dacrema et al.),
  metric-learning (Musgrave et al.). **This is your peer group.** Say so.
  It tells the reviewer what kind of paper they are holding.

### 3. The protocol — 1 pp

State it as four requirements, each with the failure it prevents:

| Requirement | Failure it prevents | Our instance |
|---|---|---|
| Capacity-matched control | attributing depth to the prior | per-depth no-graph control, same layer count |
| Verified determinism | reporting seed noise | `--repro-check`, bit-identical, diff 0.00000000 |
| Threshold sweep | attributing operating point to the model | τ ∈ [0.05, 0.95], 19 points |
| Threshold-free primary metric | base-rate confounding | AUC primary, F1 reported secondary |

Include the one-line theory: with `TPR(τ)` and `FPR(τ)` free,
`F1(p,τ) = 2·TPR·p / [TPR·p + FPR(1−p) + p]` has two degrees of freedom that
have nothing to do with model quality. Keep the derivation in the appendix.

**Honesty note for this section:** every individual ingredient is standard
practice. The contribution is the *combination* applied to a subfield that
uses none of them, plus the demonstration that it changes conclusions. Say
that explicitly — a reviewer who thinks you're claiming to invent
capacity-matched controls will be annoyed.

### 4. Negative result: graph priors — 2 pp

- Setup: 3 architectures (plain / gated / attention) × 4 depths × 2 cities
  (LA, Chicago) × 5 seeds, capacity-matched, deterministic.
- **Table 1.** Deltas vs matched control. LA 12/12 negative, 9/12 significant;
  Chicago 10/12 negative; **0/24 significantly positive.**
- Mechanism, not just outcome — this is what lifts it above a benchmark dump:
  - Plain GCN over-smooths: region diversity 1.3495 → **0.0359** at depth 4
    (97% of inter-region distinction destroyed).
  - Gated GCN learns gate ≈ 0 — it *chooses* not to propagate. The limitation
    is informational, not architectural.
- **Figure 1 (the money figure).** Threshold sweep. Tail F1 vs τ for the
  no-graph model as a curve, with the graph model as a single point sitting
  *below* the curve. Second panel: AUC, flat at 61.99 across all 19 τ.
- Limitation stated here, not buried: our graph is crime-pattern correlation,
  not street adjacency. A physically grounded graph is untested.

### 5. Positive control: physics priors — 1.75 pp

This section exists to prove the protocol is not simply insensitive.

- Short et al. (2008) reaction–diffusion system; give both PDEs and the
  term-by-term table (already written in `report/report.tex`).
- Poisson NLL for count data, `A0` as a learned spatial field, collocation
  masked to the city.
- **Table 2**, 5 seeds:

  | | pooled AUC | Δ vs control | t |
  |---|---|---|---|
  | Short PDE + Poisson | 71.49 ± 2.93 | +18.32 | 9.57 |
  | Short PDE + MSE | 71.12 ± 1.50 | +17.95 | 12.46 |
  | no physics (control) | 53.17 ± 2.46 | — | — |
  | **generic smoothness** | **44.63** | −6.85 | — |

- **Lead with the ablation.** Generic smoothing scores *below the control and
  below chance*. That is what rules out "any regulariser would do" and it is
  the most reviewer-proof result in the paper.
- Same measurement point, restated: across these configs pooled AUC spans
  18.32 points while the Head−Tail F1 gap spans 2.06. The metric the field
  reports is nearly blind to an 18-point change in model quality.
- **State the weakness before a reviewer does:** per-group AUC is 51–58. The
  model learned *where*, not *when*. Own it in the main text.

### 6. The protocol failure is live in deployed tooling — 0.75 pp

Short, punchy, one table.

- Equal-ROC-by-construction data; Fairlearn `MetricFrame` reports 52.67 points
  of precision disparity and 23.52 of F1; AUC reports 0.84.
- Fairlearn's **default** `ThresholdOptimizer` constraint opens a TPR gap of
  −21.26 ± 0.52 where none existed (5 seeds).
- AIF360 `RejectOptionClassification`: Tail FPR 13.49% → **41.55%**, AUC
  unchanged.
- **Frame carefully.** The DP/equalized-odds tension is Kleinberg et al. 2016
  and Chouldechova 2017 — cite both in the first sentence of the section. Your
  claim is narrow: this known incompatibility is the *default setting* of
  widely used software, and its cost is quantifiable.

### 7. Limitations and conclusion — 0.5 pp

Two cities, one country, one crime type for the PINN. Correlation graph only.
Per-group AUC modest. Say all of it plainly — ICLR reviewers reward this and
punish its absence.

---

## Figures (build these early; they take longer than you think)

1. **Threshold sweep** — the money figure. Tail F1 vs τ curve + graph model as
   a point below it; AUC flat panel beneath.
2. **Depth × architecture deltas** — 24 configurations, both cities, error bars.
3. **PINN ablation bar chart** — four bars, chance line at 50, control line at
   53.17. Generic smoothness visibly below both.
4. **Over-smoothing diagnostic** — region diversity vs depth.

`figures/` already has `g1_absolute.png`, `g2_zoom.png`, `g3_all24.png`,
`s1_gap.png` — these need redrawing at ICLR column width with readable fonts.

---

## 22-day schedule

| Dates | Work |
|---|---|
| **3–5 Sept** | Lock framing. ICLR style files. Rebuild 4 figures at final size. |
| **6–10 Sept** | Draft §3 protocol, §4 negative result. These are the core — write them first, while fresh. |
| **11–14 Sept** | Draft §5 PINN, §6 tooling. Assemble all tables from saved result files. |
| **15–17 Sept** | Draft §1 intro and §2 related work **last** — you can only write a good intro once the paper exists. Write the abstract. |
| **18 Sept** | **Abstract deadline.** Submit title + abstract to OpenReview. |
| **19–22 Sept** | Full revision pass. Send to Sir. Appendix: derivations, hyperparameters, repro instructions. |
| **23–24 Sept** | Sir's comments. Anonymity sweep. AI-use statement. |
| **25 Sept** | **Submit.** |

---

## Practical traps

- **Anonymity.** Your GitHub repo carries your name. A link to it is an instant
  desk reject. Mirror it to `anonymous.4open.science` and cite that. Check the
  PDF metadata too — LaTeX embeds the author field.
- **AI-use statement.** ICLR 2027 requires disclosure. You used an AI assistant
  for code and drafting; state it plainly in the required section. It is not a
  penalty, and omitting it is a policy violation.
- **Do not cite results from Phase 7.** Those numbers predate the determinism
  fix and are void.
- **Write §1 last.** Every hour spent polishing an introduction before the
  results are written is an hour wasted.
