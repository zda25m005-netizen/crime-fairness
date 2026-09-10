# Ethics section — source and argument pack (FAccT item 3)

Compiled 2026-09-10. **These are notes and verified sources, not paper text.**

## 0. Read this first: you must write the prose yourself

The FAccT 2027 Author Guide (last updated August 2026) states:

> FAccT prohibits the use of LLMs to generate text for publications, although
> authors may use it to assist with formatting (e.g., tables), or assist with
> grammar or fluency of writing, provided they disclose this usage.

and

> all authors are required to include a generative AI usage statement ...
> Failure to disclose the use of generative AI tools is grounds for desk
> rejection.

and

> Individual authors will be considered non-delegably responsible for the
> content of their submitted papers. ... Any lack of veracity in content that
> is surfaced in the Reviewing Process will be considered grounds for desk
> rejection.

So: this document supplies **sources, figures and argument structure**. Turning
it into sentences is the author's job, and every number below must be checked
against its primary source before it appears in a submission. Do not paste from
here.

Practical consequence for the whole paper, not just this section: anything in
the draft that came out of a chat assistant as prose needs rewriting in your own
words. Grammar and formatting help is permitted **with disclosure**.

## 1. What the statement has to do

From the Author Guide, Endmatter Sections:

- **Ethical Considerations Statement** — *should be included at submission time*.
  "A description of the ethical concerns and potential adverse impacts that
  authors considered and mitigated while conducting the work."
- "We also encourage authors to discuss any potential adverse or unintended
  impacts the work might have once published, and how they have mitigated those
  potential impacts."
- Endmatter goes on its own page and **does not count towards the 14-page
  limit**.
- Positionality, Acknowledgements, Author Contributions and Competing Interests
  must be **omitted** from the anonymous submission.
- Note also: the guide explicitly says endmatter "is not meant for the
  discussion of limitations of your methodology, which should be addressed in
  the main body." So the macro-vs-pooled limitations belong in the body, not
  here.

## 2. The reflexive problem — confront it in the first paragraph

This project built crime-prediction models (GNN variants, a physics-informed
model) before turning to critique. A reviewer will ask why. The honest account:

- the work began as a reproduction of a published federated crime-prediction
  result;
- the critique emerged from failing to reproduce a *fairness* claim, and then
  from finding the same failure in our own headline number;
- we retracted our own result (documented in `STATUS.md`, dated 2026-09-07)
  rather than publishing it.

Saying this plainly is a strength. Concealing it and having a reviewer find the
model code in the repo is fatal.

## 3. The central adverse impact: this paper is weaponisable

**The risk.** "Reported fairness gaps in crime prediction are largely artifacts
of base rates" is one careless inferential step from "there is no unfairness in
predictive policing." A vendor or department could cite the paper for the
second claim. This is the single most important thing the statement must
address, and it must be addressed with our own evidence rather than a
disclaimer.

**The rebuttal, from our own results — all of which point the other way:**

| finding | source in repo |
|---|---|
| macro AUC 55.77 / 54.84 / 53.54 on real Chicago — near chance in every group | PART 6, `analysis/audit_fairness_tools.py` |
| no neural model beat a 4-week moving average on the base-rate-free target | `src/pinn/signal_ceiling.py` |
| 22/24 capacity-matched graph configurations negative | earlier phases |
| under `demographic_parity`, Tail goes 0 → 3,528 crime-free cell-weeks flagged | PART 6 |

The correct reading is therefore: **the standard metrics are too weak to
detect harm, not that there is no harm.** A metric that reports a 41-point gap
where the truth is zero is equally capable of reporting zero where the truth is
large. Unreliability is not exoneration. State that explicitly.

**A second, sharper version.** Our strongest single real-data result is that
the pooled-vs-macro inflation is *largest in the Tail* (+18.18, vs +6.13 Head
and +4.00 Mid). Pooled AUC nominates the sparse region as the best-served group
at 71.72; within cells it is the worst at 53.54. Pooled and macro disagree
about the **sign** of the gap (−9.82 vs +2.23) and therefore about which
neighbourhood gets remediation. That is a claim about being *unable to see*,
not a claim that there is nothing to see.

## 4. Recorded crime is not crime — and this compounds our own analysis

The literature to engage, with what each actually says:

**Lum & Isaac (2016), "To predict and serve?", *Significance* 13(5):14–19.**
Applied a simulation of the PredPol algorithm to Oakland drug-crime police
records. Drug arrests are heavily concentrated in non-white, low-income areas
(West Oakland, near International Boulevard), while the 2011 National Survey on
Drug Use and Health indicates drug use is roughly even across the city. The
algorithm would therefore have directed patrols disproportionately to Black and
Latino neighbourhoods. *Verify the exact multiplier before quoting it — press
coverage and the paper differ in how they state it.*

**Richardson, Schultz & Crawford (2019), "Dirty Data, Bad Predictions", *NYU
Law Review Online* 94:15.** Identified **13 jurisdictions** with documented
unlawful or biased police practices that had also deployed or explored
predictive policing during the period of that unlawful activity. Core argument:
"dirty policing" produces "dirty data", and systems trained on it cannot escape
the practices they are built on.

**Ensign, Friedler, Neville, Scheidegger & Venkatasubramanian (2018),
"Runaway Feedback Loops in Predictive Policing", FAT\* / PMLR 81.** Formalises
the feedback loop: patrol where you predicted, discover crime there, feed it
back.

**Why this matters for *us* specifically, and this is the part to make our
own:** our Head/Mid/Tail groups are defined by *recorded* burglary volume. Our
base rates are therefore themselves a product of policing. So the confound we
document sits on top of a confound. The two arguments **compound**; they do not
compete:

- the literature says the data is shaped by enforcement;
- we say that even taking the data entirely at face value, the metric still
  fails.

Our argument is **conditional** — it grants the data more credibility than it
deserves and the metric still breaks. Taking the data *less* seriously only
strengthens the conclusion. Say this; it converts the most obvious attack into
support.

## 5. These systems do not work, separately from whether they are fair

Useful because the ethics debate is usually framed as an accuracy/fairness
trade-off. The record suggests there is little accuracy to trade.

**The Markup (2 Oct 2023), Plainfield NJ.** Examined 23,631 Geolitica
predictions generated 25 Feb – 18 Dec 2018 for Plainfield Police Department.
Success rate **under 0.5%** — fewer than 100 predictions matched a subsequently
reported crime of the predicted category. Plainfield Police Captain David
Guarino: the department barely used it. Geolitica ceased operations at the end
of 2023; SoundThinking (formerly ShotSpotter) hired its engineering team and
acquired IP.

**Saunders, Hunt & Hollywood (2016), RAND, on Chicago's Strategic Subject
List.** Those on the list were **no more or less likely** to be homicide or
shooting victims than a matched control group, but *were* considerably more
likely to be arrested. 426 individuals in the study period; **77% Black, 95.8%
male**. CPD discontinued the list in November 2019 (reported January 2020)
following Office of the Inspector General criticism. *Get the exact
citation for the journal version — "Predictions put into practice: a
quasi-experimental evaluation of Chicago's predictive policing pilot",
Journal of Experimental Criminology — and check the numbers there, not from
press coverage.*

**Our own contribution to this line:** macro AUC near 50 everywhere on real
Chicago data, and no neural model beating a 4-week moving average. We are
adding a measurement reason for the same conclusion.

## 6. Policy context — accurate, and do not overclaim

The **EU AI Act Article 5(1)(d)** prohibits AI used by or on behalf of law
enforcement to make individual risk assessments predicting the likelihood of a
person committing a crime **based solely on profiling or on assessment of
personality traits and characteristics**, rather than on objective verifiable
facts linked to criminal activity. Prohibitions have been enforceable since
February 2025.

**Important for honesty:** our work is **place-based**, not person-based.
Article 5(1)(d) does not cover it. Place-based law-enforcement systems fall
under the Annex III high-risk regime instead, with conformity and
risk-management obligations rather than prohibition. The Commission published
guidelines on the Article 5 / Annex III boundary (dated 19 May 2026) — **read
those before writing this paragraph**, and do not imply our system class is
banned when it is not. Overclaiming here is exactly the kind of thing a FAccT
reviewer with a law background will catch.

## 7. What we did not do — state it flatly

- We improved nothing about policing, and did not try to. That is the paper.
- No system was deployed; no police department was a partner; no operational
  decision was informed by this work.
- We do not recommend that anyone deploy any model in this paper, including the
  baselines that outperformed our neural models.
- We make no claim that FedCrime or any cited system is "wrong." FedCrime's
  actual claim was federated training without data sharing; we reproduced it
  and do not challenge it. The Head/Mid/Tail split was **ours**.

## 8. Data ethics

- All data is public open-government data (Socrata portals: Chicago, Los
  Angeles, New York, Seattle, Cincinnati). No individual-level records; every
  analysis is on counts aggregated to grid cells and weeks.
- No human subjects; no IRB required. Say so, and say why.
- **San Francisco's portal returned HTTP 403 to our client and we did not
  attempt to circumvent it.** Worth one sentence — it demonstrates the access
  norms we followed. The city is reported as excluded for that reason.
- Burglary was chosen partly because property crime is less enforcement-
  discretionary than drug offences (the Lum & Isaac case). But it is still
  reporting- and enforcement-mediated: burglary reporting varies with insurance
  coverage and with trust in police, both of which correlate with the
  neighbourhood characteristics at issue. Do not present it as clean.
- Per-city year windows differ (Los Angeles 2020–24, others 2015–19) because
  portals publish different periods. Disclose it.

## 9. Objections to pre-empt, with the answer we can actually defend

**"This is just Kleinberg et al. (2016) and Chouldechova (2017)."**
Concede immediately and completely — the impossibility result is theirs and we
claim none of it. What is ours: it is the *library default*; the cost is
measured in counts of crime-free places flagged rather than stated as a
theorem; and the metric-function result shows the confound inside the tooling
practitioners actually run.

**"This will be cited to defend predictive policing."**
See §3. Answer with our own null results, not with a disclaimer.

**"You built the thing you criticise."**
See §2. Answer with the retraction.

**"Your base rates are themselves artifacts of biased policing, so the whole
analysis is on tainted data."**
The strongest objection. See §4 — the argument is conditional and the objection
strengthens it. Prepare this answer properly; it is likely to appear in a
review.

**"Why should we believe your numbers when you retracted your last ones?"**
Because the retraction is the evidence of the method working. Point at
`macro_auc()`, the invariance proof (a scorer knowing only cell identity gets
pooled 66.67 / macro 50.45), the five-city replication, and the fact that the
critique requires no model and no GPU and can be re-run from the repo in
minutes.

## 10. Provenance of the physics model

The Short et al. (2008) burglary model derives from repeat-victimisation and
broken-windows reasoning. Broken windows is contested — Sampson & Raudenbush
(2004); Goodson & Hoyer-Leitzel (2021). We implemented it to *measure* what it
does, not because we endorse it. It did not work, which is reported.

## 11. Generative AI usage statement — required

Must be written by the authors, must be accurate, and must cover the whole
manuscript, not just this section. Disclose honestly what assistance was used
and for what (e.g. code, analysis scripts, formatting, grammar), and be clear
that no publication text was LLM-generated. If any drafted prose from an
assistant survives anywhere in the manuscript, rewrite it before submission.

## 12. Process note that affects strategy

FAccT 2027 has a new decision structure: **Accept / Revise (major or minor) /
Reject**, with a revision-and-rebuttal round, re-review, and an extra page (up
to 15) at revision stage. A "Revise" is not a rejection. This raises the value
of submitting: the realistic good outcome is a major revision with reviewer
feedback, which is precisely what the professor said the goal was.

Also note: the author list **cannot be changed after the abstract deadline**.
Settle authorship with Manoj Sir before registering the abstract.

## Sources

- FAccT 2027 Author Guide — https://facctconference.org/2027/authorguide.html
- Lum & Isaac, "To predict and serve?" — https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1740-9713.2016.00960.x
- Richardson, Schultz & Crawford, "Dirty Data, Bad Predictions" — https://www.nyulawreview.org/wp-content/uploads/2019/04/NYULawReview-94-Richardson_etal-FIN.pdf
- Ensign et al., "Runaway Feedback Loops in Predictive Policing" — https://arxiv.org/pdf/1706.09847
- The Markup, Geolitica accuracy investigation — https://themarkup.org/prediction-bias/2023/10/02/predictive-policing-software-terrible-at-predicting-crimes
- The Markup, methodology — https://themarkup.org/show-your-work/2023/10/02/how-we-assessed-the-accuracy-of-predictive-policing-software
- RAND, "CPD's 'Heat List' and the Dilemma of Predictive Policing" — https://www.rand.org/blog/2016/09/cpds-heat-list-and-the-dilemma-of-predictive-policing.html
- Chicago Sun-Times, CPD ends the Strategic Subject List — https://chicago.suntimes.com/city-hall/2020/1/27/21084030/chicago-police-strategic-subject-list-party-to-violence-inspector-general-joe-ferguson
- EU AI Act Annex III — https://artificialintelligenceact.eu/annex/3/
- Article 5 / Annex III boundary in law enforcement — https://www.aiactblog.nl/en/posts/high-risk-ai-law-enforcement
