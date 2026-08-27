# Manuscript Blueprint

**Purpose**: lock the complete manuscript design — title, storyline, figure/table order, per-section content, claim placement, and the reviewer-attack map with answers — *before* writing any Introduction/Methods/Results/Discussion prose. Per `editorial_review_bspc.md`'s scoring (Scientific quality 8.5/10, Novelty 7.5/10 conditional, Writing not yet assessable) and the working decision that followed it: this document is the thing that gets argued over and revised, not the manuscript itself. Once this is stable, writing should be closer to transcription than composition.

**Framing decision this entire blueprint is built around** (the missed point from the editorial review): **this is an audit paper, not a Mahalanobis paper.** The Introduction's failure mode is opening with "Mahalanobis distance has limitations." The correct opening move is: distance-based reliability estimation implicitly assumes representation geometry reflects uncertainty; nobody has tested whether a representation-changing training intervention preserves that assumption; we audit it. Mahalanobis, cosine, and k-NN are the three instruments used to run the audit — none of them is the subject.

---

## 1. Title (ranked)

1. **Decodable but Not Accessible: Auditing Distance-Based Reliability Estimation on Disentangled Skin-Lesion Representations** — names the actual finding, uses "auditing" explicitly, never centers Mahalanobis. Top pick.
2. **When Representation Geometry Changes but Reliability Estimation Does Not: A Multi-Scorer Audit Under Domain-Adversarial Training** — states the core result as the title; slightly long but unambiguous.
3. **Does Representation Geometry Predict Reliability? Evidence from a Disentanglement Dose-Response Ladder in Skin Lesion Classification** — question-form, signals "we tested an assumption," strong audit framing.
4. **Information Availability Versus Information Accessibility: An Audit of Distance-Based Out-of-Distribution Detection in Skin Lesion Classifiers** — most direct statement of the novelty; slightly abstract/ML-venue-flavored for BSPC's typical readership.
5. **Auditing the Geometric Assumptions Behind Distance-Based Reliability Estimation in Domain-Adversarial Skin Lesion Models** — safe, plain, correctly scoped; least memorable.
6. **Geometry Changes, Reliability Doesn't: A Dose-Response Audit of Disentangled Representations for Skin-Lesion OOD Detection** — punchy, good for abstract's first line even if not the final title.
7. **The Limits of Distance: Why Domain Information Can Be Decodable Yet Inaccessible in Disentangled Skin-Lesion Representations** — literary framing, higher risk/reward.
8. Do not use anything containing "Mahalanobis" as the headline noun (e.g. ~~"Mahalanobis Fails to..."~~) — this is the exact framing trap item 3 above exists to avoid; fine as a Methods-section term, wrong as the paper's identity.

**Recommendation**: #1 for submission, #6 as a candidate running header / first abstract sentence.

---

## 2. Storyline (one page, this is the argument the whole paper exists to make)

Distance-based reliability estimation — Mahalanobis distance and its relatives — is used throughout clinical ML as an OOD/uncertainty signal, on the implicit assumption that a representation's *geometry* (how far a point sits from a class-conditional typical region) reflects how *trustworthy* a prediction on that point is. This assumption is rarely tested directly, and almost never tested under an intervention that deliberately reshapes representation geometry — such as shortcut-mitigation or domain-adversarial disentanglement training, both increasingly common in clinical imaging to reduce spurious acquisition-site or demographic shortcuts.

We audit this assumption directly, using a disentanglement dose-response ladder (three checkpoint families sharing one architecture and one 16-dimensional representation, differing only in orthogonality regularization strength λ_orth = 0, 1, 5). First: does the intervention actually change geometry? Yes, substantially — condition number moves two orders of magnitude with λ_orth (τ=0.84, p=2.8×10⁻⁵), the single cleanest quantitative result in the study. Second: does that geometric change track reliability estimation quality? No — Mahalanobis AUROC (ISIC-test vs. PAD-UFES) stays flat at ~0.40, *below* chance, with no significant association to any of five geometry metrics tested, despite the 100-fold conditioning swing. Third, because a null result for one estimator invites the obvious rebuttal ("try a different scorer"): is this Mahalanobis-specific? No — cosine-to-centroid and pooled k-NN (three assumption levels apart: parametric with covariance, parametric without covariance, fully non-parametric) all land in the identical ~0.40–0.42 band. Three structurally unrelated distance-based instruments fail identically, which rules out the scoring rule as the site of the problem.

That leaves the representation. But "the representation lost the information" and "the representation still has the information, inaccessibly" are different, distinguishable claims — and only one is supported. A supervised probe (three classifiers, none seeing the training objective) trained on the identical embeddings decodes ISIC-vs-PAD-UFES domain membership at ~0.73–0.81 AUROC, consistently across every λ_orth level. The information a distance-based scorer would need is present in the representation; it is simply not organized in a way any of the three tested distance-based geometries can reach.

This is the paper's actual contribution: not "Mahalanobis is broken," but a demonstrated dissociation between **decodability** (is domain-relevant information present, extractable by a supervised readout) and **distance accessibility** (is that information usable by an unsupervised, geometry-based scorer) — audited, not assumed, across three qualitatively different scorer families, on a representation subjected to a training intervention that is exactly the kind increasingly recommended for shortcut mitigation in clinical imaging. The paper closes by scoping this honestly: measured within one training regime, one dataset pair (with PAD-UFES also serving as the adversarial training target, disclosed explicitly, not discovered by a reviewer), one architecture — a first demonstration, not a general law, with the specific missing experiment (a matched, non-adversarial, same-dimensionality control) named as the direct next step rather than glossed over.

---

## 3. Figure order

**Main text (6 figures)**

| # | Content | Source | New or existing |
|---|---|---|---|
| 1 | Conceptual schematic: clinical motivation → implicit assumption (geometry ≈ reliability) → ladder design → the audit question | — | **New** |
| 2 | Geometry changes across the ladder — condition number highlighted (primary panel) + Fisher-ratio-scalar as its decoupled companion (per `fisher_ratio_defense.md`'s own recommendation) | `figure1_e1_ladder_trend.png`, condensed to 2 panels | Trim existing |
| 3 | Geometry vs. Mahalanobis AUROC — condition number panel only in main text, full 5-metric grid to Supplementary | `figure2_e2_geometry_vs_auroc.png`, condensed to 1 panel | Trim existing |
| 4 | All three scorer families fail alike (Mahalanobis / cosine / k-NN×3, grouped by rung, chance line at 0.5) | `figure_e2_6_scorer_comparison.png` | Existing, as-is |
| 5 | **Headline figure — the paper's best evidence, currently unbuilt.** Distance-scorer AUROC band vs. domain-probe AUROC band, per rung, chance line at 0.5. See spec below. | — | **New — highest priority** |
| 6 | Ruling out simple mechanisms: NV-attractor gap (headline P(NV\|ID) vs. P(NV\|OOD) numbers) + feature-norm gap, one compact combined panel | `figure_id_confusion_matrix.png` + `figure_norm_*` condensed | Condense existing |

**Figure 5 spec (build this one carefully — it is the 5-second read the whole paper hinges on)**:
- X-axis: the three primary-ladder rungs, ordered by λ_orth.
- Y-axis: AUROC, 0–1, dashed reference line at 0.5.
- Two bands/series per rung: "distance-based scorers" (range or mean±SD across all 5 E2.6 variants — visually reads as one clustered, at-or-below-chance band) vs. "domain probe" (range or mean±SD across the 3 E2.7 probes — visually reads as one clustered, well-above-chance band).
- Alternative worth prototyping: a paired dumbbell/slope plot (one dot for distance-scorer mean, one for probe mean, per rung, connected by a line) — may show "the gap" even more viscerally than grouped bars.
- `baseline_soft` does **not** appear on this figure. Its analogous numbers go to a separate Supplementary figure/table, explicitly not sharing axes with the headline claim, so a reader's eye cannot average a confounded n=1 point into the clean ladder evidence.
- Caption must state plainly: same embeddings, same checkpoints, same 13-point ladder as Figures 2–4; error bands reflect variation across scorer/probe type and seed, not held-out generalization uncertainty.

**Supplementary**
- Full 5-metric versions of Figures 2 and 3 (all of condition number, Fisher-ratio-HL, Fisher-ratio-scalar, Mardia kurtosis b/z).
- `baseline_soft` reference figure/table (geometry, once E1b is run — see §9 of `editorial_review_bspc.md` — plus its E2.6/E2.7 numbers), explicitly captioned "descriptive reference, confounded by architecture/dimensionality/training recipe, n=1 checkpoint — not a controlled comparison."
- Full ID confusion-matrix heatmap (`figure_id_confusion_matrix.png`).
- Full distance-distribution histogram/KDE/ECDF panels (`figure_e2_5_distance_distributions.png`, all 3 rungs) — supports the "below chance but overlapping, not cleanly separated" honesty already established in this project's own analysis scripts.
- Feature-norm distribution/boxplot figures (`figure_norm_distributions.png`, `figure_norm_boxplot.png`).
- Full predicted-class breakdown (`figure_predicted_class_ood.png`).

---

## 4. Table order

**Main text**

1. **Table 1 — Ladder design.** Rung, λ_orth, architecture, representation dimensionality, seeds available, checkpoint count. Orients the reader before any numbers — currently doesn't exist as a standalone table anywhere (the information is scattered across `SPEC.md` prose).
2. **Table 2 — Geometry summary** (mean±SD per rung per metric). From `table1_e1_summary.csv`.
3. **Table 3 — Mahalanobis AUROC summary** per rung. From `table2_e2_summary.csv`.
4. **Table 4 — Scorer comparison summary** (5 scorers × 3 rungs + pooled). From `table_e2_6_scorer_summary.csv` — coexists with Figure 4 deliberately (readers who want exact numbers for citation shouldn't have to read them off a bar chart).
5. **Table 5 — Domain probe vs. distance-scorer summary**, the numeric companion to headline Figure 5. From `table_e2_7_domain_vs_distance.csv`, `baseline_soft` row **excluded** from this table (goes to Supplementary Table S-Baseline instead, same reasoning as Figure 5).

**Supplementary**
- Full Kendall's τ tables (`e1_kendall_tau.csv`, `e2_kendall_tau.csv`, `e2_6_kendall_tau.csv`, `e2_7_kendall_tau.csv`) — full-ladder and common-seed-subset variants both.
- Full Jonckheere-Terpstra tables (`e1_jonckheere_terpstra.csv`, `e2_jonckheere_terpstra_auroc.csv`, `e2_6_jonckheere_terpstra.csv`).
- NV-attractor headline table (`nv_attractor_headline.csv`), feature-norm summary (`norm_summary.csv`).
- Table S-Baseline: `baseline_soft`'s full row (distance scorers + domain probes), same explicit "descriptive only" caption as its figure.

---

## 5. Section-by-section content

- **Title / Abstract**: per §1/§2 above; abstract already drafted (`abstract.tex`), needs revision once Hewitt & Liang and the PAD-UFES scoping sentence are incorporated into the argument they summarize.
- **Introduction**: implicit assumption → untested under representation-changing interventions → audit framing stated explicitly in the first paragraph → contribution statement (a positive claim, not `internal_review.md`'s flagged "triple negative" pattern) → one-paragraph result preview → brief related-work pointers, full treatment deferred to §Related Work.
- **Related Work**: shortcut learning & disentanglement (motivates why this training family exists); distance-based OOD detection family (Mahalanobis, k-NN, cosine — the three audited instruments); probing-classifier literature, **including the probing-vs-selectivity critique** (Hewitt & Liang) stated upfront as the reason this paper's decodability claim is scoped carefully, not oversold; representation-geometry context (Neural Collapse) for why condition number/class-separation geometry is a meaningful thing to measure at all.
- **Methods**: dataset/domain description **with the PAD-UFES-as-adversarial-target relationship disclosed here**, not left for a reviewer to discover in Discussion; ladder/architecture description; the three geometry metrics with audit-driven selection rationale (condensed from `geometry_metric_audit.md`); the three distance scorers with exact formulas and the pre-registration discipline stated plainly (k=10 fixed before results, k=1/50 as a declared robustness grid); domain-probe methodology (3 probes, 5-fold CV, no tuning, stated explicitly); statistical testing (exact-permutation Kendall's τ/Jonckheere-Terpstra, and *why* — this is a real, citable methodological strength, not a footnote); reproducibility statement.
- **Results**: strict claim ordering, see §6. No causal or mechanistic language anywhere in this section.
- **Discussion**: the decodability≠accessibility interpretation; explicit "this is an audit, not a new estimator" restatement; PAD-UFES scoping sentence (verbatim, below); architecture/dimensionality confound disclosure for the `baseline_soft` reference; Limitations (E3/E4 not run, Fisher-entanglement check not completed, single architecture, uneven seed counts, "reliability estimation" operationalized as OOD-AUROC specifically); Future Work (matched non-adversarial same-dimensionality control — named as the single highest-value next experiment; HAM10000 replication contingent on a verified dedup); brief broader-relevance paragraph.
- **Conclusion**: 3–4 sentences, no new claims.

---

## 6. Claim placement — Results vs. Discussion-only vs. never

**Results (facts, minimal interpretation)**
- Condition number moves ~100× with λ_orth (τ=0.84, p=2.8×10⁻⁵); other four geometry metrics show no significant trend.
- No geometry metric correlates with Mahalanobis AUROC (all |τ|≤0.28, p>0.19).
- Mahalanobis / cosine / k-NN(1,10,50) AUROC all ≈0.40–0.42, consistently below chance.
- Domain probe AUROC ≈0.73–0.81 (three probes, three rungs).
- NV-attractor gap is only 9–11pp above an already-high ID baseline; feature-norm gap is small and mixed at the mean — stated as "not consistent with X being the dominant explanation," never as "X is ruled out."
- `baseline_soft`'s numbers reported plainly, once, with no comparative language beyond the numbers themselves.

**Discussion-only (interpretation, explicitly labeled)**
- "Decodability ≠ distance accessibility" as the general dissociation.
- The audit framing and what it does/doesn't claim.
- Any reading of the `baseline_soft` contrast — must carry the confound disclosure in the same sentence, every time it's mentioned.
- **PAD-UFES scoping sentence, verbatim**: *"Since PAD-UFES also serves as the adversarial domain during training, our conclusions should be interpreted as characterizing representations learned under this training regime rather than all domain shifts."*
- Mechanistic speculation (why below chance, why decodable-but-inaccessible) — labeled as open hypothesis, not established.
- Broader-relevance claims beyond skin lesions.

**Never** (no data supports these; do not write them, in Results or Discussion)
- "Information moved to another direction/subspace/manifold" — no direction/angle/subspace analysis exists.
- Any causal attribution of the baseline_soft gap to disentanglement specifically.
- Any claim of generalization beyond ISIC/PAD-UFES/CSG-Lite (E3/E4 were never run).
- "Strictly held-out" anywhere near HAM10000, unless E3 is actually run with a verified dedup.

---

## 7. Reviewer-attack map (attack → where it's answered → the answer)

| # | Attack | Answered in | Answer |
|---|---|---|---|
| 1 | PAD-UFES is the adversarial training target — is this a general phenomenon or a near-tautology? | Discussion, Limitations | Verbatim scoping sentence, §6 above. No new experiment. |
| 2 | Baseline-vs-CSG comparison is confounded (architecture + dimensionality + recipe + n=1) | Results (brief) + Discussion | Reported once, plainly, always with the confound named in the same breath; never used as the paper's evidentiary anchor. |
| 3 | HAM10000 never run; source repos document heavy ISIC/HAM10000 overlap | Future Work | Named explicitly as the generalization experiment not yet done; "strictly held-out" struck everywhere until it's actually verified. |
| 4 | Small, uneven n (13 points, `runB` n=3) | Methods | Stated upfront; exact-permutation testing presented as the direct methodological response, not hidden. |
| 5 | Multiple comparisons, uncorrected across ~15+ tests | **Open — needs your call**, see §8 | Either add a correction pass, or state explicitly in Limitations that these are exploratory/descriptive, not a family-wise test. |
| 6 | Fisher-ratio/condition-number entanglement bootstrap was pre-registered and never run | Limitations | Disclosed directly; noted that condition number (not Fisher ratio) drives the reported trend, which bounds but doesn't eliminate the concern. |
| 7 | Checkpoint best-val provenance never cross-checked against training logs | Methods/Limitations | One-sentence disclosure. |
| 8 | AUROC is *below* chance, not merely at chance — why? | Discussion | Named as an open mechanistic question; NV-attractor/norm results offered as partial, non-dominant contributors, not a full account. |
| 9 | "Reliability estimation" is never operationally defined | Introduction | Defined explicitly, early, as OOD-detection AUROC specifically — the broader term is scoped down on purpose, on the page, not left ambiguous. |
| 10 | "This reads like a Mahalanobis-limitations paper" | Introduction (framing itself) | The entire opening-paragraph design in §5 exists to prevent this reading before it starts. |

---

## 8. Open items — resolved

1. **Multiple-comparisons correction: no correction.** Rationale: this is an exploratory/hypothesis-generating audit, not a confirmatory paper trading on isolated threshold crossings; the one load-bearing result (condition_number, p=2.8×10⁻⁵) survives any reasonable correction, and every non-significant result is already reported as non-significant. Disclosure sentence, verbatim, goes in Methods: *"Because this work is exploratory and hypothesis-generating rather than confirmatory, we report exact p-values without family-wise correction. Conclusions rely on effect consistency across independent analyses rather than isolated threshold crossings."*
2. **E1b (baseline_soft geometry): not run.** `baseline_soft` is descriptive-only per this blueprint's own design (§3, §6) — its geometry numbers would never be load-bearing for any claim, so computing them adds a branch with no claim depending on it. If a reviewer asks, the answer is "outside the scope of the categorical reference's role in this paper," not a gap to backfill.
3. **Figure 5: dumbbell plot, not grouped bars.** One dot for the distance-scorer mean, one dot for the domain-probe mean, per rung, connected by a line — the paper's claim is the *gap*, not the two absolute numbers, and a dumbbell makes the gap the visually dominant feature (bar-chart height comparison makes the reader do that subtraction manually). Locked.

## 9. The one-line slogan (locked wording, do not paraphrase)

> **Information can remain decodable while becoming inaccessible to distance-based reliability estimators.**

Appears verbatim, unchanged, in exactly three places: last sentence of the Abstract, last sentence of the Introduction, last sentence of the Discussion. This is the sentence a reader should be able to quote back after skimming only those three spots.

**Amendment (decided)**: the Discussion's closing instance intentionally reads "The same information can remain decodable while becoming inaccessible to distance-based reliability estimators" — one word added on purpose, as a callback to Introduction's plain "Information can remain decodable...". This is a deliberate stylistic variant, not an inconsistency to reconcile. Abstract, when written, should use the plain Introduction wording (the first occurrence in reading order), leaving Discussion's "The same information..." as the sole, intentional callback at the close.

## 10. Next step: competing Introductions, not a full draft

Per the working decision: Claude does not write the full manuscript end-to-end. Phase 1 is Introduction only, and — because a wrong opening framing corrupts everything downstream — three competing full Introduction drafts are written before any one of them is chosen:

- **Version A**: Clinical motivation → reliability estimation → audit.
- **Version B**: Representation geometry → the distance assumption → audit.
- **Version C**: Shortcut mitigation → disentanglement → unexpected consequence → audit.

Drafts are in `docs/introduction_drafts.md`. Methods/Results/Discussion do not start until one version (or a deliberate hybrid) is chosen.

## 11. Status: Introduction, Methods, Results, Discussion all LOCKED

Introduction: Version B, locked (`introduction_drafts.md`). Methods: locked (`methods_draft.md`). Results: locked (`results_draft.md`), scored 9.6/10, with the working note that §3.2–3.5 are one evidentiary chain for a single claim (distance-based accessibility did not change), not four separate results. Discussion: locked (`discussion_draft.md`), 4 rounds of revision, scored 9.2/10 — built as a hypothesis-elimination argument (rule out "Mahalanobis is a poor instrument," rule out "the information is gone," land on decodability-≠-accessibility), preserving the §3.2–3.5 evidentiary-chain shape rather than re-narrating four separate findings. **The full body is now closed. No further edits to Introduction/Methods/Results/Discussion prose unless a reviewer requests one.**

**Next phase — packaging, not composition** (per explicit sequencing): Title finalization → Abstract (must compress exactly two results: geometry changed enormously; distance-based accessibility did not, robustly, across four independent tests) → figure captions → BSPC cover letter → BSPC author-guideline checklist → one final full read-through from a reviewer's perspective. No new experiments unless a reviewer asks.

**No new experiments from this point on, unless a reviewer requests one.** The narrative is complete: geometry changes substantially (E1); distance-based reliability does not track it (E2); the raw distance gap is real but modest, not explained by norm collapse or a dominant attractor class (E2.5); the failure is shared by three structurally distinct scorer families, so it is not a Mahalanobis-specific defect (E2.6); the information those scorers would need is nonetheless recoverable by a non-distance-based reader (E2.7). Effort from here goes to Discussion, figures, and submission polish, not further analysis.

## 12. Discussion structure — LOCKED (exactly four subsections, in this order)

- **4.1 Finding.** One sentence. Answers "what did we find," nothing else — no supporting argument, no citation, no elaboration.
- **4.2 Interpretation.** The decodability-≠-distance-accessibility argument, built by walking §3.2–3.5 as one evidentiary chain (per §11 above), not as four separate results.
- **4.3 Relation to literature.** Every citation this paper carries lives here: Mahalanobis/OOD detection, representation geometry, probing classifiers.
- **4.4 Limitations.** Written directly, not hedged around: PAD-UFES-as-adversarial-target, the baseline confound, $n=13$, no E3, no matched non-adversarial control, uncorrected multiple comparisons, unverified checkpoint provenance, the undone Fisher-ratio/condition-number entanglement check.

The locked slogan (§9) is the final sentence of §4.4, and therefore of the Discussion as a whole.
