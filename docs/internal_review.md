# Internal Review — Reviewer 2

**Recommendation: Reject.** This is a design-stage review — no experiment has been run yet — but the design as documented already contains at least one factual self-contradiction serious enough to invalidate its central mitigation strategy, one uncontrolled confound serious enough to undermine the primary independent variable, and one implemented metric whose own author-run smoke test produces a false positive on the exact null case it exists to detect. None of these are typos. They are findings the authors already had the material to catch and did not.

This review does not propose fixes. Per instructions, the job here is to find grounds for rejection, not to repair them.

---

## README.md

**Unsupported assumptions**
- Calls this "an independent research project" that is "not a continuation... of Paper 1 or Paper 2," while simultaneously depending on Paper 1's exact checkpoints, exact training code, and exact Mahalanobis-fitting code as immutable, unmodifiable substrate (per its own "Ground rules"). A project whose entire independent variable (the dose-response ladder) and whose entire measurement instrument are both inherited wholesale from two other papers is not independent in any sense a reviewer would recognize — it is a re-analysis. Calling it "independent" is doing rhetorical work the actual dependency structure doesn't support.

**Ambiguous definitions**
- "Reliability estimation" is the paper's central construct and is never operationally defined here. Is it calibration (ECE)? OOD-detection AUROC? Selective-prediction risk-coverage? The document conflates "OOD detection" (literally what DST-Skin's `OODScorer` computes) with "reliability estimation" (a broader construct) as though they were interchangeable. A reviewer cannot evaluate whether the experiments answer the research question if the dependent variable's construct is never pinned down in the paper's own front door.

**What a reviewer would attack**
- Line 18 states Paper 2's "Ledoit-Wolf shrinkage covariance in normalized feature space" is "used as the dependent-variable measurement instrument" for the whole project. This is flatly contradicted by this project's own `threats_to_validity.md` (§2) and `geometry_metric_audit.md`, both of which commit E1–E3 — the core of the paper — to CSG-SKin's own raw pooled-covariance, per-class Mahalanobis via `eval_ood_benchmarks.py`, explicitly *not* DST-Skin's `OODScorer`. A reviewer who reads both documents (which they will, since both are submitted) finds the project's own README misstating what its own measurement instrument is. This is not a nuance; it's a wrong sentence about the central method, sitting in the introduction-equivalent document.

**Unjustified scientific leaps**
- "This is not a methods paper, not an OOD-detection paper, and not a Mahalanobis paper" is a triple negative with no corresponding positive contribution statement. Stating what a paper is not is not a substitute for stating, in one sentence, what novel scientific contribution it makes and to which body of literature it adds. As written, a reviewer cannot tell whether this is intended as an empirical study, a critical/negative-result paper, a methodological-caution paper, or a case study — each of which has different standards of evidence at a venue like MICCAI, and none of which is claimed.

**Engineering decisions that increase technical debt**
- N/A for this file (no code).

**Unnecessary parts**
- The "Layout" tree (lines 27–37) duplicates information already fully specified by the actual directory structure and by the more detailed layout description implicit in `SPEC.md`/`REPOSITORY_MAP.md`. Low cost, but pure restatement.

**What should be deleted**
- The claim of independence from Papers 1/2 (line 5) should either be substantiated (it currently isn't) or removed — as written it is an assertion the rest of the project's own documents disprove.

---

## SPEC.md

**Unsupported assumptions**
- The "dose-response ladder" (§4, E1's table) presents `λ_orth` as the *only* variable distinguishing the four rungs. It is not. Per this project's own `REPOSITORY_MAP.md` (§2.8), `baseline_soft` is trained at a different learning rate (2e-4 vs. 1e-4 for all three CSG runs) and under a materially different augmentation pipeline (`--no_robust_transforms`, i.e. the "light" transform path, vs. the "robust" path used by every CSG run). The ladder's rung 0 is not a controlled zero-dose condition — it is a different training recipe entirely, confounding "shortcut-mitigation strength" with learning rate and augmentation strategy. Any geometry or reliability difference observed between rung 0 and rungs 1–3 cannot be attributed to shortcut-mitigation alone from this design. This is the single most damaging unaddressed confound in the entire specification, and the information needed to catch it was already sitting in this project's own audit document.

**Ambiguous definitions**
- H3 (§3) is stated entirely in hedged modal language: representation improvements "*may* alter the assumptions," and therefore "*may* fail to translate." A hypothesis phrased with two independent "may"s is compatible with almost any experimental outcome — if geometry changes and reliability improves anyway, H3 is not thereby falsified (H3 never claimed alteration was sufficient to cause failure, only that it "may"). No falsification criterion is stated anywhere in this document. A hypothesis a reviewer cannot imagine being disproven by a stated pattern of results is not yet a testable hypothesis.

**What a reviewer would attack**
- E3 (§4) calls HAM10000 "a strictly held-out third domain" and claims validating on it checks that the E1↔E2 relationship "isn't an artifact of the ISIC/PAD-UFES split." This project's own `REPOSITORY_MAP.md` (§3.4) states outright: "since ISIC2018 and HAM10000 overlap heavily" — because ISIC 2018 Task 3 was itself constructed substantially from the HAM10000 collection. HAM10000 is not an independent domain from ISIC in any meaningful sense; it is largely the same underlying image collection. E3's entire justification for existing — proving the core result isn't an ISIC-specific artifact — is undermined by using a "third domain" that is not third. No deduplication procedure is specified anywhere (DST-Skin's own dedup script, `split.py`, is orphaned and its own output CSV does not exist on disk per this project's own audit), so even the minimal fix (exclude overlapping image IDs) is not currently implemented or verified. As written, E3 risks re-testing on training data while believing it is testing generalization.
- E2 (§4) proposes correlating/regressing geometry metrics against Mahalanobis AUROC "across the same four-rung ladder" with, per this project's own §6, 3–5 seeds per rung — a maximum of ~18 data points across 4 discrete treatment levels, or 12 if restricted to the common-seed subset per `threats_to_validity.md` §4. This is a 4-group comparison dressed as a regression. No power analysis is presented anywhere to indicate this sample size can distinguish a real association from noise, and with this few points any correlation coefficient will carry a confidence interval wide enough to be compatible with both "strong effect" and "no effect."

**Unjustified scientific leaps**
- E4 is explicitly deprioritized ("optional... scoped last, and only if time remains," §4). This means the one experiment capable of showing the E1↔E2 relationship generalizes beyond one specific architecture (CSG-Lite), one specific training recipe, and one contaminated dataset pairing is the one most likely to be cut under the project's own stated 6-month solo-author time pressure. In the scenario where E4 is cut, the paper's actual deliverable is an N≤5-seed, single-architecture, single-(confounded)-dataset-pair association — not evidence for the general research question posed in the title ("when do representation improvements translate into better reliability estimation," stated with no architecture or dataset qualifier).

**Engineering decisions that increase technical debt**
- N/A (no code in this file).

**Unnecessary parts**
- §6's risk list and §7's open-decisions list substantially duplicate content already present in `REPOSITORY_MAP.md` and (for the geometry-metric item) resolved in `geometry_metric_audit.md`. Maintaining the same facts in three places is a documentation-consistency liability, not a neutral redundancy — it is exactly how the README's contradiction (above) and the HAM10000 contradiction (above) were able to arise undetected: nothing forces the three documents to agree when one is updated.

**What should be deleted**
- The word "strictly" before "held-out" (E3, §4) should not survive contact with this project's own repository audit.

---

## REPOSITORY_MAP.md

**Unsupported assumptions**
- Treats the underlying codebases' documented instability as background context to be worked around rather than as a threat to the paper's validity in its own right. The document itself catalogs: a checkpoint-resolution bug that silently selects non-best checkpoints for 6 seed directories including the paper's own primary rungs; a domain-adversarial "OOD" test set that is simultaneously the model's own training data; six-plus divergent, subtly-inconsistent feature-extraction implementations; a Mahalanobis fitting bug in the training script itself. The document's own framing treats this as due diligence rather than what it also is: direct evidence that Papers 1 and 2's infrastructure was not built to the standard of rigor a causal claim about representation geometry and reliability estimation would need, and that Paper 3 is contractually forbidden (per its own ground rules) from fixing any of it.

**What a reviewer would attack**
- The scale of this document (400+ lines cataloging bugs, stale artifacts, and unverifiable claims in the source repos) is itself an argument against building new causal claims on top of this substrate rather than a neutral inventory. A reviewer sympathetic to reproducibility standards would ask why a third paper's central hypothesis test is being run on checkpoints from a codebase whose own audit needed this many pages to explain which files are safe to trust.

**Unjustified scientific leaps**
- None specific to this document — it is descriptive, not a claims-making document. Its risk is one of omission at the point of use: several facts recorded here accurately (the transform/lr confound across CSG method presets; the ISIC/HAM10000 overlap) are not acted on in `SPEC.md` or `threats_to_validity.md`, meaning the map that would have caught two of this review's most serious findings already existed and wasn't consulted at the point those findings mattered.

**Engineering decisions that increase technical debt**
- N/A (descriptive document, no code of its own).

**Unnecessary parts**
- Given this document's actual function turned out to be "the thing later documents should have cross-checked against and didn't," a shorter, load-bearing subset (checkpoint pitfalls, split contamination, Mahalanobis divergences) would have served the project better than the full architectural tour, which is comprehensive but not all equally decision-relevant to Paper 3 specifically.

**What should be deleted**
- Nothing factually — but its existence as a *separate* document from `SPEC.md`/`threats_to_validity.md`, with no enforced cross-reference check, is the structural reason its most important findings (transform confound, HAM10000 overlap) failed to propagate into the documents that needed them.

---

## CLAUDE.md

**What a reviewer would attack**
- "If something in either source repo looks like a bug worth fixing, flag it to the user instead of fixing it." This hard-codes every one of the ~15–20 already-cataloged infrastructure risks (`REPOSITORY_MAP.md`) as a future manual escalation rather than a resolved precondition. For a solo-author, 6-month project, this is a standing commitment to interrupt experimental progress every time one of these known issues actually bites — which, given how many are cataloged as "confirmed to actually fire" (not hypothetical) in `REPOSITORY_MAP.md` §7.2, is not a matter of if.

**Engineering decisions that increase technical debt**
- The non-modification rule is enforced entirely by convention and reviewer/author discipline — there is no lint rule, pre-commit hook, CI check, or automated guard preventing a future edit to `CSG-SKin/` or `DST-Skin/`. For a rule load-bearing enough to be stated as a "hard constraint" in three separate documents (`README.md`, `SPEC.md`, `CLAUDE.md`), the absence of any technical enforcement mechanism is itself a technical-debt decision: the project is one distracted edit away from violating its own central methodological commitment, with nothing but documentation standing in the way.

**Unnecessary parts**
- Substantially restates ground rules already present in `README.md` and constraints already present in `SPEC.md` §5. Three copies of "never modify the source repos" is not obviously safer than one authoritative copy the other two reference — see the README/SPEC/threats-doc drift already documented above for what happens when the same fact lives in more than one place without an enforced sync.

---

## docs/geometry_metric_audit.md

**Unsupported assumptions**
- The entire document rests on a three-concern taxonomy (conditioning / class-separation / Gaussianity, §1) presented with the rhetorical weight of a formal derivation ("these three concerns are logically independent," "the formal anchor") but is, in fact, the authors' own conceptual scaffolding — it is not drawn from Lee et al. (2018) or any other cited source, and no citation is offered for this specific three-way decomposition. A differently-carved taxonomy (for instance, separating covariance *estimation error* from covariance *model misspecification*, currently bundled together as one "concern") would license a different metric selection. Presenting a self-constructed framework as though it were the necessary formal structure of Mahalanobis distance overstates the rigor actually on offer.
- §5's justification for stopping at exactly 3 metrics — "no concern from §1 may be left completely untested... adding a fourth or fifth metric... buys no new coverage" — is circular once the taxonomy itself is in question: the argument that 3 is sufficient depends entirely on 3 concerns being the right and complete decomposition, which is asserted, not established.

**Ambiguous definitions**
- The Family-A/B/C "Verdict" labels (Primary/Secondary/Rejected, §3–§5) are applied with an appearance of mechanical consistency, but the underlying judgments are graded and contestable (e.g. "redundant" is doing heavy lifting across at least six separate rejections without a shared, quantitative redundancy threshold — how correlated do two metrics need to be, empirically, before "redundant" is a fair label, versus merely "related"?). No such threshold is ever specified or tested; it is asserted per metric, in prose, by the same author who chose which three to keep.

**What a reviewer would attack**
- Mardia's multivariate kurtosis is explicitly described (§5, §3 C2) as "adapted... from its classical single-Gaussian form" to a per-class-conditional, in-sample-covariance setting. The cited asymptotic null, $b_{2,d}\sim\mathcal N(d(d+2), 8d(d+2)/n)$, is Mardia's (1970) result for the classical, single-population, out-of-sample setting. Nowhere in this document is it shown, derived, or even argued that the same asymptotic null holds once the mean is taken per-class and the covariance is estimated in-sample from the very residuals being tested. Borrowing a well-known formula from a textbook citation and applying it to a modified setting without justifying that the modification preserves the formula's validity is precisely the move that invites a statistically literate reviewer to reject the instrument outright — and this project's own subsequent implementation (`geometry_diagnostics.py`) demonstrates the concern is not academic: see below.
- The literature survey (§3) omits Neural Collapse (Papyan, Han & Donoho, PNAS 2020, and the substantial line of work it spawned), which is the current, widely-cited framework for exactly the within/between-class covariance-geometry question this document's Family B is built around, and would be the first thing a reviewer familiar with representation-geometry literature looks for. It also omits local intrinsic dimension estimators (Ma et al., 2018, ICLR, "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality"), which are specifically used in the OOD/adversarial-robustness literature this document otherwise cites (Ren et al. 2021, Lee et al. 2018) as a per-point geometric diagnostic — arguably more relevant to an OOD-adjacent question than several of the 1979–1987-vintage generic clustering-validity indices that *are* included (Davies-Bouldin, silhouette). A candidate list built by auditing an externally-supplied list of 13 names, rather than by an independent literature search, inherits whatever gaps were in that original list — and it shows.
- §5 justifies keeping only 3 metrics partly on multiple-comparisons grounds ("a multiple-comparisons liability, not added rigor") but the document never specifies a multiple-comparisons correction (Bonferroni, FDR, a single composite pre-registered test) for the 3 comparisons that remain. Raising multiple comparisons as a reason to cut from 13 to 3, then not addressing the exact same statistical issue for the surviving 3, is inconsistent.

**Unjustified scientific leaps**
- The Fisher ratio's justification (§5) claims it is "computed as $\mathrm{tr}(\text{precision}\cdot S_B)$ directly on the same `precision`/`class_means` the estimator already fit," framed as a virtue ("one estimation pass, not two divergent ones"). This framing obscures a real cost: because Fisher ratio and condition number are now both computed from the *same* `precision` matrix, they are not independent measurements of two logically distinct concerns, as the taxonomy in §1 claims. If `precision`'s conditioning is corrupted (e.g., by `reg_eps` dominating over genuinely small eigenvalues at high feature dimensionality — the exact failure mode Family A's own writeup spends several paragraphs worrying about), that same corruption flows directly into the Fisher-ratio number, since Fisher ratio is literally a linear functional of `precision`. The document presents "one estimation call, three read-outs" as efficiency; a skeptical reviewer reads it as three entangled numbers being sold as three independent tests of independent hypotheses.

**Engineering decisions that increase technical debt**
- N/A directly (this document precedes code), but see the entangled-precision-matrix point above, which became actual, load-bearing code in `geometry_diagnostics.py` without the entanglement being flagged there either.

**Unnecessary parts**
- The document runs 276 lines to audit 13 candidates and conclude "use the 3 things directly derivable from one existing function call." The 10 rejected candidates each receive a full nine-field write-up (definition, property, formula-relevance, complexity, robustness, prior use, strengths, weaknesses, verdict) before being rejected — for candidates like spectral entropy (§3, A7), which is rejected for being "A5 in different units," the full apparatus is disproportionate to the two-sentence reason it doesn't survive.

**What should be deleted**
- The claim that the three concerns are "logically independent" (§1) should be removed or defended, since §5's own chosen implementation (Fisher ratio sharing `precision` with condition number) directly undercuts measurement-level independence even where the concepts might be independent in principle.

---

## docs/threats_to_validity.md

**Unsupported assumptions**
- Threat #1's mitigation claims E3 (HAM10000) validates against PAD-UFES contamination using "a strictly held-out third domain." As established above, this project's own `REPOSITORY_MAP.md` contradicts the word "held-out." This document's single stated mitigation for its own first-listed, most consequential threat does not hold up against another document in the same project.

**Ambiguous definitions**
- No entry in this document addresses *construct* validity — whether "Mahalanobis AUROC computed on the ISIC/PAD-UFES split" actually measures "reliability estimation quality" as a general construct, as opposed to "domain-shift-detection accuracy specific to one acquisition-modality shift." All eight entries are internal-validity threats (contamination, confounds, implementation mismatches, sample size). The absence of a construct-validity entry is a gap in kind, not degree — it means the document never asks whether the thing being measured is the thing the research question is actually about.

**What a reviewer would attack**
- Threat #4's mitigation — restrict to the 3 seeds common across all rungs — directly worsens the small-sample problem raised independently in `SPEC.md` §7 and Threat #6, by discarding the 2 extra seeds available for 3 of the 4 rungs. No power calculation is offered showing this trade (symmetry vs. more data) is the right one; the document picks a side of a real statistical trade-off without quantifying it.
- Threat #2's mitigation ("use `eval_ood_benchmarks.py` only") establishes only that this script avoids one specific, previously-documented bug (the grayscale double-conversion in `train_csg.py`). It does not establish that `eval_ood_benchmarks.py` is itself correct — no independent verification, reference implementation comparison, or unit test is proposed anywhere in this project for the script this entire paper's dependent variable will come from.
- Threat #8 concludes DST-Skin's unseeded checkpoints mean E4 "cannot support a statistical claim" beyond a single-run illustration, yet `SPEC.md` still lists E4 as an experiment in the numbered plan (E1–E4) rather than downgrading it to explicitly illustrative status at the point this document's own analysis concludes it must be. The threat is correctly identified and then not acted on in the document that defines the experiment.

**Unjustified scientific leaps**
- Threat #7's mitigation — "the full audit... is written down... before any E1/E2 result is computed, precisely to prevent post-hoc metric selection" — is undercut by its own admission one sentence later: "this is a discipline commitment, not a technical guarantee." A mitigation that concedes it has no enforcement mechanism is a stated intention, not a threat mitigation in the sense the rest of the document's format implies.

**Engineering decisions that increase technical debt**
- N/A (no code in this document).

**Unnecessary parts / what should be deleted**
- "Strictly" (Threat #1's title line) — see README.md and SPEC.md entries above; the same word fails in three places for the same reason.

---

## scripts/geometry_diagnostics.py

**What a reviewer would attack — the most serious finding in this review**
- This module's own `__main__` smoke test generates synthetic, exactly-Gaussian, isotropic, per-class-separated data (8 classes, 16 dimensions, 200 samples/class) — data that is Gaussian by construction, with no possible source of real non-normality. Running it (done during this review) produces `mardia_kurtosis_z = -3.13`. Under a correctly calibrated test, a z-score this extreme on genuinely Gaussian data should be rare (roughly a 1-in-500 event under the classical asymptotic null this statistic borrows). Getting it on the very first, easiest, most favorable case the author chose to demonstrate the module is a direct empirical demonstration that the in-sample-bias caveat documented in the function's own docstring is not a minor footnote — it is large enough to produce an apparent rejection of Gaussianity on data that is Gaussian by construction. No calibration study, no simulation-based null, and no correction is proposed anywhere in this project for this. A reviewer does not need to imagine this failure mode; it is reproducible by running the file exactly as delivered.

**Unsupported assumptions**
- `condition_number` silently discards non-positive eigenvalues (`eigvals = eigvals[eigvals > 0]`) with no warning, log, or error. Given `precision` is constructed as the inverse of a matrix regularized by `+ reg_eps·I`, it should be strictly positive-definite by construction — meaning this branch should never fire in correct operation. If it ever does fire (e.g., under floating-point breakdown in a genuinely near-singular case), that is precisely the pathological condition this metric exists to surface, and the code's response is to quietly drop the offending eigenvalues and report a number computed from what's left, rather than flag the anomaly. The one failure mode this function is supposed to detect is the one case where its own defensive code would hide the evidence.
- `fisher_ratio` assumes class labels are contiguous integers `0..K-1` matching `class_means`' row order, with no validation. A mismatched or non-contiguous label encoding would silently produce a wrong `S_B` (via wrong `counts`) rather than raising a clear error.

**Ambiguous definitions**
- The module computes $J = \mathrm{tr}(\text{precision} \cdot S_B)$ and documents it as "the Fisher ratio," but `precision` is `inv(S_W/(N-K) + \text{reg\_eps}\cdot I)`, not $S_W^{-1}$ — it carries an implicit scale factor of approximately $(N-K)$ relative to the textbook quantity defined in `geometry_metric_audit.md` §3 (B1). This is only harmless if $N$ (the Mahalanobis fitting-set size) is identical across every rung and seed being compared; that assumption is never stated, checked, or asserted anywhere in this module or its documentation.

**Engineering decisions that increase technical debt**
- Reaches into a sibling repository via a hand-rolled `sys.path.insert` keyed to a hardcoded relative path (`parents[2] / "CSG-SKin"`). No package boundary, no version pin against CSG-SKin's code, no test verifying the import still resolves the expected function signature. If `CSG-SKin/` is ever moved, renamed, or restructured — outside this project's control, since it's explicitly someone else's frozen artifact — this fails, possibly silently (e.g. resolving a same-named function from an unrelated location on `sys.path` in a different environment) rather than loudly.
- No unit tests. The only executable verification is a `print()`-based smoke test that requires a human to notice the z-score above is suspicious; nothing in the module would catch this automatically on a future change.
- `GeometryDiagnostics` records `n_samples`, `feat_dim`, `num_classes`, `reg_eps` — but not which checkpoint, rung, seed, or representation ($z_{lesion}$ vs. $z_{context}$) produced the row. Any caller assembling results across the 4-rung × up-to-5-seed ladder must independently track and correctly align this provenance externally, which is exactly the kind of bookkeeping step where a silent mislabeling (attributing one seed's geometry to another) becomes possible and would not be caught by anything in this module.
- `compute_mahalanobis_params_from_arrays` (imported, unmodified per policy) raises if any class has zero samples in the fit set. Nothing in this module catches, contextualizes, or gracefully degrades on that exception — a single rare/missing class in one rung/seed's fit batch hard-stops the entire pipeline run for that condition, with no partial-results handling.

**Unnecessary parts**
- The extensive module docstring re-derives and re-justifies the metric selection already fully argued in `geometry_metric_audit.md`, duplicating rationale across two documents that can drift out of sync exactly as the other document pairs in this project already have.

---

## Cross-document findings (the reasons this gets rejected, not revised)

1. **HAM10000 is called "strictly held-out" in two documents (`SPEC.md`, `threats_to_validity.md`) and directly contradicted by a third document in the same project (`REPOSITORY_MAP.md`)**, which states the two datasets overlap heavily. E3 — the experiment meant to rule out exactly the kind of dataset-specific artifact this contradiction represents — is compromised at the design stage, not the execution stage.
2. **The dose-response ladder confounds shortcut-mitigation strength with learning rate and augmentation pipeline** (documented in `REPOSITORY_MAP.md`, never surfaced as a risk in `SPEC.md` or `threats_to_validity.md`). This is the independent variable of the entire study.
3. **The README misstates the project's own dependent-variable instrument**, attributing it to DST-Skin's Ledoit-Wolf `OODScorer` when the actual committed instrument (per `threats_to_validity.md` and `geometry_metric_audit.md`) is CSG-SKin's own raw pooled-covariance implementation.
4. **The one implemented statistical test (Mardia's kurtosis) produces an apparent false rejection of Gaussianity on synthetic data that is Gaussian by construction**, in a smoke test the author ran and printed without flagging the result as disqualifying.
5. **Two of the three chosen "independent" geometry metrics (condition number, Fisher ratio) are computed from the same shared, `reg_eps`-regularized precision matrix**, meaning a corruption in one is not statistically separable from a corruption in the other — undercutting the taxonomy's central claim that the three concerns are measured independently.

None of these were latent or hard to find. Each was either already documented in another file in this same project, or reproducible by running the one script that exists. A reviewer's job is easier, not harder, when the submission's own supporting documents contradict each other and its own demo code contradicts its own docstring.
