# Methods — LOCKED

Polished as a BSPC associate-editor pass on the prior draft: no new content beyond the explicitly requested `2.0 Overall study design` subsection, no logic or result changes, all technical detail preserved. Changes from the previous round: removed meta-commentary explaining *why* information is placed where it is (Methods should state facts, not narrate its own structure); collapsed repeated statements about `baseline_soft`'s role to one sentence; added a pre-specification statement to §2.3; tightened the multiple-comparisons sentence; tightened §2.5's probing-caveat sentence; added `2.0` as the missing pipeline overview. "Identical embeddings" is repeated deliberately across subsections and was not trimmed — the paper's entire evidentiary structure rests on same-embedding, different-scorer comparisons, and the repetition is load-bearing, not redundant.

Placeholders in `[brackets]` mark citations or values needing a final source check before submission (dataset citation keys, backbone architecture name), not open decisions.

---

## 2.0 Overall study design

For each of 13 checkpoints (§2.2), we extract $z_{lesion}$ embeddings for the ISIC training set and for the ISIC-test/PAD-UFES evaluation split. From these embeddings we compute three geometry metrics on the fitted Mahalanobis parameters (§2.3), score the evaluation split with three distance-based reliability estimators built on the identical embeddings (§2.4), and train three supervised probes on the same embeddings to quantify domain-information decodability independent of any distance-based scorer (§2.5). Association between disentanglement strength, geometry, reliability estimation, and decodability is then tested across the ladder using exact-permutation statistics (§2.6). Figure 1 summarizes this pipeline.

## 2.1 Datasets and domains

We use two publicly available skin lesion datasets: ISIC 2018 [cite] as the source domain and PAD-UFES-20 [cite] as the target domain, following the split and preprocessing pipeline of the underlying disentanglement classifier this study audits. ISIC data is partitioned into training, validation, and test subsets under a fixed, seeded split. PAD-UFES is withheld from label supervision during training. PAD-UFES serves as the adversarial target domain during training and therefore is not independent of the optimization process: the domain-adversarial branch of the audited architecture explicitly optimizes the representation to reduce discriminability between ISIC and PAD-UFES.

## 2.2 The disentanglement dose-response ladder

We use three checkpoint families from a domain-adversarial disentanglement architecture [architecture name], each producing a 16-dimensional lesion representation ($z_{lesion}$). The three families share identical architecture, training recipe, and representation dimensionality, differing only in the strength of an orthogonality regularization penalty ($\lambda_{orth}$) applied between the lesion and context branches:

| Family | $\lambda_{orth}$ | Mechanism | Seeds available |
|---|---|---|---|
| `runA_grl` | 0 | domain-adversarial component only | 42, 52, 62, 72, 82 (5) |
| `runB_orth1` | 1 | + orthogonality term | 42, 52, 62, 72, 82 (5) |
| `runB` | 5 | + stronger orthogonality | 42, 52, 62 (3) |

13 (family, seed) checkpoints total. All checkpoints are referenced by explicit, individually verified file path; none is resolved by automated directory search, to avoid a documented failure mode in which the most-recently-modified file in a run directory is not the best-performing validation epoch.

We additionally report a single conventional (non-disentangled) ResNet-50 checkpoint, `baseline_soft`, differing from the ladder in architecture, representation dimensionality (2048-d vs. 16-d), and training recipe, as a descriptive reference rather than a fourth dose level.

## 2.3 Representation geometry metrics

The selected metrics were pre-specified to represent three complementary aspects of representation geometry — covariance conditioning, class separation, and multivariate normality — rather than to maximize empirical correlation with downstream performance.

- **Condition number**: the ratio of the largest to smallest eigenvalue of the regularized, shared within-class covariance matrix used to fit the Mahalanobis estimator (§2.4), measuring how numerically stable that covariance's inversion is.
- **Fisher ratio**: the Hotelling–Lawley trace, $\mathrm{tr}(\Sigma^{-1}S_B)$, computed from the same fitted precision matrix, measuring class separation relative to within-class spread. We additionally report a decoupled scalar companion, $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$, which involves no matrix inverse and is therefore not entangled with condition number's own estimation noise — the two are reported together because they are not statistically independent measurements when both derive from the same $\Sigma^{-1}$.
- **Mardia's multivariate kurtosis**: quantifies departure from the class-conditional Gaussian shape the Mahalanobis estimator assumes. The classical closed-form asymptotic null does not apply once class means and covariance are estimated in-sample from the same residuals being tested; its null distribution is calibrated by parametric bootstrap (200 resamples per checkpoint) rather than the textbook formula.

All three metrics are computed on the identical fitted Mahalanobis parameters used for scoring (§2.4), not from an independently re-estimated covariance, so that geometry and reliability estimation describe the same fitted object.

## 2.4 Distance-based reliability scorers

Each test embedding is scored by three structurally distinct distance-based rules, applied to identical embeddings and an identical ISIC-train / ISIC-test / PAD-UFES split, for every one of the 13 checkpoints:

- **Mahalanobis distance**: the minimum, over the 8 ISIC classes, of the squared Mahalanobis distance from the test embedding to each class mean, under a shared covariance fit on the ISIC training set with regularization $\varepsilon=10^{-5}$.
- **Cosine-to-centroid**: the minimum, over the same 8 class means, of $1-\cos(z,\mu_c)$ — identical class-centroid structure to the Mahalanobis scorer, with the covariance/precision step removed, isolating that one variable.
- **Pooled $k$-nearest-neighbor distance**: the Euclidean distance from a test embedding to its $k$-th nearest neighbor in the full, class-pooled ISIC training set — no class structure, no covariance, the most assumption-free of the three scorers [Sun et al., 2022]. $k=10$ was fixed as the primary/headline value before any $k$-NN result was computed; $k=1$ and $k=50$ are reported as a pre-registered robustness grid, not selected after the fact.

All three scorers share one score convention (higher score, more out-of-distribution) and are evaluated by AUROC and FPR-at-95%-TPR on the same ISIC-test-vs-PAD-UFES split.

## 2.5 Domain-information probe

To test whether domain-discriminative information is present in an embedding independent of whether any distance-based scorer can access it, we train three supervised classifiers — logistic regression, a linear support vector machine, and a random forest, each at library-default hyperparameters with no tuning — to predict domain membership (ISIC-test vs. PAD-UFES) directly from the same $z_{lesion}$ embeddings scored in §2.4. None of the three probes has access to the original training objective; each is a fresh classifier fit only for this analysis. Probe AUROC is reported as evidence of what is linearly and nonlinearly recoverable from the representation, not as a claim about the training objective's internal computation [Hewitt and Liang, 2019].

We report 5-fold stratified cross-validated AUROC on out-of-fold predictions, avoiding the same-data-fit-and-evaluate leakage that would otherwise inflate apparent decodability at this sample size and dimensionality.

## 2.6 Statistical testing

The disentanglement ladder has only three ordinal levels with unequal per-level seed counts (5, 5, 3). We test association using Kendall's $\tau$ and, for ordered-group trend testing, the Jonckheere–Terpstra statistic, both evaluated against a full-enumeration exact permutation null — every distinct label arrangement consistent with the true per-rung seed counts — rather than the standard asymptotic approximation, which is not assumed valid at this sample size without checking.

Given the limited number of independent checkpoints ($n=13$), we emphasize consistency across independent analyses rather than statistical significance from any single test; unadjusted exact $p$-values are reported without family-wise correction.

## 2.7 Reproducibility

Every checkpoint, seed, and preprocessing parameter used in this study is recorded by explicit value or file path; no result in this paper depends on a directory-resolved or otherwise ambiguously-selected checkpoint. [Code/data availability statement — placeholder.]

---

## Open items before Results

1. **Architecture name / backbone**: `[architecture name]` in §2.2 needs the actual public-facing name for the CSG-Lite backbone (EfficientNet-B3-based, per this project's internal docs) — confirm how much architectural detail belongs in the main text vs. a Methods appendix.
2. **Dataset citation keys**: ISIC 2018 and PAD-UFES-20 citations are placeholders, not yet filled with verified BibTeX keys.
3. **Code/data availability statement**: needs the actual repository URL/DOI once one exists publicly — not fabricated here.

None of these block moving to Results — they're reference-list/appendix-level detail, not framing decisions.
