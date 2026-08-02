# Geometry Metric Audit for E1

**Purpose of this document**: E1 (per `SPEC.md`) needs to "measure geometry changes" across the CSG-SKin dose-response ladder, but *which* geometric properties to measure was left as an open design decision (`SPEC.md` §7). This document is that decision, worked from first principles: for each candidate metric, it establishes what the metric formalizes, whether it is *theoretically load-bearing for Mahalanobis reliability estimation specifically* (not just "a geometry metric someone might compute"), and whether it should be measured in E1 at all.

**What this document is not**: it does not implement anything, does not choose a specific software library or estimator, and does not commit to an E2 association test. It only decides *what to measure* and *why*, so E1's implementation has a justified target instead of an arbitrary one.

**Method**: every candidate is evaluated against how it relates to the Mahalanobis distance formula and the assumptions required for that formula to behave as a reliability estimate (defined in §1) — not against how common or well-known the metric is in general.

---

## 1. The formal anchor: what does Mahalanobis reliability estimation actually assume?

Squared Mahalanobis distance from a point $x$ to a distribution with mean $\mu$ and covariance $\Sigma$ is:

$$d_M(x)^2 = (x - \mu)^\top \Sigma^{-1} (x - \mu)$$

This is only a meaningful *reliability* signal — "large $d_M$ ⇒ untrustworthy input" — to the extent that three things hold:

1. **$\Sigma$ is well-estimated and well-conditioned**, so $\Sigma^{-1}$ doesn't amplify estimation noise in low-variance directions into spuriously large distances. This is a property of the covariance's *eigenvalue spectrum* alone.
2. **The generative structure Mahalanobis assumes matches the data.** Lee et al. (2018, "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" — the paper both CSG-SKin's and DST-Skin's Mahalanobis implementations descend from) frame this as class-conditional Gaussians sharing one covariance ($x \mid y=c \sim \mathcal{N}(\mu_c, \Sigma)$), scored by $\min_c d_M(x, \mu_c)$. Under this model, Mahalanobis distance is only discriminative between "confidently ID" and "OOD-ish" if the class-conditional clusters are actually separated relative to their spread — a **between/within-class separation** property, distinct from (1).
3. **The class-conditional distribution is actually approximately Gaussian.** $d_M(x)^2$ is literally the quadratic form inside a multivariate Gaussian's exponent; if the true distribution is heavy-tailed, skewed, or multimodal, $d_M$ stops corresponding to a density/likelihood-ratio at all, even if (1) and (2) hold.

These three concerns are logically independent — a representation can be well-conditioned but non-Gaussian, Gaussian but with overlapping classes, or well-separated but ill-conditioned. This gives the taxonomy used below: candidates are grouped by which of these three concerns they operationalize, since that determines whether a candidate is *theoretically relevant to Mahalanobis* rather than just "a geometry number."

**A repo-specific fact this taxonomy immediately surfaces** (verified by reading `DST-Skin/src/utils/scoring.py::OODScorer.fit`): DST-Skin's Mahalanobis is fit as a **single global mean/covariance over all ID training features** — `train_labels` is accepted as a parameter but never used in `fit()`. It is a one-class, not-per-class-conditional formulation. CSG-SKin's `ood_metrics.py::compute_mahalanobis_params_from_arrays`, by contrast, is per-class (Lee-et-al.-style pooled within-class covariance, per-class means). **Concern (2) — between/within-class separation — is therefore mechanistically irrelevant to DST-Skin's specific implementation**, as currently written, even though it is central to CSG-SKin's. This matters directly for scoping: Family B results can motivate and explain outcomes on the E1–E3 path (CSG-SKin) and simply may not transfer, as a mechanism, to the E4 path (DST-Skin) unless DST's `OODScorer` is itself extended to be per-class — which is out of scope for E1.

---

## 2. Taxonomy

| Family | Concern from §1 | What it operationalizes |
|---|---|---|
| **A — Covariance conditioning** | (1) | Shape/scale of $\Sigma$'s eigenspectrum: is it invertible in a numerically stable, non-noise-amplifying way? |
| **B — Class-separation geometry** | (2) | Are class-conditional clusters separated relative to their spread? (Mechanistically load-bearing only for a per-class-conditional Mahalanobis formulation — see note above.) |
| **C — Distributional shape** | (3) | Does the feature distribution actually look Gaussian? |

Family A's seven candidates (condition number, trace, log-determinant, participation ratio, effective rank, anisotropy, spectral entropy) are all, mathematically, **functions of the same object** — the eigenvalue spectrum $\{\lambda_1,\dots,\lambda_d\}$ of $\Sigma$. They are audited together precisely because their redundancy with each other is the main thing that determines how many of them E1 actually needs to compute.

---

## 3. Per-metric audit

### Family A — Covariance conditioning

#### A1. Condition number

- **Definition**: $\kappa(\Sigma) = \lambda_{\max}/\lambda_{\min}$.
- **Property measured**: worst-case eigenvalue spread — how much the most- and least-informative directions differ in scale.
- **Appears directly in Mahalanobis formulation?** Yes, directly. $\Sigma^{-1}$ has eigenvalues $1/\lambda_i$; a large $\kappa$ means $\Sigma^{-1}$ is dominated by $1/\lambda_{\min}$, so $d_M$ is driven almost entirely by displacement along the single lowest-variance direction — exactly the numerical-instability failure mode Family A exists to detect.
- **Computational complexity**: $O(d^3)$ (eigendecomposition of $\Sigma$) or $O(nd^2)$ via SVD of centered features; for this project's dims ($d=16$ for $z_{lesion}$, $d=64$ for $z_{context}$, $d=1536$ for `backbone_raw`, $d=2048$ for the ResNet-50 baseline) this is cheap-to-trivial, computed once per (rung, seed).
- **Robustness to sample size**: **poor — the central practical risk for this project.** $\lambda_{\min}$ of a *sample* covariance is a high-variance, biased estimate of the true smallest eigenvalue, especially as $d/n$ stops being small (sample eigenvalues spread out even when the population spectrum is flat). CSG's own Mahalanobis code already needs a `reg_eps·I` regularizer for exactly this reason, and DST's uses Ledoit-Wolf shrinkage for the same reason. At $d=1536$–$2048$, condition number computed from a raw (unregularized) sample covariance will be dominated by estimation noise unless per-class $n$ is large — it must be computed on the *same* (regularized/shrunk) covariance the actual Mahalanobis estimator uses, not a separately-estimated raw one, or it won't describe the object Mahalanobis actually inverts.
- **Prior use in OOD/representation literature**: standard numerical-linear-algebra diagnostic generally; specifically implicated in critiques of when Mahalanobis-based OOD detection underperforms (e.g. Ren et al., 2021, "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection," traces failure modes back to covariance estimation/conditioning).
- **Strengths**: the most direct, literal encoding of "how well-behaved is $\Sigma^{-1}$"; cheap; immediately interpretable (large = risk).
- **Weaknesses**: a single ratio is extremely sensitive to sampling noise in just the two extreme eigenvalues, not a stable summary of the whole spectrum; scale-dependent unless features are normalized consistently (CSG and DST already differ here — DST L2-normalizes before fitting, CSG does not).
- **Verdict: Primary.** The most direct operationalization of Concern (1) and the cheapest to compute; the remaining Family-A candidates either measure a strict subset of the same information (trace) or a smoothed/regularized view of the same spectrum with different noise-sensitivity trade-offs (participation ratio, effective rank, spectral entropy, anisotropy).

#### A2. Trace(covariance)

- **Definition**: $\mathrm{tr}(\Sigma) = \sum_i \lambda_i$ (sum of per-dimension variances).
- **Property measured**: total variance / overall scale, independent of how it's distributed across directions.
- **Appears directly in Mahalanobis formulation?** Only indirectly — $\mathrm{tr}(\Sigma)$ does not appear in $\Sigma^{-1}$ or in $d_M^2$; it matters mainly as a normalizing reference for other spectrum-shape metrics (participation ratio and spectral entropy are both explicitly trace-normalized, see A4/A6).
- **Computational complexity**: $O(nd)$ — trivial, cheaper than any eigendecomposition (computable without ever forming or decomposing $\Sigma$).
- **Robustness to sample size**: good — trace sums per-dimension sample variances, each a consistent, low-variance estimator individually; unlike $\lambda_{\min}$ or $\kappa$, it doesn't depend on correctly resolving the smallest eigenvalue.
- **Prior use**: common summary/normalizing statistic in high-dimensional statistics and representation-geometry work, rarely a headline result on its own.
- **Strengths**: cheap, stable, requires no decomposition.
- **Weaknesses**: not diagnostic of conditioning at all — a covariance can have large trace and still be nearly singular (variance concentrated in one direction), or small trace and well-conditioned. Cannot detect Concern (1) on its own.
- **Verdict: Secondary.** Not diagnostic of Mahalanobis validity by itself, but essentially free and needed as a normalizer for A4/A6 anyway — report it, don't treat it as a headline result.

#### A3. Log-determinant

- **Definition**: $\log\det(\Sigma) = \sum_i \log \lambda_i$.
- **Property measured**: overall "volume" of the covariance ellipsoid in log-space; heavily penalizes any single near-zero eigenvalue (since $\log \lambda_i \to -\infty$ as $\lambda_i \to 0$).
- **Appears directly in Mahalanobis formulation?** It appears in the *Gaussian log-density* that $d_M^2$ is the quadratic term of ($\log p(x) = -\tfrac{1}{2}d_M(x)^2 - \tfrac{1}{2}\log\det\Sigma - \tfrac{d}{2}\log 2\pi$), but **not in $d_M^2$ itself**, and not in the raw-distance form of Mahalanobis that CSG's `ood_metrics.py` and DST's `OODScorer.score_mahalanobis` both actually use (both return a bare distance, not a log-density). This is a precise but important distinction: log-det matters for calibrating a *likelihood-based* threshold across differently-scaled distributions, but is invisible to a pipeline that only ever computes and thresholds raw $d_M$ — which is what both repos do.
- **Computational complexity**: $O(d^3)$ (same eigendecomposition as A1) — effectively free once $\kappa$ is already being computed.
- **Robustness to sample size**: **very poor, worse than $\kappa$.** A single near-zero sample eigenvalue (guaranteed whenever $n < d$, and likely whenever $n$ is only moderately larger than $d$) sends $\log\det$ toward $-\infty$; this is a well-known numerical pathology, which is exactly why every practical Mahalanobis implementation (including both in these repos) regularizes or shrinks $\Sigma$ before inverting.
- **Prior use**: appears throughout Gaussian-model literature (log-likelihood, KL-divergence between Gaussians, generalized variance) but is used far less often as a standalone geometry diagnostic than condition number or effective-rank-family metrics, precisely because of the instability above.
- **Strengths**: theoretically clean connection to the Gaussian log-likelihood; captures the same "near-singular" failure mode as $\kappa$ but as a sum rather than a ratio, so it doesn't ignore the rest of the spectrum entirely.
- **Weaknesses**: not part of the distance formula either repo actually computes; numerically worse-behaved under small-sample/high-dimension conditions than the alternative already selected (A1); redundant with A1 as a "detect near-singularity" signal once regularization is applied consistently.
- **Verdict: Rejected** (for E1's primary/secondary set) **as a standalone metric**, but **flag for E2/documentation**: if E2 later wants to reason about likelihood-based (rather than raw-distance) reliability scores, log-det becomes relevant again as the normalizing term — this should be revisited if E2's association test moves beyond raw AUROC on raw distances.

#### A4. Participation ratio

- **Definition**: $PR(\Sigma) = \dfrac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2} = \dfrac{\mathrm{tr}(\Sigma)^2}{\mathrm{tr}(\Sigma^2)}$. Ranges from 1 (all variance in one direction) to $d$ (variance spread perfectly evenly).
- **Property measured**: the *effective number of dimensions* the variance is actually spread across — a smoothed, quadratic-moment-based dimensionality count.
- **Appears directly in Mahalanobis formulation?** Indirectly — it is a monotonic-ish summary of the same eigenvalue spectrum $\Sigma^{-1}$ depends on, but doesn't appear as a term in $d_M^2$. Its relevance to Mahalanobis is exactly the same underlying concern as $\kappa$ (Concern 1), just expressed as an "effective dimensionality" rather than an extremal ratio.
- **Computational complexity**: $O(d^3)$ for exact eigenvalues, or $O(nd)$ if computed directly from $\mathrm{tr}(\Sigma)$ and $\mathrm{tr}(\Sigma^2) = \sum_{ij}\mathrm{Cov}_{ij}^2$ (Frobenius norm) without full decomposition — cheaper than A1/A3 in practice since it avoids the full eigendecomposition.
- **Robustness to sample size**: **better than $\kappa$ or $\log\det$**, precisely because it's a ratio of two *sums* (quadratic moments) rather than a function of the single extreme eigenvalue. It still inherits some sample-covariance bias (sample eigenvalue spreading inflates $\mathrm{tr}(\Sigma^2)$ relative to the truth, biasing $PR$ downward), but is far less catastrophically sensitive than $\log\det$ to one badly-estimated eigenvalue.
- **Prior use in OOD/representation-learning literature**: widely used in neural-population and deep-representation geometry work as an "effective dimensionality" measure — e.g. Gao & Ganguli-style neural population geometry analyses, and Ansuini et al. (2019, "Intrinsic Dimension of Data Representations in Deep Neural Networks") on the effective dimensionality of DNN layer representations across training/depth.
- **Strengths**: intuitive units ("effective number of dimensions," bounded in $[1,d]$); more sample-robust than $\kappa$/$\log\det$; cheap to compute without full decomposition.
- **Weaknesses**: a single scalar necessarily discards where in the spectrum the variance concentration happens (two very different spectra can share the same $PR$); doesn't distinguish "well-conditioned but low-dimensional" from "poorly-conditioned."
- **Verdict: Primary.** Complements A1 (condition number) rather than duplicating it: $\kappa$ answers "how bad is the single worst direction," $PR$ answers "how many directions carry the representation's variance overall" — together they give a conditioning picture that is materially more sample-robust than either alone, at negligible extra compute cost.

#### A5. Effective rank

- **Definition** (Roy & Vetterli, 2007, "The Effective Rank: A Measure of Effective Dimensionality"): normalize the eigenvalues to a probability-like distribution $p_i = \lambda_i / \sum_j \lambda_j$, then $\mathrm{erank}(\Sigma) = \exp\!\left(-\sum_i p_i \log p_i\right)$ — the exponential of the Shannon entropy of the normalized spectrum.
- **Property measured**: effective dimensionality, via an entropy rather than a quadratic-moment construction (contrast with A4).
- **Appears directly in Mahalanobis formulation?** Indirectly, same as A4 — a summary of the eigenvalue spectrum, not a term in $d_M^2$ itself.
- **Computational complexity**: $O(d^3)$ for eigenvalues (or $O(nd^2)$ via SVD), then $O(d)$ for the entropy — same order as A1/A4, no extra decomposition needed since the eigenvalues are typically already computed for A1.
- **Robustness to sample size**: similar profile to A4 — better than $\kappa$/$\log\det$ because it's an aggregate over the whole spectrum, but the entropy calculation is sensitive to the many small, noisy tail eigenvalues in a high-$d$/moderate-$n$ regime (each contributes a small but nonzero $-p_i\log p_i$ term, so estimation noise in the tail inflates the entropy, biasing $\mathrm{erank}$ upward relative to A4's $PR$, which weights the tail more gently via $\lambda_i^2$).
- **Prior use**: signal-processing origin (Roy & Vetterli); adopted into deep-learning representation analysis as an alternative "intrinsic/effective dimensionality" measure, and specifically used as a collapse diagnostic in self-supervised learning (e.g. Jing et al., 2022, "Understanding Dimensional Collapse in Contrastive Self-Supervised Learning," tracks the singular-value spectrum / effective rank of learned embeddings to detect representation collapse).
- **Strengths**: theoretically well-motivated (information-theoretic), same interpretable "effective dimension" units as $PR$, and has direct precedent in exactly the representation-collapse literature this project is adjacent to.
- **Weaknesses**: near-fully redundant with A4 (both measure "effective dimensionality" from the same spectrum, differing mainly in how the tail is weighted); more sensitive to small noisy tail eigenvalues than $PR$, which is the less desirable property of the two for this project's small-$n$-relative-to-$d$ regime.
- **Verdict: Secondary.** Theoretically sound and literature-precedented, but redundant with A4 for this project's purposes and slightly worse-behaved under estimation noise. Worth computing as a robustness cross-check against $PR$ (if the two disagree materially across the ladder, that itself is informative about tail-eigenvalue instability) but not as an independent primary signal.

#### A6. Covariance anisotropy

- **Definition**: not fully standardized in the literature; common forms include $1 - \lambda_{\min}/\lambda_{\max}$ (bounded $[0,1)$, isotropic at 0) and, in the NLP embedding-geometry literature, an *average-cosine-similarity*-based anisotropy score computed directly on sampled feature vectors rather than on $\Sigma$'s eigenvalues at all (Ethayarajh, 2019, "How Contextual Are Contextualized Word Representations?").
- **Property measured**: same underlying idea as $\kappa$ — directional non-uniformity of variance — but normalized to a bounded scale, or (in the Ethayarajh form) measured via average pairwise cosine similarity of the raw feature vectors rather than the covariance spectrum.
- **Appears directly in Mahalanobis formulation?** The eigenvalue-ratio form is a monotonic reparameterization of $\kappa$ (Concern 1); the cosine-similarity form does **not** correspond to $\Sigma$'s spectrum at all — it measures whether feature vectors point in similar directions, a related but distinct notion of "spread," and is not something $\Sigma^{-1}$ depends on directly.
- **Computational complexity**: eigenvalue-ratio form — same as $\kappa$, $O(d^3)$; cosine-similarity form — $O(n^2 d)$ for exact pairwise cosines, or $O(nd)$ with the standard closed-form average-cosine identity (avoiding explicit pairwise enumeration).
- **Robustness to sample size**: eigenvalue-ratio form inherits $\kappa$'s poor robustness (same $\lambda_{\min}$/$\lambda_{\max}$ sensitivity, just rescaled); cosine-similarity form is more sample-robust since it's an average over all vectors rather than a function of two extreme eigenvalues, but measures a related-not-identical property.
- **Prior use**: the cosine-similarity form is well established in NLP embedding-space analysis (Ethayarajh 2019 and follow-ups) and in self-supervised-learning collapse diagnostics (alongside uniformity/alignment measures, e.g. Wang & Isola, 2020, "Understanding Contrastive Representation Learning Through Alignment and Uniformity on the Hypersphere").
- **Strengths**: bounded, interpretable scale; the cosine-similarity variant is cheap and well-precedented for exactly the kind of "did representation training change the geometry" question this project asks.
- **Weaknesses**: definitional ambiguity across the literature is itself a liability — "anisotropy" means at least two different things depending on source, which risks confusion or an implicit, unstated choice; the eigenvalue-ratio form is functionally redundant with A1; the cosine-similarity form measures something Mahalanobis's formula does not directly depend on (it's about raw vector directions, not the fitted covariance).
- **Verdict: Rejected as a distinct primary/secondary metric**, folded into A1. If a bounded-scale version of "directional spread" is wanted for readability in a table alongside AUROC (which is also bounded $[0,1]$-ish), report $1-\lambda_{\min}/\lambda_{\max}$ as a rescaling of A1's condition number rather than introducing it as a separate, ambiguously-defined metric.

#### A7. Spectral entropy

- **Definition**: identical construction to effective rank's inner term — Shannon entropy $H(\Sigma) = -\sum_i p_i \log p_i$ of the normalized eigenvalues $p_i = \lambda_i/\sum_j\lambda_j$ — but reported directly as the entropy (in nats/bits), rather than exponentiated into "effective rank" units.
- **Property measured**: identical to A5, differing only in units (log-scale entropy vs. exponentiated "effective dimension count").
- **Appears directly in Mahalanobis formulation?** Same answer as A5 — indirect, a spectrum summary.
- **Computational complexity**: identical to A5.
- **Robustness to sample size**: identical to A5.
- **Prior use**: same lineage as effective rank; also appears under "coding rate" framings in some representation-learning work (e.g. the entropy-like rate-distortion measures in Yu et al., 2020, "Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction") as a way to quantify how much of the representation space a set of embeddings actually occupies.
- **Strengths / weaknesses**: same as A5.
- **Verdict: Rejected.** This is A5 in different units with no added information — computing both would be redundant book-keeping, not additional evidence. If entropy units are preferred over "effective dimension count" for the eventual write-up, substitute this for A5 rather than adding it alongside.

---

### Family B — Class-separation geometry

*(Mechanistically load-bearing for CSG-SKin's per-class Mahalanobis, per §1's repo-specific note; not directly applicable to DST-Skin's current single-centroid `OODScorer` without modifying it.)*

#### B1. Fisher ratio (within/between scatter ratio)

- **Definition**: the classical Fisher linear-discriminant criterion, generalized to multi-class multivariate form: $J = \mathrm{tr}(S_W^{-1}S_B)$ (or, univariate/per-direction form, $\dfrac{\sigma_B^2}{\sigma_W^2}$), where $S_B = \sum_c n_c(\mu_c-\bar\mu)(\mu_c-\bar\mu)^\top$ (between-class scatter) and $S_W = \sum_c \sum_{i \in c} (x_i-\mu_c)(x_i-\mu_c)^\top$ (within-class scatter, i.e. exactly the pooled covariance $\Sigma$ CSG's Mahalanobis inverts, up to normalization).
- **Property measured**: how large class-mean separation is relative to within-class spread — precisely Concern (2).
- **Appears directly in Mahalanobis formulation?** **Yes, more directly than any Family-A metric appears in Family A's relationship to Concern (1).** $S_W$ *is* (up to a normalization constant) the $\Sigma$ that CSG's per-class Mahalanobis inverts; $S_B$ encodes exactly the between-class-mean geometry that determines whether $\min_c d_M(x,\mu_c)$ can actually discriminate ID-confident points from OOD-ish ones. This is the single metric on this list with the tightest formal connection to CSG's specific Mahalanobis construction.
- **Computational complexity**: $O(nd + Kd^2)$ to build $S_W$, $S_B$ from $n$ samples and $K$ classes (dominated by the same covariance-accumulation pass any Mahalanobis fit already does), plus $O(d^3)$ for $S_W^{-1}S_B$'s trace/eigenvalues if the full multivariate form is used.
- **Robustness to sample size**: the trace-based multivariate form ($\mathrm{tr}(S_W^{-1}S_B)$) inherits $S_W^{-1}$'s sample-size sensitivity (same regularization/shrinkage requirement as Family A, since $S_W \propto \Sigma$); a simpler, more robust proxy — total between-class scatter norm divided by total within-class scatter norm (a scalar ratio of two trace quantities, no matrix inverse) — avoids this at the cost of being a coarser summary.
- **Prior use in OOD/representation-learning literature**: the direct generalization of the classical Fisher/LDA discriminability criterion; used extensively in domain-adaptation and disentanglement literature to quantify class- vs. domain-separability trade-offs — directly analogous to what CSG-SKin's own domain-leakage probe (`check_leakage.py`) already measures via a logistic-regression proxy, just formalized geometrically instead of via probe accuracy.
- **Strengths**: the most theoretically direct link to CSG's actual per-class Mahalanobis construction on this entire list; a natural geometric complement to the leakage probe CSG already runs (same underlying scatter matrices, different summary statistic).
- **Weaknesses**: the full matrix-inverse form shares Family A's small-sample instability if $S_W$ isn't regularized consistently with however the actual Mahalanobis fit regularizes $\Sigma$; requires deciding between the full-matrix and scalar-ratio variants before implementation (an open item, not resolved here).
- **Verdict: Primary.** This is the single strongest theoretical candidate for Concern (2) and arguably the most directly relevant metric on the entire list to CSG-SKin's actual Mahalanobis formula.

#### B2. Within/between scatter (reported as raw scatter matrices/traces, not their ratio)

- **Definition**: $\mathrm{tr}(S_W)$ and $\mathrm{tr}(S_B)$ reported separately (rather than as the B1 ratio).
- **Property measured**: the same underlying quantities as B1, but unratioed — lets within-class compactness and between-class separation be tracked as two separate trajectories across the dose-response ladder instead of collapsed into one number.
- **Appears directly in Mahalanobis formulation?** Same as B1 — $\mathrm{tr}(S_W)$ is literally $\mathrm{tr}(\Sigma)$ up to a normalization constant (i.e. this is A2, computed per-class-pooled rather than globally).
- **Computational complexity / robustness**: identical to B1's inputs — cheaper than B1 since no matrix inverse is needed (raw traces only).
- **Prior use**: standard decomposition underlying every LDA/Fisher-ratio analysis; reported separately whenever a paper wants to distinguish "clusters got tighter" from "clusters got further apart" as different findings, which is exactly the diagnostic granularity H3 benefits from (shortcut-mitigation training could plausibly move these two quantities in different directions — e.g. tighter within-class clusters via better feature learning, but also compressed between-class separation as a side effect of adversarial domain-unlearning, which is compatible with the ratio B1 staying flat while masking two offsetting mechanisms).
- **Strengths**: decomposes B1 into diagnosable parts at essentially no extra cost, since both traces are needed to compute B1 anyway.
- **Weaknesses**: not a single summary number, so more of a diagnostic breakdown than a standalone reportable metric; scale-dependent across rungs if overall feature magnitude drifts (should be reported alongside, or normalized by, overall scale).
- **Verdict: Secondary.** Report alongside B1 as the decomposition of it — free once B1 is computed, and adds exactly the interpretability a single ratio would hide.

#### B3. Silhouette score

- **Definition**: for each point $i$ in cluster $c$, $s(i) = \dfrac{b(i)-a(i)}{\max(a(i),b(i))}$, where $a(i)$ = mean distance to other points in $i$'s own cluster, $b(i)$ = mean distance to points in the nearest other cluster; averaged over all points, ranges $[-1,1]$.
- **Property measured**: cluster cohesion and separation jointly, computed from pairwise *distances* directly on the data, not from the parametric mean/covariance summary B1/B2 use.
- **Appears directly in Mahalanobis formulation?** No — it is metric- (usually Euclidean-) based and non-parametric; it doesn't reference $\Sigma$ or $\Sigma^{-1}$ at all. It measures a *qualitatively similar* concept to B1 (are classes well-separated relative to their spread?) through an entirely different, distribution-free mechanism.
- **Computational complexity**: naively $O(n^2 d)$ for all pairwise distances (expensive for CSG's ISIC-scale training sets — thousands of samples per class); approximate/sampled variants exist but add estimator variance.
- **Robustness to sample size**: reasonably robust in the sense of not depending on any matrix inversion, but sensitive to class-size imbalance (PAD-UFES's DF/VASC classes are known to be small and imbalanced per the CSG repository audit) — silhouette scores for small clusters are noisier and can dominate or be swamped by large-cluster contributions depending on how the class-level average is taken.
- **Prior use**: classic clustering-validity index (Rousseeuw, 1987); used in representation-learning papers as a distribution-free sanity check on whether learned embeddings cluster by the intended factor (e.g. by class) and not by a confound (e.g. by domain) — directly analogous in spirit to what CSG's leakage probe already checks, but geometric/non-parametric rather than probe-based.
- **Strengths**: makes no Gaussian or covariance-based assumption at all, so it's a genuinely independent cross-check on class separation that doesn't share Family A's estimation-noise sensitivity; standard, well-understood, available in `scikit-learn`.
- **Weaknesses**: $O(n^2)$ cost is a real concern at ISIC scale without subsampling; doesn't connect to the Mahalanobis formula the way B1 does — it's evidence *about* Concern (2), not a component *of* the formula.
- **Verdict: Secondary.** Valuable as a non-parametric cross-check on B1 (if silhouette and Fisher ratio tell the same story across the ladder, that's convergent evidence; if they diverge, that itself is informative about whether B1's Gaussian-adjacent scatter assumptions are the issue), but not primary since it has no direct algebraic relationship to what Mahalanobis computes.

#### B4. Davies-Bouldin index

- **Definition**: $DB = \dfrac{1}{K}\sum_c \max_{c' \neq c} \dfrac{\sigma_c + \sigma_{c'}}{d(\mu_c,\mu_{c'})}$, where $\sigma_c$ is average within-cluster scatter and $d(\mu_c,\mu_{c'})$ is between-centroid distance. Lower is better (more separated).
- **Property measured**: same conceptual target as silhouette (B3) and Fisher ratio (B1) — within-cluster compactness vs. between-cluster separation — but via a centroid-and-scatter construction closer to B1's than to B3's pairwise-distance construction.
- **Appears directly in Mahalanobis formulation?** No, for the same reason as B3 — it's centroid/scatter-based but doesn't reference $\Sigma^{-1}$; closer in spirit to B1 than B3 is, but still not algebraically part of the Mahalanobis formula.
- **Computational complexity**: $O(nd + K^2d)$ — cheaper than silhouette (no all-pairs distance computation, only per-class centroids and worst-case pairwise centroid comparisons), comparable to B1's cost.
- **Robustness to sample size**: similar profile to B1 — depends on per-class mean and scatter estimates, which are the same quantities B1 needs, so shares B1's sample-size behavior without requiring a matrix inverse (uses scalar per-cluster scatter, not the full covariance).
- **Prior use**: classic clustering-validity index (Davies & Bouldin, 1979); used alongside silhouette in representation-quality evaluation, generally as a secondary/confirmatory measure rather than a primary result.
- **Strengths**: cheaper than silhouette, algebraically closer to B1 (uses class means and scatter directly, like Fisher ratio, rather than raw pairwise distances).
- **Weaknesses**: substantively redundant with B1 (both are ratio-of-scatter-to-separation constructions using class means and within-class scatter) and with B3 (both are classic clustering-validity indices measuring the same underlying concept); doesn't add a distinct failure mode beyond what B1+B3 already jointly cover.
- **Verdict: Rejected.** Sits directly between B1 and B3 in what it measures and how it's computed, without adding either B1's tight formal connection to Mahalanobis or B3's genuinely distribution-free, non-parametric independence. Including it alongside both would be reporting the same underlying signal three times.

---

### Family C — Distributional shape (Gaussianity)

#### C1. Henze-Zirkler test

- **Definition**: a kernel-based (Gaussian-kernel) affine-invariant test statistic comparing the empirical characteristic function of the (whitened) sample to that of a multivariate normal; produces a test statistic and p-value under $H_0$: data is multivariate normal (Henze & Zirkler, 1990).
- **Property measured**: multivariate normality directly — Concern (3) in its most literal form.
- **Appears directly in Mahalanobis formulation?** It tests the assumption *underlying* $d_M$'s interpretation as a Gaussian log-density term (per §1, point 3), but is not itself a term in the formula.
- **Computational complexity**: $O(n^2 d)$ (pairwise-distance/kernel construction over all sample pairs) — the most expensive candidate on this list at ISIC-scale per-class sample counts (potentially thousands of samples per class after the 80/20 split), though still feasible for a one-time per-rung/per-class computation rather than a per-training-step one.
- **Robustness to sample size**: **this is the crux of Family C's practical fit for this project, and it cuts both ways.** With small $n$, the test has low power to detect real non-normality (won't reject $H_0$ even if the data isn't Gaussian). With large $n$ — which per-class ISIC counts likely provide — the test becomes *extremely* sensitive, rejecting $H_0$ for trivial, practically-irrelevant deviations from exact normality (a well-known general property of goodness-of-fit hypothesis tests: power grows with $n$ regardless of effect size). **In high dimensions specifically** (this matters directly for `backbone_raw` at $d=1536$–$2048$), Henze-Zirkler's kernel-bandwidth-dependent construction degrades — the test becomes both computationally heavier and statistically less reliable as $d$ grows relative to $n$, and most standard implementations either become impractical or require $n \gg d$ to behave well. **This test is realistically only usable on $z_{lesion}$ ($d=16$) directly; $z_{context}$ ($d=64$) is borderline; `backbone_raw`/baseline features ($d\geq1536$) are not a reasonable target for this test without a dimensionality-reduction step first (which would then be testing normality of the reduction, not the original representation).**
- **Prior use in OOD/representation literature**: less common as a *headline* result in deep-learning OOD papers (which more often assume Gaussianity implicitly, per Lee et al. 2018, rather than testing it) than in the statistics literature it comes from; its relevance here is precisely that testing this normally-unstated assumption is the novel angle H3 is pointing at.
- **Strengths**: a formal, well-established, affine-invariant multivariate normality test — directly targets Concern (3) with a principled statistical procedure rather than an ad hoc proxy.
- **Weaknesses**: impractical or unreliable at the feature dimensionalities this project's higher-dimensional representations live at ($z_{context}$, `backbone_raw`); p-value interpretation at large $n$ needs care (statistical vs. practical significance); $O(n^2d)$ cost is real at ISIC scale.
- **Verdict: Primary, but scope-limited to $z_{lesion}$ (and cautiously $z_{context}$).** This is the most direct available test of Concern (3), and Concern (3) is otherwise completely unaddressed by every other candidate on this list — but its dimensionality ceiling needs to be stated explicitly rather than discovered during implementation. Do not attempt it on raw high-dimensional backbone features without a documented dimensionality-reduction step.

#### C2. Mardia's test (multivariate skewness and kurtosis)

- **Definition** (Mardia, 1970): multivariate skewness $b_{1,d} = \frac{1}{n^2}\sum_{i,j} \left[(x_i-\bar x)^\top S^{-1} (x_j - \bar x)\right]^3$ and multivariate kurtosis $b_{2,d} = \frac{1}{n}\sum_i \left[(x_i-\bar x)^\top S^{-1}(x_i-\bar x)\right]^2$, each with known asymptotic null distributions under multivariate normality, giving two separate test statistics/p-values.
- **Property measured**: Gaussianity via specific, named higher-moment deviations (asymmetry and tail-weight) rather than Henze-Zirkler's omnibus characteristic-function comparison — Concern (3), decomposed into two interpretable failure modes instead of one aggregate statistic.
- **Appears directly in Mahalanobis formulation?** Same relationship as C1 — tests an assumption behind $d_M$'s interpretation, not a term in the formula. Notably, Mardia's kurtosis statistic $b_{2,d}$ is literally the sample fourth moment of $d_M(x_i, \bar x)^2$ itself (the squared Mahalanobis distances from the mean, using the sample covariance) — this is the single tightest mechanical link between any Family-C candidate and the actual Mahalanobis-distance values a downstream OOD score would compute, since it's constructed directly from that same quantity.
- **Computational complexity**: skewness statistic is $O(n^2 d)$ (all pairs, same order as Henze-Zirkler); kurtosis statistic is cheaper, $O(nd + d^3)$ (only needs $S^{-1}$ once and one pass over $n$ points).
- **Robustness to sample size**: shares Henze-Zirkler's large-$n$-over-sensitivity and high-$d$-fragility in the skewness term (which needs $S^{-1}$, i.e. inherits exactly Family A's small-sample covariance-inversion problem); the kurtosis-only statistic is comparatively more tractable at moderate $d$ since it needs only one matrix inverse, not a full $n\times n$ pairwise construction, but both terms still degrade as $d$ approaches $n$.
- **Prior use**: one of the two classical multivariate-normality tests (alongside Henze-Zirkler) in the statistics literature; adopted less frequently than Henze-Zirkler in modern applied normality-testing software, but the *decomposition* into separate skewness/kurtosis statistics is specifically useful when a paper wants to say *which* kind of non-normality is present (asymmetric clusters vs. heavy/light tails), which is more informative for explaining *why* Mahalanobis might degrade than a single omnibus statistic would be.
- **Strengths**: the kurtosis statistic's direct algebraic identity with squared Mahalanobis distances from the mean is a uniquely strong, literal connection to this project's actual downstream quantity; decomposing into skewness vs. kurtosis gives a diagnosable failure mode rather than a single yes/no.
- **Weaknesses**: two statistics to report and interpret instead of one; the skewness term shares Henze-Zirkler's high-dimensionality fragility; both terms are, like C1, realistically only reliable at $z_{lesion}$'s dimensionality (or with a documented dimensionality-reduction step for higher-dimensional features).
- **Verdict: Primary (kurtosis term), Secondary (skewness term), scope-limited the same way as C1.** The kurtosis statistic's direct construction from squared Mahalanobis distances makes it arguably the single most mechanistically-connected Family-C candidate on this list — worth reporting as a primary companion to Henze-Zirkler rather than instead of it, since it targets a specific, diagnosable failure mode (tail weight) that HZ's omnibus statistic doesn't distinguish from asymmetry. The skewness term is secondary: informative but the more expensive and more fragile of Mardia's two statistics, and partially redundant with what HZ already screens for in aggregate. **Post-implementation note**: the scope-limitation above was not merely theoretical — applying this metric's bootstrap calibration to `baseline_soft`'s native 2048-d feature (outside the stated $z_{lesion}$ scope) produced the GPU-server performance crisis that led to §6's resolution and `SPEC.md` §4's primary/secondary ladder split. The caution held; the fix was scoping the experiment, not further optimizing the metric at that dimensionality.

---

## 4. Summary table

| # | Metric | Family | Concern | Direct in $d_M$ formula? | Complexity | Sample-size robustness | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | Condition number | A | (1) | Yes | $O(d^3)$ | Poor | **Primary** |
| A2 | Trace(Σ) | A | (1)* | Indirect | $O(nd)$ | Good | Secondary |
| A3 | Log-determinant | A | (1) | In log-density, not raw $d_M$ | $O(d^3)$ | Very poor | Rejected |
| A4 | Participation ratio | A | (1) | Indirect | $O(d^3)$ or $O(nd)$ | Moderate–good | **Primary** |
| A5 | Effective rank | A | (1) | Indirect | $O(d^3)$ | Moderate | Secondary |
| A6 | Covariance anisotropy | A | (1) | Redundant w/ A1, or off-formula | Varies | Varies | Rejected (fold into A1) |
| A7 | Spectral entropy | A | (1) | Indirect | $O(d^3)$ | Moderate | Rejected (redundant w/ A5) |
| B1 | Fisher ratio | B | (2) | Yes (CSG only) | $O(nd{+}Kd^2{+}d^3)$ | Poor (inverse) / moderate (scalar variant) | **Primary** |
| B2 | Within/between scatter | B | (2) | Yes (CSG only) | $O(nd{+}Kd^2)$ | Moderate | Secondary |
| B3 | Silhouette | B | (2) | No (non-parametric) | $O(n^2d)$ | Moderate | Secondary |
| B4 | Davies-Bouldin | B | (2) | No | $O(nd{+}K^2d)$ | Moderate | Rejected (redundant w/ B1+B3) |
| C1 | Henze-Zirkler | C | (3) | Tests the assumption | $O(n^2d)$ | Poor at high $d$ | **Primary** (scope-limited to $z_{lesion}$) |
| C2 | Mardia (kurtosis) | C | (3) | Built from $d_M^2$ directly | $O(nd{+}d^3)$ | Poor at high $d$ | **Primary** (scope-limited); skewness term Secondary |

*A2's Concern-(1) relevance is only as a normalizer for A4/A7, not standalone.

---

## 5. Final minimal metric set (defensibility-optimized)

**The goal of this section is not completeness — it is the smallest set of metrics that can carry the paper's causal claim without a reviewer being able to say "you didn't test the assumption that actually matters."** §3/§4 above enumerated 13 candidates and screened out clearly-redundant ones; this section applies a stricter bar on top of that screening: *every remaining metric must earn its place by covering a concern nothing else on the list covers*, and no concern from §1 may be left completely untested. That second requirement is the reason the set below is 3, not fewer — H3 explicitly says shortcut-mitigation training "may alter **the assumptions** required by Mahalanobis" (plural), and §1 formally decomposed "the assumptions" into three logically independent concerns. Dropping any one of the three leaves a specific, nameable hole a reviewer can point at. Adding a fourth or fifth metric, conversely, buys no new coverage — it only re-measures a concern already covered, which weakens rather than strengthens the paper (more geometry numbers correlated against one AUROC series in a small-sample study is a multiple-comparisons liability, not added rigor).

### The set (exactly 3, one per concern)

| Concern (§1) | Metric | Why this one, specifically |
|---|---|---|
| (1) Covariance conditioning | **Condition number**, $\kappa(\Sigma) = \lambda_{\max}/\lambda_{\min}$ | The only candidate that is *literally* the object $\Sigma^{-1}$ is built from — not a correlate of it. Its main weakness (small-sample fragility of $\lambda_{\min}$) is neutralized, not just noted: computed directly from the same regularized `precision` matrix CSG's `ood_metrics.py::compute_mahalanobis_params_from_arrays` already produces (condition number is invariant under matrix inversion, $\kappa(\Sigma)=\kappa(\Sigma^{-1})$, so no extra inversion or re-estimation is needed), which means the metric describes the exact object the downstream Mahalanobis score actually uses, not an independently-estimated proxy. |
| (2) Class-separation geometry | **Fisher ratio**, $J=\mathrm{tr}(\Sigma^{-1}S_B)$ | The single tightest algebraic connection to CSG's per-class Mahalanobis construction of anything considered — $S_W$ (within-class scatter) *is* the numerator of the same $\Sigma$ the precision matrix above inverts. Computed as $\mathrm{tr}(\text{precision} \cdot S_B)$ directly on the same `precision`/`class_means` the estimator already fit, for the same reason as the row above: one estimation pass, not two divergent ones. |
| (3) Distributional shape | **Mardia's multivariate kurtosis**, $b_{2,d}=\frac1n\sum_i\big[(x_i-\mu_{y_i})^\top\Sigma^{-1}(x_i-\mu_{y_i})\big]^2$ | Chosen over Henze-Zirkler *and* over Mardia's own skewness term for the same reason as both rows above: $b_{2,d}$ is not merely related to squared Mahalanobis distances, it is their empirical fourth moment, computed from the exact same per-sample $d_M^2$ values the reliability estimator produces. It is also $O(nd+d^3)$ rather than Henze-Zirkler's $O(n^2d)$, and a single scalar with a known asymptotic null ($b_{2,d}\sim\mathcal N(d(d+2),\,8d(d+2)/n)$) rather than two statistics to interpret. |

Read as one sentence per row: *is $\Sigma^{-1}$ numerically trustworthy (1), does the class structure the precision matrix is meant to discriminate actually separate (2), and does the empirical distribution of the exact distances the estimator computes actually look like what the Gaussian model underlying Mahalanobis predicts (3)?* All three are computed from the same fitted `(class_means, precision)` pair — one estimation call, three read-outs, no independently-estimated proxy objects to reconcile.

### Why every other candidate is rejected

| Metric | Reason for rejection |
|---|---|
| Trace(Σ) (A2) | Not diagnostic of Mahalanobis validity on its own (§3, A2) — it measures overall scale, not conditioning; a covariance can have large trace and still be singular. Its only role in the wider audit was as a normalizer for participation ratio / spectral entropy, both of which are themselves rejected below, so it has no remaining role. |
| Log-determinant (A3) | Appears in the Gaussian log-*density* Mahalanobis is derived from, but not in the raw squared distance both CSG's and DST's code actually compute and threshold (§3, A3) — measuring it would characterize an estimator neither repo implements. Also the single worst-behaved candidate under small-sample/high-dimension conditions (diverges to $-\infty$ whenever any sample eigenvalue is near zero), a liability condition number avoids by being a ratio rather than a sum of logs. |
| Participation ratio (A4) | Legitimate and more sample-robust than condition number in isolation, but redundant *for this paper's purpose*: once condition number is computed on the exact regularized precision matrix (neutralizing the small-sample concern that motivated wanting a second, gentler Family-A metric in the first place), a second conditioning metric adds no new concern-coverage — only a second, correlated number to the same question row 1 of the table above already answers. |
| Effective rank (A5) | Same underlying construction as participation ratio (both are functions of the eigenvalue spectrum's second moment/entropy) and rejected for the identical reason: redundant with condition number once condition number is computed correctly, and redundant with participation ratio if that were kept instead. |
| Covariance anisotropy (A6) | Not a single, standardized definition across the literature — the eigenvalue-ratio form is a rescaling of condition number (pure duplication); the cosine-similarity form measures something Mahalanobis's formula does not depend on at all (§3, A6). Neither form clears the bar of "measures something condition number doesn't." |
| Spectral entropy (A7) | Identical construction to effective rank in different units (§3, A7) — rejected for the same reason, twice over. |
| Within/between scatter, unratioed (B2) | A decomposition of the Fisher ratio into its two components, not an independent concern — informative as a diagnostic *if* the Fisher ratio result needs explaining later, but not a metric the paper's core claim needs reported on its own. Demoted from the earlier "secondary" recommendation because "smallest defensible set" means the headline result is the ratio, not its parts. |
| Silhouette (B3) | Measures the same concern as Fisher ratio (class separation) through a non-parametric, pairwise-distance route instead of the Mahalanobis-consistent scatter-matrix route. Genuinely independent methodologically, but that independence is exactly what makes it *not* part of the minimal set: it doesn't test whether $\Sigma^{-1}$-based reliability estimation specifically is affected, only whether classes look separated under an unrelated (Euclidean, non-parametric) notion of distance. Also $O(n^2d)$, the most expensive candidate in Family B at ISIC scale. |
| Davies-Bouldin (B4) | Sits between Fisher ratio and silhouette in construction and measures the same concern as both without adding a distinct failure mode (§3, B4) — the most purely redundant candidate on the entire list. |
| Henze-Zirkler (C1) | An omnibus normality test, not one built from the Mahalanobis distances themselves — it tests the raw feature vectors' characteristic function, a step removed from the actual quantity ($d_M^2$) whose distribution matters for the paper's claim. Also the most computationally expensive candidate audited ($O(n^2d)$) and the one whose reliability degrades most sharply as dimensionality grows relative to sample size (§3, C1) — a real risk at $z_{context}$'s 64 dimensions. Mardia's kurtosis targets the same concern (3) more cheaply and with a tighter mechanistic link to the paper's actual dependent variable. |
| Mardia's skewness (part of C2) | Shares Henze-Zirkler's $O(n^2d)$ cost and high-dimensionality fragility (it needs the same all-pairs construction), and is not the sub-statistic with the direct fourth-moment-of-$d_M^2$ identity that makes the kurtosis term uniquely defensible. Detects a different failure mode (asymmetry vs. tail weight) — legitimate as future/supplementary work, but the minimal set keeps one Gaussianity statistic, not two, and the kurtosis term is the more defensible of the two by the same "built directly from the actual quantity" criterion used to choose the other two metrics. |

### What this buys, stated plainly

Three numbers, one fitting call, each answering a distinct, named question a reviewer would otherwise have to ask by hand: *is the precision matrix well-conditioned, are the classes it's meant to separate actually separated, and do the distances it produces look like the Gaussian model assumes they should?* Nothing on the rejected list answers a fourth question these three don't already cover — every rejection above is either "measures the same concern as something already kept" or "not actually part of the object the paper's dependent variable (Mahalanobis AUROC) is computed from."

---

## 6. What this document does not decide

- **Regularization consistency**: resolved by §5's final set — all three chosen metrics are computed directly from the `(class_means, precision)` pair CSG's existing `compute_mahalanobis_params_from_arrays` already fits, so there is no separate raw-vs-regularized covariance choice left open.
- ~~**Dimensionality-reduction protocol** for applying the Mardia-kurtosis metric to `backbone_raw`, if that's ever attempted despite the scope caution noted in §3 for Family-C tests at higher dimensionality.~~ **Resolved, and not via dimensionality reduction.** This stopped being hypothetical: `baseline_soft`'s only native representation is its 2048-d ResNet-50 backbone feature, exactly the regime §3 (C1/C2) flagged as outside this test's validated range — confirmed empirically when its Mardia bootstrap became the GPU-server performance bottleneck. Rather than projecting that feature down to match `z_lesion`'s 16 dimensions (which would make E1's geometry describe a different object than what E2's Mahalanobis AUROC is actually computed on for that checkpoint — reintroducing the same "two divergent objects, one nominal row" problem `threats_to_validity.md` #2 already exists to prevent), the resolution was to stop treating `baseline_soft` as a rung on the same ladder as the CSG methods at all. `SPEC.md` §4 now splits it out as a categorical reference: Mardia's kurtosis is exploratory-only there (native 2048-d, outside this section's validated regime, never load-bearing for H3), while condition number and Fisher ratio — which have no analogous dimensionality floor — remain fully load-bearing for it. See `open_questions.md` Q4.
- **The E2 association test** (correlation, regression, or otherwise) between these geometry metrics and Mahalanobis AUROC — **partially resolved**: Kendall's τ + seed-level scatter on the three-rung primary ladder only (`experiment_contract.md` E2a), explicitly not a regression given only three ordinal treatment levels. `baseline_soft`'s comparison against the ladder (E2b) remains descriptive, no formal test.
- **DST-Skin (E4) applicability of Family B**: whether E4 modifies `OODScorer` to be per-class before Family B metrics can be meaningfully applied there, or whether E4 is scoped to Family A/C only given DST's current global-mean formulation. This is a design decision for whenever E4 is actually scoped (per `SPEC.md`, it is optional and scoped last).

No code has been written to implement any of the above.
