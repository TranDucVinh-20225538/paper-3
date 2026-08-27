# Experiment Contract

**Purpose**: pre-registered Input / Output / Success / Failure for E1–E4, fixed *before* any experiment runs. Once an experiment executes, its outcome is read off against the criteria below — not re-argued afterward. If a criterion turns out to be wrong or ill-posed, that is a reason to amend this document explicitly (dated, with a stated reason), not to reinterpret a result to fit.

**One global rule, stated once**: if an experiment's Input preconditions cannot be met (a checkpoint can't be resolved to a verified file, a required deduplication can't be confirmed, a required upstream script/file doesn't exist), the experiment is marked **BLOCKED**, never **FAILED**. Blocked says nothing about H3. Failed is evidence against it. Conflating the two misrepresents what was actually tested.

**No experiment below predicts a *direction* for any geometry metric unless SPEC.md's H3 actually licenses one.** H3 (`SPEC.md` §3) claims shortcut-mitigation training changes feature geometry and that this *may* affect Mahalanobis validity — it does not claim which way any specific metric moves. A contract that quietly baked in an unearned directional prediction would be exactly the kind of unjustified leap `internal_review.md` already flagged elsewhere in this project.

---

## E1 — Geometry: primary disentanglement ladder + secondary categorical reference

**Design note** (`open_questions.md` Q4): `baseline_soft` is no longer a rung on the same ordinal ladder as the three CSG methods — it differs in architecture, representation dimensionality (2048-d vs. 16-d), and training recipe, and forcing it onto one 4-point trend conflated those confounds with the actual manipulated variable (λ_orth). Split below into a primary mechanistic test (E1a) and a secondary categorical reference (E1b), mirroring `SPEC.md` §4.

### E1a — Primary: the disentanglement ladder

**Input**
- 13 CSG-SKin checkpoints: `{runA_grl, runB_orth1}` × seeds `{42,52,62,72,82}` (5 each) + `runB` × seeds `{42,52,62}` (3 only) — each referenced by an explicit, individually-verified `.ckpt` file path recorded in a checkpoint manifest. Directory-resolved paths are not a valid input (see `REPOSITORY_MAP.md` risk #5).
- Per checkpoint: $z_{lesion}$ embeddings + ISIC class labels, extracted via `collect_z_lesion_labels_csg` on the **ISIC-train-under-eval-transform** loader — the same loader `eval_ood_benchmarks.py` uses, not `train_csg.py`'s augmented paired-CSG loader (avoiding the documented grayscale-double-conversion bug, per `threats_to_validity.md` #2).

**Expected output**
- `results/e1_geometry_metrics.csv`, one row per (rung, seed), `rung ∈ {runA_grl, runB_orth1, runB}`: `rung, method, seed, checkpoint_path, n_samples, feat_dim, condition_number, fisher_ratio_HL, fisher_ratio_scalar, mardia_kurtosis_b, mardia_kurtosis_z`.
  (`fisher_ratio_HL` = $\mathrm{tr}(\Sigma^{-1}S_B)$; `fisher_ratio_scalar` = $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$, the decoupled companion required by `fisher_ratio_defense.md` §5.)

**Success**
- At least one of `{condition_number, fisher_ratio_HL, mardia_kurtosis_z}` has $|\tau_{\text{Kendall}}(\text{rung order}, \text{metric})| \geq 0.3$ across all available (rung, seed) points (13 points: 5+5+3), **and** the sign of $\tau$ is preserved when recomputed on the common-seed (42/52/62) subset alone. Kendall's τ, not Spearman's ρ or a fitted regression — with only 3 ordinal rung levels, a coefficient implying more precision than that would misrepresent the design.

**Failure**
- All three metrics have $|\tau| < 0.3$, **or** the sign of $\tau$ is not preserved between the full 13-point set and the common-seed 9-point subset — the latter means `runB`'s seed gap (3 vs. 5) is driving the apparent trend, not a genuine rung effect.

### E1b — Secondary: baseline_soft as categorical reference

**Input**
- 5 `baseline_soft` checkpoints (seeds 42/52/62/72/82), explicit file paths.
- Per checkpoint: the native 2048-d ResNet-50 backbone feature (`_extract_baseline`), ISIC class labels, same eval-transform loader convention as E1a.

**Expected output**
- Rows appended to `results/e1_geometry_metrics.csv` with `rung=baseline_soft`, same schema as E1a. `mardia_kurtosis_b`/`mardia_kurtosis_z` are computed and recorded but flagged exploratory in any downstream write-up — d=2048 is outside this test's validated regime (`geometry_metric_audit.md` §3, C1/C2) and is never used to support or refute H3.

**Success**
- No statistical criterion, matching E4's framing below. "Success" means: `condition_number`, `fisher_ratio_HL`, and `fisher_ratio_scalar` compute without error across all 5 seeds, and are reported descriptively alongside the E1a ladder's range (e.g. does baseline's condition number fall inside or outside the span `runA_grl`→`runB` covers).

**Failure**
- Any of the three load-bearing metrics fails to compute on any seed → reported as **not run** for that seed, not a null result. Mardia kurtosis failing/being slow at d=2048 is not a failure of E1b — it was never load-bearing here; if computed, report it, if not, note it as skipped and why.

---

## E2 — Geometry ↔ Mahalanobis AUROC association

### E2a — Primary: association within the disentanglement ladder

**Input**
- `results/e1_geometry_metrics.csv`, rows with `rung ∈ {runA_grl, runB_orth1, runB}` only.
- Per (rung, seed) Mahalanobis AUROC (ISIC-test vs. PAD-UFES), computed **exclusively** via `cbm_revision/scripts/eval_ood_benchmarks.py` (`threats_to_validity.md` #2). AUROC numbers from any other script (`train_csg.py`'s post-fit path, `eval_ood_scores.py`, a new reimplementation) are not a valid input to this experiment.

**Expected output**
- `results/e2_association.csv` (per-metric Kendall's τ + n) and `results/e2_association_report.md` stating the outcome against the criteria below in plain language, plus a seed-level scatter plot (rung on the ordinal axis, metric vs. AUROC).

**Success**
- For at least one metric E1a flagged as trend-bearing, $|\tau_{\text{Kendall}}(\text{metric}, \text{AUROC})| \geq 0.3$ across the same 13 points, with a sign consistent with both trending in the same rung direction (the direct test of H3's causal chain: representation change → geometry change → reliability change). This is reported as a descriptive association, not a hypothesis test with a formal p-value — three ordinal treatment levels do not support one.

**Failure**
- No E1a-trend-bearing metric reaches $|\tau| \geq 0.3$ against AUROC; **or** a metric trends with rung (per E1a) while AUROC itself shows no rung trend at all. The second case is recorded explicitly as **H3 not supported in this setting** — geometry moved, reliability didn't — not filed as merely inconclusive.

### E2b — Secondary: baseline reference vs. the ladder

**Input**
- E1b's `baseline_soft` geometry row(s) + its Mahalanobis AUROC (same canonical source as E2a).
- E2a's ladder-wide (geometry, AUROC) pattern, for context.

**Expected output**
- A descriptive paragraph in `results/e2_association_report.md`: does baseline's (geometry, AUROC) point sit inside, at the edge of, or outside the range the three-rung ladder spans — i.e. does having *any* disentanglement architecture (vs. none) look different in kind from varying its strength.

**Success / Failure**
- No statistical criterion, same reasoning as E1b/E4 — one categorical point cannot support a trend test. This section exists to contextualize E2a's magnitude, not to be judged pass/fail.

---

## E2.6 — Diagnostic: alternative scorers on the same embeddings (locked before implementation)

**Design note** (`open_questions.md` Q6): E2a found no association between geometry and Mahalanobis AUROC, and separate checks ruled out an implementation bug, a dominant NV attractor, and a large norm-collapse effect as the explanation — while confirming Mahalanobis's own Gaussian assumption is catastrophically and training-invariantly violated (Mardia z 191–824, no trend with λ_orth). This experiment asks whether an assumption-light scorer on the *same* embeddings behaves differently, to distinguish "representation problem" from "scoring-rule problem." Every formula and hyperparameter below is fixed **before** any code is written or rerun executed, specifically to prevent selecting a scorer/hyperparameter after seeing which one looks best — each of the two decisions below (pool structure, k) was an explicit user sign-off, not a default picked silently.

**Input**
- The same 13 checkpoints as E1a/E2a, same explicit-file-path manifest.
- Raw `z_lesion` embeddings (16-d) for all three splits per checkpoint: ISIC-train (the same set already used to fit Mahalanobis's per-class means/covariance), ISIC-test (ID), PAD-UFES (OOD). Saved in full, not subsampled — a k-NN reference pool must be the complete train set or k-th-neighbor distances are not meaningful. (Approx. 1.5 MB/checkpoint × 13 ≈ 20 MB total at 16-d float32 — this is also the data needed for a future UMAP illustration, satisfied as a side effect, not a separate save path.)
- Per-class means already fit for Mahalanobis (reused, not refit).

**Scorer formulas** (all: higher score = more OOD-like, matching the existing `auroc_fpr95_from_scores` convention; label OOD=1, ID=0)

1. **Mahalanobis** — unchanged, reused from E2a's existing `s_id`/`s_ood`. Included only as the comparison baseline, not recomputed.
2. **Cosine-to-centroid** — `score(z) = min_c [1 − cosine_similarity(z, mean_c)]`, `mean_c` = the same per-class means used for Mahalanobis, min over the 8 ISIC classes. No normalization decision needed (cosine similarity is scale-invariant in both arguments by construction) — the only change from Mahalanobis is dropping the covariance/precision-matrix step, isolating exactly that one variable.
3. **k-NN distance** (Sun et al. 2022-style) — `score(z) = ||z − z_(k)||₂`, where `z_(k)` is the k-th nearest neighbor of `z` in the **pooled** ISIC-train set (all 8 classes merged into one reference pool — no per-class split, no covariance, no distributional assumption of any kind; this is deliberately the most assumption-free instrument in the comparison, decided explicitly over a per-class-then-min variant that would have kept too much structural similarity to Mahalanobis to serve as a real contrast). Primary/headline: **k=10**. Pre-registered robustness grid, reported in full regardless of result: **k ∈ {1, 10, 50}**. No k value is ever selected or suppressed after seeing its AUROC.

**Expected output**
- `results/e2_6_scorer_comparison.csv`: one row per (rung, seed, scorer), `scorer ∈ {mahalanobis, cosine, knn_k1, knn_k10, knn_k50}`, columns `auroc, fpr95` (same computation as E2a's `auroc_fpr95_from_scores`).
- Per-scorer Kendall's τ vs. rung order (same exact-permutation machinery as E1a/E2a), for all 5 scorer rows — not just whichever looks most interesting.
- A raw-embedding artifact (`results/e2_distances/{rung}_s{seed}_z.npz` or equivalent), saved as a byproduct of the input requirement above, available for any future exploratory visualization (e.g. UMAP) without a further rerun.

**Success / Failure**
- No new statistical pass/fail criterion is introduced here — this is diagnostic/exploratory, same framing as E1b/E2b/E4, precisely because it was designed *after* E2a's result was already known (a formal criterion invented at that point would not be a genuine pre-registration, regardless of how it's labeled). What is locked is **only** the formulas, hyperparameters, and full-grid reporting requirement above, so that no scorer or k value can be silently dropped from the writeup after implementation. The interpretive question this answers — "does any alternative scorer materially outperform Mahalanobis and/or track λ_orth where Mahalanobis does not" — is reported descriptively against the numbers this produces, not against a threshold decided sight-unseen.

**Result** (all 65 rows validated, no unresolved duplicates): pooled AUROC across the primary ladder — `mahalanobis`=0.4023, `cosine`=0.4181, `knn_k1`=0.4239, `knn_k10`=0.4135 (headline), `knn_k50`=0.4114 — all five scorers land in the same ~0.40–0.42 band, none near the "materially higher" signature that would implicate the scoring rule specifically. Per the interpretive frame above, this outcome points toward the representation, not any one scoring rule, as the bottleneck. One non-contract-passing footnote: Cosine shows the only positive Kendall's τ against λ_orth (τ=+0.321, exact p=0.194) and the closest-to-conventional Jonckheere-Terpstra trend (p_exact=0.097), driven by its AUROC at `runB` (0.4387) — reported for completeness, well short of E1a/E2a's 0.3-τ-and-significant bar.

---

## E2.7 — Diagnostic: how much domain information does the representation itself retain?

**Design note**: E2.6 found that Mahalanobis, Cosine, and pooled k-NN (5 scorer variants total) all fail similarly (~0.40–0.42 AUROC), which rules out the scoring rule as the sole explanation and points toward the representation. The natural follow-up — proposed and then explicitly re-scoped by the user before implementation — is *why* the representation fails every distance-based scorer. The first framing considered was "does the domain-adversarial (GRL) objective succeed at making the representation domain-invariant," operationalized as reading out the trained domain discriminator's own accuracy (`_d_adv`, already computed and discarded in `extract_auroc_e2.py`'s `_extract_csg`). This was rejected before implementation for a specific reason: `_d_adv` is the output of a component the training objective directly optimizes — using it to test "did the objective succeed" is circular ("vừa đá bóng vừa thổi còi"), and a null result (`_d_adv` AUROC≈0.5) would only show the discriminator failed, not that domain information is actually absent from `z` (could be discriminator undertraining, architecture bottleneck, calibration — several alternative explanations, none ruled out). A high `_d_adv` AUROC alongside failing distance scorers would also not resolve the question either way.

**Reframed test**: instead of asking whether the trained discriminator succeeded, ask directly how much domain information the embedding `z` contains, using an *independent* probe never exposed to the training objective — the standard representation-learning practice for this question, and harder for a reviewer to dismiss as circular. Fit fresh classifiers (`LogisticRegression`, linear `SVM`, `RandomForest` — three different inductive biases, none tuned on this objective) to predict domain (ISIC-test=0 vs. PAD-UFES=1) directly from `z`, using the *exact same* ID/OOD embeddings E2/E2.6 already scored — no rerun, no GPU, no checkpoint reload required, since `z_id`/`z_ood` are already a saved artifact (E2.6's raw-embedding save).

**Input**
- `results/e2_distances/{rung}_s{seed}_z.npz` (E2.6's raw-embedding save) — `z_id`, `z_ood` only; `z_train`/`y_id`/`y_ood` (ISIC class labels) are not used here, domain is the only label of interest.
- Domain label: 0 for every `z_id` row (ISIC-test), 1 for every `z_ood` row (PAD-UFES) — the identical population and identical embeddings E2a/E2.6 computed AUROC on, so this is a direct apples-to-apples comparison, not a different data slice.

**Method**
- Three probes, none seeing any part of the original training objective: `LogisticRegression` and linear `SVM` (both with `StandardScaler` preprocessing), `RandomForestClassifier` (no scaling needed). All three use library defaults for hyperparameters — no tuning, since tuning a probe to maximize domain AUROC would bias the very question being asked.
- 5-fold stratified cross-validation, out-of-fold predictions via `cross_val_predict`, AUROC computed on the pooled out-of-fold scores — avoids the same-data-fit-and-eval leakage that would inflate domain AUROC in low-dimension (`d=16`), moderate-n settings.

**Expected output**
- `results/e2_7_domain_probe.csv`: one row per (rung, seed, probe), columns `domain_auroc`, `n_id`, `n_ood`.
- A direct joint table against E2.6's distance-scorer AUROC, per rung: e.g. "distance scorers ≈0.40, domain probe ≈0.50" vs. "distance scorers ≈0.40, domain probe ≈0.95" are two different findings requiring different write-ups (the first: domain information is genuinely gone from `z`; the second: domain information survives in `z` but in a form no distance-based scorer here can exploit — a deeper and more specific finding).

**Success / Failure**
- No pass/fail criterion, same diagnostic/exploratory framing as E2.6 — designed after E2.6's result was known. Reported descriptively against the two contrasting readings above, whichever the actual numbers support.

---

## E2.8 — Extending the scorer family past "distance-based": Energy, ViM, density

**Design note**: E2.6 showed three structurally distinct distance-based scorers (Mahalanobis, Cosine, pooled k-NN) all fail identically. The paper's claim is scoped to "distance-based reliability estimation" specifically — an anticipated reviewer question is "accessible to WHAT, exactly?" This experiment extends the family being ruled out past distance-based scorers to a logit-based confidence score (Energy), a hybrid residual-subspace/logit score (ViM), and a non-parametric density estimator, none of which are distance-based in the E2.6 sense, all fixed **before** the code below was written.

**Input**
- The same 13 primary-ladder checkpoints as E2/E2.6. `baseline_soft` is explicitly **out of scope** here: it uses `BaselineResNet50`, a different architecture/classifier head than `CSGLiteLightning`'s `lesion_classifier`, and `open_questions.md` Q4 already documents it as not a valid zero-dose point on this ladder — extending logit reconstruction to a second architecture for one reference checkpoint was judged not worth the added surface area.
- E2.6's cached raw embeddings (`results/e2_distances/{rung}_s{seed}_z.npz`: `z_train`, `y_train`, `z_id`, `z_ood`) — no re-extraction from images.
- Logits, reconstructed **offline**, not from a fresh forward pass: `logits = z @ W.T + b`, where `W`, `b` are `model.lesion_classifier.{weight,bias}` read directly from each checkpoint's `state_dict` (no `pytorch_lightning` import needed for this — only two tensors are read, not the full Lightning module). Verified correct before use: applying this to `runA_grl_s42`'s cached `z_id`/`z_train` and taking `argmax` reproduces the classifier's own predictions at 81.0% accuracy on ISIC-test and 99.4% on ISIC-train against the true labels — both squarely in the range expected of a trained 8-class classifier, confirming `z_lesion` is cached post-`lesion_bn` (i.e. exactly what `lesion_classifier` consumes), not some other stage of the forward pass.

**Scorer formulas** (all: higher score = more OOD-like, same `auroc_fpr95_from_scores` convention as E2.6; label OOD=1, ID=0)

4. **Energy** (Liu et al. 2020) — `score(z) = -logsumexp(logits(z))`. Imported, not reimplemented: `DST-Skin/src/utils/scoring.py::OODScorer.score_energy` (a `@staticmethod`, callable with no `fit()`/instantiation), negated once at the call site — DST's own docstring states it returns "higher = more ID/confident," the opposite of this project's convention; the negation is a usage decision, not a reimplementation.
5. **ViM / Virtual-Logit Matching** (Wang et al. 2022) — imported directly and unmodified: `DST-Skin/src/utils/ood_vim_react.py::fit_vim`/`vim_score`. Fit on `z_train` + reconstructed train logits + `(W, b)`; `d` (residual-subspace dimension) left at `fit_vim`'s own default, `feat_dim − num_classes = 16 − 8 = 8`, not overridden. `vim_score`'s return sign already matches this project's "higher = more OOD-like" convention as-is (verified by reading the formula: `score = vlogit − energy`, both terms individually pushing the same direction for an OOD point) — no negation applied, unlike Energy.
6. **Density (per-class KDE)** — `score(z) = -max_c[log p_c(z)]`, `p_c` = a Gaussian-kernel `KernelDensity` fit on `z_train`'s class-`c` subset only (mirroring Mahalanobis/Cosine's per-class-then-min structure, but replacing the single-Gaussian assumption with a non-parametric multi-mode density, which is the point — a per-class single-Gaussian density estimator would just be a monotonic transform of Mahalanobis and wouldn't test anything new). Bandwidth fixed by Scott's rule, `h_c = σ̄_c · n_c^{-1/(d+4)}` (`σ̄_c` = mean per-dimension std of class `c`'s training samples, `d=16`), computed once per class from the data, never tuned against AUROC. No density-based OOD scorer exists in either sibling repo — this is new code, not an import, precisely because the search confirmed nothing to reuse.

**Expected output**
- New rows appended to `results/e2_6_scorer_comparison.csv`: `scorer ∈ {energy, vim, density_kde}`, same 7-column schema as E2.6, same 13×3 = 39 rows.
- `analysis/analyze_e2_6.py`'s `SCORERS` list extended to include the three new names, so the existing summary table / Kendall's τ / Jonckheere-Terpstra / comparison figure machinery covers all 8 scorers with no separate analysis path.

**Success / Failure**
- Same framing as E2.6/E2.7: diagnostic, designed after E2.6's null result was known, no new pass/fail threshold. Reported descriptively — either the three new scorers land in the same ~0.35–0.45 band as the five E2.6 scorers (representation-level failure, now shown across an even broader scorer family), or at least one clears it materially (would narrow the claim to specifically distance-based scorers after all).

**Result** (all 39 rows, 13 checkpoints × 3 scorers, no unresolved duplicates): pooled AUROC — `energy`=0.4466±0.0465, `vim`=0.3867±0.0223, `density_kde`=0.4167±0.0245. Combined with E2.6's five, all eight scorers land in `energy`'s span down to `vim`'s span, i.e. 0.387–0.447 — no scorer approaches a "materially higher" AUROC that would overturn the representation-level reading. One footnote, same class as E2.6's Cosine footnote: Energy's Jonckheere–Terpstra trend across the ladder crosses conventional significance (J=42.0, p_exact=0.032), and its Kendall's τ is the largest of all eight scorers (τ=+0.443) but does **not** clear E2a's pre-registered |τ|≥0.3-and-significant bar (exact p=0.065). Explicit user decision on how to report this (not silently resolved): neither dismiss it as a false positive (no evidence for that beyond "it's 1 of 8 tests") nor elevate it to a discovery (would require a mechanistic account this experiment wasn't designed to give) — report the numbers plainly in Results, discuss it in Discussion as an isolated, hypothesis-generating observation (small effect size, exploratory, not pre-registered, Kendall's τ short of threshold), and leave the main conclusion as "accessibility *generally* fails" rather than "no scorer ever shows any trend." ViM and density_kde show no such signal (ViM's pooled AUROC is in fact the lowest of all eight scorers), which if anything strengthens the paper's claim: a representation-aware residual-subspace scorer (ViM) and a non-parametric density estimator both fail alongside the distance-based family.

---

## E3 — Replication on HAM10000

**Input**
- The same 18 checkpoints from E1 (13 primary-ladder + 5 baseline-reference, no retraining).
- HAM10000 images/labels, **deduplicated against `isic_2018_binary_train.csv`'s image IDs** (per DST-Skin's orphaned `split.py` logic, which must be re-verified since its own output CSV does not currently exist — `REPOSITORY_MAP.md` §3.4/§6). Deduplication having been confirmed is a precondition of a valid input, not a post-hoc cleanup step — `REPOSITORY_MAP.md` §3.4 already documents that ISIC2018 and HAM10000 overlap heavily, so an unverified dedup makes this experiment untrustworthy by construction, not merely noisy.

**Expected output**
- `results/e3_geometry_metrics_ham10000.csv`, `results/e3_association_ham10000.csv` — same schemas as E1a/E2a for the primary ladder, plus the same E1b/E2b-style descriptive baseline row, HAM10000 substituted for PAD-UFES as the shifted-domain evaluation set.

**Success**
- E2a's trend-bearing association replicates on the primary ladder only: same sign, $|\tau_{\text{Kendall}}| \geq 0.2$ (a lower bar than E2a's 0.3 — this is a secondary confirmation set, smaller and not the primary test). `baseline_soft` replicates only as the same descriptive reference (E1b/E2b-style), never pooled into this criterion.

**Failure**
- Deduplication cannot be verified as complete → the experiment is **BLOCKED**, full stop; reporting an association computed on possibly-contaminated HAM10000 data as confirmatory would misstate what was tested, regardless of what number comes out.
- Deduplication succeeds but the sign reverses or $|\tau| < 0.2$ on the primary ladder → reported as **did not replicate on an independent domain**, named explicitly as a distinct outcome from "blocked."

---

## E4 — Supplementary check on DST-Skin (optional)

**Input**
- DST-Skin's 3 existing checkpoints (ResNet-18/50, EfficientNet-B3) — unseeded, single run each, no retraining.
- DST's own binary label taxonomy (not CSG's 8-class labels — no cross-repo label mapping is invented, per `threats_to_validity.md` #5).
- DST's `OODScorer` Mahalanobis (Ledoit-Wolf, single global mean) — a different estimator from E1–E3's, not interchangeable with it.
- Only `condition_number` and `mardia_kurtosis` are computed; Fisher ratio is not applicable here, since DST's `OODScorer.fit` never uses per-class labels (`geometry_metric_audit.md` §1, §6).

**Expected output**
- `results/e4_geometry_metrics_dst.csv` (3 rows, one per backbone) and `results/e4_notes.md` stating explicitly that this is a single-run, 3-point illustration, not a statistical test.

**Success**
- There is no statistical success criterion. With 3 backbones and zero seed replication, no correlation or regression is well-posed. "Success" here means only: both applicable metrics compute without error on all 3 backbones, and their values are reported descriptively alongside each backbone's Mahalanobis AUROC.

**Failure**
- Either metric fails to compute on any backbone (e.g. a shape/label mismatch in DST's pipeline) → reported as **not run** for that backbone, not as a null result.
- If E4 runs successfully, there is no further failure criterion, precisely because no statistical claim is being tested — any future write-up must state this explicitly so E4 is never cited later as though it had a pass/fail outcome it was never designed to have.
