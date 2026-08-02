# Open Questions

Log only. Nothing here gets acted on until its own Action item is explicitly executed as a separate task — this file exists so open concerns don't get silently redesigned around or silently forgotten.

---

## Q1. Dose-response ladder may contain multiple simultaneous interventions

**Status**: Largely resolved by Q4 — the specific rung this flagged (`baseline_soft`, rung 0) is no longer part of the primary dose-response trend at all. The remaining, much narrower question is whether `runA_grl`/`runB_orth1`/`runB` (the actual primary ladder now) share the same lr/transform recipe as each other — the CSG-guide table cited below suggests they do (all three are CSG runs, distinct from `baseline_soft`'s different recipe), but this should still be spot-checked before trusting E1a/E2a's trend.

**Context**: `internal_review.md` found that CSG-SKin's four ladder rungs (`baseline_soft → runA_grl → runB_orth1 → runB`) differ by more than `λ_orth` alone — per `REPOSITORY_MAP.md` §2.8, `baseline_soft` uses a different learning rate (2e-4 vs. 1e-4) and a different augmentation pipeline (light vs. robust transforms) than the three CSG runs. Rung 0 is not a controlled zero-dose condition.

**Action**: Verify actual per-rung training configs (learning rate, transform pipeline, epoch count) against `REPOSITORY_MAP.md`'s §2.8 table before interpreting any monotonic or rank-correlation trend from E1/E2 as caused specifically by shortcut-mitigation strength.

**Priority**: Medium.

---

## Q2. Fisher ratio's shared-precision-matrix entanglement with condition number

**Status**: Open, with a partial mitigation already specified.

**Context**: `fisher_ratio_defense.md` §5 requires a bootstrap check ($\mathrm{Corr}(\kappa, J)$ at fixed true geometry) before trusting any E1/E2 result that leans on Fisher ratio, and requires reporting the decoupled $\mathrm{tr}(S_B)/\mathrm{tr}(S_W)$ companion alongside it.

**Action**: Run the bootstrap check specified in `fisher_ratio_defense.md` §4 once real per-rung sample sizes are known. Not blocking Task 1–2; blocking before E2 interprets any Fisher-ratio result.

**Priority**: Medium — has a defined trigger point (before E2), not before E1's implementation work.

---

## Q3. `eval_ood_benchmarks.py` — the canonical E2 Mahalanobis source — resolves checkpoints by directory, not explicit file

**Status**: Open.

**Context**: found while writing `extract_embeddings_e1.py` and reading `eval_ood_benchmarks.py` directly. Its `_find_ckpt(dir_path)` picks `sorted(dir_path.glob("best-*.ckpt"), key=mtime, reverse=True)[0]`, i.e. the same newest-by-mtime, directory-based resolution `REPOSITORY_MAP.md` risk #5 already flagged as confirmed-wrong for several `runB`/`runB_orth1` seed directories. `threats_to_validity.md` #2 designated this script as the canonical E1–E3 Mahalanobis source specifically to avoid `train_csg.py`'s transform-mismatch bug — but never checked it against the separate checkpoint-resolution bug. It has it too.

**Action**: before E2 runs, verify whether `eval_ood_benchmarks.py`'s per-seed checkpoint choice (for `runB` and `runB_orth1`) matches the explicit, manifest-verified paths `extract_embeddings_e1.py` uses for E1's geometry metrics on the same seeds. If they diverge, E1's geometry and E2's AUROC would be describing two different checkpoints for the same nominal (rung, seed) row.

**Priority**: High — this one directly threatens whether E1 and E2's outputs are even about the same model, not just a downstream statistical concern.

---

## Q4. `baseline_soft` is not a valid zero-dose point on the disentanglement ladder

**Status**: Resolved.

**Context**: originally treated as "λ_orth = 0," the fourth rung of one continuous dose-response trend alongside `runA_grl → runB_orth1 → runB`. It isn't a valid zero-dose condition: it differs from the three CSG rungs in architecture (ResNet-50 vs. EfficientNet-B3-based CSGLite), representation dimensionality (2048-d backbone feature vs. 16-d `z_lesion`), and training recipe (Q1, above). The dimensionality difference specifically surfaced as a concrete failure, not just a theoretical worry: computing Mardia kurtosis's bootstrap calibration on `baseline_soft`'s native 2048-d feature was the direct cause of the GPU-server performance crisis (`_simulate_mardia_null` dominated by repeated large-matrix operations) — exactly the regime `geometry_metric_audit.md` §3 (C1/C2) had already flagged as outside this test's validated range, before any checkpoint was ever run.

**Resolution**: E1/E2 are split into a **primary mechanistic test** (the three-rung disentanglement ladder — `runA_grl`, `runB_orth1`, `runB`, all sharing architecture and 16-d `z_lesion`) and a **secondary/categorical reference** (`baseline_soft`, evaluated on its own native 2048-d representation, never pooled into the primary trend statistic). The primary ladder's association test uses Kendall's τ across three ordinal points, not a 4-point regression or a Spearman threshold implying more precision than three treatment levels support. `baseline_soft`'s comparison against the ladder is reported descriptively (does its geometry/AUROC sit inside or outside the ladder's range), with no formal success/failure criterion — the same pattern already used for E4. Mardia's kurtosis remains load-bearing only for the primary ladder; for `baseline_soft` it is computed but explicitly exploratory, not used to support or refute H3.

**Updated**: `SPEC.md` §4 (E1/E2 experiment plan), `experiment_contract.md` (E1a/E1b, E2a/E2b split), `geometry_metric_audit.md` §6 and the C2 verdict note. No scripts modified.

**Priority**: Resolved — was High while open, since it bore on whether the primary claim H3 is tested against was even a coherent single comparison.

---

## Q5. E1's geometry and E2's AUROC fit Mahalanobis with different `reg_eps`

**Status**: Resolved.

**Context**: `geometry_metric_audit.md` §3 (A1) already documented that CSG-SKin's own scripts use two different `reg_eps` values for the same `compute_mahalanobis_params_from_arrays` construction — `1e-5` (the function's own default, used by `train_csg.py`/`train_baseline.py`, and what `extract_embeddings_e1.py`'s geometry metrics use) vs. `1e-3` (explicitly passed by `eval_ood_scores.py`/`eval_ood_benchmarks.py`/`run_effb3_control.py`). This surfaced concretely while writing `extract_auroc_e2.py`, which initially replicated `eval_ood_benchmarks.py`'s `reg_eps=1e-3` for literal fidelity to the canonical script.

**Investigation**: checked every occurrence of `reg_eps` in CSG-SKin before deciding. Findings: (1) `1e-5` is the library function's own default — every use of `1e-3` is an explicit override; (2) no comment anywhere in `eval_ood_benchmarks.py`, `run_effb3_control.py`, or `eval_ood_scores.py` justifies `1e-3` over the default; (3) `eval_ood_scores.py` exposes it as a CLI flag with `default=1e-3`, i.e. the original author treated it as an adjustable knob, not a fixed requirement; (4) `threats_to_validity.md` #2 designated `eval_ood_benchmarks.py` canonical for its correct *transform* (avoiding `train_csg.py`'s grayscale bug), never for its `reg_eps` choice; (5) Paper 3's E2 produces its own new `results/e2_auroc.csv`, not a reproduction of `eval_ood_benchmarks.py`'s own `ood_comparison.csv`, so there is no existing published number requiring bit-for-bit match; (6) this project already has direct empirical evidence `reg_eps=1e-5` is numerically safe at the scales involved — E1's actual server run used it successfully at `z_lesion` (d=16), and it was also used successfully (no numerical failure, only a speed problem fixed separately) at `baseline_soft`'s d=2048 during the Mardia profiling investigation.

**Resolution**: `extract_auroc_e2.py`'s `REG_EPS` changed from `1e-3` to `1e-5`, matching `extract_embeddings_e1.py` exactly. This is now the one deliberate, documented departure from `eval_ood_benchmarks.py`'s literal values — every other part of its methodology (loader, transform, split, scoring convention) is still reproduced exactly. E1's precision matrix and E2's precision matrix for the same nominal (rung, seed) row are now the literal same fitted object, not two differently-regularized approximations of it — required for the paper's actual claim (geometry of *this* precision matrix explains AUROC) to be about one consistent thing.

**Priority**: Resolved — was Medium while open.

---

## Q6. Why Mahalanobis AUROC is consistently below 0.5 — resumed as a bounded, ROI-gated investigation

**Status**: In progress. Explicitly a separate, non-pre-registered investigation, not a silent extension of E2's already-completed, already-analyzed scope (`experiment_contract.md`).

**Context**: across all 13 primary-ladder checkpoints, Mahalanobis AUROC (ISIC-test vs. PAD-UFES) sits at ~0.37–0.44 — consistently below chance, not just "not great." Checked what's checkable without a new experiment first: `build_id_ood_test_dataloaders` returns `(id_loader, ood_loader)` in that literal order (verified from `splits.py` source); `extract_auroc_e2.py`'s label/score convention matches `eval_ood_benchmarks.py`'s exactly; `mahalanobis_min_squared_distances` is confirmed a true minimum over classes, not argmax; train/id-test/ood-test loaders share one `eval_transform`. None of these turned up a bug.

**Timeline of the decision, since it moved twice**: (1) a full E2.5 implementation (persist raw distances, `analysis/analyze_e2_distances.py` for histogram/KDE/ECDF/boxplot) was drafted; (2) deliberately reverted — the question is real but post-hoc, and running the full 13-checkpoint batch to chase it wasn't yet justified; (3) resumed via a cheap intermediate step instead of an all-or-nothing call: a **temporary 4-line print check on one checkpoint** (`runA_grl` seed=42) before committing to the full rerun. Result: `median(OOD)=7.82 < median(ID)=13.78`, `mean(OOD)=16.47 < mean(ID)=20.64` — OOD sits closer to ID class centroids on both statistics, crossing the pre-declared threshold for proceeding. E2.5 was then restored in full (temporary print removed).

**Why this crossed the threshold to investigate now rather than defer**: AUROC scattered noisily around 0.5 would have meant "underpowered, no effect." Landing consistently at ~0.37–0.44 with never a single checkpoint above 0.5, now corroborated by the actual distance distribution on one checkpoint (not just the ranking statistic), is a harder pattern to explain as noise — and the three possible explanations (representation mechanism / implementation bug / Mahalanobis-specific failure) each change what gets written in a way "defer to future work" would have left unresolved.

**Action**: re-run `run_e2_all.sh` (13 checkpoints) with the restored `extract_auroc_e2.py` — note this reproduces identical AUROC/FPR95 values (unchanged computation) while additionally capturing `results/e2_distances/*.npz` and `results/distance_summary.csv`. Clear/rename the existing `e2_auroc.csv` first (append-only writer; re-running without clearing will trigger `analyze_e2.py`'s duplicate-row check by design). Then run `analysis/analyze_e2_distances.py` for the full-ladder verdict.

**Priority**: High while in progress — the outcome changes what E2's finding actually is, not just how it's explained.

**Update (mechanism candidates checked, one by one, all against the same 13-checkpoint data, no rerun needed for any of these)**:

- *Distance direction* (`analyze_e2_distances.py`): confirmed directly, not inferred from AUROC — median(OOD) < median(ID) on all 13/13 checkpoints, pooled per rung. Real, but the histogram/KDE/ECDF figure shows the separation is modest/overlapping, not a clean two-population split.
- *Feature norm collapse* (`analyze_norm.py`): `||z_ID|| > ||z_OOD||` holds on the median for all 3 primary rungs, but the gap is small (~6.2-6.9 range) and at the mean level it's mixed (e.g. runA_grl: mean(OOD)=6.83 > mean(ID)=6.72). Real but modest contributor, not sufficient alone.
- *NV (Nevus) attractor* (`analyze_predicted_class.py`, `analyze_nv_attractor.py`): 62-64% of OOD samples get nearest-centroid-assigned to NV, consistently across all 3 rungs — but ID's own baseline P(pred=NV|ID) is already ~53% (ISIC-test is NV-majority; the ID confusion matrix shows NV pulls broadly from other true classes even within ID, e.g. true_VASC→pred_NV ~24-28%, true_DF→pred_NV ~15-18%). The OOD-vs-ID gap is only ~9-11 percentage points (ratio ~1.16-1.20x), essentially identical across all 3 rungs regardless of λ_orth. **This effectively rules out a strong domain-specific NV attractor as the mechanism** — NV's pull on OOD is not qualitatively different from its pull on ID, just slightly stronger.
- *Geometry-vs-AUROC correlation, directly* (`e2_kendall_tau.csv`, already computed): none of the 5 approved geometry metrics correlate with AUROC — all `|tau| <= 0.28`, all `p > 0.2` (most `p > 0.5`). Notably this holds even though **condition number moves by two orders of magnitude across the ladder** (65-83 at runA_grl -> 3020-8257 at runB, `tau=0.84`, `p=2.8e-5` against lambda_orth per `e1_kendall_tau.csv`) — AUROC stays flat at ~0.39-0.41 (`table2_e2_summary.csv`) regardless. A 100x change in covariance conditioning produces no detectable AUROC response.
- *Gaussian-assumption violation severity* (`e1_geometry_metrics.csv`, already computed): Mardia's kurtosis bootstrap z-scores range 191-824 across all 13 checkpoints — catastrophic, uniform rejection of multivariate normality at every rung and seed, not a borderline case. Critically, this violation does **not** improve with more disentanglement training either: `mardia_kurtosis_z` vs. rung-index `tau=-0.168, p=0.52` (`e1_kendall_tau.csv`) — not significant, no trend. The orthogonality objective changes representation geometry substantially (condition number, Fisher ratio) but does not move the class-conditional distributions any closer to the Gaussian regime Mahalanobis distance assumes.

**Working hypothesis, updated**: the convergence of these checks (no implementation bug found; no dominant attractor; norm collapse and distance-gap are real but modest; zero AUROC sensitivity to a 100x conditioning change; catastrophic and training-invariant non-Gaussianity) points away from "representation geometry doesn't matter" and toward **"Mahalanobis distance's own Gaussian/covariance assumption is violated badly enough, and independently of training, that it may not be a valid instrument for reading out geometry improvements on this representation at all."** This reframes the open question from *why does OOD score closer to ID* to *what is Mahalanobis distance actually measuring here, given its assumptions are already broken before any dose-response question is asked*.

**Next planned check, now designed and pre-registered as E2.6** (`experiment_contract.md`, `SPEC.md` §4): whether an alternative OOD scorer computed on the *same* embeddings behaves differently from Mahalanobis. If an assumption-light scorer tracks AUROC with geometry (or just achieves higher AUROC generally) where Mahalanobis does not, that would be strong evidence the failure is in the scoring rule, not the representation. Deliberately designed *before* writing any code, per explicit user instruction, to avoid picking a scorer/hyperparameter after seeing which one looks best:

- Scorers locked to exactly 3: Mahalanobis (existing, unchanged), Cosine-to-centroid (same per-class-centroid structure as Mahalanobis, covariance step dropped), k-NN distance (pooled across all 8 classes, Sun et al. 2022-style — decided explicitly over a per-class variant to keep it maximally assumption-free as a genuine contrast to Mahalanobis).
- k-NN's k: primary/headline **k=10**, with **{1, 50}** reported as a pre-registered robustness grid regardless of outcome — both decisions made by explicit user sign-off, not defaulted.
- Still **blocked on implementation**: needs a raw `z_lesion` extension to `extract_auroc_e2.py` (full train/id-test/ood-test arrays per checkpoint, ~20MB total, not subsampled — same artifact resolves the UMAP blocker as a side effect) + one more 13-checkpoint rerun. Formulas are locked; code has not been written yet (explicit user instruction: design and pre-register first, code second).
