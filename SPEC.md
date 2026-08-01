# Paper 3 — Specification

**Status**: hypothesis and experiment plan only. No experiments have been implemented. This document will be revised as the design firms up; it is not a frozen artifact the way Paper 1/2 are.

---

## 1. Research question

> **When do representation improvements translate into better reliability estimation?**

## 2. Explicit non-goals

To keep scope bounded for a solo-author, 6-month project:

- This is **not** a methods paper — no new representation-learning technique is being proposed.
- This is **not** an OOD-detection paper — OOD detection is not the object of study.
- This is **not** a Mahalanobis paper — Mahalanobis distance is not being improved, tuned, or compared against other estimators as an end in itself.

Mahalanobis-based reliability estimation (as already implemented in Paper 2 / DST-Skin) is used strictly as **the current instrument** for measuring "reliability estimation quality." If a different instrument were used, the research question would be unchanged; Mahalanobis is a means, not the subject.

## 3. Mechanistic hypothesis (H3)

> Representation improvements induced by shortcut-mitigation training change the feature geometry. These geometric changes may alter the assumptions required by Mahalanobis reliability estimation. Therefore, representation improvements may fail to translate into improved reliability estimation.

In other words: a representation can get *better* by the metric shortcut-mitigation training was designed to improve (lower domain leakage, per Paper 1's leakage probe) while simultaneously becoming *worse-suited* to a downstream reliability estimator whose validity depends on specific geometric assumptions (e.g. approximately class-conditional Gaussian clusters with a shared, well-conditioned covariance — the assumption Mahalanobis distance relies on). H3 is the candidate explanation for why representation quality and reliability-estimation quality might decouple, rather than move together as commonly assumed.

## 4. Experiment plan

### E1 — Measure geometry changes across the CSG-SKin dose-response ladder

**Objective**: characterize how feature geometry changes as shortcut-mitigation strength increases.

**Independent variable**: the four CSG-SKin method checkpoints, treated as a dose-response ladder of increasing shortcut-mitigation strength:

| Rung | Method | Mechanism | λ_orth |
|---|---|---|---|
| 0 | `baseline_soft` | no disentanglement | – |
| 1 | `runA_grl` | domain-adversarial only | 0.0 |
| 2 | `runB_orth1` | + orthogonality term | 1.0 |
| 3 | `runB` | + stronger orthogonality | 5.0 |

**What "geometry" means here** needs to be fixed before implementation — candidates include per-class covariance conditioning/eigenspectrum, class-cluster separation, within- vs. between-class scatter ratio, and normality/Gaussianity of the class-conditional feature distributions (the specific assumptions Mahalanobis leans on). This choice is deliberately left open pending design work; it is the first concrete decision E1's implementation needs to make.

**Output**: a per-rung, per-seed table of geometry metrics, extracted from the existing CSG-SKin checkpoints via `CSGLite.encode_z_lesion` (no retraining).

### E2 — Test whether geometry changes are associated with Mahalanobis AUROC

**Objective**: test the second link in H3 — does the geometry shift measured in E1 predict Mahalanobis-based reliability-estimation quality (Mahalanobis OOD-detection AUROC, ISIC-test vs. PAD-UFES)?

**Design**: correlate/regress E1's geometry metrics against per-rung, per-seed Mahalanobis AUROC across the same four-rung ladder.

**Output**: an association result (e.g. correlation or regression) between geometry-shift magnitude and Mahalanobis AUROC across the ladder.

### E3 — Validate on a strictly held-out third domain (HAM10000)

**Objective**: check that the E1↔E2 relationship isn't an artifact of the ISIC/PAD-UFES split used everywhere else in Papers 1 and 2, by re-running the same geometry/reliability measurement on a domain neither CSG-SKin nor DST-Skin was trained or tuned on.

**Design**: apply the same CSG-SKin checkpoints (no retraining) and the same geometry + Mahalanobis measurement pipeline from E1/E2 to HAM10000 as a third, strictly held-out domain.

### E4 — Optional supplementary validation on DST-Skin (independent codebase)

**Objective**: check whether the E1↔E2 relationship generalizes beyond CSG-SKin's specific architecture and training recipe, using Paper 2's independently-trained backbones (ResNet-18/50, EfficientNet-B3) and its own Ledoit-Wolf-shrinkage Mahalanobis implementation as an independent measurement path.

**Status**: optional / supplementary, not required for the core claim. Scoped last, and only if E1–E3 support H3 and time remains.

---

## 5. Reuse and non-modification constraints

- **Paper 1 (CSG-SKin) and Paper 2 (DST-Skin) are frozen historical artifacts.** No code in either repository is modified as part of Paper 3 unless explicitly instructed otherwise for a specific, isolated change.
- **All new code lives under `paper-3/scripts/` and `paper-3/analysis/`.** Existing functionality (checkpoint loading, feature/embedding extraction, Mahalanobis fitting) is reused via **import**, not copy-paste or reimplementation.
- **Every experiment must be reproducible**: pinned seeds, checkpoints referenced by explicit file path (never a directory — see feasibility note below), and versioned outputs written to `paper-3/results/`.

## 6. Known feasibility risks and dependencies (from `REPOSITORY_MAP.md`)

These were identified during the repository audit and are worth resolving explicitly during design, before E1–E4 are implemented:

- **E1/E2 seed coverage is uneven across the ladder.** `runB` (λ_orth=5.0) checkpoints exist for only 3 of 5 seeds (42, 52, 62) in the current `checkpoints/` inventory; `runA_grl`, `runB_orth1`, and `baseline_soft` each have all 5. Any per-rung statistics across the ladder need to either match on the common 3 seeds or explicitly report the asymmetry.
- **Checkpoint resolution must use explicit files, not directories.** CSG-SKin's `find_checkpoint` (`src/utils/ood_metrics.py`) resolves the *newest-by-mtime* file in a directory, which is confirmed to pick a non-best-val checkpoint in several `runB`/`runB_orth1` seed directories due to stale duplicates from repeated training runs. E1/E2/E3 scripts must reference specific `.ckpt` files, verified against each run's actual best epoch, not a resolved directory path.
- **CSG-SKin's own "OOD" evaluation set is not a clean held-out set.** By construction, all of PAD-UFES is both CSG's auxiliary training-domain stream and its OOD-test set (`SkinDataModule.setup`). If E2 uses CSG-SKin's own pre-computed summary.json Mahalanobis AUROC numbers directly, those numbers reflect this contamination. This is a specific reason E3's genuinely-held-out HAM10000 check matters, and also a reason E2 may need to recompute Mahalanobis AUROC on a cleaner split rather than trusting the existing per-run summaries as-is.
- **Two Mahalanobis implementations exist and are not interchangeable.** CSG-SKin's `ood_metrics.py` uses a raw pooled within-class covariance; DST-Skin's `OODScorer` uses Ledoit-Wolf shrinkage on L2-normalized features. E1/E2/E3 (CSG-based) and E4 (DST-based) will produce numbers from different formulas — this is expected given E4's role as an independent validation path, but any cross-experiment comparison must say explicitly which formula produced which number.
- **HAM10000 data is not currently present in either repository.** DST-Skin contains an orphaned preprocessing script (`split.py`) that references `data/raw/ham10000/`, which does not exist on disk in this checkout. E3 will need this data acquired and prepared before it can run — this is a hard external dependency, not a code dependency.
- **Label-taxonomy mismatch between the two repos.** CSG-SKin uses the full 8-class ISIC label space; DST-Skin uses a binarized malignant/benign relabeling, and the two repos' malignant-class definitions for PAD-UFES don't match each other's conventions. E4 (DST-Skin path) will need its own, separately-defined label handling — it cannot silently inherit CSG-SKin's label logic.

## 7. Open design decisions

- The precise geometry metric(s) for E1 (see E1 note above) — this is the first thing to pin down before any script is written.
- Whether E2's association test is correlation, regression, or something else, and what counts as a statistically meaningful result given the small per-rung seed counts (n=3–5).
- Where/how HAM10000 data will be sourced and preprocessed for E3.
- Whether E4 is scoped at all, and if so, to what extent, given its "optional" status and the 6-month solo-author budget.
